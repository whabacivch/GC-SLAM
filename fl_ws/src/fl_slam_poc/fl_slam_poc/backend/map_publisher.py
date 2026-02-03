"""
PrimitiveMap publisher: publishes map as PointCloud2 and optional MarkerArray.

Consumes PrimitiveMapView only (extracted from PrimitiveMap). Used for Rerun
(Wayland-friendly) and downstream ROS tools. Rendering is derived output from
state; this module does not feed back into the pipeline.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md Section 9
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from fl_slam_poc.backend.structures.primitive_map import (
    AtlasMap,
    extract_primitive_map_view,
    renderable_batch_from_view,
    RenderablePrimitiveBatch,
)
from fl_slam_poc.common import constants

if TYPE_CHECKING:
    from fl_slam_poc.backend.rerun_visualizer import RerunVisualizer


def _pointcloud2_fields_xyz_intensity() -> tuple[list, int, dict]:
    """PointCloud2 schema: x, y, z (float32), intensity (float32 weight)."""
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 16
    offsets = {"x": 0, "y": 4, "z": 8, "intensity": 12}
    return fields, point_step, offsets


def _build_pointcloud2_from_view(
    positions: np.ndarray,
    weights: np.ndarray,
    colors: Optional[np.ndarray],
    frame_id: str,
    stamp_sec: float,
) -> PointCloud2:
    """
    Build PointCloud2 from primitive view arrays (numpy).

    positions: (N, 3), weights: (N,), colors: (N, 3) or None.
    """
    from builtin_interfaces.msg import Time

    n = positions.shape[0]
    fields, point_step, _ = _pointcloud2_fields_xyz_intensity()

    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = Time(sec=int(stamp_sec), nanosec=int((stamp_sec % 1) * 1e9))
    msg.height = 1
    msg.width = n
    msg.is_dense = True
    msg.is_bigendian = False
    msg.fields = fields
    msg.point_step = point_step
    msg.row_step = point_step * n

    if n == 0:
        msg.data = b""
        return msg

    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")],
        align=False,
    )
    arr = np.zeros((n,), dtype=dtype)
    arr["x"] = positions[:, 0].astype(np.float32)
    arr["y"] = positions[:, 1].astype(np.float32)
    arr["z"] = positions[:, 2].astype(np.float32)
    arr["intensity"] = np.clip(weights.astype(np.float32), 0.0, 1e6)
    msg.data = arr.tobytes()
    return msg


class PrimitiveMapPublisher:
    """
    Publishes PrimitiveMap as PointCloud2 (/gc/map/points).

    Optional: MarkerArray (/gc/map/ellipsoids) for covariance ellipsoids.
    Frame_id should match state (e.g. odom). No /tf from this module.
    """

    def __init__(
        self,
        node,
        frame_id: str = "odom",
        publish_ellipsoids: bool = False,
        max_primitives: Optional[int] = None,
        eps_lift: float = constants.GC_EPS_LIFT,
        eps_mass: float = constants.GC_EPS_MASS,
        rerun_visualizer: Optional["RerunVisualizer"] = None,
    ):
        self._node = node
        self._frame_id = frame_id
        self._publish_ellipsoids = publish_ellipsoids
        self._max_primitives = max_primitives
        self._eps_lift = eps_lift
        self._eps_mass = eps_mass
        self._rerun_visualizer = rerun_visualizer

        self._pub_cloud = node.create_publisher(
            PointCloud2,
            "/gc/map/points",
            10,
        )
        self._pub_markers = None
        if publish_ellipsoids:
            from visualization_msgs.msg import MarkerArray

            self._pub_markers = node.create_publisher(
                MarkerArray,
                "/gc/map/ellipsoids",
                10,
            )

    def publish(self, primitive_map: AtlasMap, stamp_sec: float) -> RenderablePrimitiveBatch:
        """
        Publish primitive map as PointCloud2 (and optionally MarkerArray).

        Extracts PrimitiveMapView from primitive_map; publishes positions and
        weights. Returns RenderablePrimitiveBatch with full fields.
        """
        if primitive_map.n_tiles != 1:
            raise ValueError(
                f"PrimitiveMapPublisher expects single-tile atlas, got {primitive_map.n_tiles}"
            )
        tile_id = primitive_map.tile_ids[0]
        tile = primitive_map.tiles[tile_id]
        view = extract_primitive_map_view(
            tile=tile,
            max_primitives=self._max_primitives,
            eps_lift=self._eps_lift,
            eps_mass=self._eps_mass,
        )
        n = view.count
        if n == 0:
            cloud_msg = _build_pointcloud2_from_view(
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                None,
                self._frame_id,
                stamp_sec,
            )
            self._pub_cloud.publish(cloud_msg)
            if self._rerun_visualizer is not None:
                self._rerun_visualizer.log_map(
                    np.zeros((0, 3)), np.zeros((0,)), None, stamp_sec
                )
            if self._pub_markers is not None:
                from visualization_msgs.msg import MarkerArray as MarkerArrayMsg

                self._pub_markers.publish(MarkerArrayMsg())
            return renderable_batch_from_view(view, eps_lift=self._eps_lift)

        positions = np.array(view.positions)
        weights = np.array(view.weights)
        colors = np.array(view.colors) if view.colors is not None else None

        cloud_msg = _build_pointcloud2_from_view(
            positions,
            weights,
            colors,
            self._frame_id,
            stamp_sec,
        )
        self._pub_cloud.publish(cloud_msg)

        if self._rerun_visualizer is not None:
            self._rerun_visualizer.log_map(positions, weights, colors, stamp_sec)

        if self._pub_markers is not None:
            # Optional: build ellipsoid markers from view.covariances
            # Deferred: full MarkerArray from covariances can be added later
            pass

        return renderable_batch_from_view(view, eps_lift=self._eps_lift)
