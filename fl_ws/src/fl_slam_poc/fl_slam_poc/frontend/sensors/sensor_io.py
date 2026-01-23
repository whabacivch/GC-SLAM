"""
Sensor I/O Layer - NO MATH.

Handles:
- Sensor subscriptions and buffering
- TF lookups
- Point cloud conversions (depth→3D, scan→3D)
- Timestamp-based data retrieval

All geometric transforms use geometry.se3 (exact operations).
"""

from typing import Optional, Tuple
import ast
import json
import time
import numpy as np
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan, PointCloud2, PointField
from nav_msgs.msg import Odometry
from tf2_ros import TransformException

from fl_slam_poc.common import constants
from fl_slam_poc.common.constants import (
    QOS_DEPTH_SENSOR_HIGH_FREQ,
    QOS_DEPTH_SENSOR_MED_FREQ,
)
from fl_slam_poc.common.geometry.se3_numpy import (
    quat_to_rotmat,
    se3_compose,
    se3_relative,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
)
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common.utils import stamp_to_sec

from fl_slam_poc.frontend.sensors.qos_utils import resolve_qos_profiles
from fl_slam_poc.frontend.sensors.dedup import is_duplicate


def pointcloud2_to_array(msg: PointCloud2) -> np.ndarray:
    """
    Convert PointCloud2 message to numpy array of XYZ points.

    Handles common point cloud formats (XYZ, XYZRGB, etc.)
    Returns array of shape (N, 3) containing [x, y, z] coordinates.
    
    Dynamically handles field data types (float32, float64, int32, etc.)
    based on the PointCloud2 field metadata.
    """
    # Get field info
    field_names = [f.name for f in msg.fields]
    field_map = {f.name: f for f in msg.fields}

    # Check for XYZ fields
    if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
        raise ValueError("PointCloud2 message missing x, y, or z fields")

    # Data type mapping from PointField constants to numpy dtypes
    dtype_map = {
        PointField.FLOAT32: (np.float32, 4),
        PointField.FLOAT64: (np.float64, 8),
        PointField.INT32: (np.int32, 4),
        PointField.UINT32: (np.uint32, 4),
        PointField.INT16: (np.int16, 2),
        PointField.UINT16: (np.uint16, 2),
        PointField.INT8: (np.int8, 1),
        PointField.UINT8: (np.uint8, 1),
    }

    # Get field info for x, y, z
    x_field = field_map['x']
    y_field = field_map['y']
    z_field = field_map['z']

    # Determine dtype (use x field's type, assuming x/y/z have same type)
    x_dtype_info = dtype_map.get(x_field.datatype, (np.float32, 4))
    dtype, byte_size = x_dtype_info

    point_step = msg.point_step
    n_points = msg.width * msg.height

    if n_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Parse raw data using the correct dtype
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)

    x = data[:, x_field.offset:x_field.offset+byte_size].view(dtype).flatten()
    y = data[:, y_field.offset:y_field.offset+byte_size].view(dtype).flatten()
    z = data[:, z_field.offset:z_field.offset+byte_size].view(dtype).flatten()

    points = np.stack([x, y, z], axis=1)

    # Filter invalid points (NaN, Inf)
    valid = np.isfinite(points).all(axis=1)
    return points[valid].astype(np.float32)


class SensorIO:
    """
    Sensor I/O manager - buffering, TF, and point cloud conversion.
    
    Pure I/O layer - NO mathematical inference.
    """
    
    def __init__(self, node: Node, config: dict):
        """
        Args:
            node: ROS node for subscriptions and logging
            config: Dict with keys: scan_topic, odom_topic, camera_topic, depth_topic,
                    camera_info_topic, enable_image, enable_depth, enable_camera_info,
                    odom_is_delta, odom_frame, base_frame, camera_frame, scan_frame,
                    tf_timeout_sec, feature_buffer_len, depth_stride
        """
        self.node = node
        self.config = config
        
        # Buffers (timestamp, data)
        self.odom_buffer = []
        self.image_buffer = []  # (timestamp, rgb_array, frame_id)
        self.depth_buffer = []  # (timestamp, depth_array, points, frame_id)
        self.pointcloud_buffer = []  # (timestamp, points_array, frame_id)
        self.imu_buffer = []  # (timestamp, accel_xyz, gyro_xyz) - event-driven clearing
        self.last_imu_frame_id = None

        # Flow counters for data flow audit
        self.imu_callback_count = 0
        self.imu_segment_published_count = 0

        # State
        self.last_pose = None
        self.depth_intrinsics = None
        self._last_msg_keys = {}
        
        # IMU acceleration scale (Livox outputs in g's, needs *9.81 for m/s²)
        self.imu_accel_scale = config.get("imu_accel_scale", 9.81)
        
        # 3D point cloud mode
        self.use_3d_pointcloud = config.get("use_3d_pointcloud", False)
        self.enable_pointcloud = config.get("enable_pointcloud", False)
        self._last_pointcloud_time = 0.0
        self._pointcloud_rate_limit = 1.0 / config.get("pointcloud_rate_limit_hz", 30.0) if config.get("pointcloud_rate_limit_hz", 30.0) > 0 else 0.0
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        # Rosbag playback frequently publishes static transforms on /tf_static.
        # Use TRANSIENT_LOCAL for static TF so late-joining subscribers receive it.
        tf_qos = QoSProfile(
            depth=constants.QOS_DEPTH_LOW_FREQ,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        tf_static_qos = QoSProfile(
            depth=constants.QOS_DEPTH_LOW_FREQ,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            node,
            qos=tf_qos,
            static_qos=tf_static_qos,
        )
        
        # C++ decompressor is now the only supported image path.
        # Avoid importing cv_bridge in Python to prevent NumPy ABI crashes.
        self.cv_bridge = None
        
        # Subscribe to sensors
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        """Create ROS subscriptions for all enabled sensors."""
        qos_profiles, qos_names = resolve_qos_profiles(
            reliability=str(self.config.get("sensor_qos_reliability", "reliable")).lower(),
            depth=QOS_DEPTH_SENSOR_MED_FREQ,
        )
        for qos in qos_profiles:
            # Always subscribe to odometry
            self.node.create_subscription(
                Odometry, self.config["odom_topic"], self._on_odom, qos)
            
            # Subscribe to LaserScan (2D mode) or PointCloud2 (3D mode)
            if self.use_3d_pointcloud:
                # 3D mode: Subscribe to PointCloud2 instead of LaserScan
                pointcloud_topic = self.config.get("pointcloud_topic", constants.POINTCLOUD_TOPIC_DEFAULT)
                self.node.create_subscription(
                    PointCloud2, pointcloud_topic, self._on_pointcloud, qos)
                self.node.get_logger().info(
                    f"SensorIO: 3D point cloud mode enabled, subscribing to {pointcloud_topic}"
                )
            else:
                # 2D mode: Subscribe to LaserScan
                self.node.create_subscription(
                    LaserScan, self.config["scan_topic"], self._on_scan_internal, qos)
            
            # Optionally subscribe to both PointCloud2 AND LaserScan
            if self.enable_pointcloud and not self.use_3d_pointcloud:
                pointcloud_topic = self.config.get("pointcloud_topic", constants.POINTCLOUD_TOPIC_DEFAULT)
                self.node.create_subscription(
                    PointCloud2, pointcloud_topic, self._on_pointcloud, qos)
            
            if self.config.get("enable_image", False):
                self.node.create_subscription(
                    Image, self.config["camera_topic"], self._on_image, qos)
            
            # Depth images are optional and may be used for RGB-D evidence/appearance
            # even when the primary sparse geometry source is PointCloud2.
            if self.config.get("enable_depth", False):
                self.node.create_subscription(
                    Image, self.config["depth_topic"], self._on_depth, qos)
            
            if self.config.get("enable_camera_info", False):
                self.node.create_subscription(
                    CameraInfo, self.config["camera_info_topic"], self._on_camera_info, qos)

        # IMU subscription (high-rate sensor with separate QoS for 200Hz data)
        if self.config.get("enable_imu", False):
            imu_qos_profiles, imu_qos_names = self._resolve_imu_qos_profiles()
            imu_topic = self.config.get("imu_topic", constants.IMU_TOPIC_DEFAULT)
            for imu_qos in imu_qos_profiles:
                self.node.create_subscription(Imu, imu_topic, self._on_imu, imu_qos)
            self.node.get_logger().info(
                f"SensorIO: IMU subscription enabled on {imu_topic} "
                f"with QoS reliability: {', '.join(imu_qos_names)}, depth: {QOS_DEPTH_SENSOR_HIGH_FREQ}"
            )

        mode_str = "3D PointCloud" if self.use_3d_pointcloud else "2D LaserScan"
        self.node.get_logger().info(
            f"SensorIO ({mode_str} mode) subscribed to {self.config['odom_topic']} "
            f"with QoS reliability: {', '.join(qos_names)}, depth: {QOS_DEPTH_SENSOR_MED_FREQ}"
        )

    def _resolve_imu_qos_profiles(self):
        """
        Resolve IMU QoS profiles from config.

        Uses high-frequency depth buffer to handle 200Hz IMU data.
        Separate from sensor QoS to allow different reliability/depth settings.
        """
        reliability = str(self.config.get("imu_qos_reliability", "best_effort")).lower()
        rel_map = {
            "reliable": ReliabilityPolicy.RELIABLE,
            "best_effort": ReliabilityPolicy.BEST_EFFORT,
            "system_default": ReliabilityPolicy.SYSTEM_DEFAULT,
        }
        if reliability == "both":
            rels = [ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT]
            names = ["reliable", "best_effort"]
        elif reliability in rel_map:
            rels = [rel_map[reliability]]
            names = [reliability]
        else:
            rels = [ReliabilityPolicy.BEST_EFFORT]
            names = ["best_effort"]

        profiles = [
            QoSProfile(
                reliability=rel,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=QOS_DEPTH_SENSOR_HIGH_FREQ,
            )
            for rel in rels
        ]

        return profiles, names

    def set_scan_callback(self, callback):
        """Set external callback for scan processing."""
        self._scan_callback = callback

    def set_odom_callback(self, callback):
        """Set external callback for odom processing."""
        self._odom_callback = callback

    def set_image_callback(self, callback):
        """Set external callback for image processing."""
        self._image_callback = callback

    def set_depth_callback(self, callback):
        """Set external callback for depth processing."""
        self._depth_callback = callback

    def set_pointcloud_callback(self, callback):
        """Set external callback for point cloud processing."""
        self._pointcloud_callback = callback

    def set_imu_callback(self, callback):
        """Set external callback for IMU processing."""
        self._imu_callback = callback

    def compose_pose(self, pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
        """Compose two SE(3) poses using internal geometry utilities."""
        return se3_compose(pose_a, pose_b)

    def relative_pose(self, pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
        """Compute relative SE(3) pose using internal geometry utilities."""
        return se3_relative(pose_a, pose_b)
    
    def _on_scan_internal(self, msg: LaserScan):
        """Internal scan handler - calls external callback if set."""
        if is_duplicate(self._last_msg_keys, "scan", msg.header.stamp, frame_id=msg.header.frame_id):
            return
        # Debug: Log first scan received
        if not hasattr(self, '_first_scan_logged'):
            self._first_scan_logged = True
            self.node.get_logger().info(
                f"SensorIO: First scan received, frame_id={msg.header.frame_id}, "
                f"ranges={len(msg.ranges)}, last_pose={'SET' if self.last_pose is not None else 'NONE'}"
            )
        
        if hasattr(self, '_scan_callback'):
            self._scan_callback(msg)
    
    def _on_pointcloud(self, msg: PointCloud2):
        """Buffer 3D point cloud data."""
        if is_duplicate(self._last_msg_keys, "pointcloud", msg.header.stamp, frame_id=msg.header.frame_id):
            return
        
        # Rate limiting for high-frequency point clouds
        current_time = time.time()
        if self._pointcloud_rate_limit > 0 and (current_time - self._last_pointcloud_time) < self._pointcloud_rate_limit:
            return
        self._last_pointcloud_time = current_time
        
        try:
            # Convert PointCloud2 to numpy array
            points = pointcloud2_to_array(msg)
            
            if points.shape[0] == 0:
                self.node.get_logger().debug(
                    "PointCloud2: empty after conversion (no valid points)",
                    throttle_duration_sec=5.0,
                )
                return
            
            # Debug: Log first point cloud received
            if not hasattr(self, '_first_pointcloud_logged'):
                self._first_pointcloud_logged = True
                self.node.get_logger().info(
                    f"SensorIO: First point cloud received, frame_id={msg.header.frame_id}, "
                    f"points={points.shape[0]}, last_pose={'SET' if self.last_pose is not None else 'NONE'}"
                )
            
            stamp = stamp_to_sec(msg.header.stamp)
            frame_id = msg.header.frame_id or self.config.get("camera_frame", "camera_link")
            
            # Transform to base_frame if needed
            base_frame = self.config.get("base_frame", "base_link")
            if frame_id != base_frame:
                extr = self.config.get("lidar_base_extrinsic", None)
                # Treat empty list/string or all zeros (launch default) as "not provided".
                if isinstance(extr, str) and extr.strip() == "":
                    extr = None
                elif isinstance(extr, list):
                    if len(extr) == 0:
                        extr = None
                    elif len(extr) == 6 and all(
                        abs(x) < constants.NUMERICAL_EPSILON for x in extr
                    ):
                        extr = None

                if extr is not None:
                    try:
                        # Launch files sometimes pass arrays as strings (e.g. "[0,0,0,0,0,0]").
                        if isinstance(extr, str):
                            s = extr.strip()
                            if s == "":
                                raise ValueError("empty string")
                            try:
                                extr = json.loads(s)
                            except Exception:
                                extr = ast.literal_eval(s)
                        extr = np.asarray(extr, dtype=float).reshape(-1)
                        if extr.shape != (6,):
                            raise ValueError(f"expected shape (6,), got {extr.shape}")
                        points = self._transform_points(points, extr)
                        frame_id = base_frame
                        if not hasattr(self, "_logged_pc_static_extrinsic"):
                            self._logged_pc_static_extrinsic = True
                            self.node.get_logger().info(
                                f"SensorIO: Using lidar_base_extrinsic for pointcloud transform "
                                f"({base_frame} ← {msg.header.frame_id}); TF ignored by configuration."
                            )
                            if hasattr(self.node, "_publish_report"):
                                report = OpReport(
                                    name="PointCloudTransformStatic",
                                    exact=True,
                                    family_in="PointCloud2",
                                    family_out="PointCloud2",
                                    closed_form=True,
                                    metrics={
                                        "source": "lidar_base_extrinsic",
                                        "target_frame": base_frame,
                                        "source_frame": str(msg.header.frame_id),
                                    },
                                    notes="Static extrinsic configured and applied for pointcloud transform.",
                                )
                                report.validate()
                                self.node._publish_report(report)  # type: ignore[attr-defined]
                    except Exception as e_extr:
                        if not hasattr(self, "_logged_pc_extrinsic_bad"):
                            self._logged_pc_extrinsic_bad = True
                            self.node.get_logger().error(
                                f"SensorIO: lidar_base_extrinsic provided but invalid ({e_extr}). "
                                "Pointcloud rejected to avoid frame corruption."
                            )
                            if hasattr(self.node, "_publish_report"):
                                report = OpReport(
                                    name="PointCloudExtrinsicInvalid",
                                    exact=True,
                                    family_in="PointCloud2",
                                    family_out="PointCloud2",
                                    closed_form=True,
                                    metrics={
                                        "target_frame": base_frame,
                                        "source_frame": str(msg.header.frame_id),
                                        "error": f"{type(e_extr).__name__}: {e_extr}",
                                    },
                                    notes="lidar_base_extrinsic was provided but could not be parsed/applied.",
                                )
                                report.validate()
                                self.node._publish_report(report)  # type: ignore[attr-defined]
                        return
                else:
                    transform = self._lookup_transform(base_frame, frame_id, msg.header.stamp)
                    if transform is None:
                        if not hasattr(self, '_logged_pc_tf_fail'):
                            self._logged_pc_tf_fail = True
                            self.node.get_logger().error(
                                f"CRITICAL: Cannot transform pointcloud from '{frame_id}' to '{base_frame}'. "
                                f"TF lookup failed and no lidar_base_extrinsic configured."
                            )
                            if hasattr(self.node, "_publish_report"):
                                report = OpReport(
                                    name="PointCloudTransformMissing",
                                    exact=True,
                                    family_in="PointCloud2",
                                    family_out="PointCloud2",
                                    closed_form=True,
                                    metrics={
                                        "target_frame": base_frame,
                                        "source_frame": str(msg.header.frame_id),
                                        "has_lidar_base_extrinsic": False,
                                    },
                                    notes="TF lookup failed and no usable lidar_base_extrinsic was available.",
                                )
                                report.validate()
                                self.node._publish_report(report)  # type: ignore[attr-defined]
                        return
                    points = self._transform_points(points, transform)
                    frame_id = base_frame
            else:
                if not hasattr(self, '_logged_pc_no_tf'):
                    self._logged_pc_no_tf = True
                    self.node.get_logger().info(
                        f"PointCloud frame '{frame_id}' matches base_frame, no TF transform needed"
                    )
            
            # Buffer management
            buffer_len = self.config.get("feature_buffer_len", 10)
            self.pointcloud_buffer.append((stamp, points, frame_id))
            if len(self.pointcloud_buffer) > buffer_len:
                self.pointcloud_buffer.pop(0)
            
            # Call external callback if set
            if hasattr(self, '_pointcloud_callback'):
                self._pointcloud_callback(msg, points)
                
        except Exception as e:
            self.node.get_logger().error(
                f"PointCloud2 conversion failed: {e}", throttle_duration_sec=5.0)
            if hasattr(self.node, "_publish_report"):
                report = OpReport(
                    name="PointCloudConversionFailed",
                    exact=True,
                    family_in="PointCloud2",
                    family_out="PointCloud2",
                    closed_form=True,
                    domain_projection=True,
                    metrics={"error": f"{type(e).__name__}: {e}"},
                    notes="PointCloud2 message could not be parsed into points.",
                )
                report.validate()
                self.node._publish_report(report)  # type: ignore[attr-defined]
    
    def _on_odom(self, msg: Odometry):
        """Buffer odometry (pose only)."""
        if is_duplicate(self._last_msg_keys, "odom", msg.header.stamp, frame_id=msg.header.frame_id):
            return
        stamp = stamp_to_sec(msg.header.stamp)
        
        # Debug logging for first odom message
        if self.last_pose is None:
            self.node.get_logger().info(
                f"SensorIO: First odom received at stamp {stamp:.3f}, frame_id={msg.header.frame_id}"
            )
        
        if self.config.get("odom_is_delta", False):
            # Accumulate deltas using proper SE(3) composition
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            
            # CRITICAL FIX: Convert quaternion to rotation vector (not just qx,qy,qz!)
            # The odometry message contains a quaternion, not a rotation vector
            R_delta = quat_to_rotmat(ori.x, ori.y, ori.z, ori.w)
            rotvec_delta = rotmat_to_rotvec(R_delta)
            delta = np.array([pos.x, pos.y, pos.z, rotvec_delta[0], rotvec_delta[1], rotvec_delta[2]], dtype=float)
            
            if self.last_pose is None:
                self.last_pose = np.zeros(6, dtype=float)
            
            # CRITICAL FIX: Use proper SE(3) composition instead of addition
            # pose_new = pose_old ∘ delta (compose in the body frame)
            self.last_pose = se3_compose(self.last_pose, delta)
            pose = self.last_pose.copy()
        else:
            # Absolute pose - convert quaternion to rotation vector
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            
            # Use geometry.se3 for exact conversion
            R = quat_to_rotmat(ori.x, ori.y, ori.z, ori.w)
            rotvec = rotmat_to_rotvec(R)
            
            pose = np.array([pos.x, pos.y, pos.z, rotvec[0], rotvec[1], rotvec[2]], dtype=float)
            self.last_pose = pose.copy()
        
        # Buffer management
        buffer_len = self.config.get("feature_buffer_len", 10)
        self.odom_buffer.append((stamp, pose))
        if len(self.odom_buffer) > buffer_len:
            self.odom_buffer.pop(0)

        if hasattr(self, "_odom_callback"):
            self._odom_callback(msg)
    
    def _on_image(self, msg: Image):
        """Buffer RGB image array for RGB-D evidence extraction."""
        if is_duplicate(self._last_msg_keys, "image", msg.header.stamp, frame_id=msg.header.frame_id):
            return
        encoding = str(getattr(msg, "encoding", "") or "").lower()
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)

        def warn_once(text: str):
            if not hasattr(self, "_logged_rgb_decode"):
                self._logged_rgb_decode = True
                self.node.get_logger().info(text)

        try:
            raw = memoryview(msg.data)
            row_bytes = w * 3
            if encoding in ("rgb8", "bgr8"):
                if step < row_bytes:
                    raise ValueError(f"rgb step too small: step={step}, expected>={row_bytes}")
                buf = np.frombuffer(raw, dtype=np.uint8).reshape(h, step)[:, :row_bytes]
                img = buf.reshape(h, w, 3)
                if encoding == "bgr8":
                    img = img[:, :, ::-1]
                rgb = img
                warn_once(f"SensorIO: Decoding Image via pure NumPy ({encoding})")
            elif encoding in ("rgba8", "bgra8"):
                row_bytes4 = w * 4
                if step < row_bytes4:
                    raise ValueError(f"rgba step too small: step={step}, expected>={row_bytes4}")
                buf = np.frombuffer(raw, dtype=np.uint8).reshape(h, step)[:, :row_bytes4]
                img = buf.reshape(h, w, 4)
                if encoding == "bgra8":
                    img = img[:, :, [2, 1, 0, 3]]
                rgb = img[:, :, :3]
                warn_once(f"SensorIO: Decoding Image via pure NumPy ({encoding})")
            elif encoding in ("mono8",):
                if step < w:
                    raise ValueError(f"mono step too small: step={step}, expected>={w}")
                buf = np.frombuffer(raw, dtype=np.uint8).reshape(h, step)[:, :w]
                gray = buf.reshape(h, w)
                rgb = np.repeat(gray[:, :, None], 3, axis=2)
                warn_once("SensorIO: Decoding Image via pure NumPy (mono8 -> RGB)")
            else:
                if not hasattr(self, "_logged_rgb_unsupported"):
                    self._logged_rgb_unsupported = True
                    self.node.get_logger().warn(
                        f"SensorIO: Unsupported Image encoding '{encoding}'. "
                        "RGB buffering disabled for this encoding.",
                        throttle_duration_sec=5.0,
                    )
                rgb = None
        except Exception as e:
            self.node.get_logger().warn(f"RGB decode failed: {e}", throttle_duration_sec=5.0)
            rgb = None

        stamp = stamp_to_sec(msg.header.stamp)
        frame_id = msg.header.frame_id or self.config.get("camera_frame", "camera_link")
        if rgb is not None:
            self.image_buffer.append((stamp, rgb, frame_id))
        
        buffer_len = self.config.get("feature_buffer_len", 10)
        if len(self.image_buffer) > buffer_len:
            self.image_buffer.pop(0)

        if hasattr(self, "_image_callback"):
            self._image_callback(msg)
    
    def _on_depth(self, msg: Image):
        """Buffer depth array (and optionally 3D points) for RGB-D evidence extraction."""
        if is_duplicate(self._last_msg_keys, "depth", msg.header.stamp, frame_id=msg.header.frame_id):
            return
        encoding = str(getattr(msg, "encoding", "") or "").lower()
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)

        def warn_once(text: str):
            if not hasattr(self, "_logged_depth_decode"):
                self._logged_depth_decode = True
                self.node.get_logger().info(text)

        depth: Optional[np.ndarray]
        try:
            raw = memoryview(msg.data)
            if encoding in ("32fc1",):
                row_bytes = w * 4
                if step < row_bytes:
                    raise ValueError(f"32FC1 step too small: step={step}, expected>={row_bytes}")
                buf = np.frombuffer(raw, dtype=np.float32).reshape(h, step // 4)[:, :w]
                depth = buf.reshape(h, w).astype(np.float32, copy=False)
                warn_once("SensorIO: Decoding depth via pure NumPy (32FC1 meters)")
            elif encoding in ("16uc1",):
                row_bytes = w * 2
                if step < row_bytes:
                    raise ValueError(f"16UC1 step too small: step={step}, expected>={row_bytes}")
                buf = np.frombuffer(raw, dtype=np.uint16).reshape(h, step // 2)[:, :w]
                depth_u16 = buf.reshape(h, w)
                depth = (depth_u16.astype(np.float32) * 1e-3)  # mm -> m
                warn_once("SensorIO: Decoding depth via pure NumPy (16UC1 mm -> meters)")
            else:
                if not hasattr(self, "_logged_depth_unsupported"):
                    self._logged_depth_unsupported = True
                    self.node.get_logger().warn(
                        f"SensorIO: Unsupported depth Image encoding '{encoding}'. "
                        "Depth buffering disabled for this encoding.",
                        throttle_duration_sec=5.0,
                    )
                depth = None
        except Exception as e:
            self.node.get_logger().warn(f"Depth decode failed: {e}", throttle_duration_sec=5.0)
            depth = None
        
        # Compute points for descriptor/ICP processing (2D scan mode only).
        # In 3D PointCloud mode, depth points are not required and often cannot be TF-transformed
        # during rosbag playback (missing TF), so we avoid triggering TF warnings.
        points = None
        if depth is not None and (not self.use_3d_pointcloud) and self.depth_intrinsics is not None:
            points = self._depth_to_points(depth, msg.header)
        
        stamp = stamp_to_sec(msg.header.stamp)
        frame_id = msg.header.frame_id or self.config.get("camera_frame", "camera_link")
        # Store depth array AND points (depth for normals, points for descriptors)
        if depth is not None:
            self.depth_buffer.append((stamp, depth, points, frame_id))
        
        buffer_len = self.config.get("feature_buffer_len", 10)
        if len(self.depth_buffer) > buffer_len:
            self.depth_buffer.pop(0)

        if hasattr(self, "_depth_callback"):
            self._depth_callback(msg)
    
    def _on_camera_info(self, msg: CameraInfo):
        """Extract camera intrinsics."""
        if is_duplicate(self._last_msg_keys, "camera_info", msg.header.stamp, frame_id=msg.header.frame_id):
            return
        self.depth_intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy
        
        # Log first camera info received
        if not hasattr(self, '_first_camera_info_logged'):
            self._first_camera_info_logged = True
            fx, fy, cx, cy = self.depth_intrinsics
            self.node.get_logger().info(
                f"SensorIO: Camera intrinsics received from CameraInfo: "
                f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"
            )

    def _on_imu(self, msg: Imu):
        """
        Buffer IMU measurements for preintegration.

        Uses event-driven clearing: measurements accumulate until explicitly
        cleared after preintegration (no automatic sliding window).
        """
        if is_duplicate(self._last_msg_keys, "imu", msg.header.stamp, frame_id=msg.header.frame_id):
            return

        # Flow counter for data flow audit
        self.imu_callback_count += 1

        stamp = stamp_to_sec(msg.header.stamp)
        # Apply acceleration scale (Livox IMU outputs in g's, need m/s²)
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=float) * self.imu_accel_scale
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=float)

        self.last_imu_frame_id = msg.header.frame_id
        self.imu_buffer.append((stamp, accel, gyro))

        # Debug: Log first IMU received (show both raw and scaled)
        if not hasattr(self, '_first_imu_logged'):
            self._first_imu_logged = True
            accel_raw = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            self.node.get_logger().info(
                f"SensorIO: First IMU received, frame_id={msg.header.frame_id}, "
                f"accel_raw=({accel_raw[0]:.3f}, {accel_raw[1]:.3f}, {accel_raw[2]:.3f}) [g], "
                f"accel_scaled=({accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}) [m/s²], "
                f"gyro=({gyro[0]:.4f}, {gyro[1]:.4f}, {gyro[2]:.4f}) [rad/s], "
                f"scale={self.imu_accel_scale}"
            )

        # Call external callback if set
        if hasattr(self, '_imu_callback'):
            self._imu_callback(msg)

    def _depth_to_points(self, depth: np.ndarray, header) -> Optional[np.ndarray]:
        """Convert depth image to 3D points in base_frame."""
        if self.depth_intrinsics is None:
            self.node.get_logger().debug(
                "Depth to points: depth_intrinsics not set, cannot convert depth to 3D points",
                throttle_duration_sec=5.0,
            )
            return None
        
        fx, fy, cx, cy = self.depth_intrinsics
        stride = self.config.get("depth_stride", 4)
        h, w = depth.shape
        
        ys, xs = np.arange(0, h, stride, dtype=float), np.arange(0, w, stride, dtype=float)
        grid_x, grid_y = np.meshgrid(xs, ys)
        zs = depth[::stride, ::stride]
        
        valid = np.isfinite(zs) & (zs > 0.0)
        if not np.any(valid):
            self.node.get_logger().debug(
                "Depth to points: no valid depth values (all NaN/Inf or zero)",
                throttle_duration_sec=5.0,
            )
            return None
        
        z = zs[valid]
        x = (grid_x[valid] - cx) * z / fx
        y = (grid_y[valid] - cy) * z / fy
        points_cam = np.stack([x, y, z], axis=1)
        
        # Transform to base_frame
        frame_id = header.frame_id or self.config.get("camera_frame", "camera_link")
        transform = self._lookup_transform(self.config["base_frame"], frame_id, header.stamp)
        if transform is None:
            self.node.get_logger().debug(
                f"Depth to points: transform lookup failed ({self.config['base_frame']} <- {frame_id})",
                throttle_duration_sec=5.0,
            )
            return None
        
        return self._transform_points(points_cam, transform)
    
    def scan_to_points(self, msg: LaserScan) -> Optional[np.ndarray]:
        """Convert LaserScan to 3D points in base_frame."""
        ranges = np.asarray(msg.ranges, dtype=float).reshape(-1)
        if ranges.size == 0:
            self.node.get_logger().debug(
                "Scan to points: empty ranges array",
                throttle_duration_sec=5.0,
            )
            return None
        
        angles = msg.angle_min + np.arange(ranges.size, dtype=float) * msg.angle_increment
        valid = np.isfinite(ranges)
        valid &= (ranges >= float(msg.range_min))
        valid &= (ranges <= float(msg.range_max))
        
        if not np.any(valid):
            self.node.get_logger().debug(
                "Scan to points: no valid ranges (all out of range or NaN/Inf)",
                throttle_duration_sec=5.0,
            )
            return None
        
        r = ranges[valid]
        a = angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)
        z = np.zeros_like(x)
        points_scan = np.stack([x, y, z], axis=1)
        
        # Transform to base_frame
        frame_id = msg.header.frame_id or self.config.get("scan_frame", "base_link")
        base_frame = self.config["base_frame"]
        
        # If frames are the same, no transform needed
        if frame_id == base_frame:
            if not hasattr(self, '_logged_scan_no_tf'):
                self._logged_scan_no_tf = True
                self.node.get_logger().info(
                    f"Scan frame '{frame_id}' matches base_frame, no TF transform needed"
                )
            return points_scan
        
        transform = self._lookup_transform(base_frame, frame_id, msg.header.stamp)
        if transform is None:
            if not hasattr(self, '_logged_scan_tf_fail'):
                self._logged_scan_tf_fail = True
                self.node.get_logger().error(
                    f"CRITICAL: Cannot transform scan from '{frame_id}' to '{base_frame}'. "
                    f"TF lookup failed! Check that TF tree is published in rosbag or by robot. "
                    f"Without TF, scan points cannot be used for anchors/loops."
                )
            return None
        
        return self._transform_points(points_scan, transform)

    def _coerce_time(self, stamp):
        """
        Coerce various stamp types into `rclpy.time.Time`.

        `tf2_ros.Buffer.lookup_transform()` expects an `rclpy.time.Time` object.
        Passing `builtin_interfaces.msg.Time` can raise `TypeError` (and crash the node)
        during rosbag playback.
        """
        from rclpy.time import Time

        if stamp is None:
            return Time()

        # Already an rclpy Time
        if isinstance(stamp, Time):
            return stamp

        # Likely builtin_interfaces.msg.Time
        try:
            return Time.from_msg(stamp)
        except Exception:
            # Fallback: "latest available"
            return Time()
    
    def _lookup_transform(self, target_frame: str, source_frame: str, stamp) -> Optional[np.ndarray]:
        """Lookup TF transform as SE(3) pose."""
        try:
            from rclpy.duration import Duration
            from rclpy.time import Time
            timeout = Duration(seconds=self.config.get("tf_timeout_sec", 0.05))
            
            query_time = self._coerce_time(stamp)
            try:
                t = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, query_time, timeout=timeout
                )
            except (TransformException, TypeError, ValueError) as e_first:
                # Bag playback often contains static transforms; if a time-specific lookup
                # fails (extrapolation), fall back to "latest available".
                try:
                    t = self.tf_buffer.lookup_transform(
                        target_frame, source_frame, Time(), timeout=timeout
                    )
                except (TransformException, TypeError, ValueError):
                    raise e_first
            
            trans = t.transform.translation
            rot = t.transform.rotation
            
            # Convert to rotation vector using geometry.se3
            R = quat_to_rotmat(rot.x, rot.y, rot.z, rot.w)
            rotvec = rotmat_to_rotvec(R)
            
            return np.array([trans.x, trans.y, trans.z, rotvec[0], rotvec[1], rotvec[2]], dtype=float)
        
        except (TransformException, TypeError, ValueError) as e:
            self.node.get_logger().warn(
                f"TF lookup failed ({target_frame} ← {source_frame}): {e}",
                throttle_duration_sec=5.0)
            return None
        except Exception as e:
            self.node.get_logger().warn(
                f"TF lookup unexpected error ({target_frame} ← {source_frame}): {type(e).__name__}: {e}",
                throttle_duration_sec=5.0,
            )
            return None
    
    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply SE(3) transform to points."""
        R = rotvec_to_rotmat(transform[3:6])
        t = transform[:3]
        
        # points_new = R @ points^T + t
        return (R @ points.T).T + t
    
    def get_nearest_pose(self, stamp_sec: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get pose nearest to timestamp. Returns (pose, dt)."""
        if not self.odom_buffer:
            self.node.get_logger().debug(
                "get_nearest_pose: odom_buffer empty",
                throttle_duration_sec=10.0,
            )
            return None, None
        closest = min(self.odom_buffer, key=lambda x: abs(x[0] - stamp_sec))
        return closest[1], float(stamp_sec - closest[0])
    
    def get_nearest_image(self, stamp_sec: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get RGB image array nearest to timestamp. Returns (rgb_array, dt)."""
        if not self.image_buffer:
            self.node.get_logger().debug(
                "get_nearest_image: image_buffer empty",
                throttle_duration_sec=10.0,
            )
            return None, None
        closest = min(self.image_buffer, key=lambda x: abs(x[0] - stamp_sec))
        return closest[1], float(stamp_sec - closest[0])
    
    def get_nearest_depth(self, stamp_sec: float) -> Optional[Tuple[float, Optional[np.ndarray], Optional[np.ndarray], str]]:
        """Get depth data nearest to timestamp. Returns (timestamp, depth_array, points, frame_id)."""
        if not self.depth_buffer:
            self.node.get_logger().debug(
                "get_nearest_depth: depth_buffer empty",
                throttle_duration_sec=10.0,
            )
            return None
        return min(self.depth_buffer, key=lambda x: abs(x[0] - stamp_sec))
    
    def get_synchronized_rgbd(self, stamp_sec: float, max_dt: float = 0.05) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[str]]:
        """
        Get synchronized RGB + depth pair nearest to timestamp.
        
        Args:
            stamp_sec: Target timestamp
            max_dt: Maximum time offset for synchronization (seconds)
        
        Returns:
            (rgb_array, depth_array, dt) or (None, None, None) if no sync found
        """
        if not self.image_buffer or not self.depth_buffer:
            self.node.get_logger().debug(
                f"get_synchronized_rgbd: buffers empty (image={len(self.image_buffer)}, depth={len(self.depth_buffer)})",
                throttle_duration_sec=10.0,
            )
            return None, None, None, None
        
        # Find closest depth
        depth_item = min(self.depth_buffer, key=lambda x: abs(x[0] - stamp_sec))
        depth_stamp, depth_array, _, depth_frame = depth_item
        
        # Find closest RGB to the depth timestamp
        rgb_item = min(self.image_buffer, key=lambda x: abs(x[0] - depth_stamp))
        rgb_stamp, rgb_array, _rgb_frame = rgb_item
        
        # Check sync quality
        dt_rgb_depth = abs(rgb_stamp - depth_stamp)
        if dt_rgb_depth > max_dt:
            self.node.get_logger().debug(
                f"get_synchronized_rgbd: sync failed (dt={dt_rgb_depth:.3f}s > max_dt={max_dt:.3f}s)",
                throttle_duration_sec=10.0,
            )
            return None, None, None, None
        
        dt = float(stamp_sec - depth_stamp)
        return rgb_array, depth_array, dt, depth_frame
    
    def get_nearest_pointcloud(self, stamp_sec: float) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
        """
        Get point cloud nearest to timestamp.
        
        Args:
            stamp_sec: Target timestamp in seconds
            
        Returns:
            (points_array, dt, frame_id) or (None, None, None) if no data
        """
        if not self.pointcloud_buffer:
            self.node.get_logger().debug(
                "get_nearest_pointcloud: pointcloud_buffer empty",
                throttle_duration_sec=10.0,
            )
            return None, None, None
        closest = min(self.pointcloud_buffer, key=lambda x: abs(x[0] - stamp_sec))
        stamp, points, frame_id = closest
        return points, float(stamp_sec - stamp), frame_id
    
    def get_latest_pointcloud(self) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
        """
        Get the most recent point cloud.
        
        Returns:
            (points_array, timestamp, frame_id) or (None, None, None) if no data
        """
        if not self.pointcloud_buffer:
            self.node.get_logger().debug(
                "get_latest_pointcloud: pointcloud_buffer empty",
                throttle_duration_sec=10.0,
            )
            return None, None, None
        stamp, points, frame_id = self.pointcloud_buffer[-1]
        return points, stamp, frame_id
    
    def is_3d_mode(self) -> bool:
        """Check if running in 3D point cloud mode."""
        return self.use_3d_pointcloud

    def get_imu_measurements(self, start_sec: float, end_sec: float) -> list:
        """
        Retrieve IMU measurements in time interval [start_sec, end_sec].

        Args:
            start_sec: Start of interval (seconds)
            end_sec: End of interval (seconds)

        Returns:
            List of (timestamp, accel, gyro) tuples sorted by timestamp.
            Each accel/gyro is a numpy array of shape (3,).
        """
        measurements = [
            (t, a.copy(), g.copy())
            for t, a, g in self.imu_buffer
            if start_sec <= t <= end_sec
        ]
        return sorted(measurements, key=lambda x: x[0])

    def clear_imu_buffer(self, before_sec: float) -> int:
        """
        Clear IMU measurements before the specified timestamp.

        Called after preintegration to remove consumed measurements.
        This is the event-driven clearing approach (no automatic sliding window).

        Args:
            before_sec: Clear all measurements with timestamp < before_sec

        Returns:
            Number of measurements cleared
        """
        original_len = len(self.imu_buffer)
        self.imu_buffer = [
            (t, a, g) for t, a, g in self.imu_buffer if t >= before_sec
        ]
        cleared = original_len - len(self.imu_buffer)
        return cleared

    def get_imu_buffer_size(self) -> int:
        """Get current number of buffered IMU measurements."""
        return len(self.imu_buffer)
