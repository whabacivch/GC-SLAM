#!/usr/bin/env python3
"""
Livox CustomMsg to PointCloud2 Converter Node.

Converts Livox proprietary CustomMsg format to standard sensor_msgs/PointCloud2
so FL-SLAM can process Livox LiDAR data.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from rclpy.parameter import Parameter
from typing import Any, Optional, Dict


def _try_import_msg(msg_type: str):
    """
    Import a Livox CustomMsg type by ROS interface name.

    Supported:
      - livox_ros_driver2/msg/CustomMsg
      - livox_ros_driver/msg/CustomMsg (optional; may not be installed)
    """
    if msg_type == "livox_ros_driver2/msg/CustomMsg":
        from livox_ros_driver2.msg import CustomMsg  # type: ignore

        return CustomMsg
    if msg_type == "livox_ros_driver/msg/CustomMsg":
        from livox_ros_driver.msg import CustomMsg  # type: ignore

        return CustomMsg
    raise ValueError(f"Unsupported msg_type: {msg_type}")


def _pointcloud2_fields_for_livox(has_time_offset: bool) -> tuple[list[PointField], int, dict[str, int]]:
    """
    Define a stable PointCloud2 schema that preserves as much Livox info as possible.

    Always includes:
      - x,y,z float32
      - intensity uint8 (from reflectivity)
      - ring uint8 (from line)
      - tag uint8
      - timebase_low uint32, timebase_high uint32 (constant per message; preserved for downstream)

    If per-point time offset exists, includes:
      - time_offset uint32 (units depend on driver; treated as raw)

    Returns: (fields, point_step_bytes, offsets_by_name)
    """
    # Base layout:
    #   x,y,z: 12 bytes
    #   intensity, ring, tag: 3 bytes
    #   pad: 1 byte (to 16)
    #   [time_offset uint32] optional
    #   pad to align timebase at 8-byte boundary
    #   timebase_low uint32 + timebase_high uint32
    fields: list[PointField] = []
    offsets: dict[str, int] = {}

    def add(name: str, offset: int, datatype: int, count: int = 1):
        fields.append(PointField(name=name, offset=offset, datatype=datatype, count=count))
        offsets[name] = offset

    add("x", 0, PointField.FLOAT32)
    add("y", 4, PointField.FLOAT32)
    add("z", 8, PointField.FLOAT32)
    add("intensity", 12, PointField.UINT8)
    add("ring", 13, PointField.UINT8)
    add("tag", 14, PointField.UINT8)

    if has_time_offset:
        add("time_offset", 16, PointField.UINT32)
        add("timebase_low", 20, PointField.UINT32)
        add("timebase_high", 24, PointField.UINT32)
        point_step = 28
    else:
        add("timebase_low", 16, PointField.UINT32)
        add("timebase_high", 20, PointField.UINT32)
        point_step = 24

    return fields, point_step, offsets


class LivoxConverterNode(Node):
    """Converts Livox CustomMsg to PointCloud2."""

    def __init__(self, parameter_overrides: Optional[Dict[str, Any]] = None):
        overrides = None
        if parameter_overrides:
            overrides = [Parameter(k, value=v) for k, v in parameter_overrides.items()]
        super().__init__('livox_converter', parameter_overrides=overrides)
        
        # Parameters
        self.declare_parameter("input_topic", "/livox/mid360/lidar")
        self.declare_parameter("output_topic", "/gc/sensors/lidar_points")
        # If empty, preserve msg.header.frame_id (recommended).
        self.declare_parameter("frame_id", "")
        # Message type selection (explicit, no fallback):
        # - livox_ros_driver2/msg/CustomMsg: MID360 bags (common)
        # - livox_ros_driver/msg/CustomMsg: AVIA bags (optional; may not be installed)
        self.declare_parameter("input_msg_type", "livox_ros_driver2/msg/CustomMsg")
        
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.input_msg_type = str(self.get_parameter("input_msg_type").value)
        
        # QoS: explicit, single-path.
        # - Subscription from rosbag: RELIABLE (bags typically record with RELIABLE).
        # - Publisher to backend: BEST_EFFORT to match GC backend sensor subscription.
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        # Publisher for PointCloud2
        self.publisher = self.create_publisher(
            PointCloud2,
            output_topic,
            qos_pub,
        )

        try:
            MsgT = _try_import_msg(self.input_msg_type)
        except Exception as exc:
            self.get_logger().error(f"Livox converter: cannot import {self.input_msg_type}: {exc}")
            raise RuntimeError(f"Unsupported input_msg_type: {self.input_msg_type}") from exc

        self._subscription = self.create_subscription(
            MsgT,
            input_topic,
            self._on_custom_msg,
            qos_sub,
        )
        
        self._msg_count = 0
        self._logged_schema = False
        self.get_logger().info("Livox converter node started")
        self.get_logger().info(f"  Input:  {input_topic}")
        self.get_logger().info(f"  Input type: {self.input_msg_type}")
        self.get_logger().info(f"  Output: {output_topic}")
        self.get_logger().info(f"  Frame override: {self.frame_id!r} (empty preserves bag frame)")

    def _on_custom_msg(self, msg):
        # Works for livox_ros_driver2 and (optionally) livox_ros_driver messages.
        self._msg_count += 1
        if self._msg_count <= 5 or self._msg_count % 100 == 0:
            self.get_logger().info(f"Livox converter: received CustomMsg #{self._msg_count}")

        points_list = getattr(msg, "points", None)
        if not points_list:
            return

        # Validate message-level accounting if present.
        point_num = getattr(msg, "point_num", None)
        if point_num is not None and int(point_num) != len(points_list):
            if not hasattr(self, "_logged_point_num_mismatch"):
                self._logged_point_num_mismatch = True
                self.get_logger().warn(
                    f"Livox converter: point_num ({int(point_num)}) != len(points) ({len(points_list)}). "
                    "Proceeding with len(points)."
                )

        # Extract core fields.
        xyz = np.array([(p.x, p.y, p.z) for p in points_list], dtype=np.float32)
        valid = np.isfinite(xyz).all(axis=1)
        if not np.any(valid):
            return

        xyz = xyz[valid]
        points_valid = [p for p, ok in zip(points_list, valid) if ok]

        # Intensity / ring / tag are present in livox_ros_driver2 CustomPoint.
        intensity = np.array([int(getattr(p, "reflectivity", 0)) for p in points_valid], dtype=np.uint8)
        ring = np.array([int(getattr(p, "line", 0)) for p in points_valid], dtype=np.uint8)
        tag = np.array([int(getattr(p, "tag", 0)) for p in points_valid], dtype=np.uint8)

        # Per-point time offset: ALWAYS include (required by GC v2 backend)
        # For livox_ros_driver2, offset_time is not in CustomPoint, so we compute synthetic offsets
        # assuming uniform time spacing within the scan (typical for non-repetitive scanning)
        has_time_offset_field = hasattr(points_valid[0], "offset_time") if points_valid else False
        if has_time_offset_field:
            # Use actual offset_time from message (e.g., livox_ros_driver)
            time_offset = np.array([int(getattr(p, "offset_time", 0)) for p in points_valid], dtype=np.uint32)
        else:
            # For livox_ros_driver2: compute synthetic time_offset from point index
            # Assuming uniform time spacing: offset = point_index * (scan_duration_ns / num_points)
            # For MID-360, typical scan duration is ~100ms = 100e6 ns
            # This allows per-point timestamp reconstruction in the backend
            n_points = len(points_valid)
            if n_points > 0:
                # Estimate: 100ms scan duration distributed uniformly across points
                scan_duration_ns = 100_000_000  # 100ms in nanoseconds
                point_indices = np.arange(n_points, dtype=np.uint32)
                time_offset = (point_indices * scan_duration_ns // n_points).astype(np.uint32)
            else:
                time_offset = np.array([], dtype=np.uint32)
        
        # Always include time_offset field (required by GC v2)
        has_time_offset = True

        # timebase is present in livox_ros_driver2 CustomMsg (uint64). Preserve if available.
        timebase = int(getattr(msg, "timebase", 0))
        timebase_low = np.uint32(timebase & 0xFFFFFFFF)
        timebase_high = np.uint32((timebase >> 32) & 0xFFFFFFFF)

        cloud_msg = PointCloud2()
        cloud_msg.header = Header()
        cloud_msg.header.stamp = msg.header.stamp
        cloud_msg.header.frame_id = self.frame_id or msg.header.frame_id
        cloud_msg.height = 1
        cloud_msg.width = int(xyz.shape[0])
        cloud_msg.is_dense = bool(xyz.shape[0] == len(points_list))
        cloud_msg.is_bigendian = False
        fields, point_step, offsets = _pointcloud2_fields_for_livox(has_time_offset=bool(has_time_offset))
        cloud_msg.fields = fields
        cloud_msg.point_step = int(point_step)
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width

        # Pack binary payload deterministically with explicit offsets (no numpy alignment surprises).
        if has_time_offset:
            names = ["x", "y", "z", "intensity", "ring", "tag", "time_offset", "timebase_low", "timebase_high"]
            formats = ["<f4", "<f4", "<f4", "u1", "u1", "u1", "<u4", "<u4", "<u4"]
            offsets_list = [
                offsets["x"],
                offsets["y"],
                offsets["z"],
                offsets["intensity"],
                offsets["ring"],
                offsets["tag"],
                offsets["time_offset"],
                offsets["timebase_low"],
                offsets["timebase_high"],
            ]
            dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets_list, "itemsize": point_step})
            arr = np.zeros((xyz.shape[0],), dtype=dtype)
            arr["x"] = xyz[:, 0]
            arr["y"] = xyz[:, 1]
            arr["z"] = xyz[:, 2]
            arr["intensity"] = intensity
            arr["ring"] = ring
            arr["tag"] = tag
            arr["time_offset"] = time_offset  # type: ignore[assignment]
            arr["timebase_low"] = timebase_low
            arr["timebase_high"] = timebase_high
        else:
            names = ["x", "y", "z", "intensity", "ring", "tag", "timebase_low", "timebase_high"]
            formats = ["<f4", "<f4", "<f4", "u1", "u1", "u1", "<u4", "<u4"]
            offsets_list = [
                offsets["x"],
                offsets["y"],
                offsets["z"],
                offsets["intensity"],
                offsets["ring"],
                offsets["tag"],
                offsets["timebase_low"],
                offsets["timebase_high"],
            ]
            dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets_list, "itemsize": point_step})
            arr = np.zeros((xyz.shape[0],), dtype=dtype)
            arr["x"] = xyz[:, 0]
            arr["y"] = xyz[:, 1]
            arr["z"] = xyz[:, 2]
            arr["intensity"] = intensity
            arr["ring"] = ring
            arr["tag"] = tag
            arr["timebase_low"] = timebase_low
            arr["timebase_high"] = timebase_high

        cloud_msg.data = arr.tobytes()

        # Log schema once for auditability.
        if not self._logged_schema:
            self._logged_schema = True
            self.get_logger().info(
                "Livox converter: publishing PointCloud2 fields: "
                + ", ".join([f"{f.name}@{f.offset}" for f in cloud_msg.fields])
                + f" (point_step={cloud_msg.point_step})"
            )
            if not has_time_offset_field:
                self.get_logger().info(
                    "Livox converter: per-point time offset computed synthetically from point index "
                    "(uniform spacing assumption for livox_ros_driver2)."
                )

        self.publisher.publish(cloud_msg)
        if self._msg_count <= 5 or self._msg_count % 100 == 0:
            self.get_logger().info(
                f"Livox converter: published PointCloud2 #{self._msg_count} with {cloud_msg.width} points"
            )


def main(args=None):
    rclpy.init(args=args)
    node = LivoxConverterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
