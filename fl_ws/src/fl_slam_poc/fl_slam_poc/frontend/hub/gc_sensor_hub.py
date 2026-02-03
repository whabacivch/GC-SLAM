"""
=============================================================================
GC SENSOR HUB - Single-Process Frontend for Geometric Compositional SLAM
=============================================================================

Runs all frontend preprocessing in one process (MultiThreadedExecutor).
LiDAR path: pointcloud_passthrough only (PointCloud2 bags, e.g. Kimera/VLP-16).

Architecture:
    Rosbag (raw topics)
        │
        ▼
    gc_sensor_hub: pointcloud_passthrough, odom_normalizer, imu_normalizer, dead_end_audit
        │
        ▼
    /gc/sensors/* (canonical topics for backend)

Reference: docs/GC_SLAM.md
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from fl_slam_poc.frontend.sensors.pointcloud_passthrough import PointcloudPassthroughNode
from fl_slam_poc.frontend.sensors.odom_normalizer import OdomNormalizerNode
from fl_slam_poc.frontend.sensors.imu_normalizer import ImuNormalizerNode
from fl_slam_poc.frontend.audit.dead_end_audit_node import DeadEndAuditNode


@dataclass(frozen=True)
class SensorHubConfig:
    pointcloud_passthrough: Dict[str, Any]
    odom: Dict[str, Any]
    imu: Dict[str, Any]
    dead_end: Dict[str, Any]


def _load_hub_config(path: str) -> SensorHubConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    hub = (data.get("gc_sensor_hub") or {}).get("ros__parameters") or {}
    pointcloud_passthrough = dict(hub.get("pointcloud_passthrough") or {})
    odom = dict(hub.get("odom_normalizer") or {})
    imu = dict(hub.get("imu_normalizer") or {})
    dead_end = dict(hub.get("dead_end_audit") or {})

    missing = [
        name
        for name, cfg in [
            ("pointcloud_passthrough", pointcloud_passthrough),
            ("odom_normalizer", odom),
            ("imu_normalizer", imu),
            ("dead_end_audit", dead_end),
        ]
        if not cfg
    ]
    if missing:
        raise ValueError(f"gc_sensor_hub config missing sections: {missing} (from {path})")
    if not pointcloud_passthrough.get("input_topic"):
        raise ValueError(
            f"gc_sensor_hub config: pointcloud_passthrough.input_topic required (from {path})"
        )

    return SensorHubConfig(
        pointcloud_passthrough=pointcloud_passthrough,
        odom=odom,
        imu=imu,
        dead_end=dead_end,
    )


def _resolve_default_config_path() -> str:
    # Prefer installed share dir when available, else fall back to workspace-relative.
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("fl_slam_poc")
        return os.path.join(share, "config", "gc_unified.yaml")
    except Exception:
        # Workspace-relative fallback for dev.
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "..",
            "..",
            "config",
            "gc_unified.yaml",
        )


def main() -> None:
    """
    Run all frontend nodes in a single process with MultiThreadedExecutor.
    
    Benefits:
        - Lower latency (intra-process communication)
        - Easier debugging (single process)
        - Clearer responsibility boundaries
    """
    rclpy.init()

    # A small hub node just to hold hub-level parameters and logs.
    hub_node = Node("gc_sensor_hub")
    hub_node.declare_parameter("config_path", "")
    hub_node.declare_parameter("executor_threads", 4)

    config_path = str(hub_node.get_parameter("config_path").value).strip()
    if not config_path:
        config_path = _resolve_default_config_path()

    hub_cfg = _load_hub_config(config_path)
    lidar_node = PointcloudPassthroughNode(parameter_overrides=hub_cfg.pointcloud_passthrough)

    # Create all frontend nodes with parameter overrides from config.
    # NOTE: DeadEndAuditNode handles list param type issues internally
    # (see ROS 2 Jazzy bug workaround in dead_end_audit_node.py).
    nodes = [
        hub_node,
        lidar_node,
        OdomNormalizerNode(parameter_overrides=hub_cfg.odom),
        ImuNormalizerNode(parameter_overrides=hub_cfg.imu),
        DeadEndAuditNode(parameter_overrides=hub_cfg.dead_end),
    ]

    threads = int(hub_node.get_parameter("executor_threads").value)
    if threads <= 0:
        raise ValueError("gc_sensor_hub.executor_threads must be > 0")

    executor = MultiThreadedExecutor(num_threads=threads)
    for node in nodes:
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
