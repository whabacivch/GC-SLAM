"""
Golden Child SLAM v2 Rosbag Launch File.

Launches the Golden Child backend with a rosbag for evaluation.

Architecture:
    Rosbag (raw topics)
        │
        ▼
    gc_sensor_hub (single process, MultiThreadedExecutor)
        - livox_converter:  /livox/mid360/lidar → /gc/sensors/lidar_points
        - odom_normalizer:  /odom → /gc/sensors/odom
        - imu_normalizer:   /livox/mid360/imu → /gc/sensors/imu
        - dead_end_audit:   unused topics → /gc/dead_end_status
        │
        ▼
    gc_backend_node (subscribes ONLY to /gc/sensors/*)
        │
        ▼
    Outputs: /gc/state, /gc/trajectory, /gc/status, etc.

Topic Naming Convention:
    Raw (from bag)              Canonical (for backend)
    ─────────────────────────   ────────────────────────────
    /livox/mid360/lidar     →   /gc/sensors/lidar_points
    /odom                   →   /gc/sensors/odom
    /livox/mid360/imu       →   /gc/sensors/imu
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for Golden Child SLAM evaluation."""

    # =========================================================================
    # Launch Arguments
    # =========================================================================
    bag_arg = DeclareLaunchArgument(
        "bag",
        description="Path to rosbag directory",
    )

    trajectory_path_arg = DeclareLaunchArgument(
        "trajectory_export_path",
        default_value="/tmp/gc_slam_trajectory.tum",
        description="Path to export trajectory in TUM format",
    )

    livox_input_msg_type_arg = DeclareLaunchArgument(
        "livox_input_msg_type",
        default_value="livox_ros_driver2/msg/CustomMsg",
        description="Livox CustomMsg type (explicit; no fallback).",
    )

    wiring_summary_path_arg = DeclareLaunchArgument(
        "wiring_summary_path",
        default_value="/tmp/gc_wiring_summary.json",
        description="Path to write wiring summary JSON at end of run.",
    )

    diagnostics_path_arg = DeclareLaunchArgument(
        "diagnostics_export_path",
        default_value="/tmp/gc_slam_diagnostics.npz",
        description="Path to export per-scan diagnostics for dashboard.",
    )

    # =========================================================================
    # Sensor Hub (single process)
    # =========================================================================

    unified_config_path = os.path.join(
        get_package_share_directory("fl_slam_poc"),
        "config",
        "gc_unified.yaml",
    )
    gc_sensor_hub = Node(
        package="fl_slam_poc",
        executable="gc_sensor_hub",
        name="gc_sensor_hub",
        output="screen",
        parameters=[
            {
                "config_path": unified_config_path,
                "livox_input_msg_type": LaunchConfiguration("livox_input_msg_type"),
                "executor_threads": 4,
            }
        ],
    )

    # Wiring auditor: collects status from all nodes and produces end-of-run summary
    wiring_auditor = Node(
        package="fl_slam_poc",
        executable="wiring_auditor",
        name="wiring_auditor",
        output="screen",
        parameters=[
            {
                "output_json_path": LaunchConfiguration("wiring_summary_path"),
                "summary_period_sec": 30.0,  # Periodic log every 30s during run
            }
        ],
    )

    # =========================================================================
    # Backend Node
    # Subscribes ONLY to /gc/sensors/* (canonical topics)
    # =========================================================================
    gc_backend = Node(
        package="fl_slam_poc",
        executable="gc_backend_node",
        name="gc_backend",
        output="screen",
        parameters=[
            {
                # Backend subscribes ONLY to canonical topics
                "lidar_topic": "/gc/sensors/lidar_points",
                "odom_topic": "/gc/sensors/odom",
                "imu_topic": "/gc/sensors/imu",
                # Other params
                "trajectory_export_path": LaunchConfiguration("trajectory_export_path"),
                "diagnostics_export_path": LaunchConfiguration("diagnostics_export_path"),
                "odom_frame": "odom",
                # Bag truth for M3DGR Dynamic01: odom child_frame_id is base_footprint.
                "base_frame": "base_footprint",
                # No-TF extrinsics (T_{base<-sensor}) in [x,y,z,rx,ry,rz] rotvec (rad).
                # Defaults from docs/BAG_TOPICS_AND_USAGE.md (MID-360 mount; identity rotation).
                "T_base_lidar": [-0.011, 0.0, 0.778, 0.0, 0.0, 0.0],
                "T_base_imu": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "status_check_period_sec": 5.0,
                "forgetting_factor": 0.99,
            }
        ],
    )

    # =========================================================================
    # Rosbag Playback
    # =========================================================================
    bag_play = TimerAction(
        period=3.0,  # 3 second delay to let nodes initialize
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "bag", "play",
                    LaunchConfiguration("bag"),
                    "--clock",
                    "--rate", "1.0",
                ],
                output="screen",
            ),
        ],
    )

    return LaunchDescription([
        # Arguments
        bag_arg,
        trajectory_path_arg,
        livox_input_msg_type_arg,
        wiring_summary_path_arg,
        diagnostics_path_arg,
        # Sensor Hub (single process)
        gc_sensor_hub,
        # Audit / observability
        wiring_auditor,
        # Backend
        gc_backend,
        # Rosbag
        bag_play,
    ])
