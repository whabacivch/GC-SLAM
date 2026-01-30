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

Camera (optional, enable_camera:=true):
    Kimera bag: RGB compressed, depth raw.
    - image_decompress_cpp: rgb_compressed_topic -> /gc/sensors/camera_image (rgb8)
    - depth_passthrough: depth_raw_topic -> /gc/sensors/camera_depth (32FC1 m)
    Defaults: /acl_jackal/forward/color/image_raw/compressed, /acl_jackal/forward/depth/image_rect_raw
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
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
        default_value="results/gc_slam_diagnostics.npz",
        description="Path to export per-scan diagnostics for dashboard.",
    )

    imu_gravity_scale_arg = DeclareLaunchArgument(
        "imu_gravity_scale",
        default_value="1.0",
        description="Scale gravity for IMU evidence and preintegration (1.0 = correct cancellation; 0.0 = ablation).",
    )

    deskew_rotation_only_arg = DeclareLaunchArgument(
        "deskew_rotation_only",
        default_value="false",
        description="Use rotation-only deskew (removes hidden IMU translation leak).",
    )

    enable_timing_arg = DeclareLaunchArgument(
        "enable_timing",
        default_value="true",
        description="Record per-stage timings (ms) in diagnostics for bottleneck analysis.",
    )

    bag_play_rate_arg = DeclareLaunchArgument(
        "bag_play_rate",
        default_value="0.25",
        description="Rosbag playback rate (1.0 = realtime; 0.25 = 1/4 speed to allow more processing time per scan).",
    )

    bag_duration_arg = DeclareLaunchArgument(
        "bag_duration",
        default_value="60",
        description="Play only the first N seconds of the bag (bag timeline).",
    )
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value=os.path.join(get_package_share_directory("fl_slam_poc"), "config", "gc_unified.yaml"),
        description="GC config file (sensor hub defaults).",
    )
    lidar_topic_arg = DeclareLaunchArgument("lidar_topic", default_value="/gc/sensors/lidar_points")
    odom_topic_arg = DeclareLaunchArgument("odom_topic", default_value="/gc/sensors/odom")
    imu_topic_arg = DeclareLaunchArgument("imu_topic", default_value="/gc/sensors/imu")
    base_frame_arg = DeclareLaunchArgument("base_frame", default_value="base_footprint")
    T_base_lidar_arg = DeclareLaunchArgument("T_base_lidar", default_value="[-0.011, 0.0, 0.778, 0.0, 0.0, 0.0]")
    T_base_imu_arg = DeclareLaunchArgument("T_base_imu", default_value="[0.0, 0.0, 0.0, -0.015586, 0.489293, 0.0]")

    # Camera (e.g. Kimera bag: RGB compressed, depth raw)
    enable_camera_arg = DeclareLaunchArgument(
        "enable_camera",
        default_value="false",
        description="If true, run image_decompress_cpp (RGB) and optionally depth_passthrough (raw depth).",
    )
    camera_rgb_compressed_arg = DeclareLaunchArgument(
        "camera_rgb_compressed_topic",
        default_value="/acl_jackal/forward/color/image_raw/compressed",
        description="Compressed RGB topic (e.g. Kimera: /acl_jackal/forward/color/image_raw/compressed).",
    )
    camera_rgb_output_arg = DeclareLaunchArgument(
        "camera_rgb_output_topic",
        default_value="/gc/sensors/camera_image",
        description="Canonical RGB output (sensor_msgs/Image rgb8).",
    )
    camera_depth_compressed_arg = DeclareLaunchArgument(
        "camera_depth_compressed_topic",
        default_value="",
        description="Compressed depth topic (empty for Kimera; bag has raw depth).",
    )
    camera_depth_output_arg = DeclareLaunchArgument(
        "camera_depth_output_topic",
        default_value="/gc/sensors/camera_depth",
        description="Canonical depth output (sensor_msgs/Image 32FC1 m).",
    )
    camera_depth_raw_arg = DeclareLaunchArgument(
        "camera_depth_raw_topic",
        default_value="/acl_jackal/forward/depth/image_rect_raw",
        description="Raw depth topic for passthrough (e.g. Kimera: .../depth/image_rect_raw). Empty = no passthrough.",
    )

    # =========================================================================
    # Image decompression (C++): compressed RGB -> canonical; optional compressed depth
    # =========================================================================
    image_decompress_cpp = Node(
        package="fl_slam_poc",
        executable="image_decompress_cpp",
        name="image_decompress_cpp",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_camera", default="false")),
        parameters=[
            {
                "rgb_compressed_topic": LaunchConfiguration("camera_rgb_compressed_topic"),
                "rgb_output_topic": LaunchConfiguration("camera_rgb_output_topic"),
                "depth_compressed_topic": LaunchConfiguration("camera_depth_compressed_topic"),
                "depth_output_topic": LaunchConfiguration("camera_depth_output_topic"),
                "depth_scale_mm_to_m": True,
                "qos_reliability": "best_effort",
            }
        ],
    )

    # Depth passthrough: raw depth (e.g. Kimera 16UC1) -> canonical 32FC1 m
    depth_passthrough = Node(
        package="fl_slam_poc",
        executable="depth_passthrough",
        name="depth_passthrough",
        output="screen",
        condition=IfCondition(
            PythonExpression(
                [
                    "'", LaunchConfiguration("enable_camera", default="false"), "' == 'true' and '",
                    LaunchConfiguration("camera_depth_raw_topic", default=""), "' != ''",
                ]
            )
        ),
        parameters=[
            {
                "depth_raw_topic": LaunchConfiguration("camera_depth_raw_topic"),
                "depth_output_topic": LaunchConfiguration("camera_depth_output_topic"),
                "scale_mm_to_m": True,
                "qos_depth": 10,
            }
        ],
    )

    # =========================================================================
    # Sensor Hub (single process)
    # =========================================================================

    gc_sensor_hub = Node(
        package="fl_slam_poc",
        executable="gc_sensor_hub",
        name="gc_sensor_hub",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
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
                "lidar_topic": LaunchConfiguration("lidar_topic"),
                "odom_topic": LaunchConfiguration("odom_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                # Other params
                "trajectory_export_path": LaunchConfiguration("trajectory_export_path"),
                "diagnostics_export_path": LaunchConfiguration("diagnostics_export_path"),
                "odom_frame": "odom",
                "base_frame": LaunchConfiguration("base_frame"),
                # No-TF extrinsics (T_{base<-sensor}) in [x,y,z,rx,ry,rz] rotvec (rad).
                "T_base_lidar": LaunchConfiguration("T_base_lidar"),
                "T_base_imu": LaunchConfiguration("T_base_imu"),
                "status_check_period_sec": 5.0,
                "forgetting_factor": 0.99,
                "imu_gravity_scale": LaunchConfiguration("imu_gravity_scale"),
                "deskew_rotation_only": LaunchConfiguration("deskew_rotation_only"),
                "enable_timing": LaunchConfiguration("enable_timing"),
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
                    "--rate", LaunchConfiguration("bag_play_rate"),
                    "--playback-duration", LaunchConfiguration("bag_duration"),
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
        imu_gravity_scale_arg,
        deskew_rotation_only_arg,
        enable_timing_arg,
        bag_play_rate_arg,
        bag_duration_arg,
        config_path_arg,
        lidar_topic_arg,
        odom_topic_arg,
        imu_topic_arg,
        base_frame_arg,
        T_base_lidar_arg,
        T_base_imu_arg,
        enable_camera_arg,
        camera_rgb_compressed_arg,
        camera_rgb_output_arg,
        camera_depth_compressed_arg,
        camera_depth_output_arg,
        camera_depth_raw_arg,
        # Camera: decompress RGB (C++); passthrough raw depth (Python) when camera_depth_raw_topic set
        image_decompress_cpp,
        depth_passthrough,
        # Sensor Hub (single process)
        gc_sensor_hub,
        # Audit / observability
        wiring_auditor,
        # Backend
        gc_backend,
        # Rosbag
        bag_play,
    ])
