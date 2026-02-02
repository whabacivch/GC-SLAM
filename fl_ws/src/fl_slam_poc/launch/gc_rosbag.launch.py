"""
Geometric Compositional SLAM v2 Rosbag Launch File.

Launches the Geometric Compositional backend with a rosbag for evaluation.
Default profile: Kimera (PointCloud2 LiDAR, standard IMU).

Architecture:
    Rosbag → gc_sensor_hub (pointcloud_passthrough, odom_normalizer, imu_normalizer, dead_end_audit)
           → /gc/sensors/* → gc_backend_node → /gc/state, /gc/trajectory, etc.

Camera (always enabled): camera_rgbd_node → /gc/sensors/camera_rgbd (single RGBD topic).
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

# Backend parameters that must be numeric/bool (LaunchConfiguration returns strings)
_BACKEND_NUMERIC_PARAMS = {
    "lidar_sigma_meas": float,
    "imu_gravity_scale": float,
    "imu_accel_scale": float,
    "deskew_rotation_only": lambda s: s.strip().lower() in ("true", "1", "yes"),
    "enable_timing": lambda s: s.strip().lower() in ("true", "1", "yes"),
    "use_rerun": lambda s: s.strip().lower() in ("true", "1", "yes"),
    "rerun_spawn": lambda s: s.strip().lower() in ("true", "1", "yes"),
    "odom_belief_diagnostic_max_scans": int,
}


def generate_launch_description():
    """Generate launch description for Geometric Compositional SLAM evaluation."""

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
    splat_export_path_arg = DeclareLaunchArgument(
        "splat_export_path",
        default_value="",
        description="Path to export primitive map (splats) for post-run JAXsplat viz. Empty = do not export.",
    )

    imu_gravity_scale_arg = DeclareLaunchArgument(
        "imu_gravity_scale",
        default_value="1.0",
        description="Scale gravity for IMU evidence and preintegration (1.0 = correct cancellation; 0.0 = ablation).",
    )
    imu_accel_scale_arg = DeclareLaunchArgument(
        "imu_accel_scale",
        default_value="1.0",
        description="IMU linear_acceleration scale: 1.0 when bag publishes m/s² (Kimera/ROS).",
    )

    deskew_rotation_only_arg = DeclareLaunchArgument(
        "deskew_rotation_only",
        default_value="false",
        description="Use rotation-only deskew (removes hidden IMU translation leak).",
    )

    use_rerun_arg = DeclareLaunchArgument(
        "use_rerun",
        default_value="true",
        description="Use Rerun for visualization (Wayland-friendly; replaces RViz).",
    )
    rerun_recording_path_arg = DeclareLaunchArgument(
        "rerun_recording_path",
        default_value="/tmp/gc_slam.rrd",
        description="Path for Rerun recording; open with: rerun <path>. Empty = buffer only.",
    )
    rerun_spawn_arg = DeclareLaunchArgument(
        "rerun_spawn",
        default_value="false",
        description="Spawn Rerun viewer at startup (use_rerun must be true).",
    )

    odom_belief_diagnostic_file_arg = DeclareLaunchArgument(
        "odom_belief_diagnostic_file",
        default_value="",
        description="When non-empty, backend writes CSV of raw odom vs belief (start/end) per scan.",
    )
    odom_belief_diagnostic_max_scans_arg = DeclareLaunchArgument(
        "odom_belief_diagnostic_max_scans",
        default_value="0",
        description="Max scans to log for odom/belief diagnostic (0 = all).",
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
    odom_frame_arg = DeclareLaunchArgument(
        "odom_frame",
        default_value="acl_jackal2/odom",
        description="Odom/parent frame (Kimera: acl_jackal2/odom).",
    )
    base_frame_arg = DeclareLaunchArgument(
        "base_frame",
        default_value="acl_jackal2/base",
        description="Base/child frame (Kimera: acl_jackal2/base).",
    )
    pointcloud_layout_arg = DeclareLaunchArgument(
        "pointcloud_layout",
        default_value="vlp16",
        description="PointCloud2 layout: vlp16 (Kimera VLP-16). See docs/POINTCLOUD2_LAYOUTS.md.",
    )
    # Extrinsics: loaded from config yaml (single source of truth).
    # Launch args available for override but should not be used - config yaml is authoritative.
    extrinsics_source_arg = DeclareLaunchArgument(
        "extrinsics_source",
        default_value="inline",
        description="Extrinsics source: inline (from config yaml) | file (from *_file params).",
    )
    T_base_lidar_file_arg = DeclareLaunchArgument("T_base_lidar_file", default_value="")
    T_base_imu_file_arg = DeclareLaunchArgument("T_base_imu_file", default_value="")
    # No inline defaults - extrinsics come from config yaml only
    T_base_lidar_arg = DeclareLaunchArgument("T_base_lidar", default_value="")
    T_base_imu_arg = DeclareLaunchArgument("T_base_imu", default_value="")
    lidar_sigma_meas_arg = DeclareLaunchArgument(
        "lidar_sigma_meas",
        default_value="0.001",
        description="LiDAR measurement noise prior (m² isotropic). Kimera VLP-16: 1e-3.",
    )

    # Camera (always enabled; backend requires camera for primitive pose evidence)
    camera_rgb_compressed_arg = DeclareLaunchArgument(
        "camera_rgb_compressed_topic",
        default_value="/acl_jackal/forward/color/image_raw/compressed",
        description="Compressed RGB topic (e.g. Kimera: /acl_jackal/forward/color/image_raw/compressed).",
    )
    camera_depth_raw_arg = DeclareLaunchArgument(
        "camera_depth_raw_topic",
        default_value="/acl_jackal/forward/depth/image_rect_raw",
        description="Raw depth topic (e.g. Kimera: .../depth/image_rect_raw).",
    )
    camera_rgbd_output_arg = DeclareLaunchArgument(
        "camera_rgbd_output_topic",
        default_value="/gc/sensors/camera_rgbd",
        description="Canonical RGBD output (fl_slam_poc/RGBDImage).",
    )
    camera_pair_max_dt_arg = DeclareLaunchArgument(
        "camera_pair_max_dt_sec",
        default_value="0.05",
        description="Max |t_rgb - t_depth| (sec) for pairing into one RGBD frame.",
    )

    # =========================================================================
    # Single-path camera RGBD (C++): compressed RGB + raw depth -> RGBDImage
    # =========================================================================
    camera_rgbd_node = Node(
        package="fl_slam_poc",
        executable="camera_rgbd_node",
        name="camera_rgbd_node",
        output="screen",
        parameters=[
            {
                "rgb_compressed_topic": LaunchConfiguration("camera_rgb_compressed_topic"),
                "depth_raw_topic": LaunchConfiguration("camera_depth_raw_topic"),
                "output_topic": LaunchConfiguration("camera_rgbd_output_topic"),
                "depth_scale_mm_to_m": True,
                "pair_max_dt_sec": LaunchConfiguration("camera_pair_max_dt_sec"),
                "qos_reliability": "best_effort",
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
    # Backend Node: load gc_backend.ros__parameters from config_path, merge with launch overrides
    # =========================================================================
    def backend_node_with_config(context):
        config_path = LaunchConfiguration("config_path").perform(context)
        backend_params = {}
        if config_path and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            backend_params = dict(data.get("gc_backend", {}).get("ros__parameters", {}))
        # Launch overrides (take precedence)
        overrides = {
            "lidar_topic": LaunchConfiguration("lidar_topic").perform(context),
            "odom_topic": LaunchConfiguration("odom_topic").perform(context),
            "imu_topic": LaunchConfiguration("imu_topic").perform(context),
            "trajectory_export_path": LaunchConfiguration("trajectory_export_path").perform(context),
            "diagnostics_export_path": LaunchConfiguration("diagnostics_export_path").perform(context),
            "splat_export_path": LaunchConfiguration("splat_export_path").perform(context),
            "odom_frame": LaunchConfiguration("odom_frame").perform(context),
            "base_frame": LaunchConfiguration("base_frame").perform(context),
            "pointcloud_layout": LaunchConfiguration("pointcloud_layout").perform(context),
            "lidar_sigma_meas": LaunchConfiguration("lidar_sigma_meas").perform(context),
            "extrinsics_source": LaunchConfiguration("extrinsics_source").perform(context),
            "T_base_lidar_file": LaunchConfiguration("T_base_lidar_file").perform(context),
            "T_base_imu_file": LaunchConfiguration("T_base_imu_file").perform(context),
            # T_base_lidar, T_base_imu stay from YAML (lists); launch args are string form
            "imu_gravity_scale": LaunchConfiguration("imu_gravity_scale").perform(context),
            "imu_accel_scale": LaunchConfiguration("imu_accel_scale").perform(context),
            "deskew_rotation_only": LaunchConfiguration("deskew_rotation_only").perform(context),
            "enable_timing": LaunchConfiguration("enable_timing").perform(context),
            "odom_belief_diagnostic_file": LaunchConfiguration("odom_belief_diagnostic_file").perform(context),
            "odom_belief_diagnostic_max_scans": LaunchConfiguration("odom_belief_diagnostic_max_scans").perform(context),
            "use_rerun": LaunchConfiguration("use_rerun").perform(context),
            "rerun_recording_path": LaunchConfiguration("rerun_recording_path").perform(context),
            "rerun_spawn": LaunchConfiguration("rerun_spawn").perform(context),
        }
        # Convert string overrides to expected types for rclpy parameter validation
        for key, conv in _BACKEND_NUMERIC_PARAMS.items():
            if key in overrides and overrides[key] != "":
                try:
                    overrides[key] = conv(overrides[key])
                except (ValueError, TypeError):
                    pass
        merged = {**backend_params, **overrides}
        return [
            Node(
                package="fl_slam_poc",
                executable="gc_backend_node",
                name="gc_backend",
                output="screen",
                parameters=[merged],
            )
        ]

    gc_backend_with_config = OpaqueFunction(function=backend_node_with_config)

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
        bag_arg,
        trajectory_path_arg,
        wiring_summary_path_arg,
        diagnostics_path_arg,
        splat_export_path_arg,
        imu_gravity_scale_arg,
        imu_accel_scale_arg,
        deskew_rotation_only_arg,
        enable_timing_arg,
        bag_play_rate_arg,
        bag_duration_arg,
        odom_belief_diagnostic_file_arg,
        odom_belief_diagnostic_max_scans_arg,
        use_rerun_arg,
        rerun_recording_path_arg,
        rerun_spawn_arg,
        config_path_arg,
        lidar_topic_arg,
        odom_topic_arg,
        imu_topic_arg,
        odom_frame_arg,
        base_frame_arg,
        pointcloud_layout_arg,
        lidar_sigma_meas_arg,
        extrinsics_source_arg,
        T_base_lidar_file_arg,
        T_base_imu_file_arg,
        T_base_lidar_arg,
        T_base_imu_arg,
        camera_rgb_compressed_arg,
        camera_depth_raw_arg,
        camera_rgbd_output_arg,
        camera_pair_max_dt_arg,
        # Camera: single-path RGBD (C++)
        camera_rgbd_node,
        # Sensor Hub (single process)
        gc_sensor_hub,
        # Audit / observability
        wiring_auditor,
        # Backend (params from config_path gc_backend.ros__parameters + launch overrides)
        gc_backend_with_config,
        # Rosbag
        bag_play,
    ])
