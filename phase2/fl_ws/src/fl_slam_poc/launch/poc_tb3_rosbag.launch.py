"""
ALTERNATIVE DATASET (2D/TB3 ROSBAG) - FUTURE/OPTIONAL

This launch file is kept for validating non-M3DGR datasets.
It is not used by the MVP `scripts/run_and_evaluate.sh` pipeline.
See `ROADMAP.md` for planned validation work.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    play_bag = LaunchConfiguration("play_bag")
    bag = LaunchConfiguration("bag")
    bag_start_delay_sec = LaunchConfiguration("bag_start_delay_sec")
    qos_overrides_path = LaunchConfiguration("qos_overrides_path")
    enable_qos_overrides = LaunchConfiguration("enable_qos_overrides")

    enable_frontend = LaunchConfiguration("enable_frontend")
    enable_backend = LaunchConfiguration("enable_backend")
    enable_odom_bridge = LaunchConfiguration("enable_odom_bridge")
    
    # RGB-D decompression (NEW - for rosbags with compressed images)
    enable_decompress = LaunchConfiguration("enable_decompress")
    rgb_compressed_topic = LaunchConfiguration("rgb_compressed_topic")
    depth_compressed_topic = LaunchConfiguration("depth_compressed_topic")

    scan_topic = LaunchConfiguration("scan_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    camera_topic = LaunchConfiguration("camera_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    rgbd_evidence_topic = LaunchConfiguration("rgbd_evidence_topic")
    publish_rgbd_evidence = LaunchConfiguration("publish_rgbd_evidence")
    rgbd_publish_every_n_scans = LaunchConfiguration("rgbd_publish_every_n_scans")
    rgbd_max_points_per_msg = LaunchConfiguration("rgbd_max_points_per_msg")
    enable_image = LaunchConfiguration("enable_image")
    enable_depth = LaunchConfiguration("enable_depth")
    enable_camera_info = LaunchConfiguration("enable_camera_info")
    scan_frame = LaunchConfiguration("scan_frame")
    base_frame = LaunchConfiguration("base_frame")
    odom_frame = LaunchConfiguration("odom_frame")
    sensor_qos_reliability = LaunchConfiguration("sensor_qos_reliability")
    
    # 3D Point Cloud Mode (optional)
    use_3d_pointcloud = LaunchConfiguration("use_3d_pointcloud")
    use_gpu = LaunchConfiguration("use_gpu")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    voxel_size = LaunchConfiguration("voxel_size")

    default_qos_path = (
        get_package_share_directory("fl_slam_poc") + "/config/qos_override.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("play_bag", default_value="false"),
            DeclareLaunchArgument("bag", default_value=""),
            DeclareLaunchArgument("bag_start_delay_sec", default_value="2.0"),
            DeclareLaunchArgument("enable_qos_overrides", default_value="true"),
            DeclareLaunchArgument("qos_overrides_path", default_value=default_qos_path),
            DeclareLaunchArgument("enable_frontend", default_value="true"),
            DeclareLaunchArgument("enable_backend", default_value="true"),
            DeclareLaunchArgument("enable_odom_bridge", default_value="true"),
            # RGB-D decompression (NEW)
            DeclareLaunchArgument("enable_decompress", default_value="true",
                description="Enable image decompression for rosbags with compressed RGB-D"),
            DeclareLaunchArgument("rgb_compressed_topic", 
                default_value="/stereo_camera/left/image_rect_color/compressed/throttled",
                description="Compressed RGB topic from rosbag"),
            DeclareLaunchArgument("depth_compressed_topic",
                default_value="/stereo_camera/depth/depth_registered/compressedDepth/throttled",
                description="Compressed depth topic from rosbag"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("odom_topic", default_value="/odom"),
            # Decompressed output topics (fed to frontend)
            DeclareLaunchArgument("camera_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
            # Camera info directly from rosbag (not compressed)
            DeclareLaunchArgument("camera_info_topic", default_value="/stereo_camera/left/camera_info"),
            DeclareLaunchArgument("rgbd_evidence_topic", default_value="/sim/rgbd_evidence"),
            DeclareLaunchArgument("publish_rgbd_evidence", default_value="true"),
            DeclareLaunchArgument("rgbd_publish_every_n_scans", default_value="5"),
            DeclareLaunchArgument("rgbd_max_points_per_msg", default_value="500"),
            DeclareLaunchArgument("scan_frame", default_value="base_scan"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument(
                "sensor_qos_reliability",
                default_value="both",
                description="QoS reliability for sensor subscriptions: reliable, best_effort, system_default, both",
            ),
            # Camera ENABLED by default (rosbag has RGB-D data!)
            DeclareLaunchArgument("enable_image", default_value="true",
                description="Enable RGB image processing (requires cv_bridge)"),
            DeclareLaunchArgument("enable_depth", default_value="true",
                description="Enable depth image processing"),
            DeclareLaunchArgument("enable_camera_info", default_value="true",
                description="Enable camera intrinsics subscription"),
            # 3D Point Cloud Mode (disabled by default for TurtleBot3)
            DeclareLaunchArgument("use_3d_pointcloud", default_value="false",
                description="Enable 3D point cloud mode (vs 2D LaserScan)"),
            DeclareLaunchArgument("use_gpu", default_value="false",
                description="Enable GPU acceleration for ICP"),
            DeclareLaunchArgument("pointcloud_topic", default_value="/camera/depth/points",
                description="PointCloud2 topic to subscribe to"),
            DeclareLaunchArgument("voxel_size", default_value="0.05",
                description="Voxel grid filter size in meters"),
            # Image decompression node (decompresses rosbag compressed images)
            Node(
                package="fl_slam_poc",
                executable="image_decompress",
                name="image_decompress",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "rgb_compressed_topic": rgb_compressed_topic,
                        "depth_compressed_topic": depth_compressed_topic,
                        "rgb_output_topic": camera_topic,
                        "depth_output_topic": depth_topic,
                        "depth_scale_mm_to_m": True,
                        "qos_reliability": sensor_qos_reliability,
                    }
                ],
                condition=IfCondition(enable_decompress),
            ),
            Node(
                package="fl_slam_poc",
                executable="frontend_node",
                name="fl_frontend",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "odom_is_delta": False,
                        "tf_timeout_sec": 1.0,  # Increased for bag playback tolerance
                        "scan_topic": scan_topic,
                        "odom_topic": odom_topic,
                        "camera_topic": camera_topic,
                        "depth_topic": depth_topic,
                        "camera_info_topic": camera_info_topic,
                        "enable_image": enable_image,
                        "enable_depth": enable_depth,
                        "enable_camera_info": enable_camera_info,
                        "rgbd_evidence_topic": rgbd_evidence_topic,
                        "publish_rgbd_evidence": publish_rgbd_evidence,
                        "rgbd_publish_every_n_scans": rgbd_publish_every_n_scans,
                        "rgbd_max_points_per_msg": rgbd_max_points_per_msg,
                        "scan_frame": scan_frame,
                        "base_frame": base_frame,
                        "odom_frame": odom_frame,
                        "sensor_qos_reliability": sensor_qos_reliability,
                        # 3D Point Cloud Mode
                        "use_3d_pointcloud": use_3d_pointcloud,
                        "enable_pointcloud": use_3d_pointcloud,  # Auto-enable with 3D mode
                        "pointcloud_topic": pointcloud_topic,
                        "use_gpu": use_gpu,
                        "voxel_size": voxel_size,
                        "gpu_fallback_to_cpu": True,
                    }
                ],
                condition=IfCondition(enable_frontend),
            ),
            Node(
                package="fl_slam_poc",
                executable="tb3_odom_bridge",
                name="tb3_odom_bridge",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "input_topic": odom_topic,
                        "output_topic": "/sim/odom",
                        "output_frame": odom_frame,
                        "child_frame": base_frame,
                        "qos_reliability": sensor_qos_reliability,
                    }
                ],
                condition=IfCondition(enable_odom_bridge),
            ),
            Node(
                package="fl_slam_poc",
                executable="backend_node",
                name="fl_backend",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "odom_frame": odom_frame,
                        "rgbd_evidence_topic": rgbd_evidence_topic,
                    }
                ],
                condition=IfCondition(enable_backend),
            ),
            TimerAction(
                period=bag_start_delay_sec,
                actions=[
                    ExecuteProcess(
                        cmd=["ros2", "bag", "play", bag, "--clock"],
                        output="screen",
                    ),
                ],
                condition=IfCondition(PythonExpression(["'", play_bag, "' == 'true' and '", enable_qos_overrides, "' != 'true'"])),
            ),
            TimerAction(
                period=bag_start_delay_sec,
                actions=[
                    ExecuteProcess(
                        cmd=[
                            "ros2",
                            "bag",
                            "play",
                            bag,
                            "--clock",
                            "--qos-profile-overrides-path",
                            qos_overrides_path,
                        ],
                        output="screen",
                    ),
                ],
                condition=IfCondition(PythonExpression(["'", play_bag, "' == 'true' and '", enable_qos_overrides, "' == 'true'"])),
            ),
        ]
    )
