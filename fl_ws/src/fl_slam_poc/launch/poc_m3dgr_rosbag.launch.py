from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    play_bag = LaunchConfiguration("play_bag")
    bag = LaunchConfiguration("bag")
    bag_start_delay_sec = LaunchConfiguration("bag_start_delay_sec")

    enable_frontend = LaunchConfiguration("enable_frontend")
    enable_backend = LaunchConfiguration("enable_backend")
    enable_odom_bridge = LaunchConfiguration("enable_odom_bridge")

    enable_livox_convert = LaunchConfiguration("enable_livox_convert")
    livox_input_topic = LaunchConfiguration("livox_input_topic")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    pointcloud_frame_id = LaunchConfiguration("pointcloud_frame_id")

    odom_topic = LaunchConfiguration("odom_topic")
    odom_frame = LaunchConfiguration("odom_frame")
    base_frame = LaunchConfiguration("base_frame")

    enable_decompress = LaunchConfiguration("enable_decompress")
    rgb_compressed_topic = LaunchConfiguration("rgb_compressed_topic")
    depth_compressed_topic = LaunchConfiguration("depth_compressed_topic")
    camera_topic = LaunchConfiguration("camera_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    publish_rgbd_evidence = LaunchConfiguration("publish_rgbd_evidence")
    rgbd_evidence_topic = LaunchConfiguration("rgbd_evidence_topic")
    enable_image = LaunchConfiguration("enable_image")
    enable_depth = LaunchConfiguration("enable_depth")
    enable_camera_info = LaunchConfiguration("enable_camera_info")
    camera_fx = LaunchConfiguration("camera_fx")
    camera_fy = LaunchConfiguration("camera_fy")
    camera_cx = LaunchConfiguration("camera_cx")
    camera_cy = LaunchConfiguration("camera_cy")

    sensor_qos_reliability = LaunchConfiguration("sensor_qos_reliability")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("play_bag", default_value="true"),
            DeclareLaunchArgument("bag", default_value=""),
            DeclareLaunchArgument("bag_start_delay_sec", default_value="2.0"),

            DeclareLaunchArgument("enable_frontend", default_value="true"),
            DeclareLaunchArgument("enable_backend", default_value="true"),
            DeclareLaunchArgument("enable_odom_bridge", default_value="true"),

            # Livox CustomMsg -> PointCloud2
            DeclareLaunchArgument("enable_livox_convert", default_value="true"),
            DeclareLaunchArgument("livox_input_topic", default_value="/livox/mid360/lidar"),
            DeclareLaunchArgument("pointcloud_topic", default_value="/camera/depth/points"),
            # CRITICAL FIX: Publish in base_link frame since M3DGR rosbag has no TF
            DeclareLaunchArgument("pointcloud_frame_id", default_value="base_link"),

            # Core topics / frames
            DeclareLaunchArgument("odom_topic", default_value="/odom"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),

            # RGB-D decompression (M3DGR: compressed RGB + compressedDepth)
            DeclareLaunchArgument("enable_decompress", default_value="true"),
            DeclareLaunchArgument("rgb_compressed_topic", default_value="/camera/color/image_raw/compressed"),
            DeclareLaunchArgument("depth_compressed_topic", default_value="/camera/aligned_depth_to_color/image_raw/compressedDepth"),
            DeclareLaunchArgument("camera_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
            # M3DGR Dynamic01 has no CameraInfo topic in the bag by default.
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/depth/camera_info"),

            DeclareLaunchArgument("publish_rgbd_evidence", default_value="true"),
            DeclareLaunchArgument("rgbd_evidence_topic", default_value="/sim/rgbd_evidence"),
            DeclareLaunchArgument("enable_image", default_value="true"),
            DeclareLaunchArgument("enable_depth", default_value="true"),
            DeclareLaunchArgument("enable_camera_info", default_value="false"),
            # Intrinsics fallback (used when enable_camera_info=false)
            # M3DGR RealSense D435i camera intrinsics (640x480)
            # Source: M3DGR dataset documentation
            DeclareLaunchArgument("camera_fx", default_value="383.0"),
            DeclareLaunchArgument("camera_fy", default_value="383.0"),
            DeclareLaunchArgument("camera_cx", default_value="320.0"),
            DeclareLaunchArgument("camera_cy", default_value="240.0"),

            DeclareLaunchArgument(
                "sensor_qos_reliability",
                default_value="both",
                description="QoS reliability for sensor subscriptions: reliable, best_effort, system_default, both",
            ),

            # Livox converter node
            Node(
                package="fl_slam_poc",
                executable="livox_converter",
                name="livox_converter",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "input_topic": livox_input_topic,
                        "output_topic": pointcloud_topic,
                        "frame_id": pointcloud_frame_id,
                    }
                ],
                condition=IfCondition(enable_livox_convert),
            ),

            # Image decompression node
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

            # Frontend (3D pointcloud mode)
            Node(
                package="fl_slam_poc",
                executable="frontend_node",
                name="fl_frontend",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        # CRITICAL FIX: Frontend must use delta odom from odom bridge
                        "odom_is_delta": True,  # Changed from False
                        "odom_topic": "/sim/odom",  # Changed from odom_topic (/odom)
                        "odom_frame": odom_frame,
                        "base_frame": base_frame,

                        "use_3d_pointcloud": True,
                        "enable_pointcloud": True,
                        "pointcloud_topic": pointcloud_topic,

                        "enable_image": enable_image,
                        "enable_depth": enable_depth,
                        "enable_camera_info": enable_camera_info,
                        "camera_topic": camera_topic,
                        "depth_topic": depth_topic,
                        "camera_info_topic": camera_info_topic,

                        "publish_rgbd_evidence": publish_rgbd_evidence,
                        "rgbd_evidence_topic": rgbd_evidence_topic,
                        "camera_fx": camera_fx,
                        "camera_fy": camera_fy,
                        "camera_cx": camera_cx,
                        "camera_cy": camera_cy,

                        "sensor_qos_reliability": sensor_qos_reliability,

                        # Reduced birth intensity to prevent too many anchors diluting responsibilities
                        "birth_intensity": 5.0,
                    }
                ],
                condition=IfCondition(enable_frontend),
            ),

            # Odom bridge (absolute -> delta)
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
                        # Use "both" to handle rosbag QoS variations
                        "qos_reliability": "both",
                        "validate_frames": False,  # Disable frame validation for rosbag compatibility
                    }
                ],
                condition=IfCondition(enable_odom_bridge),
            ),

            # Backend
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
                        "trajectory_export_path": "/tmp/fl_slam_trajectory.tum",
                    }
                ],
                condition=IfCondition(enable_backend),
            ),

            # Rosbag playback (delayed to allow nodes to initialize)
            TimerAction(
                period=bag_start_delay_sec,
                actions=[
                    ExecuteProcess(
                        cmd=["ros2", "bag", "play", bag, "--clock"],
                        output="screen",
                    ),
                ],
                condition=IfCondition(PythonExpression(["'", play_bag, "' == 'true'"])),
            ),
        ]
    )
