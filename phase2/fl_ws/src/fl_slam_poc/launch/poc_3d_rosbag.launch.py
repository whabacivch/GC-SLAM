"""
FL-SLAM 3D Point Cloud Mode Launch File

This launch file configures the system for 3D point cloud input with GPU acceleration.
ALTERNATIVE DATASET/PIPELINE (FUTURE/OPTIONAL): not used by the MVP M3DGR evaluation.
Designed for use with:
- RealSense D455 rosbags
- NVIDIA r2b_storage dataset
- Any rosbag with PointCloud2 messages

Usage:
    # With r2b dataset
    ros2 launch fl_slam_poc poc_3d_rosbag.launch.py bag:=/path/to/r2b_storage

    # With RealSense bag
    ros2 launch fl_slam_poc poc_3d_rosbag.launch.py bag:=/path/to/realsense.db3

    # CPU-only mode
    ros2 launch fl_slam_poc poc_3d_rosbag.launch.py bag:=/path/to/bag use_gpu:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Core configuration
    use_sim_time = LaunchConfiguration("use_sim_time")
    play_bag = LaunchConfiguration("play_bag")
    bag = LaunchConfiguration("bag")
    bag_start_delay_sec = LaunchConfiguration("bag_start_delay_sec")
    qos_overrides_path = LaunchConfiguration("qos_overrides_path")
    enable_qos_overrides = LaunchConfiguration("enable_qos_overrides")

    # Node enables
    enable_frontend = LaunchConfiguration("enable_frontend")
    enable_backend = LaunchConfiguration("enable_backend")
    enable_odom_bridge = LaunchConfiguration("enable_odom_bridge")
    
    # 3D Point Cloud Configuration
    use_3d_pointcloud = LaunchConfiguration("use_3d_pointcloud")
    use_gpu = LaunchConfiguration("use_gpu")
    gpu_device_index = LaunchConfiguration("gpu_device_index")
    gpu_fallback_to_cpu = LaunchConfiguration("gpu_fallback_to_cpu")
    voxel_size = LaunchConfiguration("voxel_size")
    max_correspondence_distance = LaunchConfiguration("max_correspondence_distance")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    pointcloud_rate_limit_hz = LaunchConfiguration("pointcloud_rate_limit_hz")
    min_points_for_icp = LaunchConfiguration("min_points_for_icp")

    # Sensor topics
    scan_topic = LaunchConfiguration("scan_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    camera_topic = LaunchConfiguration("camera_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    
    # RGB-D evidence
    rgbd_evidence_topic = LaunchConfiguration("rgbd_evidence_topic")
    publish_rgbd_evidence = LaunchConfiguration("publish_rgbd_evidence")
    rgbd_publish_every_n_scans = LaunchConfiguration("rgbd_publish_every_n_scans")
    rgbd_max_points_per_msg = LaunchConfiguration("rgbd_max_points_per_msg")
    
    # Feature enables
    enable_image = LaunchConfiguration("enable_image")
    enable_depth = LaunchConfiguration("enable_depth")
    enable_camera_info = LaunchConfiguration("enable_camera_info")
    enable_pointcloud = LaunchConfiguration("enable_pointcloud")
    
    # Frames
    scan_frame = LaunchConfiguration("scan_frame")
    base_frame = LaunchConfiguration("base_frame")
    odom_frame = LaunchConfiguration("odom_frame")
    sensor_qos_reliability = LaunchConfiguration("sensor_qos_reliability")

    default_qos_path = (
        get_package_share_directory("fl_slam_poc") + "/config/qos_override.yaml"
    )

    return LaunchDescription(
        [
            # Core arguments
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("play_bag", default_value="false"),
            DeclareLaunchArgument("bag", default_value=""),
            DeclareLaunchArgument("bag_start_delay_sec", default_value="2.0"),
            DeclareLaunchArgument("enable_qos_overrides", default_value="true"),
            DeclareLaunchArgument("qos_overrides_path", default_value=default_qos_path),
            DeclareLaunchArgument("enable_frontend", default_value="true"),
            DeclareLaunchArgument("enable_backend", default_value="true"),
            DeclareLaunchArgument("enable_odom_bridge", default_value="true"),
            
            # 3D Point Cloud Mode Arguments
            DeclareLaunchArgument("use_3d_pointcloud", default_value="true",
                description="Enable 3D point cloud mode (vs 2D LaserScan)"),
            DeclareLaunchArgument("use_gpu", default_value="true",
                description="Enable GPU acceleration for ICP"),
            DeclareLaunchArgument("gpu_device_index", default_value="0",
                description="CUDA device index"),
            DeclareLaunchArgument("gpu_fallback_to_cpu", default_value="true",
                description="Fall back to CPU if GPU unavailable"),
            DeclareLaunchArgument("voxel_size", default_value="0.05",
                description="Voxel grid filter size in meters"),
            DeclareLaunchArgument("max_correspondence_distance", default_value="0.5",
                description="Maximum ICP correspondence distance in meters"),
            DeclareLaunchArgument("pointcloud_topic", default_value="/camera/depth/points",
                description="PointCloud2 topic to subscribe to"),
            DeclareLaunchArgument("pointcloud_rate_limit_hz", default_value="30.0",
                description="Rate limit for point cloud processing"),
            DeclareLaunchArgument("min_points_for_icp", default_value="100",
                description="Minimum points required for ICP"),
            
            # Sensor topics (r2b_storage / RealSense defaults)
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("odom_topic", default_value="/odom"),
            DeclareLaunchArgument("camera_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/color/camera_info"),
            
            # RGB-D evidence
            DeclareLaunchArgument("rgbd_evidence_topic", default_value="/sim/rgbd_evidence"),
            DeclareLaunchArgument("publish_rgbd_evidence", default_value="true"),
            DeclareLaunchArgument("rgbd_publish_every_n_scans", default_value="5"),
            DeclareLaunchArgument("rgbd_max_points_per_msg", default_value="500"),
            
            # Feature enables (3D mode typically uses pointcloud, but can also use RGB-D)
            DeclareLaunchArgument("enable_image", default_value="true",
                description="Enable RGB image processing"),
            DeclareLaunchArgument("enable_depth", default_value="true",
                description="Enable depth image processing"),
            DeclareLaunchArgument("enable_camera_info", default_value="true",
                description="Enable camera intrinsics subscription"),
            DeclareLaunchArgument("enable_pointcloud", default_value="true",
                description="Subscribe to PointCloud2 messages (automatically enabled with use_3d_pointcloud)"),
            
            # Frames
            DeclareLaunchArgument("scan_frame", default_value="base_link"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument(
                "sensor_qos_reliability",
                default_value="both",
                description="QoS reliability for sensor subscriptions: reliable, best_effort, system_default, both",
            ),
            
            # Frontend Node (3D Point Cloud Mode)
            Node(
                package="fl_slam_poc",
                executable="frontend_node",
                name="fl_frontend",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "odom_is_delta": False,
                        "tf_timeout_sec": 1.0,
                        
                        # 3D Point Cloud Configuration
                        "use_3d_pointcloud": use_3d_pointcloud,
                        "enable_pointcloud": enable_pointcloud,
                        "pointcloud_topic": pointcloud_topic,
                        
                        # GPU Configuration
                        "use_gpu": use_gpu,
                        "gpu_device_index": gpu_device_index,
                        "gpu_fallback_to_cpu": gpu_fallback_to_cpu,
                        
                        # Point Cloud Processing
                        "voxel_size": voxel_size,
                        "icp_max_correspondence_distance": max_correspondence_distance,
                        "pointcloud_rate_limit_hz": pointcloud_rate_limit_hz,
                        "min_points_for_icp": min_points_for_icp,
                        "max_points_after_filter": 50000,
                        "normal_estimation_radius": 0.1,
                        
                        # Sensor topics
                        "scan_topic": scan_topic,
                        "odom_topic": odom_topic,
                        "camera_topic": camera_topic,
                        "depth_topic": depth_topic,
                        "camera_info_topic": camera_info_topic,
                        
                        # Feature enables
                        "enable_image": enable_image,
                        "enable_depth": enable_depth,
                        "enable_camera_info": enable_camera_info,
                        
                        # RGB-D evidence
                        "rgbd_evidence_topic": rgbd_evidence_topic,
                        "publish_rgbd_evidence": publish_rgbd_evidence,
                        "rgbd_publish_every_n_scans": rgbd_publish_every_n_scans,
                        "rgbd_max_points_per_msg": rgbd_max_points_per_msg,
                        
                        # Frames
                        "scan_frame": scan_frame,
                        "base_frame": base_frame,
                        "odom_frame": odom_frame,
                        "sensor_qos_reliability": sensor_qos_reliability,
                    }
                ],
                condition=IfCondition(enable_frontend),
            ),
            
            # Odom Bridge
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
            
            # Backend Node
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
            
            # Bag playback (without QoS overrides)
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
            
            # Bag playback (with QoS overrides)
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
