from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from typing import List


def launch_setup(context, *args, **kwargs):
    # NOTE:
    # The launcher process cannot initialize JAX "for" child node processes.
    # The correct way to ensure JAX is configured before import in node processes
    # is to pass environment variables into the backend node process (see below).

    # Evaluate LaunchConfiguration values to actual Python values for gravity list
    gravity_x_val = float(context.launch_configurations.get("gravity_x", "0.0"))
    gravity_y_val = float(context.launch_configurations.get("gravity_y", "0.0"))
    gravity_z_val = float(context.launch_configurations.get("gravity_z", "-9.81"))
    gravity_list = [gravity_x_val, gravity_y_val, gravity_z_val]  # Real Python list
    # NOTE: launch's typed substitution requires typing.List[T], not built-in list.
    gravity_value = ParameterValue(gravity_list, value_type=List[float])

    # Create LaunchConfiguration objects for other parameters (not evaluated yet)
    use_sim_time = LaunchConfiguration("use_sim_time")
    play_bag = LaunchConfiguration("play_bag")
    bag = LaunchConfiguration("bag")
    bag_start_delay_sec = LaunchConfiguration("bag_start_delay_sec")

    enable_frontend = LaunchConfiguration("enable_frontend")
    enable_backend = LaunchConfiguration("enable_backend")
    enable_odom_bridge = LaunchConfiguration("enable_odom_bridge")

    enable_livox_convert = LaunchConfiguration("enable_livox_convert")
    livox_input_topic = LaunchConfiguration("livox_input_topic")
    livox_input_msg_type = LaunchConfiguration("livox_input_msg_type")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    pointcloud_frame_id = LaunchConfiguration("pointcloud_frame_id")
    lidar_base_extrinsic = LaunchConfiguration("lidar_base_extrinsic")

    odom_topic = LaunchConfiguration("odom_topic")
    odom_frame = LaunchConfiguration("odom_frame")
    base_frame = LaunchConfiguration("base_frame")

    enable_decompress_cpp = LaunchConfiguration("enable_decompress_cpp")
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

    # IMU Integration
    enable_imu = LaunchConfiguration("enable_imu")
    imu_topic = LaunchConfiguration("imu_topic")
    imu_gyro_noise_density = LaunchConfiguration("imu_gyro_noise_density")
    imu_accel_noise_density = LaunchConfiguration("imu_accel_noise_density")
    keyframe_translation_threshold = LaunchConfiguration("keyframe_translation_threshold")
    keyframe_rotation_threshold = LaunchConfiguration("keyframe_rotation_threshold")

    return [
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
                        "input_msg_type": livox_input_msg_type,
                        "output_topic": pointcloud_topic,
                        "frame_id": pointcloud_frame_id,
                    }
                ],
                condition=IfCondition(enable_livox_convert),
            ),

            # Image decompression node (C++ / cv_bridge)
            Node(
                package="fl_slam_poc",
                executable="image_decompress_cpp",
                name="image_decompress_cpp",
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
                condition=IfCondition(enable_decompress_cpp),
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
                        # Optional no-TF LiDAR extrinsic
                        "lidar_base_extrinsic": lidar_base_extrinsic,

                        # IMU Integration
                        "enable_imu": enable_imu,
                        "imu_topic": imu_topic,
                        "imu_gyro_noise_density": imu_gyro_noise_density,
                        "imu_accel_noise_density": imu_accel_noise_density,
                        "keyframe_translation_threshold": keyframe_translation_threshold,
                        "keyframe_rotation_threshold": keyframe_rotation_threshold,
                        "gravity": gravity_value,

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
                        # Use "reliable" to avoid race condition with dual subscriptions
                        "qos_reliability": "reliable",
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
                additional_env={
                    # Ensure backend process inherits JAX config BEFORE any import.
                    # Use "cuda" (not "gpu") to avoid JAX attempting ROCm first.
                    "JAX_PLATFORMS": "cuda",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                },
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "odom_frame": odom_frame,
                        "rgbd_evidence_topic": rgbd_evidence_topic,
                        "trajectory_export_path": "/tmp/fl_slam_trajectory.tum",

                        # IMU Integration parameters
                        "enable_imu_fusion": enable_imu,
                        "gravity": gravity_value,
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


def generate_launch_description():
    return LaunchDescription([
        # Declare all launch arguments first
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("play_bag", default_value="true"),
        DeclareLaunchArgument("bag", default_value=""),
        DeclareLaunchArgument("bag_start_delay_sec", default_value="2.0"),
        DeclareLaunchArgument("enable_frontend", default_value="true"),
        DeclareLaunchArgument("enable_backend", default_value="true"),
        DeclareLaunchArgument("enable_odom_bridge", default_value="true"),
        DeclareLaunchArgument("enable_livox_convert", default_value="true"),
        DeclareLaunchArgument("livox_input_topic", default_value="/livox/mid360/lidar"),
        DeclareLaunchArgument("pointcloud_topic", default_value="/lidar/points"),
        DeclareLaunchArgument("pointcloud_frame_id", default_value="livox_frame"),
        DeclareLaunchArgument("livox_input_msg_type", default_value="auto"),
        # MID-360 mounting extrinsic (no-TF bags): T_b_mid360 = [x,y,z,rx,ry,rz]
        # For M3DGR Dynamic01, we default this to the provided MID-360 calibration:
        # t=[-0.011, 0.0, 0.778], R=I.
        DeclareLaunchArgument("lidar_base_extrinsic", default_value="[-0.011, 0.0, 0.778, 0.0, 0.0, 0.0]"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("odom_frame", default_value="odom_combined"),
        DeclareLaunchArgument("base_frame", default_value="base_footprint"),
        DeclareLaunchArgument("enable_decompress_cpp", default_value="true"),
        DeclareLaunchArgument("rgb_compressed_topic", default_value="/camera/color/image_raw/compressed"),
        DeclareLaunchArgument("depth_compressed_topic", default_value="/camera/aligned_depth_to_color/image_raw/compressedDepth"),
        DeclareLaunchArgument("camera_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/depth/camera_info"),
        DeclareLaunchArgument("publish_rgbd_evidence", default_value="false"),
        DeclareLaunchArgument("rgbd_evidence_topic", default_value="/sim/rgbd_evidence"),
        DeclareLaunchArgument("enable_image", default_value="false"),
        DeclareLaunchArgument("enable_depth", default_value="false"),
        DeclareLaunchArgument("enable_camera_info", default_value="false"),
        # M3DGR: bag does not include CameraInfo; intrinsics must be declared.
        # RealSense 640x480 (dataset values).
        DeclareLaunchArgument("camera_fx", default_value="610.16"),
        DeclareLaunchArgument("camera_fy", default_value="610.45"),
        DeclareLaunchArgument("camera_cx", default_value="326.35"),
        DeclareLaunchArgument("camera_cy", default_value="244.68"),
        DeclareLaunchArgument("sensor_qos_reliability", default_value="reliable"),
        DeclareLaunchArgument("enable_imu", default_value="true"),
        # M3DGR: primary IMU topic should be Livox MID-360 IMU.
        DeclareLaunchArgument("imu_topic", default_value="/livox/mid360/imu"),
        # M3DGR (Xsens-class IMU parameters; random-walk terms are intentionally NOT used).
        DeclareLaunchArgument("imu_gyro_noise_density", default_value="1.7e-4"),
        DeclareLaunchArgument("imu_accel_noise_density", default_value="1.9e-4"),
        DeclareLaunchArgument("keyframe_translation_threshold", default_value="0.5"),
        DeclareLaunchArgument("keyframe_rotation_threshold", default_value="0.26"),
        DeclareLaunchArgument("gravity_x", default_value="0.0"),
        DeclareLaunchArgument("gravity_y", default_value="0.0"),
        DeclareLaunchArgument("gravity_z", default_value="-9.8051"),
        # Use OpaqueFunction to evaluate context and create nodes
        OpaqueFunction(function=launch_setup),
    ])
