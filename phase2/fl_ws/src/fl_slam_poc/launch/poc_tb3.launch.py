"""
GAZEBO SIMULATION (FUTURE/OPTIONAL)

This launch file is intended for live testing in TurtleBot3 Gazebo worlds.
It is not used by the MVP M3DGR rosbag evaluation pipeline.
See `ROADMAP.md` for planned validation work.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    model = LaunchConfiguration("model")
    launch_gazebo = LaunchConfiguration("launch_gazebo")
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_frontend = LaunchConfiguration("enable_frontend")
    scan_topic = LaunchConfiguration("scan_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    camera_topic = LaunchConfiguration("camera_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    sensor_qos_reliability = LaunchConfiguration("sensor_qos_reliability")

    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    tb3_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [tb3_gazebo_dir, "/launch/turtlebot3_world.launch.py"]
        ),
        condition=IfCondition(launch_gazebo),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value="waffle"),
            DeclareLaunchArgument("launch_gazebo", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("enable_frontend", default_value="true"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("odom_topic", default_value="/odom"),
            DeclareLaunchArgument("camera_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/depth/camera_info"),
            DeclareLaunchArgument(
                "sensor_qos_reliability",
                default_value="both",
                description="QoS reliability for sensor subscriptions: reliable, best_effort, system_default, both",
            ),
            SetEnvironmentVariable("TURTLEBOT3_MODEL", model),
            tb3_world,
            Node(
                package="fl_slam_poc",
                executable="frontend_node",
                name="fl_frontend",
                output="screen",
                parameters=[{
                    "use_sim_time": use_sim_time,
                    "odom_is_delta": False,
                    "scan_topic": scan_topic,
                    "odom_topic": odom_topic,
                    "camera_topic": camera_topic,
                    "depth_topic": depth_topic,
                    "camera_info_topic": camera_info_topic,
                    "sensor_qos_reliability": sensor_qos_reliability,
                }],
                condition=IfCondition(enable_frontend),
            ),
            Node(
                package="fl_slam_poc",
                executable="tb3_odom_bridge",
                name="tb3_odom_bridge",
                output="screen",
                parameters=[{
                    "use_sim_time": use_sim_time,
                    "input_topic": odom_topic,
                    "output_topic": "/sim/odom",
                    "qos_reliability": sensor_qos_reliability,
                }],
            ),
            Node(
                package="fl_slam_poc",
                executable="backend_node",
                name="fl_backend",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]
    )
