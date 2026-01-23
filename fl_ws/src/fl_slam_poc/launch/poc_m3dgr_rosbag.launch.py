import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    # NOTE:
    # The launcher process cannot initialize JAX "for" child node processes.
    # The correct way to ensure JAX is configured before import in node processes
    # is to pass environment variables into the backend node process (see below).

    # LaunchConfiguration objects (not evaluated yet)
    play_bag = LaunchConfiguration("play_bag")
    bag = LaunchConfiguration("bag")
    bag_start_delay_sec = LaunchConfiguration("bag_start_delay_sec")

    enable_frontend = LaunchConfiguration("enable_frontend")
    enable_backend = LaunchConfiguration("enable_backend")
    enable_odom_bridge = LaunchConfiguration("enable_odom_bridge")

    config_base = LaunchConfiguration("config_base")
    config_preset = LaunchConfiguration("config_preset")

    enable_livox_convert = LaunchConfiguration("enable_livox_convert")
    enable_decompress_cpp = LaunchConfiguration("enable_decompress_cpp")

    return [
            # Livox converter node
            Node(
                package="fl_slam_poc",
                executable="livox_converter",
                name="livox_converter",
                output="screen",
                parameters=[config_base, config_preset],
                condition=IfCondition(enable_livox_convert),
            ),

            # Image decompression node (C++ / cv_bridge)
            Node(
                package="fl_slam_poc",
                executable="image_decompress_cpp",
                name="image_decompress_cpp",
                output="screen",
                parameters=[config_base, config_preset],
                condition=IfCondition(enable_decompress_cpp),
            ),

            # Frontend (3D pointcloud mode)
            Node(
                package="fl_slam_poc",
                executable="frontend_node",
                name="fl_frontend",
                output="screen",
                parameters=[config_base, config_preset],
                condition=IfCondition(enable_frontend),
            ),

            # Odom bridge (absolute -> delta)
            Node(
                package="fl_slam_poc",
                executable="odom_bridge",
                name="odom_bridge",
                output="screen",
                parameters=[config_base, config_preset],
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
                parameters=[config_base, config_preset],
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
    pkg_share = get_package_share_directory("fl_slam_poc")
    config_base_default = os.path.join(pkg_share, "config", "fl_slam_poc_base.yaml")
    config_preset_default = os.path.join(pkg_share, "config", "presets", "m3dgr.yaml")
    return LaunchDescription([
        # Declare all launch arguments first
        DeclareLaunchArgument("play_bag", default_value="true"),
        DeclareLaunchArgument("bag", default_value=""),
        DeclareLaunchArgument("bag_start_delay_sec", default_value="2.0"),
        DeclareLaunchArgument("enable_frontend", default_value="true"),
        DeclareLaunchArgument("enable_backend", default_value="true"),
        DeclareLaunchArgument("enable_odom_bridge", default_value="true"),
        DeclareLaunchArgument("config_base", default_value=config_base_default),
        DeclareLaunchArgument("config_preset", default_value=config_preset_default),
        DeclareLaunchArgument("enable_livox_convert", default_value="true"),
        DeclareLaunchArgument("enable_decompress_cpp", default_value="true"),
        # Use OpaqueFunction to evaluate context and create nodes
        OpaqueFunction(function=launch_setup),
    ])
