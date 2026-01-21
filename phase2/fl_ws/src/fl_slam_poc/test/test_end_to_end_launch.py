"""
End-to-end ROS launch tests for FL-SLAM.

Requires full ROS 2 runtime (rclpy + launch_ros + message types).
These tests ONLY run when executed via `launch_test` in a properly built workspace.

To run:
    cd fl_ws
    colcon build --packages-select fl_slam_poc
    source install/setup.bash
    launch_test src/fl_slam_poc/test/test_end_to_end_launch.py
"""

import json
import time
import unittest

import launch
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from launch.actions import TimerAction
from launch_ros.actions import Node as LaunchNode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from fl_slam_poc.msg import AnchorCreate, LoopFactor


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """Generate the launch description for the test."""
    
    static_tf = LaunchNode(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_cam",
        arguments=["0", "0", "0", "0", "0", "0", "odom", "camera_link"],
        output="screen",
    )

    sim_world = LaunchNode(
        package="fl_slam_poc",
        executable="sim_world_node",
        name="sim_world",
        output="screen",
        parameters=[
            {"publish_sensors": True},
            {"publish_anchor": False},
            {"publish_loop_factor": False},
            {"sim_dt": 0.05},
        ],
    )

    frontend = LaunchNode(
        package="fl_slam_poc",
        executable="frontend_node",
        name="fl_frontend",
        output="screen",
        parameters=[
            {"odom_topic": "/sim/odom"},
            {"odom_is_delta": True},
            {"birth_intensity": 1000.0},
            {"scan_period": 0.05},
        ],
    )

    backend = LaunchNode(
        package="fl_slam_poc",
        executable="fl_backend_node",
        name="fl_backend",
        output="screen",
    )

    return (
        launch.LaunchDescription([
            static_tf,
            sim_world,
            frontend,
            backend,
            TimerAction(period=3.0, actions=[launch_testing.actions.ReadyToTest()]),
        ]),
        {
            "static_tf": static_tf,
            "sim_world": sim_world,
            "frontend": frontend,
            "backend": backend,
        },
    )


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for FL-SLAM nodes."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("e2e_test_node")
        cls.loop_msgs = []
        cls.state_msgs = []
        cls.report_msgs = []
        cls.node.create_subscription(LoopFactor, "/sim/loop_factor", cls.loop_msgs.append, 10)
        cls.node.create_subscription(Odometry, "/cdwm/state", cls.state_msgs.append, 10)
        cls.node.create_subscription(String, "/cdwm/op_report", cls.report_msgs.append, 50)
        cls.anchor_pub = cls.node.create_publisher(AnchorCreate, "/sim/anchor_create", 10)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _spin_until(self, condition, timeout_sec: float):
        start = time.time()
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if condition():
                return True
        return False

    def test_1_backend_state_frame(self):
        """Test that backend publishes state with correct frame."""
        ok = self._spin_until(lambda: len(self.state_msgs) > 0, timeout_sec=15.0)
        self.assertTrue(ok, "No backend state published.")

        state = self.state_msgs[-1]
        self.assertEqual(state.header.frame_id, "odom")
        self.assertEqual(len(state.pose.covariance), 36)

    def test_2_loop_factor_emits_with_correct_metadata(self):
        """Test that frontend publishes loop factors with correct metadata."""
        ok = self._spin_until(lambda: len(self.loop_msgs) > 0, timeout_sec=20.0)
        self.assertTrue(ok, "No LoopFactor published by frontend.")

        loop = self.loop_msgs[-1]
        self.assertEqual(loop.header.frame_id, "odom")
        self.assertEqual(loop.solver_name, "ICP")
        self.assertIn("Linearization", list(loop.approximation_triggers))

    def test_3_op_reports_emitted(self):
        """Test that OpReports are being published."""
        ok = self._spin_until(lambda: len(self.report_msgs) > 5, timeout_sec=10.0)
        self.assertTrue(ok, "Not enough OpReports published.")
        
        # Verify reports are valid JSON
        valid_count = 0
        for msg in self.report_msgs[:10]:
            try:
                data = json.loads(msg.data)
                if "name" in data:
                    valid_count += 1
            except json.JSONDecodeError:
                pass
        
        self.assertGreater(valid_count, 0, "No valid OpReports found.")
