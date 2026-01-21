"""
Simulation World Node - Ground Truth & Odometry Provider.

GAZEBO ONLY: Not used in the MVP M3DGR rosbag evaluation pipeline.

Provides ground truth trajectory and noisy odometry for testing/evaluation.
Does NOT provide synthetic sensors - those should come from a real simulator
(e.g., Gazebo) for physically consistent data.

Publishes:
    /sim/ground_truth (Odometry): True pose for ATE/RPE evaluation
    /odom (Odometry): Noisy absolute odometry (standard topic for frontend/odom_bridge)
    /cdwm/world_markers (MarkerArray): Optional world visualization

Does NOT publish:
    - Synthetic sensors (/scan, /camera/*) - use a real simulator
    - Synthetic loop factors - frontend generates from real associations
    - Synthetic anchors - frontend creates from real observations

State Representation:
    Uses rotation vector (axis-angle) for state, NOT RPY.
    This maintains consistency with the backend's SE(3) representation.

Reference: Barfoot (2017), Sola et al. (2018)
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from fl_slam_poc.common.transforms.se3 import (
    rotvec_to_rotmat,
    rotmat_to_quat,
    se3_compose,
)


class SimWorld(Node):
    def __init__(self):
        super().__init__("sim_world")

        # Publishers
        self.pub_ground_truth = self.create_publisher(Odometry, "/sim/ground_truth", 50)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 50)
        self.pub_world_markers = self.create_publisher(MarkerArray, "/cdwm/world_markers", 1)
        self.pub_gt_path = self.create_publisher(Path, "/sim/ground_truth_path", 10)
        
        # Ground truth path storage for Foxglove visualization
        self.gt_path_poses: list[PoseStamped] = []
        self.max_path_length = 2000

        # Frame parameters
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")
        
        # Simulation parameters
        self.declare_parameter("sim_dt", 0.05)
        self.declare_parameter("sim_duration", 60.0)
        self.declare_parameter("linear_velocity", 0.3)
        self.declare_parameter("angular_velocity_factor", 1.0)
        
        # Trajectory type: "circle", "figure8", "straight"
        self.declare_parameter("trajectory_type", "circle")
        
        # Noise parameters for odometry
        self.declare_parameter("odom_noise_trans", 0.005)
        self.declare_parameter("odom_noise_rot", 0.002)
        
        # Optional features
        self.declare_parameter("publish_world_markers", True)
        self.declare_parameter("world_marker_period", 2.0)

        self._init_from_params()

    def _init_from_params(self):
        """Initialize from parameters."""
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        
        self.dt = float(self.get_parameter("sim_dt").value)
        self.T_total = float(self.get_parameter("sim_duration").value)
        self.v = float(self.get_parameter("linear_velocity").value)
        w_factor = float(self.get_parameter("angular_velocity_factor").value)
        self.w = w_factor * 2.0 * math.pi / self.T_total
        
        self.trajectory_type = str(self.get_parameter("trajectory_type").value)
        
        noise_trans = float(self.get_parameter("odom_noise_trans").value)
        noise_rot = float(self.get_parameter("odom_noise_rot").value)
        self.odom_sigma = np.array([noise_trans, noise_trans, noise_trans,
                                     noise_rot * 0.5, noise_rot * 0.5, noise_rot])
        
        self.publish_world_markers = bool(self.get_parameter("publish_world_markers").value)
        world_marker_period = float(self.get_parameter("world_marker_period").value)
        
        # World marker timer
        self._world_timer = None
        if self.publish_world_markers:
            self._world_timer = self.create_timer(world_marker_period, self._publish_world_timer)

        # Ground truth state in SE(3) form: (x, y, z, rx, ry, rz) with rotation vector
        self.x_true = np.zeros(6, dtype=float)
        
        # Odometry state (accumulated with noise)
        self.x_odom = np.zeros(6, dtype=float)
        
        # Accumulated covariance for odometry
        self.odom_cov = np.diag([0.01**2, 0.01**2, 0.01**2, 
                                  0.005**2, 0.005**2, 0.005**2])
        
        self.t = 0.0
        self.timer = self.create_timer(self.dt, self.step)
        
        self.get_logger().info(
            f"SimWorld started: trajectory={self.trajectory_type}, "
            f"duration={self.T_total}s, v={self.v}m/s"
        )
        self.get_logger().info(
            "Publishing ground truth to /sim/ground_truth and noisy odom to /odom. "
            "Sensors should come from a real simulator (e.g., Gazebo)."
        )

    def _publish_world_timer(self):
        stamp = self.get_clock().now().to_msg()
        self._publish_world_markers(stamp)

    def _publish_world_markers(self, stamp):
        """Publish world geometry as markers for visualization."""
        ma = MarkerArray()
        
        # Define a simple world with obstacles
        # These could eventually be used for ray-casting in a physics-based sim
        obstacles = [
            # (x, y, z, sx, sy, sz, r, g, b)
            (3.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.8, 0.2, 0.2),   # Red box
            (0.0, 3.0, 0.25, 0.5, 0.5, 0.5, 0.2, 0.8, 0.2),   # Green box
            (-3.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.2, 0.2, 0.8),  # Blue box
            (0.0, -3.0, 0.25, 0.5, 0.5, 0.5, 0.8, 0.8, 0.2),  # Yellow box
            (2.0, 2.0, 0.25, 0.3, 0.3, 0.5, 0.6, 0.4, 0.2),   # Brown pillar
            (-2.0, -2.0, 0.25, 0.3, 0.3, 0.5, 0.6, 0.4, 0.2), # Brown pillar
        ]
        
        for i, (x, y, z, sx, sy, sz, r, g, b) in enumerate(obstacles):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.odom_frame
            m.ns = "world_obstacles"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = float(z)
            m.pose.orientation.w = 1.0
            m.scale.x = float(sx)
            m.scale.y = float(sy)
            m.scale.z = float(sz)
            m.color.a = 0.85
            m.color.r = float(r)
            m.color.g = float(g)
            m.color.b = float(b)
            ma.markers.append(m)
        
        # Ground plane marker
        ground = Marker()
        ground.header.stamp = stamp
        ground.header.frame_id = self.odom_frame
        ground.ns = "world_ground"
        ground.id = 0
        ground.type = Marker.CUBE
        ground.action = Marker.ADD
        ground.pose.position.x = 0.0
        ground.pose.position.y = 0.0
        ground.pose.position.z = -0.025
        ground.pose.orientation.w = 1.0
        ground.scale.x = 10.0
        ground.scale.y = 10.0
        ground.scale.z = 0.05
        ground.color.a = 0.3
        ground.color.r = 0.5
        ground.color.g = 0.5
        ground.color.b = 0.5
        ma.markers.append(ground)
        
        self.pub_world_markers.publish(ma)

    def _compute_delta(self) -> np.ndarray:
        """Compute commanded delta based on trajectory type."""
        if self.trajectory_type == "straight":
            # Pure forward motion
            return np.array([self.v * self.dt, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        elif self.trajectory_type == "figure8":
            # Figure-8 trajectory: sinusoidal angular velocity
            omega = self.w * math.cos(2.0 * self.w * self.t)
            return np.array([self.v * self.dt, 0.0, 0.0, 0.0, 0.0, omega * self.dt])
        
        else:  # "circle" (default)
            # Circular trajectory
            return np.array([self.v * self.dt, 0.0, 0.0, 0.0, 0.0, self.w * self.dt])

    def step(self):
        """Simulation step: update ground truth and publish."""
        self.t += self.dt

        # Compute commanded delta motion
        delta = self._compute_delta()
        
        # Update ground truth (no noise)
        self.x_true = se3_compose(self.x_true, delta)
        
        # Update odometry with accumulated noise
        noisy_delta = delta + np.random.randn(6) * self.odom_sigma
        self.x_odom = se3_compose(self.x_odom, noisy_delta)
        
        # Accumulate covariance (simplified - proper would use adjoint transport)
        delta_cov = np.diag(self.odom_sigma ** 2)
        self.odom_cov = self.odom_cov + delta_cov

        stamp = self.get_clock().now().to_msg()
        
        # Publish ground truth
        self._publish_pose(self.pub_ground_truth, self.x_true, 
                          np.zeros((6, 6)), stamp, 
                          self.map_frame, self.base_frame)
        
        # Publish noisy odometry (absolute pose, for odom_bridge to convert to delta)
        self._publish_pose(self.pub_odom, self.x_odom, 
                          self.odom_cov, stamp,
                          self.odom_frame, self.base_frame)
        
        # Publish ground truth path for Foxglove visualization
        self._publish_gt_path(stamp)

        # Check for completion
        if self.t >= self.T_total:
            self._log_final_stats()
            self.get_logger().info("Simulation complete. Continuing to publish final pose.")
            self.timer.cancel()
            # Keep node alive for visualization but stop advancing

    def _publish_pose(self, publisher, pose: np.ndarray, cov: np.ndarray,
                      stamp, frame_id: str, child_frame_id: str):
        """Publish an Odometry message from SE(3) state."""
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.child_frame_id = child_frame_id

        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        msg.pose.pose.position.z = float(pose[2])
        
        R = rotvec_to_rotmat(pose[3:6])
        qx, qy, qz, qw = rotmat_to_quat(R)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        
        msg.pose.covariance = cov.reshape(-1).tolist()
        
        publisher.publish(msg)

    def _publish_gt_path(self, stamp):
        """Publish ground truth path for Foxglove visualization."""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = self.map_frame
        pose_stamped.pose.position.x = float(self.x_true[0])
        pose_stamped.pose.position.y = float(self.x_true[1])
        pose_stamped.pose.position.z = float(self.x_true[2])
        
        R = rotvec_to_rotmat(self.x_true[3:6])
        qx, qy, qz, qw = rotmat_to_quat(R)
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw
        
        self.gt_path_poses.append(pose_stamped)
        
        # Trim if too long
        if len(self.gt_path_poses) > self.max_path_length:
            self.gt_path_poses = self.gt_path_poses[-self.max_path_length:]
        
        # Publish path
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.map_frame
        path.poses = self.gt_path_poses
        self.pub_gt_path.publish(path)

    def _log_final_stats(self):
        """Log final trajectory statistics."""
        # Compute drift (difference between odom and ground truth)
        drift = self.x_odom - self.x_true
        drift_trans = np.linalg.norm(drift[:3])
        drift_rot = np.linalg.norm(drift[3:6])
        
        self.get_logger().info(
            f"Final stats: "
            f"drift_trans={drift_trans:.4f}m, drift_rot={drift_rot:.4f}rad, "
            f"total_time={self.t:.1f}s"
        )
        
        # Log final positions
        self.get_logger().info(
            f"Ground truth final: x={self.x_true[0]:.3f}, y={self.x_true[1]:.3f}, "
            f"yaw={self.x_true[5]:.3f}"
        )
        self.get_logger().info(
            f"Odometry final: x={self.x_odom[0]:.3f}, y={self.x_odom[1]:.3f}, "
            f"yaw={self.x_odom[5]:.3f}"
        )


def main():
    rclpy.init()
    node = SimWorld()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
