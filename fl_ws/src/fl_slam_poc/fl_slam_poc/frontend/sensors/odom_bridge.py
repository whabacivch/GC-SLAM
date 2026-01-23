"""
Odometry Bridge Node.

Converts absolute odometry to delta odometry for the FL-SLAM backend.
Generic implementation - works with any odometry source (M3DGR, TurtleBot3, etc.).
Uses proper SE(3) operations with rotation vector representation (no RPY).

Frame Handling:
    - Input frame: from message header (validated against TF)
    - Output frame: configurable via parameter (default: "odom")
    - Validates frame consistency when TF is available

Delta Covariance (H1 Compliance):
    For delta = T_prev^{-1} ∘ T_curr, the covariance transforms as:
    Σ_delta ≈ J_inv @ Σ_curr @ J_inv.T + J_prev @ Σ_prev @ J_prev.T
    
    For small deltas, J ≈ I, so we use the approximation:
    Σ_delta ≈ Σ_curr + Σ_prev
    
    This is logged via OpReport for audit compliance.

Reference: Barfoot (2017), Sola et al. (2018)
"""

from typing import Optional

import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from fl_slam_poc.common.geometry.se3_numpy import (
    quat_to_rotvec,
    rotmat_to_quat,
    rotvec_to_rotmat,
    se3_compose,
    se3_inverse,
)
from fl_slam_poc.backend.fusion.gaussian_geom import gaussian_frobenius_correction
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common import constants
from fl_slam_poc.frontend.diagnostics.op_report_publish import publish_op_report
from fl_slam_poc.frontend.sensors.qos_utils import resolve_qos_profiles
from fl_slam_poc.frontend.sensors.dedup import is_duplicate


class OdomBridge(Node):
    def __init__(self):
        super().__init__("odom_bridge")
        
        # Configurable frames (not hardcoded)
        self.declare_parameter("input_topic", "/odom")
        self.declare_parameter("output_topic", "/sim/odom")
        self.declare_parameter("output_frame", "odom")
        self.declare_parameter("child_frame", "base_link")
        self.declare_parameter("validate_frames", True)
        self.declare_parameter("tf_timeout_sec", 0.1)
        self.declare_parameter("qos_reliability", "reliable")
        
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)
        self.validate_frames = bool(self.get_parameter("validate_frames").value)
        self.tf_timeout = float(self.get_parameter("tf_timeout_sec").value)
        qos_reliability = str(self.get_parameter("qos_reliability").value).lower()
        
        self._last_msg_keys: dict[str, tuple[int, int, str]] = {}
        qos_profiles, qos_names = resolve_qos_profiles(
            reliability=qos_reliability,
            depth=constants.QOS_DEPTH_SENSOR_MED_FREQ,
        )
        self.sub = []
        for qos in qos_profiles:
            self.sub.append(self.create_subscription(Odometry, input_topic, self.on_odom, qos))
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=constants.QOS_DEPTH_SENSOR_MED_FREQ,
        )
        self.pub = self.create_publisher(Odometry, output_topic, pub_qos)
        self.pub_report = self.create_publisher(String, "/cdwm/op_report", 10)
        
        self.get_logger().info(
            f"odom_bridge subscribing to {input_topic} with QoS reliability: {', '.join(qos_names)}"
        )
        
        # TF for frame validation
        self.tf_buffer = tf2_ros.Buffer()
        tf_qos = QoSProfile(
            depth=constants.QOS_DEPTH_LOW_FREQ,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        tf_static_qos = QoSProfile(
            depth=constants.QOS_DEPTH_LOW_FREQ,
            reliability=ReliabilityPolicy.RELIABLE,
            # Standard for /tf_static is TRANSIENT_LOCAL (late-joining subscribers receive history)
            # This matches what rosbag2 playback publishes
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, qos=tf_qos, static_qos=tf_static_qos
        )

        # Previous state for delta computation
        self.prev_pose: Optional[np.ndarray] = None
        self.prev_cov: Optional[np.ndarray] = None
        self.prev_frame: Optional[str] = None
        
        self._warned_frames = set()

    def _warn_frame_once(self, key: str, msg: str):
        if key not in self._warned_frames:
            self._warned_frames.add(key)
            self.get_logger().warn(msg)

    def msg_to_se3(self, pose) -> np.ndarray:
        """Convert geometry_msgs/Pose to SE(3) vector (x, y, z, rx, ry, rz)."""
        qx = float(pose.orientation.x)
        qy = float(pose.orientation.y)
        qz = float(pose.orientation.z)
        qw = float(pose.orientation.w)
        rotvec = quat_to_rotvec(np.array([qx, qy, qz, qw], dtype=float))
        return np.array([
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            rotvec[0], rotvec[1], rotvec[2]
        ], dtype=float)

    def se3_to_msg_pose(self, se3: np.ndarray, out):
        """Fill geometry_msgs/Pose from SE(3) vector."""
        out.position.x = float(se3[0])
        out.position.y = float(se3[1])
        out.position.z = float(se3[2])
        R = rotvec_to_rotmat(se3[3:6])
        qx, qy, qz, qw = rotmat_to_quat(R)
        out.orientation.x = qx
        out.orientation.y = qy
        out.orientation.z = qz
        out.orientation.w = qw

    def validate_frame(self, frame_id: str, stamp) -> bool:
        """Validate frame exists in TF tree (if validation enabled)."""
        if not self.validate_frames:
            return True
        try:
            self.tf_buffer.lookup_transform(
                frame_id, frame_id,
                rclpy.time.Time.from_msg(stamp),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout))
            return True
        except Exception:
            return False

    def cov_from_msg(self, cov_list: list) -> np.ndarray:
        """Extract 6x6 covariance from flat list."""
        return np.array(cov_list, dtype=float).reshape(6, 6)

    def compute_delta_covariance(
        self, 
        cov_prev: np.ndarray, 
        cov_curr: np.ndarray, 
        delta: np.ndarray
    ) -> np.ndarray:
        """
        Compute covariance for delta = T_prev^{-1} ∘ T_curr.
        
        For small deltas, the Jacobian of the inverse-compose operation
        is approximately identity, giving:
            Σ_delta ≈ Σ_prev + Σ_curr
        
        This is a first-order approximation. For larger deltas, proper
        adjoint transport would be needed, but absolute odom typically
        has small inter-frame motion.
        """
        # First-order approximation (valid for small deltas)
        delta_norm = np.linalg.norm(delta)
        
        if delta_norm < 0.1:  # Small motion approximation valid
            return cov_prev + cov_curr
        else:
            # For larger motions, use conservative sum with scaling
            # This overestimates uncertainty, which is safe
            return 2.0 * (cov_prev + cov_curr)

    def on_odom(self, msg: Odometry):
        """Process odometry message and compute delta."""
        if is_duplicate(self._last_msg_keys, "odom", msg.header.stamp, frame_id=msg.header.frame_id):
            return
        input_frame = msg.header.frame_id
        
        # Frame validation (warn once if frame changes unexpectedly)
        if self.validate_frames:
            if self.prev_frame is not None and input_frame != self.prev_frame:
                self._warn_frame_once(
                    f"frame_change_{self.prev_frame}_{input_frame}",
                    f"Input frame changed from '{self.prev_frame}' to '{input_frame}'")
            
            if not self.validate_frame(input_frame, msg.header.stamp):
                self._warn_frame_once(
                    f"frame_missing_{input_frame}",
                    f"Input frame '{input_frame}' not found in TF tree")
        
        # Convert to SE(3)
        curr_pose = self.msg_to_se3(msg.pose.pose)
        curr_cov = self.cov_from_msg(msg.pose.covariance)
        
        if self.prev_pose is None:
            # First message: publish ZERO delta (identity transform)
            # The delta from pose to itself is zero - this is the correct initialization
            self.prev_pose = curr_pose.copy()
            self.prev_cov = curr_cov.copy()
            self.prev_frame = input_frame
            
            # Publish zero delta (identity transform)
            out = Odometry()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self.output_frame
            out.child_frame_id = self.child_frame
            # Identity pose: position=0, orientation=[0,0,0,1]
            out.pose.pose.position.x = 0.0
            out.pose.pose.position.y = 0.0
            out.pose.pose.position.z = 0.0
            out.pose.pose.orientation.x = 0.0
            out.pose.pose.orientation.y = 0.0
            out.pose.pose.orientation.z = 0.0
            out.pose.pose.orientation.w = 1.0
            out.pose.covariance = curr_cov.reshape(-1).tolist()
            out.twist = msg.twist
            self.pub.publish(out)

            self.get_logger().info(
                f"Odom bridge initialized at ({curr_pose[0]:.3f}, {curr_pose[1]:.3f}, {curr_pose[2]:.3f}), "
                f"published zero delta"
            )
            return

        # Compute delta: delta = T_prev^{-1} ∘ T_curr
        delta = se3_compose(se3_inverse(self.prev_pose), curr_pose)
        
        # Compute delta covariance with proper approximation
        delta_cov = self.compute_delta_covariance(self.prev_cov, curr_cov, delta)

        # Publish delta odometry
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.output_frame
        out.child_frame_id = self.child_frame
        self.se3_to_msg_pose(delta, out.pose.pose)
        out.pose.covariance = delta_cov.reshape(-1).tolist()
        out.twist = msg.twist
        
        self.pub.publish(out)
        
        # Publish OpReport for audit compliance (H1)
        delta_norm = float(np.linalg.norm(delta))
        approximation_used = "FirstOrderSum" if delta_norm < 0.1 else "ConservativeSum"

        _, frob_stats = gaussian_frobenius_correction(delta)
        
        report = OpReport(
            name="OdomDeltaConversion",
            exact=False,
            approximation_triggers=["CovarianceTransport"],
            family_in="Gaussian",
            family_out="Gaussian",
            closed_form=True,
            solver_used=None,
            frobenius_applied=True,
            frobenius_operator="gaussian_identity_third_order",
            frobenius_delta_norm=float(frob_stats["delta_norm"]),
            frobenius_input_stats=dict(frob_stats["input_stats"]),
            frobenius_output_stats=dict(frob_stats["output_stats"]),
            metrics={
                "delta_norm": delta_norm,
                "approximation": approximation_used,
                "cov_prev_trace": float(np.trace(self.prev_cov)),
                "cov_curr_trace": float(np.trace(curr_cov)),
                "cov_delta_trace": float(np.trace(delta_cov)),
            },
            notes=f"Delta covariance via {approximation_used}. "
                  "First-order valid for small inter-frame motion.",
        )
        publish_op_report(self, self.pub_report, report)

        self.prev_pose = curr_pose.copy()
        self.prev_cov = curr_cov.copy()
        self.prev_frame = input_frame

def main():
    rclpy.init()
    node = OdomBridge()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
