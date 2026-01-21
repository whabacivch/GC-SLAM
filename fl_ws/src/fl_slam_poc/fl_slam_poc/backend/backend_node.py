"""
Frobenius-Legendre SLAM Backend Node.

State representation: SE(3) in rotation vector form (x, y, z, rx, ry, rz)
Covariance is in se(3) tangent space, transported via adjoint.

Loop Factor Convention (EXPLICIT):
    Z = T_anchor^{-1} ∘ T_current
    Backend reconstruction: T_current = T_anchor ∘ Z

Two-Pose Factor Semantics (G1 Compliance):
    Loop factors update BOTH anchor and current pose beliefs via joint
    Gaussian update, then marginalize anchor.

Hybrid Dual-Layer Architecture:
    - Sparse Anchor Modules: Laser-based keyframes for pose estimation
    - Dense 3D Modules: RGB-D-based modules for dense mapping + appearance
    - Multi-modal fusion: Laser 2D + RGB-D 3D via information form addition

Following information geometry principles:
- Gaussian fusion in information form (exact, closed-form)
- vMF fusion for surface normals (exact via Bessel barycenter)
- Covariance transport via adjoint (exact)
- Linearization only for predict step (explicit ablation)
- Probabilistic timestamp model (no hard gates)

Observability:
    Publishes /cdwm/backend_status (JSON) with input data status.
    You will KNOW if the system is running dead-reckoning only (no loop factors).

Reference: Barfoot (2017), Miyamoto et al. (2024), Combe (2022-2025)
"""

import json
import math
import struct
import time
from collections import deque
from typing import Optional, Dict, List

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from fl_slam_poc.common.transforms.se3 import (
    quat_to_rotmat,
    rotmat_to_quat,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
    se3_compose,
    se3_inverse,
    se3_adjoint,
    se3_cov_compose,
)
from fl_slam_poc.backend.parameters import TimeAlignmentModel, AdaptiveProcessNoise
from fl_slam_poc.backend.fusion.gaussian_info import make_evidence, fuse_info, mean_cov
from fl_slam_poc.backend.fusion.gaussian_geom import gaussian_frobenius_correction
from fl_slam_poc.frontend.loops.vmf_geometry import vmf_barycenter, vmf_mean_param
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.msg import AnchorCreate, LoopFactor


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# =============================================================================
# Module Classes for Dual-Layer Atlas
# =============================================================================

class BaseModule:
    """Base class for all modules in the atlas."""
    def __init__(self, module_id: int, module_type: str):
        self.module_id = module_id
        self.module_type = module_type  # "sparse_anchor" or "dense_3d"
        self.mass = 1.0
        self.last_updated = time.time()


class SparseAnchorModule(BaseModule):
    """
    Sparse anchor from laser SLAM.
    
    Stores:
    - SE(3) pose (mu) and covariance in information form (L, h)
    - Point cloud for visualization
    - Optional: NIG descriptor model
    
    Can be upgraded to 3D when fused with RGB-D evidence.
    """
    def __init__(self, anchor_id: int, mu: np.ndarray, cov: np.ndarray, points: np.ndarray = None):
        super().__init__(anchor_id, "sparse_anchor")
        self.mu = mu.copy()
        self.cov = cov.copy()
        self.L, self.h = make_evidence(mu, cov)
        self.points = points.copy() if points is not None else np.empty((0, 3))
        self.desc_model = None  # NIG descriptor (set by frontend)
        self.rgbd_fused = False  # True if RGB-D evidence has been fused
    
    def fuse_rgbd_position(self, rgbd_L: np.ndarray, rgbd_h: np.ndarray, weight: float = 1.0):
        """
        Fuse RGB-D 3D position evidence at this anchor.
        
        Uses information form addition (exact, closed-form).
        """
        # Direct 3D fusion (anchor is already 6D SE(3), use position part)
        # Extract position-only information
        L_pos = self.L[:3, :3]
        h_pos = self.h[:3]
        
        # Fuse position evidence
        L_pos_fused = L_pos + weight * rgbd_L
        h_pos_fused = h_pos + weight * rgbd_h.reshape(-1)
        
        # Update anchor's position components
        self.L[:3, :3] = L_pos_fused
        self.h[:3] = h_pos_fused
        
        # Recover mean/cov
        self.mu, self.cov = mean_cov(self.L, self.h)
        self.mass += weight
        self.last_updated = time.time()
        self.rgbd_fused = True


class Dense3DModule(BaseModule):
    """
    Dense 3D Gaussian module from RGB-D.
    
    Stores:
    - 3D position + covariance in information form
    - vMF normal (surface normal as θ = κμ)
    - Color (RGB Gaussian)
    - Opacity (scalar)
    """
    def __init__(self, module_id: int, mu: np.ndarray, cov: np.ndarray):
        super().__init__(module_id, "dense_3d")
        self.mu = mu.copy()
        self.cov = cov.copy()
        self.L, self.h = make_evidence(mu, cov)
        
        # vMF normal (default: pointing up, κ=0 isotropic)
        self.normal_theta = np.array([0.0, 0.0, 1.0])
        
        # Color (RGB Gaussian)
        self.color_mean = np.array([0.5, 0.5, 0.5])
        self.color_cov = np.eye(3) * 0.01
        self.color_L, self.color_h = make_evidence(self.color_mean, self.color_cov)
        
        # Opacity
        self.alpha_mean = 1.0
        self.alpha_var = 0.1
    
    def update_from_evidence(self, evidence: dict, weight: float = 1.0):
        """
        Update module from RGB-D evidence dict.
        
        All operations use exact closed-form exponential family fusion.
        """
        # Position fusion (Gaussian info form)
        self.L, self.h = fuse_info(
            self.L, self.h,
            evidence["position_L"], evidence["position_h"],
            weight=weight
        )
        self.mu, self.cov = mean_cov(self.L, self.h)
        
        # Normal fusion (vMF barycenter - exact via Bessel)
        thetas = [self.normal_theta, evidence["normal_theta"]]
        weights_vmf = [self.mass, weight]
        self.normal_theta, _ = vmf_barycenter(thetas, weights_vmf, d=3)
        
        # Color fusion (Gaussian info form)
        self.color_L, self.color_h = fuse_info(
            self.color_L, self.color_h,
            evidence["color_L"], evidence["color_h"],
            weight=weight
        )
        self.color_mean, self.color_cov = mean_cov(self.color_L, self.color_h)
        
        # Opacity fusion (weighted average)
        obs_alpha = evidence.get("alpha_mean", 1.0)
        self.alpha_mean = (self.mass * self.alpha_mean + weight * obs_alpha) / (self.mass + weight)
        
        self.mass += weight
        self.last_updated = time.time()


class FLBackend(Node):
    def __init__(self):
        super().__init__("fl_backend")
        self._declare_parameters()
        self._init_from_params()

    def _declare_parameters(self):
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("rgbd_evidence_topic", "/sim/rgbd_evidence")
        self.declare_parameter("alignment_sigma_prior", 0.1)
        self.declare_parameter("alignment_prior_strength", 5.0)
        self.declare_parameter("alignment_sigma_floor", 0.001)
        self.declare_parameter("process_noise_trans_prior", 0.03)
        self.declare_parameter("process_noise_rot_prior", 0.015)
        self.declare_parameter("process_noise_prior_strength", 10.0)

    def _init_from_params(self):
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.rgbd_evidence_topic = str(self.get_parameter("rgbd_evidence_topic").value)

        # QoS profile for subscriptions (MUST match frontend RELIABLE QoS)
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,  # Increased to match odom bridge publisher depth
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscriptions (with RELIABLE QoS to match frontend)
        self.sub_odom = self.create_subscription(Odometry, "/sim/odom", self.on_odom, qos)
        self.sub_loop = self.create_subscription(LoopFactor, "/sim/loop_factor", self.on_loop, qos)
        self.sub_anchor = self.create_subscription(AnchorCreate, "/sim/anchor_create", self.on_anchor_create, qos)
        self.sub_rgbd = self.create_subscription(String, self.rgbd_evidence_topic, self.on_rgbd_evidence, qos)

        # Publishers
        self.pub_state = self.create_publisher(Odometry, "/cdwm/state", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/cdwm/markers", 10)
        self.pub_dbg = self.create_publisher(String, "/cdwm/debug", 10)
        self.pub_report = self.create_publisher(String, "/cdwm/op_report", 10)
        self.pub_loop_markers = self.create_publisher(MarkerArray, "/cdwm/loop_markers", 10)
        
        # Map publisher (point cloud)
        from sensor_msgs.msg import PointCloud2, PointField
        self.pub_map = self.create_publisher(PointCloud2, "/cdwm/map", 10)
        self.PointCloud2 = PointCloud2
        self.PointField = PointField
        
        # Trajectory path publisher for Foxglove visualization
        self.pub_path = self.create_publisher(Path, "/cdwm/trajectory", 10)
        self.trajectory_poses: list[PoseStamped] = []
        
        # Trajectory export for ground truth comparison
        self.trajectory_export_path = self.declare_parameter(
            "trajectory_export_path", "/tmp/fl_slam_trajectory.tum"
        ).value
        self.trajectory_file = None
        if self.trajectory_export_path:
            self.trajectory_file = open(self.trajectory_export_path, "w")
            self.trajectory_file.write("# timestamp x y z qx qy qz qw\n")
            self.get_logger().info(f"Exporting trajectory to: {self.trajectory_export_path}")
        self.max_path_length = 1000  # Limit path history

        # State belief in information form
        mu0 = np.zeros(6)
        cov0 = np.diag([0.2**2, 0.2**2, 0.2**2, 0.1**2, 0.1**2, 0.1**2])
        self.L, self.h = make_evidence(mu0, cov0)

        # Adaptive process noise
        trans_prior = float(self.get_parameter("process_noise_trans_prior").value)
        rot_prior = float(self.get_parameter("process_noise_rot_prior").value)
        noise_strength = float(self.get_parameter("process_noise_prior_strength").value)
        prior_diag = np.array([trans_prior**2] * 3 + [rot_prior**2] * 3)
        self.process_noise = AdaptiveProcessNoise.create(6, prior_diag, noise_strength)

        # Timestamp alignment
        align_sigma = float(self.get_parameter("alignment_sigma_prior").value)
        align_strength = float(self.get_parameter("alignment_prior_strength").value)
        align_floor = float(self.get_parameter("alignment_sigma_floor").value)
        self.timestamp_model = TimeAlignmentModel(align_sigma, align_strength, align_floor)

        # Anchor storage: anchor_id -> (mu, cov, L, h, points)
        # LEGACY: Kept for backward compatibility, new code should use sparse_anchors
        self.anchors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        
        # Dual-layer module atlas (NEW)
        self.sparse_anchors: Dict[int, SparseAnchorModule] = {}
        self.dense_modules: Dict[int, Dense3DModule] = {}
        self.next_dense_id = 1000000  # High IDs for dense modules (avoid collision with anchor IDs)
        
        # Dense module configuration
        self.dense_association_radius = 0.5  # meters - fuse RGB-D at anchors within this radius
        self.max_dense_modules = 10000  # Prevent unbounded memory growth
        
        # Loop factor buffer for race condition protection
        # Stores loop factors that arrive before their anchor is created
        self.pending_loop_factors: dict[int, list] = {}
        self.max_pending_loops_per_anchor = 100  # Prevent unbounded growth
        
        # State buffer for timestamp alignment
        self.state_buffer = deque(maxlen=500)
        self.prev_mu = None

        # TF broadcaster so Foxglove/TF consumers can place frames correctly
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Status publisher
        self.pub_status = self.create_publisher(String, "/cdwm/backend_status", 10)
        
        # Input tracking - YOU WILL KNOW WHAT'S HAPPENING
        self.odom_count = 0
        self.loop_factor_count = 0
        self.anchor_count = 0
        self.last_odom_time: Optional[float] = None
        self.last_loop_time: Optional[float] = None
        self.last_odom_stamp: Optional[float] = None  # Odometry message timestamp for trajectory export
        self.node_start_time = time.time()
        self.status_period = 5.0
        self.warned_no_loops = False
        
        # Status timer uses wall time so it continues even when /clock pauses.
        self._status_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.status_timer = self.create_timer(
            self.status_period, self._check_status, clock=self._status_clock
        )
        
        # Startup log
        self.get_logger().info("=" * 60)
        self.get_logger().info("FL-SLAM Backend starting")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Subscriptions:")
        self.get_logger().info("  Delta odom:    /sim/odom (MUST come from tb3_odom_bridge)")
        self.get_logger().info("  Loop factors:  /sim/loop_factor (from frontend)")
        self.get_logger().info("  Anchors:       /sim/anchor_create (from frontend)")
        self.get_logger().info("  RGB-D evidence: " + self.rgbd_evidence_topic)
        self.get_logger().info("")
        self.get_logger().info("Status monitoring: Will report DEAD_RECKONING if no loop factors")
        self.get_logger().info("  Check /cdwm/backend_status for real-time status")
        self.get_logger().info("=" * 60)

    def on_odom(self, msg: Odometry):
        """Process delta odometry with adjoint covariance transport."""
        self.odom_count += 1
        self.last_odom_time = time.time()
        # Store odometry message timestamp for trajectory export (NOT wall clock!)
        self.last_odom_stamp = stamp_to_sec(msg.header.stamp)
        
        # Log first few odom messages for debugging
        if self.odom_count <= 3:
            self.get_logger().info(
                f"Backend received odom #{self.odom_count}, "
                f"delta=({msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f}, {msg.pose.pose.position.z:.3f})"
            )
        
        # Extract delta
        dx = float(msg.pose.pose.position.x)
        dy = float(msg.pose.pose.position.y)
        dz = float(msg.pose.pose.position.z)
        
        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)
        R_delta = quat_to_rotmat(qx, qy, qz, qw)
        rotvec_delta = rotmat_to_rotvec(R_delta)
        
        delta = np.array([dx, dy, dz, rotvec_delta[0], rotvec_delta[1], rotvec_delta[2]], dtype=float)

        # Get current state
        mu, cov = mean_cov(self.L, self.h)
        linearization_point = mu.copy()
        
        # Compose pose
        mu_pred = se3_compose(mu, delta)
        
        # Get adaptive process noise and transport covariance
        Q = self.process_noise.estimate()
        cov_pred = se3_cov_compose(cov, Q, mu)

        # Update state
        self.L, self.h = make_evidence(mu_pred, cov_pred)
        self.state_buffer.append((stamp_to_sec(msg.header.stamp), mu_pred.copy(), cov_pred.copy()))
        
        # Update process noise from residuals
        residual_vec = np.zeros(6, dtype=float)
        if self.prev_mu is not None:
            predicted_from_prev = se3_compose(self.prev_mu, delta)
            residual_vec = (mu_pred - predicted_from_prev).astype(float)
            self.process_noise.update(residual_vec)
        self.prev_mu = mu.copy()
        
        self._publish_state(tag="odom")

        _, frob_stats = gaussian_frobenius_correction(residual_vec)
        
        self._publish_report(OpReport(
            name="GaussianPredictSE3",
            exact=False,
            approximation_triggers=["Linearization"],
            family_in="Gaussian",
            family_out="Gaussian",
            closed_form=True,
            frobenius_applied=True,
            frobenius_operator="gaussian_identity_third_order",
            frobenius_delta_norm=float(frob_stats["delta_norm"]),
            frobenius_input_stats=dict(frob_stats["input_stats"]),
            frobenius_output_stats=dict(frob_stats["output_stats"]),
            metrics={
                "covariance_transport": "adjoint",
                "linearization_point": linearization_point.tolist(),
                "process_noise_trace": float(np.trace(Q)),
                "process_noise_confidence": self.process_noise.confidence(),
            },
            notes="Delta-odom composed in SE(3) with adjoint covariance transport.",
        ))

    def _get_state_at_stamp(self, stamp_sec: float):
        if not self.state_buffer:
            mu, cov = mean_cov(self.L, self.h)
            return mu, cov, None
        closest = min(self.state_buffer, key=lambda item: abs(item[0] - stamp_sec))
        return closest[1], closest[2], float(stamp_sec - closest[0])

    def on_anchor_create(self, msg: AnchorCreate):
        """Create anchor with probabilistic timestamp weighting."""
        self.anchor_count += 1
        
        anchor_id = int(msg.anchor_id)
        stamp = stamp_to_sec(msg.header.stamp)
        mu, cov, dt = self._get_state_at_stamp(stamp)
        
        self.get_logger().info(f"Backend received anchor {anchor_id} with {len(msg.points)} points")
        
        self.timestamp_model.update(dt)
        timestamp_weight = self.timestamp_model.weight(dt)
        
        # Scale covariance by inverse weight
        if timestamp_weight > 1e-6:
            cov_scaled = cov / timestamp_weight
        else:
            cov_scaled = cov * 1e6
        
        # Convert points from message
        points = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=float) if len(msg.points) > 0 else np.empty((0, 3))
        
        L_anchor, h_anchor = make_evidence(mu, cov_scaled)
        self.anchors[anchor_id] = (mu.copy(), cov_scaled.copy(), L_anchor.copy(), h_anchor.copy(), points.copy())
        self._publish_anchor_marker(anchor_id, mu)
        
        # Publish updated map
        self._publish_map()

        _, frob_stats = gaussian_frobenius_correction(np.zeros(6, dtype=float))

        self._publish_report(OpReport(
            name="AnchorCreate",
            exact=dt is None or abs(dt) < 1e-9,
            approximation_triggers=["TimestampAlignment"] if dt is not None and abs(dt) >= 1e-9 else [],
            family_in="Gaussian",
            family_out="Gaussian",
            closed_form=True,
            frobenius_applied=bool(dt is not None and abs(dt) >= 1e-9),
            frobenius_operator="gaussian_identity_third_order" if (dt is not None and abs(dt) >= 1e-9) else None,
            frobenius_delta_norm=float(frob_stats["delta_norm"]) if (dt is not None and abs(dt) >= 1e-9) else None,
            frobenius_input_stats=dict(frob_stats["input_stats"]) if (dt is not None and abs(dt) >= 1e-9) else None,
            frobenius_output_stats=dict(frob_stats["output_stats"]) if (dt is not None and abs(dt) >= 1e-9) else None,
            metrics={
                "anchor_id": anchor_id,
                "dt_sec": dt,
                "timestamp_weight": timestamp_weight,
            },
            notes="Anchor with probabilistic timestamp weighting.",
        ))
        
        # Process any pending loop factors for this anchor (race condition protection)
        if anchor_id in self.pending_loop_factors:
            pending = self.pending_loop_factors.pop(anchor_id)
            self.get_logger().info(
                f"Processing {len(pending)} pending loop factors for anchor {anchor_id}"
            )
            for pending_msg in pending:
                self.on_loop(pending_msg)

    def on_rgbd_evidence(self, msg: String):
        """
        Receive RGB-D evidence (JSON payload) and update dense map layer.

        Payload schema:
          {"evidence": [ {position_L, position_h, color_L, color_h, normal_theta, alpha_mean, alpha_var}, ... ]}
        """
        try:
            payload = json.loads(msg.data)
            evidence_in = payload.get("evidence", [])
            if not isinstance(evidence_in, list) or len(evidence_in) == 0:
                return

            evidence_list: List[dict] = []
            for ev in evidence_in:
                L = np.asarray(ev["position_L"], dtype=float)
                h = np.asarray(ev["position_h"], dtype=float).reshape(-1)
                out = {"position_L": L, "position_h": h}

                if "color_L" in ev and "color_h" in ev:
                    out["color_L"] = np.asarray(ev["color_L"], dtype=float)
                    out["color_h"] = np.asarray(ev["color_h"], dtype=float).reshape(-1)
                if "normal_theta" in ev:
                    out["normal_theta"] = np.asarray(ev["normal_theta"], dtype=float).reshape(-1)
                if "alpha_mean" in ev:
                    out["alpha_mean"] = float(ev["alpha_mean"])
                if "alpha_var" in ev:
                    out["alpha_var"] = float(ev["alpha_var"])

                evidence_list.append(out)

            self.process_rgbd_evidence(evidence_list)
        except Exception as e:
            self.get_logger().warn(
                f"RGB-D evidence parse/update failed: {e}",
                throttle_duration_sec=5.0,
            )

    def on_loop(self, msg: LoopFactor):
        """Two-pose factor update with bidirectional information flow."""
        self.loop_factor_count += 1
        self.last_loop_time = time.time()
        
        anchor_id = int(msg.anchor_id)
        
        if not hasattr(self, '_loop_recv_count'):
            self._loop_recv_count = 0
        self._loop_recv_count += 1
        if self._loop_recv_count <= 3:  # Reduced from 5 to 3
            self.get_logger().info(f"Backend received loop factor #{self._loop_recv_count} for anchor {anchor_id}")
        
        anchor_data = self.anchors.get(anchor_id)

        # Debug: Check if anchor exists
        if self.loop_factor_count <= 5:
            self.get_logger().info(
                f"Loop factor #{self.loop_factor_count}: anchor {anchor_id} "
                f"{'FOUND' if anchor_data is not None else 'NOT FOUND'}, "
                f"total anchors: {len(self.anchors)}"
            )

        if anchor_data is None:
            # Race condition: loop factor arrived before anchor creation
            # Buffer it for later processing
            if anchor_id not in self.pending_loop_factors:
                self.pending_loop_factors[anchor_id] = []
            
            if len(self.pending_loop_factors[anchor_id]) < self.max_pending_loops_per_anchor:
                self.pending_loop_factors[anchor_id].append(msg)
                self.get_logger().debug(
                    f"Buffering loop factor for unknown anchor {anchor_id} "
                    f"({len(self.pending_loop_factors[anchor_id])} pending)"
                )
            else:
                self.get_logger().warn(
                    f"Dropping loop factor for anchor {anchor_id}: buffer full "
                    f"({self.max_pending_loops_per_anchor} pending)"
                )
            
            self._publish_report(OpReport(
                name="LoopFactorBuffered",
                exact=True,
                family_in="Gaussian",
                family_out="Gaussian",
                closed_form=True,
                domain_projection=False,
                metrics={"anchor_id": anchor_id, "buffered": True},
                notes="Loop factor arrived before anchor creation - buffered for processing.",
            ))
            return

        mu_anchor, cov_anchor, L_anchor, h_anchor, _ = anchor_data  # Unpack 5-tuple, ignore points
        
        # Extract relative pose
        rx = float(msg.rel_pose.position.x)
        ry = float(msg.rel_pose.position.y)
        rz = float(msg.rel_pose.position.z)
        qx = float(msg.rel_pose.orientation.x)
        qy = float(msg.rel_pose.orientation.y)
        qz = float(msg.rel_pose.orientation.z)
        qw = float(msg.rel_pose.orientation.w)
        R_rel = quat_to_rotmat(qx, qy, qz, qw)
        rotvec_rel = rotmat_to_rotvec(R_rel)
        rel = np.array([rx, ry, rz, rotvec_rel[0], rotvec_rel[1], rotvec_rel[2]], dtype=float)

        cov_rel = np.array(msg.covariance, dtype=float).reshape(6, 6)
        weight = max(float(msg.weight), 0.0)
        
        if weight < 1e-12:
            return

        # Two-pose factor update
        mu_current, cov_current = mean_cov(self.L, self.h)
        
        # Predicted relative transform
        T_anchor_inv = se3_inverse(mu_anchor)
        Z_pred = se3_compose(T_anchor_inv, mu_current)
        innovation = rel - Z_pred
        
        # Jacobians
        Ad_anchor_inv = se3_adjoint(T_anchor_inv)
        H_anchor = -Ad_anchor_inv
        H_current = np.eye(6)
        
        # Build joint system
        L_joint = np.zeros((12, 12), dtype=float)
        h_joint = np.zeros(12, dtype=float)
        
        L_joint[:6, :6] = L_anchor
        h_joint[:6] = h_anchor.reshape(-1)
        L_joint[6:12, 6:12] = self.L
        h_joint[6:12] = self.h.reshape(-1)
        
        H = np.zeros((6, 12), dtype=float)
        H[:, :6] = H_anchor
        H[:, 6:12] = H_current
        
        try:
            cov_rel_inv = np.linalg.inv(cov_rel)
        except np.linalg.LinAlgError:
            cov_rel_inv = np.linalg.pinv(cov_rel)
        
        cov_rel_inv_weighted = weight * cov_rel_inv
        L_joint += H.T @ cov_rel_inv_weighted @ H
        
        mu_joint_prior = np.concatenate([mu_anchor, mu_current])
        h_joint += H.T @ cov_rel_inv_weighted @ (innovation + H @ mu_joint_prior)
        
        # Marginalize anchor
        L_aa = L_joint[:6, :6]
        L_ac = L_joint[:6, 6:12]
        L_ca = L_joint[6:12, :6]
        L_cc = L_joint[6:12, 6:12]
        h_a = h_joint[:6]
        h_c = h_joint[6:12]
        
        try:
            L_aa_inv = np.linalg.inv(L_aa)
        except np.linalg.LinAlgError:
            L_aa_inv = np.linalg.pinv(L_aa)
        
        self.L = L_cc - L_ca @ L_aa_inv @ L_ac
        self.h = h_c - L_ca @ L_aa_inv @ h_a
        
        # Update anchor belief (preserve points)
        mu_anchor_new, cov_anchor_new = mean_cov(L_aa, h_a)
        anchor_points = anchor_data[4]  # Get existing points
        self.anchors[int(msg.anchor_id)] = (
            mu_anchor_new, cov_anchor_new, L_aa.copy(), h_a.reshape(-1).copy(), anchor_points
        )

        mu_updated, cov_updated = mean_cov(self.L, self.h)
        
        # Publish updated map after loop closure
        self._publish_map()

        _, frob_stats = gaussian_frobenius_correction(innovation)
        
        self._publish_report(OpReport(
            name="TwoPoseFactorUpdate",
            exact=False,
            approximation_triggers=["Linearization"],
            family_in="Gaussian",
            family_out="Gaussian",
            closed_form=True,
            frobenius_applied=True,
            frobenius_operator="gaussian_identity_third_order",
            frobenius_delta_norm=float(frob_stats["delta_norm"]),
            frobenius_input_stats=dict(frob_stats["input_stats"]),
            frobenius_output_stats=dict(frob_stats["output_stats"]),
            metrics={
                "anchor_id": int(msg.anchor_id),
                "weight": weight,
                "innovation_norm": float(np.linalg.norm(innovation)),
                "anchor_cov_trace_after": float(np.trace(cov_anchor_new)),
                "current_cov_trace_after": float(np.trace(cov_updated)),
            },
            notes="Two-pose factor with joint update and marginalization.",
        ))

        self._publish_loop_marker(int(msg.anchor_id), mu_anchor_new, mu_updated)
        self._publish_state(tag="loop")

        # Debug: Log state update
        if self.loop_factor_count <= 3:
            self.get_logger().info(
                f"Loop #{self.loop_factor_count} processed: innovation_norm={np.linalg.norm(innovation):.6f}, "
                f"weight={weight:.6f}, cov_trace_before={float(np.trace(cov_current)):.3f}, "
                f"cov_trace_after={float(np.trace(cov_updated)):.3f}"
            )

    def _check_status(self):
        """Periodic status check - warns if running dead-reckoning only."""
        elapsed = time.time() - self.node_start_time
        
        # Compute odom rate
        odom_rate = self.odom_count / max(elapsed, 1.0)
        
        # Check if we're getting loop factors
        receiving_loops = self.loop_factor_count > 0
        loops_recent = (self.last_loop_time is not None and 
                       (time.time() - self.last_loop_time) < 30.0)
        
        # Determine mode
        if not receiving_loops:
            mode = "DEAD_RECKONING"
        elif loops_recent:
            mode = "SLAM_ACTIVE"
        else:
            mode = "SLAM_STALE"
        
        # Count pending loop factors
        total_pending = sum(len(v) for v in self.pending_loop_factors.values())
        
        status = {
            "timestamp": time.time(),
            "elapsed_sec": elapsed,
            "mode": mode,
            "odom_count": self.odom_count,
            "odom_rate_hz": round(odom_rate, 1),
            "loop_factor_count": self.loop_factor_count,
            "anchor_count": self.anchor_count,
            "anchors_stored": len(self.anchors),
            "pending_loop_factors": total_pending,
            "last_loop_age_sec": (time.time() - self.last_loop_time) if self.last_loop_time else None,
            # Dual-layer statistics (NEW)
            "sparse_anchors": len(self.sparse_anchors),
            "dense_modules": len(self.dense_modules),
            "rgbd_fused_anchors": sum(1 for a in self.sparse_anchors.values() if a.rgbd_fused),
        }
        
        # Warn if no loop factors after startup period
        if elapsed > 15.0 and not receiving_loops and not self.warned_no_loops:
            self.warned_no_loops = True
            self.get_logger().warn(
                "=" * 60 + "\n"
                "BACKEND RUNNING DEAD-RECKONING ONLY\n"
                "No loop factors received from frontend.\n"
                "This means: NO SLAM, just accumulating odometry drift.\n"
                "Check: Are sensors connected? Is frontend running?\n"
                f"Stats: odom={self.odom_count}, anchors={self.anchor_count}, loops={self.loop_factor_count}\n"
                "=" * 60
            )
        
        # Periodic status log
        if elapsed > 10.0 and int(elapsed) % 30 == 0:
            self.get_logger().info(
                f"Backend status: mode={mode}, odom={self.odom_count} ({odom_rate:.1f}Hz), "
                f"loops={self.loop_factor_count}, anchors={len(self.anchors)}"
            )
        
        # Publish status
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)

    def _publish_state(self, tag: str):
        mu, cov = mean_cov(self.L, self.h)

        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = "base_link"

        out.pose.pose.position.x = float(mu[0])
        out.pose.pose.position.y = float(mu[1])
        out.pose.pose.position.z = float(mu[2])
        
        R = rotvec_to_rotmat(mu[3:6])
        qx, qy, qz, qw = rotmat_to_quat(R)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.pose.covariance = cov.reshape(-1).tolist()
        self.pub_state.publish(out)

        # Publish TF: odom -> base_link using the same pose we publish in /cdwm/state
        tf_msg = TransformStamped()
        tf_msg.header = out.header
        tf_msg.child_frame_id = out.child_frame_id
        tf_msg.transform.translation.x = float(mu[0])
        tf_msg.transform.translation.y = float(mu[1])
        tf_msg.transform.translation.z = float(mu[2])
        tf_msg.transform.rotation.x = float(out.pose.pose.orientation.x)
        tf_msg.transform.rotation.y = float(out.pose.pose.orientation.y)
        tf_msg.transform.rotation.z = float(out.pose.pose.orientation.z)
        tf_msg.transform.rotation.w = float(out.pose.pose.orientation.w)
        self.tf_broadcaster.sendTransform(tf_msg)

        # Covariance ellipse marker
        ma = MarkerArray()
        m = Marker()
        m.header = out.header
        m.ns = "pose"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(mu[0])
        m.pose.position.y = float(mu[1])
        m.pose.position.z = float(mu[2])
        m.scale.x = float(2.0 * math.sqrt(max(cov[0, 0], 1e-9)))
        m.scale.y = float(2.0 * math.sqrt(max(cov[1, 1], 1e-9)))
        m.scale.z = float(2.0 * math.sqrt(max(cov[2, 2], 1e-9)))
        m.color.a = 0.8
        m.color.r = 0.2
        m.color.g = 0.8
        m.color.b = 0.2
        ma.markers.append(m)
        self.pub_markers.publish(ma)
        
        # Trajectory path for Foxglove visualization
        pose_stamped = PoseStamped()
        pose_stamped.header = out.header
        pose_stamped.pose = out.pose.pose
        self.trajectory_poses.append(pose_stamped)
        
        # Trim trajectory if too long
        if len(self.trajectory_poses) > self.max_path_length:
            self.trajectory_poses = self.trajectory_poses[-self.max_path_length:]
        
        # Publish path
        path = Path()
        path.header = out.header
        path.poses = self.trajectory_poses
        self.pub_path.publish(path)
        
        # Export trajectory to file with ODOMETRY timestamp (not wall clock!)
        # Using odometry msg timestamp ensures proper alignment with ground truth
        if self.trajectory_file and self.last_odom_stamp is not None:
            timestamp = self.last_odom_stamp  # Use odometry message timestamp
            self.trajectory_file.write(
                f"{timestamp:.6f} {mu[0]:.6f} {mu[1]:.6f} {mu[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
            self.trajectory_file.flush()

    def _publish_report(self, report: OpReport):
        try:
            report.validate()
        except ValueError as exc:
            self.get_logger().error(f"OpReport validation failed: {exc}")
            raise
        msg = String()
        msg.data = report.to_json()
        self.pub_report.publish(msg)

    def _publish_anchor_marker(self, anchor_id: int, mu: np.ndarray):
        ma = MarkerArray()
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.odom_frame
        m.ns = "anchors"
        m.id = anchor_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(mu[0])
        m.pose.position.y = float(mu[1])
        m.pose.position.z = float(mu[2])
        m.scale.x = 0.08
        m.scale.y = 0.08
        m.scale.z = 0.08
        m.color.a = 0.9
        m.color.r = 0.9
        m.color.g = 0.7
        m.color.b = 0.1
        ma.markers.append(m)
        self.pub_loop_markers.publish(ma)

    def _publish_loop_marker(self, anchor_id: int, mu_anchor: np.ndarray, mu_current: np.ndarray):
        ma = MarkerArray()
        line = Marker()
        line.header.stamp = self.get_clock().now().to_msg()
        line.header.frame_id = self.odom_frame
        line.ns = "loops"
        line.id = anchor_id
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.03
        line.color.a = 0.8
        line.color.r = 0.2
        line.color.g = 0.6
        line.color.b = 0.9
        line.points = []

        start = Point()
        start.x = float(mu_anchor[0])
        start.y = float(mu_anchor[1])
        start.z = float(mu_anchor[2])
        end = Point()
        end.x = float(mu_current[0])
        end.y = float(mu_current[1])
        end.z = float(mu_current[2])
        line.points.append(start)
        line.points.append(end)
        ma.markers.append(line)
        self.pub_loop_markers.publish(ma)
    
    def _publish_map(self):
        """
        Publish dual-layer point cloud map.
        
        Two layers:
        - Sparse anchors (yellow) - laser keyframes
        - Dense modules (true color) - RGB-D modules
        """
        # Collect all points with colors
        points_with_color = []  # List of (x, y, z, r, g, b)
        
        # Layer 1: Sparse anchor point clouds (yellow)
        for anchor_id, (mu_anchor, cov_anchor, L, h, points) in self.anchors.items():
            if len(points) == 0:
                continue
            
            # Transform points from anchor frame to global frame
            R = rotvec_to_rotmat(mu_anchor[3:6])
            t = mu_anchor[:3]
            points_transformed = (R @ points.T).T + t
            
            # Yellow color for sparse anchors
            for pt in points_transformed:
                points_with_color.append((
                    float(pt[0]), float(pt[1]), float(pt[2]),
                    255, 255, 0  # Yellow
                ))
        
        # Layer 2: Dense modules (true RGB color)
        for mod in self.dense_modules.values():
            # Get module color (clamped to [0, 255])
            rgb = np.clip(mod.color_mean * 255, 0, 255).astype(np.uint8)
            points_with_color.append((
                float(mod.mu[0]), float(mod.mu[1]), float(mod.mu[2]),
                int(rgb[0]), int(rgb[1]), int(rgb[2])
            ))
        
        if len(points_with_color) == 0:
            self.get_logger().debug("No points to publish in map")
            return
        
        self.get_logger().info(f"Publishing map with {len(points_with_color)} points to /cdwm/map")
        
        # Create PointCloud2 message with XYZRGB
        msg = self.PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.height = 1
        msg.width = len(points_with_color)
        msg.is_dense = True
        msg.is_bigendian = False
        
        # Define fields (XYZRGB) - Foxglove compatible format
        # Pack RGB as a single float32 for maximum compatibility
        msg.fields = [
            self.PointField(name='x', offset=0, datatype=self.PointField.FLOAT32, count=1),
            self.PointField(name='y', offset=4, datatype=self.PointField.FLOAT32, count=1),
            self.PointField(name='z', offset=8, datatype=self.PointField.FLOAT32, count=1),
            self.PointField(name='rgb', offset=12, datatype=self.PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16  # 4 floats = 16 bytes
        msg.row_step = msg.point_step * msg.width
        
        # Pack points to bytes
        cloud_data = bytearray()
        for pt in points_with_color:
            # Pack RGB into a single float32 - Foxglove/RViz use BGR order (little-endian)
            r, g, b = int(pt[3]), int(pt[4]), int(pt[5])
            # BGR order: blue in LSB, then green, then red in MSB
            rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
            cloud_data.extend(struct.pack('<ffff', pt[0], pt[1], pt[2], rgb_packed))
        
        msg.data = bytes(cloud_data)
        self.pub_map.publish(msg)
    
    def add_dense_module(self, evidence: dict) -> int:
        """
        Add a new dense module from RGB-D evidence.
        
        Returns module ID.
        """
        if len(self.dense_modules) >= self.max_dense_modules:
            # Cull oldest modules
            self._cull_dense_modules()
        
        mu, cov = mean_cov(evidence["position_L"], evidence["position_h"])
        mod = Dense3DModule(self.next_dense_id, mu, cov)
        mod.update_from_evidence(evidence, weight=1.0)
        
        self.dense_modules[self.next_dense_id] = mod
        self.next_dense_id += 1
        
        return mod.module_id
    
    def _cull_dense_modules(self, keep_fraction: float = 0.8):
        """Remove oldest dense modules to free memory."""
        if len(self.dense_modules) == 0:
            return
        
        # Sort by last_updated
        sorted_mods = sorted(
            self.dense_modules.items(),
            key=lambda x: x[1].last_updated
        )
        
        # Keep most recent fraction
        keep_count = int(len(sorted_mods) * keep_fraction)
        remove_ids = [mod_id for mod_id, _ in sorted_mods[:-keep_count]]
        
        for mod_id in remove_ids:
            del self.dense_modules[mod_id]
        
        self.get_logger().info(f"Culled {len(remove_ids)} dense modules, {len(self.dense_modules)} remaining")
    
    def process_rgbd_evidence(self, evidence_list: List[dict]):
        """
        Process RGB-D evidence from frontend.
        
        Strategy (ORDER-INVARIANT):
        1. Assign each evidence to nearest anchor (deterministic tiebreak on anchor_id)
        2. Accumulate evidence per anchor BEFORE updating
        3. Apply all updates (order doesn't matter due to information addition)
        
        This ensures: Evidence_A + Evidence_B + Anchor_1 + Anchor_2 
                   == Evidence_B + Evidence_A + Anchor_2 + Anchor_1
        """
        # Phase 1: Assign evidence to anchors (deterministic assignment)
        evidence_for_anchor = {}  # anchor_id -> list of evidence
        unassigned_evidence = []
        
        for evidence in evidence_list:
            mu_obs, _ = mean_cov(evidence["position_L"], evidence["position_h"])
            
            # Find nearest sparse anchor with DETERMINISTIC tiebreaker
            candidates = []
            for anchor_id, (mu_anchor, _, _, _, _) in self.anchors.items():
                dist = np.linalg.norm(mu_anchor[:2] - mu_obs[:2])  # 2D XY distance
                if dist < self.dense_association_radius:
                    candidates.append((dist, anchor_id))
            
            if len(candidates) > 0:
                # Sort by distance, then by anchor_id (deterministic tiebreak)
                candidates.sort(key=lambda x: (x[0], x[1]))
                nearest_anchor_id = candidates[0][1]
                
                if nearest_anchor_id not in evidence_for_anchor:
                    evidence_for_anchor[nearest_anchor_id] = []
                evidence_for_anchor[nearest_anchor_id].append(evidence)
            else:
                unassigned_evidence.append(evidence)
        
        # Phase 2: Apply accumulated evidence to anchors (order-invariant)
        for anchor_id, evidence_batch in evidence_for_anchor.items():
            anchor_data = self.anchors[anchor_id]
            mu_a, cov_a, L_a, h_a, points_a = anchor_data
            
            # Create sparse anchor module if not exists
            if anchor_id not in self.sparse_anchors:
                self.sparse_anchors[anchor_id] = SparseAnchorModule(
                    anchor_id, mu_a, cov_a, points_a
                )
            
            # Fuse all evidence at once (information addition is associative/commutative)
            for evidence in evidence_batch:
                self.sparse_anchors[anchor_id].fuse_rgbd_position(
                    evidence["position_L"], evidence["position_h"], weight=1.0
                )
        
        # Phase 3: Create new dense modules for unassigned evidence
        for evidence in unassigned_evidence:
            self.add_dense_module(evidence)
    
    def destroy_node(self):
        """Clean up trajectory file on shutdown."""
        if self.trajectory_file:
            self.trajectory_file.close()
            self.get_logger().info("Trajectory export closed")
        super().destroy_node()


def main():
    rclpy.init()
    node = FLBackend()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
