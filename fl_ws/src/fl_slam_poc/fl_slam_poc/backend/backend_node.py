"""
Frobenius-Legendre SLAM Backend Node.

State representation: SE(3) in rotation vector form (x, y, z, rx, ry, rz)
Covariance is in se(3) tangent space, transported via adjoint.

Loop Factor Convention (EXPLICIT):
    Z = T_anchor^{-1} ∘ T_current
    Backend reconstruction: T_current = T_anchor ∘ Z

Loop Closure Semantics (G1 Compliance):
    Loop factors update BOTH anchor and current pose beliefs via one-shot
    recomposition (bidirectional message passing + Gaussian fusion).
    No Schur complement, no per-anchor loop, and no Jacobians in core fusion.

Hybrid Dual-Layer Architecture:
    - Sparse Anchor Modules: Laser-based keyframes for pose estimation
    - Dense 3D Modules: RGB-D-based modules for dense mapping + appearance
    - Multi-modal fusion: Laser 2D + RGB-D 3D via information form addition

Following information geometry principles:
- Gaussian fusion in information form (exact, closed-form)
- vMF fusion for surface normals (exact via Bessel barycenter)
- Covariance transport via adjoint (exact)
- Linearization only where explicitly declared (e.g., predict)
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

# JAX is initialized via common.jax_init module (imported lazily when needed)
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from fl_slam_poc.common.se3 import (
    quat_to_rotvec,
    rotmat_to_quat,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
    se3_compose,
    se3_exp,
    se3_inverse,
    se3_adjoint,
    se3_cov_compose,
    se3_relative,
)
from fl_slam_poc.backend import TimeAlignmentModel, AdaptiveProcessNoise, AdaptiveIMUNoiseModel, WishartPrior
from fl_slam_poc.backend.gaussian_info import make_evidence, fuse_info, mean_cov
from fl_slam_poc.backend.gaussian_geom import (
    gaussian_frobenius_correction, 
    se3_tangent_frobenius_correction,
    imu_tangent_frobenius_correction
)
from fl_slam_poc.backend.information_distances import hellinger_gaussian
from fl_slam_poc.frontend.vmf_geometry import vmf_barycenter, vmf_mean_param
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common import constants
from fl_slam_poc.msg import AnchorCreate, IMUSegment, LoopFactor


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
        
        # IMU Integration (15D state extension)
        self.declare_parameter("enable_imu_fusion", True)
        self.declare_parameter("imu_gyro_noise_density", constants.IMU_GYRO_NOISE_DENSITY_DEFAULT)
        self.declare_parameter("imu_accel_noise_density", constants.IMU_ACCEL_NOISE_DENSITY_DEFAULT)
        self.declare_parameter("imu_bias_adapt_forgetting", 0.995)
        self.declare_parameter("gravity", list(constants.GRAVITY_DEFAULT))

    def _init_from_params(self):
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.rgbd_evidence_topic = str(self.get_parameter("rgbd_evidence_topic").value)

        # IMU fusion configuration (must be read before subscriptions)
        enable_imu_param = self.get_parameter("enable_imu_fusion")
        gyro_noise_param = self.get_parameter("imu_gyro_noise_density")
        accel_noise_param = self.get_parameter("imu_accel_noise_density")
        bias_forgetting_param = self.get_parameter("imu_bias_adapt_forgetting")

        self.enable_imu_fusion = bool(enable_imu_param.value if enable_imu_param.value is not None else True)
        self.imu_gyro_noise_density = float(
            gyro_noise_param.value if gyro_noise_param.value is not None else constants.IMU_GYRO_NOISE_DENSITY_DEFAULT
        )
        self.imu_accel_noise_density = float(
            accel_noise_param.value if accel_noise_param.value is not None else constants.IMU_ACCEL_NOISE_DENSITY_DEFAULT
        )
        self.imu_bias_adapt_forgetting = float(
            bias_forgetting_param.value if bias_forgetting_param.value is not None else 0.995
        )
        gravity_param = self.get_parameter("gravity").value
        self.gravity = np.array(gravity_param, dtype=float)

        self.imu_adaptive_noise = None
        # Fixed prior for bias innovation covariance (adaptive model learns online).
        # This is intentionally NOT exposed as an IMU "random-walk" parameter.
        self._imu_bias_innov_gyro_cov_prior = np.eye(3, dtype=float) * (
            float(constants.IMU_GYRO_BIAS_INNOV_STD_PRIOR) ** 2
        )
        self._imu_bias_innov_accel_cov_prior = np.eye(3, dtype=float) * (
            float(constants.IMU_ACCEL_BIAS_INNOV_STD_PRIOR) ** 2
        )
        if self.enable_imu_fusion:
            gyro_prior = WishartPrior.from_mean_covariance(self._imu_bias_innov_gyro_cov_prior)
            accel_prior = WishartPrior.from_mean_covariance(self._imu_bias_innov_accel_cov_prior)
            self.imu_adaptive_noise = AdaptiveIMUNoiseModel(
                accel_bias_prior=accel_prior,
                gyro_bias_prior=gyro_prior,
                forgetting_factor=self.imu_bias_adapt_forgetting,
            )

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
        
        # IMU segment subscription (Phase 2: 15D state extension)
        if self.enable_imu_fusion:
            self.sub_imu_segment = self.create_subscription(
                IMUSegment, "/sim/imu_segment", self.on_imu_segment, qos
            )

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
        
        # State dimension: 15D when IMU enabled, 6D otherwise
        self.state_dim = constants.STATE_DIM_FULL if self.enable_imu_fusion else constants.STATE_DIM_POSE
        
        # State belief in information form
        # 15D state: [p(3), R(3), v(3), b_g(3), b_a(3)]
        if self.enable_imu_fusion:
            mu0 = np.zeros(constants.STATE_DIM_FULL)
            cov0_diag = np.concatenate([
                np.array([constants.STATE_PRIOR_POSE_TRANS_STD**2] * 3),   # Position [0:3]
                np.array([constants.STATE_PRIOR_POSE_ROT_STD**2] * 3),     # Rotation [3:6]
                np.array([constants.STATE_PRIOR_VELOCITY_STD**2] * 3),     # Velocity [6:9]
                np.array([constants.STATE_PRIOR_GYRO_BIAS_STD**2] * 3),    # Gyro bias [9:12]
                np.array([constants.STATE_PRIOR_ACCEL_BIAS_STD**2] * 3),   # Accel bias [12:15]
            ])
            cov0 = np.diag(cov0_diag)
            self.get_logger().info(f"Backend: 15D state enabled (pose + velocity + biases)")
        else:
            mu0 = np.zeros(6)
            cov0 = np.diag([0.2**2, 0.2**2, 0.2**2, 0.1**2, 0.1**2, 0.1**2])
            self.get_logger().info(f"Backend: 6D state (pose only, IMU fusion disabled)")
        self.L, self.h = make_evidence(mu0, cov0)

        # Adaptive process noise (matches state dimension)
        trans_prior = float(self.get_parameter("process_noise_trans_prior").value)
        rot_prior = float(self.get_parameter("process_noise_rot_prior").value)
        noise_strength = float(self.get_parameter("process_noise_prior_strength").value)
        
        if self.enable_imu_fusion:
            # 15D process noise: pose + velocity + bias random walks
            prior_diag = np.concatenate([
                np.array([trans_prior**2] * 3),                            # Position
                np.array([rot_prior**2] * 3),                              # Rotation
                np.array([constants.PROCESS_NOISE_VELOCITY_STD**2] * 3),   # Velocity
                np.array([constants.PROCESS_NOISE_GYRO_BIAS_STD**2] * 3),  # Gyro bias
                np.array([constants.PROCESS_NOISE_ACCEL_BIAS_STD**2] * 3), # Accel bias
            ])
            self.process_noise = AdaptiveProcessNoise.create(constants.STATE_DIM_FULL, prior_diag, noise_strength)
        else:
            prior_diag = np.array([trans_prior**2] * 3 + [rot_prior**2] * 3)
            self.process_noise = AdaptiveProcessNoise.create(6, prior_diag, noise_strength)

        # Timestamp alignment
        align_sigma = float(self.get_parameter("alignment_sigma_prior").value)
        align_strength = float(self.get_parameter("alignment_prior_strength").value)
        align_floor = float(self.get_parameter("alignment_sigma_floor").value)
        self.timestamp_model = TimeAlignmentModel(align_sigma, align_strength, align_floor)

        # Anchor storage: anchor_id -> (mu, cov, L, h, points)
        # Primary anchor storage for pose estimation (used by odom, loop, IMU segments)
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
        self.pending_imu_factors: dict[int, list] = {}
        self.max_pending_imu_per_anchor = 100  # Prevent unbounded growth
        
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
        self.imu_factor_count = 0  # IMU segment tracking
        self.last_odom_time: Optional[float] = None
        self.last_loop_time: Optional[float] = None
        self.last_imu_time: Optional[float] = None  # IMU segment tracking
        self.last_odom_stamp: Optional[float] = None  # Odometry message timestamp for trajectory export
        self._last_odom_key: Optional[tuple] = None  # For duplicate detection
        self.node_start_time = time.time()
        self.status_period = 5.0
        self.warned_no_loops = False
        
        # Keyframe to anchor mapping (for IMU segment fusion)
        # Maps frontend keyframe IDs to backend anchor IDs
        self.keyframe_to_anchor: Dict[int, int] = {}
        
        # Post-rosbag queue: buffer last messages for processing on shutdown
        # This ensures complete trajectory coverage when rosbag playback ends
        self.post_rosbag_odom_queue = deque(maxlen=10)  # Last 10 odom messages
        self.post_rosbag_imu_queue = deque(maxlen=10)   # Last 10 IMU segments
        
        # Status timer uses wall time so it continues even when /clock pauses.
        self._status_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.status_timer = self.create_timer(
            self.status_period, self._check_status, clock=self._status_clock
        )
        
        # Startup log
        self.get_logger().info("=" * 60)
        self.get_logger().info("FL-SLAM Backend starting")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"State dimension: {self.state_dim}D")
        self.get_logger().info(f"IMU fusion: {'ENABLED' if self.enable_imu_fusion else 'DISABLED'}")
        self.get_logger().info("Subscriptions:")
        self.get_logger().info("  Delta odom:    /sim/odom (MUST come from tb3_odom_bridge)")
        self.get_logger().info("  Loop factors:  /sim/loop_factor (from frontend)")
        self.get_logger().info("  Anchors:       /sim/anchor_create (from frontend)")
        if self.enable_imu_fusion:
            self.get_logger().info("  IMU segments:  /sim/imu_segment (from frontend)")
        self.get_logger().info("  RGB-D evidence: " + self.rgbd_evidence_topic)
        self.get_logger().info("")
        self.get_logger().info("Status monitoring: Will report DEAD_RECKONING if no loop factors")
        self.get_logger().info("  Check /cdwm/backend_status for real-time status")
        self.get_logger().info("=" * 60)

        if self.enable_imu_fusion:
            # Early GPU availability check - fail fast at startup
            self.get_logger().info("IMU fusion enabled: GPU is required (fail-fast contract).")
            self._check_gpu_availability()
            self._warmup_imu_kernel()

    def on_odom(self, msg: Odometry):
        """Process delta odometry with adjoint covariance transport."""
        # Duplicate detection: skip if we've already processed this exact message
        odom_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if odom_key == self._last_odom_key:
            return  # Skip duplicate
        self._last_odom_key = odom_key
        
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
        
        # Extract delta (6D pose)
        dx = float(msg.pose.pose.position.x)
        dy = float(msg.pose.pose.position.y)
        dz = float(msg.pose.pose.position.z)
        
        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)
        rotvec_delta = quat_to_rotvec(qx, qy, qz, qw)
        
        delta_pose = np.array([dx, dy, dz, rotvec_delta[0], rotvec_delta[1], rotvec_delta[2]], dtype=float)

        # Get current state
        mu, cov = mean_cov(self.L, self.h)
        
        if self.state_dim == constants.STATE_DIM_FULL:
            # 15D state: Extract pose, compose, then rebuild
            mu_pose = mu[:6]
            mu_vel = mu[6:9]
            mu_bias = mu[9:15]
            
            linearization_point = mu_pose.copy()
            
            # Compose pose (6D)
            mu_pose_pred = se3_compose(mu_pose, delta_pose)
            
            # Get process noise (15D) - extract 6x6 pose portion for se3_cov_compose
            Q_full = self.process_noise.estimate()
            Q_pose = Q_full[:6, :6]
            
            # Transport pose covariance
            cov_pose = cov[:6, :6]
            cov_pose_pred = se3_cov_compose(cov_pose, Q_pose, mu_pose)
            
            # Rebuild full 15D state
            mu_pred = np.concatenate([mu_pose_pred, mu_vel, mu_bias])
            
            # Rebuild full 15D covariance
            # Pose block is updated, velocity/bias blocks get process noise
            cov_pred = cov.copy()
            cov_pred[:6, :6] = cov_pose_pred
            # Add process noise to velocity and bias (diagonal only)
            cov_pred[6:9, 6:9] += Q_full[6:9, 6:9]
            cov_pred[9:12, 9:12] += Q_full[9:12, 9:12]
            cov_pred[12:15, 12:15] += Q_full[12:15, 12:15]
            
            # Residual for process noise adaptation (pose only)
            residual_vec = np.zeros(self.state_dim, dtype=float)
            if self.prev_mu is not None:
                prev_pose = self.prev_mu[:6]
                predicted_from_prev = se3_compose(prev_pose, delta_pose)
                residual_vec[:6] = se3_relative(mu_pose_pred, predicted_from_prev).astype(float)
                self.process_noise.update(residual_vec)
            self.prev_mu = mu.copy()
        else:
            # 6D state: Original behavior
            linearization_point = mu.copy()
            
            # Compose pose
            mu_pred = se3_compose(mu, delta_pose)
            
            # Get adaptive process noise and transport covariance
            Q = self.process_noise.estimate()
            cov_pred = se3_cov_compose(cov, Q, mu)

            # Residual for process noise adaptation
            residual_vec = np.zeros(6, dtype=float)
            if self.prev_mu is not None:
                predicted_from_prev = se3_compose(self.prev_mu, delta_pose)
                residual_vec = se3_relative(mu_pred, predicted_from_prev).astype(float)
                self.process_noise.update(residual_vec)
            self.prev_mu = mu.copy()

        # Update state
        self.L, self.h = make_evidence(mu_pred, cov_pred)
        self.state_buffer.append((stamp_to_sec(msg.header.stamp), mu_pred.copy(), cov_pred.copy()))
        
        self._publish_state(tag="odom")

        # Frobenius correction (pose residual only for 15D)
        residual_pose = residual_vec[:6] if self.state_dim == constants.STATE_DIM_FULL else residual_vec
        _, frob_stats = gaussian_frobenius_correction(residual_pose)
        
        # Get process noise trace (full state)
        Q_full = self.process_noise.estimate()
        
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
                "state_dim": self.state_dim,
                "linearization_point": linearization_point.tolist(),
                "process_noise_trace": float(np.trace(Q_full)),
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
        # Declared initialization policy: keyframe id maps to anchor id for routing priors.
        self.keyframe_to_anchor[anchor_id] = anchor_id
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

        # Process any pending IMU segments waiting on this anchor
        if anchor_id in self.pending_imu_factors:
            pending_imu = self.pending_imu_factors.pop(anchor_id)
            self.get_logger().info(
                f"Processing {len(pending_imu)} pending IMU segments for anchor {anchor_id}"
            )
            for imu_msg in pending_imu:
                self.on_imu_segment(imu_msg)

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
        """Loop closure update via one-shot barycentric recomposition (no Schur complement)."""
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
        rotvec_rel = quat_to_rotvec(qx, qy, qz, qw)
        rel = np.array([rx, ry, rz, rotvec_rel[0], rotvec_rel[1], rotvec_rel[2]], dtype=float)

        cov_rel = np.array(msg.covariance, dtype=float).reshape(6, 6)
        weight = max(float(msg.weight), 0.0)
        
        if weight < 1e-12:
            return

        # ---------------------------------------------------------------------
        # One-shot recomposition (no Jacobians; no Schur complement)
        #
        # Factor semantics: T_current ≈ T_anchor ∘ rel, with uncertainty cov_rel.
        # We push anchor -> current, and current -> anchor (via rel^{-1}), then
        # fuse each predicted Gaussian with the corresponding prior.
        # ---------------------------------------------------------------------
        mu_full, cov_full = mean_cov(self.L, self.h)
        mu_current_pose = mu_full[:6]
        cov_current_pose = cov_full[:6, :6]
        mu_anchor_pose = mu_anchor[:6]
        cov_anchor_pose = cov_anchor[:6, :6]

        def _spd_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
            try:
                Lc = np.linalg.cholesky(A)
                y = np.linalg.solve(Lc, b)
                return np.linalg.solve(Lc.T, y)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(A, b, rcond=None)[0]

        def _gaussian_product(mu_a: np.ndarray, cov_a: np.ndarray,
                              mu_b: np.ndarray, cov_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            cov_a = np.asarray(cov_a, dtype=float)
            cov_b = np.asarray(cov_b, dtype=float)
            mu_a = np.asarray(mu_a, dtype=float).reshape(-1)
            mu_b = np.asarray(mu_b, dtype=float).reshape(-1)

            # Σ = (Σa^{-1} + Σb^{-1})^{-1}, μ = Σ(Σa^{-1}μa + Σb^{-1}μb)
            I_a = _spd_solve(cov_a, np.eye(cov_a.shape[0]))
            I_b = _spd_solve(cov_b, np.eye(cov_b.shape[0]))
            Sigma_inv = I_a + I_b
            Sigma = _spd_solve(Sigma_inv, np.eye(Sigma_inv.shape[0]))
            mu = Sigma @ (I_a @ mu_a + I_b @ mu_b)
            Sigma = 0.5 * (Sigma + Sigma.T)
            return mu, Sigma

        # Apply factor weight as precision scaling (weight * Σ^{-1} == (Σ/weight)^{-1})
        cov_rel_eff = cov_rel / max(weight, 1e-12)

        # Predict current from anchor + measurement
        mu_curr_pred = se3_compose(mu_anchor_pose, rel)
        cov_curr_pred = se3_cov_compose(cov_anchor_pose, cov_rel_eff, mu_anchor_pose)

        # Predict anchor from current + inverse(measurement)
        rel_inv = se3_inverse(rel)
        Ad_rel_inv = se3_adjoint(rel_inv)
        cov_rel_inv = Ad_rel_inv @ cov_rel_eff @ Ad_rel_inv.T
        mu_anchor_pred = se3_compose(mu_current_pose, rel_inv)
        cov_anchor_pred = se3_cov_compose(cov_current_pose, cov_rel_inv, mu_current_pose)

        # Fuse (product-of-experts) in moment space (no natural-parameter weighted sum)
        mu_pose_new, cov_pose_new = _gaussian_product(mu_current_pose, cov_current_pose, mu_curr_pred, cov_curr_pred)
        mu_anchor_new, cov_anchor_new = _gaussian_product(mu_anchor_pose, cov_anchor_pose, mu_anchor_pred, cov_anchor_pred)

        # Embed updated current pose back into full state
        if self.state_dim == constants.STATE_DIM_FULL:
            mu_new = mu_full.copy()
            mu_new[:6] = mu_pose_new

            cov_new = cov_full.copy()
            cov_new[:6, :6] = cov_pose_new
            self.L, self.h = make_evidence(mu_new, cov_new)
        else:
            self.L, self.h = make_evidence(mu_pose_new, cov_pose_new)

        # Update anchor belief (preserve points)
        anchor_points = anchor_data[4]
        
        # Embed 6D pose update back into 15D anchor state (preserve velocity and biases)
        if self.state_dim == constants.STATE_DIM_FULL:
            mu_anchor_full = mu_anchor.copy()  # Start with full 15D anchor
            mu_anchor_full[:6] = mu_anchor_new  # Update pose part
            
            cov_anchor_full = cov_anchor.copy()  # Start with full 15D covariance
            cov_anchor_full[:6, :6] = cov_anchor_new  # Update pose block
            # Cross-terms: assume pose-velocity/bias correlation is preserved (or zero)
            cov_anchor_full[:6, 6:] = 0.0
            cov_anchor_full[6:, :6] = 0.0
            
            L_anchor_new, h_anchor_new = make_evidence(mu_anchor_full, cov_anchor_full)
            self.anchors[anchor_id] = (
                mu_anchor_full.copy(), cov_anchor_full.copy(), L_anchor_new.copy(), h_anchor_new.reshape(-1).copy(), anchor_points
            )
        else:
            # 6D state: direct update
            L_anchor_new, h_anchor_new = make_evidence(mu_anchor_new, cov_anchor_new)
            self.anchors[anchor_id] = (
                mu_anchor_new, cov_anchor_new, L_anchor_new.copy(), h_anchor_new.reshape(-1).copy(), anchor_points
            )

        mu_updated, cov_updated = mean_cov(self.L, self.h)
        cov_current = cov_full  # For logging below

        # Innovation for logging/correction: rel - (T_anchor^{-1} ∘ T_current)
        Z_pred = se3_compose(se3_inverse(mu_anchor_pose), mu_current_pose)
        innovation = se3_relative(rel, Z_pred)
        
        # Publish updated map after loop closure
        self._publish_map()

        _, frob_stats = gaussian_frobenius_correction(innovation)
        
        self._publish_report(OpReport(
            name="LoopFactorRecomposition",
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
                "cov_rel_trace": float(np.trace(cov_rel)),
                "anchor_cov_trace_after": float(np.trace(cov_anchor_new)),
                "current_cov_trace_after": float(np.trace(cov_updated)),
            },
            notes="Loop closure via one-shot recomposition (anchor↔current message passing), no Jacobians and no Schur complement.",
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

    def _integrate_raw_imu_segment(
        self,
        stamps: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        bias_gyro: np.ndarray,
        bias_accel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate raw IMU measurements into (delta_p, delta_v, delta_rotvec).
        """
        if stamps.size < 2:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        delta_R = np.eye(3)
        delta_v = np.zeros(3)
        delta_p = np.zeros(3)

        for idx in range(len(stamps) - 1):
            dt = float(stamps[idx + 1] - stamps[idx])
            if dt <= 0:
                continue
            omega = (gyro[idx] - bias_gyro) * dt
            delta_R = delta_R @ rotvec_to_rotmat(omega)
            accel_corr = accel[idx] - bias_accel
            accel_rot = delta_R @ accel_corr
            # CRITICAL: Position update uses OLD velocity (tangent-space correct)
            # This is the Forster preintegration model: predict-then-retract
            delta_p = delta_p + delta_v * dt + 0.5 * accel_rot * dt * dt
            delta_v = delta_v + accel_rot * dt

        delta_rotvec = np.array(rotmat_to_rotvec(delta_R), dtype=float)
        return delta_p, delta_v, delta_rotvec

    def _estimate_preint_covariance(self, dt_total: float, n_meas: int) -> np.ndarray:
        """
        Approximate preintegration covariance from noise densities.
        """
        if n_meas < 2:
            return np.eye(9, dtype=float) * 1e6
        dt_avg = dt_total / max(n_meas - 1, 1)
        sigma_g = self.imu_gyro_noise_density / np.sqrt(max(dt_avg, 1e-12))
        sigma_a = self.imu_accel_noise_density / np.sqrt(max(dt_avg, 1e-12))

        cov_preint = np.zeros((9, 9), dtype=float)
        cov_preint[:3, :3] = np.eye(3) * (sigma_a**2) * dt_total**2
        cov_preint[3:6, 3:6] = np.eye(3) * (sigma_a**2) * dt_total
        cov_preint[6:9, 6:9] = np.eye(3) * (sigma_g**2) * dt_total
        cov_preint += np.eye(9, dtype=float) * 1e-8
        return cov_preint

    def on_imu_segment(self, msg: IMUSegment):
        """
        Process IMU segment with batched moment matching across anchors.
        
        This is the core of the Hellinger-Dirichlet IMU integration (Phase 2).
        
        Algorithm:
        1. Extract raw IMU segment and build measurement covariance
        2. Build batched anchor data (embed 6D anchors into 15D)
        3. Initialize/update Dirichlet routing module
        4. Call JAX batched projection kernel (one call per IMU packet)
        5. Exact Schur marginalization for current state
        6. Apply bias random walk noise and update 15D posterior
        
        Key invariants:
        - NO per-anchor loops (batched in kernel)
        - Global moment matching (NOT natural-param weighted sum)
        - Hellinger-tilted likelihood for robustness
        - Dirichlet routing with Frobenius retention
        """
        from fl_slam_poc.common.jax_init import jnp
        from fl_slam_poc.backend.imu_jax_kernel import imu_batched_projection_kernel
        from fl_slam_poc.backend.dirichlet_routing import DirichletRoutingModule
        
        if not self.enable_imu_fusion:
            return
            
        self.imu_factor_count += 1
        self.last_imu_time = time.time()
        
        # =====================================================================
        # Extract IMU segment data (Contract B)
        # =====================================================================
        keyframe_i = int(msg.keyframe_i)  # Reference keyframe (anchor)
        keyframe_j = int(msg.keyframe_j)  # Current keyframe
        t_i = float(msg.t_i)
        t_j = float(msg.t_j)
        dt_header = t_j - t_i

        stamps = np.asarray(msg.stamp, dtype=np.float64).reshape(-1)
        accel_raw = np.asarray(msg.accel, dtype=np.float64).reshape(-1)
        gyro_raw = np.asarray(msg.gyro, dtype=np.float64).reshape(-1)

        if stamps.size == 0 or accel_raw.size % 3 != 0 or gyro_raw.size % 3 != 0:
            self.get_logger().warn("IMU segment malformed: empty stamps or invalid accel/gyro length")
            return

        accel = accel_raw.reshape((-1, 3))
        gyro = gyro_raw.reshape((-1, 3))
        if accel.shape[0] != stamps.shape[0] or gyro.shape[0] != stamps.shape[0]:
            self.get_logger().warn("IMU segment malformed: stamps/accel/gyro length mismatch")
            return

        dt_stamps = float(stamps[-1] - stamps[0])
        dt = max(dt_stamps, 0.0)
        stamp_deltas = np.diff(stamps) if stamps.size > 1 else np.array([], dtype=np.float64)
        non_monotonic_count = int(np.sum(stamp_deltas <= 0)) if stamp_deltas.size > 0 else 0
        stamp_delta_min = float(np.min(stamp_deltas)) if stamp_deltas.size > 0 else None
        stamp_delta_mean = float(np.mean(stamp_deltas)) if stamp_deltas.size > 0 else None
        stamp_delta_max = float(np.max(stamp_deltas)) if stamp_deltas.size > 0 else None
        dt_gap_start = float(stamps[0] - t_i)
        dt_gap_end = float(t_j - stamps[-1])

        bias_gyro = np.asarray(msg.bias_ref_bg, dtype=np.float64).reshape(3)
        bias_accel = np.asarray(msg.bias_ref_ba, dtype=np.float64).reshape(3)
        bias_ref = np.concatenate([bias_gyro, bias_accel])

        gravity_msg = np.asarray(msg.gravity_world, dtype=np.float64).reshape(-1)
        if gravity_msg.size == 3 and np.linalg.norm(gravity_msg) > 0.0:
            gravity_world = gravity_msg
        else:
            gravity_world = self.gravity

        n_measurements = int(stamps.shape[0])
        R_imu = self._estimate_preint_covariance(dt, n_measurements)
        R_nom = R_imu.copy()

        delta_p, delta_v, delta_rotvec = self._integrate_raw_imu_segment(
            stamps, accel, gyro, bias_gyro, bias_accel
        )
        
        # Apply Frobenius (BCH third-order) correction to preintegrated IMU deltas
        # This corrects for SE(3) manifold curvature before deltas are used in residuals
        delta_p, delta_v, delta_rotvec, imu_frob_stats = imu_tangent_frobenius_correction(
            delta_p, delta_v, delta_rotvec, cov_preint=R_imu
        )
        
        z_imu = np.concatenate([delta_p, delta_v, delta_rotvec])
        
        # Debug logging
        if self.imu_factor_count <= 5:
            self.get_logger().info(
                f"IMU segment #{self.imu_factor_count}: kf_{keyframe_i}->kf_{keyframe_j}, "
                f"dt={dt:.3f}s, n_meas={n_measurements}, "
                f"delta_p_norm={np.linalg.norm(delta_p):.4f}m"
            )
        
        # =====================================================================
        # Validation
        # =====================================================================
        if len(self.anchors) == 0 or keyframe_i not in self.anchors:
            if keyframe_i not in self.pending_imu_factors:
                self.pending_imu_factors[keyframe_i] = []
            if len(self.pending_imu_factors[keyframe_i]) < self.max_pending_imu_per_anchor:
                self.pending_imu_factors[keyframe_i].append(msg)
            if self.imu_factor_count <= 3:
                self.get_logger().info(
                    f"IMU segment buffered: kf_{keyframe_i} (anchors={len(self.anchors)})"
                )
            self._publish_report(OpReport(
                name="IMUSegmentBuffered",
                exact=True,
                family_in="IMU",
                family_out="Gaussian",
                closed_form=True,
                domain_projection=False,
                metrics={"keyframe_i": keyframe_i, "buffered": True},
                notes="IMU segment arrived before anchor creation - buffered for processing.",
            ))
            return
        
        if self.state_dim != constants.STATE_DIM_FULL:
            # 6D state cannot use IMU segments (no velocity/bias)
            return
        
        # =====================================================================
        # Get current state (15D)
        # =====================================================================
        mu_current, cov_current = mean_cov(self.L, self.h)
        bias_prev = mu_current[9:15].copy()
        
        # =====================================================================
        # Build anchor data for batched processing
        # Anchors are 6D (pose); embed into 15D with velocity=0 and bias=bias_ref
        # =====================================================================
        anchor_ids = sorted(self.anchors.keys())
        M = len(anchor_ids)
        
        anchor_mus = np.zeros((M, 15), dtype=np.float64)
        anchor_covs = np.zeros((M, 15, 15), dtype=np.float64)
        
        for idx, aid in enumerate(anchor_ids):
            mu_a, cov_a, _, _, _ = self.anchors[aid]
            
            # Embed 6D anchor pose into 15D state
            anchor_mus[idx, :6] = mu_a[:6]           # Pose
            anchor_mus[idx, 6:9] = 0.0               # Velocity (unknown at anchor time)
            anchor_mus[idx, 9:15] = bias_ref         # Bias from preintegration reference
            
            # Embed covariance
            anchor_covs[idx, :6, :6] = cov_a[:6, :6]
            anchor_covs[idx, 6:9, 6:9] = np.eye(3) * constants.STATE_PRIOR_VELOCITY_STD**2
            anchor_covs[idx, 9:12, 9:12] = np.eye(3) * constants.STATE_PRIOR_GYRO_BIAS_STD**2
            anchor_covs[idx, 12:15, 12:15] = np.eye(3) * constants.STATE_PRIOR_ACCEL_BIAS_STD**2
        
        # =====================================================================
        # Enhanced Anchor Matching: keyframe_to_anchor mapping + Hellinger fallback
        # =====================================================================
        # Predict anchor pose from IMU segment: T_anchor_pred = T_current ∘ rel_inv
        # where rel is the IMU measurement (T_anchor^{-1} ∘ T_current)
        # IMU segment provides: rel = [delta_p, delta_v, delta_rotvec] (9D)
        # Extract 6D pose part: [delta_p, delta_rotvec]
        rel_pose = np.concatenate([delta_p, delta_rotvec])
        rel_pose_inv = se3_inverse(rel_pose)
        mu_current_pose = mu_current[:6]
        mu_anchor_pred = se3_compose(mu_current_pose, rel_pose_inv)
        
        # Compute Hellinger distances to all anchors (for fallback matching)
        # Extract 6x6 pose covariance from 9x9 IMU covariance
        # R_imu is [p(3), v(3), R(3)] so pose part is [0:3, 6:9] for position and rotation
        cov_pred_pose = np.zeros((6, 6), dtype=np.float64)
        cov_pred_pose[:3, :3] = R_imu[:3, :3]  # Position covariance
        cov_pred_pose[3:6, 3:6] = R_imu[6:9, 6:9]  # Rotation covariance
        # Cross terms (position-rotation correlation)
        cov_pred_pose[:3, 3:6] = R_imu[:3, 6:9]
        cov_pred_pose[3:6, :3] = R_imu[6:9, :3]
        
        hellinger_distances = np.zeros(M, dtype=np.float64)
        for idx, aid in enumerate(anchor_ids):
            mu_a_pose = anchor_mus[idx, :6]
            cov_a_pose = anchor_covs[idx, :6, :6]
            h_dist = hellinger_gaussian(cov_a_pose, cov_pred_pose, mu_a_pose, mu_anchor_pred)
            hellinger_distances[idx] = h_dist
        
        # =====================================================================
        # Dirichlet routing with enhanced initial logits
        # =====================================================================
        if not hasattr(self, '_imu_routing_module') or self._imu_routing_module is None:
            self._imu_routing_module = DirichletRoutingModule(n_anchors=M)
        elif self._imu_routing_module.n_anchors != M:
            self._imu_routing_module.resize(M)
        
        # Dense initial logits from Hellinger distances (soft association)
        hellinger_logit_scale = 10.0  # Tune based on typical Hellinger values
        initial_logits = -hellinger_logit_scale * (hellinger_distances ** 2)
        
        # Optional bias: Use keyframe_to_anchor mapping if available
        if keyframe_i in self.keyframe_to_anchor:
            mapped_anchor_id = self.keyframe_to_anchor[keyframe_i]
            if mapped_anchor_id in anchor_ids:
                mapped_idx = anchor_ids.index(mapped_anchor_id)
                initial_logits[mapped_idx] += 5.0  # Strong prior for mapped anchor
                if self.imu_factor_count <= 3:
                    self.get_logger().info(
                        f"IMU segment: using keyframe mapping kf_{keyframe_i} -> anchor_{mapped_anchor_id}"
                    )
        else:
            if self.imu_factor_count <= 3 and hellinger_distances.size > 0:
                min_h_idx = int(np.argmin(hellinger_distances))
                self.get_logger().info(
                    f"IMU segment: dense Hellinger logits (kf_{keyframe_i} -> anchor_{anchor_ids[min_h_idx]}, "
                    f"H²_min={hellinger_distances[min_h_idx]:.4f})"
                )
        
        routing_weights = self._imu_routing_module.update(initial_logits)
        
        # =====================================================================
        # JAX Batched Projection Kernel (Contract B)
        # =====================================================================
        imu_valid = np.ones((n_measurements,), dtype=bool)
        joint_mean_jax, cov_joint_jax, diagnostics = imu_batched_projection_kernel(
            anchor_mus=jnp.array(anchor_mus),
            anchor_covs=jnp.array(anchor_covs),
            current_mu=jnp.array(mu_current),
            current_cov=jnp.array(cov_current),
            routing_weights=jnp.array(routing_weights),
            imu_stamps=jnp.array(stamps),
            imu_accel=jnp.array(accel),
            imu_gyro=jnp.array(gyro),
            imu_valid=jnp.array(imu_valid),
            R_imu=jnp.array(R_imu),
            R_nom=jnp.array(R_nom),
            dt_total=dt,
            gravity=jnp.array(gravity_world),
        )

        joint_mean = np.array(joint_mean_jax)
        cov_joint = np.array(cov_joint_jax)

        # =====================================================================
        # Exact marginalization (Schur) on joint Gaussian
        # =====================================================================
        L_joint, h_joint = make_evidence(joint_mean, cov_joint)
        L_ii = L_joint[:15, :15]
        L_ij = L_joint[:15, 15:]
        L_ji = L_joint[15:, :15]
        L_jj = L_joint[15:, 15:]
        h_i = h_joint[:15]
        h_j = h_joint[15:]

        # Robust Schur complement using Cholesky solve
        # Regularize L_ii to prevent singular matrices
        L_ii_reg = L_ii + np.eye(15, dtype=L_ii.dtype) * 1e-8
        try:
            L_ii_chol = np.linalg.cholesky(L_ii_reg)
            L_ii_inv_L_ij = np.linalg.solve(L_ii_chol, L_ij)
            L_ii_inv_h_i = np.linalg.solve(L_ii_chol, h_i)
            L_j = L_jj - L_ji @ L_ii_inv_L_ij
            h_j = h_j - L_ji @ L_ii_inv_h_i
        except np.linalg.LinAlgError:
            # Fallback to regularized pseudo-inverse
            L_ii_reg = L_ii + np.eye(15, dtype=L_ii.dtype) * 1e-6
            L_ii_inv = np.linalg.pinv(L_ii_reg)
            L_j = L_jj - L_ji @ L_ii_inv @ L_ij
            h_j = h_j - L_ji @ L_ii_inv @ h_i

        delta_mu, delta_cov = mean_cov(L_j, h_j)
        
        # Apply SE(3) tangent-space Frobenius (BCH third-order) correction
        # This corrects linearization error from manifold curvature before retraction
        # Per Self-Adaptive Systems Guide: tangent ops + proper retraction
        delta_mu_corr, frob_stats = se3_tangent_frobenius_correction(
            delta_mu,
            state_uncertainty=np.diag(delta_cov) if delta_cov.ndim == 2 else None
        )

        # Retract delta onto current state (after Frobenius correction)
        pose_delta = delta_mu_corr[:6]
        rest_delta = delta_mu_corr[6:]
        pose_new = se3_compose(mu_current[:6], se3_exp(pose_delta))
        rest_new = mu_current[6:] + rest_delta
        mu_new = np.concatenate([pose_new, rest_new])
        cov_new = delta_cov

        # =====================================================================
        # Apply bias random walk noise (adaptive Wishart intensity)
        # =====================================================================
        bias_curr = mu_new[9:15]
        bias_innovation = bias_curr - bias_prev
        bias_rw_cov_adaptive = False
        bias_rw_cov_trace_gyro = None
        bias_rw_cov_trace_accel = None
        noise_params = None
        if self.imu_adaptive_noise is not None and dt > 0.0:
            self.imu_adaptive_noise.update_from_bias_innovations(
                accel_bias_innovations=np.atleast_2d(bias_innovation[3:6]),
                gyro_bias_innovations=np.atleast_2d(bias_innovation[:3]),
                dt=dt,
            )
            noise_params = self.imu_adaptive_noise.get_current_noise_params()
            bias_rw_cov_adaptive = True
            bias_rw_cov_trace_gyro = float(np.trace(noise_params["gyro_bias_cov"]))
            bias_rw_cov_trace_accel = float(np.trace(noise_params["accel_bias_cov"]))

        if noise_params is None:
            noise_params = {
                "gyro_bias_cov": self._imu_bias_innov_gyro_cov_prior,
                "accel_bias_cov": self._imu_bias_innov_accel_cov_prior,
            }

        Q_bias = np.zeros((6, 6), dtype=np.float64)
        Q_bias[:3, :3] = noise_params["gyro_bias_cov"] * dt
        Q_bias[3:6, 3:6] = noise_params["accel_bias_cov"] * dt
        cov_new[9:15, 9:15] += Q_bias

        # =====================================================================
        # Update state
        # =====================================================================
        self.L, self.h = make_evidence(mu_new, cov_new)
        self._publish_state(tag="imu")
        
        # =====================================================================
        # Diagnostics
        # =====================================================================
        routing_diag = self._imu_routing_module.get_update_diagnostics()
        max_resp = float(np.max(routing_weights)) if routing_weights.size > 0 else 0.0
        bias_update_source = "measurement" if np.linalg.norm(bias_innovation) > 1e-12 else "prior_only"
        
        if self.imu_factor_count <= 5:
            v_new = mu_new[6:9]
            self.get_logger().info(
                f"IMU segment #{self.imu_factor_count} applied: "
                f"v_norm={np.linalg.norm(v_new):.3f}m/s, "
                f"valid_anchors={diagnostics.get('valid_anchors', 0)}/{M}, "
                f"hellinger_shift={routing_diag['hellinger_shift']:.4f}"
            )
        
        # Emit OpReport
        self._publish_report(OpReport(
            name="IMUFactorUpdate",
            exact=False,
            approximation_triggers=["LegendreEProjection"],
            family_in="IMU",
            family_out="Gaussian",
            closed_form=False,
            frobenius_applied=True,
            frobenius_operator="gaussian_identity_third_order",
            frobenius_delta_norm=float(frob_stats["delta_norm"]),
            frobenius_input_stats=dict(frob_stats["input_stats"]),
            frobenius_output_stats=dict(frob_stats["output_stats"]),
            metrics={
                "keyframe_i": keyframe_i,
                "keyframe_j": keyframe_j,
                "dt_header": float(dt_header),
                "dt_stamps": float(dt_stamps),
                "dt_gap_start": float(dt_gap_start),
                "dt_gap_end": float(dt_gap_end),
                "stamp_delta_min": stamp_delta_min,
                "stamp_delta_mean": stamp_delta_mean,
                "stamp_delta_max": stamp_delta_max,
                "non_monotonic_count": non_monotonic_count,
                "dt_sec": dt,
                "n_measurements": n_measurements,
                "delta_p_norm_m": float(np.linalg.norm(delta_p)),
                "delta_v_norm_ms": float(np.linalg.norm(delta_v)),
                "delta_rot_norm_rad": float(np.linalg.norm(delta_rotvec)),
                "residual_p_norm": float(np.linalg.norm(delta_p)),
                "residual_v_norm": float(np.linalg.norm(delta_v)),
                "residual_rot_norm": float(np.linalg.norm(delta_rotvec)),
                "integration_order": "v_then_p",
                "velocity_usage": "updated",
                "bias_in_model": True,
                "factor_scope": "two_state",
                "projection": "e_projection(moment_match)",
                "sigma_scheme": "spherical_radial_cubature",
                "marginalization": "Schur",
                "convention_delta_R": "R_i^T R_j",
                "convention_delta_frame": "i_frame",
                "gravity_world": gravity_world.tolist(),
                "state_dim": self.state_dim,
                "n_anchors": M,
                "valid_anchors": diagnostics.get("valid_anchors", 0),
                "degenerate_weights": diagnostics.get("degenerate_weights", False),
                "ess": diagnostics.get("ess", None),
                "hellinger_mean": diagnostics.get("hellinger_mean", None),
                "weight_entropy": diagnostics.get("weight_entropy", None),
                "routing_alpha": routing_diag["alpha"].tolist(),
                "routing_w": routing_diag["responsibilities"].tolist(),
                "routing_retention": routing_diag["retention"],
                "routing_hellinger_shift": routing_diag["hellinger_shift"],
                "routing_max_resp": max_resp,
                "bias_update_source": bias_update_source,
                "bias_anchor_norm": float(np.linalg.norm(bias_ref)),
                "bias_current_norm": float(np.linalg.norm(bias_curr)),
                "bias_delta_norm": float(np.linalg.norm(bias_curr - bias_ref)),
                "bias_innovation_norm": float(np.linalg.norm(bias_innovation)),
                "bias_random_walk_applied": bool(dt > 0.0),
                "bias_rw_cov_adaptive": bias_rw_cov_adaptive,
                "bias_rw_cov_trace_gyro": bias_rw_cov_trace_gyro,
                "bias_rw_cov_trace_accel": bias_rw_cov_trace_accel,
            },
            notes="IMU two-state factor: joint update + single e-projection + Schur marginalization.",
        ))

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
        
        # Count pending factors
        total_pending = sum(len(v) for v in self.pending_loop_factors.values())
        total_pending_imu = sum(len(v) for v in self.pending_imu_factors.values())
        
        status = {
            "timestamp": time.time(),
            "elapsed_sec": elapsed,
            "mode": mode,
            "state_dim": self.state_dim,
            "odom_count": self.odom_count,
            "odom_rate_hz": round(odom_rate, 1),
            "loop_factor_count": self.loop_factor_count,
            "imu_factor_count": self.imu_factor_count,
            "anchor_count": self.anchor_count,
            "anchors_stored": len(self.anchors),
            "pending_loop_factors": total_pending,
            "pending_imu_factors": total_pending_imu,
            "last_loop_age_sec": (time.time() - self.last_loop_time) if self.last_loop_time else None,
            "last_imu_age_sec": (time.time() - self.last_imu_time) if self.last_imu_time else None,
            # Dual-layer statistics
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
        
        # Extract pose (6D) from potentially 15D state
        mu_pose = mu[:6]
        cov_pose = cov[:6, :6]  # Pose covariance only

        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = "base_link"

        out.pose.pose.position.x = float(mu_pose[0])
        out.pose.pose.position.y = float(mu_pose[1])
        out.pose.pose.position.z = float(mu_pose[2])
        
        R = rotvec_to_rotmat(mu_pose[3:6])
        qx, qy, qz, qw = rotmat_to_quat(R)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.pose.covariance = cov_pose.reshape(-1).tolist()  # Always 6x6 = 36 elements
        self.pub_state.publish(out)

        # Publish TF: odom -> base_link using the same pose we publish in /cdwm/state
        tf_msg = TransformStamped()
        tf_msg.header = out.header
        tf_msg.child_frame_id = out.child_frame_id
        tf_msg.transform.translation.x = float(mu_pose[0])
        tf_msg.transform.translation.y = float(mu_pose[1])
        tf_msg.transform.translation.z = float(mu_pose[2])
        tf_msg.transform.rotation.x = float(out.pose.pose.orientation.x)
        tf_msg.transform.rotation.y = float(out.pose.pose.orientation.y)
        tf_msg.transform.rotation.z = float(out.pose.pose.orientation.z)
        tf_msg.transform.rotation.w = float(out.pose.pose.orientation.w)
        self.tf_broadcaster.sendTransform(tf_msg)

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
        # ONLY write on odom updates (tag=="odom"), not on loop updates
        # Loop closures are corrections to existing poses, not new trajectory points
        if self.trajectory_file and self.last_odom_stamp is not None and tag == "odom":
            timestamp = self.last_odom_stamp  # Use odometry message timestamp
            self.trajectory_file.write(
                f"{timestamp:.6f} {mu_pose[0]:.6f} {mu_pose[1]:.6f} {mu_pose[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
            self.trajectory_file.flush()

    def _check_gpu_availability(self):
        """
        Check GPU availability early at startup.
        
        Raises RuntimeError if GPU is required but not available.
        This ensures failures happen during node initialization, not when
        the first IMU message arrives.
        """
        try:
            from fl_slam_poc.common.jax_init import jax
            devices = jax.devices()
            has_gpu = any(d.platform == "gpu" for d in devices)
            
            if not has_gpu:
                available_devices = [f"{d.platform}:{d.device_kind}" for d in devices]
                raise RuntimeError(
                    f"JAX GPU backend is required for IMU fusion but not available.\n"
                    f"Available JAX devices: {available_devices}\n"
                    f"To fix: Ensure CUDA is installed and JAX can detect GPU devices.\n"
                    f"Alternatively, disable IMU fusion by setting enable_imu_fusion:=false"
                )
            
            self.get_logger().info(
                f"GPU availability confirmed: {[f'{d.platform}:{d.device_kind}' for d in devices if d.platform == 'gpu']}"
            )
        except ImportError:
            raise RuntimeError(
                "JAX is required for IMU fusion but not installed.\n"
                "Install JAX with GPU support: pip install 'jax[cuda12]' or 'jax[cuda11]'"
            )
        except RuntimeError:
            # Re-raise RuntimeError (GPU not available)
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to check GPU availability: {exc}\n"
                f"This may indicate a JAX configuration issue."
            ) from exc

    def _warmup_imu_kernel(self):
        """Warm up JAX IMU kernel compilation to avoid first-call latency."""
        try:
            from fl_slam_poc.common.jax_init import jnp
            from fl_slam_poc.backend.imu_jax_kernel import imu_batched_projection_kernel

            # Minimal dummy data for warmup
            anchor_mus = jnp.zeros((1, 15), dtype=jnp.float64)
            anchor_covs = jnp.eye(15, dtype=jnp.float64)[None, :, :] * 1e-3
            current_mu = jnp.zeros((15,), dtype=jnp.float64)
            current_cov = jnp.eye(15, dtype=jnp.float64) * 1e-3
            routing_weights = jnp.array([1.0], dtype=jnp.float64)
            imu_stamps = jnp.array([0.0, 0.001], dtype=jnp.float64)
            imu_accel = jnp.zeros((2, 3), dtype=jnp.float64)
            imu_gyro = jnp.zeros((2, 3), dtype=jnp.float64)
            imu_valid = jnp.array([True, True], dtype=bool)
            R_imu = jnp.eye(9, dtype=jnp.float64) * 1e-3
            R_nom = R_imu.copy()

            imu_batched_projection_kernel(
                anchor_mus=anchor_mus,
                anchor_covs=anchor_covs,
                current_mu=current_mu,
                current_cov=current_cov,
                routing_weights=routing_weights,
                imu_stamps=imu_stamps,
                imu_accel=imu_accel,
                imu_gyro=imu_gyro,
                imu_valid=imu_valid,
                R_imu=R_imu,
                R_nom=R_nom,
                dt_total=0.001,
                gravity=jnp.array(self.gravity, dtype=jnp.float64),
            )
            self.get_logger().info("IMU kernel warmup complete.")
        except RuntimeError as exc:
            # Re-raise RuntimeError (GPU not available) - fail fast
            self.get_logger().error(f"IMU kernel warmup failed: {exc}")
            raise
        except Exception as exc:
            # Other exceptions (e.g., compilation errors) are warnings
            self.get_logger().warn(f"IMU kernel warmup failed (non-fatal): {exc}")

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
        """Clean up trajectory file and process post-rosbag queue on shutdown."""
        # Process post-rosbag queue: handle last messages that arrived after rosbag ended
        if hasattr(self, 'post_rosbag_odom_queue') and len(self.post_rosbag_odom_queue) > 0:
            self.get_logger().info(
                f"Processing {len(self.post_rosbag_odom_queue)} queued odom messages on shutdown"
            )
            for odom_msg in self.post_rosbag_odom_queue:
                try:
                    self.on_odom(odom_msg)
                except Exception as e:
                    self.get_logger().warn(f"Error processing queued odom: {e}")
        
        if hasattr(self, 'post_rosbag_imu_queue') and len(self.post_rosbag_imu_queue) > 0:
            self.get_logger().info(
                f"Processing {len(self.post_rosbag_imu_queue)} queued IMU segments on shutdown"
            )
            for imu_msg in self.post_rosbag_imu_queue:
                try:
                    self.on_imu_segment(imu_msg)
                except Exception as e:
                    self.get_logger().warn(f"Error processing queued IMU segment: {e}")
        
        # Flush and close trajectory file
        if self.trajectory_file:
            self.trajectory_file.flush()
            self.trajectory_file.close()
            self.get_logger().info("Trajectory export closed and flushed")
        
        super().destroy_node()


def main():
    rclpy.init()
    node = FLBackend()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
