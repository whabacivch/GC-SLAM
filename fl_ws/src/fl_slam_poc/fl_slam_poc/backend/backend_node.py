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
import time
from collections import deque
from typing import Optional, Dict

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from fl_slam_poc.backend import TimeAlignmentModel, AdaptiveProcessNoise
from fl_slam_poc.backend.fusion.gaussian_info import make_evidence, mean_cov
from fl_slam_poc.backend.factors.imu import process_imu_segment
from fl_slam_poc.backend.factors.odom import process_odom
from fl_slam_poc.backend.factors.loop import process_loop
from fl_slam_poc.backend.state import (
    create_anchor,
    parse_rgbd_evidence,
    process_rgbd_evidence,
)
from fl_slam_poc.backend.state.modules import Dense3DModule, SparseAnchorModule
from fl_slam_poc.backend.config import validate_backend_params
from fl_slam_poc.backend.diagnostics import (
    check_gpu_availability,
    check_status,
    publish_loop_marker,
    publish_map,
    publish_report,
    publish_state,
    warmup_imu_kernel,
)
from fl_slam_poc.common import constants
from fl_slam_poc.common.param_models import BackendParams
from fl_slam_poc.msg import AnchorCreate, IMUSegment, LoopFactor

class FLBackend(Node):
    def __init__(self):
        super().__init__("fl_backend")
        self._declare_parameters()
        self.params = self._validate_params()
        self._log_final_parameters()
        self._init_from_params()

    def _declare_parameters(self):
        self.declare_parameter("use_sim_time", False)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("rgbd_evidence_topic", "/sim/rgbd_evidence")
        
        # AUD-002: Parametrize /sim/* subscription topics (previously hardcoded)
        self.declare_parameter("odom_topic", "/sim/odom")
        self.declare_parameter("loop_factor_topic", "/sim/loop_factor")
        self.declare_parameter("anchor_create_topic", "/sim/anchor_create")
        self.declare_parameter("imu_segment_topic", "/sim/imu_segment")
        
        self.declare_parameter("alignment_sigma_prior", 0.1)
        self.declare_parameter("alignment_prior_strength", 5.0)
        self.declare_parameter("alignment_sigma_floor", 0.001)
        self.declare_parameter("process_noise_trans_prior", 0.03)
        self.declare_parameter("process_noise_rot_prior", 0.015)
        self.declare_parameter("process_noise_prior_strength", 10.0)
        
        # IMU Integration (always 15D state)
        # NOTE: enable_imu_fusion removed — 15D is the only path
        self.declare_parameter("imu_gyro_noise_density", constants.IMU_GYRO_NOISE_DENSITY_DEFAULT)
        self.declare_parameter("imu_accel_noise_density", constants.IMU_ACCEL_NOISE_DENSITY_DEFAULT)
        self.declare_parameter("imu_gyro_random_walk", constants.IMU_GYRO_RANDOM_WALK_DEFAULT)
        self.declare_parameter("imu_accel_random_walk", constants.IMU_ACCEL_RANDOM_WALK_DEFAULT)
        self.declare_parameter("gravity", list(constants.GRAVITY_DEFAULT))
        self.declare_parameter("trajectory_export_path", "/tmp/fl_slam_trajectory.tum")
        self.declare_parameter("trajectory_path_max_length", constants.TRAJECTORY_PATH_MAX_LENGTH)
        self.declare_parameter("status_check_period_sec", constants.STATUS_CHECK_PERIOD)
        self.declare_parameter("dense_association_radius", constants.DENSE_ASSOCIATION_RADIUS_DEFAULT)
        self.declare_parameter("max_dense_modules", constants.DENSE_MODULE_COMPUTE_BUDGET)
        self.declare_parameter("dense_module_keep_fraction", constants.DENSE_MODULE_KEEP_FRACTION)
        self.declare_parameter("max_pending_loops_per_anchor", constants.LOOP_PENDING_BUFFER_BUDGET)
        self.declare_parameter("max_pending_imu_per_anchor", constants.IMU_PENDING_BUFFER_BUDGET)
        self.declare_parameter("state_buffer_max_length", constants.STATE_BUFFER_MAX_LENGTH)

    def _validate_params(self) -> BackendParams:
        return validate_backend_params(self)

    def _log_final_parameters(self) -> None:
        """Log resolved parameters after precedence rules are applied."""
        params_dict = self.params.model_dump()
        self.get_logger().info(
            f"Backend parameters (final): {json.dumps(params_dict, sort_keys=True)}"
        )

    def _init_from_params(self):
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.rgbd_evidence_topic = str(self.get_parameter("rgbd_evidence_topic").value)

        # IMU noise parameters (15D state is always enabled - no 6D path)
        gyro_noise_param = self.get_parameter("imu_gyro_noise_density")
        accel_noise_param = self.get_parameter("imu_accel_noise_density")
        gyro_walk_param = self.get_parameter("imu_gyro_random_walk")
        accel_walk_param = self.get_parameter("imu_accel_random_walk")

        self.imu_gyro_noise_density = float(
            gyro_noise_param.value if gyro_noise_param.value is not None else constants.IMU_GYRO_NOISE_DENSITY_DEFAULT
        )
        self.imu_accel_noise_density = float(
            accel_noise_param.value if accel_noise_param.value is not None else constants.IMU_ACCEL_NOISE_DENSITY_DEFAULT
        )
        self.imu_gyro_random_walk = float(
            gyro_walk_param.value if gyro_walk_param.value is not None else constants.IMU_GYRO_RANDOM_WALK_DEFAULT
        )
        self.imu_accel_random_walk = float(
            accel_walk_param.value if accel_walk_param.value is not None else constants.IMU_ACCEL_RANDOM_WALK_DEFAULT
        )
        gravity_param = self.get_parameter("gravity").value
        self.gravity = np.array(gravity_param, dtype=float)

        # QoS profile for subscriptions (MUST match frontend RELIABLE QoS)
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=constants.QOS_DEPTH_SENSOR_MED_FREQ,
            durability=DurabilityPolicy.VOLATILE,
        )

        # AUD-002: Use parametrized topic names instead of hardcoded /sim/* strings
        odom_topic = str(self.get_parameter("odom_topic").value)
        loop_topic = str(self.get_parameter("loop_factor_topic").value)
        anchor_topic = str(self.get_parameter("anchor_create_topic").value)
        imu_segment_topic = str(self.get_parameter("imu_segment_topic").value)

        # Subscriptions (with RELIABLE QoS to match frontend)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, qos)
        self.sub_loop = self.create_subscription(LoopFactor, loop_topic, self.on_loop, qos)
        self.sub_anchor = self.create_subscription(AnchorCreate, anchor_topic, self.on_anchor_create, qos)
        self.sub_rgbd = self.create_subscription(String, self.rgbd_evidence_topic, self.on_rgbd_evidence, qos)
        
        # IMU segment subscription (15D state - always enabled)
        self.sub_imu_segment = self.create_subscription(
            IMUSegment, imu_segment_topic, self.on_imu_segment, qos
        )
        
        # Store topic names for wiring banner
        self._topics = {
            "odom": odom_topic,
            "loop_factor": loop_topic,
            "anchor_create": anchor_topic,
            "imu_segment": imu_segment_topic,
            "rgbd_evidence": self.rgbd_evidence_topic,
        }

        # Publishers
        self.pub_state = self.create_publisher(Odometry, "/cdwm/state", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/cdwm/markers", 10)
        self.pub_dbg = self.create_publisher(String, "/cdwm/debug", 10)
        self.pub_report = self.create_publisher(String, "/cdwm/op_report", 10)
        self.pub_loop_markers = self.create_publisher(MarkerArray, "/cdwm/loop_markers", 10)
        
        # Map publisher (point cloud)
        self.pub_map = self.create_publisher(PointCloud2, "/cdwm/map", 10)
        self.PointCloud2 = PointCloud2
        self.PointField = PointField
        
        # Trajectory path publisher for Foxglove visualization
        self.pub_path = self.create_publisher(Path, "/cdwm/trajectory", 10)
        self.trajectory_poses: list[PoseStamped] = []
        
        # Trajectory export for ground truth comparison
        self.trajectory_export_path = str(self.get_parameter("trajectory_export_path").value)
        self.trajectory_file = None
        if self.trajectory_export_path:
            self.trajectory_file = open(self.trajectory_export_path, "w")
            self.trajectory_file.write("# timestamp x y z qx qy qz qw\n")
            self.get_logger().info(f"Exporting trajectory to: {self.trajectory_export_path}")
        self.max_path_length = int(self.get_parameter("trajectory_path_max_length").value)
        
        # State dimension: always 15D (no 6D pose-only path)
        self.state_dim = constants.STATE_DIM_FULL
        
        # State belief in information form
        # 15D state: [p(3), R(3), v(3), b_g(3), b_a(3)]
        mu0 = np.zeros(constants.STATE_DIM_FULL)
        cov0_diag = np.concatenate([
            np.array([constants.STATE_PRIOR_POSE_TRANS_STD**2] * 3),   # Position [0:3]
            np.array([constants.STATE_PRIOR_POSE_ROT_STD**2] * 3),     # Rotation [3:6]
            np.array([constants.STATE_PRIOR_VELOCITY_STD**2] * 3),     # Velocity [6:9]
            np.array([constants.STATE_PRIOR_GYRO_BIAS_STD**2] * 3),    # Gyro bias [9:12]
            np.array([constants.STATE_PRIOR_ACCEL_BIAS_STD**2] * 3),   # Accel bias [12:15]
        ])
        cov0 = np.diag(cov0_diag)
        self.get_logger().info("Backend: 15D state (pose + velocity + biases)")
        self.L, self.h = make_evidence(mu0, cov0)

        # Adaptive process noise (15D state - no 6D path)
        trans_prior = float(self.get_parameter("process_noise_trans_prior").value)
        rot_prior = float(self.get_parameter("process_noise_rot_prior").value)
        noise_strength = float(self.get_parameter("process_noise_prior_strength").value)
        
        # 15D process noise: pose + velocity + bias random walks
        prior_diag = np.concatenate([
            np.array([trans_prior**2] * 3),                            # Position
            np.array([rot_prior**2] * 3),                              # Rotation
            np.array([constants.PROCESS_NOISE_VELOCITY_STD**2] * 3),   # Velocity
            np.array([constants.PROCESS_NOISE_GYRO_BIAS_STD**2] * 3),  # Gyro bias
            np.array([constants.PROCESS_NOISE_ACCEL_BIAS_STD**2] * 3), # Accel bias
        ])
        self.process_noise = AdaptiveProcessNoise.create(constants.STATE_DIM_FULL, prior_diag, noise_strength)

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
        
        # Dense module configuration (soft association scale; not a hard radius)
        self.dense_association_radius = float(self.get_parameter("dense_association_radius").value)
        self.max_dense_modules = int(self.get_parameter("max_dense_modules").value)
        self.dense_module_keep_fraction = float(self.get_parameter("dense_module_keep_fraction").value)
        
        # Loop factor buffer for race condition protection
        # Stores loop factors that arrive before their anchor is created
        self.pending_loop_factors: dict[int, list] = {}
        self.max_pending_loops_per_anchor = int(self.get_parameter("max_pending_loops_per_anchor").value)
        self.pending_imu_factors: dict[int, list] = {}
        self.max_pending_imu_per_anchor = int(self.get_parameter("max_pending_imu_per_anchor").value)
        
        # State buffer for timestamp alignment
        self.state_buffer = deque(maxlen=int(self.get_parameter("state_buffer_max_length").value))
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
        self.status_period = float(self.get_parameter("status_check_period_sec").value)
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
        
        # AUD-002: Startup wiring banner for observability
        logger = self.get_logger()
        logger.info("=" * 60)
        logger.info("BACKEND WIRING BANNER")
        logger.info("=" * 60)
        logger.info(f"State dimension: {self.state_dim}D (15D state - pose + velocity + biases)")
        logger.info("")
        logger.info("Subscriptions:")
        logger.info(f"  {self._topics['odom']} (nav_msgs/Odometry) QoS: RELIABLE")
        logger.info(f"  {self._topics['loop_factor']} (fl_slam_poc/LoopFactor) QoS: RELIABLE")
        logger.info(f"  {self._topics['anchor_create']} (fl_slam_poc/AnchorCreate) QoS: RELIABLE")
        logger.info(f"  {self._topics['imu_segment']} (fl_slam_poc/IMUSegment) QoS: RELIABLE")
        logger.info(f"  {self._topics['rgbd_evidence']} (std_msgs/String) QoS: RELIABLE")
        logger.info("")
        logger.info("Publications:")
        logger.info("  /cdwm/state (nav_msgs/Odometry)")
        logger.info("  /cdwm/trajectory (nav_msgs/Path)")
        logger.info("  /cdwm/map (sensor_msgs/PointCloud2)")
        logger.info("  /cdwm/backend_status (std_msgs/String)")
        logger.info("  /cdwm/op_report (std_msgs/String)")
        logger.info("  /cdwm/markers (visualization_msgs/MarkerArray)")
        logger.info("  /cdwm/loop_markers (visualization_msgs/MarkerArray)")
        logger.info("")
        logger.info("Status monitoring: Will report DEAD_RECKONING if no loop factors")
        logger.info("  Check /cdwm/backend_status for real-time status")
        logger.info("=" * 60)

        # Early GPU availability check - fail fast at startup
        check_gpu_availability(self)
        warmup_imu_kernel(self, self.gravity.tolist())

    def on_odom(self, msg: Odometry):
        process_odom(self, msg)

    def on_anchor_create(self, msg: AnchorCreate):
        """Create anchor with probabilistic timestamp weighting."""
        create_anchor(self, msg)

    def on_rgbd_evidence(self, msg: String):
        """
        Receive RGB-D evidence (JSON payload) and update dense map layer.

        Payload schema:
          {"evidence": [ {position_L, position_h, color_L, color_h, normal_theta, alpha_mean, alpha_var}, ... ]}
        """
        evidence_list = parse_rgbd_evidence(msg.data)
        if len(evidence_list) == 0:
            return
        process_rgbd_evidence(self, evidence_list)

    def on_loop(self, msg: LoopFactor):
        process_loop(self, msg)

    def on_imu_segment(self, msg: IMUSegment):
        process_imu_segment(self, msg)

    def _check_status(self):
        """Periodic status check - warns if running dead-reckoning only."""
        check_status(
            self,
            self.node_start_time,
            self.odom_count,
            self.loop_factor_count,
            self.anchor_count,
            self.imu_factor_count,
            self.last_loop_time,
            self.pending_loop_factors,
            self.pending_imu_factors,
            self.anchors,
            self.sparse_anchors,
            self.dense_modules,
            self.pub_status,
        )

    def _publish_state(self, tag: str):
        publish_state(
            self, tag, self.L, self.h, self.odom_frame,
            self.pub_state, self.pub_path, self.tf_broadcaster,
            self.trajectory_poses, self.max_path_length,
            self.trajectory_file, self.last_odom_stamp,
        )

    def get_state_summary(self) -> dict:
        """Return a structured summary of backend state for inspection."""
        mu, cov = mean_cov(self.L, self.h)
        return {
            "state_dim": int(self.state_dim),
            "odom_count": int(self.odom_count),
            "loop_factor_count": int(self.loop_factor_count),
            "imu_factor_count": int(self.imu_factor_count),
            "anchor_count": int(len(self.anchors)),
            "dense_module_count": int(len(self.dense_modules)),
            "sparse_anchor_count": int(len(self.sparse_anchors)),
            "pending_loop_factors": int(sum(len(v) for v in self.pending_loop_factors.values())),
            "pending_imu_factors": int(sum(len(v) for v in self.pending_imu_factors.values())),
            "last_odom_time": float(self.last_odom_time) if self.last_odom_time is not None else None,
            "last_loop_time": float(self.last_loop_time) if self.last_loop_time is not None else None,
            "last_imu_time": float(self.last_imu_time) if self.last_imu_time is not None else None,
            "last_odom_stamp": float(self.last_odom_stamp) if self.last_odom_stamp is not None else None,
            "mean_head": mu[:6].tolist(),
            "cov_trace": float(np.trace(cov)),
        }
    
    def destroy_node(self):
        """Clean up trajectory file and process post-rosbag queue on shutdown."""
        # Process post-rosbag queue: handle last messages that arrived after rosbag ended
        if hasattr(self, 'post_rosbag_odom_queue') and len(self.post_rosbag_odom_queue) > 0:
            self.get_logger().info(
                f"Processing {len(self.post_rosbag_odom_queue)} queued odom messages on shutdown"
            )
            for odom_msg in self.post_rosbag_odom_queue:
                self.on_odom(odom_msg)
        
        if hasattr(self, 'post_rosbag_imu_queue') and len(self.post_rosbag_imu_queue) > 0:
            self.get_logger().info(
                f"Processing {len(self.post_rosbag_imu_queue)} queued IMU segments on shutdown"
            )
            for imu_msg in self.post_rosbag_imu_queue:
                self.on_imu_segment(imu_msg)
        
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
