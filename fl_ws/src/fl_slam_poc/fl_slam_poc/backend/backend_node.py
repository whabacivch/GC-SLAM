"""
Golden Child SLAM v2 Backend Node.

Actually uses the GC operators to process LiDAR scans.
This is NOT passthrough - it runs the full 14-step pipeline.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md
"""

import json
import struct
import time
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, Imu, PointField
from std_msgs.msg import String

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    D_Z,
    se3_identity,
    se3_from_rotvec_trans,
    se3_to_rotvec_trans,
)
from fl_slam_poc.common.certificates import CertBundle
from fl_slam_poc.common.certificates import InfluenceCert, aggregate_certificates
from fl_slam_poc.backend.pipeline import (
    PipelineConfig,
    RuntimeManifest,
    process_scan_single_hypothesis,
    process_hypotheses,
    ScanPipelineResult,
)
from fl_slam_poc.backend.structures.inverse_wishart_jax import (
    ProcessNoiseIWState,
    create_datasheet_process_noise_state,
)
from fl_slam_poc.backend.operators.inverse_wishart_jax import (
    process_noise_state_to_Q_jax,
    process_noise_iw_apply_suffstats_jax,
)
from fl_slam_poc.backend.structures.measurement_noise_iw_jax import (
    MeasurementNoiseIWState,
    create_datasheet_measurement_noise_state,
)
from fl_slam_poc.backend.operators.measurement_noise_iw_jax import (
    measurement_noise_mean_jax,
    measurement_noise_apply_suffstats_jax,
)
from fl_slam_poc.backend.structures.lidar_bucket_noise_iw_jax import (
    LidarBucketNoiseIWState,
    create_datasheet_lidar_bucket_noise_state,
)
from fl_slam_poc.backend.operators.lidar_bucket_noise_iw_jax import (
    lidar_bucket_iw_apply_suffstats_jax,
)
from fl_slam_poc.backend.structures.bin_atlas import (
    BinAtlas,
    MapBinStats,
    create_fibonacci_atlas,
    create_empty_map_stats,
    apply_forgetting,
    update_map_stats,
)
from fl_slam_poc.backend.diagnostics import ScanDiagnostics, DiagnosticsLog

from scipy.spatial.transform import Rotation


def _smooth_window_weight(dist: float, min_r: float, max_r: float, sigma: float) -> float:
    """Continuous range weighting without hard gates."""
    # Smooth window: sigmoid(dist-min_r) * sigmoid(max_r-dist)
    a = (dist - min_r) / sigma
    b = (max_r - dist) / sigma
    w_min = 1.0 / (1.0 + np.exp(-a))
    w_max = 1.0 / (1.0 + np.exp(-b))
    return float(w_min * w_max)


def _pointfield_to_dtype(datatype: int) -> np.dtype:
    """Map sensor_msgs.msg.PointField datatype to numpy dtype."""
    if datatype == PointField.INT8:
        return np.dtype("i1")
    if datatype == PointField.UINT8:
        return np.dtype("u1")
    if datatype == PointField.INT16:
        return np.dtype("<i2")
    if datatype == PointField.UINT16:
        return np.dtype("<u2")
    if datatype == PointField.INT32:
        return np.dtype("<i4")
    if datatype == PointField.UINT32:
        return np.dtype("<u4")
    if datatype == PointField.FLOAT32:
        return np.dtype("<f4")
    if datatype == PointField.FLOAT64:
        return np.dtype("<f8")
    raise ValueError(f"Unsupported PointField datatype: {datatype}")


def parse_pointcloud2_vectorized(
    msg: PointCloud2,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Parse PointCloud2 message to extract xyz points.
    
    Returns:
        points: (N, 3) array of xyz coordinates
        timestamps: (N,) array of per-point timestamps (or zeros if unavailable)
        weights: (N,) array of continuous point weights
        ring: (N,) uint8 ring/line id (0 if unavailable)
        tag: (N,) uint8 tag class (0 if unavailable)
    """
    n_points = msg.width * msg.height
    if n_points <= 0:
        return (
            jnp.zeros((0, 3), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.uint8),
            jnp.zeros((0,), dtype=jnp.uint8),
        )

    field_map = {f.name: (f.offset, f.datatype) for f in msg.fields}
    # Fail-fast: no silent fallbacks for required fields (single math path).
    required = ["x", "y", "z", "ring", "tag", "timebase_low", "timebase_high", "time_offset"]
    missing = [k for k in required if k not in field_map]
    if missing:
        raise RuntimeError(
            f"PointCloud2 missing required fields for GC v2 (no fallback): {missing}. "
            f"Present fields: {sorted(list(field_map.keys()))}"
        )
    if "x" not in field_map or "y" not in field_map or "z" not in field_map:
        return (
            jnp.zeros((0, 3), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.uint8),
            jnp.zeros((0,), dtype=jnp.uint8),
        )

    needed = ["x", "y", "z", "intensity", "ring", "tag", "time_offset", "timebase_low", "timebase_high"]
    names = []
    formats = []
    offsets = []
    for name in needed:
        if name in field_map:
            off, dt = field_map[name]
            names.append(name)
            formats.append(_pointfield_to_dtype(dt))
            offsets.append(off)

    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dtype, count=n_points)

    # Domain projection (wrapper boundary): replace non-finite values with large finite sentinels.
    # This avoids NaN propagation without discrete gating inside likelihood math.
    sentinel = float(constants.GC_NONFINITE_SENTINEL)
    x = np.nan_to_num(np.asarray(arr["x"], dtype=np.float64), nan=sentinel, posinf=sentinel, neginf=-sentinel)
    y = np.nan_to_num(np.asarray(arr["y"], dtype=np.float64), nan=sentinel, posinf=sentinel, neginf=-sentinel)
    z = np.nan_to_num(np.asarray(arr["z"], dtype=np.float64), nan=sentinel, posinf=sentinel, neginf=-sentinel)

    low = np.uint64(arr["timebase_low"][0])
    high = np.uint64(arr["timebase_high"][0])
    timebase = (high << np.uint64(32)) | low
    timebase_sec = float(timebase) * 1e-9
    offs = np.asarray(arr["time_offset"], dtype=np.uint64)
    t = timebase_sec + offs.astype(np.float64) * 1e-9

    # Metadata (optional)
    ring = np.zeros((n_points,), dtype=np.uint8)
    tag = np.zeros((n_points,), dtype=np.uint8)
    if "ring" in arr.dtype.names:
        ring = np.asarray(arr["ring"], dtype=np.uint8)
    if "tag" in arr.dtype.names:
        tag = np.asarray(arr["tag"], dtype=np.uint8)

    # Continuous range-based weighting (vectorized; strictly positive floor)
    dist = np.sqrt(x * x + y * y + z * z)
    sigma = float(constants.GC_RANGE_WEIGHT_SIGMA)
    min_r = float(constants.GC_RANGE_WEIGHT_MIN_R)
    max_r = float(constants.GC_RANGE_WEIGHT_MAX_R)
    a = (dist - min_r) / sigma
    b = (max_r - dist) / sigma
    w_min = 1.0 / (1.0 + np.exp(-a))
    w_max = 1.0 / (1.0 + np.exp(-b))
    w_raw = (w_min * w_max).astype(np.float64)
    wf = float(constants.GC_WEIGHT_FLOOR)
    w = w_raw * (1.0 - wf) + wf

    pts = np.stack([x, y, z], axis=1)
    return (
        jnp.array(pts, dtype=jnp.float64),
        jnp.array(t, dtype=jnp.float64),
        jnp.array(w, dtype=jnp.float64),
        jnp.array(ring),
        jnp.array(tag),
    )


class GoldenChildBackend(Node):
    """
    Golden Child SLAM v2 Backend.
    
    Actually runs the 14-step pipeline on each LiDAR scan.
    """

    def __init__(self):
        super().__init__("gc_backend")
        
        self._declare_parameters()
        self._init_state()
        self._init_ros()
        self._publish_runtime_manifest()

        self.get_logger().info("Golden Child SLAM v2 Backend initialized - PIPELINE ENABLED")

    def _declare_parameters(self):
        """Declare ROS parameters."""
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        # Backend subscribes ONLY to /gc/sensors/* (canonical topics from sensor hub)
        self.declare_parameter("lidar_topic", "/gc/sensors/lidar_points")
        self.declare_parameter("odom_topic", "/gc/sensors/odom")
        self.declare_parameter("imu_topic", "/gc/sensors/imu")
        self.declare_parameter("trajectory_export_path", "/tmp/gc_slam_trajectory.tum")
        self.declare_parameter("diagnostics_export_path", "/tmp/gc_slam_diagnostics.npz")
        self.declare_parameter("status_check_period_sec", 5.0)
        self.declare_parameter("forgetting_factor", 0.99)
        # Hard single-path enforcement: if enabled, missing topics are hard errors.
        self.declare_parameter("use_imu", True)
        self.declare_parameter("use_odom", True)

    def _init_state(self):
        """Initialize Golden Child state."""
        # Pipeline configuration
        self.config = PipelineConfig()
        
        # Adaptive process noise IW state (datasheet priors) + derived Q
        self.process_noise_state: ProcessNoiseIWState = create_datasheet_process_noise_state()
        self.Q = process_noise_state_to_Q_jax(self.process_noise_state)
        
        # Adaptive measurement noise IW state (per-sensor, phase 1) + derived Sigma_meas (LiDAR)
        self.measurement_noise_state: MeasurementNoiseIWState = create_datasheet_measurement_noise_state()
        self.config.Sigma_meas = measurement_noise_mean_jax(self.measurement_noise_state, idx=2)

        # LiDAR per-(line,tag) bucket noise IW state (Phase 3 part 2)
        self.lidar_bucket_noise_state: LidarBucketNoiseIWState = create_datasheet_lidar_bucket_noise_state()

        # Bin atlas for directional binning
        self.bin_atlas = create_fibonacci_atlas(self.config.B_BINS)
        
        # Map statistics (accumulates over time with forgetting)
        self.map_stats = create_empty_map_stats(self.config.B_BINS)
        self.forgetting_factor = float(self.get_parameter("forgetting_factor").value)
        
        # Initialize hypotheses with identity prior
        self.hypotheses: List[BeliefGaussianInfo] = []
        self.hyp_weights = jnp.ones(self.config.K_HYP) / self.config.K_HYP
        
        for i in range(self.config.K_HYP):
            belief = BeliefGaussianInfo.create_identity_prior(
                anchor_id=f"hyp_{i}_anchor_0",
                stamp_sec=0.0,
                prior_precision=1e-6,
            )
            self.hypotheses.append(belief)
        
        # Current best estimate from hypotheses
        self.current_belief: Optional[BeliefGaussianInfo] = self.hypotheses[0]
        
        # Odometry for initial guess / prediction
        self.last_odom_pose = None
        self.last_odom_stamp = 0.0
        self.last_odom_cov_se3 = None  # (6,6) in [x,y,z,roll,pitch,yaw] ~ [trans, rotvec]
        
        # IMU buffer for high-rate prediction
        self.imu_buffer: List[Tuple[float, jnp.ndarray, jnp.ndarray]] = []
        self.max_imu_buffer = 200
        
        # Tracking
        self.imu_count = 0
        self.odom_count = 0
        self.scan_count = 0
        self.pipeline_runs = 0
        self.last_scan_stamp = 0.0
        self.node_start_time = time.time()
        
        # Certificate history
        self.cert_history: List[CertBundle] = []

        # Diagnostics log for dashboard
        self.diagnostics_log = DiagnosticsLog(
            run_id=f"gc_slam_{int(self.node_start_time)}",
            start_time=self.node_start_time,
        )

    def _init_ros(self):
        """Initialize ROS interfaces."""
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        lidar_topic = str(self.get_parameter("lidar_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        imu_topic = str(self.get_parameter("imu_topic").value)
        
        self.sub_lidar = self.create_subscription(
            PointCloud2, lidar_topic, self.on_lidar, qos_sensor
        )
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.on_odom, qos_reliable
        )
        self.sub_imu = self.create_subscription(
            Imu, imu_topic, self.on_imu, qos_sensor
        )
        
        self.get_logger().info(f"LiDAR: {lidar_topic} (PIPELINE ACTIVE)")
        self.get_logger().info(f"Odom: {odom_topic}")
        self.get_logger().info(f"IMU: {imu_topic}")
        
        # Publishers
        self.pub_state = self.create_publisher(Odometry, "/gc/state", 10)
        self.pub_path = self.create_publisher(Path, "/gc/trajectory", 10)
        self.pub_manifest = self.create_publisher(String, "/gc/runtime_manifest", 10)
        self.pub_cert = self.create_publisher(String, "/gc/certificate", 10)
        self.pub_status = self.create_publisher(String, "/gc/status", 10)
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Trajectory export
        self.trajectory_export_path = str(self.get_parameter("trajectory_export_path").value)
        self.trajectory_file = None
        if self.trajectory_export_path:
            self.trajectory_file = open(self.trajectory_export_path, "w")
            self.trajectory_file.write("# timestamp x y z qx qy qz qw\n")
        
        self.trajectory_poses: List[PoseStamped] = []
        self.max_path_length = 1000
        
        # Status timer
        status_period = float(self.get_parameter("status_check_period_sec").value)
        self._status_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.status_timer = self.create_timer(
            status_period, self._publish_status, clock=self._status_clock
        )

    def _publish_runtime_manifest(self):
        """Publish RuntimeManifest at startup."""
        manifest = RuntimeManifest()
        manifest_dict = manifest.to_dict()
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("GOLDEN CHILD RUNTIME MANIFEST")
        self.get_logger().info("=" * 60)
        for key, value in manifest_dict.items():
            self.get_logger().info(f"  {key}: {value}")
        self.get_logger().info("=" * 60)
        
        msg = String()
        msg.data = json.dumps(manifest_dict)
        self.pub_manifest.publish(msg)

    def on_imu(self, msg: Imu):
        """Buffer IMU measurements for prediction."""
        self.imu_count += 1
        
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Keep callback CPU-only (no JAX ops per message).
        gyro = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=np.float64,
        )
        accel = np.array(
            [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            dtype=np.float64,
        )

        self.imu_buffer.append((stamp_sec, gyro, accel))
        
        # Keep buffer bounded
        if len(self.imu_buffer) > self.max_imu_buffer:
            self.imu_buffer.pop(0)

    def on_odom(self, msg: Odometry):
        """Store latest odometry for delta computation."""
        self.odom_count += 1
        
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        quat = [ori.x, ori.y, ori.z, ori.w]
        R = Rotation.from_quat(quat)
        rotvec = R.as_rotvec()
        
        self.last_odom_pose = se3_from_rotvec_trans(
            jnp.array(rotvec, dtype=jnp.float64),
            jnp.array([pos.x, pos.y, pos.z], dtype=jnp.float64)
        )
        self.last_odom_stamp = stamp_sec
        # Pose covariance is row-major 6x6: [x,y,z,roll,pitch,yaw]
        cov = np.array(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        self.last_odom_cov_se3 = jnp.array(cov, dtype=jnp.float64)

    def on_lidar(self, msg: PointCloud2):
        """
        Process LiDAR scan through the full GC pipeline.

        This is where the actual SLAM happens!
        """
        self.scan_count += 1
        self.get_logger().info(f"on_lidar callback #{self.scan_count} received")

        use_imu = bool(self.get_parameter("use_imu").value)
        use_odom = bool(self.get_parameter("use_odom").value)

        # Sensor warmup check: skip scans until required sensors have data
        # This allows the system to warm up without crashing on startup race conditions
        # Check BEFORE parsing pointcloud to avoid wasted computation
        if use_odom and (self.last_odom_pose is None or self.last_odom_cov_se3 is None):
            self.get_logger().warn(
                f"Scan #{self.scan_count} skipped: waiting for odometry (use_odom=True)"
            )
            return
        if use_imu and (len(self.imu_buffer) == 0):
            self.get_logger().warn(
                f"Scan #{self.scan_count} skipped: waiting for IMU (use_imu=True)"
            )
            return

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Parse point cloud (vectorized; preserves Livox metadata)
        points, timestamps, weights, ring, tag = parse_pointcloud2_vectorized(msg)
        n_points = points.shape[0]

        if n_points == 0:
            # Keep pipeline total by supplying a zero-weight dummy point.
            points = jnp.zeros((1, 3), dtype=jnp.float64)
            timestamps = jnp.zeros(1, dtype=jnp.float64)
            weights = jnp.zeros(1, dtype=jnp.float64)
            n_points = 1

        # Compute dt since last scan
        # Derive scan bounds from per-point timestamps when available; always include header stamp.
        stamp_j = jnp.array(stamp_sec, dtype=jnp.float64)
        scan_start_time = float(jnp.minimum(stamp_j, jnp.min(timestamps)))
        scan_end_time = float(jnp.maximum(stamp_j, jnp.max(timestamps)))

        dt_raw = (
            scan_end_time - self.last_scan_stamp
            if self.last_scan_stamp > 0
            else (scan_end_time - scan_start_time)
        )
        eps_dt = np.finfo(np.float64).eps
        dt_sec = float(jnp.sqrt(jnp.array(dt_raw, dtype=jnp.float64) ** 2 + eps_dt))
        self.last_scan_stamp = scan_end_time

        # Apply forgetting to map stats
        self.map_stats = apply_forgetting(self.map_stats, self.forgetting_factor)

        # Build fixed-size IMU arrays (pads with zeros; contributions are controlled by continuous weights only)
        M = self.max_imu_buffer
        imu_stamps = np.zeros((M,), dtype=np.float64)
        imu_gyro = np.zeros((M, 3), dtype=np.float64)
        imu_accel = np.zeros((M, 3), dtype=np.float64)
        tail = self.imu_buffer[-M:]
        for i, (t, g, a) in enumerate(tail):
            imu_stamps[i] = float(t)
            imu_gyro[i, :] = np.array(g)
            imu_accel[i, :] = np.array(a)
        imu_stamps_j = jnp.array(imu_stamps, dtype=jnp.float64)
        imu_gyro_j = jnp.array(imu_gyro, dtype=jnp.float64)
        imu_accel_j = jnp.array(imu_accel, dtype=jnp.float64)
        
        # Run pipeline for each hypothesis
        results: List[ScanPipelineResult] = []
        # Commutative IW sufficient-statistics accumulation (order-robust; applied once per scan)
        accum_dPsi = jnp.zeros((7, 6, 6), dtype=jnp.float64)
        accum_dnu = jnp.zeros((7,), dtype=jnp.float64)
        accum_meas_dPsi = jnp.zeros((3, 3, 3), dtype=jnp.float64)
        accum_meas_dnu = jnp.zeros((3,), dtype=jnp.float64)
        accum_lidar_bucket_dPsi = jnp.zeros((constants.GC_LIDAR_N_BUCKETS, 3, 3), dtype=jnp.float64)
        accum_lidar_bucket_dnu = jnp.zeros((constants.GC_LIDAR_N_BUCKETS,), dtype=jnp.float64)
        
        try:
            # Update per-scan measurement covariance from IW state (shared across hypotheses)
            self.config.Sigma_meas = measurement_noise_mean_jax(self.measurement_noise_state, idx=2)
            self.config.Sigma_g = measurement_noise_mean_jax(self.measurement_noise_state, idx=0)
            self.config.Sigma_a = measurement_noise_mean_jax(self.measurement_noise_state, idx=1)

            # Precompute Q once for this scan (shared across hypotheses)
            Q_scan = process_noise_state_to_Q_jax(self.process_noise_state)

            for i, belief in enumerate(self.hypotheses):
                result = process_scan_single_hypothesis(
                    belief_prev=belief,
                    raw_points=points,
                    raw_timestamps=timestamps,
                    raw_weights=weights,
                    raw_ring=ring,
                    raw_tag=tag,
                    lidar_bucket_state=self.lidar_bucket_noise_state,
                    imu_stamps=imu_stamps_j,
                    imu_gyro=imu_gyro_j,
                    imu_accel=imu_accel_j,
                    odom_pose=self.last_odom_pose if self.last_odom_pose is not None else se3_identity(),
                    odom_cov_se3=self.last_odom_cov_se3 if self.last_odom_cov_se3 is not None else jnp.eye(6, dtype=jnp.float64),
                    scan_start_time=scan_start_time,
                    scan_end_time=scan_end_time,
                    dt_sec=dt_sec,
                    Q=Q_scan,
                    bin_atlas=self.bin_atlas,
                    map_stats=self.map_stats,
                    config=self.config,
                )
                results.append(result)
                self.hypotheses[i] = result.belief_updated

                # Accumulate commutative IW sufficient statistics
                w_h = float(self.hyp_weights[i])
                accum_dPsi = accum_dPsi + w_h * result.iw_process_dPsi
                accum_dnu = accum_dnu + w_h * result.iw_process_dnu
                accum_meas_dPsi = accum_meas_dPsi + w_h * result.iw_meas_dPsi
                accum_meas_dnu = accum_meas_dnu + w_h * result.iw_meas_dnu
                accum_lidar_bucket_dPsi = accum_lidar_bucket_dPsi + w_h * result.iw_lidar_bucket_dPsi
                accum_lidar_bucket_dnu = accum_lidar_bucket_dnu + w_h * result.iw_lidar_bucket_dnu
            
            # Combine hypotheses
            combined_belief, combo_cert, combo_effect = process_hypotheses(
                hypotheses=self.hypotheses,
                weights=self.hyp_weights,
                config=self.config,
            )
            
            self.current_belief = combined_belief
            self.pipeline_runs += 1

            # Apply process-noise IW update ONCE per scan (after hypothesis combine)
            self.process_noise_state, proc_iw_cert_vec = process_noise_iw_apply_suffstats_jax(
                pn_state=self.process_noise_state,
                dPsi=accum_dPsi,
                dnu=accum_dnu,
                dt_sec=dt_sec,
                eps_psd=self.config.eps_psd,
            )
            self.Q = process_noise_state_to_Q_jax(self.process_noise_state)

            # Apply measurement-noise IW update ONCE per scan (after hypothesis combine)
            self.measurement_noise_state, meas_iw_cert_vec = measurement_noise_apply_suffstats_jax(
                mn_state=self.measurement_noise_state,
                dPsi_blocks=accum_meas_dPsi,
                dnu=accum_meas_dnu,
                eps_psd=self.config.eps_psd,
            )

            # Apply LiDAR bucket IW update ONCE per scan (after hypothesis combine)
            self.lidar_bucket_noise_state = lidar_bucket_iw_apply_suffstats_jax(
                state=self.lidar_bucket_noise_state,
                dPsi=accum_lidar_bucket_dPsi,
                dnu=accum_lidar_bucket_dnu,
                eps_psd=self.config.eps_psd,
            )
            
            # Update map statistics with weighted increments
            delta_S_dir = jnp.zeros_like(self.map_stats.S_dir)
            delta_N_dir = jnp.zeros_like(self.map_stats.N_dir)
            delta_N_pos = jnp.zeros_like(self.map_stats.N_pos)
            delta_sum_p = jnp.zeros_like(self.map_stats.sum_p)
            delta_sum_ppT = jnp.zeros_like(self.map_stats.sum_ppT)
            for i, result in enumerate(results):
                w_h = float(self.hyp_weights[i])
                delta_S_dir = delta_S_dir + w_h * result.map_increments.delta_S_dir
                delta_N_dir = delta_N_dir + w_h * result.map_increments.delta_N_dir
                delta_N_pos = delta_N_pos + w_h * result.map_increments.delta_N_pos
                delta_sum_p = delta_sum_p + w_h * result.map_increments.delta_sum_p
                delta_sum_ppT = delta_sum_ppT + w_h * result.map_increments.delta_sum_ppT
            self.map_stats = update_map_stats(
                self.map_stats,
                delta_S_dir,
                delta_N_dir,
                delta_N_pos,
                delta_sum_p,
                delta_sum_ppT,
            )

            # Store certificate
            if results:
                iw_process_cert = CertBundle.create_approx(
                    chart_id=combined_belief.chart_id,
                    anchor_id=combined_belief.anchor_id,
                    triggers=["ProcessNoiseIWUpdate"],
                    influence=InfluenceCert(
                        lift_strength=0.0,
                        psd_projection_delta=float(proc_iw_cert_vec[0]),
                        nu_projection_delta=float(proc_iw_cert_vec[1]),
                        mass_epsilon_ratio=0.0,
                        anchor_drift_rho=0.0,
                        dt_scale=1.0,
                        extrinsic_scale=1.0,
                        trust_alpha=1.0,
                    ),
                )
                iw_meas_cert = CertBundle.create_approx(
                    chart_id=combined_belief.chart_id,
                    anchor_id=combined_belief.anchor_id,
                    triggers=["MeasurementNoiseIWUpdate"],
                    influence=InfluenceCert(
                        lift_strength=0.0,
                        psd_projection_delta=float(meas_iw_cert_vec[0]),
                        nu_projection_delta=float(meas_iw_cert_vec[1]),
                        mass_epsilon_ratio=0.0,
                        anchor_drift_rho=0.0,
                        dt_scale=1.0,
                        extrinsic_scale=1.0,
                        trust_alpha=1.0,
                    ),
                )
                self.cert_history.append(aggregate_certificates([results[0].aggregated_cert, iw_process_cert, iw_meas_cert]))
                if len(self.cert_history) > 100:
                    self.cert_history.pop(0)

            # Collect diagnostics for dashboard (from first hypothesis)
            if results and results[0].diagnostics is not None:
                diag = results[0].diagnostics
                # Update scan number and add noise trace info
                diag.scan_number = self.scan_count
                diag.trace_Q_mode = float(jnp.trace(self.Q))
                diag.trace_Sigma_lidar_mode = float(jnp.trace(self.config.Sigma_meas))
                diag.trace_Sigma_g_mode = float(jnp.trace(self.config.Sigma_g))
                diag.trace_Sigma_a_mode = float(jnp.trace(self.config.Sigma_a))
                self.diagnostics_log.append(diag)

            # Extract pose from belief and publish
            pose_6d = combined_belief.mean_world_pose()
            self._publish_state_from_pose(pose_6d, stamp_sec)
            
            if self.scan_count <= 10 or self.scan_count % 50 == 0:
                self.get_logger().info(
                    f"Scan {self.scan_count}: {n_points} pts, pipeline #{self.pipeline_runs}, "
                    f"dt={dt_sec:.3f}s"
                )
        
        except Exception as e:
            # Log error and fail fast (no silent fallbacks)
            self.get_logger().error(f"Pipeline error on scan {self.scan_count}: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise

    def _publish_state_from_pose(self, pose_6d: jnp.ndarray, stamp_sec: float):
        """Publish state from a 6D pose [trans, rotvec]."""
        rotvec, trans = se3_to_rotvec_trans(pose_6d)
        R = Rotation.from_rotvec(np.array(rotvec))
        quat = R.as_quat()
        
        # Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        
        odom_msg.pose.pose.position.x = float(trans[0])
        odom_msg.pose.pose.position.y = float(trans[1])
        odom_msg.pose.pose.position.z = float(trans[2])
        odom_msg.pose.pose.orientation.x = float(quat[0])
        odom_msg.pose.pose.orientation.y = float(quat[1])
        odom_msg.pose.pose.orientation.z = float(quat[2])
        odom_msg.pose.pose.orientation.w = float(quat[3])
        
        self.pub_state.publish(odom_msg)
        
        # TF
        tf_msg = TransformStamped()
        tf_msg.header = odom_msg.header
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = float(trans[0])
        tf_msg.transform.translation.y = float(trans[1])
        tf_msg.transform.translation.z = float(trans[2])
        tf_msg.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)
        
        # Trajectory
        pose_stamped = PoseStamped()
        pose_stamped.header = odom_msg.header
        pose_stamped.pose = odom_msg.pose.pose
        self.trajectory_poses.append(pose_stamped)
        
        if len(self.trajectory_poses) > self.max_path_length:
            self.trajectory_poses.pop(0)
        
        path_msg = Path()
        path_msg.header = odom_msg.header
        path_msg.poses = self.trajectory_poses
        self.pub_path.publish(path_msg)
        
        # TUM export
        if self.trajectory_file:
            self.trajectory_file.write(
                f"{stamp_sec:.9f} {trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )
            self.trajectory_file.flush()

    def _publish_status(self):
        """Publish periodic status."""
        elapsed = time.time() - self.node_start_time
        
        status = {
            "elapsed_sec": elapsed,
            "odom_count": self.odom_count,
            "scan_count": self.scan_count,
            "imu_count": self.imu_count,
            "pipeline_runs": self.pipeline_runs,
            "hypotheses": self.config.K_HYP,
            "map_bins_active": int(jnp.sum(self.map_stats.N_dir > 0)),
        }
        
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)
        
        self.get_logger().info(
            f"GC Status: odom={self.odom_count}, scans={self.scan_count}, "
            f"imu={self.imu_count}, pipeline={self.pipeline_runs}, "
            f"map_bins={status['map_bins_active']}/{self.config.B_BINS}"
        )

    def destroy_node(self):
        """Clean up."""
        if self.trajectory_file:
            self.trajectory_file.flush()
            self.trajectory_file.close()
            self.get_logger().info(f"Trajectory saved: {self.trajectory_export_path}")

        # Save diagnostics log for dashboard
        diagnostics_path = str(self.get_parameter("diagnostics_export_path").value)
        if diagnostics_path and self.diagnostics_log.total_scans > 0:
            self.diagnostics_log.end_time = time.time()
            try:
                self.diagnostics_log.save_npz(diagnostics_path)
                self.get_logger().info(
                    f"Diagnostics saved: {diagnostics_path} ({self.diagnostics_log.total_scans} scans)"
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to save diagnostics: {e}")

        super().destroy_node()


def main():
    rclpy.init()
    node = GoldenChildBackend()
    node.get_logger().info("Backend node created, entering spin loop...")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Shutting down. Final counts: scans={node.scan_count}, odom={node.odom_count}, imu={node.imu_count}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
