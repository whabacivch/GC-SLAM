"""
Geometric Compositional SLAM v2 Backend Node.

Actually uses the GC operators to process LiDAR scans.
This is NOT passthrough - it runs the full 14-step pipeline.

Reference: docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md
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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, Imu, PointField, Image
from std_msgs.msg import String
from fl_slam_poc.msg import RGBDImage

from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    se3_identity,
    se3_from_rotvec_trans,
    se3_to_rotvec_trans,
    se3_compose,
    se3_inverse,
)
from fl_slam_poc.common.geometry import se3_jax
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
from fl_slam_poc.backend.diagnostics import DiagnosticsLog

# Stage 1: PrimitiveMap imports
from fl_slam_poc.backend.structures.primitive_map import (
    PrimitiveMap,
    create_empty_primitive_map,
)
from fl_slam_poc.backend.map_publisher import PrimitiveMapPublisher
from fl_slam_poc.backend.rerun_visualizer import RerunVisualizer
from fl_slam_poc.backend.camera_batch_utils import feature_list_to_camera_batch
from fl_slam_poc.frontend.sensors.visual_feature_extractor import (
    VisualFeatureExtractor,
    VisualFeatureExtractorConfig,
    PinholeIntrinsics,
)
from fl_slam_poc.frontend.sensors.splat_prep import splat_prep_fused
from fl_slam_poc.frontend.sensors.lidar_camera_depth_fusion import (
    LidarCameraDepthFusionConfig,
)

from scipy.spatial.transform import Rotation


def _parse_T_base_sensor_6d(xyz_rxyz) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse 6D extrinsic parameter [x, y, z, rx, ry, rz] (rotvec radians) into (R, t) where:
      p_base = R @ p_sensor + t
    """
    v = np.array(xyz_rxyz, dtype=np.float64).reshape(-1)
    if v.shape[0] != 6:
        raise ValueError(f"Expected 6D extrinsic [x,y,z,rx,ry,rz], got shape {v.shape}")
    t = v[:3]
    rotvec = v[3:6]
    Rm = Rotation.from_rotvec(rotvec).as_matrix()
    return Rm, t


def _load_extrinsics_6d_from_file(path: str, key: str) -> list:
    """
    Load 6D [x,y,z,rx,ry,rz] from a YAML file. Path must exist (fail-fast).
    File may be: a list [x,y,z,rx,ry,rz], or a dict with the given key mapping to that list.
    """
    import os
    import yaml
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Extrinsics file missing when extrinsics_source=file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Extrinsics file empty or invalid: {path}")
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict) and key in data:
        raw = data[key]
    else:
        raise ValueError(f"Extrinsics file must be a 6D list or dict with key {key!r}: {path}")
    v = list(raw)
    if len(v) != 6:
        raise ValueError(f"Expected 6D [x,y,z,rx,ry,rz], got length {len(v)} in {path}")
    return v


def _smooth_window_weight(dist: float, min_r: float, max_r: float, sigma: float) -> float:
    """Continuous range weighting without hard gates."""
    # Smooth window: sigmoid(dist-min_r) * sigmoid(max_r-dist)
    a = (dist - min_r) / sigma
    b = (max_r - dist) / sigma
    w_min = 1.0 / (1.0 + np.exp(-a))
    w_max = 1.0 / (1.0 + np.exp(-b))
    return float(w_min * w_max)


def _polar_so3(M: np.ndarray) -> np.ndarray:
    """Project 3x3 matrix to SO(3) via polar decomposition (one SVD). R_bar = polar(M)."""
    U, _S, Vh = np.linalg.svd(M)
    R = U @ Vh
    if np.linalg.det(R) < 0:
        U_c = U.copy()
        U_c[:, -1] *= -1
        R = U_c @ Vh
    return R


def _imu_stability_weights(
    stamps: List[float],
    imu_buffer: List[Tuple[float, np.ndarray, np.ndarray]],
    c_gyro: float,
    c_accel: float,
    g: float,
) -> List[float]:
    """
    Per-timestamp stability weight from IMU: w_k ∝ exp(-c_gyro ‖ω_k‖²) · exp(-c_accel (‖a_k‖ - g)²).
    No gates; smooth downweighting when robot is ringing at bag start.
    imu_buffer entries are (stamp_sec, gyro_3, accel_3).
    """
    if not imu_buffer:
        return [1.0] * len(stamps)
    ts = np.array([t for t, _, _ in imu_buffer], dtype=np.float64)
    weights = []
    for s in stamps:
        i = np.argmin(np.abs(ts - s))
        _t, gyro, accel = imu_buffer[i]
        gyro = np.asarray(gyro, dtype=np.float64)
        accel = np.asarray(accel, dtype=np.float64)
        w_gyro = np.exp(-c_gyro * float(np.dot(gyro, gyro)))
        a_norm = float(np.linalg.norm(accel))
        w_accel = np.exp(-c_accel * (a_norm - g) ** 2)
        weights.append(w_gyro * w_accel)
    return weights


def _yaw_deg_from_pose_6d(pose_6d) -> float:
    """Extract yaw (degrees) from 6D pose [trans, rotvec]. Uses atan2(R[1,0], R[0,0])."""
    pose_6d = np.asarray(pose_6d, dtype=np.float64).ravel()[:6]
    R = np.array(se3_jax.so3_exp(pose_6d[3:6]), dtype=np.float64)
    yaw_rad = np.arctan2(R[1, 0], R[0, 0])
    return float(np.degrees(yaw_rad))


def _belief_xyyaw_vel(belief: BeliefGaussianInfo, eps_lift: float = 1e-9) -> Tuple[float, float, float, float, float]:
    """Extract (x, y, yaw_deg, vx_world, vy_world) from belief for diagnostic."""
    pose_6d = np.array(belief.mean_world_pose(eps_lift=eps_lift), dtype=np.float64)
    x, y = float(pose_6d[0]), float(pose_6d[1])
    yaw_deg = _yaw_deg_from_pose_6d(pose_6d)
    delta_z = np.array(belief.mean_increment(eps_lift=eps_lift), dtype=np.float64)
    z_lin = np.array(belief.z_lin, dtype=np.float64)
    vel = z_lin + delta_z
    vx = float(vel[6])
    vy = float(vel[7])
    return (x, y, yaw_deg, vx, vy)


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


def parse_pointcloud2_vlp16(
    msg: PointCloud2,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Parse VLP-16 (Velodyne Puck) PointCloud2: x, y, z, ring; optional t/time.
    Outputs (points, timestamps, weights, ring, tag) for pipeline.
    tag=0; timebase from header.stamp; time_offset from per-point t if present else 0.
    See docs/POINTCLOUD2_LAYOUTS.md.
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
    required_vlp16 = ["x", "y", "z", "ring"]
    missing = [k for k in required_vlp16 if k not in field_map]
    if missing:
        raise RuntimeError(
            f"PointCloud2 (VLP-16 layout) missing required fields: {missing}. "
            f"Present fields: {sorted(list(field_map.keys()))}"
        )

    needed = ["x", "y", "z", "ring"]
    if "intensity" in field_map:
        needed.append("intensity")
    time_field = None
    if "t" in field_map:
        time_field = "t"
    elif "time" in field_map:
        time_field = "time"
    if time_field and time_field not in needed:
        needed.append(time_field)

    names = []
    formats = []
    offsets = []
    for name in needed:
        off, dt = field_map[name]
        names.append(name)
        formats.append(_pointfield_to_dtype(dt))
        offsets.append(off)

    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dtype, count=n_points)

    sentinel = float(constants.GC_NONFINITE_SENTINEL)
    x = np.nan_to_num(np.asarray(arr["x"], dtype=np.float64), nan=sentinel, posinf=sentinel, neginf=-sentinel)
    y = np.nan_to_num(np.asarray(arr["y"], dtype=np.float64), nan=sentinel, posinf=sentinel, neginf=-sentinel)
    z = np.nan_to_num(np.asarray(arr["z"], dtype=np.float64), nan=sentinel, posinf=sentinel, neginf=-sentinel)
    ring = np.asarray(arr["ring"], dtype=np.uint8)

    # Per-point time: t/time in seconds or ns (driver-dependent). If present, use it; else header.stamp for whole scan.
    header_stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if time_field is not None:
        t_raw = np.asarray(arr[time_field], dtype=np.float64)
        # If values are large (e.g. ns), convert to seconds
        if np.any(t_raw > 1e6):
            t = t_raw * 1e-9
        else:
            t = t_raw
    else:
        t = np.full((n_points,), header_stamp_sec, dtype=np.float64)

    tag = np.zeros((n_points,), dtype=np.uint8)

    # Range-based weighting
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


class GeometricCompositionalBackend(Node):
    """
    Geometric Compositional SLAM v2 Backend.
    
    Actually runs the 14-step pipeline on each LiDAR scan.
    """

    def __init__(self):
        super().__init__("gc_backend")
        
        self._declare_parameters()
        self._init_state()
        self._init_ros()
        self._publish_runtime_manifest()

        self.get_logger().info("Geometric Compositional SLAM v2 Backend initialized - PIPELINE ENABLED")

    def _declare_parameters(self):
        """Declare ROS parameters."""
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        # Backend subscribes ONLY to /gc/sensors/* (canonical topics from sensor hub)
        self.declare_parameter("lidar_topic", "/gc/sensors/lidar_points")
        self.declare_parameter("odom_topic", "/gc/sensors/odom")
        self.declare_parameter("imu_topic", "/gc/sensors/imu")
        self.declare_parameter("trajectory_export_path", "/tmp/gc_slam_trajectory.tum")
        self.declare_parameter("diagnostics_export_path", "results/gc_slam_diagnostics.npz")
        self.declare_parameter("splat_export_path", "")
        self.declare_parameter("status_check_period_sec", 5.0)
        self.declare_parameter("forgetting_factor", 0.99)
        # No-TF extrinsics (T_{base<-sensor}) in [x, y, z, rx, ry, rz] rotvec (radians).
        # extrinsics_source: inline | file. When file: load from T_base_lidar_file, T_base_imu_file (fail if missing).
        self.declare_parameter("extrinsics_source", "inline")
        self.declare_parameter("T_base_lidar_file", "")
        self.declare_parameter("T_base_imu_file", "")
        self.declare_parameter("T_base_lidar", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("T_base_imu", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # LiDAR measurement noise prior (m² isotropic). Kimera VLP-16: 1e-3.
        self.declare_parameter("lidar_sigma_meas", 0.01)
        # When true, derive Sigma_g/Sigma_a from first N IMU messages (units/fallback doc'd). When false, use priors only.
        self.declare_parameter("use_imu_message_covariance", False)
        # Hard single-path enforcement: if enabled, missing topics are hard errors.
        self.declare_parameter("use_imu", True)
        self.declare_parameter("use_odom", True)
        # IMU gravity scaling (1.0 = nominal; 0.0 disables gravity contribution)
        self.declare_parameter("imu_gravity_scale", 1.0)
        self.declare_parameter("imu_accel_scale", 1.0)  # 1.0 when bag publishes m/s² (Kimera/ROS)
        # Deskew rotation-only mode: removes hidden IMU translation leak through deskew
        self.declare_parameter("deskew_rotation_only", False)
        # Timing/profiling
        self.declare_parameter("enable_timing", False)
        # Smoothed initial reference: buffer first K odom, then set first_odom_pose = aggregate (PIPELINE_DESIGN_GAPS §5.4.1)
        self.declare_parameter("init_window_odom_count", 10)
        # PointCloud2 layout: vlp16 (Kimera/VLP-16). See docs/POINTCLOUD2_LAYOUTS.md.
        self.declare_parameter("pointcloud_layout", "vlp16")
        # Odom vs belief diagnostic: when non-empty, write CSV (raw odom, belief start, belief end) per scan.
        self.declare_parameter("odom_belief_diagnostic_file", "")
        self.declare_parameter("odom_belief_diagnostic_max_scans", 0)  # 0 = all scans
        # Camera (required): single RGBD topic from camera_rgbd_node
        self.declare_parameter("camera_rgbd_topic", "/gc/sensors/camera_rgbd")
        self.declare_parameter("camera_K", [500.0, 500.0, 320.0, 240.0])  # [fx, fy, cx, cy]
        self.declare_parameter("T_base_camera", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6D rotvec
        self.declare_parameter("ringbuf_len", constants.GC_RINGBUF_LEN)
        self.declare_parameter("use_rerun", True)
        self.declare_parameter("rerun_recording_path", "")
        self.declare_parameter("rerun_spawn", False)

    def _init_state(self):
        """Initialize Geometric Compositional state."""
        # Pipeline configuration
        self.config = PipelineConfig()
        
        # Adaptive process noise IW state (datasheet priors) + derived Q
        self.process_noise_state: ProcessNoiseIWState = create_datasheet_process_noise_state()
        self.Q = process_noise_state_to_Q_jax(self.process_noise_state)
        
        # Adaptive measurement noise IW state (per-sensor, phase 1) + derived Sigma_meas (LiDAR)
        lidar_sigma_meas = float(self.get_parameter("lidar_sigma_meas").value)
        self.measurement_noise_state: MeasurementNoiseIWState = create_datasheet_measurement_noise_state(
            lidar_sigma_meas=lidar_sigma_meas
        )
        self.config.Sigma_meas = measurement_noise_mean_jax(self.measurement_noise_state, idx=2)

        self.forgetting_factor = float(self.get_parameter("forgetting_factor").value)

        # PrimitiveMap is the canonical map (single-path primitives)
        self.primitive_map = create_empty_primitive_map(
            max_size=self.config.primitive_map_max_size,
        )
        self.get_logger().info(
            f"PrimitiveMap initialized: max_size={self.config.primitive_map_max_size}, "
            f"n_feat={self.config.n_feat}, n_surfel={self.config.n_surfel}"
        )

        # Parse and cache no-TF extrinsics (inline or from file; fail-fast if file missing when source=file).
        extrinsics_source = str(self.get_parameter("extrinsics_source").value).strip().lower()
        if extrinsics_source == "file":
            lidar_file = str(self.get_parameter("T_base_lidar_file").value).strip()
            imu_file = str(self.get_parameter("T_base_imu_file").value).strip()
            if not lidar_file or not imu_file:
                raise ValueError(
                    "extrinsics_source=file requires T_base_lidar_file and T_base_imu_file to be set"
                )
            T_base_lidar_list = _load_extrinsics_6d_from_file(lidar_file, "T_base_lidar")
            T_base_imu_list = _load_extrinsics_6d_from_file(imu_file, "T_base_imu")
            self.R_base_lidar, self.t_base_lidar = _parse_T_base_sensor_6d(T_base_lidar_list)
            self.R_base_imu, self.t_base_imu = _parse_T_base_sensor_6d(T_base_imu_list)
        else:
            if extrinsics_source != "inline":
                raise ValueError(
                    f"extrinsics_source must be 'inline' or 'file'; got {extrinsics_source!r}"
                )
            self.R_base_lidar, self.t_base_lidar = _parse_T_base_sensor_6d(
                self.get_parameter("T_base_lidar").value
            )
            self.R_base_imu, self.t_base_imu = _parse_T_base_sensor_6d(
                self.get_parameter("T_base_imu").value
            )

        # Log extrinsics for runtime audit (verifies config is applied correctly)
        from scipy.spatial.transform import Rotation as R_scipy
        lidar_rotvec = R_scipy.from_matrix(self.R_base_lidar).as_rotvec()
        imu_rotvec = R_scipy.from_matrix(self.R_base_imu).as_rotvec()
        self.get_logger().info("=" * 60)
        self.get_logger().info("EXTRINSICS (T_{base<-sensor})")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"T_base_lidar: t=({self.t_base_lidar[0]:.6f}, {self.t_base_lidar[1]:.6f}, {self.t_base_lidar[2]:.6f}) rotvec=({lidar_rotvec[0]:.6f}, {lidar_rotvec[1]:.6f}, {lidar_rotvec[2]:.6f})")
        self.get_logger().info(f"T_base_imu:   t=({self.t_base_imu[0]:.6f}, {self.t_base_imu[1]:.6f}, {self.t_base_imu[2]:.6f}) rotvec=({imu_rotvec[0]:.6f}, {imu_rotvec[1]:.6f}, {imu_rotvec[2]:.6f})")
        imu_angle_deg = np.linalg.norm(imu_rotvec) * 180.0 / np.pi
        self.get_logger().info(f"IMU rotation angle: {imu_angle_deg:.3f}° (should be ~28° if gravity-aligned)")
        self.get_logger().info("=" * 60)

        # Camera (required): intrinsics and T_base_camera; fail-fast if missing
        camera_K = list(self.get_parameter("camera_K").value)
        if len(camera_K) != 4:
            raise ValueError("camera_K must be [fx, fy, cx, cy]; got length %d" % len(camera_K))
        self.camera_K = camera_K  # [fx, fy, cx, cy]
        self.R_base_camera, self.t_base_camera = _parse_T_base_sensor_6d(
            list(self.get_parameter("T_base_camera").value)
        )
        ringbuf_len = int(self.get_parameter("ringbuf_len").value)
        if ringbuf_len < 1:
            raise ValueError("ringbuf_len must be >= 1; got %d" % ringbuf_len)
        self.camera_ringbuf: List[Tuple[float, np.ndarray, np.ndarray]] = []  # (stamp_sec, rgb, depth)
        self.camera_ringbuf_max = ringbuf_len
        self.get_logger().info(
            "Camera: K=[fx=%s fy=%s cx=%s cy=%s], T_base_camera cached, ringbuf_len=%d"
            % (camera_K[0], camera_K[1], camera_K[2], camera_K[3], ringbuf_len)
        )
        # VisualFeatureExtractor and splat_prep config (camera -> MeasurementBatch)
        fx, fy, cx, cy = float(camera_K[0]), float(camera_K[1]), float(camera_K[2]), float(camera_K[3])
        self.camera_intrinsics = PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
        self.visual_extractor = VisualFeatureExtractor(
            self.camera_intrinsics,
            config=VisualFeatureExtractorConfig(max_features=self.config.n_feat),
        )
        self.depth_fusion_config = LidarCameraDepthFusionConfig()

        # Wire LiDAR origin (in base frame) into the pipeline so direction features are computed
        # from the sensor origin, not the base origin.
        self.config.lidar_origin_base = jnp.array(self.t_base_lidar, dtype=jnp.float64)
        self.config.imu_gravity_scale = float(self.get_parameter("imu_gravity_scale").value)
        self.get_logger().info(f"IMU gravity scale: {self.config.imu_gravity_scale:.6f}")
        self.config.deskew_rotation_only = bool(self.get_parameter("deskew_rotation_only").value)
        _et = self.get_parameter("enable_timing").value
        self.config.enable_timing = _et if isinstance(_et, bool) else (str(_et).lower() == "true")
        self.get_logger().info(f"Timing diagnostics: {'enabled' if self.config.enable_timing else 'disabled'}")
        self.get_logger().info(f"Deskew rotation-only: {self.config.deskew_rotation_only}")
        self.init_window_odom_count = int(self.get_parameter("init_window_odom_count").value)
        self.get_logger().info(f"Init window: first_odom_pose = aggregate of first {self.init_window_odom_count} odom")
        self.pointcloud_layout = str(self.get_parameter("pointcloud_layout").value).strip().lower()
        if self.pointcloud_layout not in ("vlp16",):
            raise ValueError(
                f"pointcloud_layout must be 'vlp16'; got {self.pointcloud_layout!r}. "
                "See docs/POINTCLOUD2_LAYOUTS.md."
            )
        self.get_logger().info(f"PointCloud2 layout: {self.pointcloud_layout}")

        # Odom vs belief diagnostic (raw vs estimate)
        self._odom_belief_diagnostic_file = str(self.get_parameter("odom_belief_diagnostic_file").value).strip()
        self._odom_belief_diagnostic_max = int(self.get_parameter("odom_belief_diagnostic_max_scans").value)
        if self._odom_belief_diagnostic_file:
            self.get_logger().info(
                f"Odom/belief diagnostic: writing to {self._odom_belief_diagnostic_file} "
                f"(max_scans={self._odom_belief_diagnostic_max or 'all'})"
            )
            self._odom_belief_diagnostic_header_written = False
        else:
            self._odom_belief_diagnostic_header_written = True  # no-op

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

        # Odometry twist (velocity) for kinematic coupling
        # Initialize to zeros with huge covariance (negligible precision).
        # By construction, this means "no information" without needing a gate.
        self.last_odom_twist = jnp.zeros(6, dtype=jnp.float64)  # (6,) [vx, vy, vz, wx, wy, wz]
        self.last_odom_twist_cov = 1e12 * jnp.eye(6, dtype=jnp.float64)  # (6,6) huge covariance

        # Explicit anchor A (anchor-to-world). Belief lives in anchor frame; export uses A.
        # Provisional A0 = first odom sample (no scan drops); after K samples, A_smoothed from polar + weighted mean.
        self.first_odom_pose = None  # (6,) SE3 = A0 then unchanged; odom_relative = first_odom^{-1} ∘ odom_absolute
        self.odom_init_buffer: List[Tuple[float, jnp.ndarray]] = []  # (stamp, pose_6d) for first K
        self.anchor_correction = se3_identity()  # (6,) SE3: pose_export = anchor_correction ∘ pose_belief (identity until A_smoothed set)
        self._anchor_smoothed_done = False  # True after first K odom → A_smoothed and anchor_correction set

        # IMU buffer for high-rate prediction
        # CRITICAL: Must be large enough to cover scan-to-scan intervals.
        # At 200Hz IMU and up to 20s scan intervals, need ~4000 samples.
        # Previous value of 200 only covered 1s, causing dt_int ≈ 0 for most scans!
        self.imu_buffer: List[Tuple[float, jnp.ndarray, jnp.ndarray]] = []
        self.max_imu_buffer = 4000  # Covers ~20s at 200Hz IMU rate
        
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

        # Deferred publish: drain at start of next callback so pipeline hot path doesn't block on ROS
        self._pending_publish: Optional[Tuple[jnp.ndarray, float]] = None

    def _init_ros(self):
        """Initialize ROS interfaces."""
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,  # Increased from 10 for better IMU burst handling
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
        
        # Separate callback groups so IMU/odom can be buffered while lidar pipeline runs.
        # Without this, single-threaded or MutuallyExclusive groups would block IMU during
        # the ~1-2s pipeline processing, causing 200Hz IMU messages to overflow the queue.
        self.cb_group_lidar = MutuallyExclusiveCallbackGroup()
        self.cb_group_sensors = ReentrantCallbackGroup()  # IMU + odom can run concurrently
        
        self.sub_lidar = self.create_subscription(
            PointCloud2, lidar_topic, self.on_lidar, qos_sensor,
            callback_group=self.cb_group_lidar
        )
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.on_odom, qos_reliable,
            callback_group=self.cb_group_sensors
        )
        self.sub_imu = self.create_subscription(
            Imu, imu_topic, self.on_imu, qos_sensor,
            callback_group=self.cb_group_sensors
        )
        
        self.get_logger().info(f"LiDAR: {lidar_topic} (PIPELINE ACTIVE)")
        self.get_logger().info(f"Odom: {odom_topic}")
        self.get_logger().info(f"IMU: {imu_topic}")

        # Camera (required): single RGBD subscription with ring buffer
        camera_rgbd_topic = str(self.get_parameter("camera_rgbd_topic").value).strip()
        if not camera_rgbd_topic:
            raise ValueError("camera_rgbd_topic is required but empty")
        self.sub_camera_rgbd = self.create_subscription(
            RGBDImage, camera_rgbd_topic, self._on_camera_rgbd, qos_sensor,
            callback_group=self.cb_group_sensors,
        )
        self.get_logger().info(f"Camera RGBD: {camera_rgbd_topic} (ring buffer len={self.camera_ringbuf_max})")
        
        # Publishers
        self.pub_state = self.create_publisher(Odometry, "/gc/state", 10)
        self.pub_path = self.create_publisher(Path, "/gc/trajectory", 10)
        self.pub_manifest = self.create_publisher(String, "/gc/runtime_manifest", 10)
        self.pub_cert = self.create_publisher(String, "/gc/certificate", 10)
        self.pub_status = self.create_publisher(String, "/gc/status", 10)

        # Rerun visualization (Wayland-friendly; replaces RViz)
        self.rerun_visualizer: Optional[RerunVisualizer] = None
        use_rerun = bool(self.get_parameter("use_rerun").value)
        rerun_recording_path = str(self.get_parameter("rerun_recording_path").value).strip()
        rerun_spawn = bool(self.get_parameter("rerun_spawn").value)
        if use_rerun:
            self.rerun_visualizer = RerunVisualizer(
                application_id="fl_slam_poc",
                spawn=rerun_spawn,
                recording_path=rerun_recording_path or None,
            )
            if self.rerun_visualizer.init():
                self.get_logger().info(
                    "Rerun visualization: enabled (spawn=%s, recording=%s)"
                    % (rerun_spawn, rerun_recording_path or "none")
                )
            else:
                self.get_logger().warn("Rerun visualization: requested but rerun-sdk not available; install with: pip install rerun-sdk")
                self.rerun_visualizer = None
        else:
            self.rerun_visualizer = None

        # PrimitiveMap publisher: /gc/map/points (PointCloud2)
        self.map_publisher: Optional[PrimitiveMapPublisher] = None
        if self.primitive_map is not None:
            self.map_publisher = PrimitiveMapPublisher(
                self,
                frame_id=self.odom_frame,
                publish_ellipsoids=False,
                rerun_visualizer=self.rerun_visualizer,
            )
            self.get_logger().info("PrimitiveMap publisher: /gc/map/points (PointCloud2)")
        
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
        manifest = RuntimeManifest(
            imu_gravity_scale=float(self.config.imu_gravity_scale),
            deskew_rotation_only=bool(self.config.deskew_rotation_only),
            power_beta_min=float(self.config.power_beta_min),
            power_beta_exc_c=float(self.config.power_beta_exc_c),
            power_beta_z_c=float(self.config.power_beta_z_c),
            pose_evidence_backend=constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES,
            map_backend=constants.GC_MAP_BACKEND_PRIMITIVE_MAP,
            topics={
                "lidar": str(self.get_parameter("lidar_topic").value),
                "odom": str(self.get_parameter("odom_topic").value),
                "imu": str(self.get_parameter("imu_topic").value),
                "camera_rgbd": str(self.get_parameter("camera_rgbd_topic").value),
                "runtime_manifest": "/gc/runtime_manifest",
                "certificate": "/gc/certificate",
                "status": "/gc/status",
            },
        )
        manifest_dict = manifest.to_dict()
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("GEOMETRIC COMPOSITIONAL RUNTIME MANIFEST")
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
        # Apply acceleration scale: 1.0 when bag publishes m/s² (Kimera/ROS).
        accel_raw = np.array(
            [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            dtype=np.float64,
        )
        accel_scale = float(self.get_parameter("imu_accel_scale").value)
        accel = accel_raw * accel_scale

        # No-TF mode: rotate IMU measurements into the base/body frame.
        # This is a numeric transform (not just frame_id relabeling).
        gyro = self.R_base_imu @ gyro
        accel = self.R_base_imu @ accel

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
        R_parent_child = Rotation.from_quat(quat)
        rotvec = R_parent_child.as_rotvec()

        # Convert to SE3 pose [trans, rotvec]
        #
        # IMPORTANT (frame convention, by construction):
        # - msg.header.frame_id is the parent frame (e.g., acl_jackal2/odom)
        # - msg.child_frame_id is the child frame (e.g., acl_jackal2/base)
        # - ROS Odometry pose encodes T_{parent<-child} in the usual rigid-transform form:
        #       p_parent = R_parent_child * p_child + t_parent_child
        #   This matches se3_jax/se3_compose convention (t_out = t_a + R_a @ t_b).
        #
        # Therefore: DO NOT invert here. Downstream code treats odom_pose as a pose in the
        # parent/world frame of the body, consistent with LiDAR/Wahba and IMU operators.
        #
        # Planar robot: wheel odom often does not observe Z (cov ~1e6 for z/roll/pitch).
        # Use actual pos.z from the message; trust is capped via GC_ODOM_Z_VARIANCE_PRIOR below.
        # Anchor smoothing still uses GC_PLANAR_Z_REF as a reference height (line 807).
        odom_pose_absolute = se3_from_rotvec_trans(
            jnp.array(rotvec, dtype=jnp.float64),
            jnp.array([pos.x, pos.y, pos.z], dtype=jnp.float64),
        )
        
        # Explicit anchor: provisional A0 on first odom (no scan drops); after K samples, set anchor_correction from A_smoothed
        if self.first_odom_pose is None:
            self.first_odom_pose = odom_pose_absolute  # A0 = first sample (provisional anchor)
            self.odom_init_buffer.append((stamp_sec, odom_pose_absolute))
            self.get_logger().info(
                "Anchor A0 (provisional) set from first odom: "
                f"frame_id='{msg.header.frame_id}' child_frame_id='{msg.child_frame_id}'"
            )
        else:
            if not self._anchor_smoothed_done:
                self.odom_init_buffer.append((stamp_sec, odom_pose_absolute))
            if not self._anchor_smoothed_done and len(self.odom_init_buffer) >= self.init_window_odom_count:
                # Closed-form smoothed anchor: weighted t̄ + polar(M) for R̄ (PIPELINE_DESIGN_GAPS §5.4.1)
                A0 = self.first_odom_pose
                stamps = [s for s, _ in self.odom_init_buffer]
                poses = [p for _, p in self.odom_init_buffer]
                weights = _imu_stability_weights(
                    stamps, self.imu_buffer,
                    constants.GC_INIT_ANCHOR_GYRO_SCALE,
                    constants.GC_INIT_ANCHOR_ACCEL_SCALE,
                    constants.GRAVITY_MAG,
                )
                w_sum = sum(weights)
                if w_sum <= 0.0:
                    weights = None  # uniform weighting when IMU weights sum to zero (e.g. wrong accel units)
                trans_mean = np.average(
                    [np.array(p[:3], dtype=np.float64) for p in poses],
                    axis=0, weights=weights,
                )
                # Planar: anchor Z is a reference, not average of odom Z (odom Z is unobserved).
                trans_mean[2] = float(constants.GC_PLANAR_Z_REF)
                R_matrices = [Rotation.from_rotvec(np.array(p[3:6], dtype=np.float64)).as_matrix() for p in poses]
                M = np.average(R_matrices, axis=0, weights=weights)
                R_polar = _polar_so3(M)
                rotvec_mean = Rotation.from_matrix(R_polar).as_rotvec()
                A_smoothed = se3_from_rotvec_trans(
                    jnp.array(rotvec_mean, dtype=jnp.float64),
                    jnp.array(trans_mean, dtype=jnp.float64),
                )
                self.anchor_correction = se3_compose(se3_inverse(A_smoothed), A0)
                self.odom_init_buffer.clear()
                self._anchor_smoothed_done = True
                yaw_deg = float(Rotation.from_matrix(R_polar).as_euler("xyz", degrees=True)[2])
                self.get_logger().info(
                    f"Anchor smoothed: K={self.init_window_odom_count}, A_smoothed set; "
                    f"trans=({trans_mean[0]:.3f}, {trans_mean[1]:.3f}, {trans_mean[2]:.3f}) yaw={yaw_deg:+.2f}deg"
                )

        # odom_relative = first_odom^{-1} ∘ odom_absolute (belief stays in A0 frame)
        first_odom_inv = se3_inverse(self.first_odom_pose)
        self.last_odom_pose = se3_compose(first_odom_inv, odom_pose_absolute)
        self.last_odom_stamp = stamp_sec
        # Pose covariance is row-major 6x6: [x,y,z,roll,pitch,yaw]
        # Note: Covariance is unchanged by the relative transformation (same uncertainty)
        cov = np.array(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        # Cap z variance to limit trust in odom z (planar robots often have bad/unobserved z)
        cov[2, 2] = max(cov[2, 2], constants.GC_ODOM_Z_VARIANCE_PRIOR)
        self.last_odom_cov_se3 = jnp.array(cov, dtype=jnp.float64)

        # Read twist (velocity) from odometry message
        # ROS twist is in body frame: [linear.x/y/z, angular.x/y/z]
        twist = msg.twist.twist
        self.last_odom_twist = jnp.array([
            twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.x, twist.angular.y, twist.angular.z
        ], dtype=jnp.float64)  # (6,) [vx, vy, vz, wx, wy, wz] in body frame
        # Twist covariance is row-major 6x6
        twist_cov = np.array(msg.twist.covariance, dtype=np.float64).reshape(6, 6)
        self.last_odom_twist_cov = jnp.array(twist_cov, dtype=jnp.float64)

    def _on_camera_rgbd(self, msg: RGBDImage):
        """Store coherent (rgb, depth) pair in ring buffer from camera_rgbd_node."""
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            # Decode RGB (rgb8)
            rgb_msg = msg.rgb
            height = int(rgb_msg.height)
            width = int(rgb_msg.width)
            if height <= 0 or width <= 0:
                raise ValueError(f"invalid RGB size {height}x{width}")
            row_elems = max(1, int(rgb_msg.step) // 3)
            rgb_data = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(height, row_elems, 3)
            rgb = rgb_data[:, :width, :].copy()

            # Decode depth (32FC1, meters)
            depth_msg = msg.depth
            dh, dw = int(depth_msg.height), int(depth_msg.width)
            if dh != height or dw != width:
                raise ValueError(f"RGB/depth size mismatch: rgb {height}x{width}, depth {dh}x{dw}")
            if depth_msg.encoding != "32FC1":
                raise ValueError(f"depth encoding must be 32FC1; got {depth_msg.encoding!r}")
            depth_row_elems = max(1, int(depth_msg.step) // 4)
            depth_data = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(dh, depth_row_elems)
            depth = depth_data[:, :dw].astype(np.float64)

            # Push to ring buffer (stamp, rgb, depth)
            self.camera_ringbuf.append((stamp_sec, rgb, depth))
            if len(self.camera_ringbuf) > self.camera_ringbuf_max:
                self.camera_ringbuf.pop(0)
        except Exception as e:
            self.get_logger().warn("Camera RGBD buffer failed: %s" % e)

    def on_lidar(self, msg: PointCloud2):
        """
        Process LiDAR scan through the full GC pipeline.

        This is where the actual SLAM happens!
        """
        self.scan_count += 1
        self.get_logger().info(f"on_lidar callback #{self.scan_count} received")

        # Do not skip scans when odom/IMU are missing. Use LiDAR when we have it.
        # When odom is missing we pass identity pose + large covariance (negligible odom evidence).
        # When IMU buffer is empty we pass zero arrays (deskew no-op, IMU evidence near zero).

        # =====================================================================
        # TIMESTAMP AUDIT: Log header stamp and per-point time offsets
        # =====================================================================
        header_stamp_sec = msg.header.stamp.sec
        header_stamp_nsec = msg.header.stamp.nanosec
        stamp_sec = header_stamp_sec + header_stamp_nsec * 1e-9
        
        # Parse point cloud (vlp16 layout; Kimera/VLP-16 PointCloud2).
        points, timestamps, weights, ring, tag = parse_pointcloud2_vlp16(msg)

        # No-TF mode: transform LiDAR points into the base/body frame before any inference.
        # p_base = R_base_lidar @ p_lidar + t_base_lidar
        if points.shape[0] > 0:
            pts_np = np.array(points)
            pts_base = (self.R_base_lidar @ pts_np.T).T + self.t_base_lidar[None, :]
            points = jnp.array(pts_base, dtype=jnp.float64)
        n_points = points.shape[0]

        if n_points == 0:
            # Keep pipeline total by supplying a zero-weight dummy point.
            points = jnp.zeros((1, 3), dtype=jnp.float64)
            timestamps = jnp.zeros(1, dtype=jnp.float64)
            weights = jnp.zeros(1, dtype=jnp.float64)
            n_points = 1
            per_point_offset_min = 0.0
            per_point_offset_max = 0.0
            per_point_offset_units = "ns (synthetic, empty scan)"
        else:
            # Extract per-point time_offset values for audit logging
            # Note: time_offset is stored as uint32 nanoseconds in the PointCloud2 message
            # The parse function converts it to seconds, so we need to get raw offsets
            field_map = {f.name: (f.offset, f.datatype) for f in msg.fields}
            if "time_offset" in field_map:
                off, dt = field_map["time_offset"]
                # Read raw uint32 time_offset values (vectorized for efficiency)
                # time_offset is at offset 'off' within each point, so we need to stride by point_step
                # Create a view that reads uint32 from each point's time_offset field
                offsets_raw = np.array([
                    struct.unpack_from('<I', msg.data, i * msg.point_step + off)[0]
                    for i in range(n_points)
                ], dtype=np.uint32)
                per_point_offset_min = float(np.min(offsets_raw)) if len(offsets_raw) > 0 else 0.0
                per_point_offset_max = float(np.max(offsets_raw)) if len(offsets_raw) > 0 else 0.0
                per_point_offset_units = "ns (uint32, relative to timebase)"
            else:
                per_point_offset_min = 0.0
                per_point_offset_max = 0.0
                per_point_offset_units = "ns (not available)"

        # =====================================================================
        # SCAN TIME RULE: t_scan comes from PointCloud2.header.stamp
        # Per-point offsets are for deskew INSIDE the scan, not scan-to-scan dt
        # =====================================================================
        t_scan = stamp_sec  # CORRECT: scan time is header.stamp, not per-point timestamps

        if self.rerun_visualizer is not None and points.shape[0] > 0:
            self.rerun_visualizer.log_lidar(np.array(points, dtype=np.float64), t_scan)

        # Preserve previous scan time for scan-to-scan interval handling.
        # This must be captured BEFORE updating self.last_scan_stamp.
        t_prev_scan = float(self.last_scan_stamp) if self.last_scan_stamp > 0.0 else 0.0
        
        # Derive scan bounds for deskew (within-scan only).
        # Scan time bounds (PointCloud2 vlp16: header.stamp or per-point t):
        # - timebase_sec = start of accumulation window
        # - header.stamp = end of accumulation window (when message published)
        # - All time_offset = 0 (no per-point timestamps available)
        # So we MUST use timebase_sec as scan_start_time and header.stamp as scan_end_time
        # to capture the accumulation window, even though we can't deskew individual points.
        stamp_j = jnp.array(stamp_sec, dtype=jnp.float64)
        timestamps_min = float(jnp.min(timestamps))
        timestamps_max = float(jnp.max(timestamps))
        
        # If all timestamps are the same (rosette pattern with time_offset=0),
        # use timebase_sec (from timestamps) as scan_start and header.stamp as scan_end.
        if jnp.abs(timestamps_max - timestamps_min) < 1e-9:  # All timestamps identical
            # Rosette pattern: use accumulation window
            scan_start_time = timestamps_min  # = timebase_sec
            scan_end_time = float(stamp_j)    # = header.stamp
        else:
            # Per-point timestamps available: use actual min/max
            scan_start_time = float(jnp.minimum(stamp_j, timestamps_min))
            scan_end_time = float(jnp.maximum(stamp_j, timestamps_max))

        # Compute dt since last scan (using t_scan, not scan_end_time)
        dt_raw = (
            t_scan - t_prev_scan
            if t_prev_scan > 0.0
            else (scan_end_time - scan_start_time)
        )
        eps_dt = np.finfo(np.float64).eps
        dt_sec = float(jnp.sqrt(jnp.array(dt_raw, dtype=jnp.float64) ** 2 + eps_dt))
        self.last_scan_stamp = t_scan  # CORRECT: update using t_scan (header.stamp)

        # =====================================================================
        # TIMESTAMP AUDIT LOGGING (before IMU processing)
        # =====================================================================
        # Scan-to-scan IMU integration interval is (t_prev_scan, t_scan).
        # For the first scan, the interval is empty by definition.
        t_last_scan = t_prev_scan if t_prev_scan > 0.0 else t_scan
        self.get_logger().info(
            f"Scan #{self.scan_count} timestamp audit: "
            f"header.stamp=({header_stamp_sec}.{header_stamp_nsec:09d}) "
            f"per_point_offset=[{per_point_offset_min:.0f}, {per_point_offset_max:.0f}] {per_point_offset_units} "
            f"t_scan={t_scan:.9f} "
            f"scan_bounds=[{scan_start_time:.9f}, {scan_end_time:.9f}] "
            f"dt_scan_to_scan={dt_sec:.6f} "
            f"IMU_interval=({t_last_scan:.9f}, {t_scan:.9f})"
        )

        # Drain deferred publish from previous scan (keeps pipeline hot path free of ROS publish)
        if self._pending_publish is not None:
            pose_6d, stamp = self._pending_publish
            self._publish_state_from_pose(pose_6d, stamp)
            self._pending_publish = None

        # Camera is a first-class sensor, but the pipeline must remain total (no gating).
        # If the ring buffer is empty, run with an empty camera batch and rely on LiDAR/IMU/odom.
        if not self.camera_ringbuf:
            from fl_slam_poc.backend.structures.measurement_batch import create_empty_measurement_batch

            self.get_logger().warn(
                f"Scan {self.scan_count}: camera ring buffer empty; using empty camera MeasurementBatch (no gating)."
            )
            camera_batch = create_empty_measurement_batch(
                n_feat=self.config.n_feat,
                n_surfel=self.config.n_surfel,
            )
        else:
            # Select frame closest to scan time (argmin |t_frame - t_scan|)
            best_idx = 0
            best_dt = abs(self.camera_ringbuf[0][0] - t_scan)
            for i, (t_frame, _, _) in enumerate(self.camera_ringbuf):
                dt_frame = abs(t_frame - t_scan)
                if dt_frame < best_dt:
                    best_dt = dt_frame
                    best_idx = i
            stamp_rgb, rgb, depth = self.camera_ringbuf[best_idx]
            if self.scan_count <= 5:
                self.get_logger().info(
                    f"Camera frame selected: idx={best_idx}/{len(self.camera_ringbuf)}, "
                    f"t_frame={stamp_rgb:.6f}, t_scan={t_scan:.6f}, dt={best_dt:.6f}s"
                )
            if self.rerun_visualizer is not None:
                self.rerun_visualizer.log_rgbd(rgb, depth, stamp_rgb)
            extraction_result = self.visual_extractor.extract(
                rgb, depth, color_order="RGB", timestamp_ns=int(stamp_rgb * 1e9)
            )
            # LiDAR points in base frame -> camera frame for lidar_depth_evidence
            points_base = np.array(points, dtype=np.float64)
            if points_base.shape[0] > 0:
                points_cam = (self.R_base_camera.T @ (points_base.T - self.t_base_camera[:, None])).T
            else:
                points_cam = np.zeros((0, 3), dtype=np.float64)
            fused = splat_prep_fused(
                extraction_result,
                points_cam,
                self.camera_intrinsics,
                self.depth_fusion_config,
            )
            # Camera features are extracted in the camera frame; convert to base frame so all
            # MeasurementBatch primitives share a common sensor frame (base) before world transform.
            if fused:
                R = self.R_base_camera
                t = self.t_base_camera
                fused_base = []
                for f in fused:
                    xyz_cam = np.asarray(f.xyz, dtype=np.float64).reshape(3)
                    cov_cam = np.asarray(f.cov_xyz, dtype=np.float64).reshape(3, 3)
                    xyz_base = R @ xyz_cam + t
                    cov_base = R @ cov_cam @ R.T
                    if f.mu_app is not None:
                        mu_app_base = (R @ np.asarray(f.mu_app, dtype=np.float64).reshape(3))
                    else:
                        mu_app_base = None
                    fused_base.append(
                        f.__class__(
                            u=f.u,
                            v=f.v,
                            xyz=xyz_base,
                            cov_xyz=cov_base,
                            info_xyz=f.info_xyz,  # not used downstream (MeasurementBatch recomputes in info form)
                            logdet_cov=f.logdet_cov,
                            canonical_theta=f.canonical_theta,
                            canonical_log_partition=f.canonical_log_partition,
                            desc=f.desc,
                            weight=f.weight,
                            meta=f.meta,
                            mu_app=mu_app_base,
                            kappa_app=f.kappa_app,
                            color=f.color,
                        )
                    )
                fused = fused_base
            camera_batch = feature_list_to_camera_batch(
                fused,
                stamp_rgb,
                n_feat=self.config.n_feat,
                n_surfel=self.config.n_surfel,
                eps_lift=self.config.eps_lift,
            )

        # Slice IMU to integration window only (PIPELINE_DESIGN_GAPS §5.6): avoids 4000-step scan.
        # Window = [min(t_last_scan, scan_start), max(t_scan, scan_end)]; pad to GC_MAX_IMU_PREINT_LEN.
        t_min = min(t_last_scan, scan_start_time)
        t_max = max(t_scan, scan_end_time)
        eps_t = 1e-9
        window = [
            (t, g, a)
            for (t, g, a) in self.imu_buffer
            if t_min - eps_t <= t <= t_max + eps_t
        ]
        if len(window) > constants.GC_MAX_IMU_PREINT_LEN:
            window = window[-constants.GC_MAX_IMU_PREINT_LEN:]
        M_preint = constants.GC_MAX_IMU_PREINT_LEN
        imu_stamps = np.zeros((M_preint,), dtype=np.float64)
        imu_gyro = np.zeros((M_preint, 3), dtype=np.float64)
        imu_accel = np.zeros((M_preint, 3), dtype=np.float64)
        for i, (t, g, a) in enumerate(window):
            imu_stamps[i] = float(t)
            imu_gyro[i, :] = np.array(g)
            imu_accel[i, :] = np.array(a)
        imu_stamps_j = jnp.array(imu_stamps, dtype=jnp.float64)
        imu_gyro_j = jnp.array(imu_gyro, dtype=jnp.float64)
        imu_accel_j = jnp.array(imu_accel, dtype=jnp.float64)

        # Diagnostic: log IMU buffer state
        n_valid_imu = int(np.sum(imu_stamps > 0.0))
        if n_valid_imu > 0:
            valid_stamps = imu_stamps[imu_stamps > 0.0]
            t_imu_min = float(np.min(valid_stamps))
            t_imu_max = float(np.max(valid_stamps))
            self.get_logger().info(
                f"Scan #{self.scan_count} IMU buffer: {n_valid_imu}/{len(self.imu_buffer)} valid, "
                f"stamp_range=[{t_imu_min:.6f}, {t_imu_max:.6f}], "
                f"integration_window=({t_last_scan:.6f}, {t_scan:.6f})"
            )
        
        # =====================================================================
        # IMU INTEGRATION TIME COMPUTATION (after IMU arrays are built)
        # =====================================================================
        # CRITICAL: IMU is path-integral data over (t_last_scan, t_scan)
        # dt_int = sum of IMU sample intervals in that interval
        from fl_slam_poc.backend.pipeline import compute_imu_integration_time
        dt_int = compute_imu_integration_time(
            imu_stamps=imu_stamps,  # Use numpy array directly
            t_start=t_last_scan,
            t_end=t_scan,
        )
        
        # Log dt_int with invariant check
        self.get_logger().info(
            f"Scan #{self.scan_count} IMU integration: "
            f"dt_int={dt_int:.6f} dt_scan={dt_sec:.6f} "
            f"invariant_check={0.0 <= dt_int <= dt_sec}"
        )
        
        # Run pipeline for each hypothesis
        results: List[ScanPipelineResult] = []
        # Commutative IW sufficient-statistics accumulation (order-robust; applied once per scan)
        accum_dPsi = jnp.zeros((7, 6, 6), dtype=jnp.float64)
        accum_dnu = jnp.zeros((7,), dtype=jnp.float64)
        accum_meas_dPsi = jnp.zeros((3, 3, 3), dtype=jnp.float64)
        accum_meas_dnu = jnp.zeros((3,), dtype=jnp.float64)

        # Odom vs belief diagnostic: snapshot raw odom and belief before this scan
        _diag_odom_x = _diag_odom_y = _diag_odom_yaw = _diag_odom_vx = _diag_odom_vy = _diag_odom_wz = 0.0
        _diag_bel_x = _diag_bel_y = _diag_bel_yaw = _diag_bel_vx = _diag_bel_vy = 0.0
        if self._odom_belief_diagnostic_file and (self._odom_belief_diagnostic_max <= 0 or self.scan_count <= self._odom_belief_diagnostic_max):
            if self.current_belief is not None:
                _diag_bel_x, _diag_bel_y, _diag_bel_yaw, _diag_bel_vx, _diag_bel_vy = _belief_xyyaw_vel(self.current_belief)
            if self.last_odom_pose is not None:
                op = np.array(self.last_odom_pose, dtype=np.float64).ravel()[:6]
                _diag_odom_x, _diag_odom_y = float(op[0]), float(op[1])
                _diag_odom_yaw = _yaw_deg_from_pose_6d(op)
            ot = np.array(self.last_odom_twist, dtype=np.float64).ravel()[:6]
            _diag_odom_vx, _diag_odom_vy, _diag_odom_wz = float(ot[0]), float(ot[1]), float(ot[5])

        try:
            # Update per-scan measurement covariance from IW state (shared across hypotheses)
            self.config.Sigma_meas = measurement_noise_mean_jax(self.measurement_noise_state, idx=2)
            self.config.Sigma_g = measurement_noise_mean_jax(self.measurement_noise_state, idx=0)
            self.config.Sigma_a = measurement_noise_mean_jax(self.measurement_noise_state, idx=1)

            # Precompute Q once for this scan (shared across hypotheses)
            Q_scan = process_noise_state_to_Q_jax(self.process_noise_state)

            # t_last_scan is already defined above (line 547) for IMU integration interval
            # IMU is path-integral data over (t_last_scan, t_scan), not a snapshot
            
            for i, belief in enumerate(self.hypotheses):
                result = process_scan_single_hypothesis(
                    belief_prev=belief,
                    raw_points=points,
                    raw_timestamps=timestamps,
                    raw_weights=weights,
                    raw_ring=ring,
                    raw_tag=tag,
                    imu_stamps=imu_stamps_j,
                    imu_gyro=imu_gyro_j,
                    imu_accel=imu_accel_j,
                    odom_pose=self.last_odom_pose if self.last_odom_pose is not None else se3_identity(),
                    odom_cov_se3=(
                        self.last_odom_cov_se3
                        if self.last_odom_cov_se3 is not None
                        else (1e12 * jnp.eye(6, dtype=jnp.float64))
                    ),
                    scan_start_time=scan_start_time,
                    scan_end_time=scan_end_time,
                    dt_sec=dt_sec,
                    t_last_scan=t_last_scan,  # IMU integration interval start
                    t_scan=t_scan,            # IMU integration interval end
                    Q=Q_scan,
                    config=self.config,
                    odom_twist=self.last_odom_twist,  # Phase 2: odom twist for velocity factors
                    odom_twist_cov=self.last_odom_twist_cov,
                    primitive_map=self.primitive_map,
                    camera_batch=camera_batch,
                )
                results.append(result)
                self.hypotheses[i] = result.belief_updated

                # Stage 1: Update PrimitiveMap from first hypothesis
                # (For multi-hypothesis, we use hypothesis 0's map update; proper multi-hyp requires merging)
                if i == 0 and result.primitive_map_updated is not None:
                    self.primitive_map = result.primitive_map_updated

                # Accumulate commutative IW sufficient statistics
                w_h = float(self.hyp_weights[i])
                accum_dPsi = accum_dPsi + w_h * result.iw_process_dPsi
                accum_dnu = accum_dnu + w_h * result.iw_process_dnu
                accum_meas_dPsi = accum_meas_dPsi + w_h * result.iw_meas_dPsi
                accum_meas_dnu = accum_meas_dnu + w_h * result.iw_meas_dnu

            # Combine hypotheses
            combined_belief, combo_cert, combo_effect = process_hypotheses(
                hypotheses=self.hypotheses,
                weights=self.hyp_weights,
                config=self.config,
            )
            
            self.current_belief = combined_belief
            self.pipeline_runs += 1

            # Apply IW updates ONCE per scan (after hypothesis combine). No gates: always apply;
            # readiness is a weight on sufficient stats (process has no prediction at scan 0 -> weight 0).
            w_process = min(1, self.scan_count)  # 0 at scan 0, 1 from scan 1 (prior/budget)
            w_meas = 1.0
            self.process_noise_state, proc_iw_cert_vec = process_noise_iw_apply_suffstats_jax(
                pn_state=self.process_noise_state,
                dPsi=w_process * accum_dPsi,
                dnu=w_process * accum_dnu,
                dt_sec=dt_sec,
                eps_psd=self.config.eps_psd,
            )
            self.Q = process_noise_state_to_Q_jax(self.process_noise_state)
            self.measurement_noise_state, meas_iw_cert_vec = measurement_noise_apply_suffstats_jax(
                mn_state=self.measurement_noise_state,
                dPsi_blocks=w_meas * accum_meas_dPsi,
                dnu=w_meas * accum_meas_dnu,
                eps_psd=self.config.eps_psd,
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

            # Collect diagnostics: minimal tape (default) or full ScanDiagnostics
            if results and results[0].diagnostics_tape is not None:
                from dataclasses import replace
                entry = replace(results[0].diagnostics_tape, scan_number=self.scan_count)
                self.diagnostics_log.append_tape(entry)
            elif results and results[0].diagnostics is not None:
                diag = results[0].diagnostics
                diag.scan_number = self.scan_count
                diag.trace_Q_mode = float(jnp.trace(self.Q))
                diag.trace_Sigma_lidar_mode = float(jnp.trace(self.config.Sigma_meas))
                diag.trace_Sigma_g_mode = float(jnp.trace(self.config.Sigma_g))
                diag.trace_Sigma_a_mode = float(jnp.trace(self.config.Sigma_a))
                self.diagnostics_log.append(diag)

            # Defer publish to next callback (state/TF/path published when next scan starts)
            pose_6d = combined_belief.mean_world_pose()
            self._pending_publish = (jnp.array(pose_6d), stamp_sec)

            # Odom vs belief diagnostic: append one row (raw odom, belief start, belief end)
            if self._odom_belief_diagnostic_file and (self._odom_belief_diagnostic_max <= 0 or self.scan_count <= self._odom_belief_diagnostic_max):
                bel_end_x, bel_end_y, bel_end_yaw, bel_end_vx, bel_end_vy = _belief_xyyaw_vel(combined_belief)
                header = (
                    "scan,t_sec,odom_x,odom_y,odom_yaw_deg,odom_vx,odom_vy,odom_wz,"
                    "bel_start_x,bel_start_y,bel_start_yaw_deg,bel_start_vx,bel_start_vy,"
                    "bel_end_x,bel_end_y,bel_end_yaw_deg,bel_end_vx,bel_end_vy\n"
                )
                row = (
                    f"{self.scan_count},{stamp_sec:.6f},"
                    f"{_diag_odom_x:.6f},{_diag_odom_y:.6f},{_diag_odom_yaw:.6f},{_diag_odom_vx:.6f},{_diag_odom_vy:.6f},{_diag_odom_wz:.6f},"
                    f"{_diag_bel_x:.6f},{_diag_bel_y:.6f},{_diag_bel_yaw:.6f},{_diag_bel_vx:.6f},{_diag_bel_vy:.6f},"
                    f"{bel_end_x:.6f},{bel_end_y:.6f},{bel_end_yaw:.6f},{bel_end_vx:.6f},{bel_end_vy:.6f}\n"
                )
                try:
                    with open(self._odom_belief_diagnostic_file, "a", encoding="utf-8") as f:
                        if not self._odom_belief_diagnostic_header_written:
                            f.write(header)
                            self._odom_belief_diagnostic_header_written = True
                        f.write(row)
                except OSError as e:
                    self.get_logger().warn(f"Odom/belief diagnostic write failed: {e}")

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
        """Publish state from a 6D pose [trans, rotvec] in anchor frame. Export uses anchor_correction ∘ pose."""
        pose_export = se3_compose(self.anchor_correction, pose_6d)
        rotvec, trans = se3_to_rotvec_trans(pose_export)
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

        if self.rerun_visualizer is not None:
            path_xyz = np.array(
                [
                    [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                    for p in self.trajectory_poses
                ],
                dtype=np.float64,
            )
            self.rerun_visualizer.log_trajectory(path_xyz, stamp_sec)

        # PrimitiveMap as PointCloud2 (/gc/map/points)
        if self.map_publisher is not None and self.primitive_map is not None:
            self.map_publisher.publish(self.primitive_map, stamp_sec)
        
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
        
        prim_count = int(self.primitive_map.count) if self.primitive_map is not None else 0
        status = {
            "elapsed_sec": elapsed,
            "odom_count": self.odom_count,
            "scan_count": self.scan_count,
            "imu_count": self.imu_count,
            "pipeline_runs": self.pipeline_runs,
            "hypotheses": self.config.K_HYP,
            "primitive_map_count": prim_count,
        }
        
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)
        
        self.get_logger().info(
            f"GC Status: odom={self.odom_count}, scans={self.scan_count}, "
            f"imu={self.imu_count}, pipeline={self.pipeline_runs}, "
            f"primitive_map={prim_count}"
        )

    def destroy_node(self):
        """Clean up."""
        if self.rerun_visualizer is not None:
            self.rerun_visualizer.flush()
        # Drain deferred publish so last scan state is written
        if self._pending_publish is not None:
            pose_6d, stamp = self._pending_publish
            self._publish_state_from_pose(pose_6d, stamp)
            self._pending_publish = None
        if self.trajectory_file:
            self.trajectory_file.flush()
            self.trajectory_file.close()
            self.get_logger().info(f"Trajectory saved: {self.trajectory_export_path}")

        # Save diagnostics log for dashboard
        diagnostics_path = str(self.get_parameter("diagnostics_export_path").value)
        if diagnostics_path and self.diagnostics_log.total_scans > 0:
            self.diagnostics_log.end_time = time.time()
            try:
                import os
                parent = os.path.dirname(diagnostics_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self.diagnostics_log.save_npz(diagnostics_path)
                self.get_logger().info(
                    f"Diagnostics saved: {diagnostics_path} ({self.diagnostics_log.total_scans} scans)"
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to save diagnostics: {e}")

        # Export primitive map for post-run JAXsplat (or other) visualization
        splat_path = str(self.get_parameter("splat_export_path").value).strip()
        if splat_path and self.primitive_map is not None and self.primitive_map.count > 0:
            try:
                import os
                from fl_slam_poc.backend.structures.primitive_map import extract_primitive_map_view
                view = extract_primitive_map_view(self.primitive_map)
                n = view.count
                # Convert to numpy for NPZ (JAX arrays may be on device)
                positions = np.asarray(view.positions)
                covariances = np.asarray(view.covariances)
                colors = np.asarray(view.colors) if view.colors is not None else np.zeros((n, 3), dtype=np.float64)
                weights = np.asarray(view.weights)
                directions = np.asarray(view.directions)
                kappas = np.asarray(view.kappas)
                os.makedirs(os.path.dirname(splat_path) or ".", exist_ok=True)
                np.savez_compressed(
                    splat_path,
                    positions=positions,
                    covariances=covariances,
                    colors=colors,
                    weights=weights,
                    directions=directions,
                    kappas=kappas,
                    n=n,
                )
                self.get_logger().info(f"Splat export saved: {splat_path} ({n} primitives)")
            except Exception as e:
                self.get_logger().warn(f"Failed to save splat export: {e}")

        super().destroy_node()


def main():
    rclpy.init()
    node = GeometricCompositionalBackend()
    node.get_logger().info("Backend node created, entering spin loop...")
    
    # Use MultiThreadedExecutor to allow IMU callbacks to run while lidar pipeline processes.
    # With single-threaded spin, IMU callbacks can't run during the ~1-2s pipeline processing,
    # causing 200Hz IMU messages to overflow the depth=10 queue.
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Shutting down. Final counts: scans={node.scan_count}, odom={node.odom_count}, imu={node.imu_count}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
