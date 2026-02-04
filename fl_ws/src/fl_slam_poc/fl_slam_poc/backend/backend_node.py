"""
Geometric Compositional SLAM v2 Backend Node.

Actually uses the GC operators to process LiDAR scans.
This is NOT passthrough - it runs the full 14-step pipeline.

Reference: docs/GC_SLAM.md
"""

import json
import struct
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Any, Dict

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
from fl_slam_poc.msg import RGBDImage, VisualFeatureBatch

from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.tiling import ma_hex_stencil_tile_ids
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    se3_identity,
    se3_from_rotvec_trans,
    se3_to_rotvec_trans,
    se3_compose,
    se3_inverse,
)
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.common.certificates import CertBundle, ScanIOCert, DeviceRuntimeCert
from fl_slam_poc.common.certificates import InfluenceCert, aggregate_certificates
from fl_slam_poc.common.runtime_counters import (
    reset_runtime_counters,
    record_host_to_device,
    record_device_to_host,
    record_jit_recompile,
    consume_runtime_counters,
)
from fl_slam_poc.backend.pipeline import (
    PipelineConfig,
    RuntimeManifest,
    process_scan_single_hypothesis,
    process_hypotheses,
    ScanPipelineResult,
)
from fl_slam_poc.backend.structures import (
    AtlasMap,
    create_empty_atlas_map,
    extract_primitive_map_view,
    create_empty_measurement_batch,
    ProcessNoiseIWState,
    create_datasheet_process_noise_state,
    MeasurementNoiseIWState,
    create_datasheet_measurement_noise_state,
    primitive_map_fuse,
)
from fl_slam_poc.backend.operators import (
    process_noise_state_to_Q_jax,
    process_noise_iw_apply_suffstats_jax,
    measurement_noise_mean_jax,
    measurement_noise_apply_suffstats_jax,
)
from fl_slam_poc.backend.diagnostics import DiagnosticsLog

from fl_slam_poc.backend.map_publisher import PrimitiveMapPublisher
from fl_slam_poc.backend.rerun_visualizer import RerunVisualizer
from fl_slam_poc.backend.camera_batch_utils import feature_list_to_camera_batch
from fl_slam_poc.backend.structures.measurement_batch import (
    MeasurementBatch,
    measurement_batch_mean_positions,
    measurement_batch_mean_directions,
    measurement_batch_kappas,
)
from fl_slam_poc.backend.operators.primitive_association import (
    _compute_sparse_cost_matrix_jax,
    _sinkhorn_unbalanced_fixed_k_jax,
)
from fl_slam_poc.frontend.sensors.visual_types import (
    Feature3D,
    ExtractionResult,
    PinholeIntrinsics,
)
from fl_slam_poc.frontend.sensors.splat_prep import splat_prep_fused
from fl_slam_poc.frontend.sensors.lidar_camera_depth_fusion import (
    LidarCameraDepthFusionConfig,
)

from scipy.spatial.transform import Rotation


def _cast_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: Any
    cast: Optional[Callable[[Any], Any]] = None


PARAM_SPECS: List[ParamSpec] = [
    ParamSpec("odom_frame", "odom", str),
    ParamSpec("base_frame", "base_link", str),
    # Backend subscribes ONLY to /gc/sensors/* (canonical topics from sensor hub)
    ParamSpec("lidar_topic", "/gc/sensors/lidar_points", str),
    ParamSpec("odom_topic", "/gc/sensors/odom", str),
    ParamSpec("imu_topic", "/gc/sensors/imu", str),
    ParamSpec("trajectory_export_path", "/tmp/gc_slam_trajectory.tum", str),
    ParamSpec("diagnostics_export_path", "results/gc_slam_diagnostics.npz", str),
    ParamSpec("splat_export_path", "", str),
    ParamSpec("status_check_period_sec", 5.0, float),
    # No-TF extrinsics (T_{base<-sensor}) in [x, y, z, rx, ry, rz] rotvec (radians).
    # extrinsics_source: inline | file. When file: load from T_base_lidar_file, T_base_imu_file (fail if missing).
    ParamSpec("extrinsics_source", "inline", str),
    ParamSpec("T_base_lidar_file", "", str),
    ParamSpec("T_base_imu_file", "", str),
    ParamSpec("T_base_lidar", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], list),
    ParamSpec("T_base_imu", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], list),
    # LiDAR measurement noise prior (m² isotropic). Kimera VLP-16: 1e-3.
    ParamSpec("lidar_sigma_meas", 0.01, float),
    # When true, derive Sigma_g/Sigma_a from first N IMU messages (units/fallback doc'd). When false, use priors only.
    ParamSpec("use_imu_message_covariance", False, _cast_bool),
    # Hard single-path enforcement: if enabled, missing topics are hard errors.
    ParamSpec("use_imu", True, _cast_bool),
    ParamSpec("use_odom", True, _cast_bool),
    # IMU gravity scaling (1.0 = nominal; 0.0 disables gravity contribution)
    ParamSpec("imu_gravity_scale", 1.0, float),
    ParamSpec("imu_accel_scale", 1.0, float),
    # Deskew rotation-only mode: removes hidden IMU translation leak through deskew
    ParamSpec("deskew_rotation_only", False, _cast_bool),
    # Timing/profiling
    ParamSpec("enable_timing", False, _cast_bool),
    # Parallelize independent stages
    ParamSpec("enable_parallel_stages", False, _cast_bool),
    # JAX warmup (compile before first scan; no state mutation)
    ParamSpec("warmup_enable", True, _cast_bool),
    # Numerical epsilons (domain stabilization)
    ParamSpec("eps_psd", constants.GC_EPS_PSD, float),
    ParamSpec("eps_lift", constants.GC_EPS_LIFT, float),
    ParamSpec("eps_mass", constants.GC_EPS_MASS, float),
    # Fusion conditioning
    ParamSpec("alpha_min", constants.GC_ALPHA_MIN, float),
    ParamSpec("alpha_max", constants.GC_ALPHA_MAX, float),
    ParamSpec("kappa_scale", constants.GC_KAPPA_SCALE, float),
    ParamSpec("c0_cond", constants.GC_C0_COND, float),
    # Power tempering
    ParamSpec("power_beta_min", 0.25, float),
    ParamSpec("power_beta_exc_c", 50.0, float),
    ParamSpec("power_beta_z_c", 1.0, float),
    # Excitation coupling
    ParamSpec("c_dt", constants.GC_C_DT, float),
    ParamSpec("c_ex", constants.GC_C_EX, float),
    ParamSpec("c_frob", constants.GC_C_FROB, float),
    # Planar prior (soft constraints)
    ParamSpec("planar_z_ref", constants.GC_PLANAR_Z_REF, float),
    ParamSpec("planar_z_sigma", constants.GC_PLANAR_Z_SIGMA, float),
    ParamSpec("planar_vz_sigma", constants.GC_PLANAR_VZ_SIGMA, float),
    ParamSpec("enable_planar_prior", True, _cast_bool),
    # Odom twist evidence
    ParamSpec("enable_odom_twist", True, _cast_bool),
    ParamSpec("odom_twist_vel_sigma", constants.GC_ODOM_TWIST_VEL_SIGMA, float),
    ParamSpec("odom_twist_wz_sigma", constants.GC_ODOM_TWIST_WZ_SIGMA, float),
    ParamSpec("odom_z_variance_prior", constants.GC_ODOM_Z_VARIANCE_PRIOR, float),
    # Association OT params
    ParamSpec("ot_epsilon", 0.1, float),
    ParamSpec("ot_tau_a", 0.5, float),
    ParamSpec("ot_tau_b", 0.5, float),
    # PrimitiveMap + association budgets (fixed-cost)
    ParamSpec("n_feat", constants.GC_N_FEAT, int),
    ParamSpec("n_surfel", constants.GC_N_SURFEL, int),
    ParamSpec("k_assoc", constants.GC_K_ASSOC, int),
    ParamSpec("k_sinkhorn", constants.GC_K_SINKHORN, int),
    ParamSpec("primitive_map_max_size", constants.GC_PRIMITIVE_MAP_MAX_SIZE, int),
    ParamSpec("primitive_forgetting_factor", constants.GC_PRIMITIVE_FORGETTING_FACTOR, float),
    ParamSpec("k_insert_tile", constants.GC_K_INSERT_TILE, int),
    ParamSpec("k_merge_pairs_tile", constants.GC_K_MERGE_PAIRS_PER_TILE, int),
    ParamSpec("primitive_merge_max_tile_size", constants.GC_PRIMITIVE_MERGE_MAX_TILE_SIZE, int),
    # Phase 6 (Real tiling): deterministic MA-Hex 3D tiling + fixed active/stencil budgets.
    # These are fixed-cost budgets; runtime YAML must match compiled constants (fail-fast).
    ParamSpec("H_TILE", constants.GC_H_TILE, float),
    ParamSpec("R_ACTIVE_TILES_XY", constants.GC_R_ACTIVE_TILES_XY, int),
    ParamSpec("R_ACTIVE_TILES_Z", constants.GC_R_ACTIVE_TILES_Z, int),
    ParamSpec("N_ACTIVE_TILES", constants.GC_N_ACTIVE_TILES, int),
    ParamSpec("R_STENCIL_TILES_XY", constants.GC_R_STENCIL_TILES_XY, int),
    ParamSpec("R_STENCIL_TILES_Z", constants.GC_R_STENCIL_TILES_Z, int),
    ParamSpec("N_STENCIL_TILES", constants.GC_N_STENCIL_TILES, int),
    ParamSpec("M_TILE_VIEW", constants.GC_M_TILE_VIEW, int),
    ParamSpec("primitive_merge_threshold", constants.GC_PRIMITIVE_MERGE_THRESHOLD, float),
    ParamSpec("primitive_cull_weight_threshold", constants.GC_PRIMITIVE_CULL_WEIGHT_THRESHOLD, float),
    # Surfel extraction config
    ParamSpec("surfel_voxel_size_m", 0.1, float),
    ParamSpec("surfel_min_points_per_voxel", 3, int),
    # Diagnostics
    ParamSpec("save_full_diagnostics", False, _cast_bool),
    # Smoothed initial reference: buffer first K odom, then set first_odom_pose = aggregate (PIPELINE_DESIGN_GAPS §5.4.1)
    ParamSpec("init_window_odom_count", 10, int),
    # PointCloud2 layout: vlp16 (Kimera/VLP-16). See docs/KIMERA_DATASET_AND_PIPELINE.md (§6 PointCloud2 layout).
    ParamSpec("pointcloud_layout", "vlp16", str),
    # Odom vs belief diagnostic: when non-empty, write CSV (raw odom, belief start, belief end) per scan.
    ParamSpec("odom_belief_diagnostic_file", "", str),
    ParamSpec("odom_belief_diagnostic_max_scans", 0, int),
    # Camera (required): single RGBD topic from camera_rgbd_node
    ParamSpec("camera_rgbd_topic", "/gc/sensors/camera_rgbd", str),
    # Visual feature batch (from C++ visual_feature_node)
    ParamSpec("visual_feature_topic", "/gc/sensors/visual_features", str),
    ParamSpec("camera_K", [500.0, 500.0, 320.0, 240.0], list),
    ParamSpec("T_base_camera", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], list),
    ParamSpec("ringbuf_len", constants.GC_RINGBUF_LEN, int),
    # Backend time alignment (calibrated profile; continuous, no gates)
    ParamSpec("time_alignment_profile", "", str),
    ParamSpec("time_alignment_reference_topic", "", str),
    # Async LiDAR processing: bounded queues to avoid callback blocking.
    ParamSpec("lidar_queue_len", 2, int),
    ParamSpec("publish_queue_len", 4, int),
    ParamSpec("publish_timer_period_sec", 0.01, float),
    ParamSpec("use_rerun", True, _cast_bool),
    ParamSpec("rerun_recording_path", "", str),
    ParamSpec("rerun_spawn", False, _cast_bool),
    ParamSpec("event_log_path", "", str),
    # Manifest-only flags (view layer)
    ParamSpec("bev_backend_enabled", False, _cast_bool),
    ParamSpec("bev_views_n", 0, int),
    # Explicit backend selection (single-path)
    ParamSpec("map_backend", constants.GC_MAP_BACKEND_PRIMITIVE_MAP, str),
    ParamSpec("pose_evidence_backend", constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES, str),
]

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
    See docs/KIMERA_DATASET_AND_PIPELINE.md (§6 PointCloud2 layout).
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
    record_host_to_device(pts)
    record_host_to_device(t)
    record_host_to_device(w)
    record_host_to_device(ring)
    record_host_to_device(tag)
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
        for spec in PARAM_SPECS:
            self.declare_parameter(spec.name, spec.default)

    def _load_parameters(self) -> None:
        """Load parameters into a typed cache (single source of truth)."""
        params: Dict[str, Any] = {}
        for spec in PARAM_SPECS:
            value = self.get_parameter(spec.name).value
            cast = spec.cast
            if cast is None:
                if isinstance(spec.default, bool):
                    cast = _cast_bool
                elif isinstance(spec.default, int):
                    cast = int
                elif isinstance(spec.default, float):
                    cast = float
                elif isinstance(spec.default, str):
                    cast = str
                elif isinstance(spec.default, list):
                    cast = list
                else:
                    cast = lambda v: v
            params[spec.name] = cast(value)
        self._params = params

    def _init_state(self):
        """Initialize Geometric Compositional state."""
        # Pipeline configuration
        self.config = PipelineConfig()
        self._load_parameters()
        p = self._params

        # Apply fixed-cost budgets from parameters (single source: config yaml)
        for name in (
            "n_feat",
            "n_surfel",
            "k_assoc",
            "k_sinkhorn",
            "k_insert_tile",
            "k_merge_pairs_tile",
            "primitive_map_max_size",
            "primitive_merge_max_tile_size",
        ):
            setattr(self.config, name, p[name])
        self.config.primitive_forgetting_factor = p["primitive_forgetting_factor"]
        # Phase 6 tiling budgets (fail-fast if YAML drifts from compiled constants).
        for name in (
            "H_TILE",
            "R_ACTIVE_TILES_XY",
            "R_ACTIVE_TILES_Z",
            "N_ACTIVE_TILES",
            "R_STENCIL_TILES_XY",
            "R_STENCIL_TILES_Z",
            "N_STENCIL_TILES",
            "M_TILE_VIEW",
        ):
            setattr(self.config, name, p[name])

        if self.config.H_TILE != constants.GC_H_TILE:
            raise ValueError(
                f"H_TILE must equal compiled GC_H_TILE={constants.GC_H_TILE}; got {self.config.H_TILE}"
            )
        if self.config.R_ACTIVE_TILES_XY != constants.GC_R_ACTIVE_TILES_XY:
            raise ValueError(
                f"R_ACTIVE_TILES_XY must equal compiled GC_R_ACTIVE_TILES_XY={constants.GC_R_ACTIVE_TILES_XY}; "
                f"got {self.config.R_ACTIVE_TILES_XY}"
            )
        if self.config.R_ACTIVE_TILES_Z != constants.GC_R_ACTIVE_TILES_Z:
            raise ValueError(
                f"R_ACTIVE_TILES_Z must equal compiled GC_R_ACTIVE_TILES_Z={constants.GC_R_ACTIVE_TILES_Z}; "
                f"got {self.config.R_ACTIVE_TILES_Z}"
            )
        if self.config.R_STENCIL_TILES_XY != constants.GC_R_STENCIL_TILES_XY:
            raise ValueError(
                f"R_STENCIL_TILES_XY must equal compiled GC_R_STENCIL_TILES_XY={constants.GC_R_STENCIL_TILES_XY}; "
                f"got {self.config.R_STENCIL_TILES_XY}"
            )
        if self.config.R_STENCIL_TILES_Z != constants.GC_R_STENCIL_TILES_Z:
            raise ValueError(
                f"R_STENCIL_TILES_Z must equal compiled GC_R_STENCIL_TILES_Z={constants.GC_R_STENCIL_TILES_Z}; "
                f"got {self.config.R_STENCIL_TILES_Z}"
            )
        if self.config.N_ACTIVE_TILES != constants.GC_N_ACTIVE_TILES:
            raise ValueError(
                f"N_ACTIVE_TILES must equal compiled GC_N_ACTIVE_TILES={constants.GC_N_ACTIVE_TILES}; "
                f"got {self.config.N_ACTIVE_TILES}"
            )
        if self.config.N_STENCIL_TILES != constants.GC_N_STENCIL_TILES:
            raise ValueError(
                f"N_STENCIL_TILES must equal compiled GC_N_STENCIL_TILES={constants.GC_N_STENCIL_TILES}; "
                f"got {self.config.N_STENCIL_TILES}"
            )
        if self.config.M_TILE_VIEW != constants.GC_M_TILE_VIEW:
            raise ValueError(
                f"M_TILE_VIEW must equal compiled GC_M_TILE_VIEW={constants.GC_M_TILE_VIEW}; "
                f"got {self.config.M_TILE_VIEW}"
            )
        # Keep config budgets consistent for diagnostics/manifesting
        self.config.N_FEAT = int(self.config.n_feat)
        self.config.N_SURFEL = int(self.config.n_surfel)
        self.config.K_SINKHORN = int(self.config.k_sinkhorn)

        # Numerical epsilons
        self.config.eps_psd = p["eps_psd"]
        self.config.eps_lift = p["eps_lift"]
        self.config.eps_mass = p["eps_mass"]
        # Fusion conditioning
        self.config.alpha_min = p["alpha_min"]
        self.config.alpha_max = p["alpha_max"]
        self.config.kappa_scale = p["kappa_scale"]
        self.config.c0_cond = p["c0_cond"]
        # Power tempering
        self.config.power_beta_min = p["power_beta_min"]
        self.config.power_beta_exc_c = p["power_beta_exc_c"]
        self.config.power_beta_z_c = p["power_beta_z_c"]
        # Excitation coupling
        self.config.c_dt = p["c_dt"]
        self.config.c_ex = p["c_ex"]
        self.config.c_frob = p["c_frob"]
        # Planar prior
        self.config.planar_z_ref = p["planar_z_ref"]
        self.config.planar_z_sigma = p["planar_z_sigma"]
        self.config.planar_vz_sigma = p["planar_vz_sigma"]
        self.config.enable_planar_prior = p["enable_planar_prior"]
        # Odom twist evidence
        self.config.enable_odom_twist = p["enable_odom_twist"]
        self.config.odom_twist_vel_sigma = p["odom_twist_vel_sigma"]
        self.config.odom_twist_wz_sigma = p["odom_twist_wz_sigma"]
        self.odom_z_variance_prior = p["odom_z_variance_prior"]
        # OT params
        self.config.ot_epsilon = p["ot_epsilon"]
        self.config.ot_tau_a = p["ot_tau_a"]
        self.config.ot_tau_b = p["ot_tau_b"]
        # Primitive map thresholds
        self.config.primitive_merge_threshold = p["primitive_merge_threshold"]
        self.config.primitive_cull_weight_threshold = p["primitive_cull_weight_threshold"]
        # Surfel extraction
        self.config.surfel_voxel_size_m = p["surfel_voxel_size_m"]
        self.config.surfel_min_points_per_voxel = p["surfel_min_points_per_voxel"]
        # Diagnostics
        self.config.save_full_diagnostics = p["save_full_diagnostics"]
        self.config.enable_parallel_stages = p["enable_parallel_stages"]
        if self.config.save_full_diagnostics:
            self.get_logger().warn(
                "save_full_diagnostics is deprecated; only minimal tape diagnostics are recorded."
            )

        # Explicit backend selection (single-path, fail-fast)
        self.odom_topic = p["odom_topic"].strip()
        self.map_backend = p["map_backend"].strip()
        self.pose_evidence_backend = p["pose_evidence_backend"].strip()
        if self.map_backend != constants.GC_MAP_BACKEND_PRIMITIVE_MAP:
            raise ValueError(
                f"map_backend must be {constants.GC_MAP_BACKEND_PRIMITIVE_MAP!r}; got {self.map_backend!r}"
            )
        if self.pose_evidence_backend != constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES:
            raise ValueError(
                f"pose_evidence_backend must be {constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES!r}; "
                f"got {self.pose_evidence_backend!r}"
            )
        self.bev_backend_enabled = p["bev_backend_enabled"]
        self.bev_views_n = p["bev_views_n"]
        
        # Adaptive process noise IW state (datasheet priors) + derived Q
        self.process_noise_state: ProcessNoiseIWState = create_datasheet_process_noise_state()
        self.Q = process_noise_state_to_Q_jax(self.process_noise_state)
        
        # Adaptive measurement noise IW state (per-sensor, phase 1) + derived Sigma_meas (LiDAR)
        lidar_sigma_meas = p["lidar_sigma_meas"]
        self.measurement_noise_state: MeasurementNoiseIWState = create_datasheet_measurement_noise_state(
            lidar_sigma_meas=lidar_sigma_meas
        )
        self.config.Sigma_meas = measurement_noise_mean_jax(self.measurement_noise_state, idx=2)

        # AtlasMap is the canonical map (single-path primitives; single tile in Phase 2.1)
        self.primitive_map = create_empty_atlas_map(
            m_tile=self.config.primitive_map_max_size,
        )
        self.get_logger().info(
            f"AtlasMap initialized: m_tile={self.config.primitive_map_max_size}, "
            f"n_feat={self.config.n_feat}, n_surfel={self.config.n_surfel}"
        )

        # Optional JAX warmup to reduce first-scan latency (no state mutation).
        if p["warmup_enable"]:
            self._jit_warmup_primitives()

        # Parse and cache no-TF extrinsics (inline or from file; fail-fast if file missing when source=file).
        extrinsics_source = p["extrinsics_source"].strip().lower()
        if extrinsics_source == "file":
            lidar_file = p["T_base_lidar_file"].strip()
            imu_file = p["T_base_imu_file"].strip()
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
            self.R_base_lidar, self.t_base_lidar = _parse_T_base_sensor_6d(p["T_base_lidar"])
            self.R_base_imu, self.t_base_imu = _parse_T_base_sensor_6d(p["T_base_imu"])

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
        camera_K = list(p["camera_K"])
        if len(camera_K) != 4:
            raise ValueError("camera_K must be [fx, fy, cx, cy]; got length %d" % len(camera_K))
        self.camera_K = camera_K  # [fx, fy, cx, cy]
        self.R_base_camera, self.t_base_camera = _parse_T_base_sensor_6d(
            list(p["T_base_camera"])
        )
        ringbuf_len = p["ringbuf_len"]
        if ringbuf_len < 1:
            raise ValueError("ringbuf_len must be >= 1; got %d" % ringbuf_len)
        self.camera_ringbuf: List[Tuple[float, np.ndarray, np.ndarray]] = []  # (stamp_sec, rgb, depth)
        self.camera_ringbuf_max = ringbuf_len
        self.camera_drop_count = 0
        self.visual_feature_ringbuf: List[Tuple[float, VisualFeatureBatch]] = []
        self.visual_feature_ringbuf_max = ringbuf_len
        self.visual_feature_drop_count = 0
        self.odom_ringbuf: List[Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
        self.odom_ringbuf_max = ringbuf_len

        # Time alignment profile (calibrated offsets + drift per stream).
        self._time_align_enabled = False
        self._time_align_profile = {}
        self._time_align_reference = ""
        profile_path = p["time_alignment_profile"].strip()
        self._time_align_reference = p["time_alignment_reference_topic"].strip()
        if profile_path:
            import yaml
            with open(profile_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            ta = data.get("time_alignment", {})
            self._time_align_reference = str(ta.get("reference", self._time_align_reference or "")).strip()
            streams = ta.get("streams", {}) or {}
            for topic, cfg in streams.items():
                self._time_align_profile[str(topic)] = {
                    "offset": float(cfg.get("offset_sec", 0.0)),
                    "drift": float(cfg.get("drift_sec_per_sec", 0.0)),
                    "t0": float(cfg.get("t0_sec", 0.0)),
                }
            if not self._time_align_reference:
                raise ValueError("time_alignment_profile requires time_alignment.reference or time_alignment_reference_topic")
            self._time_align_enabled = True
        # Shared buffer lock for async callbacks (IMU/odom/camera/features).
        self._buffer_lock = threading.Lock()
        # Shared state lock for map/anchor access across threads.
        self._state_lock = threading.Lock()

        # Async LiDAR processing queues (bounded, single-path).
        self._lidar_queue_len = p["lidar_queue_len"]
        self._publish_queue_len = p["publish_queue_len"]
        self._publish_timer_period_sec = p["publish_timer_period_sec"]
        if self._lidar_queue_len < 1:
            raise ValueError(f"lidar_queue_len must be >= 1; got {self._lidar_queue_len}")
        if self._publish_queue_len < 1:
            raise ValueError(f"publish_queue_len must be >= 1; got {self._publish_queue_len}")
        if self._publish_timer_period_sec <= 0.0:
            raise ValueError(
                f"publish_timer_period_sec must be > 0; got {self._publish_timer_period_sec}"
            )
        self._lidar_queue: deque = deque(maxlen=self._lidar_queue_len)
        self._lidar_cv = threading.Condition()
        self._lidar_worker_stop = False
        self._lidar_worker = None
        self._lidar_msg_count = 0
        self._lidar_drop_count = 0
        self._lidar_error_count = 0
        self._publish_queue: deque = deque(maxlen=self._publish_queue_len)
        self._publish_drop_count = 0
        self.get_logger().info(
            "Camera: K=[fx=%s fy=%s cx=%s cy=%s], T_base_camera cached, ringbuf_len=%d"
            % (camera_K[0], camera_K[1], camera_K[2], camera_K[3], ringbuf_len)
        )
        if self._time_align_enabled:
            self.get_logger().info(
                f"Backend time alignment: enabled (reference={self._time_align_reference})"
            )
        # Camera intrinsics and splat_prep config (camera -> MeasurementBatch)
        fx, fy, cx, cy = float(camera_K[0]), float(camera_K[1]), float(camera_K[2]), float(camera_K[3])
        self.camera_intrinsics = PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
        self.depth_fusion_config = LidarCameraDepthFusionConfig()

        # Wire LiDAR origin (in base frame) into the pipeline so direction features are computed
        # from the sensor origin, not the base origin.
        self.config.lidar_origin_base = jnp.array(self.t_base_lidar, dtype=jnp.float64)
        self.config.imu_gravity_scale = p["imu_gravity_scale"]
        self.get_logger().info(f"IMU gravity scale: {self.config.imu_gravity_scale:.6f}")
        self.config.deskew_rotation_only = p["deskew_rotation_only"]
        self.config.enable_timing = p["enable_timing"]
        self.get_logger().info(f"Timing diagnostics: {'enabled' if self.config.enable_timing else 'disabled'}")
        self.get_logger().info(
            f"Parallel stages: {'enabled' if self.config.enable_parallel_stages else 'disabled'}"
        )
        self.get_logger().info(f"Deskew rotation-only: {self.config.deskew_rotation_only}")
        self.init_window_odom_count = p["init_window_odom_count"]
        self.get_logger().info(f"Init window: first_odom_pose = aggregate of first {self.init_window_odom_count} odom")
        self.pointcloud_layout = p["pointcloud_layout"].strip().lower()
        if self.pointcloud_layout not in ("vlp16",):
            raise ValueError(
                f"pointcloud_layout must be 'vlp16'; got {self.pointcloud_layout!r}. "
                "See docs/KIMERA_DATASET_AND_PIPELINE.md (§6 PointCloud2 layout)."
            )
        self.get_logger().info(f"PointCloud2 layout: {self.pointcloud_layout}")

        # Odom vs belief diagnostic (raw vs estimate)
        self._odom_belief_diagnostic_file = p["odom_belief_diagnostic_file"].strip()
        self._odom_belief_diagnostic_max = p["odom_belief_diagnostic_max_scans"]
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
        self.last_imu_cov_g = None  # (3,3) gyro covariance from IMU messages
        self.last_imu_cov_a = None  # (3,3) accel covariance from IMU messages

    def _jit_warmup_primitives(self) -> None:
        """Warm up JAX compile for association + fuse without mutating state."""
        t0 = time.perf_counter()
        n_feat = int(self.config.n_feat)
        n_surfel = int(self.config.n_surfel)
        k_assoc = int(self.config.k_assoc)

        batch = create_empty_measurement_batch(n_feat=n_feat, n_surfel=n_surfel)
        Lambdas = batch.Lambdas.at[0].set(jnp.eye(3, dtype=jnp.float64))
        thetas = batch.thetas.at[0].set(jnp.zeros((3,), dtype=jnp.float64))
        etas = batch.etas.at[0, 0].set(jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64))
        weights = batch.weights.at[0].set(1.0)
        valid_mask = batch.valid_mask.at[0].set(True)
        timestamps = batch.timestamps.at[0].set(0.0)
        colors = batch.colors.at[0].set(jnp.array([0.5, 0.5, 0.5], dtype=jnp.float64))

        batch = MeasurementBatch(
            Lambdas=Lambdas,
            thetas=thetas,
            etas=etas,
            weights=weights,
            sources=batch.sources,
            source_indices=batch.source_indices,
            valid_mask=valid_mask,
            timestamps=timestamps,
            colors=colors,
            n_feat=batch.n_feat,
            n_surfel=batch.n_surfel,
            n_camera_valid=1,
            n_lidar_valid=0,
        )

        meas_positions = measurement_batch_mean_positions(batch, eps_lift=self.config.eps_lift)
        meas_directions = measurement_batch_mean_directions(batch, eps_mass=self.config.eps_mass)
        meas_kappas = measurement_batch_kappas(batch)

        map_positions = jnp.zeros((1, 3), dtype=jnp.float64)
        map_directions = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float64)
        map_kappas = jnp.array([1.0], dtype=jnp.float64)
        candidate_indices = jnp.zeros((batch.n_total, k_assoc), dtype=jnp.int32)

        cost_matrix = _compute_sparse_cost_matrix_jax(
            meas_positions=meas_positions,
            meas_directions=meas_directions,
            meas_kappas=meas_kappas,
            map_positions=map_positions,
            map_directions=map_directions,
            map_kappas=map_kappas,
            candidate_indices=candidate_indices,
        )
        a = batch.valid_mask.astype(jnp.float64)
        a = a / jnp.maximum(jnp.sum(a), 1e-12)
        b = jnp.ones((k_assoc,), dtype=jnp.float64) / float(k_assoc)
        _sinkhorn_unbalanced_fixed_k_jax(
            C=cost_matrix,
            a=a,
            b=b,
            epsilon=0.1,
            tau_a=0.5,
            tau_b=0.5,
            K=int(self.config.k_sinkhorn),
        )

        block = min(constants.GC_ASSOC_BLOCK_SIZE, batch.n_total)
        K_flat = int(block * k_assoc)
        target_slots = jnp.zeros((K_flat,), dtype=jnp.int32)
        Lambdas_meas = jnp.tile(jnp.eye(3, dtype=jnp.float64)[None, :, :], (K_flat, 1, 1))
        thetas_meas = jnp.zeros((K_flat, 3), dtype=jnp.float64)
        etas_meas = jnp.zeros((K_flat, constants.GC_VMF_N_LOBES, 3), dtype=jnp.float64)
        weights_meas = jnp.ones((K_flat,), dtype=jnp.float64)
        responsibilities = jnp.ones((K_flat,), dtype=jnp.float64)
        valid_flat = jnp.ones((K_flat,), dtype=bool)
        colors_meas = jnp.zeros((K_flat, 3), dtype=jnp.float64)

        _ = primitive_map_fuse(
            atlas_map=self.primitive_map,
            tile_id=0,
            target_slots=target_slots,
            Lambdas_meas=Lambdas_meas,
            thetas_meas=thetas_meas,
            etas_meas=etas_meas,
            weights_meas=weights_meas,
            responsibilities=responsibilities,
            timestamp=0.0,
            valid_mask=valid_flat,
            colors_meas=colors_meas,
            eps_mass=self.config.eps_mass,
        )
        dt = time.perf_counter() - t0
        self.get_logger().info(f"JAX warmup complete: {dt:.3f}s")

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
        self.imu_drop_count = 0
        self.lidar_drop_count = 0
        self.odom_drop_count = 0
        
        # Tracking
        self.imu_count = 0
        self.odom_count = 0
        self.scan_count = 0
        self.pipeline_runs = 0
        self.last_scan_stamp = 0.0
        self.node_start_time = time.time()

        # Best-effort JIT shape signature cache for recompilation estimates.
        self._jit_signature_cache: set[tuple] = set()
        
        # Certificate history
        self.cert_history: List[CertBundle] = []

        # Diagnostics log for dashboard
        self.diagnostics_log = DiagnosticsLog(
            run_id=f"gc_slam_{int(self.node_start_time)}",
            start_time=self.node_start_time,
        )

        # Deferred publish: drain at start of next callback so pipeline hot path doesn't block on ROS

    def _buffer_time_window(self, buffer: list, time_index: int = 0) -> tuple[float, float, float]:
        if not buffer:
            return 0.0, 0.0, 0.0
        stamps = [float(entry[time_index]) for entry in buffer]
        return float(min(stamps)), float(max(stamps)), float(stamps[-1])

    def _align_stamp(self, topic: str, t: float) -> float:
        if not self._time_align_enabled:
            return float(t)
        cfg = self._time_align_profile.get(topic)
        if cfg is None:
            return float(t)
        t0 = float(cfg.get("t0", 0.0))
        offset = float(cfg.get("offset", 0.0))
        drift = float(cfg.get("drift", 0.0))
        return float(t) + offset + drift * (float(t) - t0)

    def _stream_stats(
        self,
        *,
        buffer_len: int,
        buffer_max: int,
        drops_total: int,
        window_start_sec: float,
        window_end_sec: float,
        last_stamp_sec: float,
    ) -> dict:
        return {
            "buffer_len": float(buffer_len),
            "buffer_max": float(buffer_max),
            "drops_total": float(drops_total),
            "window_start_sec": float(window_start_sec),
            "window_end_sec": float(window_end_sec),
            "last_stamp_sec": float(last_stamp_sec),
        }

    def _build_scan_io_cert(
        self,
        *,
        scan_seq: int,
        scan_stamp_sec: float,
        scan_window_start_sec: float,
        scan_window_end_sec: float,
        n_lidar_points: int,
    ) -> ScanIOCert:
        imu_start, imu_end, imu_last = self._buffer_time_window(self.imu_buffer)
        cam_start, cam_end, cam_last = self._buffer_time_window(self.camera_ringbuf)
        odom_last = float(self.last_odom_stamp) if self.last_odom_stamp > 0.0 else 0.0
        odom_len = 1 if self.last_odom_stamp > 0.0 else 0
        streams = {
            "lidar": self._stream_stats(
                buffer_len=n_lidar_points,
                buffer_max=int(self.config.N_POINTS_CAP),
                drops_total=self.lidar_drop_count,
                window_start_sec=scan_window_start_sec,
                window_end_sec=scan_window_end_sec,
                last_stamp_sec=scan_stamp_sec,
            ),
            "imu": self._stream_stats(
                buffer_len=len(self.imu_buffer),
                buffer_max=self.max_imu_buffer,
                drops_total=self.imu_drop_count,
                window_start_sec=imu_start,
                window_end_sec=imu_end,
                last_stamp_sec=imu_last,
            ),
            "camera_rgbd": self._stream_stats(
                buffer_len=len(self.camera_ringbuf),
                buffer_max=self.camera_ringbuf_max,
                drops_total=self.camera_drop_count,
                window_start_sec=cam_start,
                window_end_sec=cam_end,
                last_stamp_sec=cam_last,
            ),
            "visual_features": self._stream_stats(
                buffer_len=len(self.visual_feature_ringbuf),
                buffer_max=self.visual_feature_ringbuf_max,
                drops_total=self.visual_feature_drop_count,
                window_start_sec=self._buffer_time_window(self.visual_feature_ringbuf)[0],
                window_end_sec=self._buffer_time_window(self.visual_feature_ringbuf)[1],
                last_stamp_sec=self._buffer_time_window(self.visual_feature_ringbuf)[2],
            ),
            "odom": self._stream_stats(
                buffer_len=odom_len,
                buffer_max=1,
                drops_total=self.odom_drop_count,
                window_start_sec=odom_last,
                window_end_sec=odom_last,
                last_stamp_sec=odom_last,
            ),
        }
        return ScanIOCert(
            scan_seq=int(scan_seq),
            scan_stamp_sec=float(scan_stamp_sec),
            scan_window_start_sec=float(scan_window_start_sec),
            scan_window_end_sec=float(scan_window_end_sec),
            streams=streams,
        )

    def _record_jit_signature(
        self,
        *,
        points: jnp.ndarray,
        timestamps: jnp.ndarray,
        weights: jnp.ndarray,
        ring: jnp.ndarray,
        tag: jnp.ndarray,
        imu_stamps: np.ndarray,
        imu_gyro: np.ndarray,
        imu_accel: np.ndarray,
        camera_batch: MeasurementBatch,
    ) -> None:
        signature = (
            ("points", tuple(points.shape), str(points.dtype)),
            ("timestamps", tuple(timestamps.shape), str(timestamps.dtype)),
            ("weights", tuple(weights.shape), str(weights.dtype)),
            ("ring", tuple(ring.shape), str(ring.dtype)),
            ("tag", tuple(tag.shape), str(tag.dtype)),
            ("imu_stamps", tuple(imu_stamps.shape), str(imu_stamps.dtype)),
            ("imu_gyro", tuple(imu_gyro.shape), str(imu_gyro.dtype)),
            ("imu_accel", tuple(imu_accel.shape), str(imu_accel.dtype)),
            ("camera_Lambdas", tuple(camera_batch.Lambdas.shape), str(camera_batch.Lambdas.dtype)),
            ("camera_thetas", tuple(camera_batch.thetas.shape), str(camera_batch.thetas.dtype)),
            ("camera_etas", tuple(camera_batch.etas.shape), str(camera_batch.etas.dtype)),
            ("camera_weights", tuple(camera_batch.weights.shape), str(camera_batch.weights.dtype)),
            ("camera_valid", tuple(camera_batch.valid_mask.shape), str(camera_batch.valid_mask.dtype)),
            ("camera_timestamps", tuple(camera_batch.timestamps.shape), str(camera_batch.timestamps.dtype)),
            ("camera_colors", tuple(camera_batch.colors.shape), str(camera_batch.colors.dtype)),
        )
        if signature not in self._jit_signature_cache:
            self._jit_signature_cache.add(signature)
            record_jit_recompile(1)

    def _init_ros(self):
        """Initialize ROS interfaces."""
        p = self._params
        self.odom_frame = p["odom_frame"]
        self.base_frame = p["base_frame"]
        
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
        
        lidar_topic = p["lidar_topic"]
        odom_topic = p["odom_topic"]
        self.odom_topic = odom_topic
        imu_topic = p["imu_topic"]
        self.imu_topic = imu_topic
        
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

        # Camera (required): RGBD subscription for debug/visualization
        camera_rgbd_topic = p["camera_rgbd_topic"].strip()
        self.camera_rgbd_topic = camera_rgbd_topic
        if not camera_rgbd_topic:
            raise ValueError("camera_rgbd_topic is required but empty")
        self.sub_camera_rgbd = self.create_subscription(
            RGBDImage, camera_rgbd_topic, self._on_camera_rgbd, qos_sensor,
            callback_group=self.cb_group_sensors,
        )
        self.get_logger().info(f"Camera RGBD: {camera_rgbd_topic} (ring buffer len={self.camera_ringbuf_max})")

        # Visual feature batch (C++ preprocessing)
        visual_feature_topic = p["visual_feature_topic"].strip()
        self.visual_feature_topic = visual_feature_topic
        if not visual_feature_topic:
            raise ValueError("visual_feature_topic is required but empty")
        self.sub_visual_features = self.create_subscription(
            VisualFeatureBatch, visual_feature_topic, self._on_visual_features, qos_sensor,
            callback_group=self.cb_group_sensors,
        )
        self.get_logger().info(
            f"Visual features: {visual_feature_topic} (ring buffer len={self.visual_feature_ringbuf_max})"
        )
        
        # Publishers
        self.pub_state = self.create_publisher(Odometry, "/gc/state", 10)
        self.pub_path = self.create_publisher(Path, "/gc/trajectory", 10)
        self.pub_manifest = self.create_publisher(String, "/gc/runtime_manifest", 10)
        self.pub_cert = self.create_publisher(String, "/gc/certificate", 10)
        self.pub_status = self.create_publisher(String, "/gc/status", 10)

        # Publish queue drain timer (keeps ROS publish on the ROS thread).
        self._publish_timer = self.create_timer(
            self._publish_timer_period_sec,
            self._drain_publish_queue,
            callback_group=self.cb_group_sensors,
        )
        # Start async LiDAR worker thread.
        self._start_lidar_worker()

        # Rerun visualization (Wayland-friendly; replaces RViz)
        self.rerun_visualizer: Optional[RerunVisualizer] = None
        use_rerun = p["use_rerun"]
        rerun_recording_path = p["rerun_recording_path"].strip()
        rerun_spawn = p["rerun_spawn"]
        if use_rerun:
            if not rerun_spawn and not rerun_recording_path:
                self.get_logger().warning(
                    "use_rerun=true but rerun_spawn=false and rerun_recording_path empty; "
                    "no viewer or recording will be created."
                )
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
                raise RuntimeError(
                    "Rerun visualization requested but initialization failed. "
                    "Install rerun-sdk or disable use_rerun."
                )
        else:
            self.rerun_visualizer = None

        # PrimitiveMap publisher: /gc/map/points (PointCloud2)
        self.map_publisher: Optional[PrimitiveMapPublisher] = None
        if self.primitive_map is not None:
            self.map_publisher = PrimitiveMapPublisher(
                self,
                frame_id=self.odom_frame,
                publish_ellipsoids=False,
                eps_lift=float(self.config.eps_lift),
                eps_mass=float(self.config.eps_mass),
                rerun_visualizer=self.rerun_visualizer,
            )
            self.get_logger().info("PrimitiveMap publisher: /gc/map/points (PointCloud2)")
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Event log (append-only JSONL for replay)
        self.event_log_path = self._params["event_log_path"].strip()
        self._event_log_file = None
        if self.event_log_path:
            import os
            os.makedirs(os.path.dirname(self.event_log_path) or ".", exist_ok=True)
            self._event_log_file = open(self.event_log_path, "a", encoding="utf-8")
        
        # Trajectory export
        self.trajectory_export_path = self._params["trajectory_export_path"]
        self.trajectory_file = None
        if self.trajectory_export_path:
            self.trajectory_file = open(self.trajectory_export_path, "w")
            self.trajectory_file.write("# timestamp x y z qx qy qz qw\n")
        
        self.trajectory_poses: List[PoseStamped] = []
        self.max_path_length = 1000
        
        # Status timer
        status_period = float(self._params["status_check_period_sec"])
        self._status_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.status_timer = self.create_timer(
            status_period, self._publish_status, clock=self._status_clock
        )

    def _publish_runtime_manifest(self):
        """Publish RuntimeManifest at startup."""
        manifest = RuntimeManifest(
            eps_psd=float(self.config.eps_psd),
            eps_lift=float(self.config.eps_lift),
            eps_mass=float(self.config.eps_mass),
            alpha_min=float(self.config.alpha_min),
            alpha_max=float(self.config.alpha_max),
            kappa_scale=float(self.config.kappa_scale),
            c0_cond=float(self.config.c0_cond),
            c_dt=float(self.config.c_dt),
            c_ex=float(self.config.c_ex),
            c_frob=float(self.config.c_frob),
            imu_gravity_scale=float(self.config.imu_gravity_scale),
            deskew_rotation_only=bool(self.config.deskew_rotation_only),
            power_beta_min=float(self.config.power_beta_min),
            power_beta_exc_c=float(self.config.power_beta_exc_c),
            power_beta_z_c=float(self.config.power_beta_z_c),
            enable_parallel_stages=bool(self.config.enable_parallel_stages),
            ot_epsilon=float(self.config.ot_epsilon),
            ot_tau_a=float(self.config.ot_tau_a),
            ot_tau_b=float(self.config.ot_tau_b),
            ot_iters=int(self.config.k_sinkhorn),
            N_POINTS_CAP=int(self.config.N_POINTS_CAP),
            N_FEAT=int(self.config.n_feat),
            N_SURFEL=int(self.config.n_surfel),
            K_ASSOC=int(self.config.k_assoc),
            K_SINKHORN=int(self.config.k_sinkhorn),
            K_INSERT_TILE=int(self.config.k_insert_tile),
            K_MERGE_PAIRS_TILE=int(self.config.k_merge_pairs_tile),
            MERGE_MAX_TILE_SIZE=int(self.config.primitive_merge_max_tile_size),
            M_TILE=int(self.config.primitive_map_max_size),
            H_TILE=float(self.config.H_TILE),
            N_ACTIVE_TILES=int(self.config.N_ACTIVE_TILES),
            R_ACTIVE_TILES_XY=int(self.config.R_ACTIVE_TILES_XY),
            R_ACTIVE_TILES_Z=int(self.config.R_ACTIVE_TILES_Z),
            M_TILE_VIEW=int(self.config.M_TILE_VIEW),
            N_STENCIL_TILES=int(self.config.N_STENCIL_TILES),
            R_STENCIL_TILES_XY=int(self.config.R_STENCIL_TILES_XY),
            R_STENCIL_TILES_Z=int(self.config.R_STENCIL_TILES_Z),
            bev_backend_enabled=bool(self.bev_backend_enabled),
            bev_views_n=int(self.bev_views_n),
            pose_evidence_backend=self.pose_evidence_backend,
            map_backend=self.map_backend,
            topics={
                "lidar": self._params["lidar_topic"],
                "odom": self._params["odom_topic"],
                "imu": self._params["imu_topic"],
                "camera_rgbd": self._params["camera_rgbd_topic"],
                "visual_features": self._params["visual_feature_topic"],
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

    def _start_lidar_worker(self) -> None:
        """Start the async LiDAR worker thread (single-path)."""
        if self._lidar_worker is not None:
            return
        self._lidar_worker = threading.Thread(
            target=self._lidar_worker_loop,
            name="gc_lidar_worker",
            daemon=True,
        )
        self._lidar_worker.start()

    def _enqueue_lidar_msg(self, msg: PointCloud2) -> None:
        """Enqueue LiDAR scan for async processing (bounded queue)."""
        with self._lidar_cv:
            self._lidar_msg_count += 1
            if len(self._lidar_queue) >= self._lidar_queue.maxlen:
                # Drop oldest to keep newest (real-time, bounded).
                self._lidar_queue.popleft()
                self._lidar_drop_count += 1
                if self._lidar_drop_count % 50 == 0:
                    self.get_logger().warn(
                        f"LiDAR queue overflow: dropped {self._lidar_drop_count} scans "
                        f"(queue_len={self._lidar_queue.maxlen})"
                    )
            self._lidar_queue.append(msg)
            self._lidar_cv.notify()

    def _lidar_worker_loop(self) -> None:
        """Worker loop: process LiDAR scans asynchronously."""
        while True:
            with self._lidar_cv:
                while not self._lidar_queue and not self._lidar_worker_stop:
                    self._lidar_cv.wait(timeout=0.1)
                if self._lidar_worker_stop:
                    return
                msg = self._lidar_queue.popleft()
            # Process outside lock to avoid blocking enqueue.
            try:
                self._process_lidar_msg(msg)
            except Exception as exc:
                self._lidar_error_count += 1
                self.get_logger().error(f"LiDAR processing failed (count={self._lidar_error_count}): {exc}")

    def _drain_publish_queue(self) -> None:
        """Publish processed poses from the queue (ROS thread)."""
        if not self._publish_queue:
            return
        pose_6d, stamp = self._publish_queue.popleft()
        self._publish_state_from_pose(pose_6d, stamp)

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
        accel_scale = float(self._params["imu_accel_scale"])
        accel = accel_raw * accel_scale

        # No-TF mode: rotate IMU measurements into the base/body frame.
        # This is a numeric transform (not just frame_id relabeling).
        gyro = self.R_base_imu @ gyro
        accel = self.R_base_imu @ accel

        with self._buffer_lock:
            self.imu_buffer.append((stamp_sec, gyro, accel))

        if bool(self._params["use_imu_message_covariance"]):
            cov_g = np.array(msg.angular_velocity_covariance, dtype=np.float64).reshape(3, 3)
            cov_a = np.array(msg.linear_acceleration_covariance, dtype=np.float64).reshape(3, 3)
            if np.all(np.isfinite(cov_g)) and np.all(np.isfinite(cov_a)):
                if not np.allclose(cov_g, 0.0) and not np.allclose(cov_a, 0.0):
                    with self._buffer_lock:
                        self.last_imu_cov_g = cov_g
                        self.last_imu_cov_a = cov_a
        
        # Keep buffer bounded
        with self._buffer_lock:
            if len(self.imu_buffer) > self.max_imu_buffer:
                self.imu_buffer.pop(0)
                self.imu_drop_count += 1

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
        # Use actual pos.z from the message; trust is capped via odom_z_variance_prior below.
        # Anchor smoothing uses planar_z_ref as a reference height.
        odom_pose_absolute = se3_from_rotvec_trans(
            jnp.array(rotvec, dtype=jnp.float64),
            jnp.array([pos.x, pos.y, pos.z], dtype=jnp.float64),
        )
        
        with self._buffer_lock:
            imu_buffer_local = list(self.imu_buffer)
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
                        stamps, imu_buffer_local,
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
                    trans_mean[2] = float(self.config.planar_z_ref)
                    R_matrices = [Rotation.from_rotvec(np.array(p[3:6], dtype=np.float64)).as_matrix() for p in poses]
                    M = np.average(R_matrices, axis=0, weights=weights)
                    R_polar = _polar_so3(M)
                    rotvec_mean = Rotation.from_matrix(R_polar).as_rotvec()
                    A_smoothed = se3_from_rotvec_trans(
                        jnp.array(rotvec_mean, dtype=jnp.float64),
                        jnp.array(trans_mean, dtype=jnp.float64),
                    )
                    with self._state_lock:
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
            cov[2, 2] = max(cov[2, 2], float(self.odom_z_variance_prior))
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

            # Cache odom samples for time-aligned selection at scan time.
            self.odom_ringbuf.append(
                (stamp_sec, self.last_odom_pose, self.last_odom_cov_se3, self.last_odom_twist, self.last_odom_twist_cov)
            )
            if len(self.odom_ringbuf) > self.odom_ringbuf_max:
                self.odom_ringbuf.pop(0)

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
            with self._buffer_lock:
                self.camera_ringbuf.append((stamp_sec, rgb, depth))
                if len(self.camera_ringbuf) > self.camera_ringbuf_max:
                    self.camera_ringbuf.pop(0)
                    self.camera_drop_count += 1
        except Exception as e:
            self.get_logger().warn("Camera RGBD buffer failed: %s" % e)

    def _on_visual_features(self, msg: VisualFeatureBatch) -> None:
        """Store visual feature batch in ring buffer (C++ preprocessing output)."""
        stamp_sec = float(msg.stamp_sec) if msg.stamp_sec > 0.0 else (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )
        with self._buffer_lock:
            self.visual_feature_ringbuf.append((stamp_sec, msg))
            if len(self.visual_feature_ringbuf) > self.visual_feature_ringbuf_max:
                self.visual_feature_ringbuf.pop(0)
                self.visual_feature_drop_count += 1

    def _feature_batch_to_list(self, batch: VisualFeatureBatch) -> List[Feature3D]:
        """Convert VisualFeatureBatch to Feature3D list (camera frame)."""
        features: List[Feature3D] = []
        n = min(int(batch.count), len(batch.features))
        log_2pi = np.log(2.0 * np.pi)
        for i in range(n):
            f = batch.features[i]
            if not bool(f.valid):
                continue
            xyz = np.array(f.xyz, dtype=np.float64).reshape(3)
            cov = np.array(f.cov_xyz, dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(cov)):
                cov = np.eye(3, dtype=np.float64) * 1e6
            cov_reg = cov + float(self.config.eps_lift) * np.eye(3, dtype=np.float64)
            try:
                sign, logdet = np.linalg.slogdet(cov_reg)
                if sign <= 0:
                    cov_reg = cov + 1e-6 * np.eye(3, dtype=np.float64)
                    sign, logdet = np.linalg.slogdet(cov_reg)
                info = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                info = np.linalg.inv(cov_reg + 1e-3 * np.eye(3, dtype=np.float64))
                logdet = float(np.linalg.slogdet(cov_reg)[1])
            canonical_theta = info @ xyz
            canonical_log_partition = (
                0.5 * float(xyz @ info @ xyz) + 0.5 * float(logdet) + 1.5 * float(log_2pi)
            )
            mu_app = np.array(f.mu_app, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(mu_app)) or np.linalg.norm(mu_app) <= 0.0:
                mu_app_out = None
            else:
                mu_app_out = mu_app / (np.linalg.norm(mu_app) + 1e-12)
            meta = {
                "depth_sigma_c_sq": float(f.depth_sigma_c_sq),
                "depth_Lambda_c": float(f.depth_lambda_c),
                "depth_theta_c": float(f.depth_theta_c),
            }
            color = np.array(f.color, dtype=np.float64).reshape(3) if len(f.color) >= 3 else None
            features.append(
                Feature3D(
                    u=float(f.u),
                    v=float(f.v),
                    xyz=xyz,
                    cov_xyz=cov,
                    info_xyz=info,
                    logdet_cov=float(logdet),
                    canonical_theta=canonical_theta,
                    canonical_log_partition=canonical_log_partition,
                    desc=np.zeros((0,), dtype=np.uint8),
                    weight=float(f.weight),
                    meta=meta,
                    mu_app=mu_app_out,
                    kappa_app=float(f.kappa_app),
                    color=color,
                )
            )
        return features

    def on_lidar(self, msg: PointCloud2):
        """Enqueue LiDAR scan for async processing."""
        self._enqueue_lidar_msg(msg)

    def _process_lidar_msg(self, msg: PointCloud2):
        """
        Process LiDAR scan through the full GC pipeline.

        This is where the actual SLAM happens!
        """
        self.scan_count += 1
        self.get_logger().info(f"on_lidar callback #{self.scan_count} received")

        # Reset per-scan runtime counters (host/device transfer + sync estimates).
        reset_runtime_counters()

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
            record_device_to_host(points, syncs=1)
            pts_np = np.array(points)
            pts_base = (self.R_base_lidar @ pts_np.T).T + self.t_base_lidar[None, :]
            record_host_to_device(pts_base)
            points = jnp.array(pts_base, dtype=jnp.float64)
        n_points = points.shape[0]
        n_points_raw = int(n_points)

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

        # Snapshot shared buffers for async processing (avoid holding lock during compute).
        with self._buffer_lock:
            imu_buffer = list(self.imu_buffer)
            last_odom_pose = self.last_odom_pose
            last_odom_cov_se3 = self.last_odom_cov_se3
            last_odom_twist = self.last_odom_twist
            last_odom_twist_cov = self.last_odom_twist_cov
            last_odom_stamp = self.last_odom_stamp
            odom_ringbuf = list(self.odom_ringbuf)
            last_imu_cov_g = self.last_imu_cov_g
            last_imu_cov_a = self.last_imu_cov_a
            visual_feature_ringbuf = list(self.visual_feature_ringbuf)
            camera_ringbuf = list(self.camera_ringbuf)
        odom_twist_local = (
            last_odom_twist if last_odom_twist is not None else jnp.zeros((6,), dtype=jnp.float64)
        )
        odom_twist_cov_local = (
            last_odom_twist_cov
            if last_odom_twist_cov is not None
            else (1e6 * jnp.eye(6, dtype=jnp.float64))
        )

        if self.rerun_visualizer is not None and points.shape[0] > 0:
            record_device_to_host(points, syncs=1)
            self.rerun_visualizer.log_lidar(np.array(points, dtype=np.float64), t_scan)

        # Preserve previous scan time for scan-to-scan interval handling.
        # This must be captured BEFORE updating self.last_scan_stamp.
        t_prev_scan = float(self.last_scan_stamp) if self.last_scan_stamp > 0.0 else 0.0

        # =====================================================================
        # TIME SYNC ONCE AT START: materialize scan bounds and dt from device
        # All later time use is Python floats; no further device sync for time.
        # =====================================================================
        time_bounds = jnp.array([jnp.min(timestamps), jnp.max(timestamps)], dtype=jnp.float64)
        record_device_to_host(time_bounds, syncs=1)
        timestamps_min, timestamps_max = float(time_bounds[0]), float(time_bounds[1])
        if abs(timestamps_max - timestamps_min) < 1e-9:
            scan_start_time = timestamps_min
            scan_end_time = stamp_sec  # header.stamp
        else:
            scan_start_time = min(stamp_sec, timestamps_min)
            scan_end_time = max(stamp_sec, timestamps_max)
        dt_raw = (
            (t_scan - t_prev_scan)
            if t_prev_scan > 0.0
            else (scan_end_time - scan_start_time)
        )
        eps_dt = np.finfo(np.float64).eps
        dt_sec = float(np.sqrt(dt_raw**2 + eps_dt))
        self.last_scan_stamp = t_scan

        # Apply alignment to IMU buffer timestamps (continuous; no gates).
        if self._time_align_enabled and imu_buffer:
            imu_buffer = [
                (self._align_stamp(self.imu_topic, float(t)), g, a)
                for (t, g, a) in imu_buffer
            ]

        # Select closest odom sample to scan time (aligned).
        if odom_ringbuf:
            best_idx = 0
            best_dt = abs(self._align_stamp(self.odom_topic, float(odom_ringbuf[0][0])) - t_scan)
            for i in range(1, len(odom_ringbuf)):
                t_i = self._align_stamp(self.odom_topic, float(odom_ringbuf[i][0]))
                dt_i = abs(t_i - t_scan)
                if dt_i < best_dt:
                    best_dt = dt_i
                    best_idx = i
            odom_stamp, last_odom_pose, last_odom_cov_se3, last_odom_twist, last_odom_twist_cov = odom_ringbuf[best_idx]
            last_odom_stamp = self._align_stamp(self.odom_topic, float(odom_stamp))

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

        # Visual features are required when the visual feature topic is configured.
        if not visual_feature_ringbuf:
            raise RuntimeError(
                "Visual feature ring buffer is empty; visual features are required for GC v2. "
                "Ensure the C++ visual_feature_node is running and publishing /gc/sensors/visual_features."
            )
        else:
            best_idx = 0
            best_dt = abs(self._align_stamp(self.visual_feature_topic, float(visual_feature_ringbuf[0][0])) - t_scan)
            for i, (t_frame, _) in enumerate(visual_feature_ringbuf):
                dt_frame = abs(self._align_stamp(self.visual_feature_topic, float(t_frame)) - t_scan)
                if dt_frame < best_dt:
                    best_dt = dt_frame
                    best_idx = i
            stamp_feat, feat_batch = visual_feature_ringbuf[best_idx]
            if self.scan_count <= 5:
                self.get_logger().info(
                    f"Visual feature batch selected: idx={best_idx}/{len(visual_feature_ringbuf)}, "
                    f"t_frame={stamp_feat:.6f}, t_scan={t_scan:.6f}, dt={best_dt:.6f}s"
                )
            # Optional RGBD logging for Rerun (debug only)
            if self.rerun_visualizer is not None and camera_ringbuf:
                rb_idx = 0
                rb_dt = abs(self._align_stamp(self.camera_rgbd_topic, float(camera_ringbuf[0][0])) - t_scan)
                for i, (t_frame, _, _) in enumerate(camera_ringbuf):
                    dt_frame = abs(self._align_stamp(self.camera_rgbd_topic, float(t_frame)) - t_scan)
                    if dt_frame < rb_dt:
                        rb_dt = dt_frame
                        rb_idx = i
                stamp_rgb, rgb, depth = camera_ringbuf[rb_idx]
                self.rerun_visualizer.log_rgbd(rgb, depth, stamp_rgb)

            features = self._feature_batch_to_list(feat_batch)
            extraction_result = ExtractionResult(
                features=features,
                op_report=[],
                timestamp_ns=int(stamp_feat * 1e9),
            )

            # LiDAR points in base frame -> camera frame for lidar_depth_evidence
            record_device_to_host(points, syncs=1)
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
            # Camera features are in camera frame; convert to base frame so all
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
                stamp_feat,
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
            for (t, g, a) in imu_buffer
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
        record_host_to_device(imu_stamps)
        record_host_to_device(imu_gyro)
        record_host_to_device(imu_accel)
        self._record_jit_signature(
            points=points,
            timestamps=timestamps,
            weights=weights,
            ring=ring,
            tag=tag,
            imu_stamps=imu_stamps,
            imu_gyro=imu_gyro,
            imu_accel=imu_accel,
            camera_batch=camera_batch,
        )

        # Diagnostic: log IMU buffer state
        n_valid_imu = int(np.sum(imu_stamps > 0.0))
        if n_valid_imu > 0:
            valid_stamps = imu_stamps[imu_stamps > 0.0]
            t_imu_min = float(np.min(valid_stamps))
            t_imu_max = float(np.max(valid_stamps))
            self.get_logger().info(
                f"Scan #{self.scan_count} IMU buffer: {n_valid_imu}/{len(imu_buffer)} valid, "
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
            if last_odom_pose is not None:
                record_device_to_host(last_odom_pose, syncs=1)
                op = np.array(last_odom_pose, dtype=np.float64).ravel()[:6]
                _diag_odom_x, _diag_odom_y = float(op[0]), float(op[1])
                _diag_odom_yaw = _yaw_deg_from_pose_6d(op)
            record_device_to_host(odom_twist_local, syncs=1)
            ot = np.array(odom_twist_local, dtype=np.float64).ravel()[:6]
            _diag_odom_vx, _diag_odom_vy, _diag_odom_wz = float(ot[0]), float(ot[1]), float(ot[5])

        try:
            # Update per-scan measurement covariance from IW state (shared across hypotheses)
            self.config.Sigma_meas = measurement_noise_mean_jax(self.measurement_noise_state, idx=2)
            self.config.Sigma_g = measurement_noise_mean_jax(self.measurement_noise_state, idx=0)
            self.config.Sigma_a = measurement_noise_mean_jax(self.measurement_noise_state, idx=1)
            if bool(self._params["use_imu_message_covariance"]):
                if last_imu_cov_g is not None:
                    self.config.Sigma_g = jnp.array(last_imu_cov_g, dtype=jnp.float64)
                if last_imu_cov_a is not None:
                    self.config.Sigma_a = jnp.array(last_imu_cov_a, dtype=jnp.float64)

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
                    odom_pose=last_odom_pose if last_odom_pose is not None else se3_identity(),
                    odom_cov_se3=(
                        last_odom_cov_se3
                        if last_odom_cov_se3 is not None
                        else (1e12 * jnp.eye(6, dtype=jnp.float64))
                    ),
                    scan_start_time=scan_start_time,
                    scan_end_time=scan_end_time,
                    dt_sec=dt_sec,
                    t_last_scan=t_last_scan,  # IMU integration interval start
                    t_scan=t_scan,            # IMU integration interval end
                    Q=Q_scan,
                    config=self.config,
                    odom_twist=odom_twist_local,  # Phase 2: odom twist for velocity factors
                    odom_twist_cov=odom_twist_cov_local,
                    primitive_map=self.primitive_map,
                    camera_batch=camera_batch,
                    scan_seq=int(self.scan_count),
                )
                results.append(result)
                self.hypotheses[i] = result.belief_updated

                if i == 0 and self._event_log_file and result.event_log_entries:
                    import json
                    for entry in result.event_log_entries:
                        payload = {
                            "scan_time": float(scan_end_time),
                            "entry": entry,
                        }
                        self._event_log_file.write(json.dumps(payload) + "\n")
                    self._event_log_file.flush()

                # Stage 1: Update PrimitiveMap from first hypothesis
                # (For multi-hypothesis, we use hypothesis 0's map update; proper multi-hyp requires merging)
                if i == 0 and result.primitive_map_updated is not None:
                    with self._state_lock:
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

            scan_io_cert = self._build_scan_io_cert(
                scan_seq=self.scan_count,
                scan_stamp_sec=t_scan,
                scan_window_start_sec=scan_start_time,
                scan_window_end_sec=scan_end_time,
                n_lidar_points=n_points_raw,
            )

            # Store certificate
            if results:
                iw_process_cert = CertBundle.create_approx(
                    chart_id=combined_belief.chart_id,
                    anchor_id=combined_belief.anchor_id,
                    triggers=["ProcessNoiseIWUpdate"],
                    influence=InfluenceCert.identity().with_overrides(
                        psd_projection_delta=float(proc_iw_cert_vec[0]),
                        nu_projection_delta=float(proc_iw_cert_vec[1]),
                    ),
                )
                iw_meas_cert = CertBundle.create_approx(
                    chart_id=combined_belief.chart_id,
                    anchor_id=combined_belief.anchor_id,
                    triggers=["MeasurementNoiseIWUpdate"],
                    influence=InfluenceCert.identity().with_overrides(
                        psd_projection_delta=float(meas_iw_cert_vec[0]),
                        nu_projection_delta=float(meas_iw_cert_vec[1]),
                    ),
                )

            # Enqueue publish (drained by ROS timer; non-blocking).
            pose_6d = combined_belief.mean_world_pose()
            if len(self._publish_queue) >= self._publish_queue.maxlen:
                self._publish_queue.popleft()
                self._publish_drop_count += 1
            self._publish_queue.append((jnp.array(pose_6d), stamp_sec))

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

            # Diagnostics at end of callback (after publish and odom-belief); run when needed.
            if results:
                runtime_counts = consume_runtime_counters()
                device_runtime_cert = DeviceRuntimeCert(
                    host_sync_count_est=runtime_counts.host_sync_count_est,
                    device_to_host_bytes_est=runtime_counts.device_to_host_bytes_est,
                    host_to_device_bytes_est=runtime_counts.host_to_device_bytes_est,
                    jit_recompile_count=runtime_counts.jit_recompile_count,
                )
                results[0].aggregated_cert.compute.scan_io = scan_io_cert
                results[0].aggregated_cert.compute.device_runtime = device_runtime_cert
                self.cert_history.append(aggregate_certificates([results[0].aggregated_cert, iw_process_cert, iw_meas_cert]))
                if len(self.cert_history) > 100:
                    self.cert_history.pop(0)
                if results[0].diagnostics_tape is not None:
                    from dataclasses import replace
                    entry = replace(results[0].diagnostics_tape, scan_number=self.scan_count)
                    self.diagnostics_log.append_tape(entry)

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
        with self._state_lock:
            anchor_correction = self.anchor_correction
            primitive_map = self.primitive_map
            current_belief = self.current_belief
        pose_export = se3_compose(anchor_correction, pose_6d)
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
        if self.map_publisher is not None and primitive_map is not None:
            tile_ids = None
            if current_belief is not None:
                pose = current_belief.mean_world_pose(eps_lift=self.config.eps_lift)
                center_xyz = np.asarray(pose[:3], dtype=np.float64).reshape(3)
                tile_ids = ma_hex_stencil_tile_ids(
                    center_xyz=center_xyz,
                    h_tile=float(self.config.H_TILE),
                    radius_xy=int(self.config.R_ACTIVE_TILES_XY),
                    radius_z=int(self.config.R_ACTIVE_TILES_Z),
                )
            self.map_publisher.publish(primitive_map, stamp_sec, tile_ids=tile_ids)
        
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

        with self._state_lock:
            primitive_map = self.primitive_map
        prim_count = int(primitive_map.total_count) if primitive_map is not None else 0
        status = {
            "elapsed_sec": elapsed,
            "odom_count": self.odom_count,
            "scan_count": self.scan_count,
            "imu_count": self.imu_count,
            "lidar_msg_count": self._lidar_msg_count,
            "lidar_drop_count": self._lidar_drop_count,
            "lidar_error_count": self._lidar_error_count,
            "lidar_queue_len": len(self._lidar_queue),
            "lidar_queue_max": self._lidar_queue.maxlen,
            "publish_drop_count": self._publish_drop_count,
            "publish_queue_len": len(self._publish_queue),
            "publish_queue_max": self._publish_queue.maxlen,
            "pipeline_runs": self.pipeline_runs,
            "hypotheses": self.config.K_HYP,
            "primitive_map_count": prim_count,
            "time_align_enabled": bool(self._time_align_enabled),
            "time_align_reference": self._time_align_reference,
        }
        
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)
        
        self.get_logger().info(
            f"GC Status: odom={self.odom_count}, scans={self.scan_count}, "
            f"imu={self.imu_count}, pipeline={self.pipeline_runs}, "
            f"primitive_map={prim_count}, "
            f"lidar_q={len(self._lidar_queue)}/{self._lidar_queue.maxlen} drops={self._lidar_drop_count} errors={self._lidar_error_count}, "
            f"pub_q={len(self._publish_queue)}/{self._publish_queue.maxlen} drops={self._publish_drop_count}"
        )

    def destroy_node(self):
        """Clean up."""
        if self.rerun_visualizer is not None:
            self.rerun_visualizer.flush()
        # Stop LiDAR worker thread.
        with self._lidar_cv:
            self._lidar_worker_stop = True
            self._lidar_cv.notify_all()
        if self._lidar_worker is not None:
            self._lidar_worker.join(timeout=2.0)
        # Drain publish queue so last scan state is written
        while self._publish_queue:
            self._drain_publish_queue()
        if self.trajectory_file:
            self.trajectory_file.flush()
            self.trajectory_file.close()
            self.get_logger().info(f"Trajectory saved: {self.trajectory_export_path}")
        if self._event_log_file:
            self._event_log_file.flush()
            self._event_log_file.close()

        # Save diagnostics log for dashboard
        diagnostics_path = self._params["diagnostics_export_path"]
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
        splat_path = self._params["splat_export_path"].strip()
        if splat_path and self.primitive_map is not None and self.primitive_map.total_count > 0:
            try:
                import os
                # Multi-tile export: concatenate all tiles deterministically (sorted by tile_id).
                positions_list = []
                cov_list = []
                colors_list = []
                weights_list = []
                directions_list = []
                kappas_list = []
                timestamps_list = []
                created_list = []
                ids_list = []

                for tile_id in sorted(self.primitive_map.tile_ids):
                    tile = self.primitive_map.tiles.get(int(tile_id))
                    if tile is None or int(tile.count) == 0:
                        continue
                    view = extract_primitive_map_view(tile=tile)
                    if view.count == 0:
                        continue
                    positions_list.append(np.asarray(view.positions))
                    cov_list.append(np.asarray(view.covariances))
                    colors_list.append(np.asarray(view.colors) if view.colors is not None else np.zeros((view.count, 3), dtype=np.float64))
                    weights_list.append(np.asarray(view.weights))
                    directions_list.append(np.asarray(view.directions))
                    kappas_list.append(np.asarray(view.kappas))
                    ids_list.append(np.asarray(view.primitive_ids))
                    # Slot-local timestamps are direct (avoid ID lookups).
                    slot_idx = np.asarray(view.slot_indices, dtype=np.int64)
                    timestamps_list.append(np.asarray(tile.timestamps[slot_idx], dtype=np.float64))
                    created_list.append(np.asarray(tile.created_timestamps[slot_idx], dtype=np.float64))

                if not positions_list:
                    self.get_logger().info("Splat export skipped: no valid primitives in atlas tiles.")
                    positions = covariances = colors = weights = directions = kappas = timestamps = created_timestamps = view_ids = None
                    n = 0
                else:
                    positions = np.concatenate(positions_list, axis=0)
                    covariances = np.concatenate(cov_list, axis=0)
                    colors = np.concatenate(colors_list, axis=0)
                    weights = np.concatenate(weights_list, axis=0)
                    directions = np.concatenate(directions_list, axis=0)
                    kappas = np.concatenate(kappas_list, axis=0)
                    timestamps = np.concatenate(timestamps_list, axis=0)
                    created_timestamps = np.concatenate(created_list, axis=0)
                    view_ids = np.concatenate(ids_list, axis=0)
                    n = int(positions.shape[0])

                if n > 0:
                    os.makedirs(os.path.dirname(splat_path) or ".", exist_ok=True)
                    np.savez_compressed(
                        splat_path,
                        positions=positions,
                        covariances=covariances,
                        colors=colors,
                        weights=weights,
                        directions=directions,
                        kappas=kappas,
                        timestamps=timestamps,
                        created_timestamps=created_timestamps,
                        primitive_ids=view_ids,
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
