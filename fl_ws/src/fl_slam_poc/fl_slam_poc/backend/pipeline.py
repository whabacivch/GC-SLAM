"""
Golden Child SLAM v2 Pipeline.

Main per-scan execution following spec Section 7.
All steps run every time; influence may go to ~0 smoothly. No gates.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.common.belief import BeliefGaussianInfo, D_Z, HypothesisSet
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    aggregate_certificates,
    InfluenceCert,
)
from fl_slam_poc.backend.structures.bin_atlas import (
    BinAtlas,
    MapBinStats,
    create_fibonacci_atlas,
    compute_map_derived_stats,
    apply_forgetting,
)

# Import all operators
from fl_slam_poc.backend.operators.point_budget import (
    point_budget_resample,
    PointBudgetResult,
)
from fl_slam_poc.backend.operators.predict import (
    predict_diffusion,
)
from fl_slam_poc.backend.operators.imu_preintegration import (
    smooth_window_weights,
    preintegrate_imu_relative_pose_jax,
)
from fl_slam_poc.backend.operators.deskew_constant_twist import (
    deskew_constant_twist,
    DeskewConstantTwistResult,
)
from fl_slam_poc.backend.operators.binning import (
    bin_soft_assign,
    scan_bin_moment_match,
    create_bin_atlas,
    ScanBinStats,
)
from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2
from fl_slam_poc.backend.operators.matrix_fisher_evidence import (
    matrix_fisher_rotation_evidence,
    planar_translation_evidence,
    build_combined_lidar_evidence_22d,
)
# Legacy imports kept for IW residual computation (will be removed later)
from fl_slam_poc.backend.operators.measurement_noise_iw_jax import (
    lidar_meas_iw_suffstats_from_translation_residuals_jax,
    imu_gyro_meas_iw_suffstats_from_avg_rate_jax,
    imu_accel_meas_iw_suffstats_from_gravity_dir_jax,
)
from fl_slam_poc.backend.operators.odom_evidence import odom_quadratic_evidence
from fl_slam_poc.backend.operators.odom_twist_evidence import (
    odom_velocity_evidence,
    odom_yawrate_evidence,
    pose_twist_kinematic_consistency,
)
from fl_slam_poc.backend.operators.imu_evidence import (
    imu_vmf_gravity_evidence_time_resolved,
    TimeResolvedImuResult,
)
from fl_slam_poc.backend.operators.planar_prior import (
    planar_z_prior,
    velocity_z_prior,
)
from fl_slam_poc.backend.operators.imu_gyro_evidence import imu_gyro_rotation_evidence
from fl_slam_poc.backend.operators.imu_preintegration_factor import imu_preintegration_factor
from fl_slam_poc.backend.operators.lidar_bucket_noise_iw_jax import (
    lidar_point_reliability_from_buckets_jax,
    lidar_bucket_iw_suffstats_from_bin_residuals_jax,
)
from fl_slam_poc.backend.operators.fusion import (
    fusion_scale_from_certificates,
    info_fusion_additive,
)
from fl_slam_poc.backend.operators.excitation import (
    compute_excitation_scales_jax,
    apply_excitation_prior_scaling_jax,
)
from fl_slam_poc.backend.operators.recompose import pose_update_frobenius_recompose
from fl_slam_poc.backend.operators.inverse_wishart_jax import process_noise_iw_suffstats_from_info_jax
from fl_slam_poc.backend.operators.map_update import (
    pos_cov_inflation_pushforward,
    MapUpdateResult,
)
from fl_slam_poc.backend.operators.anchor_drift import anchor_drift_update
from fl_slam_poc.backend.operators.hypothesis import hypothesis_barycenter_projection
from fl_slam_poc.backend.diagnostics import ScanDiagnostics


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the Golden Child pipeline."""
    # Budgets (hard constants)
    K_HYP: int = constants.GC_K_HYP
    B_BINS: int = constants.GC_B_BINS
    N_POINTS_CAP: int = constants.GC_N_POINTS_CAP
    
    # Epsilon constants
    eps_psd: float = constants.GC_EPS_PSD
    eps_lift: float = constants.GC_EPS_LIFT
    eps_mass: float = constants.GC_EPS_MASS
    
    # Fusion parameters
    alpha_min: float = constants.GC_ALPHA_MIN
    alpha_max: float = constants.GC_ALPHA_MAX
    kappa_scale: float = constants.GC_KAPPA_SCALE
    c0_cond: float = constants.GC_C0_COND
    
    # Excitation coupling
    c_dt: float = constants.GC_C_DT
    c_ex: float = constants.GC_C_EX
    c_frob: float = constants.GC_C_FROB
    
    # Soft assign
    tau_soft_assign: float = constants.GC_TAU_SOFT_ASSIGN
    
    # Forgetting
    forgetting_factor: float = 0.99
    
    # Measurement noise
    Sigma_meas: jnp.ndarray = None
    Sigma_g: jnp.ndarray = None  # (3,3) gyro covariance proxy (from measurement IW)
    Sigma_a: jnp.ndarray = None  # (3,3) accel covariance proxy (from measurement IW)

    # Single IMU gravity scale: used for vMF gravity evidence and preintegration.
    # 1.0 = nominal (correct gravity cancellation: a_world = R@a_body + g_W -> 0 when level).
    # 0.0 = disable (ablation only).
    imu_gravity_scale: float = 1.0

    # Deskew translation control: if True, zero out translation in deskew twist (rotation-only).
    # This removes the hidden IMU translation leak through deskew → LiDAR evidence.
    # Use for A/B testing; proper fix is Gaussian twist marginalization.
    deskew_rotation_only: bool = False

    # No-TF extrinsics: LiDAR origin expressed in base frame. Used to compute
    # ray directions from the sensor origin (not the base origin).
    lidar_origin_base: jnp.ndarray = None  # (3,)

    # Planar robot constraints (Phase 1: z fix via soft prior)
    # For ground-hugging robots, these prevent z runaway feedback loop.
    planar_z_ref: float = constants.GC_PLANAR_Z_REF  # Reference z height (m)
    planar_z_sigma: float = constants.GC_PLANAR_Z_SIGMA  # Soft z constraint std dev (m)
    planar_vz_sigma: float = constants.GC_PLANAR_VZ_SIGMA  # Soft vel_z=0 std dev (m/s)
    enable_planar_prior: bool = True  # Enable/disable planar constraints
    # z_precision_scale REMOVED - now self-adaptive from map scatter eigenvalues

    # Odometry twist evidence (Phase 2: velocity factors)
    enable_odom_twist: bool = True  # Enable/disable odom twist evidence
    odom_twist_vel_sigma: float = constants.GC_ODOM_TWIST_VEL_SIGMA  # Velocity std dev (m/s)
    odom_twist_wz_sigma: float = constants.GC_ODOM_TWIST_WZ_SIGMA  # Yaw rate std dev (rad/s)

    def __post_init__(self):
        if self.Sigma_meas is None:
            # Default measurement noise (3x3 isotropic)
            self.Sigma_meas = 0.01 * jnp.eye(3, dtype=jnp.float64)
        if self.Sigma_g is None:
            self.Sigma_g = constants.GC_IMU_GYRO_NOISE_DENSITY * jnp.eye(3, dtype=jnp.float64)
        if self.Sigma_a is None:
            self.Sigma_a = constants.GC_IMU_ACCEL_NOISE_DENSITY * jnp.eye(3, dtype=jnp.float64)
        if self.lidar_origin_base is None:
            self.lidar_origin_base = jnp.zeros((3,), dtype=jnp.float64)


# =============================================================================
# Per-Scan Pipeline Result
# =============================================================================


@dataclass
class ScanPipelineResult:
    """Result of processing a single scan for one hypothesis."""
    belief_updated: BeliefGaussianInfo
    map_increments: MapUpdateResult  # Increments to map statistics
    # Process-noise IW sufficient statistics proposal (commutative; aggregated once per scan).
    iw_process_dPsi: jnp.ndarray  # (7, 6, 6) padded
    iw_process_dnu: jnp.ndarray   # (7,)
    # Measurement-noise IW sufficient statistics proposal (per-sensor; aggregated once per scan).
    iw_meas_dPsi: jnp.ndarray     # (3, 3, 3) [gyro, accel, lidar]
    iw_meas_dnu: jnp.ndarray      # (3,)
    # LiDAR bucket IW sufficient statistics proposal (per-(line,tag); aggregated once per scan).
    iw_lidar_bucket_dPsi: jnp.ndarray  # (K, 3, 3)
    iw_lidar_bucket_dnu: jnp.ndarray   # (K,)
    all_certs: List[CertBundle]
    aggregated_cert: CertBundle
    # Per-scan diagnostics for dashboard (Stage-0 schema)
    diagnostics: Optional[ScanDiagnostics] = None


# =============================================================================
# Main Pipeline Functions
# =============================================================================


def compute_imu_integration_time(
    imu_stamps: jnp.ndarray,
    t_start: float,
    t_end: float,
) -> float:
    """
    Compute dt_int = sum of IMU sample intervals in (t_start, t_end).
    
    This is the IMU integration time: dt_int = Σ_i (t_{i+1} - t_i) for all
    IMU samples where t_i ∈ (t_start, t_end).
    
    Absolute invariants (non-negotiable):
    - 0 ≤ dt_int ≤ (t_end - t_start)
    - dt_int resets every scan
    - dt_int is never cumulative
    
    Args:
        imu_stamps: (M,) array of IMU timestamps (may include zeros for padding)
        t_start: Start of integration interval (t_last_scan)
        t_end: End of integration interval (t_scan)
    
    Returns:
        dt_int: Sum of actual IMU sample intervals in the interval
    """
    import numpy as np
    imu_stamps_arr = np.asarray(imu_stamps, dtype=np.float64)
    t_start = float(t_start)
    t_end = float(t_end)
    
    # Filter to samples in (t_start, t_end) and exclude zero-padded entries
    eps = 1e-9
    mask = (imu_stamps_arr > t_start - eps) & (imu_stamps_arr <= t_end + eps) & (imu_stamps_arr > 0.0)
    valid_stamps = imu_stamps_arr[mask]
    
    if len(valid_stamps) < 2:
        return 0.0
    
    # Sort to ensure chronological order
    valid_stamps = np.sort(valid_stamps)
    
    # Compute dt_i = t_{i+1} - t_i for consecutive samples
    dt_intervals = valid_stamps[1:] - valid_stamps[:-1]
    dt_intervals = np.maximum(dt_intervals, 0.0)  # Ensure non-negative
    
    # dt_int = sum of all intervals
    dt_int = float(np.sum(dt_intervals))
    
    # Enforce invariant: 0 ≤ dt_int ≤ (t_end - t_start)
    dt_max = t_end - t_start
    dt_int = max(0.0, min(dt_int, dt_max))
    
    return dt_int


def process_scan_single_hypothesis(
    belief_prev: BeliefGaussianInfo,
    raw_points: jnp.ndarray,
    raw_timestamps: jnp.ndarray,
    raw_weights: jnp.ndarray,
    raw_ring: jnp.ndarray,
    raw_tag: jnp.ndarray,
    lidar_bucket_state,
    imu_stamps: jnp.ndarray,
    imu_gyro: jnp.ndarray,
    imu_accel: jnp.ndarray,
    odom_pose: jnp.ndarray,
    odom_cov_se3: jnp.ndarray,
    scan_start_time: float,
    scan_end_time: float,
    dt_sec: float,
    t_last_scan: float,  # NEW: Previous scan time for IMU interval
    t_scan: float,        # NEW: Current scan time for IMU interval
    Q: jnp.ndarray,
    bin_atlas: BinAtlas,
    map_stats: MapBinStats,
    config: PipelineConfig,
    odom_twist: jnp.ndarray,  # (6,) [vx,vy,vz,wx,wy,wz] in body frame (never None)
    odom_twist_cov: jnp.ndarray,  # (6,6) twist covariance (never None)
) -> ScanPipelineResult:
    """
    Process a single scan for one hypothesis.
    
    Follows the fixed-cost scan pipeline:
    1. PointBudgetResample
    2. PredictDiffusion
    3. DeskewConstantTwist
    4. BinSoftAssign
    5. ScanBinMomentMatch
    6. KappaFromResultant (map and scan)
    7. Matrix Fisher rotation + planar translation
    8. OdomEvidence + ImuEvidence + LidarEvidence (closed-form/Laplace; no moment-matching)
    9. FusionScaleFromCertificates
    10. InfoFusionAdditive
    11. PoseUpdateFrobeniusRecompose
    12. PoseCovInflationPushforward
    13. AnchorDriftUpdate
    
    All steps run every time. No gates.
    
    Args:
        belief_prev: Previous belief
        raw_points: Raw LiDAR points (N, 3)
        raw_timestamps: Per-point timestamps (N,)
        raw_weights: Per-point weights (N,)
        scan_start_time: Scan start timestamp
        scan_end_time: Scan end timestamp
        dt_sec: Time delta since last update
        Q: Process noise matrix (D_Z, D_Z)
        bin_atlas: Bin atlas for binning
        map_stats: Current map statistics
        config: Pipeline configuration
        
    Returns:
        ScanPipelineResult with updated belief and certificates
    """
    all_certs = []
    
    # =========================================================================
    # Step 1: PointBudgetResample
    # =========================================================================
    budget_result, budget_cert, budget_effect = point_budget_resample(
        points=raw_points,
        timestamps=raw_timestamps,
        weights=raw_weights,
        ring=raw_ring,
        tag=raw_tag,
        n_points_cap=config.N_POINTS_CAP,
        chart_id=belief_prev.chart_id,
        anchor_id=belief_prev.anchor_id,
    )
    all_certs.append(budget_cert)
    
    points = budget_result.points
    timestamps = budget_result.timestamps
    weights = budget_result.weights
    ring = budget_result.ring
    tag = budget_result.tag
    
    # =========================================================================
    # Step 2: PredictDiffusion
    # =========================================================================
    belief_pred, pred_cert, pred_effect = predict_diffusion(
        belief_prev=belief_prev,
        Q=Q,
        dt_sec=dt_sec,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
    )
    all_certs.append(pred_cert)
    
    # =========================================================================
    # Step 3: DeskewConstantTwist (IMU-derived constant twist)
    # =========================================================================
    # Time-warp width from dt uncertainty (continuous; no thresholding)
    _mu_pred, Sigma_pred, _lift = belief_pred.to_moments(eps_lift=config.eps_lift)
    dt_std = jnp.sqrt(Sigma_pred[15, 15])
    sigma_warp = float(jnp.maximum(dt_std, 0.01))
    # Two IMU membership windows:
    # - w_imu_scan: within-scan window for deskew (uses per-point timestamps)
    # - w_imu_int:  scan-to-scan window for gyro evidence / noise updates (uses header stamps)
    w_imu_scan = smooth_window_weights(
        imu_stamps=imu_stamps,
        scan_start_time=scan_start_time,
        scan_end_time=scan_end_time,
        sigma=sigma_warp,
    )
    w_imu_int = smooth_window_weights(
        imu_stamps=imu_stamps,
        scan_start_time=t_last_scan,
        scan_end_time=t_scan,
        sigma=sigma_warp,
    )

    # Biases from predicted belief increment
    mu_inc = belief_pred.mean_increment(eps_lift=config.eps_lift)
    gyro_bias = mu_inc[9:12]
    accel_bias = mu_inc[12:15]

    pose0 = belief_prev.mean_world_pose(eps_lift=config.eps_lift)
    rotvec0 = pose0[3:6]
    gravity_W = jnp.array(constants.GC_GRAVITY_W, dtype=jnp.float64) * float(config.imu_gravity_scale)

    # Preintegration for deskew (within-scan).
    delta_pose_scan, _R_end_scan, _p_end_scan, _v_end_scan, ess_imu_scan, _, _, _, _ = preintegrate_imu_relative_pose_jax(
        imu_stamps=imu_stamps,
        imu_gyro=imu_gyro,
        imu_accel=imu_accel,
        weights=w_imu_scan,
        rotvec_start_WB=rotvec0,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        gravity_W=gravity_W,
    )
    xi_body = se3_jax.se3_log(delta_pose_scan)  # (6,) twist over scan (used for deskew)
    # Rotation-only deskew: zero out translation component [rho, phi] -> [0, phi]
    # This removes the hidden IMU translation leak through deskew → LiDAR evidence.
    # Branch-free: uses continuous scaling (0.0 vs 1.0) instead of if-statement.
    trans_scale = jnp.where(config.deskew_rotation_only, 0.0, 1.0)
    xi_body = xi_body.at[:3].set(xi_body[:3] * trans_scale)

    # =====================================================================
    # IMU INTEGRATION TIME: dt_int = Σ_i Δt_i over (t_last_scan, t_scan)
    # =====================================================================
    # CRITICAL: IMU is path-integral data, not a snapshot
    # dt_int is the sum of actual IMU sample intervals in the integration window
    # Absolute invariants: 0 ≤ dt_int ≤ dt_sec
    dt_int = compute_imu_integration_time(
        imu_stamps=imu_stamps,
        t_start=t_last_scan,
        t_end=t_scan,
    )

    # Preintegration for scan-to-scan interval (used for gyro evidence + IMU noise updates).
    (
        delta_pose_int,
        _R_end_int,
        delta_p_int,
        delta_v_int,
        ess_imu_int,
        imu_a_body_mean,
        imu_a_world_nog_mean,
        imu_a_world_mean,
        imu_dt_eff_sum,
    ) = preintegrate_imu_relative_pose_jax(
        imu_stamps=imu_stamps,
        imu_gyro=imu_gyro,
        imu_accel=imu_accel,
        weights=w_imu_int,
        rotvec_start_WB=rotvec0,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        gravity_W=gravity_W,
    )
    
    # IMU measurement-noise IW sufficient stats (Σg, Σa) from commutative residuals
    # Average IMU sampling period for PSD mapping (Var * dt -> PSD).
    # NOTE: imu_stamps is padded with zeros; exclude padded entries by construction.
    import numpy as np
    imu_stamps_np = np.asarray(imu_stamps, dtype=np.float64).reshape(-1)
    valid_mask_np = imu_stamps_np > 0.0
    n_valid = int(np.sum(valid_mask_np))
    if n_valid >= 2:
        valid_stamps_raw = imu_stamps_np[valid_mask_np]
        dt_valid = np.diff(valid_stamps_raw)
        imu_dt_valid_min = float(np.min(dt_valid))
        imu_dt_valid_max = float(np.max(dt_valid))
        imu_dt_valid_mean = float(np.mean(dt_valid))
        imu_dt_valid_median = float(np.median(dt_valid))
        imu_dt_valid_std = float(np.std(dt_valid, ddof=1)) if dt_valid.size > 1 else 0.0
        imu_dt_valid_nonpos = int(np.sum(dt_valid <= 0.0))
    else:
        imu_dt_valid_min = 0.0
        imu_dt_valid_max = 0.0
        imu_dt_valid_mean = 0.0
        imu_dt_valid_median = 0.0
        imu_dt_valid_std = 0.0
        imu_dt_valid_nonpos = 0

    dt_full = np.concatenate([imu_stamps_np[1:] - imu_stamps_np[:-1], np.zeros((1,), dtype=np.float64)], axis=0)
    dt_full = np.maximum(dt_full, 0.0)
    w_imu_int_np = np.asarray(w_imu_int, dtype=np.float64).reshape(-1)
    w_floor = float(constants.GC_WEIGHT_FLOOR)
    denom = max(1.0 - w_floor, 1e-12)
    w_raw = (w_imu_int_np - w_floor) / denom
    w_raw = np.maximum(w_raw, 0.0)
    w_raw = w_raw * valid_mask_np.astype(np.float64)
    w_sum = float(np.sum(w_raw))
    if w_sum > 0.0:
        imu_dt_weighted_mean = float(np.sum(w_raw * dt_full) / w_sum)
        imu_dt_weighted_std = float(np.sqrt(np.sum(w_raw * (dt_full - imu_dt_weighted_mean) ** 2) / w_sum))
        imu_dt_weighted_sum = float(np.sum(w_raw * dt_full))
    else:
        imu_dt_weighted_mean = 0.0
        imu_dt_weighted_std = 0.0
        imu_dt_weighted_sum = 0.0

    if n_valid >= 2:
        valid = np.sort(imu_stamps_np[valid_mask_np])
        dt_imu = float((valid[-1] - valid[0]) / max(n_valid - 1, 1))
    else:
        dt_imu = 0.0
    dt_imu = jnp.maximum(jnp.array(dt_imu, dtype=jnp.float64), 1e-12)

    # CRITICAL: omega_avg must be an angular-rate proxy (rad/s), not a finite-rotation / dt surrogate.
    # Using so3_log(delta_R) / dt is only valid in the small-angle limit and can destabilize IW updates.
    # Use the weighted mean of the debiased gyro measurements over the integration window instead.
    imu_stamps_j = jnp.asarray(imu_stamps, dtype=jnp.float64).reshape(-1)
    valid_mask = (imu_stamps_j > 0.0).astype(jnp.float64)
    w_imu_int_valid = w_imu_int * valid_mask
    w_sum_imu_int = jnp.sum(w_imu_int_valid) + config.eps_mass
    w_norm_imu_int = w_imu_int_valid / w_sum_imu_int
    omega_avg = jnp.einsum("m,mi->i", w_norm_imu_int, (imu_gyro - gyro_bias[None, :]))
    
    # Diagnostic: omega_avg sanity (no heuristics / no gating).
    _omega_avg_np = np.array(omega_avg)
    if not np.all(np.isfinite(_omega_avg_np)):
        raise ValueError(f"omega_avg contains non-finite values: {_omega_avg_np}")
    _omega_avg_norm = float(np.linalg.norm(_omega_avg_np))
    
    iw_meas_gyro_dPsi, iw_meas_gyro_dnu = imu_gyro_meas_iw_suffstats_from_avg_rate_jax(
        imu_gyro=imu_gyro,
        weights=w_imu_int_valid,
        gyro_bias=gyro_bias,
        omega_avg=omega_avg,
        dt_imu_sec=dt_imu,
        eps_mass=config.eps_mass,
    )
    iw_meas_accel_dPsi, iw_meas_accel_dnu = imu_accel_meas_iw_suffstats_from_gravity_dir_jax(
        rotvec_world_body=rotvec0,
        imu_accel=imu_accel,
        weights=w_imu_int_valid,
        accel_bias=accel_bias,
        gravity_W=gravity_W,
        dt_imu_sec=dt_imu,
        eps_mass=config.eps_mass,
    )

    deskew_twist_result, deskew_cert, deskew_effect = deskew_constant_twist(
        points=points,
        timestamps=timestamps,
        weights=weights,
        scan_start_time=scan_start_time,
        scan_end_time=scan_end_time,
        xi_body=xi_body,
        ess_imu=float(ess_imu_scan),
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(deskew_cert)

    deskewed_points = deskew_twist_result.points
    deskewed_weights = deskew_twist_result.weights
    # Point covariances are not used by the new deskew. Keep zeros (deterministic).
    deskewed_covs = jnp.zeros((deskewed_points.shape[0], 3, 3), dtype=jnp.float64)
    
    # Compute point directions for binning from the LiDAR origin (not base origin).
    # points are in base frame; subtract sensor origin to recover true ray directions.
    rays = deskewed_points - config.lidar_origin_base[None, :]
    norms = jnp.linalg.norm(rays, axis=1, keepdims=True)
    point_directions = rays / (norms + config.eps_mass)
    
    # =========================================================================
    # Step 4: BinSoftAssign
    # =========================================================================
    assign_result, assign_cert, assign_effect = bin_soft_assign(
        point_directions=point_directions,
        bin_directions=bin_atlas.dirs,
        tau=config.tau_soft_assign,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(assign_cert)
    responsibilities = assign_result.responsibilities
    
    # =========================================================================
    # Step 5: ScanBinMomentMatch
    # =========================================================================
    point_lambda = lidar_point_reliability_from_buckets_jax(lidar_bucket_state, ring=ring, tag=tag)
    scan_bins, scan_cert, scan_effect = scan_bin_moment_match(
        points=deskewed_points,
        point_covariances=deskewed_covs,
        weights=deskewed_weights,
        responsibilities=responsibilities,
        point_lambda=point_lambda,
        direction_origin=config.lidar_origin_base,
        eps_psd=config.eps_psd,
        eps_mass=config.eps_mass,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(scan_cert)
    
    # =========================================================================
    # Step 6: KappaFromResultant (map and scan) - already done in scan_bin_moment_match
    # =========================================================================
    # Compute map derived stats including kappa
    mu_map, kappa_map, c_map, Sigma_c_map = compute_map_derived_stats(
        map_stats=map_stats,
        eps_mass=config.eps_mass,
        eps_psd=config.eps_psd,
    )
    
    # Compute scan mean directions (batched, no per-bin host sync)
    s_norms = jnp.linalg.norm(scan_bins.s_dir, axis=1, keepdims=True)
    mu_scan = scan_bins.s_dir / (s_norms + config.eps_mass)

    # =========================================================================
    # Step 7: Matrix Fisher Rotation Evidence (replaces WahbaSVD)
    # =========================================================================
    # Uses full scatter tensors for principled rotation estimation with
    # Fisher-derived information matrix. No heuristic kappa weighting.
    mf_result, mf_cert, mf_effect = matrix_fisher_rotation_evidence(
        belief_pred=belief_pred,
        scan_s_dir=scan_bins.s_dir,
        scan_S_dir_scatter=scan_bins.S_dir_scatter,
        scan_N=scan_bins.N,
        map_S_dir=map_stats.S_dir,
        map_S_dir_scatter=map_stats.S_dir_scatter,
        map_N_dir=map_stats.N_dir,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        eps_mass=config.eps_mass,
    )
    all_certs.append(mf_cert)
    R_hat = mf_result.R_mf  # ML rotation from Matrix Fisher

    # =========================================================================
    # Step 8: Planar Translation Evidence (replaces TranslationWLS)
    # =========================================================================
    # Uses WLS translation but scales down z information for planar robots.
    # The planar prior handles z estimation, not scan matching.
    planar_trans_result, planar_trans_cert, planar_trans_effect = planar_translation_evidence(
        belief_pred=belief_pred,
        scan_p_bar=scan_bins.p_bar,
        scan_Sigma_p=scan_bins.Sigma_p,
        scan_N=scan_bins.N,
        map_centroid=c_map,
        map_Sigma_c=Sigma_c_map,
        map_N_pos=map_stats.N_pos,
        map_S_dir_scatter=map_stats.S_dir_scatter,  # For self-adaptive z precision
        map_N_dir=map_stats.N_dir,  # For normalizing scatter
        R_hat=R_hat,
        # z_precision_scale is now SELF-ADAPTIVE from map scatter eigenvalues
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        eps_mass=config.eps_mass,
    )
    all_certs.append(planar_trans_cert)
    t_hat = planar_trans_result.t_wls

    # Compute weights for IW stats (legacy - uses geometric mean of masses)
    mf_weights = jnp.sqrt(scan_bins.N * map_stats.N_dir + config.eps_mass)

    # Measurement-noise IW sufficient stats from translation residuals (LiDAR meas block)
    # residual_b = c_map[b] - R_hat @ p_bar_scan[b] - t_hat
    p_rot = (R_hat @ scan_bins.p_bar.T).T  # (B,3)
    residuals_t = c_map - p_rot - t_hat[None, :]
    iw_meas_dPsi, iw_meas_dnu = lidar_meas_iw_suffstats_from_translation_residuals_jax(
        residuals=residuals_t,
        weights=mf_weights,
        eps_mass=config.eps_mass,
    )
    iw_meas_dPsi = iw_meas_dPsi + iw_meas_gyro_dPsi + iw_meas_accel_dPsi
    iw_meas_dnu = iw_meas_dnu + iw_meas_gyro_dnu + iw_meas_accel_dnu

    # LiDAR bucket IW sufficient stats from translation residuals apportioned by responsibilities.
    iw_lidar_bucket_dPsi, iw_lidar_bucket_dnu = lidar_bucket_iw_suffstats_from_bin_residuals_jax(
        residuals_bin=residuals_t,
        responsibilities=responsibilities,
        weights=deskewed_weights,
        ring=ring,
        tag=tag,
        eps_mass=config.eps_mass,
    )
    
    # =========================================================================
    # Step 9: Odom + IMU + LiDAR evidence (no moment matching)
    # =========================================================================
    
    # Odom evidence (Gaussian)
    odom_pose = jnp.asarray(odom_pose, dtype=jnp.float64).reshape(-1)
    odom_cov_se3 = jnp.asarray(odom_cov_se3, dtype=jnp.float64)
    odom_result, odom_cert, odom_effect = odom_quadratic_evidence(
        belief_pred_pose=belief_pred.mean_world_pose(eps_lift=config.eps_lift),
        odom_pose=odom_pose,
        odom_cov_se3=odom_cov_se3,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(odom_cert)

    # IMU accel direction evidence (time-resolved vMF with transport consistency weighting)
    # Per-sample reliability is computed from gyro-accel transport: e_k = df/dt + ω × f
    # Samples with small e_k (gravity-dominant) get high weight, linear acceleration downweighted.
    # sigma is self-adaptive from data (MAD-based), no manual knobs.
    pose_pred = belief_pred.mean_world_pose(eps_lift=config.eps_lift)
    imu_result, imu_cert, imu_effect = imu_vmf_gravity_evidence_time_resolved(
        rotvec_world_body=pose_pred[3:6],
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,  # For transport consistency computation
        weights=w_imu_int,
        accel_bias=accel_bias,
        gravity_W=gravity_W,
        dt_imu=dt_imu,  # Time step between IMU samples
        eps_psd=config.eps_psd,
        eps_mass=config.eps_mass,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(imu_cert)

    # IMU gyro rotation evidence (Gaussian on SO(3) using preintegrated delta rotation)
    # CRITICAL: Pass dt_int (IMU integration time), not dt_scan (LiDAR scan duration)
    Sigma_g = jnp.asarray(config.Sigma_g, dtype=jnp.float64)
    
    # Diagnostic: check inputs for NaN before gyro evidence
    _rotvec0_np = np.array(rotvec0)
    _pose_pred_np = np.array(pose_pred)
    _delta_pose_int_np = np.array(delta_pose_int)
    if not np.all(np.isfinite(_rotvec0_np)):
        raise ValueError(f"rotvec0 contains NaN: {_rotvec0_np}")
    if not np.all(np.isfinite(_pose_pred_np)):
        raise ValueError(f"pose_pred contains NaN: {_pose_pred_np}")
    if not np.all(np.isfinite(_delta_pose_int_np)):
        raise ValueError(f"delta_pose_int contains NaN: {_delta_pose_int_np}")
    
    gyro_result, gyro_cert, gyro_effect = imu_gyro_rotation_evidence(
        rotvec_start_WB=rotvec0,
        rotvec_end_pred_WB=pose_pred[3:6],
        delta_rotvec_meas=delta_pose_int[3:6],
        Sigma_g=Sigma_g,
        dt_int=dt_int,  # CORRECT: IMU integration time, not scan duration
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(gyro_cert)

    # =====================================================================
    # INVARIANT TEST: Compare yaw increments from three sources
    # =====================================================================
    # This diagnostic compares the yaw direction from three sources:
    # 1. Gyro-integrated: R_start @ Exp(delta_rotvec_meas)
    # 2. Odom: odom_pose rotation
    # 3. Matrix Fisher (LiDAR): R_hat
    # If they have opposite signs, we have a sign convention mismatch.
    R_start_mat = se3_jax.so3_exp(rotvec0)
    R_delta_gyro = se3_jax.so3_exp(delta_pose_int[3:6])
    R_gyro_end = R_start_mat @ R_delta_gyro  # Gyro-predicted end orientation

    # Extract yaw (rotation about Z) using atan2
    def _yaw_from_R(R):
        return float(jnp.arctan2(R[1, 0], R[0, 0]))

    yaw_start = np.degrees(_yaw_from_R(R_start_mat))
    yaw_gyro = np.degrees(_yaw_from_R(R_gyro_end))
    yaw_mf = np.degrees(_yaw_from_R(R_hat))  # Matrix Fisher rotation

    # Compute odom yaw (need to convert odom_pose rotation to matrix)
    R_odom_mat = se3_jax.so3_exp(odom_pose[3:6])
    yaw_odom = np.degrees(_yaw_from_R(R_odom_mat))

    # Compute yaw increments (handle wraparound)
    dyaw_gyro = yaw_gyro - yaw_start
    dyaw_odom = yaw_odom - yaw_start
    dyaw_mf = yaw_mf - yaw_start

    # Normalize to [-180, 180]
    def normalize_angle(angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    dyaw_gyro = normalize_angle(dyaw_gyro)
    dyaw_odom = normalize_angle(dyaw_odom)
    dyaw_mf = normalize_angle(dyaw_mf)

    # Log only first 10 scans for brevity
    if hasattr(config, '_scan_count'):
        config._scan_count += 1
    else:
        config._scan_count = 1
    if config._scan_count <= 10:
        sign_match_gyro_mf = 'YES' if dyaw_gyro * dyaw_mf > 0 else 'NO'
        sign_match_gyro_odom = 'YES' if dyaw_gyro * dyaw_odom > 0 else 'NO'
        sign_match_odom_mf = 'YES' if dyaw_odom * dyaw_mf > 0 else 'NO'
        print(f"[INVARIANT] Scan {config._scan_count}: "
              f"yaw_start={yaw_start:+7.2f}°, "
              f"Δyaw_gyro={dyaw_gyro:+7.2f}°, "
              f"Δyaw_odom={dyaw_odom:+7.2f}°, "
              f"Δyaw_mf={dyaw_mf:+7.2f}°, "
              f"gyro↔mf={sign_match_gyro_mf}, "
              f"gyro↔odom={sign_match_gyro_odom}, "
              f"odom↔mf={sign_match_odom_mf}")

    # IMU preintegration factor for velocity and position evidence
    # Get velocities from state (indices 6:9)
    mu_prev = belief_prev.mean_increment(eps_lift=config.eps_lift)
    v_start_world = mu_prev[6:9]  # Previous velocity in world frame
    v_pred_world = mu_inc[6:9]    # Predicted velocity in world frame (mu_inc already computed above)
    Sigma_a = jnp.asarray(config.Sigma_a, dtype=jnp.float64)

    preint_result, preint_cert, preint_effect = imu_preintegration_factor(
        p_start_world=pose0[0:3],
        rotvec_start_WB=rotvec0,
        v_start_world=v_start_world,
        p_end_pred_world=pose_pred[0:3],
        v_end_pred_world=v_pred_world,
        delta_v_body=delta_v_int,
        delta_p_body=delta_p_int,
        Sigma_a=Sigma_a,
        dt_int=dt_int,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(preint_cert)

    # Build combined LiDAR evidence from Matrix Fisher rotation + Planar Translation
    # This replaces the old lidar_quadratic_evidence which used Wahba + TranslationWLS
    L_lidar, h_lidar = build_combined_lidar_evidence_22d(
        mf_result=mf_result,
        trans_result=planar_trans_result,
    )

    # Combine evidence terms additively before fusion scaling
    L_evidence = (L_lidar + odom_result.L_odom + imu_result.L_imu +
                  gyro_result.L_gyro + preint_result.L_imu_preint)
    h_evidence = (h_lidar + odom_result.h_odom + imu_result.h_imu +
                  gyro_result.h_gyro + preint_result.h_imu_preint)

    # =========================================================================
    # Phase 1: Planar Z Prior (soft z = z_ref constraint)
    # =========================================================================
    # This prevents z runaway by adding soft constraints:
    # 1. z ≈ z_ref (robot height)
    # 2. vel_z ≈ 0 (robot doesn't fly)
    # NO GATE: Always runs. Influence controlled by sigma parameters.
    planar_result, planar_cert, _ = planar_z_prior(
        belief_pred_pose=pose_pred,
        z_ref=config.planar_z_ref,
        sigma_z=config.planar_z_sigma,
        eps_psd=config.eps_psd,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(planar_cert)

    # vel_z prior: v_z should be ~0 for ground robot
    vz_pred = float(mu_inc[8])  # z velocity from state (index 8)
    vz_result, vz_cert, _ = velocity_z_prior(
        v_z_pred=vz_pred,
        sigma_vz=config.planar_vz_sigma,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(vz_cert)

    # Add planar priors to evidence
    L_evidence = L_evidence + planar_result.L_planar + vz_result.L_vz
    h_evidence = h_evidence + planar_result.h_planar + vz_result.h_vz

    # =========================================================================
    # Phase 2: Odom Twist Evidence (velocity factors from wheel odometry)
    # =========================================================================
    # This adds kinematic coupling from wheel odometry twist that was previously unused.
    # NO GATE: Always runs. Covariance controls influence (huge cov = negligible precision).

    # Get predicted velocity in world frame
    v_pred_world = mu_inc[6:9]  # velocity block from state

    # Get rotation matrix from predicted pose
    R_world_body = se3_jax.so3_exp(pose_pred[3:6])

    # odom_twist is always initialized (never None). Covariance controls influence.
    # When no odom has been received, cov is huge (1e12) -> precision is negligible.
    v_odom_body = odom_twist[0:3]
    omega_z_odom = float(odom_twist[5])
    Sigma_v = odom_twist_cov[0:3, 0:3]  # Linear velocity covariance

    odom_vel_result, odom_vel_cert, _ = odom_velocity_evidence(
        v_pred_world=v_pred_world,
        R_world_body=R_world_body,
        v_odom_body=v_odom_body,
        Sigma_v=Sigma_v,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(odom_vel_cert)

    # Add odom twist evidence
    L_evidence = L_evidence + odom_vel_result.L_vel
    h_evidence = h_evidence + odom_vel_result.h_vel

    # Yaw rate evidence: compare odom yaw rate to gyro-derived yaw rate
    # Predicted yaw rate from gyro (z component of omega_avg, already computed above)
    omega_z_pred = float(omega_avg[2])

    # Use yaw rate covariance from odom_twist_cov[5,5] to derive sigma
    # sqrt(cov) = sigma. When cov is huge (1e12), sigma is huge (1e6) -> negligible precision.
    sigma_wz_from_cov = jnp.sqrt(jnp.maximum(odom_twist_cov[5, 5], 1e-12))
    sigma_wz_effective = float(sigma_wz_from_cov)

    odom_wz_result, odom_wz_cert, _ = odom_yawrate_evidence(
        omega_z_pred=omega_z_pred,
        omega_z_odom=omega_z_odom,
        sigma_wz=sigma_wz_effective,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(odom_wz_cert)

    L_evidence = L_evidence + odom_wz_result.L_wz
    h_evidence = h_evidence + odom_wz_result.h_wz

    # =========================================================================
    # Phase 2b: Pose-Twist Kinematic Consistency Factor (6.1.2 #3)
    # =========================================================================
    # Enforces: Log(X_prev^{-1} @ X_curr) ≈ [R_prev @ v_body * dt; omega_body * dt]
    # This directly couples pose change to twist, fixing "no dynamic linkage" issue
    # Extract velocity covariance from twist covariance (upper-left 3x3 block)
    Sigma_v_odom = odom_twist_cov[0:3, 0:3]
    # Extract angular velocity covariance (lower-right 3x3 block)
    Sigma_omega_odom = odom_twist_cov[3:6, 3:6]

    kinematic_result, kinematic_cert, _ = pose_twist_kinematic_consistency(
        pose_prev=pose0,  # Previous pose from belief_prev
        pose_curr=pose_pred,  # Current (predicted) pose
        v_body=odom_twist[0:3],  # Body linear velocity
        omega_body=odom_twist[3:6],  # Body angular velocity
        dt=dt_sec,  # Time between scans
        Sigma_v=Sigma_v_odom,
        Sigma_omega=Sigma_omega_odom,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(kinematic_cert)

    L_evidence = L_evidence + kinematic_result.L_consistency
    h_evidence = h_evidence + kinematic_result.h_consistency

    # Diagnostic: check which evidence component has NaN
    for name, L in [("L_lidar", L_lidar), ("L_odom", odom_result.L_odom),
                    ("L_imu", imu_result.L_imu), ("L_gyro", gyro_result.L_gyro),
                    ("L_imu_preint", preint_result.L_imu_preint)]:
        L_np = np.array(L)
        if not np.all(np.isfinite(L_np)):
            nan_pos = np.argwhere(~np.isfinite(L_np))[:5]
            raise ValueError(f"{name} contains NaN at positions {nan_pos.tolist()}")

    # Fisher-derived excitation scaling (Contract 1): scale dt/ex prior strength by (1 - s)
    s_dt, s_ex = compute_excitation_scales_jax(L_evidence=L_evidence, L_prior=belief_pred.L)
    L_prior_scaled, h_prior_scaled = apply_excitation_prior_scaling_jax(
        L_prior=belief_pred.L,
        h_prior=belief_pred.h,
        s_dt=s_dt,
        s_ex=s_ex,
    )
    exc_dt_scale = float(1.0 - s_dt)
    exc_ex_scale = float(1.0 - s_ex)
    exc_cert = CertBundle.create_approx(
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
        triggers=["ExcitationPriorScaling"],
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=exc_dt_scale,
            extrinsic_scale=exc_ex_scale,
            trust_alpha=1.0,
        ),
    )
    all_certs.append(exc_cert)
    belief_pred = BeliefGaussianInfo(
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
        X_anchor=belief_pred.X_anchor,
        stamp_sec=belief_pred.stamp_sec,
        z_lin=belief_pred.z_lin,
        L=L_prior_scaled,
        h=h_prior_scaled,
        cert=belief_pred.cert,
    )
    
    # =========================================================================
    # Step 10: FusionScaleFromCertificates
    # =========================================================================
    # Use a combined certificate for fusion scale (single-path, includes IMU+odom+lidar).
    evidence_cert = aggregate_certificates([deskew_cert, assign_cert, scan_cert, mf_cert, planar_trans_cert])
    combined_evidence_cert = aggregate_certificates([evidence_cert, odom_cert, imu_cert, gyro_cert])

    # Effective conditioning for trust alpha should be evaluated on the subspace we intend to update.
    # Using the full 22x22 conditioning can be dominated by physically-null directions (e.g., yaw under gravity,
    # weak bias/extrinsic blocks), pinning alpha at floor even when pose evidence is strong.
    import numpy as np

    eps_cond = float(config.eps_psd)
    L_pose = np.array(L_evidence[0:6, 0:6], dtype=np.float64)
    L_pose = 0.5 * (L_pose + L_pose.T)

    eig_pose: np.ndarray | None = None
    if np.all(np.isfinite(L_pose)) and float(np.trace(np.abs(L_pose))) > 1e-12:
        try:
            eig_pose = np.linalg.eigvalsh(L_pose)
        except (np.linalg.LinAlgError, ValueError):
            # Fall back to singular values: robust proxy for "strength" and conditioning.
            try:
                eig_pose = np.linalg.svd(L_pose, compute_uv=False)
            except (np.linalg.LinAlgError, ValueError):
                eig_pose = None

    if eig_pose is None or eig_pose.shape != (6,) or (not np.all(np.isfinite(eig_pose))):
        eig_pose = np.full((6,), eps_cond, dtype=np.float64)
    eig_pose = np.sort(eig_pose)

    eig_pose_clipped = np.maximum(eig_pose, eps_cond)
    eig_min_pose = float(eig_pose_clipped[0])
    eig_max_pose = float(eig_pose_clipped[-1])
    cond_pose6 = float(eig_max_pose / eig_min_pose)

    combined_evidence_cert.conditioning = ConditioningCert(
        eig_min=eig_min_pose,
        eig_max=eig_max_pose,
        cond=cond_pose6,
        near_null_count=int(np.sum(eig_pose <= eps_cond)),
    )
    combined_evidence_cert.approximation_triggers.append("FusionScaleConditioningPose6")

    fusion_scale_result, fusion_scale_cert, fusion_scale_effect = fusion_scale_from_certificates(
        cert_evidence=combined_evidence_cert,
        cert_belief=pred_cert,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        kappa_scale=config.kappa_scale,
        c0_cond=config.c0_cond,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(fusion_scale_cert)
    alpha = fusion_scale_result.alpha
    
    # =========================================================================
    # Step 11: InfoFusionAdditive
    # =========================================================================
    belief_post, fusion_cert, fusion_effect = info_fusion_additive(
        belief_pred=belief_pred,
        L_evidence=L_evidence,
        h_evidence=h_evidence,
        alpha=alpha,
        eps_psd=config.eps_psd,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(fusion_cert)
    
    # =========================================================================
    # Step 12: PoseUpdateFrobeniusRecompose
    # =========================================================================
    total_trigger_magnitude = sum(c.total_trigger_magnitude() for c in all_certs)
    
    recompose_result, belief_recomposed, recompose_cert, recompose_effect = pose_update_frobenius_recompose(
        belief_post=belief_post,
        total_trigger_magnitude=total_trigger_magnitude,
        c_frob=config.c_frob,
        eps_lift=config.eps_lift,
    )
    all_certs.append(recompose_cert)

    # =========================================================================
    # Process-noise IW sufficient statistics (NEW; commutative, no state mutation here)
    # =========================================================================
    iw_process_dPsi, iw_process_dnu = process_noise_iw_suffstats_from_info_jax(
        L_pred=belief_pred.L,
        h_pred=belief_pred.h,
        L_post=belief_recomposed.L,
        h_post=belief_recomposed.h,
        eps_lift=config.eps_lift,
    )
    
    # =========================================================================
    # Step 13: PoseCovInflationPushforward
    # =========================================================================
    # CRITICAL FIX: When the map is empty (first scan), R_hat and t_hat from
    # Wahba/WLS are meaningless (identity/zero). Instead, use the POSTERIOR
    # BELIEF's pose to place the scan in the map. This ensures the first scan
    # is conditioned on IMU/odom data, not just placed at identity.
    #
    # The posterior belief already incorporates:
    # - Odom evidence (L_odom, h_odom)
    # - IMU gravity evidence (L_imu, h_imu)
    # - IMU gyro evidence (L_gyro, h_gyro)
    #
    # When map has data, R_hat/t_hat from scan-to-map alignment are valid.
    map_total_mass = float(jnp.sum(map_stats.N_dir))
    map_is_empty = map_total_mass < config.eps_mass

    if map_is_empty:
        # First scan: use POSTERIOR belief pose for map placement.
        #
        # SIGN FIX: Previously used belief_pred (START-of-scan) which caused a
        # rotation mismatch. The posterior (belief_recomposed) incorporates the
        # gyro evidence which pulls toward END-of-scan rotation. Since the next
        # scan's prior IS the previous posterior (at END), we must place the
        # map at END rotation for consistency. Otherwise:
        #   - Scan 1 map at R_start_scan1
        #   - Scan 2 prior at R_end_scan1 = R_start_scan1 @ R_delta
        #   - Matrix Fisher estimates R_start_scan1 (from map), prior says R_end_scan1
        #   - dyaw_mf = -dyaw_gyro (SIGN MISMATCH!)
        #
        # With this fix: map at R_end_scan1, prior at R_end_scan1, no mismatch.
        #
        # Note: scan_bins.s_dir is in START-of-scan body frame (from deskew).
        # We transform these to world frame using the POSTERIOR rotation. This
        # means the map directions correspond to "where the robot was at END
        # of scan 1, looking at directions that were measured at START".
        # This is geometrically consistent because Matrix Fisher alignment will
        # also use posterior-like rotations on subsequent scans.
        pose_for_map = belief_recomposed.mean_world_pose(eps_lift=config.eps_lift)
        R_for_map = se3_jax.so3_exp(pose_for_map[3:6])  # Rotation from posterior
        t_for_map = pose_for_map[:3]  # Translation from posterior
    else:
        # Subsequent scans: use Matrix Fisher/planar translation alignment (scan-to-map matching)
        R_for_map = R_hat
        t_for_map = t_hat

    map_update_result, map_update_cert, map_update_effect = pos_cov_inflation_pushforward(
        belief_post=belief_recomposed,
        scan_N=scan_bins.N,
        scan_s_dir=scan_bins.s_dir,
        scan_S_dir_scatter=scan_bins.S_dir_scatter,
        scan_p_bar=scan_bins.p_bar,
        scan_Sigma_p=scan_bins.Sigma_p,
        R_hat=R_for_map,
        t_hat=t_for_map,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
    )
    all_certs.append(map_update_cert)
    
    # =========================================================================
    # Step 14: AnchorDriftUpdate
    # =========================================================================
    drift_result, belief_final, drift_cert, drift_effect = anchor_drift_update(
        belief=belief_recomposed,
        eps_lift=config.eps_lift,
        eps_psd=config.eps_psd,
    )
    all_certs.append(drift_cert)
    
    # =========================================================================
    # Aggregate certificates
    # =========================================================================
    aggregated_cert = aggregate_certificates(all_certs)

    # =========================================================================
    # Build diagnostics (Stage-0 schema for dashboard)
    # =========================================================================
    import numpy as np

    # Extract pose from final belief
    pose_final = belief_final.mean_world_pose(eps_lift=config.eps_lift)
    trans_final = np.array(pose_final[:3])
    rotvec_final = np.array(pose_final[3:6])

    # Convert rotvec to rotation matrix
    from scipy.spatial.transform import Rotation as R_scipy
    R_final = R_scipy.from_rotvec(rotvec_final).as_matrix()

    # Rotation-binding diagnostics: verify residuals decrease after fusion.
    pose_pred_np = np.array(belief_pred.mean_world_pose(eps_lift=config.eps_lift))
    R_pred = R_scipy.from_rotvec(pose_pred_np[3:6]).as_matrix()

    odom_pose_np = np.array(odom_pose)
    R_odom = R_scipy.from_rotvec(odom_pose_np[3:6]).as_matrix()

    R_lidar = np.array(R_hat)

    # Compute rotation errors robustly (handle non-orthogonal matrices from numerical issues)
    def _safe_rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
        """Compute rotation error in degrees, handling numerical edge cases."""
        try:
            R_diff = R1.T @ R2
            # Force orthogonality via SVD projection
            U, _, Vt = np.linalg.svd(R_diff)
            R_diff_ortho = U @ Vt
            # Ensure proper rotation (det = +1)
            if np.linalg.det(R_diff_ortho) < 0:
                R_diff_ortho = -R_diff_ortho
            return float(np.linalg.norm(R_scipy.from_matrix(R_diff_ortho).as_rotvec()) * (180.0 / np.pi))
        except Exception:
            return float("nan")

    rot_err_lidar_deg_pred = _safe_rotation_error_deg(R_pred, R_lidar)
    rot_err_lidar_deg_post = _safe_rotation_error_deg(R_final, R_lidar)
    rot_err_odom_deg_pred = _safe_rotation_error_deg(R_pred, R_odom)
    rot_err_odom_deg_post = _safe_rotation_error_deg(R_final, R_odom)

    # IMU gravity direction coherence probe (in body frame).
    imu_accel_np = np.array(imu_accel, dtype=np.float64)
    w_imu_np = np.array(w_imu_int, dtype=np.float64).reshape(-1)
    accel_bias_np = np.array(accel_bias, dtype=np.float64).reshape(-1)
    a_corr = imu_accel_np - accel_bias_np[None, :]
    a_norm = np.linalg.norm(a_corr, axis=1)
    accel_mag_mean = float(np.sum(w_imu_np * a_norm) / (np.sum(w_imu_np) + 1e-12))
    x = a_corr / (a_norm[:, None] + 1e-12)
    S = np.sum(w_imu_np[:, None] * x, axis=0)
    xbar = S / (np.linalg.norm(S) + 1e-12)
    g = np.array(constants.GC_GRAVITY_W, dtype=np.float64).reshape(-1)
    g_hat = g / (np.linalg.norm(g) + 1e-12)
    mu0 = R_pred.T @ (-g_hat)
    accel_dir_dot_mu0 = float(np.dot(xbar, mu0))

    # Compute evidence diagnostics
    L_total_np = np.array(L_evidence)
    h_total_np = np.array(h_evidence)

    # Diagnostic: check for non-finite values before eigvalsh (deterministic check, not a gate)
    if not np.all(np.isfinite(L_total_np)):
        nan_count = np.sum(~np.isfinite(L_total_np))
        max_abs = np.nanmax(np.abs(L_total_np))
        # Identify which block has the issue
        block_info = []
        for name, sl in [("trans", slice(0, 3)), ("rot", slice(3, 6)), ("vel", slice(6, 9)),
                         ("bg", slice(9, 12)), ("ba", slice(12, 15)), ("dt", slice(15, 16)),
                         ("ex", slice(16, 22))]:
            block = L_total_np[sl, sl]
            if not np.all(np.isfinite(block)):
                block_info.append(f"{name}:NaN")
        raise ValueError(
            f"L_total contains {nan_count} non-finite values (max_abs={max_abs:.2e}). "
            f"Blocks with NaN: {block_info}. Check evidence operators for source."
        )

    # Safe logdet computation (handle near-singular matrices)
    eigvals = np.linalg.eigvalsh(L_total_np)
    eigvals_pos = np.maximum(eigvals, 1e-12)
    logdet_L = float(np.sum(np.log(eigvals_pos)))
    trace_L = float(np.trace(L_total_np))
    L_dt_val = float(L_total_np[15, 15])
    trace_L_ex = float(np.trace(L_total_np[16:22, 16:22]))

    # PSD diagnostics from aggregated certificate
    psd_delta = 0.0
    psd_min_eig_before = 0.0
    psd_min_eig_after = float(np.min(eigvals))
    if aggregated_cert.influence is not None:
        psd_delta = aggregated_cert.influence.psd_projection_delta
    if aggregated_cert.conditioning is not None:
        psd_min_eig_before = aggregated_cert.conditioning.eig_min
        cond_number = aggregated_cert.conditioning.cond
    else:
        cond_number = 1.0

    # Total trigger magnitude
    total_trigger_mag = sum(c.total_trigger_magnitude() for c in all_certs)

    # Matrix Fisher and scatter sentinels for the dashboard
    mf_svd_np = np.array(mf_result.svd_singular_values, dtype=np.float64).reshape(3,)
    scan_scatter_eigs = np.linalg.eigvalsh(np.array(scan_bins.S_dir_scatter, dtype=np.float64))  # (B, 3)
    map_scatter_eigs = np.linalg.eigvalsh(np.array(map_stats.S_dir_scatter, dtype=np.float64))  # (B, 3)

    diagnostics = ScanDiagnostics(
        scan_number=0,  # Will be set by backend_node
        timestamp=scan_end_time,
        dt_sec=dt_sec,
        dt_scan=float(scan_end_time - scan_start_time),
        dt_int=dt_int,  # IMU integration time: sum of sample intervals in (t_last_scan, t_scan)
        num_imu_samples=int(
            np.sum(
                (np.asarray(imu_stamps, dtype=np.float64) > (float(t_last_scan) - 1e-9))
                & (np.asarray(imu_stamps, dtype=np.float64) <= (float(t_scan) + 1e-9))
                & (np.asarray(imu_stamps, dtype=np.float64) > 0.0)
            )
        ),
        n_points_raw=int(raw_points.shape[0]),
        n_points_budget=int(points.shape[0]),
        p_W=trans_final,
        R_WL=R_final,
        N_bins=np.array(scan_bins.N),
        S_bins=np.array(scan_bins.s_dir),
        kappa_bins=np.array(scan_bins.kappa_scan),
        kappa_map_bins=np.array(kappa_map),
        L_total=L_total_np,
        h_total=h_total_np,
        L_lidar=np.array(L_lidar),
        L_odom=np.array(odom_result.L_odom),
        L_imu=np.array(imu_result.L_imu),
        L_gyro=np.array(gyro_result.L_gyro),
        L_imu_preint=np.array(preint_result.L_imu_preint),
        logdet_L_total=logdet_L,
        trace_L_total=trace_L,
        L_dt=L_dt_val,
        trace_L_ex=trace_L_ex,
        s_dt=float(s_dt),
        s_ex=float(s_ex),
        psd_delta_fro=psd_delta,
        psd_min_eig_before=psd_min_eig_before,
        psd_min_eig_after=psd_min_eig_after,
        wahba_cost=float(jnp.sum(mf_result.svd_singular_values)),  # MF cost proxy (SVD sum)
        mf_svd=mf_svd_np,
        scan_scatter_eigs=scan_scatter_eigs,
        map_scatter_eigs=map_scatter_eigs,
        translation_residual_norm=float(jnp.linalg.norm(planar_trans_result.delta_trans)),
        rot_err_lidar_deg_pred=rot_err_lidar_deg_pred,
        rot_err_lidar_deg_post=rot_err_lidar_deg_post,
        rot_err_odom_deg_pred=rot_err_odom_deg_pred,
        rot_err_odom_deg_post=rot_err_odom_deg_post,
        accel_dir_dot_mu0=accel_dir_dot_mu0,
        accel_mag_mean=accel_mag_mean,
        imu_a_body_mean=np.array(imu_a_body_mean),
        imu_a_world_nog_mean=np.array(imu_a_world_nog_mean),
        imu_a_world_mean=np.array(imu_a_world_mean),
        imu_dt_eff_sum=float(imu_dt_eff_sum),
        imu_dt_valid_min=imu_dt_valid_min,
        imu_dt_valid_max=imu_dt_valid_max,
        imu_dt_valid_mean=imu_dt_valid_mean,
        imu_dt_valid_median=imu_dt_valid_median,
        imu_dt_valid_std=imu_dt_valid_std,
        imu_dt_valid_nonpos=imu_dt_valid_nonpos,
        imu_dt_weighted_mean=imu_dt_weighted_mean,
        imu_dt_weighted_std=imu_dt_weighted_std,
        imu_dt_weighted_sum=imu_dt_weighted_sum,
        fusion_alpha=float(alpha),
        total_trigger_magnitude=total_trigger_mag,
        conditioning_number=cond_number,
        conditioning_pose6=float(cond_pose6),
        preint_r_vel=np.array(preint_result.r_vel),
        preint_r_pos=np.array(preint_result.r_pos),
        dyaw_gyro=float(dyaw_gyro),
        dyaw_odom=float(dyaw_odom),
        dyaw_wahba=float(dyaw_mf),  # Now Matrix Fisher, kept field name for compatibility
    )

    return ScanPipelineResult(
        belief_updated=belief_final,
        map_increments=map_update_result,
        iw_process_dPsi=iw_process_dPsi,
        iw_process_dnu=iw_process_dnu,
        iw_meas_dPsi=iw_meas_dPsi,
        iw_meas_dnu=iw_meas_dnu,
        iw_lidar_bucket_dPsi=iw_lidar_bucket_dPsi,
        iw_lidar_bucket_dnu=iw_lidar_bucket_dnu,
        all_certs=all_certs,
        aggregated_cert=aggregated_cert,
        diagnostics=diagnostics,
    )


def process_hypotheses(
    hypotheses: List[BeliefGaussianInfo],
    weights: jnp.ndarray,
    config: PipelineConfig,
) -> Tuple[BeliefGaussianInfo, CertBundle, ExpectedEffect]:
    """
    Combine hypotheses via barycenter projection.
    
    Step 15 in the pipeline.
    
    Args:
        hypotheses: List of K_HYP beliefs (one per hypothesis)
        weights: Hypothesis weights (K_HYP,)
        config: Pipeline configuration
        
    Returns:
        Tuple of (combined_belief, CertBundle, ExpectedEffect)
    """
    result, cert, effect = hypothesis_barycenter_projection(
        hypotheses=hypotheses,
        weights=weights,
        K_HYP=config.K_HYP,
        HYP_WEIGHT_FLOOR=constants.GC_HYP_WEIGHT_FLOOR,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
    )
    
    return result.belief_out, cert, effect


# =============================================================================
# Runtime Manifest
# =============================================================================


@dataclass
class RuntimeManifest:
    """
    Runtime manifest per spec Section 6.
    
    Nodes must publish/log this at startup.
    """
    chart_id: str = constants.GC_CHART_ID
    
    D_Z: int = constants.GC_D_Z
    D_DESKEW: int = constants.GC_D_DESKEW
    K_HYP: int = constants.GC_K_HYP
    HYP_WEIGHT_FLOOR: float = constants.GC_HYP_WEIGHT_FLOOR
    B_BINS: int = constants.GC_B_BINS
    N_POINTS_CAP: int = constants.GC_N_POINTS_CAP
    
    tau_soft_assign: float = constants.GC_TAU_SOFT_ASSIGN
    
    eps_psd: float = constants.GC_EPS_PSD
    eps_lift: float = constants.GC_EPS_LIFT
    eps_mass: float = constants.GC_EPS_MASS
    eps_r: float = constants.GC_EPS_R
    eps_den: float = constants.GC_EPS_DEN
    
    alpha_min: float = constants.GC_ALPHA_MIN
    alpha_max: float = constants.GC_ALPHA_MAX
    kappa_scale: float = constants.GC_KAPPA_SCALE
    c0_cond: float = constants.GC_C0_COND
    
    c_dt: float = constants.GC_C_DT
    c_ex: float = constants.GC_C_EX
    c_frob: float = constants.GC_C_FROB
    imu_gravity_scale: float = 1.0

    # Explicit backend/operator selections (single-path; no fallback).
    # This is required for auditability of the "no multipaths" invariant.
    backends: Dict[str, str] = None

    def __post_init__(self):
        if self.backends is None:
            # Keep this list intentionally small and explicit: only things that
            # materially affect runtime behavior and could otherwise drift.
            self.backends = {
                "core_array": "jax",
                "se3": "fl_slam_poc.common.geometry.se3_jax",
                "so3_right_jacobian": "fl_slam_poc.common.geometry.se3_jax.so3_right_jacobian",
                "domain_projection_psd": "fl_slam_poc.common.primitives.domain_projection_psd_core",
                "lifted_spd_solve": "fl_slam_poc.common.primitives.spd_cholesky_solve_lifted_core",
                "lifted_spd_inverse": "fl_slam_poc.common.primitives.spd_cholesky_inverse_lifted_core",
                "process_noise_model": "fl_slam_poc.backend.operators.inverse_wishart_jax (IW, commutative per-scan)",
                "measurement_noise_model": "fl_slam_poc.backend.operators.measurement_noise_iw_jax (IW per-sensor; lidar enabled)",
                "deskew": "fl_slam_poc.backend.operators.deskew_constant_twist (constant twist; IMU preintegration)",
                "imu_preintegration": "fl_slam_poc.backend.operators.imu_preintegration",
                "imu_evidence": "fl_slam_poc.backend.operators.imu_evidence (Laplace over intrinsic ops)",
                "odom_evidence": "fl_slam_poc.backend.operators.odom_evidence (Gaussian SE(3) pose factor)",
                "lidar_evidence": "fl_slam_poc.backend.operators.matrix_fisher_evidence (Matrix Fisher + planar translation)",
                "hypothesis_barycenter": "fl_slam_poc.backend.operators.hypothesis (vectorized over hypotheses)",
                "map_update": "fl_slam_poc.backend.operators.map_update (vectorized over bins)",
                "lidar_converter": "fl_slam_poc.frontend.sensors.livox_converter",
                "pointcloud_parser": "fl_slam_poc.backend.backend_node.parse_pointcloud2",
            }
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/publishing."""
        return {
            "chart_id": self.chart_id,
            "D_Z": self.D_Z,
            "D_DESKEW": self.D_DESKEW,
            "K_HYP": self.K_HYP,
            "HYP_WEIGHT_FLOOR": self.HYP_WEIGHT_FLOOR,
            "B_BINS": self.B_BINS,
            "N_POINTS_CAP": self.N_POINTS_CAP,
            "tau_soft_assign": self.tau_soft_assign,
            "eps_psd": self.eps_psd,
            "eps_lift": self.eps_lift,
            "eps_mass": self.eps_mass,
            "eps_r": self.eps_r,
            "eps_den": self.eps_den,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "kappa_scale": self.kappa_scale,
            "c0_cond": self.c0_cond,
            "c_dt": self.c_dt,
            "c_ex": self.c_ex,
            "c_frob": self.c_frob,
            "imu_gravity_scale": self.imu_gravity_scale,
            "backends": dict(self.backends),
        }
