"""
Geometric Compositional SLAM v2 Pipeline.

Main per-scan execution following spec Section 7.
All steps run every time; influence may go to ~0 smoothly. No gates.

Reference: docs/GC_SLAM.md Section 7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

import time
from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.runtime_counters import record_host_sync
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.common.belief import BeliefGaussianInfo
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    aggregate_certificates,
    InfluenceCert,
    MapUpdateCert,
)
# PrimitiveMap-only pipeline (no bin backend)

# Import all operators
from fl_slam_poc.backend.operators import (
    point_budget_resample,
    predict_diffusion,
    smooth_window_weights,
    preintegrate_imu_relative_pose_jax,
    deskew_constant_twist,
    imu_gyro_meas_iw_suffstats_from_avg_rate_jax,
    imu_accel_meas_iw_suffstats_from_gravity_dir_jax,
    odom_quadratic_evidence,
    odom_velocity_evidence,
    odom_yawrate_evidence,
    pose_twist_kinematic_consistency,
    odom_dependence_inflation,
    imu_vmf_gravity_evidence_time_resolved,
    imu_dependence_inflation,
    planar_z_prior,
    velocity_z_prior,
    imu_gyro_rotation_evidence,
    imu_preintegration_factor,
    fusion_scale_from_certificates,
    info_fusion_additive,
    compute_excitation_scales_jax,
    apply_excitation_prior_scaling_jax,
    pose_update_frobenius_recompose,
    process_noise_iw_suffstats_from_info_jax,
    anchor_drift_update,
    hypothesis_barycenter_projection,
    extract_lidar_surfels,
    SurfelExtractionConfig,
    associate_primitives_ot,
    AssociationConfig,
    block_associations_for_fuse,
    visual_pose_evidence,
    build_visual_pose_evidence_22d,
)
from fl_slam_poc.backend.diagnostics import MinimalScanTape

# Stage 1: PrimitiveMap + OT imports
from fl_slam_poc.backend.structures import (
    AtlasMap,
    create_empty_atlas_map,
    extract_atlas_map_view,
    primitive_map_fuse,
    primitive_map_insert_masked,
    primitive_map_cull,
    primitive_map_forget,
    primitive_map_recency_inflate,
    primitive_map_merge_reduce,
    MeasurementBatch,
    create_empty_measurement_batch,
)
from fl_slam_poc.common.tiling import ma_hex_stencil_tile_ids, tile_ids_from_xyz_batch_jax
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_solve_lifted,
)


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the Geometric Compositional pipeline."""
    # Budgets (hard constants)
    K_HYP: int = constants.GC_K_HYP
    N_POINTS_CAP: int = constants.GC_N_POINTS_CAP
    N_FEAT: int = constants.GC_N_FEAT
    N_SURFEL: int = constants.GC_N_SURFEL
    K_SINKHORN: int = constants.GC_K_SINKHORN
    
    # Epsilon constants
    eps_psd: float = constants.GC_EPS_PSD
    eps_lift: float = constants.GC_EPS_LIFT
    eps_mass: float = constants.GC_EPS_MASS
    
    # Fusion parameters
    alpha_min: float = constants.GC_ALPHA_MIN
    alpha_max: float = constants.GC_ALPHA_MAX
    kappa_scale: float = constants.GC_KAPPA_SCALE
    c0_cond: float = constants.GC_C0_COND

    # Power tempering (generalized Bayes / power EP): L,h ← beta*(L,h)
    # beta is computed continuously from certificate sentinels; fixed-budget; no iteration.
    power_beta_min: float = 0.25
    power_beta_exc_c: float = 50.0  # ess_to_excitation scale (larger => less tempering)
    power_beta_z_c: float = 1.0     # z_to_xy_ratio scale for saturation (larger => more permissive)
    
    # Excitation coupling
    c_dt: float = constants.GC_C_DT
    c_ex: float = constants.GC_C_EX
    c_frob: float = constants.GC_C_FROB
    
    # Soft assign
    
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

    # Profiling / timing
    enable_timing: bool = False  # If True, record per-stage timings (ms) in diagnostics

    # Diagnostics: minimal tape only (full ScanDiagnostics removed)

    # Parallelize independent stages (IMU+odom evidence vs surfel/association prep)
    enable_parallel_stages: bool = False

    # =========================================================================
    # Map + Pose Evidence (single-path, primitives only)
    # =========================================================================
    # The canonical backend is PrimitiveMap + primitive alignment evidence.

    # PrimitiveMap budgets
    n_feat: int = constants.GC_N_FEAT
    n_surfel: int = constants.GC_N_SURFEL
    k_assoc: int = constants.GC_K_ASSOC
    k_sinkhorn: int = constants.GC_K_SINKHORN
    ot_epsilon: float = 0.1
    ot_tau_a: float = 0.5
    ot_tau_b: float = 0.5
    primitive_map_max_size: int = constants.GC_PRIMITIVE_MAP_MAX_SIZE
    primitive_forgetting_factor: float = constants.GC_PRIMITIVE_FORGETTING_FACTOR
    primitive_merge_threshold: float = constants.GC_PRIMITIVE_MERGE_THRESHOLD
    primitive_cull_weight_threshold: float = constants.GC_PRIMITIVE_CULL_WEIGHT_THRESHOLD
    # Fixed-budget insertion each scan (per-tile)
    k_insert_tile: int = constants.GC_K_INSERT_TILE
    # Fixed-budget merge-reduce per tile (number of pairs to merge)
    k_merge_pairs_tile: int = constants.GC_K_MERGE_PAIRS_PER_TILE
    # Fixed-budget merge-reduce cap (avoid O(M^2) when tile is large)
    primitive_merge_max_tile_size: int = constants.GC_PRIMITIVE_MERGE_MAX_TILE_SIZE

    # Atlas tiling budgets (Phase 6 end state; spec §5.7)
    H_TILE: float = constants.GC_H_TILE
    N_ACTIVE_TILES: int = constants.GC_N_ACTIVE_TILES
    R_ACTIVE_TILES_XY: int = constants.GC_R_ACTIVE_TILES_XY
    R_ACTIVE_TILES_Z: int = constants.GC_R_ACTIVE_TILES_Z
    M_TILE_VIEW: int = constants.GC_M_TILE_VIEW
    N_STENCIL_TILES: int = constants.GC_N_STENCIL_TILES
    R_STENCIL_TILES_XY: int = constants.GC_R_STENCIL_TILES_XY
    R_STENCIL_TILES_Z: int = constants.GC_R_STENCIL_TILES_Z
    RECENCY_DECAY_LAMBDA: float = constants.GC_RECENCY_DECAY_LAMBDA
    RECENCY_MIN_SCALE: float = constants.GC_RECENCY_MIN_SCALE

    # Surfel extraction config
    surfel_voxel_size_m: float = 0.1
    surfel_min_points_per_voxel: int = 3

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
    # Per-scan diagnostics tape for dashboard (minimal-only)
    diagnostics_tape: Optional[MinimalScanTape] = None
    # PrimitiveMap update results
    primitive_map_updated: Optional[AtlasMap] = None
    measurement_batch: Optional[MeasurementBatch] = None
    n_primitives_inserted: int = 0
    n_primitives_fused: int = 0
    n_primitives_culled: int = 0
    n_primitives_merged: int = 0
    event_log_entries: Optional[list] = None


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
    config: PipelineConfig,
    odom_twist: jnp.ndarray,  # (6,) [vx,vy,vz,wx,wy,wz] in body frame (never None)
    odom_twist_cov: jnp.ndarray,  # (6,6) twist covariance (never None)
    camera_batch: MeasurementBatch,  # Camera splats from VisualFeatureExtractor + splat_prep (required)
    scan_seq: int = 0,
    primitive_map: Optional[AtlasMap] = None,  # AtlasMap is the canonical map
) -> ScanPipelineResult:
    """
    Process a single scan for one hypothesis.
    
    Follows the fixed-cost scan pipeline:
    1. PointBudgetResample
    2. PredictDiffusion
    3. IMU membership weights (soft window)
    4. IMU preintegration (deskew + scan-to-scan)
    5. DeskewConstantTwist
    6. IMU + Odom evidence; compute z_lin
    7. Surfel extraction + OT association
    8. Visual pose evidence (Laplace at z_lin)
    9. Tempered evidence (power EP; closed form)
    10. FusionScaleFromCertificates
    11. InfoFusionAdditive
    12. PoseUpdateFrobeniusRecompose
    13. IW sufficient-stat updates (process + measurement)
    14. PrimitiveMap update (fuse/insert/cull/forget)
    15. AnchorDriftUpdate
    
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
        config: Pipeline configuration
        primitive_map: PrimitiveMap (canonical map; always used)
        camera_batch: Camera MeasurementBatch to merge with LiDAR surfels (required)
        
    Returns:
        ScanPipelineResult with updated belief and certificates
    """
    all_certs = []
    timing_ms = {} if config.enable_timing else None
    t_total_start = time.perf_counter() if config.enable_timing else None

    def _record_timing(label: str, start: float, out):
        if not config.enable_timing:
            return
        try:
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                out,
            )
            record_host_sync(syncs=1)
        except Exception as exc:
            raise RuntimeError(f"Timing sync failed for {label}") from exc
        timing_ms[label] = (time.perf_counter() - start) * 1000.0
    
    # =========================================================================
    # Step 1: PointBudgetResample
    # =========================================================================
    t0 = time.perf_counter() if config.enable_timing else None
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
    if config.enable_timing:
        _record_timing("point_budget_ms", t0, budget_result)
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
    dt_std = jnp.sqrt(Sigma_pred[constants.GC_IDX_DT, constants.GC_IDX_DT])
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
    gyro_bias = mu_inc[constants.GC_IDX_BG]
    accel_bias = mu_inc[constants.GC_IDX_BA]

    pose0 = belief_prev.mean_world_pose(eps_lift=config.eps_lift)
    rotvec0 = pose0[3:6]
    gravity_W = jnp.array(constants.GC_GRAVITY_W, dtype=jnp.float64) * float(config.imu_gravity_scale)

    # Preintegration for deskew (within-scan).
    t0 = time.perf_counter() if config.enable_timing else None
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
    if config.enable_timing:
        _record_timing("imu_preint_scan_ms", t0, delta_pose_scan)
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
    t0 = time.perf_counter() if config.enable_timing else None
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
    if config.enable_timing:
        _record_timing("imu_preint_int_ms", t0, delta_pose_int)
    
    # IMU measurement-noise IW sufficient stats (Σg, Σa) from commutative residuals
    # Average IMU sampling period for PSD mapping (Var * dt -> PSD).
    # NOTE: imu_stamps is padded with zeros; exclude padded entries by construction.
    import numpy as np
    imu_stamps_np = np.asarray(imu_stamps, dtype=np.float64).reshape(-1)
    valid_mask_np = imu_stamps_np > 0.0
    n_valid = int(np.sum(valid_mask_np))
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
    
    # Diagnostic: omega_avg sanity (no heuristics / no gating). JAX check, sync only on failure.
    if not bool(jnp.all(jnp.isfinite(omega_avg))):
        raise ValueError(f"omega_avg contains non-finite values: {omega_avg}")

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

    t0 = time.perf_counter() if config.enable_timing else None
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
    if config.enable_timing:
        _record_timing("deskew_ms", t0, deskew_twist_result.points)
    all_certs.append(deskew_cert)

    deskewed_points = deskew_twist_result.points
    deskewed_weights = deskew_twist_result.weights
    # Point covariances are not used by the new deskew. Keep zeros (deterministic).
    deskewed_covs = jnp.zeros((deskewed_points.shape[0], 3, 3), dtype=jnp.float64)

    # Compute point directions from the LiDAR origin (not base origin).
    # points are in base frame; subtract sensor origin to recover true ray directions.
    rays = deskewed_points - config.lidar_origin_base[None, :]
    norms = jnp.linalg.norm(rays, axis=1, keepdims=True)
    point_directions = rays / (norms + config.eps_mass)

    def _compute_imu_odom_branch():
        local_certs: List = []
        pose_pred_local = belief_pred.mean_world_pose(eps_lift=config.eps_lift)
        odom_pose_local = jnp.asarray(odom_pose, dtype=jnp.float64).reshape(-1)
        odom_cov_se3_local = jnp.asarray(odom_cov_se3, dtype=jnp.float64)
        odom_result_local, odom_cert_local, _ = odom_quadratic_evidence(
            belief_pred_pose=pose_pred_local,
            odom_pose=odom_pose_local,
            odom_cov_se3=odom_cov_se3_local,
            eps_psd=config.eps_psd,
            eps_lift=config.eps_lift,
            chart_id=belief_pred.chart_id,
            anchor_id=belief_pred.anchor_id,
        )
        local_certs.append(odom_cert_local)
        imu_result_local, imu_cert_local, _ = imu_vmf_gravity_evidence_time_resolved(
            rotvec_world_body=pose_pred_local[3:6],
            imu_accel=imu_accel,
            imu_gyro=imu_gyro,
            weights=w_imu_int,
            accel_bias=accel_bias,
            gravity_W=gravity_W,
            dt_imu=dt_imu,
            eps_psd=config.eps_psd,
            eps_mass=config.eps_mass,
            chart_id=belief_pred.chart_id,
            anchor_id=belief_pred.anchor_id,
        )
        local_certs.append(imu_cert_local)
        dep_result_local, dep_cert_local, _ = imu_dependence_inflation(
            transport_sigma=float(imu_result_local.transport_sigma),
            eps_mass=config.eps_mass,
            chart_id=belief_pred.chart_id,
            anchor_id="imu_dependence_inflation",
        )
        local_certs.append(dep_cert_local)
        Sigma_g_local = jnp.asarray(config.Sigma_g, dtype=jnp.float64)
        # JAX finite checks; sync only on failure for error message.
        if not bool(jnp.all(jnp.isfinite(rotvec0))):
            raise ValueError(f"rotvec0 contains NaN: {rotvec0}")
        if not bool(jnp.all(jnp.isfinite(pose_pred_local))):
            raise ValueError(f"pose_pred contains NaN: {pose_pred_local}")
        if not bool(jnp.all(jnp.isfinite(delta_pose_int))):
            raise ValueError(f"delta_pose_int contains NaN: {delta_pose_int}")
        gyro_result_local, gyro_cert_local, _ = imu_gyro_rotation_evidence(
            rotvec_start_WB=rotvec0,
            rotvec_end_pred_WB=pose_pred_local[3:6],
            delta_rotvec_meas=delta_pose_int[3:6],
            Sigma_g=Sigma_g_local,
            dt_int=dt_int,
            eps_psd=config.eps_psd,
            eps_lift=config.eps_lift,
            chart_id=belief_pred.chart_id,
            anchor_id=belief_pred.anchor_id,
        )
        local_certs.append(gyro_cert_local)
        mu_prev_local = belief_prev.mean_increment(eps_lift=config.eps_lift)
        v_start_world_local = mu_prev_local[constants.GC_IDX_VEL]
        v_pred_world_local = mu_inc[constants.GC_IDX_VEL]
        Sigma_a_local = jnp.asarray(config.Sigma_a, dtype=jnp.float64)
        preint_result_local, preint_cert_local, _ = imu_preintegration_factor(
            p_start_world=pose0[0:3],
            rotvec_start_WB=rotvec0,
            v_start_world=v_start_world_local,
            p_end_pred_world=pose_pred_local[0:3],
            v_end_pred_world=v_pred_world_local,
            delta_v_body=delta_v_int,
            delta_p_body=delta_p_int,
            Sigma_a=Sigma_a_local,
            dt_int=dt_int,
            eps_psd=config.eps_psd,
            eps_lift=config.eps_lift,
            chart_id=belief_pred.chart_id,
            anchor_id=belief_pred.anchor_id,
        )
        local_certs.append(preint_cert_local)
        planar_result_local, planar_cert_local, _ = planar_z_prior(
            belief_pred_pose=pose_pred_local,
            z_ref=config.planar_z_ref,
            sigma_z=config.planar_z_sigma,
            eps_psd=config.eps_psd,
            chart_id=belief_pred.chart_id,
            anchor_id=belief_pred.anchor_id,
        )
        local_certs.append(planar_cert_local)
        vz_pred_local = float(mu_inc[constants.GC_IDX_VEL][2])
        vz_result_local, vz_cert_local, _ = velocity_z_prior(
            v_z_pred=vz_pred_local,
            sigma_vz=config.planar_vz_sigma,
            chart_id=belief_pred.chart_id,
            anchor_id=belief_pred.anchor_id,
        )
        local_certs.append(vz_cert_local)
        v_pred_world_early = mu_inc[constants.GC_IDX_VEL]
        R_world_body_early = se3_jax.so3_exp(pose_pred_local[3:6])
        odom_vel_result_local, odom_vel_cert_local, _ = odom_velocity_evidence(
            v_pred_world=v_pred_world_early,
            R_world_body=R_world_body_early,
            v_odom_body=odom_twist[0:3],
            Sigma_v=odom_twist_cov[0:3, 0:3],
            eps_psd=config.eps_psd,
            eps_lift=config.eps_lift,
            chart_id=belief_pred.chart_id,
            anchor_id=belief_pred.anchor_id,
        )
        local_certs.append(odom_vel_cert_local)
        sigma_wz_from_cov_local = jnp.sqrt(jnp.maximum(odom_twist_cov[5, 5], 1e-12))
        sigma_wz_effective_local = float(sigma_wz_from_cov_local)
        odom_wz_result_local, odom_wz_cert_local, _ = odom_yawrate_evidence(
            omega_z_pred=float(omega_avg[2]),
            omega_z_odom=float(odom_twist[5]),
            sigma_wz=sigma_wz_effective_local,
            chart_id=belief_pred.chart_id,
            anchor_id=belief_pred.anchor_id,
        )
        local_certs.append(odom_wz_cert_local)
        kinematic_result_local, kinematic_cert_local, _ = pose_twist_kinematic_consistency(
            pose_prev=pose0,
            pose_curr=pose_pred_local,
            v_body=odom_twist[0:3],
            omega_body=odom_twist[3:6],
            dt=dt_sec,
            Sigma_v=odom_twist_cov[0:3, 0:3],
            Sigma_omega=odom_twist_cov[3:6, 3:6],
            eps_psd=config.eps_psd,
            eps_lift=config.eps_lift,
            chart_id=belief_pred.chart_id,
            anchor_id=belief_pred.anchor_id,
        )
        local_certs.append(kinematic_cert_local)
        odom_dep_result_local, odom_dep_cert_local, _ = odom_dependence_inflation(
            r_trans=kinematic_result_local.r_trans,
            r_rot=kinematic_result_local.r_rot,
            eps_mass=config.eps_mass,
            chart_id=belief_pred.chart_id,
            anchor_id="odom_dependence_inflation",
        )
        local_certs.append(odom_dep_cert_local)
        scale_dep_local = float(dep_result_local.scale)
        L_imu_scaled = imu_result_local.L_imu * scale_dep_local
        h_imu_scaled = imu_result_local.h_imu * scale_dep_local
        L_gyro_scaled = gyro_result_local.L_gyro * scale_dep_local
        h_gyro_scaled = gyro_result_local.h_gyro * scale_dep_local
        odom_scale_local = float(odom_dep_result_local.scale)
        L_odom_scaled = odom_result_local.L_odom * odom_scale_local
        h_odom_scaled = odom_result_local.h_odom * odom_scale_local
        L_vel_scaled = odom_vel_result_local.L_vel * odom_scale_local
        h_vel_scaled = odom_vel_result_local.h_vel * odom_scale_local
        L_wz_scaled = odom_wz_result_local.L_wz * odom_scale_local
        h_wz_scaled = odom_wz_result_local.h_wz * odom_scale_local
        L_imu_odom_local = (L_odom_scaled + L_imu_scaled + L_gyro_scaled + preint_result_local.L_imu_preint
                            + planar_result_local.L_planar + vz_result_local.L_vz + L_vel_scaled + L_wz_scaled
                            + kinematic_result_local.L_consistency)
        h_imu_odom_local = (h_odom_scaled + h_imu_scaled + h_gyro_scaled + preint_result_local.h_imu_preint
                            + planar_result_local.h_planar + vz_result_local.h_vz + h_vel_scaled + h_wz_scaled
                            + kinematic_result_local.h_consistency)
        L_fused_local = belief_pred.L + L_imu_odom_local
        h_fused_local = belief_pred.h + h_imu_odom_local
        L_fused_psd_local = domain_projection_psd(L_fused_local, config.eps_psd).M_psd
        z_lin_22d_local = spd_cholesky_solve_lifted(L_fused_psd_local, h_fused_local, config.eps_lift).x
        z_lin_pose_local = z_lin_22d_local[constants.GC_IDX_POSE]
        return {
            "certs": local_certs,
            "odom_cert": odom_cert_local,
            "imu_cert": imu_cert_local,
            "gyro_cert": gyro_cert_local,
            "pose_pred": pose_pred_local,
            "odom_result": odom_result_local,
            "imu_result": imu_result_local,
            "gyro_result": gyro_result_local,
            "preint_result": preint_result_local,
            "planar_result": planar_result_local,
            "vz_result": vz_result_local,
            "odom_vel_result": odom_vel_result_local,
            "odom_wz_result": odom_wz_result_local,
            "kinematic_result": kinematic_result_local,
            "dep_result": dep_result_local,
            "odom_dep_result": odom_dep_result_local,
            "L_imu_odom": L_imu_odom_local,
            "h_imu_odom": h_imu_odom_local,
            "z_lin_pose": z_lin_pose_local,
        }

    def _compute_map_branch():
        local_certs: List = []
        t0_prim_local = time.perf_counter() if config.enable_timing else None
        surfel_config = SurfelExtractionConfig(
            n_surfel=config.n_surfel,
            n_feat=config.n_feat,
            voxel_size_m=config.surfel_voxel_size_m,
            min_points_per_voxel=config.surfel_min_points_per_voxel,
            eps_lift=config.eps_lift,
        )
        measurement_batch_local, surfel_cert_local, _ = extract_lidar_surfels(
            points=deskewed_points,
            timestamps=timestamps,
            weights=deskewed_weights,
            config=surfel_config,
            base_batch=camera_batch,
            chart_id=belief_pred.chart_id,
            anchor_id="surfel_extraction",
        )
        local_certs.append(surfel_cert_local)

        pose_pred_for_map = belief_pred.mean_world_pose(eps_lift=config.eps_lift)
        t_pred_for_map = pose_pred_for_map[:3]

        map_local = primitive_map
        if map_local is None:
            map_local = create_empty_atlas_map(m_tile=config.primitive_map_max_size)

        import numpy as np

        center_xyz = np.asarray(t_pred_for_map, dtype=np.float64).reshape(3)
        active_tile_ids_local = ma_hex_stencil_tile_ids(
            center_xyz=center_xyz,
            h_tile=float(config.H_TILE),
            radius_xy=int(config.R_ACTIVE_TILES_XY),
            radius_z=int(config.R_ACTIVE_TILES_Z),
        )
        stencil_tile_ids_local = ma_hex_stencil_tile_ids(
            center_xyz=center_xyz,
            h_tile=float(config.H_TILE),
            radius_xy=int(config.R_STENCIL_TILES_XY),
            radius_z=int(config.R_STENCIL_TILES_Z),
        )
        if len(active_tile_ids_local) != int(config.N_ACTIVE_TILES):
            raise ValueError(
                f"active tile stencil size mismatch: expected N_ACTIVE_TILES={config.N_ACTIVE_TILES}, "
                f"got {len(active_tile_ids_local)}"
            )
        if len(stencil_tile_ids_local) != int(config.N_STENCIL_TILES):
            raise ValueError(
                f"stencil tile size mismatch: expected N_STENCIL_TILES={config.N_STENCIL_TILES}, "
                f"got {len(stencil_tile_ids_local)}"
            )

        map_local, inflate_cert_local, _, inflate_stats_local = primitive_map_recency_inflate(
            atlas_map=map_local,
            tile_ids=active_tile_ids_local,
            scan_seq=int(scan_seq),
            recency_decay_lambda=float(config.RECENCY_DECAY_LAMBDA),
            min_scale=float(config.RECENCY_MIN_SCALE),
            chart_id=belief_pred.chart_id,
            anchor_id="primitive_map_recency_inflate",
        )
        local_certs.append(inflate_cert_local)

        map_view_local = extract_atlas_map_view(
            atlas_map=map_local,
            tile_ids=stencil_tile_ids_local,
            m_tile_view=int(config.M_TILE_VIEW),
            eps_lift=config.eps_lift,
            eps_mass=config.eps_mass,
        )

        assoc_config = AssociationConfig(
            k_assoc=config.k_assoc,
            k_sinkhorn=config.k_sinkhorn,
            epsilon=config.ot_epsilon,
            tau_a=config.ot_tau_a,
            tau_b=config.ot_tau_b,
            eps_mass=config.eps_mass,
            h_tile=float(config.H_TILE),
            r_stencil_tiles_xy=int(config.R_STENCIL_TILES_XY),
            r_stencil_tiles_z=int(config.R_STENCIL_TILES_Z),
            scan_seq=int(scan_seq),
            recency_decay_lambda=float(config.RECENCY_DECAY_LAMBDA),
        )
        assoc_result_local, assoc_cert_local, _ = associate_primitives_ot(
            measurement_batch=measurement_batch_local,
            map_view=map_view_local,
            config=assoc_config,
            chart_id=belief_pred.chart_id,
            anchor_id="primitive_ot",
        )
        local_certs.append(assoc_cert_local)

        candidate_tiles_per_meas_mean_local = 0.0
        candidate_primitives_per_meas_mean_local = 0.0
        candidate_primitives_per_meas_p95_local = 0.0
        if measurement_batch_local is not None:
            n_valid_meas = int(jnp.sum(measurement_batch_local.valid_mask))
            if n_valid_meas > 0:
                cand_valid = map_view_local.valid_mask[assoc_result_local.candidate_pool_indices]
                cand_tiles = jnp.where(cand_valid, assoc_result_local.candidate_tile_ids, -1)
                cand_counts = jnp.sum(cand_valid.astype(jnp.float64), axis=1)

                cand_tiles_sorted = jnp.sort(cand_tiles, axis=1)
                is_valid_tile = cand_tiles_sorted != -1
                is_new = jnp.concatenate(
                    [jnp.ones((cand_tiles_sorted.shape[0], 1), dtype=bool), cand_tiles_sorted[:, 1:] != cand_tiles_sorted[:, :-1]],
                    axis=1,
                )
                distinct_tiles = jnp.sum((is_new & is_valid_tile).astype(jnp.float64), axis=1)

                valid_rows_f = measurement_batch_local.valid_mask.astype(jnp.float64)
                denom = jnp.maximum(jnp.sum(valid_rows_f), config.eps_mass)
                candidate_tiles_per_meas_mean_local = float(jnp.sum(distinct_tiles * valid_rows_f) / denom)
                candidate_primitives_per_meas_mean_local = float(jnp.sum(cand_counts * valid_rows_f) / denom)
                cand_counts_valid = jnp.where(measurement_batch_local.valid_mask, cand_counts, -1.0)
                cand_sorted = jnp.sort(cand_counts_valid)
                idx_p95 = int(0.95 * float(cand_sorted.shape[0]))
                idx_p95 = min(idx_p95, int(cand_sorted.shape[0]) - 1)
                candidate_primitives_per_meas_p95_local = float(cand_sorted[idx_p95])

        return {
            "certs": local_certs,
            "t0_prim": t0_prim_local,
            "measurement_batch": measurement_batch_local,
            "surfel_cert": surfel_cert_local,
            "assoc_result": assoc_result_local,
            "assoc_cert": assoc_cert_local,
            "map_view": map_view_local,
            "primitive_map": map_local,
            "active_tile_ids": active_tile_ids_local,
            "stencil_tile_ids": stencil_tile_ids_local,
            "candidate_tiles_per_meas_mean": candidate_tiles_per_meas_mean_local,
            "candidate_primitives_per_meas_mean": candidate_primitives_per_meas_mean_local,
            "candidate_primitives_per_meas_p95": candidate_primitives_per_meas_p95_local,
            "staleness_inflation_strength": float(inflate_stats_local.staleness_inflation_strength),
            "staleness_cov_inflation_trace": float(inflate_stats_local.staleness_cov_inflation_trace),
            "stale_precision_downscale_total": float(inflate_stats_local.stale_precision_downscale_total),
        }

    primitive_map_updated = None
    measurement_batch = None
    n_primitives_inserted = 0
    n_primitives_fused = 0
    n_primitives_culled = 0
    n_primitives_merged = 0
    fused_mass_total = 0.0
    insert_mass_total = 0.0
    insert_mass_p95 = 0.0
    evicted_mass_total = 0.0
    candidate_tiles_per_meas_mean = 0.0
    candidate_primitives_per_meas_mean = 0.0
    candidate_primitives_per_meas_p95 = 0.0
    staleness_inflation_strength = 0.0
    staleness_cov_inflation_trace = 0.0
    stale_precision_downscale_total = 0.0
    assoc_result = None
    map_view = None
    active_tile_ids: Optional[List[int]] = None
    stencil_tile_ids: Optional[List[int]] = None
    event_log_entries: List[dict] = []

    surfel_cert = None
    assoc_cert = None
    visual_cert = None

    if config.enable_parallel_stages and primitive_map is not None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            imu_future = executor.submit(_compute_imu_odom_branch)
            map_future = executor.submit(_compute_map_branch)
            imu_out = imu_future.result()
            map_out = map_future.result()
    else:
        imu_out = _compute_imu_odom_branch()
        map_out = _compute_map_branch() if primitive_map is not None else None

    all_certs.extend(imu_out["certs"])
    odom_result = imu_out["odom_result"]
    imu_result = imu_out["imu_result"]
    gyro_result = imu_out["gyro_result"]
    preint_result = imu_out["preint_result"]
    planar_result = imu_out["planar_result"]
    vz_result = imu_out["vz_result"]
    odom_vel_result = imu_out["odom_vel_result"]
    odom_wz_result = imu_out["odom_wz_result"]
    kinematic_result = imu_out["kinematic_result"]
    dep_result = imu_out["dep_result"]
    odom_dep_result = imu_out["odom_dep_result"]
    L_imu_odom = imu_out["L_imu_odom"]
    h_imu_odom = imu_out["h_imu_odom"]
    z_lin_pose = imu_out["z_lin_pose"]
    pose_pred = imu_out["pose_pred"]

    if map_out is not None:
        all_certs.extend(map_out["certs"])
        measurement_batch = map_out["measurement_batch"]
        surfel_cert = map_out["surfel_cert"]
        assoc_result = map_out["assoc_result"]
        assoc_cert = map_out["assoc_cert"]
        map_view = map_out["map_view"]
        primitive_map = map_out["primitive_map"]
        active_tile_ids = map_out["active_tile_ids"]
        stencil_tile_ids = map_out["stencil_tile_ids"]
        candidate_tiles_per_meas_mean = map_out["candidate_tiles_per_meas_mean"]
        candidate_primitives_per_meas_mean = map_out["candidate_primitives_per_meas_mean"]
        candidate_primitives_per_meas_p95 = map_out["candidate_primitives_per_meas_p95"]
        staleness_inflation_strength = map_out["staleness_inflation_strength"]
        staleness_cov_inflation_trace = map_out["staleness_cov_inflation_trace"]
        stale_precision_downscale_total = map_out["stale_precision_downscale_total"]

        visual_result, visual_cert, _ = visual_pose_evidence(
            association_result=assoc_result,
            measurement_batch=measurement_batch,
            map_view=map_view,
            belief_pred=belief_pred,
            eps_lift=config.eps_lift,
            eps_mass=config.eps_mass,
            chart_id=belief_pred.chart_id,
            anchor_id="visual_pose_evidence",
            z_lin_pose=z_lin_pose,
        )
        all_certs.append(visual_cert)
        L_lidar, h_lidar = build_visual_pose_evidence_22d(visual_result)
        if config.enable_timing:
            _record_timing("visual_pose_evidence_ms", map_out["t0_prim"], L_lidar)
    else:
        L_lidar = config.eps_lift * jnp.eye(22, dtype=jnp.float64)
        h_lidar = jnp.zeros((22,), dtype=jnp.float64)

    # =========================================================================
    # Step 5: Fuse evidence (IMU+odom+L_pose) and recompose -> z_t
    # =========================================================================
    # Evidence uses pre-update map; z_lin was IMU+odom-informed. No second odom/IMU block.

    # Measurement-noise IW sufficient stats: only IMU (gyro + accel)
    # LiDAR IW stats are now implicit in the primitive precisions
    iw_meas_dPsi = iw_meas_gyro_dPsi + iw_meas_accel_dPsi
    iw_meas_dnu = iw_meas_gyro_dnu + iw_meas_accel_dnu

    # LiDAR bucket IW stats are removed
    # Fixed-cost: primitive precisions handle measurement noise
    iw_lidar_bucket_dPsi = jnp.zeros((64, 3, 3), dtype=jnp.float64)  # Placeholder shape
    iw_lidar_bucket_dnu = jnp.zeros((64,), dtype=jnp.float64)
    
    # =========================================================================
    # Step 9: Evidence (raw) = IMU+odom + visual pose
    # =========================================================================
    # No duplicate odom/IMU block; L_imu_odom, h_imu_odom already built for z_lin.
    # We first construct the raw evidence, then optionally apply a single-pass tempered-posterior
    # scaling (power EP / generalized Bayes) as continuous conservatism (no gating, no iteration).
    L_evidence_raw = L_imu_odom + L_lidar
    h_evidence_raw = h_imu_odom + h_lidar

    # Diagnostic: check which evidence component has NaN. JAX checks; sync only on failure.
    for name, L in [("L_lidar", L_lidar), ("L_odom", odom_result.L_odom),
                    ("L_imu", imu_result.L_imu), ("L_gyro", gyro_result.L_gyro),
                    ("L_imu_preint", preint_result.L_imu_preint)]:
        if not bool(jnp.all(jnp.isfinite(L))):
            nan_pos = np.array(jnp.argwhere(~jnp.isfinite(L))[:5])
            raise ValueError(f"{name} contains NaN at positions {nan_pos.tolist()}")

    # -------------------------------------------------------------------------
    # Power tempering (closed form, fixed budget):
    # L_evidence = beta * L_evidence_raw, h_evidence = beta * h_evidence_raw
    #
    # beta is computed from certificate-level sentinels to reduce overconfidence under dependence
    # and partial observability. This is a single, declared scaling (no hidden heuristics).
    # -------------------------------------------------------------------------
    # Build evidence cert from available LiDAR-related certificates (deskew always; surfel/assoc/visual when primitive path ran)
    lidar_certs = [deskew_cert]
    if surfel_cert is not None:
        lidar_certs.append(surfel_cert)
    if assoc_cert is not None:
        lidar_certs.append(assoc_cert)
    if visual_cert is not None:
        lidar_certs.append(visual_cert)
    evidence_cert = aggregate_certificates(lidar_certs)
    combined_evidence_cert = aggregate_certificates(
        [evidence_cert, imu_out["odom_cert"], imu_out["imu_cert"], imu_out["gyro_cert"]]
    )

    # Observability sentinels are computed from the *raw* evidence to avoid fixed-point iteration.
    eps = float(config.eps_mass)
    dt_pose_raw = jnp.linalg.norm(
        L_evidence_raw[constants.GC_IDX_DT, constants.GC_IDX_POSE]
    ) + jnp.linalg.norm(
        L_evidence_raw[constants.GC_IDX_POSE, constants.GC_IDX_DT]
    )
    dt_vel_raw = jnp.linalg.norm(
        L_evidence_raw[constants.GC_IDX_DT, constants.GC_IDX_VEL]
    ) + jnp.linalg.norm(
        L_evidence_raw[constants.GC_IDX_VEL, constants.GC_IDX_DT]
    )
    dt_asym_raw = jnp.abs(dt_vel_raw - dt_pose_raw) / (dt_vel_raw + dt_pose_raw + eps)
    dt_asym_raw = jnp.clip(dt_asym_raw, 0.0, 1.0)
    L_xx_raw = jnp.abs(L_evidence_raw[0, 0])
    L_yy_raw = jnp.abs(L_evidence_raw[1, 1])
    L_zz_raw = jnp.abs(L_evidence_raw[2, 2])
    z_to_xy_raw = L_zz_raw / (0.5 * (L_xx_raw + L_yy_raw) + eps)

    combined_evidence_cert.overconfidence.dt_asymmetry = float(dt_asym_raw)
    combined_evidence_cert.overconfidence.z_to_xy_ratio = float(z_to_xy_raw)

    # Closed-form tempering beta from sentinels (continuous; bounded).
    # This mapping is a declared control law; tune parameters in PipelineConfig.
    exc_total = combined_evidence_cert.excitation.dt_effect + combined_evidence_cert.excitation.extrinsic_effect
    ess_total = combined_evidence_cert.support.ess_total
    ess_to_exc = float(ess_total) / (float(exc_total) + float(config.eps_mass))

    s_dt = dt_asym_raw
    s_z = float(z_to_xy_raw) / (float(z_to_xy_raw) + float(config.power_beta_z_c))
    s_exc = 1.0 / (1.0 + (ess_to_exc / float(config.power_beta_exc_c)))
    s = jnp.clip(jnp.asarray(s_dt * s_z * s_exc, dtype=jnp.float64), 0.0, 1.0)
    beta = float(config.power_beta_min + (1.0 - config.power_beta_min) * float(s))
    beta = float(jnp.clip(jnp.asarray(beta, dtype=jnp.float64), config.power_beta_min, 1.0))

    # Apply tempering to evidence (power posterior)
    L_evidence = beta * L_evidence_raw
    h_evidence = beta * h_evidence_raw
    combined_evidence_cert.influence.power_beta = beta
    temper_cert = CertBundle.create_approx(
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
        triggers=["PowerTempering"],
        frobenius_applied=abs(1.0 - beta) > 0.0,
        influence=InfluenceCert.identity().with_overrides(
            power_beta=beta,
        ),
    )
    all_certs.append(temper_cert)

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
        influence=InfluenceCert.identity().with_overrides(
            dt_scale=exc_dt_scale,
            extrinsic_scale=exc_ex_scale,
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

    # Effective conditioning for trust alpha: evaluated on pose block (6x6) in JAX; no host round-trip.
    # Full 22x22 would be dominated by physically-null directions (yaw under gravity, weak bias/extrinsic).
    eps_cond = float(config.eps_psd)
    L_pose = 0.5 * (
        L_evidence[constants.GC_IDX_POSE, constants.GC_IDX_POSE]
        + L_evidence[constants.GC_IDX_POSE, constants.GC_IDX_POSE].T
    )
    L_pose = jnp.nan_to_num(L_pose, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals_pose = jnp.linalg.eigvalsh(L_pose)
    eigvals_safe = jnp.nan_to_num(eigvals_pose, nan=eps_cond, posinf=eps_cond, neginf=eps_cond)
    eigvals_clipped = jnp.maximum(eigvals_safe, eps_cond)
    eig_min_pose = eigvals_clipped[0]
    eig_max_pose = eigvals_clipped[-1]
    cond_pose6 = eig_max_pose / eig_min_pose
    near_null_count = jnp.sum(eigvals_pose <= eps_cond).astype(jnp.int32)

    combined_evidence_cert.conditioning = ConditioningCert(
        eig_min=float(eig_min_pose),
        eig_max=float(eig_max_pose),
        cond=float(cond_pose6),
        near_null_count=int(near_null_count),
    )
    combined_evidence_cert.approximation_triggers.append("FusionScaleConditioningPose6")  # pose6 block only

    # Note: dt/z sentinels already computed from raw evidence and stored in combined_evidence_cert.overconfidence.

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
    # Step 12b: Primitive map update with z_t (post-recompose pose)
    # =========================================================================
    # Evidence used pre-update map; now update map using z_t (belief_recomposed).
    if (
        primitive_map is not None
        and assoc_result is not None
        and map_view is not None
        and measurement_batch is not None
        and active_tile_ids is not None
    ):
        z_t = belief_recomposed.mean_world_pose(eps_lift=config.eps_lift)
        R_t = se3_jax.so3_exp(z_t[3:6])
        t_t = z_t[:3]

        def transform_gaussian_to_world(Lambda_b, theta_b, eta_b):
            Lambda_w = R_t @ Lambda_b @ R_t.T
            Lambda_reg = Lambda_b + config.eps_lift * jnp.eye(3)
            mu_b = jnp.linalg.solve(Lambda_reg, theta_b)
            mu_w = R_t @ mu_b + t_t
            theta_w = Lambda_w @ mu_w
            # Multi-lobe vMF: eta_b is (B,3); rotate each lobe into world.
            eta_w = (R_t @ eta_b.T).T
            return Lambda_w, theta_w, eta_w

        if active_tile_ids is not None:
            (
                meas_idx_blocks,
                tile_blocks,
                slot_blocks,
                resp_blocks,
                valid_rows,
            ) = block_associations_for_fuse(
                result=assoc_result,
                valid_mask=measurement_batch.valid_mask,
                block_size=constants.GC_ASSOC_BLOCK_SIZE,
            )
            n_blocks = meas_idx_blocks.shape[0]
            k_assoc = slot_blocks.shape[2]
            for b in range(n_blocks):
                meas_idx = meas_idx_blocks[b]
                tile_blk = tile_blocks[b]
                slot_blk = slot_blocks[b]
                resp_blk = resp_blocks[b]
                valid_rows_blk = valid_rows[b]

                Lambdas_blk = measurement_batch.Lambdas[meas_idx]
                thetas_blk = measurement_batch.thetas[meas_idx]
                etas_blk = measurement_batch.etas[meas_idx]
                weights_blk = measurement_batch.weights[meas_idx]
                colors_blk = measurement_batch.colors[meas_idx]

                target_flat = slot_blk.reshape(-1).astype(jnp.int32)
                tile_ids_flat = tile_blk.reshape(-1).astype(jnp.int64)
                responsibilities_fuse = resp_blk.reshape(-1)
                valid_flat = jnp.repeat(valid_rows_blk, k_assoc)
                Lambdas_meas = jnp.repeat(Lambdas_blk, k_assoc, axis=0)
                thetas_meas = jnp.repeat(thetas_blk, k_assoc, axis=0)
                etas_meas = jnp.repeat(etas_blk, k_assoc, axis=0)
                weights_meas = jnp.repeat(weights_blk, k_assoc, axis=0)
                colors_meas_flat = jnp.repeat(colors_blk, k_assoc, axis=0)

                Lambdas_world, thetas_world, etas_world = jax.vmap(transform_gaussian_to_world)(
                    Lambdas_meas, thetas_meas, etas_meas
                )

                # Per-tile fuse: run fixed loop over active tiles; use masks (no gating).
                for tid in active_tile_ids:
                    tid_i = int(tid)
                    tile_mask = (tile_ids_flat == tid_i)
                    valid_t = valid_flat & tile_mask
                    fused_mass_total += float(
                        jnp.sum(weights_meas * responsibilities_fuse * valid_t.astype(jnp.float64))
                    )
                    fuse_result, fuse_cert, fuse_effect = primitive_map_fuse(
                        atlas_map=primitive_map,
                        tile_id=tid_i,
                        target_slots=target_flat,
                        Lambdas_meas=Lambdas_world,
                        thetas_meas=thetas_world,
                        etas_meas=etas_world,
                        weights_meas=weights_meas,
                        responsibilities=responsibilities_fuse,
                        timestamp=scan_end_time,
                        scan_seq=int(scan_seq),
                        valid_mask=valid_t,
                        colors_meas=colors_meas_flat,
                        eps_mass=config.eps_mass,
                    )
                    primitive_map = fuse_result.atlas_map
                    all_certs.append(fuse_cert)
                    n_primitives_fused += fuse_result.n_fused

        # Fixed-budget insertion every scan: novelty mass from unbalanced OT coupling.
        # This replaces the legacy "insert only when map empty" behavior.
        if assoc_result is not None and active_tile_ids is not None:
            a = measurement_batch.valid_mask.astype(jnp.float64)
            a = a / jnp.maximum(jnp.sum(a), config.eps_mass)  # fixed total mass
            row_mass = assoc_result.row_masses.astype(jnp.float64)
            novelty = jnp.maximum(a - row_mass, 0.0)
            # Prefer inserting genuinely novel evidence; never select padded invalid rows.
            score = novelty * measurement_batch.weights.astype(jnp.float64)
            score = score - (1.0 - measurement_batch.valid_mask.astype(jnp.float64)) * 1e6
            k_ins = int(config.k_insert_tile)

            # Assign each measurement to a tile by its world-frame mean position at z_t.
            Lambda_reg_all = measurement_batch.Lambdas + config.eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]
            mu_body = jax.vmap(jnp.linalg.solve)(Lambda_reg_all, measurement_batch.thetas)
            mu_world = (R_t @ mu_body.T).T + t_t[None, :]
            meas_tile_ids = tile_ids_from_xyz_batch_jax(mu_world, h_tile=float(config.H_TILE))

            # Per-tile insertion with fixed budget K_INSERT_TILE, masked when no matching measurements exist.
            for tid in active_tile_ids:
                tid_i = int(tid)
                in_tile = meas_tile_ids == jnp.asarray(tid_i, dtype=jnp.int64)
                score_t = jnp.where(in_tile, score, jnp.asarray(-1e30, dtype=jnp.float64))
                ins_idx_j = jnp.argsort(-score_t)[:k_ins].astype(jnp.int32)
                valid_new_j = in_tile[ins_idx_j] & (score_t[ins_idx_j] > -1e20)
                # Enforce fixed insert count: if no in-tile measurements, insert zero-mass placeholders.
                valid_new_j = jnp.where(jnp.any(valid_new_j), valid_new_j, jnp.ones_like(valid_new_j, dtype=bool))

                Lambdas_ins = measurement_batch.Lambdas[ins_idx_j]
                thetas_ins = measurement_batch.thetas[ins_idx_j]
                etas_ins = measurement_batch.etas[ins_idx_j]
                weights_ins = (novelty[ins_idx_j] * measurement_batch.weights[ins_idx_j]).astype(jnp.float64)
                # If we had to fill budget, zero out those insert masses explicitly.
                weights_ins = jnp.where(in_tile[ins_idx_j], weights_ins, 0.0)
                colors_ins = measurement_batch.colors[ins_idx_j]

                insert_mass_total += float(jnp.sum(weights_ins))
                weights_sorted = jnp.sort(weights_ins)
                if int(weights_sorted.shape[0]) > 0:
                    idx_p95 = int(0.95 * float(weights_sorted.shape[0]))
                    idx_p95 = min(idx_p95, int(weights_sorted.shape[0]) - 1)
                    insert_mass_p95 = max(insert_mass_p95, float(weights_sorted[idx_p95]))

                Lambdas_world_ins, thetas_world_ins, etas_world_ins = jax.vmap(transform_gaussian_to_world)(
                    Lambdas_ins, thetas_ins, etas_ins
                )
                result_insert, insert_cert, insert_effect = primitive_map_insert_masked(
                    atlas_map=primitive_map,
                    tile_id=tid_i,
                    Lambdas_new=Lambdas_world_ins,
                    thetas_new=thetas_world_ins,
                    etas_new=etas_world_ins,
                    weights_new=weights_ins,
                    timestamp=scan_end_time,
                    scan_seq=int(scan_seq),
                    valid_new_mask=valid_new_j,
                    recency_decay_lambda=float(config.RECENCY_DECAY_LAMBDA),
                    colors_new=colors_ins,
                )
                primitive_map = result_insert.atlas_map
                all_certs.append(insert_cert)
                n_primitives_inserted += int(result_insert.n_inserted)
                # Event log payloads (for Rerun replay): record inserted primitives
                if result_insert.n_inserted > 0:
                    import numpy as np
                    Lambda_reg = Lambdas_world_ins + config.eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]
                    mu_world_ins = jax.vmap(jnp.linalg.solve)(Lambda_reg, thetas_world_ins)
                    mu_np = np.array(mu_world_ins, dtype=np.float64)
                    w_np = np.array(weights_ins, dtype=np.float64)
                    c_np = np.array(colors_ins, dtype=np.float64)
                    for i_ins in range(mu_np.shape[0]):
                        event_log_entries.append(
                            {
                                "tile_id": tid_i,
                                "mu_world": mu_np[i_ins].tolist(),
                                "weight": float(w_np[i_ins]),
                                "color": c_np[i_ins].tolist(),
                                "timestamp": float(scan_end_time),
                            }
                        )

        # Per-tile maintenance (local by construction): cull + forget for active tiles only.
        if active_tile_ids is not None:
            for tid in active_tile_ids:
                tid_i = int(tid)
                cull_result, cull_cert, cull_effect = primitive_map_cull(
                    atlas_map=primitive_map,
                    tile_id=tid_i,
                    weight_threshold=config.primitive_cull_weight_threshold,
                )
                primitive_map = cull_result.atlas_map
                all_certs.append(cull_cert)
                n_primitives_culled += int(cull_result.n_culled)
                evicted_mass_total += float(cull_result.mass_dropped)

                forget_result, forget_cert, forget_effect = primitive_map_forget(
                    atlas_map=primitive_map,
                    tile_id=tid_i,
                    forgetting_factor=config.primitive_forgetting_factor,
                )
                primitive_map = forget_result.atlas_map
                all_certs.append(forget_cert)

                merge_result, merge_cert, merge_effect = primitive_map_merge_reduce(
                    atlas_map=primitive_map,
                    tile_id=tid_i,
                    merge_threshold=config.primitive_merge_threshold,
                    max_pairs=int(config.k_merge_pairs_tile),
                    max_tile_size=int(config.primitive_merge_max_tile_size),
                    eps_psd=config.eps_psd,
                    eps_lift=config.eps_lift,
                    chart_id=belief_pred.chart_id,
                    anchor_id="primitive_map_merge_reduce",
                )
                primitive_map = merge_result.atlas_map
                all_certs.append(merge_cert)
                n_primitives_merged += int(merge_result.n_merged)

        # Cache hit/miss for active tiles.
        active_set = set(active_tile_ids or [])
        tile_cache_hits = len([t for t in (active_tile_ids or []) if t in primitive_map.tiles])
        tile_cache_misses = len(active_set) - tile_cache_hits

        map_update_cert = CertBundle.create_exact(
            chart_id=belief_pred.chart_id,
            anchor_id="map_update",
            map_update=MapUpdateCert(
                n_active_tiles=int(len(active_tile_ids)) if active_tile_ids is not None else 0,
                tile_ids_active=[int(t) for t in (active_tile_ids or [])],
                n_inactive_tiles=int(
                    len([t for t in primitive_map.tile_ids if t not in set(active_tile_ids or [])])
                )
                if primitive_map is not None
                else 0,
                staleness_inflation_strength=float(staleness_inflation_strength),
                staleness_cov_inflation_trace=float(staleness_cov_inflation_trace),
                stale_precision_downscale_total=float(stale_precision_downscale_total),
                tile_ids_inactive=[
                    int(t) for t in primitive_map.tile_ids if t not in set(active_tile_ids or [])
                ]
                if primitive_map is not None
                else [],
                tile_cache_hits=int(tile_cache_hits),
                tile_cache_misses=int(tile_cache_misses),
                candidate_tiles_per_meas_mean=float(candidate_tiles_per_meas_mean),
                candidate_primitives_per_meas_mean=float(candidate_primitives_per_meas_mean),
                candidate_primitives_per_meas_p95=float(candidate_primitives_per_meas_p95),
                insert_count_total=int(n_primitives_inserted),
                insert_mass_total=float(insert_mass_total),
                insert_mass_p95=float(insert_mass_p95),
                evicted_count=int(n_primitives_culled),
                evicted_mass_total=float(evicted_mass_total),
                fused_count=int(n_primitives_fused),
                fused_mass_total=float(fused_mass_total),
                merged_count=int(n_primitives_merged),
            ),
        )
        all_certs.append(map_update_cert)

        primitive_map_updated = primitive_map
    
    # =========================================================================
    # Step 13: AnchorDriftUpdate
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
    total_trigger_mag = sum(c.total_trigger_magnitude() for c in all_certs)
    cond_number = (
        aggregated_cert.conditioning.cond
        if aggregated_cert.conditioning is not None
        else 1.0
    )

    diagnostics_tape = None
    # Minimal tape only (crash-tolerant, low overhead) + certificate summary.
    import numpy as np
    cert_support = aggregated_cert.support
    cert_mismatch = aggregated_cert.mismatch
    cert_excitation = aggregated_cert.excitation
    cert_influence = aggregated_cert.influence
    cert_over = aggregated_cert.overconfidence

    if config.enable_timing:
        timing_ms["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0

    diagnostics_tape = MinimalScanTape(
        scan_number=0,
        timestamp=scan_end_time,
        dt_sec=dt_sec,
        n_points_raw=int(raw_points.shape[0]),
        n_points_budget=int(points.shape[0]),
        fusion_alpha=float(alpha),
        cond_pose6=float(cond_pose6),
        conditioning_number=cond_number,
        eigmin_pose6=float(eig_min_pose),
        L_pose6=np.array(L_evidence[constants.GC_IDX_POSE, constants.GC_IDX_POSE], dtype=np.float64),
        total_trigger_magnitude=total_trigger_mag,
        cert_exact=bool(aggregated_cert.exact),
        cert_frobenius_applied=bool(aggregated_cert.frobenius_applied),
        cert_n_triggers=int(len(aggregated_cert.approximation_triggers)),
        support_ess_total=float(cert_support.ess_total),
        support_frac=float(cert_support.support_frac),
        mismatch_nll_per_ess=float(cert_mismatch.nll_per_ess),
        mismatch_directional_score=float(cert_mismatch.directional_score),
        excitation_dt_effect=float(cert_excitation.dt_effect),
        excitation_extrinsic_effect=float(cert_excitation.extrinsic_effect),
        influence_psd_projection_delta=float(cert_influence.psd_projection_delta),
        influence_mass_epsilon_ratio=float(cert_influence.mass_epsilon_ratio),
        influence_anchor_drift_rho=float(cert_influence.anchor_drift_rho),
        influence_dt_scale=float(cert_influence.dt_scale),
        influence_extrinsic_scale=float(cert_influence.extrinsic_scale),
        influence_trust_alpha=float(cert_influence.trust_alpha),
        influence_power_beta=float(cert_influence.power_beta),
        overconfidence_excitation_total=float(cert_over.excitation_total),
        overconfidence_ess_to_excitation=float(cert_over.ess_to_excitation),
        overconfidence_cond_to_support=float(cert_over.cond_to_support),
        overconfidence_dt_asymmetry=float(cert_over.dt_asymmetry),
        overconfidence_z_to_xy_ratio=float(cert_over.z_to_xy_ratio),
        t_total_ms=timing_ms.get("total_ms", 0.0) if timing_ms is not None else 0.0,
        t_point_budget_ms=timing_ms.get("point_budget_ms", 0.0) if timing_ms is not None else 0.0,
        t_deskew_ms=timing_ms.get("deskew_ms", 0.0) if timing_ms is not None else 0.0,
    )

    return ScanPipelineResult(
        belief_updated=belief_final,
        iw_process_dPsi=iw_process_dPsi,
        iw_process_dnu=iw_process_dnu,
        iw_meas_dPsi=iw_meas_dPsi,
        iw_meas_dnu=iw_meas_dnu,
        iw_lidar_bucket_dPsi=iw_lidar_bucket_dPsi,
        iw_lidar_bucket_dnu=iw_lidar_bucket_dnu,
        all_certs=all_certs,
        aggregated_cert=aggregated_cert,
        diagnostics_tape=diagnostics_tape,
        # Stage 1: PrimitiveMap results
        primitive_map_updated=primitive_map_updated,
        measurement_batch=measurement_batch,
        n_primitives_inserted=n_primitives_inserted,
        n_primitives_fused=n_primitives_fused,
        n_primitives_culled=n_primitives_culled,
        n_primitives_merged=n_primitives_merged,
        event_log_entries=event_log_entries if event_log_entries else None,
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
    N_POINTS_CAP: int = constants.GC_N_POINTS_CAP
    N_FEAT: int = constants.GC_N_FEAT
    N_SURFEL: int = constants.GC_N_SURFEL
    K_SINKHORN: int = constants.GC_K_SINKHORN
    
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
    deskew_rotation_only: bool = False
    power_beta_min: float = 0.25
    power_beta_exc_c: float = 50.0
    power_beta_z_c: float = 1.0
    enable_parallel_stages: bool = False

    # OT association parameters (spec §5.7)
    ot_epsilon: float = 0.1  # Entropic regularization
    ot_tau_a: float = 0.5  # Unbalanced KL for measurement marginal
    ot_tau_b: float = 0.5  # Unbalanced KL for map marginal
    ot_iters: int = constants.GC_K_SINKHORN  # Fixed Sinkhorn iterations (k_sinkhorn)

    # Fixed-cost budgets (compile-time constants, spec §6)
    K_ASSOC: int = constants.GC_K_ASSOC  # Candidate neighborhood size
    K_INSERT_TILE: int = constants.GC_K_INSERT_TILE  # Insertion budget per tile
    K_MERGE_PAIRS_TILE: int = constants.GC_K_MERGE_PAIRS_PER_TILE  # Merge-reduce budget per tile
    MERGE_MAX_TILE_SIZE: int = constants.GC_PRIMITIVE_MERGE_MAX_TILE_SIZE  # Merge-reduce cap
    M_TILE: int = constants.GC_PRIMITIVE_MAP_MAX_SIZE  # Max primitives per tile (single-tile for now)
    # Atlas tiling budgets (spec §5.7)
    H_TILE: float = constants.GC_H_TILE
    N_ACTIVE_TILES: int = constants.GC_N_ACTIVE_TILES
    R_ACTIVE_TILES_XY: int = constants.GC_R_ACTIVE_TILES_XY
    R_ACTIVE_TILES_Z: int = constants.GC_R_ACTIVE_TILES_Z
    M_TILE_VIEW: int = constants.GC_M_TILE_VIEW
    N_STENCIL_TILES: int = constants.GC_N_STENCIL_TILES
    R_STENCIL_TILES_XY: int = constants.GC_R_STENCIL_TILES_XY
    R_STENCIL_TILES_Z: int = constants.GC_R_STENCIL_TILES_Z
    ASSOC_BLOCK_SIZE: int = constants.GC_ASSOC_BLOCK_SIZE
    FUSE_CHUNK_SIZE: int = constants.GC_FUSE_CHUNK_SIZE
    MAX_IMU_PREINT_LEN: int = constants.GC_MAX_IMU_PREINT_LEN
    VMF_N_LOBES: int = constants.GC_VMF_N_LOBES

    # BEV (future view-layer) flags
    bev_backend_enabled: bool = False
    bev_views_n: int = 0

    # Explicit backend/operator selections (single-path; no fallback).
    # Required for auditability of the "no multipaths" invariant.
    # pose_evidence_backend and map_backend are the single source of truth for pose/map path.
    pose_evidence_backend: str = constants.GC_POSE_EVIDENCE_BACKEND_PRIMITIVES
    map_backend: str = constants.GC_MAP_BACKEND_PRIMITIVE_MAP
    backends: Dict[str, str] = None
    topics: Dict[str, str] = None

    def __post_init__(self):
        if self.topics is None:
            self.topics = {}
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
                "lidar_evidence": "fl_slam_poc.backend.operators.visual_pose_evidence (primitive alignment; Laplace at z_lin)",
                "hypothesis_barycenter": "fl_slam_poc.backend.operators.hypothesis (vectorized over hypotheses)",
                "map_update": "fl_slam_poc.backend.structures.primitive_map (Fuse/Insert/Cull/Forget/MergeReduce)",
                "lidar_converter": "fl_slam_poc.frontend.sensors.pointcloud_passthrough",
                "pointcloud_parser": "fl_slam_poc.backend.backend_node.parse_pointcloud2",
            }
        # Single OT path: unbalanced Sinkhorn only (no balanced path).
        self.backends["sinkhorn_backend"] = "unbalanced_fixed_k"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/publishing."""
        return {
            "chart_id": self.chart_id,
            "pose_evidence_backend": self.pose_evidence_backend,
            "map_backend": self.map_backend,
            "topics": dict(self.topics),
            "D_Z": self.D_Z,
            "D_DESKEW": self.D_DESKEW,
            "K_HYP": self.K_HYP,
            "HYP_WEIGHT_FLOOR": self.HYP_WEIGHT_FLOOR,
            "N_POINTS_CAP": self.N_POINTS_CAP,
            "N_FEAT": self.N_FEAT,
            "N_SURFEL": self.N_SURFEL,
            "K_SINKHORN": self.K_SINKHORN,
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
            "deskew_rotation_only": self.deskew_rotation_only,
            "power_beta_min": self.power_beta_min,
            "power_beta_exc_c": self.power_beta_exc_c,
            "power_beta_z_c": self.power_beta_z_c,
            "enable_parallel_stages": self.enable_parallel_stages,
            # OT params (spec §5.7)
            "ot_epsilon": self.ot_epsilon,
            "ot_tau_a": self.ot_tau_a,
            "ot_tau_b": self.ot_tau_b,
            "ot_iters": self.ot_iters,
            # Fixed-cost budgets (spec §6)
            "K_ASSOC": self.K_ASSOC,
            "K_INSERT_TILE": self.K_INSERT_TILE,
            "K_MERGE_PAIRS_TILE": self.K_MERGE_PAIRS_TILE,
            "M_TILE": self.M_TILE,
            # Atlas tiling budgets (spec §5.7)
            "H_TILE": self.H_TILE,
            "N_ACTIVE_TILES": self.N_ACTIVE_TILES,
            "R_ACTIVE_TILES_XY": self.R_ACTIVE_TILES_XY,
            "R_ACTIVE_TILES_Z": self.R_ACTIVE_TILES_Z,
            "M_TILE_VIEW": self.M_TILE_VIEW,
            "N_STENCIL_TILES": self.N_STENCIL_TILES,
            "R_STENCIL_TILES_XY": self.R_STENCIL_TILES_XY,
            "R_STENCIL_TILES_Z": self.R_STENCIL_TILES_Z,
            "ASSOC_BLOCK_SIZE": self.ASSOC_BLOCK_SIZE,
            "FUSE_CHUNK_SIZE": self.FUSE_CHUNK_SIZE,
            "MAX_IMU_PREINT_LEN": self.MAX_IMU_PREINT_LEN,
            "VMF_N_LOBES": self.VMF_N_LOBES,
            "bev_backend_enabled": self.bev_backend_enabled,
            "bev_views_n": self.bev_views_n,
            "backends": dict(self.backends),
        }
