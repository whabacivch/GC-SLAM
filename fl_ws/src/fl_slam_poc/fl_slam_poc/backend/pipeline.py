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
from fl_slam_poc.backend.operators.wahba import wahba_svd
from fl_slam_poc.backend.operators.translation import translation_wls
from fl_slam_poc.backend.operators.measurement_noise_iw_jax import (
    lidar_meas_iw_suffstats_from_translation_residuals_jax,
    imu_gyro_meas_iw_suffstats_from_avg_rate_jax,
    imu_accel_meas_iw_suffstats_from_gravity_dir_jax,
)
from fl_slam_poc.backend.operators.odom_evidence import odom_quadratic_evidence
from fl_slam_poc.backend.operators.imu_evidence import imu_vmf_gravity_evidence
from fl_slam_poc.backend.operators.imu_gyro_evidence import imu_gyro_rotation_evidence
from fl_slam_poc.backend.operators.lidar_evidence import (
    lidar_quadratic_evidence,
    MapBinStats as LidarMapBinStats,
)
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
    
    def __post_init__(self):
        if self.Sigma_meas is None:
            # Default measurement noise (3x3 isotropic)
            self.Sigma_meas = 0.01 * jnp.eye(3, dtype=jnp.float64)
        if self.Sigma_g is None:
            self.Sigma_g = constants.GC_IMU_GYRO_NOISE_DENSITY * jnp.eye(3, dtype=jnp.float64)
        if self.Sigma_a is None:
            self.Sigma_a = constants.GC_IMU_ACCEL_NOISE_DENSITY * jnp.eye(3, dtype=jnp.float64)


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
    Q: jnp.ndarray,
    bin_atlas: BinAtlas,
    map_stats: MapBinStats,
    config: PipelineConfig,
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
    7. WahbaSVD
    8. TranslationWLS
    9. OdomEvidence + ImuEvidence + LidarEvidence (closed-form/Laplace; no moment-matching)
    10. FusionScaleFromCertificates
    11. InfoFusionAdditive
    12. PoseUpdateFrobeniusRecompose
    13. PoseCovInflationPushforward
    14. AnchorDriftUpdate
    
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
    w_imu = smooth_window_weights(
        imu_stamps=imu_stamps,
        scan_start_time=scan_start_time,
        scan_end_time=scan_end_time,
        sigma=sigma_warp,
    )

    # Biases from predicted belief increment
    mu_inc = belief_pred.mean_increment(eps_lift=config.eps_lift)
    gyro_bias = mu_inc[9:12]
    accel_bias = mu_inc[12:15]

    pose0 = belief_prev.mean_world_pose(eps_lift=config.eps_lift)
    rotvec0 = pose0[3:6]
    gravity_W = jnp.array(constants.GC_GRAVITY_W, dtype=jnp.float64)

    delta_pose, _R_end, _p_end, ess_imu = preintegrate_imu_relative_pose_jax(
        imu_stamps=imu_stamps,
        imu_gyro=imu_gyro,
        imu_accel=imu_accel,
        weights=w_imu,
        rotvec_start_WB=rotvec0,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        gravity_W=gravity_W,
    )
    xi_body = se3_jax.se3_log(delta_pose)  # (6,) twist over interval

    # IMU measurement-noise IW sufficient stats (Σg, Σa) from commutative residuals
    # Average IMU sampling period for PSD mapping (Var * dt -> PSD).
    m_imu = imu_stamps.shape[0]
    dt_imu = (imu_stamps[-1] - imu_stamps[0]) / jnp.maximum(jnp.array(m_imu - 1, dtype=jnp.float64), 1.0)
    dt_imu = jnp.maximum(dt_imu, 1e-12)

    dt_scan = jnp.maximum(jnp.array(scan_end_time - scan_start_time, dtype=jnp.float64), 1e-12)
    omega_avg = delta_pose[3:6] / dt_scan
    iw_meas_gyro_dPsi, iw_meas_gyro_dnu = imu_gyro_meas_iw_suffstats_from_avg_rate_jax(
        imu_gyro=imu_gyro,
        weights=w_imu,
        gyro_bias=gyro_bias,
        omega_avg=omega_avg,
        dt_imu_sec=dt_imu,
        eps_mass=config.eps_mass,
    )
    iw_meas_accel_dPsi, iw_meas_accel_dnu = imu_accel_meas_iw_suffstats_from_gravity_dir_jax(
        rotvec_world_body=rotvec0,
        imu_accel=imu_accel,
        weights=w_imu,
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
        ess_imu=float(ess_imu),
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(deskew_cert)

    deskewed_points = deskew_twist_result.points
    deskewed_weights = deskew_twist_result.weights
    # Point covariances are not used by the new deskew. Keep zeros (deterministic).
    deskewed_covs = jnp.zeros((deskewed_points.shape[0], 3, 3), dtype=jnp.float64)
    
    # Compute point directions for binning (batched, no per-point host sync)
    norms = jnp.linalg.norm(deskewed_points, axis=1, keepdims=True)
    point_directions = deskewed_points / (norms + config.eps_mass)
    
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
    # Step 7: WahbaSVD
    # =========================================================================
    # Compute weights: w_b = N[b] * kappa_map[b] * kappa_scan[b]
    wahba_weights = scan_bins.N * kappa_map * scan_bins.kappa_scan
    
    wahba_result, wahba_cert, wahba_effect = wahba_svd(
        mu_map=mu_map,
        mu_scan=mu_scan,
        weights=wahba_weights,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(wahba_cert)
    R_hat = wahba_result.R_hat
    
    # =========================================================================
    # Step 8: TranslationWLS
    # =========================================================================
    trans_result, trans_cert, trans_effect = translation_wls(
        c_map=c_map,
        Sigma_c_map=Sigma_c_map,
        p_bar_scan=scan_bins.p_bar,
        Sigma_p_scan=scan_bins.Sigma_p,
        R_hat=R_hat,
        weights=wahba_weights,
        Sigma_meas=config.Sigma_meas,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(trans_cert)
    t_hat = trans_result.t_hat

    # Measurement-noise IW sufficient stats from TranslationWLS residuals (LiDAR meas block)
    # residual_b = c_map[b] - R_hat @ p_bar_scan[b] - t_hat
    p_rot = (R_hat @ scan_bins.p_bar.T).T  # (B,3)
    residuals_t = c_map - p_rot - t_hat[None, :]
    iw_meas_dPsi, iw_meas_dnu = lidar_meas_iw_suffstats_from_translation_residuals_jax(
        residuals=residuals_t,
        weights=wahba_weights,
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
    # Build map bins structure for lidar evidence
    lidar_map_bins = LidarMapBinStats(
        S_dir=map_stats.S_dir,
        N_dir=map_stats.N_dir,
        N_pos=map_stats.N_pos,
        sum_p=map_stats.sum_p,
        sum_ppT=map_stats.sum_ppT,
        mu_dir=mu_map,
        kappa_map=kappa_map,
        centroid=c_map,
        Sigma_c=Sigma_c_map,
    )
    
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

    # IMU accel direction evidence (vMF Laplace on rotation perturbation)
    pose_pred = belief_pred.mean_world_pose(eps_lift=config.eps_lift)
    imu_result, imu_cert, imu_effect = imu_vmf_gravity_evidence(
        rotvec_world_body=pose_pred[3:6],
        imu_accel=imu_accel,
        weights=w_imu,
        accel_bias=accel_bias,
        gravity_W=gravity_W,
        eps_psd=config.eps_psd,
        eps_mass=config.eps_mass,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(imu_cert)

    # IMU gyro rotation evidence (Gaussian on SO(3) using preintegrated delta rotation)
    Sigma_g = jnp.asarray(config.Sigma_g, dtype=jnp.float64)
    gyro_result, gyro_cert, gyro_effect = imu_gyro_rotation_evidence(
        rotvec_start_WB=rotvec0,
        rotvec_end_pred_WB=pose_pred[3:6],
        delta_rotvec_meas=delta_pose[3:6],
        Sigma_g=Sigma_g,
        dt_scan=float(dt_scan),
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
    )
    all_certs.append(gyro_cert)

    evidence_result, evidence_cert, evidence_effect = lidar_quadratic_evidence(
        belief_pred=belief_pred,
        scan_bins=scan_bins,
        map_bins=lidar_map_bins,
        R_hat=R_hat,
        t_hat=t_hat,
        t_cov=trans_result.t_cov,
        c_dt=config.c_dt,
        c_ex=config.c_ex,
        eps_psd=config.eps_psd,
        eps_lift=config.eps_lift,
    )
    all_certs.append(evidence_cert)

    # Combine evidence terms additively before fusion scaling
    L_evidence = evidence_result.L_lidar + odom_result.L_odom + imu_result.L_imu + gyro_result.L_gyro
    h_evidence = evidence_result.h_lidar + odom_result.h_odom + imu_result.h_imu + gyro_result.h_gyro

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
    combined_evidence_cert = aggregate_certificates([evidence_cert, odom_cert, imu_cert, gyro_cert])
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
    map_update_result, map_update_cert, map_update_effect = pos_cov_inflation_pushforward(
        belief_post=belief_recomposed,
        scan_N=scan_bins.N,
        scan_s_dir=scan_bins.s_dir,
        scan_p_bar=scan_bins.p_bar,
        scan_Sigma_p=scan_bins.Sigma_p,
        R_hat=R_hat,
        t_hat=t_hat,
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

    # Compute evidence diagnostics
    L_total_np = np.array(L_evidence)
    h_total_np = np.array(h_evidence)

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

    diagnostics = ScanDiagnostics(
        scan_number=0,  # Will be set by backend_node
        timestamp=scan_end_time,
        dt_sec=dt_sec,
        n_points_raw=int(raw_points.shape[0]),
        n_points_budget=int(points.shape[0]),
        p_W=trans_final,
        R_WL=R_final,
        N_bins=np.array(scan_bins.N),
        S_bins=np.array(scan_bins.s_dir),
        kappa_bins=np.array(scan_bins.kappa_scan),
        L_total=L_total_np,
        h_total=h_total_np,
        L_lidar=np.array(evidence_result.L_lidar),
        L_odom=np.array(odom_result.L_odom),
        L_imu=np.array(imu_result.L_imu),
        L_gyro=np.array(gyro_result.L_gyro),
        logdet_L_total=logdet_L,
        trace_L_total=trace_L,
        L_dt=L_dt_val,
        trace_L_ex=trace_L_ex,
        s_dt=float(s_dt),
        s_ex=float(s_ex),
        psd_delta_fro=psd_delta,
        psd_min_eig_before=psd_min_eig_before,
        psd_min_eig_after=psd_min_eig_after,
        wahba_cost=float(wahba_result.cost),
        translation_residual_norm=float(trans_result.residual_norm),
        fusion_alpha=float(alpha),
        total_trigger_magnitude=total_trigger_mag,
        conditioning_number=cond_number,
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
                "lidar_evidence": "fl_slam_poc.backend.operators.lidar_evidence (closed-form pose info; no sigma-point regression)",
                "translation_wls": "fl_slam_poc.backend.operators.translation (vectorized over bins; Cholesky inverse)",
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
            "backends": dict(self.backends),
        }
