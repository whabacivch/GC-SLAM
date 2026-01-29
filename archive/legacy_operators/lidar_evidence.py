"""
LidarQuadraticEvidence operator for Golden Child SLAM v2.

Produces quadratic evidence on full 22D tangent at fixed cost.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.9
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    D_Z,
    SLICE_POSE,
    pose_se3_to_z_delta,
)
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    ExcitationCert,
    MismatchCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_solve_lifted,
    spd_cholesky_inverse_lifted,
)
from fl_slam_poc.backend.operators.binning import ScanBinStats
from fl_slam_poc.common.geometry import se3_jax


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class MapBinStats:
    """Map bin sufficient statistics."""
    S_dir: jnp.ndarray  # (B_BINS, 3) directional resultants Σ w u
    S_dir_scatter: jnp.ndarray  # (B_BINS, 3, 3) directional scatter Σ w u u^T
    N_dir: jnp.ndarray  # (B_BINS,) directional mass
    N_pos: jnp.ndarray  # (B_BINS,) position mass
    sum_p: jnp.ndarray  # (B_BINS, 3) position sums
    sum_ppT: jnp.ndarray  # (B_BINS, 3, 3) scatter matrices
    mu_dir: jnp.ndarray  # (B_BINS, 3) mean directions
    kappa_map: jnp.ndarray  # (B_BINS,) concentration parameters
    centroid: jnp.ndarray  # (B_BINS, 3) centroids
    Sigma_c: jnp.ndarray  # (B_BINS, 3, 3) centroid covariances


@dataclass 
class LidarEvidenceResult:
    """Result of LidarQuadraticEvidence operator."""
    L_lidar: jnp.ndarray  # (D_Z, D_Z) information matrix
    h_lidar: jnp.ndarray  # (D_Z,) information vector
    delta_z_star: jnp.ndarray  # (D_Z,) MAP increment


# =============================================================================
# Main Operator
# =============================================================================


def lidar_quadratic_evidence(
    belief_pred: BeliefGaussianInfo,
    scan_bins: ScanBinStats,
    map_bins: MapBinStats,
    R_hat: jnp.ndarray,
    t_hat: jnp.ndarray,
    t_cov: jnp.ndarray,
    c_dt: float = constants.GC_C_DT,
    c_ex: float = constants.GC_C_EX,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> Tuple[LidarEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Produce quadratic evidence on full 22D tangent at fixed cost.
    
    Branch-free coupling rule:
        s_dt = dt_effect / (dt_effect + c_dt)
        s_ex = extrinsic_effect / (extrinsic_effect + c_ex)
    
    Blocks involving index 15 multiplied by s_dt.
    Blocks involving indices 16..21 multiplied by s_ex.
    
    Args:
        belief_pred: Predicted belief
        scan_bins: Scan bin statistics
        map_bins: Map bin statistics
        R_hat: Estimated rotation (3, 3)
        t_hat: Estimated translation (3,)
        t_cov: Translation covariance from TranslationWLS (3,3)
        c_dt: Time offset coupling constant
        c_ex: Extrinsic coupling constant
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        
    Returns:
        Tuple of (LidarEvidenceResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.9
    """
    R_hat = jnp.asarray(R_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(t_hat, dtype=jnp.float64)
    t_cov = jnp.asarray(t_cov, dtype=jnp.float64)

    # Predicted world pose (measurement will be compared as a right-perturbation in this chart).
    pose_pred = belief_pred.mean_world_pose(eps_lift=eps_lift)  # (6,) [trans, rotvec]
    R_pred = se3_jax.so3_exp(pose_pred[3:6])

    # Step 1: Compute excitation scales (continuous, no branching)
    # Use belief covariance as the continuous excitation proxy (no legacy cache dependency).
    _mu_pred, Sigma_pred, _lift = belief_pred.to_moments(eps_lift)
    dt_effect = float(jnp.sqrt(Sigma_pred[15, 15]))
    extrinsic_effect = float(jnp.sqrt(jnp.trace(Sigma_pred[16:22, 16:22])))
    
    s_dt = dt_effect / (dt_effect + c_dt)
    s_ex = extrinsic_effect / (extrinsic_effect + c_ex)

    # Step 2: Build delta_z* as a right-perturbation error:
    #   X_meas is the absolute scan pose in world from (R_hat, t_hat)
    #   xi_err = Log( X_pred^{-1} ∘ X_meas )   (se3_jax ordering: [rho, phi])
    #   delta_pose_z = [rho, phi]             (GC ordering: [trans, rot] - same as se3_jax!)
    rotvec_meas = se3_jax.so3_log(R_hat)
    pose_meas = jnp.concatenate([t_hat, rotvec_meas])
    T_err = se3_jax.se3_relative(pose_meas, pose_pred)  # pose_pred^{-1} ∘ pose_meas
    xi_err = se3_jax.se3_log(T_err)  # [rho, phi] = [trans, rot]
    delta_pose_z = pose_se3_to_z_delta(xi_err)  # identity now - [trans, rot]

    delta_z_star = jnp.zeros(D_Z, dtype=jnp.float64).at[SLICE_POSE].set(delta_pose_z)

    # Step 3: Build L_lidar from closed-form Fisher-style pose information (no sigma-point regression)
    #
    # Rotation: vMF directional curvature proxy aggregated over bins:
    #   H_rot ≈ Σ_b w_b * (I - μ_b μ_b^T)    (PSD, rank-2 per bin)
    #
    # Translation: use TranslationWLS covariance as a Gaussian measurement term:
    #   H_trans = (t_cov + eps I)^{-1}
    mu_map = jnp.asarray(map_bins.mu_dir, dtype=jnp.float64)  # (B,3) in world frame
    I3 = jnp.eye(3, dtype=jnp.float64)
    P = I3[None, :, :] - jnp.einsum("bi,bj->bij", mu_map, mu_map)  # (B,3,3)
    w_b = scan_bins.N * map_bins.kappa_map * scan_bins.kappa_scan  # (B,)
    H_rot_W = jnp.einsum("b,bij->ij", w_b, P)
    # Convert curvature to right-perturbation body coordinates: δθ_world ≈ R_pred δθ_body
    H_rot = R_pred.T @ H_rot_W @ R_pred
    H_rot = domain_projection_psd(H_rot, eps_psd).M_psd

    # TranslationWLS returns covariance in world translation coordinates; convert to right-perturbation coords.
    t_cov_body = R_pred.T @ t_cov @ R_pred
    t_cov_psd = domain_projection_psd(t_cov_body, eps_psd).M_psd
    H_trans, _ = spd_cholesky_inverse_lifted(t_cov_psd, eps_lift)

    total_mass = float(jnp.sum(scan_bins.N))
    info_scale = total_mass / (total_mass + constants.GC_EPS_MASS)

    L_lidar_raw = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    # GC ordering: [trans(0:3), rot(3:6)]
    L_lidar_raw = L_lidar_raw.at[0:3, 0:3].set(info_scale * H_trans)
    L_lidar_raw = L_lidar_raw.at[3:6, 3:6].set(info_scale * H_rot)
    
    # Step 4: Apply excitation scaling to relevant blocks (always)
    # Time offset: index 15
    L_lidar_raw = L_lidar_raw.at[15, :].set(s_dt * L_lidar_raw[15, :])
    L_lidar_raw = L_lidar_raw.at[:, 15].set(s_dt * L_lidar_raw[:, 15])
    
    # Extrinsic: indices 16..21
    L_lidar_raw = L_lidar_raw.at[16:22, :].set(s_ex * L_lidar_raw[16:22, :])
    L_lidar_raw = L_lidar_raw.at[:, 16:22].set(s_ex * L_lidar_raw[:, 16:22])
    
    # Step 5: Apply DomainProjectionPSD (always)
    L_psd_result = domain_projection_psd(L_lidar_raw, eps_psd)
    L_lidar = L_psd_result.M_psd
    
    # Step 6: Compute h_lidar = L_lidar @ delta_z*
    h_lidar = L_lidar @ delta_z_star
    
    # Build result
    result = LidarEvidenceResult(
        L_lidar=L_lidar,
        h_lidar=h_lidar,
        delta_z_star=delta_z_star,
    )
    
    # Compute mismatch proxy
    nll_proxy = 0.5 * float(delta_z_star @ L_lidar @ delta_z_star)
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
        triggers=["LidarQuadraticEvidence"],
        conditioning=ConditioningCert(
            eig_min=L_psd_result.conditioning.eig_min,
            eig_max=L_psd_result.conditioning.eig_max,
            cond=L_psd_result.conditioning.cond,
            near_null_count=L_psd_result.conditioning.near_null_count,
        ),
        mismatch=MismatchCert(
            nll_per_ess=nll_proxy / (total_mass + constants.GC_EPS_MASS),
            directional_score=0.0,
        ),
        excitation=ExcitationCert(
            dt_effect=dt_effect,
            extrinsic_effect=extrinsic_effect,
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=L_psd_result.projection_delta,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=s_dt,
            extrinsic_scale=s_ex,
            trust_alpha=1.0,
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_quadratic_nll_decrease",
        predicted=nll_proxy,
        realized=None,
    )
    
    return result, cert, expected_effect
