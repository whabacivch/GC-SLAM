"""
TranslationWLS operator for Golden Child SLAM v2.

Weighted least squares translation estimation.
Always uses lifted solve.

Fully vectorized over bins - no Python for-loops in hot path.
Uses Cholesky-based inverse (not jnp.linalg.inv).

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.8
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    domain_projection_psd_core,
    spd_cholesky_solve_lifted,
    spd_cholesky_solve_lifted_core,
    spd_cholesky_inverse_lifted,
    spd_cholesky_inverse_lifted_core,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TranslationWLSResult:
    """Result of TranslationWLS operator."""
    t_hat: jnp.ndarray  # (3,) estimated translation
    t_cov: jnp.ndarray  # (3, 3) covariance of estimate
    residual_norm: float  # Weighted residual norm


# =============================================================================
# Vectorized Core (JIT-safe)
# =============================================================================


@jax.jit
def _translation_wls_core(
    c_map: jnp.ndarray,
    Sigma_c_map: jnp.ndarray,
    p_bar_scan: jnp.ndarray,
    Sigma_p_scan: jnp.ndarray,
    R_hat: jnp.ndarray,
    weights: jnp.ndarray,
    Sigma_meas: jnp.ndarray,
    eps_psd: float,
    eps_lift: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vectorized core computation for TranslationWLS (JIT-compiled).

    Returns:
        t_hat: (3,) estimated translation
        t_cov: (3, 3) covariance of estimate
        residual_norm_sq: scalar, weighted squared residual norm
        total_projection_delta: scalar
        eig_A: (3,) eigenvalues of A for conditioning
    """
    # Rotate scan covariances: R @ Sigma_p @ R^T for all bins
    # Sigma_p_scan: (B, 3, 3), R_hat: (3, 3)
    # Result: (B, 3, 3)
    Sigma_scan_rotated = jnp.einsum("ij,bjk,lk->bil", R_hat, Sigma_p_scan, R_hat)

    # Total covariance per bin: Sigma_b = Sigma_c_map + Sigma_scan_rotated + Sigma_meas
    # (B, 3, 3)
    Sigma_b_raw = Sigma_c_map + Sigma_scan_rotated + Sigma_meas[None, :, :]

    # Project each to PSD (vectorized)
    def psd_project_one(S):
        S_psd, cert_vec = domain_projection_psd_core(S, eps_psd)
        return S_psd, cert_vec[0]  # projection_delta is index 0

    Sigma_b_psd, proj_deltas = jax.vmap(psd_project_one)(Sigma_b_raw)
    total_projection_delta = jnp.sum(proj_deltas)

    # Invert each Sigma_b using Cholesky-based lifted inverse (vectorized)
    def chol_inv_one(S):
        S_inv, lift = spd_cholesky_inverse_lifted_core(S, eps_lift)
        return S_inv, lift

    L_info_all, lifts = jax.vmap(chol_inv_one)(Sigma_b_psd)  # (B, 3, 3), (B,)
    total_lift = jnp.sum(lifts)

    # Compute rotated scan centroids: R @ p_bar_scan for all bins
    # p_bar_scan: (B, 3), R_hat: (3, 3)
    p_rotated = jnp.einsum("ij,bj->bi", R_hat, p_bar_scan)  # (B, 3)

    # Residuals: c_map - p_rotated (before translation estimate)
    residual_b = c_map - p_rotated  # (B, 3)

    # Weighted accumulation of normal equations: A = Σ w_b * L_info_b
    # weights: (B,), L_info_all: (B, 3, 3)
    A = jnp.einsum("b,bij->ij", weights, L_info_all)  # (3, 3)

    # b = Σ w_b * L_info_b @ residual_b
    # L_info_all @ residual_b: (B, 3)
    L_info_residual = jnp.einsum("bij,bj->bi", L_info_all, residual_b)  # (B, 3)
    b = jnp.einsum("b,bi->i", weights, L_info_residual)  # (3,)

    # Project A to PSD
    A_psd, A_cert_vec = domain_projection_psd_core(A, eps_psd)
    total_projection_delta = total_projection_delta + A_cert_vec[0]

    # Solve for t_hat using Cholesky solve
    t_hat, solve_lift = spd_cholesky_solve_lifted_core(A_psd, b, eps_lift)
    total_lift = total_lift + solve_lift

    # Covariance of estimate: inverse of A (Cholesky-based)
    t_cov_raw, cov_lift = spd_cholesky_inverse_lifted_core(A_psd, eps_lift)
    t_cov, _ = domain_projection_psd_core(t_cov_raw, eps_psd)

    # Compute weighted residual norm (vectorized)
    # Final residual: c_map - p_rotated - t_hat
    final_residual = c_map - p_rotated - t_hat[None, :]  # (B, 3)
    residual_sq_per_bin = jnp.sum(final_residual * final_residual, axis=1)  # (B,)
    residual_norm_sq = jnp.sum(weights * residual_sq_per_bin)

    # Eigenvalues of A for conditioning
    eig_A = jnp.linalg.eigvalsh(A_psd)

    return t_hat, t_cov, residual_norm_sq, total_projection_delta, total_lift, eig_A


# =============================================================================
# Main Operator (Wrapper)
# =============================================================================


def translation_wls(
    c_map: jnp.ndarray,
    Sigma_c_map: jnp.ndarray,
    p_bar_scan: jnp.ndarray,
    Sigma_p_scan: jnp.ndarray,
    R_hat: jnp.ndarray,
    weights: jnp.ndarray,
    Sigma_meas: jnp.ndarray,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[TranslationWLSResult, CertBundle, ExpectedEffect]:
    """
    Weighted least squares translation estimation.

    Model: c_map_b = R_hat @ p_bar_scan_b + t + noise

    Always uses lifted solve and DomainProjectionPSD.
    Fully vectorized over bins (no Python for-loops).

    Args:
        c_map: Map centroids (B, 3)
        Sigma_c_map: Map centroid covariances (B, 3, 3)
        p_bar_scan: Scan centroids (B, 3)
        Sigma_p_scan: Scan centroid covariances (B, 3, 3)
        R_hat: Estimated rotation (3, 3)
        weights: Per-bin weights (B,)
        Sigma_meas: Measurement noise covariance (3, 3)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        chart_id: Chart identifier
        anchor_id: Anchor identifier

    Returns:
        Tuple of (TranslationWLSResult, CertBundle, ExpectedEffect)

    Spec ref: Section 5.8
    """
    c_map = jnp.asarray(c_map, dtype=jnp.float64)
    Sigma_c_map = jnp.asarray(Sigma_c_map, dtype=jnp.float64)
    p_bar_scan = jnp.asarray(p_bar_scan, dtype=jnp.float64)
    Sigma_p_scan = jnp.asarray(Sigma_p_scan, dtype=jnp.float64)
    R_hat = jnp.asarray(R_hat, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    Sigma_meas = jnp.asarray(Sigma_meas, dtype=jnp.float64)

    # Call vectorized core
    t_hat, t_cov, residual_norm_sq, total_projection_delta, total_lift, eig_A = (
        _translation_wls_core(
            c_map,
            Sigma_c_map,
            p_bar_scan,
            Sigma_p_scan,
            R_hat,
            weights,
            Sigma_meas,
            eps_psd,
            eps_lift,
        )
    )

    residual_norm = float(jnp.sqrt(residual_norm_sq))

    # Build result
    result = TranslationWLSResult(
        t_hat=t_hat,
        t_cov=t_cov,
        residual_norm=residual_norm,
    )

    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["TranslationWLS"],
        conditioning=ConditioningCert(
            eig_min=float(jnp.min(eig_A)),
            eig_max=float(jnp.max(eig_A)),
            cond=float(jnp.max(eig_A) / (jnp.min(eig_A) + eps_psd)),
            near_null_count=int(jnp.sum(eig_A < 10 * eps_psd)),
        ),
        influence=InfluenceCert(
            lift_strength=float(total_lift),
            psd_projection_delta=float(total_projection_delta),
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    expected_effect = ExpectedEffect(
        objective_name="predicted_translation_residual",
        predicted=residual_norm,
        realized=None,
    )

    return result, cert, expected_effect
