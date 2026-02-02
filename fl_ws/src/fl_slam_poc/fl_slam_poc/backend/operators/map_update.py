"""
PoseCovInflationPushforward operator for Geometric Compositional SLAM v2.

Push scan statistics into map frame with pose covariance inflation.

Fully vectorized over bins - no Python for-loops in hot path.

Reference: docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md Section 5.13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import BeliefGaussianInfo, SLICE_POSE
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    domain_projection_psd_core,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class MapUpdateResult:
    """Result of PoseCovInflationPushforward operator."""
    # Increments to add to map sufficient statistics
    delta_S_dir: jnp.ndarray  # (B_BINS, 3)
    delta_S_dir_scatter: jnp.ndarray  # (B_BINS, 3, 3) directional scatter Σ w u u^T
    delta_N_dir: jnp.ndarray  # (B_BINS,)
    delta_N_pos: jnp.ndarray  # (B_BINS,)
    delta_sum_p: jnp.ndarray  # (B_BINS, 3)
    delta_sum_ppT: jnp.ndarray  # (B_BINS, 3, 3)
    inflation_magnitude: float  # Total covariance inflation


# =============================================================================
# Vectorized Core (JIT-safe)
# =============================================================================


def _skew_matrix(v: jnp.ndarray) -> jnp.ndarray:
    """Build 3x3 skew-symmetric matrix from 3-vector."""
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=jnp.float64)


@jax.jit
def _map_update_core(
    scan_N: jnp.ndarray,
    scan_s_dir: jnp.ndarray,
    scan_S_dir_scatter: jnp.ndarray,
    scan_p_bar: jnp.ndarray,
    scan_Sigma_p: jnp.ndarray,
    R_hat: jnp.ndarray,
    t_hat: jnp.ndarray,
    Sigma_rot: jnp.ndarray,
    Sigma_trans: jnp.ndarray,
    eps_psd: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vectorized core computation for map update (JIT-compiled).

    Args:
        scan_N: Bin masses (B,)
        scan_s_dir: Direction resultants (B, 3)
        scan_S_dir_scatter: Directional scatter tensors (B, 3, 3)
        scan_p_bar: Centroids (B, 3)
        scan_Sigma_p: Centroid covariances (B, 3, 3)
        R_hat: Estimated rotation (3, 3)
        t_hat: Estimated translation (3,)
        Sigma_rot: Rotation covariance (3, 3)
        Sigma_trans: Translation covariance (3, 3)
        eps_psd: PSD projection epsilon

    Returns:
        delta_S_dir: (B, 3)
        delta_S_dir_scatter: (B, 3, 3)
        delta_N_dir: (B,)
        delta_N_pos: (B,)
        delta_sum_p: (B, 3)
        delta_sum_ppT: (B, 3, 3)
        total_inflation: scalar
        total_psd_projection_delta: scalar
    """
    # Transform centroids to map frame: p_map = R @ p_bar + t
    # scan_p_bar: (B, 3), R_hat: (3, 3)
    p_rotated = jnp.einsum("ij,bj->bi", R_hat, scan_p_bar)  # (B, 3)

    # PLANAR FIX: Zero out t_hat[2] before map update.
    # This prevents the belief's z coordinate from being placed into the map,
    # breaking the z feedback loop that causes drift to -50 to -80m.
    # The map stays in the z=0 plane, while the belief's z is constrained
    # by the planar prior (z ≈ z_ref).
    t_hat_planar = t_hat.at[2].set(0.0)
    p_map = p_rotated + t_hat_planar[None, :]  # (B, 3)

    # Transform directions to map frame: s_dir_map = R @ s_dir
    s_dir_map = jnp.einsum("ij,bj->bi", R_hat, scan_s_dir)  # (B, 3)

    # Transform directional scatter tensors: S_scatter_map = R @ S_scatter @ R^T
    S_dir_scatter_map = jnp.einsum("ij,bjk,lk->bil", R_hat, scan_S_dir_scatter, R_hat)  # (B, 3, 3)

    # Transform measurement covariances: R @ Sigma_p @ R^T
    # scan_Sigma_p: (B, 3, 3)
    Sigma_p_rotated = jnp.einsum("ij,bjk,lk->bil", R_hat, scan_Sigma_p, R_hat)  # (B, 3, 3)

    # Build skew matrices for all rotated centroids (vectorized)
    # p_skew[b] is the skew-symmetric matrix of p_rotated[b]
    def skew_one(p):
        return jnp.array([
            [0.0, -p[2], p[1]],
            [p[2], 0.0, -p[0]],
            [-p[1], p[0], 0.0]
        ], dtype=jnp.float64)

    p_skew_all = jax.vmap(skew_one)(p_rotated)  # (B, 3, 3)

    # Rotation contribution to position uncertainty: p_skew @ Sigma_rot @ p_skew^T
    # (B, 3, 3) @ (3, 3) @ (B, 3, 3)^T
    Sigma_rot_contribution = jnp.einsum("bij,jk,bkl->bil", p_skew_all, Sigma_rot, p_skew_all)  # (B, 3, 3)

    # Total position covariance in map frame (before PSD projection)
    Sigma_map_raw = (
        Sigma_p_rotated  # Transformed measurement cov
        + Sigma_trans[None, :, :]  # Translation uncertainty
        + Sigma_rot_contribution  # Rotation uncertainty
    )  # (B, 3, 3)

    # Project each to PSD (vectorized)
    def psd_project_one(S):
        S_psd, cert_vec = domain_projection_psd_core(S, eps_psd)
        return S_psd, cert_vec[0]  # projection_delta is index 0

    Sigma_map, proj_deltas = jax.vmap(psd_project_one)(Sigma_map_raw)
    total_psd_projection_delta = jnp.sum(proj_deltas)

    # Compute inflation: trace(Sigma_map) - trace(Sigma_p_rotated)
    trace_map = jnp.trace(Sigma_map, axis1=1, axis2=2)  # (B,)
    trace_rotated = jnp.trace(Sigma_p_rotated, axis1=1, axis2=2)  # (B,)
    inflation_per_bin = trace_map - trace_rotated  # (B,)
    total_inflation = jnp.sum(inflation_per_bin * scan_N)

    # Build increments to sufficient statistics
    delta_N_dir = scan_N  # (B,)
    delta_N_pos = scan_N  # (B,)
    delta_S_dir = s_dir_map  # (B, 3)
    delta_S_dir_scatter = S_dir_scatter_map  # (B, 3, 3)
    delta_sum_p = scan_N[:, None] * p_map  # (B, 3)

    # For scatter: sum_ppT increment = N * (Sigma + p @ p^T)
    p_outer = jnp.einsum("bi,bj->bij", p_map, p_map)  # (B, 3, 3)
    delta_sum_ppT = scan_N[:, None, None] * (Sigma_map + p_outer)  # (B, 3, 3)

    return delta_S_dir, delta_S_dir_scatter, delta_N_dir, delta_N_pos, delta_sum_p, delta_sum_ppT, total_inflation, total_psd_projection_delta


# =============================================================================
# Main Operator (Wrapper)
# =============================================================================


def pos_cov_inflation_pushforward(
    belief_post: BeliefGaussianInfo,
    scan_N: jnp.ndarray,
    scan_s_dir: jnp.ndarray,
    scan_S_dir_scatter: jnp.ndarray,
    scan_p_bar: jnp.ndarray,
    scan_Sigma_p: jnp.ndarray,
    R_hat: jnp.ndarray,
    t_hat: jnp.ndarray,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> Tuple[MapUpdateResult, CertBundle, ExpectedEffect]:
    """
    Push scan statistics into map frame with pose covariance inflation.

    The pose uncertainty from the belief is propagated into the
    position covariance of each bin.

    Fully vectorized over bins (no Python for-loops).

    Args:
        belief_post: Posterior belief
        scan_N: Scan bin masses (B,)
        scan_s_dir: Scan direction resultants (B, 3)
        scan_S_dir_scatter: Scan directional scatter tensors (B, 3, 3)
        scan_p_bar: Scan centroids (B, 3)
        scan_Sigma_p: Scan centroid covariances (B, 3, 3)
        R_hat: Estimated rotation (3, 3)
        t_hat: Estimated translation (3,)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon

    Returns:
        Tuple of (MapUpdateResult, CertBundle, ExpectedEffect)

    Spec ref: Section 5.13
    """
    scan_N = jnp.asarray(scan_N, dtype=jnp.float64)
    scan_s_dir = jnp.asarray(scan_s_dir, dtype=jnp.float64)
    scan_S_dir_scatter = jnp.asarray(scan_S_dir_scatter, dtype=jnp.float64)
    scan_p_bar = jnp.asarray(scan_p_bar, dtype=jnp.float64)
    scan_Sigma_p = jnp.asarray(scan_Sigma_p, dtype=jnp.float64)
    R_hat = jnp.asarray(R_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(t_hat, dtype=jnp.float64)

    # Extract pose covariance from belief
    _, cov_full, _ = belief_post.to_moments(eps_lift)

    # Pose covariance is the 6x6 block
    # GC ordering: [trans(0:3), rot(3:6)]
    # [translation cov (3x3), trans-rot cross (3x3)]
    # [rot-trans cross (3x3), rotation cov (3x3)]
    Sigma_pose = cov_full[SLICE_POSE, SLICE_POSE]  # (6, 6)

    # For position inflation, we need the translation uncertainty
    # and how rotation uncertainty affects position
    Sigma_trans = Sigma_pose[0:3, 0:3]  # Translation covariance
    Sigma_rot = Sigma_pose[3:6, 3:6]  # Rotation covariance

    # Call vectorized core
    delta_S_dir, delta_S_dir_scatter, delta_N_dir, delta_N_pos, delta_sum_p, delta_sum_ppT, total_inflation, total_psd_projection_delta = (
        _map_update_core(
            scan_N,
            scan_s_dir,
            scan_S_dir_scatter,
            scan_p_bar,
            scan_Sigma_p,
            R_hat,
            t_hat,
            Sigma_rot,
            Sigma_trans,
            eps_psd,
        )
    )

    # Build result
    result = MapUpdateResult(
        delta_S_dir=delta_S_dir,
        delta_S_dir_scatter=delta_S_dir_scatter,
        delta_N_dir=delta_N_dir,
        delta_N_pos=delta_N_pos,
        delta_sum_p=delta_sum_p,
        delta_sum_ppT=delta_sum_ppT,
        inflation_magnitude=float(total_inflation),
    )

    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=belief_post.chart_id,
        anchor_id=belief_post.anchor_id,
        triggers=["PoseCovInflationPushforward"],
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=float(total_psd_projection_delta),
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    expected_effect = ExpectedEffect(
        objective_name="predicted_inflation_magnitude",
        predicted=float(total_inflation),
        realized=None,
    )

    return result, cert, expected_effect
