"""
Binning operators for Golden Child SLAM v2.

Soft assignment and moment matching for directional bins.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Sections 5.4, 5.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    SupportCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd_core,
    inv_mass_core,
)
from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_batch


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class BinSoftAssignResult:
    """Result of BinSoftAssign operator."""
    responsibilities: jnp.ndarray  # (N, B_BINS) soft assignment probabilities


@dataclass
class ScanBinStats:
    """Scan bin sufficient statistics."""
    N: jnp.ndarray  # (B_BINS,) total mass per bin
    s_dir: jnp.ndarray  # (B_BINS, 3) direction resultant vectors
    p_bar: jnp.ndarray  # (B_BINS, 3) centroids
    Sigma_p: jnp.ndarray  # (B_BINS, 3, 3) centroid covariances
    kappa_scan: jnp.ndarray  # (B_BINS,) concentration parameters


# =============================================================================
# Bin Atlas Creation
# =============================================================================


def create_bin_atlas(n_bins: int = constants.GC_B_BINS) -> jnp.ndarray:
    """
    Create Fibonacci lattice bin directions.
    
    Args:
        n_bins: Number of bins
        
    Returns:
        Bin directions (n_bins, 3)
    """
    indices = jnp.arange(n_bins, dtype=jnp.float64) + 0.5
    phi = jnp.arccos(1 - 2 * indices / n_bins)
    theta = jnp.pi * (1 + jnp.sqrt(5)) * indices
    
    x = jnp.sin(phi) * jnp.cos(theta)
    y = jnp.sin(phi) * jnp.sin(theta)
    z = jnp.cos(phi)
    
    dirs = jnp.stack([x, y, z], axis=1)
    
    # Normalize
    norms = jnp.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (norms + constants.GC_EPS_MASS)
    
    return dirs


# =============================================================================
# Bin Soft Assign Operator
# =============================================================================


def bin_soft_assign(
    point_directions: jnp.ndarray,
    bin_directions: jnp.ndarray,
    tau: float = constants.GC_TAU_SOFT_ASSIGN,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[BinSoftAssignResult, CertBundle, ExpectedEffect]:
    """
    Soft assignment of points to bins using softmax.
    
    Never argmax - uses continuous soft responsibilities.
    
    Args:
        point_directions: Normalized point directions (N, 3)
        bin_directions: Normalized bin directions (B, 3)
        tau: Temperature parameter for softmax
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (BinSoftAssignResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.4
    """
    point_directions = jnp.asarray(point_directions, dtype=jnp.float64)
    bin_directions = jnp.asarray(bin_directions, dtype=jnp.float64)
    
    n_points = point_directions.shape[0]

    # Compute similarities (dot products): (N, B)
    similarities = point_directions @ bin_directions.T

    # Batched soft assignment (no per-point Python loop)
    responsibilities = jax.nn.softmax(similarities / tau, axis=1)

    # Entropy-based quality metrics (continuous, branch-free)
    max_resp = jnp.max(responsibilities)
    entropy_per_point = -jnp.sum(
        responsibilities * jnp.log(responsibilities + constants.GC_EPS_MASS), axis=1
    )
    total_entropy = jnp.sum(entropy_per_point)
    avg_entropy = float(total_entropy / (n_points + constants.GC_EPS_MASS))
    
    # Build result
    result = BinSoftAssignResult(responsibilities=responsibilities)
    
    # Build certificate
    cert = CertBundle.create_exact(
        chart_id=chart_id,
        anchor_id=anchor_id,
        support=SupportCert(
            ess_total=float(jnp.exp(avg_entropy)),  # Effective number of bins per point
            support_frac=float(max_resp),
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_assignment_entropy",
        predicted=avg_entropy,
        realized=None,
    )
    
    return result, cert, expected_effect


# =============================================================================
# Scan Bin Moment Match Operator
# =============================================================================


def scan_bin_moment_match(
    points: jnp.ndarray,
    point_covariances: jnp.ndarray,
    weights: jnp.ndarray,
    responsibilities: jnp.ndarray,
    point_lambda: jnp.ndarray | None = None,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_mass: float = constants.GC_EPS_MASS,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[ScanBinStats, CertBundle, ExpectedEffect]:
    """
    Compute scan bin statistics using soft responsibilities.
    
    Uses InvMass for all mass-based computations (no N==0 branches).
    
    Args:
        points: Point positions (N, 3)
        point_covariances: Per-point covariances (N, 3, 3)
        weights: Per-point weights (N,)
        responsibilities: Soft assignments (N, B_BINS)
        eps_psd: PSD projection epsilon
        eps_mass: Mass regularization
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (ScanBinStats, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.5
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    point_covariances = jnp.asarray(point_covariances, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    responsibilities = jnp.asarray(responsibilities, dtype=jnp.float64)
    
    n_points = points.shape[0]
    n_bins = responsibilities.shape[1]

    # Noise-weighting (Contract 3): per-point reliability enters as a multiplicative weight.
    if point_lambda is None:
        point_lambda = jnp.ones((n_points,), dtype=jnp.float64)
    else:
        point_lambda = jnp.asarray(point_lambda, dtype=jnp.float64).reshape(-1)
        if point_lambda.shape[0] != n_points:
            raise ValueError(f"point_lambda must be (N,), got {point_lambda.shape} for N={n_points}")
    
    # Weighted responsibilities per bin: (N, B)
    w_eff = weights * point_lambda
    w_r = w_eff[:, None] * responsibilities

    # Normalize point directions (batched, branch-free)
    norms = jnp.linalg.norm(points, axis=1, keepdims=True)
    d = points / (norms + eps_mass)  # (N,3)

    # Sufficient statistics (batched; avoids O(N*B) Python loops)
    N = jnp.sum(w_r, axis=0)  # (B,)
    s_dir = w_r.T @ d  # (B,3)
    sum_p = w_r.T @ points  # (B,3)

    ppT = points[:, :, None] * points[:, None, :]  # (N,3,3)
    sum_ppT = jnp.einsum("nb,nij->bij", w_r, ppT)  # (B,3,3)
    sum_cov = jnp.einsum("nb,nij->bij", w_r, point_covariances)  # (B,3,3)
    
    # Compute derived quantities using batched operations (no per-bin Python loop)
    # InvMass - batched (Contract: InvMass is always 1/(m+eps), spec Section 3.4)
    inv_N, eps_ratio_N = inv_mass_core(N, eps_mass)  # (B,), (B,)
    
    # Centroids: p_bar = sum_p * inv_N
    p_bar = sum_p * inv_N[:, None]  # (B, 3)
    
    # Scatter covariance: scatter = sum_ppT * inv_N - outer(p_bar, p_bar)
    scatter = sum_ppT * inv_N[:, None, None] - jnp.einsum("bi,bj->bij", p_bar, p_bar)  # (B, 3, 3)
    
    # Measurement covariance (weighted average)
    meas_cov = sum_cov * inv_N[:, None, None]  # (B, 3, 3)
    
    # Total covariance
    Sigma_raw = scatter + meas_cov  # (B, 3, 3)
    
    # Project to PSD (batched; declared projection op).
    def proj_one(S):
        S_psd, cert_vec = domain_projection_psd_core(S, eps_psd)
        return S_psd, cert_vec

    Sigma_p, Sigma_cert = jax.vmap(proj_one)(Sigma_raw)  # (B,3,3), (B,6)
    psd_projection_delta_total = jnp.sum(Sigma_cert[:, 0])
    
    # Kappa from resultant length (batched)
    S_norms = jnp.linalg.norm(s_dir, axis=1)  # (B,)
    Rbar = S_norms * inv_N  # R-bar in (0, 1) for each bin
    kappa_scan = kappa_from_resultant_batch(Rbar, eps_r=constants.GC_EPS_R)  # (B,)
    
    # Build result
    result = ScanBinStats(
        N=N,
        s_dir=s_dir,
        p_bar=p_bar,
        Sigma_p=Sigma_p,
        kappa_scan=kappa_scan,
    )
    
    # Compute ESS
    total_mass = float(jnp.sum(N))
    ess = total_mass ** 2 / (jnp.sum(N ** 2) + eps_mass)
    
    # Build certificate
    # Continuous support fraction proxy in [0,1]: bins with mass contribute smoothly.
    support_frac = float(jnp.mean(N / (N + eps_mass)))
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ScanBinMomentMatch"],
        support=SupportCert(
            ess_total=float(ess),
            support_frac=support_frac,
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=float(psd_projection_delta_total),
            mass_epsilon_ratio=float(jnp.max(eps_ratio_N)),
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_ess",
        predicted=float(ess),
        realized=None,
    )
    
    return result, cert, expected_effect
