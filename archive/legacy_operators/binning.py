"""
Binning operators for Geometric Compositional SLAM v2.

Soft assignment and moment matching for directional bins.

Reference: docs/GC_SLAM.md Sections 5.4, 5.5
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
    s_dir: jnp.ndarray  # (B_BINS, 3) direction resultant vectors Σ w u
    S_dir_scatter: jnp.ndarray  # (B_BINS, 3, 3) directional scatter Σ w u u^T
    p_bar: jnp.ndarray  # (B_BINS, 3) centroids
    Sigma_p: jnp.ndarray  # (B_BINS, 3, 3) centroid covariances
    kappa_scan: jnp.ndarray  # (B_BINS,) concentration parameters


# =============================================================================
# Bin Soft Assign Operator
# =============================================================================


@jax.jit(static_argnames=("n_points", "n_bins"))
def _bin_soft_assign_core(
    point_directions: jnp.ndarray,
    bin_directions: jnp.ndarray,
    tau: jnp.ndarray,
    eps_mass: jnp.ndarray,
    n_points: int,
    n_bins: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT'd core: returns (responsibilities, avg_entropy, max_resp) for wrapper to build cert.
    """
    similarities = point_directions @ bin_directions.T
    responsibilities = jax.nn.softmax(similarities / tau, axis=1)
    max_resp = jnp.max(responsibilities)
    entropy_per_point = -jnp.sum(
        responsibilities * jnp.log(responsibilities + eps_mass), axis=1
    )
    total_entropy = jnp.sum(entropy_per_point)
    avg_entropy = total_entropy / (jnp.array(n_points, dtype=jnp.float64) + eps_mass)
    return responsibilities, avg_entropy, max_resp


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
    n_bins = bin_directions.shape[0]

    responsibilities, avg_entropy, max_resp = _bin_soft_assign_core(
        point_directions,
        bin_directions,
        jnp.array(tau, dtype=jnp.float64),
        jnp.array(constants.GC_EPS_MASS, dtype=jnp.float64),
        n_points=n_points,
        n_bins=n_bins,
    )

    result = BinSoftAssignResult(responsibilities=responsibilities)
    cert = CertBundle.create_exact(
        chart_id=chart_id,
        anchor_id=anchor_id,
        support=SupportCert(
            ess_total=float(jnp.exp(avg_entropy)),
            support_frac=float(max_resp),
        ),
    )
    expected_effect = ExpectedEffect(
        objective_name="predicted_assignment_entropy",
        predicted=float(avg_entropy),
        realized=None,
    )
    return result, cert, expected_effect


# =============================================================================
# Scan Bin Moment Match Operator
# =============================================================================


@jax.jit(static_argnames=("n_points", "n_bins"))
def _scan_bin_moment_match_core(
    points: jnp.ndarray,
    point_covariances: jnp.ndarray,
    weights: jnp.ndarray,
    responsibilities: jnp.ndarray,
    point_lambda: jnp.ndarray,
    direction_origin: jnp.ndarray,
    eps_psd: jnp.ndarray,
    eps_mass: jnp.ndarray,
    n_points: int,
    n_bins: int,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
]:
    """
    JIT'd core: array math only; returns (N, s_dir, S_dir_scatter, p_bar, Sigma_p, kappa_scan,
    ess, support_frac, psd_projection_delta_total, max_eps_ratio_N) for wrapper to build cert.
    """
    w_eff = weights * point_lambda
    w_r = w_eff[:, None] * responsibilities

    rays = points - direction_origin[None, :]
    norms = jnp.linalg.norm(rays, axis=1, keepdims=True)
    d = rays / (norms + eps_mass)

    N = jnp.sum(w_r, axis=0)
    s_dir = w_r.T @ d
    S_dir_scatter = jnp.einsum("nb,ni,nj->bij", w_r, d, d)
    sum_p = w_r.T @ points

    ppT = points[:, :, None] * points[:, None, :]
    sum_ppT = jnp.einsum("nb,nij->bij", w_r, ppT)
    sum_cov = jnp.einsum("nb,nij->bij", w_r, point_covariances)

    inv_N, eps_ratio_N = inv_mass_core(N, eps_mass)
    p_bar = sum_p * inv_N[:, None]

    scatter = sum_ppT * inv_N[:, None, None] - jnp.einsum("bi,bj->bij", p_bar, p_bar)
    meas_cov = sum_cov * inv_N[:, None, None]
    Sigma_raw = scatter + meas_cov

    def proj_one(S: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        S_psd, cert_vec = domain_projection_psd_core(S, eps_psd)
        return S_psd, cert_vec

    Sigma_p, Sigma_cert = jax.vmap(proj_one)(Sigma_raw)
    psd_projection_delta_total = jnp.sum(Sigma_cert[:, 0])

    S_norms = jnp.linalg.norm(s_dir, axis=1)
    Rbar = S_norms * inv_N
    kappa_scan = kappa_from_resultant_batch(Rbar, eps_r=constants.GC_EPS_R)

    total_mass = jnp.sum(N)
    ess = total_mass ** 2 / (jnp.sum(N ** 2) + eps_mass)
    support_frac = jnp.mean(N / (N + eps_mass))
    max_eps_ratio_N = jnp.max(eps_ratio_N)

    return (
        N,
        s_dir,
        S_dir_scatter,
        p_bar,
        Sigma_p,
        kappa_scan,
        ess,
        support_frac,
        psd_projection_delta_total,
        max_eps_ratio_N,
    )


def scan_bin_moment_match(
    points: jnp.ndarray,
    point_covariances: jnp.ndarray,
    weights: jnp.ndarray,
    responsibilities: jnp.ndarray,
    point_lambda: jnp.ndarray | None = None,
    direction_origin: jnp.ndarray | None = None,
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

    if direction_origin is None:
        direction_origin = jnp.zeros((3,), dtype=jnp.float64)
    else:
        direction_origin = jnp.asarray(direction_origin, dtype=jnp.float64).reshape(-1)
        if direction_origin.shape[0] != 3:
            raise ValueError(f"direction_origin must be (3,), got {direction_origin.shape}")
    
    n_points = points.shape[0]
    n_bins = responsibilities.shape[1]

    if point_lambda is None:
        point_lambda = jnp.ones((n_points,), dtype=jnp.float64)
    else:
        point_lambda = jnp.asarray(point_lambda, dtype=jnp.float64).reshape(-1)
        if point_lambda.shape[0] != n_points:
            raise ValueError(f"point_lambda must be (N,), got {point_lambda.shape} for N={n_points}")

    (
        N,
        s_dir,
        S_dir_scatter,
        p_bar,
        Sigma_p,
        kappa_scan,
        ess,
        support_frac,
        psd_projection_delta_total,
        max_eps_ratio_N,
    ) = _scan_bin_moment_match_core(
        points,
        point_covariances,
        weights,
        responsibilities,
        point_lambda,
        direction_origin,
        jnp.array(eps_psd, dtype=jnp.float64),
        jnp.array(eps_mass, dtype=jnp.float64),
        n_points=n_points,
        n_bins=n_bins,
    )

    result = ScanBinStats(
        N=N,
        s_dir=s_dir,
        S_dir_scatter=S_dir_scatter,
        p_bar=p_bar,
        Sigma_p=Sigma_p,
        kappa_scan=kappa_scan,
    )

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ScanBinMomentMatch"],
        support=SupportCert(
            ess_total=float(ess),
            support_frac=float(support_frac),
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=float(psd_projection_delta_total),
            mass_epsilon_ratio=float(max_eps_ratio_N),
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
