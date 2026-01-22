"""
von Mises-Fisher (vMF) geometry for directional data.

Closed-form operations for directional distributions on the unit sphere S^{d-1}:
- Bregman barycenter (dual-space averaging + series inversion)
- Fisher-Rao distance (exact via Bessel affinity)
- Natural parameter conversion
- Third-order Frobenius correction (stub - vMF cubic is small)

vMF density: p(x|μ,κ) = C_d(κ) exp(κ μ·x)
where C_d(κ) = κ^(d/2-1) / ((2π)^(d/2) I_{d/2-1}(κ))

The natural parameter is θ = κμ (d-dimensional vector).

Reference:
- Miyamoto et al. (2024) Section 4.5 (vMF Fisher-Rao)
- exponential_families_complete-2.pdf page 12 (vMF parametrization)
- Mardia & Jupp (2000) for series expansions
"""

import math
import numpy as np
from scipy.special import iv, ive  # Modified Bessel I_n
from scipy.optimize import brentq
from typing import Tuple, List

from fl_slam_poc.common.op_report import OpReport


def A_d(kappa: float, d: int = 3) -> float:
    """
    Compute A_d(κ) = I_{d/2}(κ) / I_{d/2-1}(κ).
    
    This is the ratio of modified Bessel functions that appears in the
    dual coordinate (expectation parameter) for vMF.
    
    Uses scaled Bessel functions (ive) for numerical stability at large κ.
    
    Args:
        kappa: Concentration parameter (κ ≥ 0)
        d: Dimension of the sphere S^{d-1}
    
    Returns:
        A_d(κ) ∈ [0, 1)
    """
    if kappa < 1e-10:
        return 0.0
    
    nu = d / 2.0 - 1.0
    
    # Use scaled Bessel (ive) for numerical stability
    # ive(v, z) = iv(v, z) * exp(-|Re(z)|)
    # A_d = I_{nu+1}(κ) / I_nu(κ) = ive(nu+1, κ) / ive(nu, κ)
    i_nu_scaled = ive(nu, kappa)
    i_nu1_scaled = ive(nu + 1, kappa)
    
    if i_nu_scaled < 1e-300:
        # Asymptotic: A_d(κ) → 1 as κ → ∞
        return 1.0 - (d - 1) / (2 * kappa)
    
    return float(i_nu1_scaled / i_nu_scaled)


def A_d_inverse_series(r: float, d: int = 3, order: int = 10) -> float:
    """
    Inverse of A_d(κ) = r.
    
    For auditability and "exactness" in the information-geometry sense, this is
    implemented as a monotone, bracketed root solve of the defining special
    function equation A_d(κ) - r = 0. (No heuristic truncation.)
    
    Args:
        r: Mean resultant length (‖η*‖ in dual space)
        d: Dimension
        order: Kept for API compatibility (unused)
    
    Returns:
        κ: Concentration parameter
    """
    _ = order
    if r < 1e-12:
        return 0.0
    
    # Clamp to valid range
    r = float(np.clip(r, 0.0, 1.0 - 1e-12))

    def f(kappa: float) -> float:
        return A_d(kappa, d) - r

    lo = 0.0  # A_d(0)=0

    # Heuristic upper bound; for r→1, κ grows ~ O((d-1)/(1-r)).
    hi = float((d - 1.0) / max(1e-12, (1.0 - r)))
    hi = max(hi, 1.0)
    hi = min(hi, 1e6)  # safety cap

    # Ensure we bracket the root.
    while hi < 1e6 and f(hi) < 0.0:
        hi *= 2.0
    hi = min(hi, 1e6)

    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=256))


def vmf_make_evidence(mu: np.ndarray, kappa: float, d: int = 3) -> np.ndarray:
    """
    Convert vMF parameters (μ, κ) to natural parameter θ.
    
    For vMF: θ = κμ (d-dimensional vector)
    
    Args:
        mu: Unit direction vector (will be normalized)
        kappa: Concentration parameter (κ ≥ 0)
        d: Dimension (default 3 for S²)
    
    Returns:
        theta: Natural parameter (d,)
    """
    mu = np.asarray(mu, dtype=float)
    norm = np.linalg.norm(mu)
    if norm > 1e-10:
        mu = mu / norm
    else:
        # Default direction if zero
        mu = np.zeros(d)
        mu[0] = 1.0
    
    return kappa * mu


def vmf_mean_param(theta: np.ndarray, d: int = 3) -> Tuple[np.ndarray, float]:
    """
    Convert natural parameter θ to vMF parameters (μ, κ).
    
    κ = ‖θ‖, μ = θ/‖θ‖
    
    Args:
        theta: Natural parameter (d,)
        d: Dimension
    
    Returns:
        (mu, kappa): Unit direction and concentration
    """
    theta = np.asarray(theta, dtype=float)
    kappa = float(np.linalg.norm(theta))
    
    if kappa < 1e-12:
        # Isotropic (κ = 0), return default direction
        mu = np.zeros(d)
        mu[0] = 1.0
        return mu, 0.0
    
    mu = theta / kappa
    return mu, kappa


def vmf_barycenter(
    thetas: List[np.ndarray],
    weights: List[float],
    d: int = 3
) -> Tuple[np.ndarray, OpReport]:
    """
    Closed-form Bregman barycenter for vMF mixture.
    
    Algorithm (exact):
    1. Dual-space averaging: η* = Σ w_i A_d(κ_i) μ_i / W
    2. Primal recovery: r = ‖η*‖, μ* = η*/r
    3. Concentration: κ* = A_d^{-1}(r) via series
    
    This is ASSOCIATIVE and COMMUTATIVE (dual averaging).
    
    Args:
        thetas: List of natural parameters (each d-dimensional)
        weights: List of weights (positive)
        d: Dimension
    
    Returns:
        (theta_star, report): Fused natural parameter and OpReport
    """
    if len(thetas) == 0:
        raise ValueError("Need at least one distribution to compute barycenter")
    
    if len(thetas) != len(weights):
        raise ValueError("thetas and weights must have same length")
    
    weights = np.array(weights, dtype=float)
    W = float(np.sum(weights))
    
    if W < 1e-12:
        raise ValueError("Total weight must be positive")
    
    # Dual-space averaging: η* = Σ w_i A_d(κ_i) μ_i / W
    eta_star = np.zeros(d, dtype=float)
    
    for theta, w in zip(thetas, weights):
        theta = np.asarray(theta, dtype=float)
        mu, kappa = vmf_mean_param(theta, d)
        a_d = A_d(kappa, d)
        eta_star += w * a_d * mu
    
    eta_star /= W
    
    # Primal recovery
    r = float(np.linalg.norm(eta_star))
    
    if r < 1e-12:
        # Isotropic limit (no preferred direction)
        mu_star = np.zeros(d)
        mu_star[0] = 1.0
        kappa_star = 0.0
    else:
        mu_star = eta_star / r
        kappa_star = A_d_inverse_series(r, d, order=10)
    
    theta_star = kappa_star * mu_star
    
    # OpReport
    report = OpReport(
        name="vMF_Barycenter",
        exact=True,  # Series is exact in limit
        approximation_triggers=[],
        family_in="vMF",
        family_out="vMF",
        closed_form=True,
        solver_used=None,
        frobenius_applied=False,  # No approximation
        metrics={
            "n_components": len(thetas),
            "concentration_out": float(kappa_star),
            "dual_norm": float(r),
            "total_weight": float(W),
        },
    )
    
    return theta_star, report


def vmf_fisher_rao_distance(
    theta1: np.ndarray,
    theta2: np.ndarray,
    d: int = 3
) -> float:
    """
    Fisher-Rao distance between two vMF distributions.
    
    From Miyamoto (2024) and standard information geometry:
        d_FR = 2 arccos(√BC)
    where BC (Bhattacharyya coefficient) is exact via Bessel affinity.
    
    For vMF: BC = exp(log I_ν(‖κ₁μ₁ + κ₂μ₂‖) - log I_ν(κ₁) - log I_ν(κ₂))
    
    This is a TRUE METRIC (symmetric, triangle inequality).
    
    Args:
        theta1: Natural parameter of first vMF
        theta2: Natural parameter of second vMF
        d: Dimension
    
    Returns:
        Fisher-Rao distance
    """
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    
    mu1, kappa1 = vmf_mean_param(theta1, d)
    mu2, kappa2 = vmf_mean_param(theta2, d)
    
    # Handle edge case: one or both κ = 0 (isotropic)
    if kappa1 < 1e-10 and kappa2 < 1e-10:
        return 0.0  # Both isotropic
    if kappa1 < 1e-10 or kappa2 < 1e-10:
        # One isotropic, one concentrated - max distance
        return float(math.pi / 2)
    
    nu = d / 2.0 - 1.0
    
    # Sum of natural parameters
    theta_sum = kappa1 * mu1 + kappa2 * mu2
    kappa_sum = float(np.linalg.norm(theta_sum))
    
    # Log Bhattacharyya coefficient (exact via scaled Bessel)
    # BC = I_ν(κ_sum) / (I_ν(κ₁) * I_ν(κ₂))
    # Use log for numerical stability
    
    if kappa_sum < 1e-10:
        log_i_sum = float(np.log(ive(nu, 1e-10) + 1e-300))
    else:
        log_i_sum = float(np.log(ive(nu, kappa_sum) + 1e-300)) + kappa_sum
    
    log_i_1 = float(np.log(ive(nu, kappa1) + 1e-300)) + kappa1
    log_i_2 = float(np.log(ive(nu, kappa2) + 1e-300)) + kappa2
    
    log_bc = log_i_sum - log_i_1 - log_i_2
    
    # Clamp BC to valid range [0, 1]
    bc = float(np.exp(np.clip(log_bc, -700, 0)))
    bc = max(0.0, min(1.0, bc))
    
    # Fisher-Rao distance
    if bc > 1.0 - 1e-10:
        return 0.0
    if bc < 1e-10:
        return float(math.pi)
    
    d_fr = 2.0 * math.acos(math.sqrt(bc))
    return float(d_fr)


def vmf_third_order_correction(
    theta: np.ndarray,
    delta: np.ndarray,
    d: int = 3
) -> Tuple[np.ndarray, dict]:
    """
    Frobenius third-order correction for vMF updates.
    
    For vMF, the cubic tensor C = ∇³ψ(θ) is non-zero but relatively small
    for moderate concentrations. This provides a stub implementation.
    
    Full implementation requires symbolic Bessel derivatives, which is
    deferred to Phase 2.
    
    Args:
        theta: Current natural parameter
        delta: Proposed update in natural parameter space
        d: Dimension
    
    Returns:
        (delta_corrected, stats): Corrected delta and diagnostic stats
    """
    delta = np.asarray(delta, dtype=float)
    
    # For Phase 1: identity (no correction)
    # Full Frobenius correction would require computing C_ijk via
    # derivatives of log I_ν, which involves Bessel function recursions
    
    delta_corrected = delta.copy()
    
    stats = {
        "delta_norm": 0.0,  # No change in Phase 1
        "input_stats": {"delta": float(np.linalg.norm(delta))},
        "output_stats": {"delta_corr": float(np.linalg.norm(delta_corrected))},
    }
    
    return delta_corrected, stats


def vmf_hellinger_distance(
    theta1: np.ndarray,
    theta2: np.ndarray,
    d: int = 3
) -> float:
    """
    Hellinger distance between two vMF distributions.
    
    H² = 1 - BC where BC is the Bhattacharyya coefficient.
    
    Args:
        theta1: Natural parameter of first vMF
        theta2: Natural parameter of second vMF
        d: Dimension
    
    Returns:
        Hellinger distance
    """
    # Compute BC via the same method as Fisher-Rao
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    
    mu1, kappa1 = vmf_mean_param(theta1, d)
    mu2, kappa2 = vmf_mean_param(theta2, d)
    
    if kappa1 < 1e-10 and kappa2 < 1e-10:
        return 0.0
    if kappa1 < 1e-10 or kappa2 < 1e-10:
        return 1.0
    
    nu = d / 2.0 - 1.0
    
    theta_sum = kappa1 * mu1 + kappa2 * mu2
    kappa_sum = float(np.linalg.norm(theta_sum))
    
    if kappa_sum < 1e-10:
        log_i_sum = float(np.log(ive(nu, 1e-10) + 1e-300))
    else:
        log_i_sum = float(np.log(ive(nu, kappa_sum) + 1e-300)) + kappa_sum
    
    log_i_1 = float(np.log(ive(nu, kappa1) + 1e-300)) + kappa1
    log_i_2 = float(np.log(ive(nu, kappa2) + 1e-300)) + kappa2
    
    log_bc = log_i_sum - log_i_1 - log_i_2
    bc = float(np.exp(np.clip(log_bc, -700, 0)))
    bc = max(0.0, min(1.0, bc))
    
    h_sq = 1.0 - bc
    return float(math.sqrt(max(0.0, h_sq)))
