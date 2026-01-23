"""
Gaussian operations in information (natural parameter) form.

Following information geometry principles:
- Represent beliefs in canonical coordinates: θ = (Λ, η) where Λ = Σ⁻¹, η = Σ⁻¹μ
- Fusion is EXACT addition in natural parameters (no Jacobians)
- Order-invariant: (L₁ + L₂) + L₃ = L₁ + (L₂ + L₃)
- Commutative: L₁ + L₂ = L₂ + L₁

Why information form replaces EKF-style updates:
- EKF: Jacobian-based linearization + Kalman gain computation
- Here: θ_posterior = θ_prior + θ_likelihood (additive, exact)
- O(n) vs O(n² iter) for iterative methods

Reference: Barndorff-Nielsen (1978), Amari (2016), Combe (2022-2025)
"""

import math
import numpy as np
from typing import Tuple

from fl_slam_poc.common import constants


def _as_vector(x: np.ndarray) -> np.ndarray:
    """
    Normalize any (n,), (n,1), (1,n) into a flat (n,) float vector.

    This prevents silent NumPy broadcasting bugs when mixing column vectors
    and 1D arrays in information-form operations.
    """
    x = np.asarray(x, dtype=float)
    return x.reshape(-1)


def _spd_solve(A: np.ndarray, b: np.ndarray, name: str) -> np.ndarray:
    """Strict SPD solve via Cholesky (raises on failure)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: expected square matrix, got {A.shape}")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"{name}: rhs shape {b.shape} incompatible with {A.shape}")
    L_chol = np.linalg.cholesky(A)
    y = np.linalg.solve(L_chol, b)
    return np.linalg.solve(L_chol.T, y)


def _spd_inv(A: np.ndarray, name: str) -> np.ndarray:
    """Strict SPD inverse via Cholesky (raises on failure)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: expected square matrix, got {A.shape}")
    eye = np.eye(A.shape[0], dtype=A.dtype)
    return _spd_solve(A, eye, name)


def make_evidence(mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (mean, covariance) to information form (Lambda, eta).
    
    Natural parameters for Gaussian:
        Λ = Σ⁻¹ (precision matrix)
        η = Σ⁻¹μ (information vector)
    
    The log-density is: log p(x) ∝ -½x'Λx + η'x - ψ(Λ,η)
    
    Uses Cholesky-based solve for numerical stability and adds regularization
    to prevent singular matrix errors.
    """
    cov = np.asarray(cov, dtype=float)
    mean = _as_vector(mean)
    
    # Regularize to prevent singular matrices
    reg = np.eye(cov.shape[0], dtype=cov.dtype) * constants.COV_REGULARIZATION_MIN
    cov_reg = cov + reg
    
    # Use Cholesky solve for stability (strict)
    L_chol = np.linalg.cholesky(cov_reg)
    L = np.linalg.solve(L_chol, np.eye(cov.shape[0], dtype=cov.dtype))
    L = L @ L.T  # Reconstruct precision matrix
    h = np.linalg.solve(L_chol, mean.reshape(-1, 1)).reshape(-1)
    
    return L, h


def fuse_info(
    L: np.ndarray,
    h: np.ndarray,
    L_obs: np.ndarray,
    h_obs: np.ndarray,
    weight: float = 1.0,
    rho: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Order-invariant additive fusion in information space.
    
    For exponential families, Bayesian fusion in natural coordinates is:
        θ_posterior = ρ·θ_prior + w·θ_likelihood
    
    This is EXACT (no approximation) when both prior and likelihood
    are in the same exponential family.
    
    Properties:
    - Commutative: fuse(A, B) = fuse(B, A)
    - Associative: fuse(fuse(A, B), C) = fuse(A, fuse(B, C))
    - No Jacobians required
    """
    h = _as_vector(h)
    h_obs = _as_vector(h_obs)
    return (rho * L + weight * L_obs, rho * h + weight * h_obs)


def mean_cov(L: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert information form (Lambda, eta) to (mean, covariance).
    
    Recovery from natural parameters:
        Σ = Λ⁻¹
        μ = Ση = Λ⁻¹η
    
    Uses Cholesky-based solve for numerical stability and adds regularization
    to prevent singular matrix errors.
    """
    L = np.asarray(L, dtype=float)
    h = _as_vector(h)
    
    # Regularize to prevent singular matrices
    # Add small diagonal term to ensure positive definiteness
    reg = np.eye(L.shape[0], dtype=L.dtype) * constants.COV_REGULARIZATION_MIN
    L_reg = L + reg
    
    # Use Cholesky solve for stability (strict)
    L_chol = np.linalg.cholesky(L_reg)
    cov = np.linalg.solve(L_chol, np.eye(L.shape[0], dtype=L.dtype))
    cov = cov @ cov.T  # Reconstruct from Cholesky factor
    mean = np.linalg.solve(L_chol, h.reshape(-1, 1)).reshape(-1)
    
    return mean, cov


def log_partition(L: np.ndarray, h: np.ndarray) -> float:
    """
    Log-partition function for Gaussian in information form.
    
    ψ(Λ, η) = ½η'Λ⁻¹η + ½(d log(2π) - log|Λ|)
    
    This is the convex potential that generates all cumulants.
    """
    d = L.shape[0]
    cov = np.linalg.inv(L)
    _, logdet_L = np.linalg.slogdet(L)
    h = _as_vector(h)
    quad = 0.5 * float(h @ (cov @ h))
    return quad + 0.5 * (d * math.log(2 * math.pi) - logdet_L)


def kl_divergence(L1: np.ndarray, h1: np.ndarray,
                  L2: np.ndarray, h2: np.ndarray) -> float:
    """
    KL divergence D_KL(N₁ || N₂) in information form.
    
    D_KL = ψ(θ₂) - ψ(θ₁) - ⟨∇ψ(θ₁), θ₂ - θ₁⟩
    
    This is the Bregman divergence induced by the log-partition.
    Closed-form, O(n³) for matrix operations.
    """
    mu1, cov1 = mean_cov(L1, h1)
    mu2, cov2 = mean_cov(L2, h2)
    
    d = cov1.shape[0]
    cov2_inv = L2  # L2 = Σ₂⁻¹
    
    term1 = np.trace(cov2_inv @ cov1)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    
    diff = mu2 - mu1
    kl = 0.5 * (term1 - d - logdet1 + logdet2 + float(diff @ cov2_inv @ diff))
    return float(kl)


def hellinger_distance(L1: np.ndarray, h1: np.ndarray,
                       L2: np.ndarray, h2: np.ndarray) -> float:
    """
    Hellinger distance between two Gaussians.
    
    H²(p₁, p₂) = 1 - BC where BC = ∫√(p₁p₂)dx
    
    For Gaussians, BC has closed form via the log-partition.
    """
    mu1, cov1 = mean_cov(L1, h1)
    mu2, cov2 = mean_cov(L2, h2)
    
    cov_avg = 0.5 * (cov1 + cov2)
    _, logdet_avg = np.linalg.slogdet(cov_avg)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    
    log_bc = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdet_avg
    
    diff = mu1 - mu2
    cov_avg_inv = _spd_inv(cov_avg, "hellinger_distance.cov_avg")
    log_bc -= 0.125 * float(diff @ cov_avg_inv @ diff)
    
    bc = np.exp(log_bc)
    return float(math.sqrt(max(0.0, 1.0 - bc)))


def bhattacharyya_coefficient(L1: np.ndarray, h1: np.ndarray,
                               L2: np.ndarray, h2: np.ndarray) -> float:
    """
    Bhattacharyya coefficient BC = ∫√(p₁p₂)dx.
    
    BC ∈ [0, 1], with BC = 1 for identical distributions.
    Can be used as a soft similarity measure for evidence weighting.
    """
    mu1, cov1 = mean_cov(L1, h1)
    mu2, cov2 = mean_cov(L2, h2)
    
    cov_avg = 0.5 * (cov1 + cov2)
    _, logdet_avg = np.linalg.slogdet(cov_avg)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    
    db = 0.5 * logdet_avg - 0.25 * (logdet1 + logdet2)
    
    diff = mu1 - mu2
    cov_avg_inv = _spd_inv(cov_avg, "bhattacharyya_coefficient.cov_avg")
    db += 0.125 * float(diff @ cov_avg_inv @ diff)
    
    return float(np.exp(-db))


def fisher_information(L: np.ndarray) -> np.ndarray:
    """
    Fisher information matrix for Gaussian.
    
    For the mean parameter: I(μ) = Σ⁻¹ = Λ
    
    The Fisher metric is the Hessian of the log-partition:
        g = ∇²ψ = Cov(T) = Σ
    """
    return L.copy()


def natural_gradient(loss_grad: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Natural gradient: g⁻¹∇L = Σ∇L.
    
    The natural gradient accounts for the information geometry,
    giving steepest descent in the Fisher-Rao metric.
    
    O(n³) one-shot solve vs O(n² iter) for iterative methods.
    """
    cov = np.linalg.inv(L)
    return cov @ loss_grad


def marginalize(L: np.ndarray, h: np.ndarray, 
               keep_dims: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marginalize out dimensions not in keep_dims.
    
    In information form, marginalization requires converting to moment
    form, selecting dimensions, then converting back.
    """
    mu, cov = mean_cov(L, h)
    keep_dims = np.asarray(keep_dims, dtype=int)
    
    mu_marg = mu[keep_dims]
    cov_marg = cov[np.ix_(keep_dims, keep_dims)]
    
    return make_evidence(mu_marg, cov_marg)


def condition(L: np.ndarray, h: np.ndarray,
             obs_dims: np.ndarray, obs_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Condition on observed dimensions.
    
    Returns the posterior (L, h) for the unobserved dimensions
    given observations at obs_dims = obs_vals.
    """
    mu, cov = mean_cov(L, h)
    obs_dims = np.asarray(obs_dims, dtype=int)
    obs_vals = np.asarray(obs_vals, dtype=float)
    
    n = L.shape[0]
    all_dims = np.arange(n)
    unobs_dims = np.array([i for i in all_dims if i not in obs_dims])
    
    if len(unobs_dims) == 0:
        return np.array([[1.0]]), np.array([[0.0]])
    
    # Partition covariance
    Sigma_aa = cov[np.ix_(unobs_dims, unobs_dims)]
    Sigma_ab = cov[np.ix_(unobs_dims, obs_dims)]
    Sigma_bb = cov[np.ix_(obs_dims, obs_dims)]
    
    mu_a = mu[unobs_dims]
    mu_b = mu[obs_dims]
    
    # Conditional: μ_a|b = μ_a + Σ_ab Σ_bb⁻¹ (b - μ_b)
    Sigma_bb_inv = np.linalg.inv(Sigma_bb)
    mu_cond = mu_a + Sigma_ab @ Sigma_bb_inv @ (obs_vals - mu_b)
    cov_cond = Sigma_aa - Sigma_ab @ Sigma_bb_inv @ Sigma_ab.T
    
    return make_evidence(mu_cond, cov_cond)


# NOTE: product_of_experts() removed - unused function


def mixture_moment_match(
    components: list[Tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse a Gaussian mixture via moment matching (e-projection).
    
    This is the CORRECT way to summarize a mixture posterior as a single
    Gaussian. It minimizes KL divergence from the mixture to the Gaussian
    family (Legendre e-projection).
    
    Algorithm:
        μ = Σᵢ wᵢ μᵢ
        Σ = Σᵢ wᵢ (Σᵢ + (μᵢ - μ)(μᵢ - μ)ᵀ)
    
    Then convert to information form:
        Λ = Σ⁻¹, h = Λμ
    
    Args:
        components: List of (mu, cov) tuples for each mixture component
        weights: Array of mixture weights (must sum to 1)
    
    Returns:
        (L, h): Information form of the moment-matched Gaussian
    
    Note:
        This is NOT the same as weighted sum of natural parameters!
        That would give incorrect uncertainty (often overconfident).
    """
    if len(components) == 0:
        raise ValueError("Need at least one component")
    
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)  # Ensure normalized
    
    n = len(components[0][0])
    
    # Step 1: Compute mixture mean
    mu_mixture = np.zeros(n, dtype=float)
    for (mu_i, _), w_i in zip(components, weights):
        mu_mixture += w_i * np.asarray(mu_i, dtype=float)
    
    # Step 2: Compute mixture covariance (includes spread of means)
    cov_mixture = np.zeros((n, n), dtype=float)
    for (mu_i, cov_i), w_i in zip(components, weights):
        mu_i = np.asarray(mu_i, dtype=float)
        cov_i = np.asarray(cov_i, dtype=float)
        diff = mu_i - mu_mixture
        cov_mixture += w_i * (cov_i + np.outer(diff, diff))
    
    # Ensure symmetric
    cov_mixture = 0.5 * (cov_mixture + cov_mixture.T)
    
    # Convert to information form
    return make_evidence(mu_mixture, cov_mixture)


def embed_info_form(
    L_small: np.ndarray,
    h_small: np.ndarray,
    indices: np.ndarray,
    full_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embed lower-dimensional information form into full-dimensional space.
    
    Used to fuse evidence that only affects a subset of dimensions.
    For example, embedding 6D odometry into 15D state.
    
    Args:
        L_small: Information matrix (m×m)
        h_small: Information vector (m,)
        indices: Which dimensions in full space (m,), e.g., [0,1,2,3,4,5]
        full_dim: Full state dimension (n), e.g., 15
    
    Returns:
        (L_full, h_full): Embedded information form (n×n, n)
    
    Note:
        The embedded information matrix has zeros in rows/columns not
        in indices, meaning those dimensions have zero information gain
        (infinite variance) from this factor.
    """
    L_small = np.asarray(L_small, dtype=float)
    h_small = _as_vector(h_small)
    indices = np.asarray(indices, dtype=int)
    
    L_full = np.zeros((full_dim, full_dim), dtype=float)
    h_full = np.zeros(full_dim, dtype=float)
    
    L_full[np.ix_(indices, indices)] = L_small
    h_full[indices] = h_small
    
    return L_full, h_full


def hellinger_squared_from_moments(
    mu1: np.ndarray, cov1: np.ndarray,
    mu2: np.ndarray, cov2: np.ndarray,
) -> float:
    """
    Squared Hellinger distance H²(N₁, N₂) from moment parameters.
    
    This is the same as hellinger_distance() but takes moment form
    directly, avoiding redundant conversions.
    
    H²(p, q) = 1 - BC where BC = ∫√(pq)dx
    
    For Gaussians:
        BC = |Σ₁|^{1/4} |Σ₂|^{1/4} |Σ_avg|^{-1/2} exp(-⅛ Δμᵀ Σ_avg⁻¹ Δμ)
    
    Used for Hellinger-tilted likelihood in robust IMU fusion.
    """
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    mu2 = np.asarray(mu2, dtype=float).reshape(-1)
    cov1 = np.asarray(cov1, dtype=float)
    cov2 = np.asarray(cov2, dtype=float)
    
    cov_avg = 0.5 * (cov1 + cov2)
    
    _, logdet_avg = np.linalg.slogdet(cov_avg)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    
    # log(BC) = 0.25*(logdet1 + logdet2) - 0.5*logdet_avg - 0.125*mahal²
    log_bc = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdet_avg
    
    diff = mu1 - mu2
    cov_avg_inv = _spd_inv(cov_avg, "hellinger_squared_from_moments.cov_avg")
    mahal_sq = float(diff @ cov_avg_inv @ diff)
    
    log_bc -= 0.125 * mahal_sq
    
    bc = np.exp(log_bc)
    h_sq = max(0.0, 1.0 - bc)
    
    return float(h_sq)


# =============================================================================
# α-Divergence and Trust-Scaled Fusion (Information-Geometric Trust Region)
# =============================================================================

# Prior: α = 0.5 gives symmetric divergence (related to Hellinger)
# α → 1 gives KL(p||q), α → 0 gives KL(q||p)
# ESS interpretation: α = 0.5 means "penalize both over-updating and under-updating equally"
ALPHA_DIVERGENCE_DEFAULT = 0.5

# Prior: maximum α-divergence per update step
# ESS interpretation: ~1 pseudocount's worth of information change is the expected maximum
# This prevents catastrophic jumps while allowing meaningful updates
MAX_ALPHA_DIVERGENCE_PRIOR = 1.0


def alpha_divergence(
    L1: np.ndarray, h1: np.ndarray,
    L2: np.ndarray, h2: np.ndarray,
    alpha: float = ALPHA_DIVERGENCE_DEFAULT,
) -> float:
    """
    α-divergence D_α(p₁ || p₂) between two Gaussians.
    
    The α-divergence family generalizes KL:
        D_α(p||q) = (1/(α(1-α))) * (1 - ∫ p^α q^{1-α} dx)
    
    Special cases:
        α → 1: KL(p||q)  "information gained" (forward KL)
        α → 0: KL(q||p)  "information lost" (reverse KL)
        α = 0.5: Symmetric, related to Hellinger (√2 * H²)
    
    For Gaussians, this has closed form via the moment-generating function.
    
    Properties:
        - α = 0.5 satisfies symmetry: D_0.5(p||q) = D_0.5(q||p)
        - Does NOT satisfy triangle inequality (not a metric)
        - Bounded below by 0, unbounded above
    
    Args:
        L1, h1: Information form of distribution p₁
        L2, h2: Information form of distribution p₂
        alpha: Divergence parameter in (0, 1), default 0.5 for symmetry
    
    Returns:
        D_α(p₁ || p₂) ≥ 0
    
    Reference: Amari (2016), Chapter 3
    """
    # Clamp alpha to valid range (avoid singularities at 0 and 1)
    alpha = float(np.clip(alpha, 0.01, 0.99))
    
    mu1, cov1 = mean_cov(L1, h1)
    mu2, cov2 = mean_cov(L2, h2)
    
    d = cov1.shape[0]
    
    # Weighted covariance: Σ_α = α*Σ₁ + (1-α)*Σ₂
    cov_alpha = alpha * cov1 + (1.0 - alpha) * cov2
    
    # Log determinants
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    _, logdet_alpha = np.linalg.slogdet(cov_alpha)
    
    # Determinant ratio term
    # log|Σ_α| - α*log|Σ₁| - (1-α)*log|Σ₂|
    log_det_term = logdet_alpha - alpha * logdet1 - (1.0 - alpha) * logdet2
    
    # Mahalanobis term
    diff = mu1 - mu2
    cov_alpha_inv = _spd_inv(cov_alpha, "alpha_divergence.cov_alpha")
    mahal_sq = float(diff @ cov_alpha_inv @ diff)
    
    # α-divergence formula for Gaussians:
    # D_α = (1/(2α(1-α))) * (α(1-α)*mahal² + log_det_term)
    # Simplified: D_α = 0.5 * mahal² + log_det_term / (2α(1-α))
    d_alpha = 0.5 * alpha * (1.0 - alpha) * mahal_sq + 0.5 * log_det_term
    d_alpha = d_alpha / (alpha * (1.0 - alpha))
    
    return float(max(0.0, d_alpha))


def trust_scaled_fusion(
    prior_L: np.ndarray,
    prior_h: np.ndarray,
    factor_L: np.ndarray,
    factor_h: np.ndarray,
    max_divergence: float = MAX_ALPHA_DIVERGENCE_PRIOR,
    alpha: float = ALPHA_DIVERGENCE_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fuse a factor with trust-region scaling using α-divergence.
    
    This is the power posterior / tempered likelihood approach:
        posterior ∝ prior × likelihood^β
    
    In information form:
        L_post = L_prior + β × L_factor
        h_post = h_prior + β × h_factor
    
    where β ∈ [0, 1] is computed to satisfy the trust region constraint.
    
    Key property: This stays ENTIRELY in the Gaussian family!
    - No mixture reduction needed
    - Preserves conjugacy and associativity
    - β = 1 gives full update, β → 0 gives "do nothing"
    
    The trust region is based on α-divergence:
        β = argmax β' s.t. D_α(posterior(β') || prior) ≤ max_divergence
    
    For efficiency, we use a closed-form approximation:
        β ≈ exp(-D_α(full_posterior || prior) / max_divergence)
    
    This gives smooth scaling: high divergence → low β, low divergence → β ≈ 1.
    
    Args:
        prior_L, prior_h: Prior in information form
        factor_L, factor_h: Factor (evidence) to fuse
        max_divergence: Trust region radius (prior: 1.0 = ~1 pseudocount change)
        alpha: α-divergence parameter (prior: 0.5 for symmetry)
    
    Returns:
        (L_post, h_post, diagnostics): Posterior and diagnostic dict
        
    Diagnostics include:
        - beta: Tempering coefficient used
        - divergence_full: D_α if full update were applied
        - divergence_actual: D_α of actual (scaled) update
        - trust_quality: exp(-divergence_actual / max_divergence) ∈ (0, 1]
    """
    prior_L = np.asarray(prior_L, dtype=float)
    prior_h = _as_vector(prior_h)
    factor_L = np.asarray(factor_L, dtype=float)
    factor_h = _as_vector(factor_h)
    
    # Compute what the full posterior would be
    full_post_L = prior_L + factor_L
    full_post_h = prior_h + factor_h
    
    # Compute α-divergence for full update
    div_full = alpha_divergence(full_post_L, full_post_h, prior_L, prior_h, alpha=alpha)
    
    # Compute trust scaling factor β
    # β = exp(-div_full / max_divergence) gives smooth decay
    # High divergence → β → 0, low divergence → β → 1
    if max_divergence > 0:
        beta = float(np.exp(-div_full / max_divergence))
    else:
        beta = 1.0
    
    # Clamp β to [0, 1]
    beta = float(np.clip(beta, 0.0, 1.0))
    
    # Apply scaled fusion (power posterior)
    # L_post = L_prior + β * L_factor
    # h_post = h_prior + β * h_factor
    post_L = prior_L + beta * factor_L
    post_h = prior_h + beta * factor_h
    
    # Compute actual divergence (for diagnostics)
    div_actual = alpha_divergence(post_L, post_h, prior_L, prior_h, alpha=alpha)
    
    # Trust quality: how much of the update was we able to apply
    trust_quality = float(np.exp(-div_actual / max(max_divergence, 1e-10)))
    
    diagnostics = {
        "beta": beta,
        "divergence_full": div_full,
        "divergence_actual": div_actual,
        "trust_quality": trust_quality,
        "alpha": alpha,
        "max_divergence": max_divergence,
    }
    
    return post_L, post_h, diagnostics


def compute_odom_precision_from_covariance(
    odom_cov: np.ndarray,
    min_precision: float = constants.COV_REGULARIZATION_MIN,
    max_precision: float = 1e6,
) -> np.ndarray:
    """
    Convert odometry covariance to precision matrix with bounds.
    
    The odom covariance encodes sensor confidence:
        - Low variance (0.001 for XY) → high precision → strong influence
        - High variance (1e6 for Z) → low precision → weak influence
    
    This function inverts the covariance to get precision, with safeguards:
        - Clamp diagonal to [1/max_precision, 1/min_precision] before inversion
        - This prevents numerical issues from huge variances (1e6) or tiny variances
    
    Args:
        odom_cov: 6x6 odometry covariance matrix
        min_precision: Minimum precision (prevents infinite variance from dominating)
        max_precision: Maximum precision (prevents tiny variance from over-weighting)
    
    Returns:
        6x6 precision matrix (information form Λ)
    
    Note:
        For M3DGR odom, Z/roll/pitch have variance ~1e6, giving precision ~1e-6.
        This means Z/roll/pitch contribute almost nothing to updates - which is correct!
    """
    odom_cov = np.asarray(odom_cov, dtype=float)
    
    if odom_cov.shape != (6, 6):
        raise ValueError(f"Expected 6x6 covariance, got {odom_cov.shape}")
    
    # Clamp diagonal to valid range before inversion
    cov_clamped = odom_cov.copy()
    diag = np.diag(cov_clamped)
    
    # Variance bounds: [1/max_precision, 1/min_precision]
    min_var = 1.0 / max_precision
    max_var = 1.0 / min_precision
    diag_clamped = np.clip(diag, min_var, max_var)
    
    # Replace diagonal with clamped values
    np.fill_diagonal(cov_clamped, diag_clamped)
    
    # Ensure positive definiteness with regularization
    cov_clamped = 0.5 * (cov_clamped + cov_clamped.T)  # Symmetrize
    eigvals = np.linalg.eigvalsh(cov_clamped)
    if eigvals.min() < 1e-10:
        cov_clamped += np.eye(6) * (1e-10 - eigvals.min())
    
    # Invert to get precision (strict)
    L_chol = np.linalg.cholesky(cov_clamped)
    precision = np.linalg.solve(L_chol, np.eye(6))
    precision = precision @ precision.T
    
    return precision
