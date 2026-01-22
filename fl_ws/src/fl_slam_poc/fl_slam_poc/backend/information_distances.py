"""
Closed-form information-geometric distances.

Following Miyamoto et al. (2024) and Combe (2022-2025):
- Hellinger for exponential families (via log-partition)
- Fisher-Rao for Gaussian, Student-t, SPD
- Product manifold aggregation
- Bregman/KL divergences

These are JACOBIAN-FREE, closed-form O(n³) operations that provide
proper Riemannian metrics (symmetry + triangle inequality) for
data association, gating, and robustness.

Why these replace ad-hoc methods:
- Log-likelihoods are NOT proper metrics (no triangle inequality)
- Euclidean distances on parameters ignore manifold geometry
- Mahalanobis depends on linearization choices
- These are geometry-native, model-consistent distances
"""

import math
import numpy as np
from typing import Callable, Sequence, Any


# =============================================================================
# A) Hellinger distance for any regular exponential family
# =============================================================================


def hellinger_sq_expfam(A: Callable[[np.ndarray], float],
                        eta: np.ndarray, theta: np.ndarray) -> float:
    """
    Squared Hellinger distance between two members of the same regular
    exponential family, using the log-partition A(·).
    
    For p(x|η) = h(x) exp(η·T(x) - A(η)):
        H²(p_η, p_θ) = 1 - exp(A((η+θ)/2) - (A(η)+A(θ))/2)
    
    This is EXACT and closed-form for any exponential family.
    """
    eta = np.asarray(eta, dtype=float)
    theta = np.asarray(theta, dtype=float)
    mid = 0.5 * (eta + theta)
    bc = np.exp(A(mid) - 0.5 * (A(eta) + A(theta)))
    return float(max(0.0, 1.0 - bc))


def hellinger_expfam(A: Callable[[np.ndarray], float],
                     eta: np.ndarray, theta: np.ndarray) -> float:
    """Hellinger distance (not squared) for exponential family."""
    return float(np.sqrt(hellinger_sq_expfam(A, eta, theta)))


def hellinger_gaussian(Sigma1: np.ndarray, Sigma2: np.ndarray,
                       mu1: np.ndarray = None, mu2: np.ndarray = None) -> float:
    """
    Closed-form Hellinger distance for multivariate Gaussians.
    
    For zero-mean or when means are equal:
        BC = (det(Σ₁)^(1/4) det(Σ₂)^(1/4)) / det((Σ₁+Σ₂)/2)^(1/2)
        H = √(1 - BC)
    
    With different means, includes exponential term.
    """
    Sigma1 = np.asarray(Sigma1, dtype=float)
    Sigma2 = np.asarray(Sigma2, dtype=float)
    Sigma_avg = 0.5 * (Sigma1 + Sigma2)
    
    _, logdet_avg = np.linalg.slogdet(Sigma_avg)
    _, logdet1 = np.linalg.slogdet(Sigma1)
    _, logdet2 = np.linalg.slogdet(Sigma2)
    
    # BC = exp(0.25*logdet1 + 0.25*logdet2 - 0.5*logdet_avg)
    log_bc = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdet_avg
    
    if mu1 is not None and mu2 is not None:
        mu1 = np.asarray(mu1, dtype=float).reshape(-1)
        mu2 = np.asarray(mu2, dtype=float).reshape(-1)
        diff = mu1 - mu2
        # Add mean term: exp(-0.125 * diff' @ Σ_avg^{-1} @ diff)
        Sigma_avg_inv = np.linalg.inv(Sigma_avg)
        log_bc -= 0.125 * float(diff @ Sigma_avg_inv @ diff)
    
    bc = np.exp(log_bc)
    return float(np.sqrt(max(0.0, 1.0 - bc)))


# =============================================================================
# B) Fisher-Rao distance for univariate Gaussian (location-scale)
# =============================================================================


def fisher_rao_gaussian_1d(mu1: float, sigma1: float,
                           mu2: float, sigma2: float) -> float:
    """
    Fisher-Rao distance for univariate Gaussian N(μ, σ²).
    
    d_FR = √2 · arctanh(√((μ₁-μ₂)² + (σ₁-σ₂)²) / ((μ₁-μ₂)² + (σ₁+σ₂)²))
    
    This is a TRUE METRIC (symmetric, triangle inequality) unlike
    Mahalanobis or log-likelihood ratios.
    
    Reference: Miyamoto et al. (2024), equation (4.20)
    """
    num = (mu1 - mu2)**2 + (sigma1 - sigma2)**2
    den = (mu1 - mu2)**2 + (sigma1 + sigma2)**2
    
    if den < 1e-15:
        return 0.0
    
    r = math.sqrt(max(0.0, min(1.0 - 1e-15, num / den)))
    return float(math.sqrt(2.0) * math.atanh(r))


# =============================================================================
# C) Fisher-Rao distance for Student-t (location-scale)
# =============================================================================


def fisher_rao_student_t(mu1: float, sigma1: float,
                         mu2: float, sigma2: float,
                         nu: float) -> float:
    """
    Fisher-Rao distance for location-scale Student's t with ν degrees of freedom.
    
    d_FR = 2√(2ν/(ν+3)) · arctanh(√((ν+1)(μ₂-μ₁)² + 2ν(σ₂-σ₁)²) / 
                                    ((ν+1)(μ₂-μ₁)² + 2ν(σ₂+σ₁)²))
    
    This is the EXACT closed-form for Student-t, which is the predictive
    distribution of the Normal-Inverse-Gamma (NIG) model.
    
    Reference: Miyamoto et al. (2024), equation (4.22)
    """
    if nu <= 0:
        raise ValueError("ν must be positive")
    
    a = (nu + 1.0) * (mu2 - mu1)**2 + 2.0 * nu * (sigma2 - sigma1)**2
    b = (nu + 1.0) * (mu2 - mu1)**2 + 2.0 * nu * (sigma2 + sigma1)**2
    
    if b < 1e-15:
        return 0.0
    
    r = math.sqrt(max(0.0, min(1.0 - 1e-15, a / b)))
    scale = 2.0 * math.sqrt(2.0 * nu / (nu + 3.0))
    return float(scale * math.atanh(r))


def fisher_rao_student_t_vec(mu1: np.ndarray, sigma1: np.ndarray,
                              mu2: np.ndarray, sigma2: np.ndarray,
                              nu: np.ndarray) -> float:
    """
    Fisher-Rao distance for vector of independent Student-t components.
    
    Uses product manifold structure: d = √(Σᵢ dᵢ²)
    
    This is the natural distance for NIG descriptor models where each
    dimension is independent.
    """
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    mu2 = np.asarray(mu2, dtype=float).reshape(-1)
    sigma1 = np.asarray(sigma1, dtype=float).reshape(-1)
    sigma2 = np.asarray(sigma2, dtype=float).reshape(-1)
    nu = np.asarray(nu, dtype=float).reshape(-1)
    
    d_sq = 0.0
    for i in range(len(mu1)):
        d_i = fisher_rao_student_t(mu1[i], sigma1[i], mu2[i], sigma2[i], nu[i])
        d_sq += d_i * d_i
    
    return float(math.sqrt(d_sq))


# =============================================================================
# D) Fisher-Rao distance on SPD covariances (affine-invariant metric)
# =============================================================================


def fisher_rao_spd(Sigma1: np.ndarray, Sigma2: np.ndarray, n: float = 1.0) -> float:
    """
    Fisher-Rao / affine-invariant distance on SPD matrices.
    
    d_FR(Σ₁, Σ₂) = √(n/2 · Σₖ (log λₖ)²)
    
    where λₖ are eigenvalues of Σ₁⁻¹Σ₂.
    
    This is the geometry-native distance on the SPD cone, NOT Euclidean
    Frobenius norm which ignores the manifold structure.
    
    Use cases:
    - Covariance consistency checking
    - Adaptive process noise tuning
    - Map fusion / multi-robot covariance reconciliation
    
    Reference: Miyamoto et al. (2024), Combe (2024) on Monge-Ampère domains
    """
    Sigma1 = np.asarray(Sigma1, dtype=float)
    Sigma2 = np.asarray(Sigma2, dtype=float)
    
    if Sigma1.shape != Sigma2.shape or Sigma1.shape[0] != Sigma1.shape[1]:
        raise ValueError("Sigma1 and Sigma2 must be same-shape square matrices")
    
    # Eigenvalues of Σ₁⁻¹Σ₂ (same as Σ₁^{-1/2} Σ₂ Σ₁^{-1/2})
    A = np.linalg.solve(Sigma1, Sigma2)
    eigvals = np.real(np.linalg.eigvals(A))
    
    # Safeguard for numerical issues
    eigvals = np.maximum(eigvals, 1e-12)
    
    logsq = np.square(np.log(eigvals))
    return float(math.sqrt((n / 2.0) * np.sum(logsq)))


# =============================================================================
# E) Product manifold distance (Pythagorean aggregation)
# =============================================================================


def product_distance(component_distances: Sequence[float]) -> float:
    """
    Product manifold distance via Pythagorean aggregation.
    
    For independent components: d((x₁,...,xₘ), (y₁,...,yₘ)) = √(Σᵢ dᵢ²)
    
    This is the natural way to combine distances from different
    sensor modalities or descriptor channels.
    """
    d_sq = sum(d * d for d in component_distances)
    return float(math.sqrt(d_sq))


def product_distance_weighted(component_distances: Sequence[float],
                               weights: Sequence[float]) -> float:
    """
    Weighted product manifold distance.
    
    d = √(Σᵢ wᵢ dᵢ²)
    
    Weights can encode confidence or information content per channel.
    """
    d_sq = sum(w * d * d for w, d in zip(weights, component_distances))
    return float(math.sqrt(d_sq))


# =============================================================================
# F) Gaussian KL divergence (Bregman divergence)
# =============================================================================


def gaussian_kl(Sigma1: np.ndarray, Sigma2: np.ndarray,
                mu1: np.ndarray = None, mu2: np.ndarray = None) -> float:
    """
    KL divergence between two Gaussians (closed-form Bregman).
    
    D_KL(N₁ || N₂) = ½[tr(Σ₂⁻¹Σ₁) - d - log(|Σ₁|/|Σ₂|) + (μ₂-μ₁)'Σ₂⁻¹(μ₂-μ₁)]
    
    This is EXACT and closed-form, O(n³) for the matrix operations.
    """
    Sigma1 = np.asarray(Sigma1, dtype=float)
    Sigma2 = np.asarray(Sigma2, dtype=float)
    d = Sigma1.shape[0]
    
    Sigma2_inv = np.linalg.inv(Sigma2)
    term1 = np.trace(Sigma2_inv @ Sigma1)
    
    _, logdet1 = np.linalg.slogdet(Sigma1)
    _, logdet2 = np.linalg.slogdet(Sigma2)
    
    kl = 0.5 * (term1 - d - logdet1 + logdet2)
    
    if mu1 is not None and mu2 is not None:
        mu1 = np.asarray(mu1, dtype=float).reshape(-1)
        mu2 = np.asarray(mu2, dtype=float).reshape(-1)
        diff = mu2 - mu1
        kl += 0.5 * float(diff @ Sigma2_inv @ diff)
    
    return float(kl)


def gaussian_kl_symmetric(Sigma1: np.ndarray, Sigma2: np.ndarray,
                          mu1: np.ndarray = None, mu2: np.ndarray = None) -> float:
    """
    Symmetric KL (Jeffreys divergence) between two Gaussians.
    
    D_J = ½(D_KL(N₁||N₂) + D_KL(N₂||N₁))
    """
    kl12 = gaussian_kl(Sigma1, Sigma2, mu1, mu2)
    kl21 = gaussian_kl(Sigma2, Sigma1, mu2, mu1)
    return float(0.5 * (kl12 + kl21))


# =============================================================================
# G) Wishart / SPD Bregman divergence
# =============================================================================


def wishart_bregman(Sigma1: np.ndarray, Sigma2: np.ndarray, p: int = None) -> float:
    """
    Wishart Bregman divergence for SPD matrices.
    
    D(Σ₁ || Σ₂) = ½[tr(Σ₂⁻¹Σ₁) - log|Σ₁Σ₂⁻¹| - p]
    
    where p is the dimension.
    
    This is the natural divergence on the Wishart/SPD cone.
    """
    Sigma1 = np.asarray(Sigma1, dtype=float)
    Sigma2 = np.asarray(Sigma2, dtype=float)
    
    if p is None:
        p = Sigma1.shape[0]
    
    term1 = np.trace(np.linalg.solve(Sigma2, Sigma1))
    _, logdet1 = np.linalg.slogdet(Sigma1)
    _, logdet2 = np.linalg.slogdet(Sigma2)
    
    return float(0.5 * (term1 - (logdet1 - logdet2) - p))


# =============================================================================
# H) Bhattacharyya coefficient and distance
# =============================================================================


def bhattacharyya_coefficient_gaussian(Sigma1: np.ndarray, Sigma2: np.ndarray,
                                        mu1: np.ndarray = None, 
                                        mu2: np.ndarray = None) -> float:
    """
    Bhattacharyya coefficient for Gaussians.
    
    BC = ∫√(p₁p₂) dx = exp(-D_B)
    
    where D_B is the Bhattacharyya distance.
    """
    Sigma1 = np.asarray(Sigma1, dtype=float)
    Sigma2 = np.asarray(Sigma2, dtype=float)
    Sigma_avg = 0.5 * (Sigma1 + Sigma2)
    
    _, logdet_avg = np.linalg.slogdet(Sigma_avg)
    _, logdet1 = np.linalg.slogdet(Sigma1)
    _, logdet2 = np.linalg.slogdet(Sigma2)
    
    # D_B for covariance part
    db = 0.5 * logdet_avg - 0.25 * (logdet1 + logdet2)
    
    if mu1 is not None and mu2 is not None:
        mu1 = np.asarray(mu1, dtype=float).reshape(-1)
        mu2 = np.asarray(mu2, dtype=float).reshape(-1)
        diff = mu1 - mu2
        Sigma_avg_inv = np.linalg.inv(Sigma_avg)
        db += 0.125 * float(diff @ Sigma_avg_inv @ diff)
    
    return float(np.exp(-db))


def bhattacharyya_distance_gaussian(Sigma1: np.ndarray, Sigma2: np.ndarray,
                                     mu1: np.ndarray = None,
                                     mu2: np.ndarray = None) -> float:
    """
    Bhattacharyya distance for Gaussians.
    
    D_B = -log(BC)
    """
    bc = bhattacharyya_coefficient_gaussian(Sigma1, Sigma2, mu1, mu2)
    return float(-math.log(max(bc, 1e-15)))

