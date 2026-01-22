"""
Adaptive process noise estimation.

Replaces hardcoded Q with online Bayesian estimation from prediction residuals.

Generative model:
    Q ~ InverseWishart(ν₀, Ψ₀)
    residuals | Q ~ N(0, Q)
    
Posterior:
    Q | residuals ~ InverseWishart(ν₀ + n, Ψ₀ + Σ residuals residuals^T)
    
Point estimate (posterior mean):
    E[Q] = Ψ / (ν - p - 1) where p = dimension

Reference: Gelman et al. (2013) Bayesian Data Analysis, Chapter 3
"""

import numpy as np


class AdaptiveProcessNoise:
    """
    Inverse-Wishart model for process noise covariance.
    
    Learns the noise covariance from observed prediction residuals,
    starting from a diagonal prior.
    """
    def __init__(
        self,
        dim: int,
        prior_scale: np.ndarray,
        prior_dof: float,
        residual_sum: np.ndarray,
        count: int,
    ):
        self.dim = dim
        self.prior_scale = prior_scale
        self.prior_dof = prior_dof
        self.residual_sum = residual_sum
        self.count = count
    
    @classmethod
    def create(cls, dim: int, prior_diagonal: np.ndarray, prior_strength: float = 10.0):
        """
        Create with diagonal prior.
        
        Args:
            dim: State dimension (e.g., 6 for SE(3))
            prior_diagonal: Prior diagonal variances [σ²_x, σ²_y, ..., σ²_rz]
            prior_strength: Pseudo-count (higher = slower adaptation)
        """
        prior_diagonal = np.asarray(prior_diagonal, dtype=float)
        prior_scale = np.diag(prior_diagonal) * prior_strength
        return cls(
            dim=dim,
            prior_scale=prior_scale,
            prior_dof=prior_strength + dim + 1,  # Ensure E[Q] exists
            residual_sum=np.zeros((dim, dim), dtype=float),
            count=0,
        )
    
    def update(self, residual: np.ndarray) -> None:
        """Update with observed prediction residual."""
        r = np.asarray(residual, dtype=float).reshape(-1, 1)
        self.residual_sum += r @ r.T
        self.count += 1
    
    def estimate(self) -> np.ndarray:
        """
        Posterior mean estimate of Q.
        
        E[Q] = (Ψ₀ + Σrr^T) / (ν₀ + n - p - 1)
        """
        posterior_scale = self.prior_scale + self.residual_sum
        posterior_dof = self.prior_dof + self.count
        divisor = posterior_dof - self.dim - 1
        if divisor <= 0:
            divisor = 1.0  # Fallback if insufficient data
        return posterior_scale / divisor
    
    def confidence(self) -> float:
        """Confidence in estimate (0 = prior only, 1 = data-dominated)."""
        if self.count == 0:
            return 0.0
        return self.count / (self.prior_dof + self.count)

