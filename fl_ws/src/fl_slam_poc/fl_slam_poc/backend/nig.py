"""
Normal-Inverse-Gamma (NIG) conjugate model for unknown mean and variance.

Generative model:
    σ² ~ InverseGamma(α, β)
    μ | σ² ~ N(μ₀, σ²/κ)
    x | μ, σ² ~ N(μ, σ²)

The predictive distribution p(x | data) is Student-t with:
    ν = 2α (degrees of freedom)
    σ² = β(κ+1)/(ακ) (scale)

This admits CLOSED-FORM Fisher-Rao distance (Miyamoto et al. 2024).

Reference: Murphy (2007) "Conjugate Bayesian analysis of the Gaussian distribution"
"""

import math

import numpy as np
from scipy.special import gammaln

from fl_slam_poc.backend.information_distances import fisher_rao_student_t


# Principled priors: α > 1 ensures finite predictive variance
NIG_PRIOR_KAPPA = 1.0
NIG_PRIOR_ALPHA = 2.0  # α > 1 required for finite variance
NIG_PRIOR_BETA = 1.0


class NIGModel:
    """
    Normal-Inverse-Gamma model for descriptor components.
    
    Each component is modeled independently with shared prior structure.
    The predictive is Student-t, enabling Fisher-Rao association.
    """
    def __init__(self, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
        self.mu = np.asarray(mu, dtype=float)
        self.kappa = np.asarray(kappa, dtype=float)
        self.alpha = np.asarray(alpha, dtype=float)
        self.beta = np.asarray(beta, dtype=float)

    @classmethod
    def from_prior(cls, mu: np.ndarray, kappa: float, alpha: float, beta: float):
        """Create from scalar prior parameters."""
        mu = np.asarray(mu, dtype=float)
        return cls(
            mu=mu.copy(),
            kappa=np.full_like(mu, float(kappa)),
            alpha=np.full_like(mu, float(alpha)),
            beta=np.full_like(mu, float(beta)),
        )

    def copy(self):
        return NIGModel(
            mu=self.mu.copy(),
            kappa=self.kappa.copy(),
            alpha=self.alpha.copy(),
            beta=self.beta.copy(),
        )

    def predictive_variance(self) -> np.ndarray:
        """Variance of Student-t predictive."""
        alpha = np.maximum(self.alpha, 1.0 + 1e-6)  # Ensure finite
        return self.beta * (self.kappa + 1.0) / (self.kappa * (alpha - 1.0))

    def predictive_sigma(self) -> np.ndarray:
        """Standard deviation of Student-t predictive."""
        return np.sqrt(self.predictive_variance())

    def predictive_nu(self) -> np.ndarray:
        """Degrees of freedom of Student-t predictive."""
        return 2.0 * self.alpha

    def update(self, x: np.ndarray, weight: float = 1.0):
        """
        Bayesian update with weighted observation.
        
        Updates follow standard NIG conjugacy:
            κ' = κ + w
            μ' = (κμ + wx) / κ'
            α' = α + w/2
            β' = β + w(x-μ)²κ/(2κ')
        """
        if weight <= 0.0:
            return
        w = float(weight)
        x = np.asarray(x, dtype=float)
        
        kappa_new = self.kappa + w
        mu_new = (self.kappa * self.mu + w * x) / kappa_new
        alpha_new = self.alpha + 0.5 * w
        beta_new = self.beta + 0.5 * (self.kappa * w / kappa_new) * (x - self.mu) ** 2

        self.kappa = kappa_new
        self.mu = mu_new
        self.alpha = alpha_new
        self.beta = beta_new

    def log_predictive(self, x: np.ndarray) -> float:
        """Log-probability under Student-t predictive (for debugging)."""
        nu = 2.0 * self.alpha
        scale2 = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        z = (x - self.mu) ** 2 / (nu * scale2)
        log_norm = (gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) 
                   - 0.5 * (np.log(np.pi * nu * scale2)))
        log_prob = log_norm - 0.5 * (nu + 1.0) * np.log1p(z)
        return float(np.sum(log_prob))

    def fisher_rao_distance(self, other: "NIGModel") -> float:
        """
        Fisher-Rao distance via Student-t predictive (product manifold).
        
        This is a TRUE METRIC satisfying symmetry and triangle inequality.
        """
        d_sq = 0.0
        for i in range(len(self.mu)):
            mu1, mu2 = float(self.mu[i]), float(other.mu[i])
            sigma1 = float(self.predictive_sigma()[i])
            sigma2 = float(other.predictive_sigma()[i])
            # Average degrees of freedom for comparison
            nu = 0.5 * (float(self.predictive_nu()[i]) + float(other.predictive_nu()[i]))
            nu = max(nu, 1.0)
            d_i = fisher_rao_student_t(mu1, sigma1, mu2, sigma2, nu)
            d_sq += d_i * d_i
        return math.sqrt(d_sq)

