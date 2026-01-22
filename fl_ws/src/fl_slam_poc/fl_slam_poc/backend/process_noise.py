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


class WishartPrior:
    """
    Wishart prior for precision matrix (inverse covariance).

    Maintains a conjugate prior for adaptive noise estimation.
    """
    def __init__(self, n: float, S: np.ndarray):
        self.n = float(n)
        self.S = np.asarray(S, dtype=float)

    @property
    def dim(self) -> int:
        return self.S.shape[0]

    @property
    def mean_precision(self) -> np.ndarray:
        return self.n * self.S

    @property
    def mean_covariance(self) -> np.ndarray:
        denom = self.n - self.dim - 1
        if denom <= 0:
            denom = 1.0
        return np.linalg.inv(self.S) / denom

    @classmethod
    def from_mean_covariance(cls, mean_cov: np.ndarray, dof=None) -> "WishartPrior":
        mean_cov = np.asarray(mean_cov, dtype=float)
        dim = mean_cov.shape[0]
        n = float(dof) if dof is not None else float(dim + 2)
        denom = n - dim - 1
        if denom <= 0:
            denom = 1.0
        S = np.linalg.inv(mean_cov) / denom
        return cls(n=n, S=S)

    def update_with_forgetting(self, residuals: np.ndarray, forgetting_factor: float) -> "WishartPrior":
        residuals = np.asarray(residuals, dtype=float)
        if residuals.size == 0:
            return self
        if residuals.ndim == 1:
            residuals = residuals.reshape(1, -1)
        scatter = residuals.T @ residuals

        n_forgotten = forgetting_factor * self.n
        # Maintain consistent scaling between (n, S) under forgetting.
        # We treat inv(S) as the accumulated pseudo-scatter scale; scale it by
        # the forgotten effective sample size n_forgotten.
        S_inv_forgotten = n_forgotten * np.linalg.inv(self.S)
        S_inv_new = S_inv_forgotten + scatter
        S_new = np.linalg.inv(S_inv_new)
        n_new = n_forgotten + residuals.shape[0]
        return WishartPrior(n=n_new, S=S_new)


class AdaptiveIMUNoiseModel:
    """
    Self-adaptive IMU noise model using Wishart conjugate updates.

    Maintains priors for bias random walk covariances.
    """
    def __init__(
        self,
        accel_bias_prior: WishartPrior,
        gyro_bias_prior: WishartPrior,
        forgetting_factor: float = 0.995,
    ):
        self.accel_bias = accel_bias_prior
        self.gyro_bias = gyro_bias_prior
        self.gamma = float(forgetting_factor)
        self.history = {
            "accel_bias_trace": [],
            "gyro_bias_trace": [],
        }

    def update_from_bias_innovations(
        self,
        accel_bias_innovations: np.ndarray,
        gyro_bias_innovations: np.ndarray,
        dt: float,
    ) -> None:
        if dt <= 0.0:
            return
        accel_bias_innovations = np.asarray(accel_bias_innovations, dtype=float)
        gyro_bias_innovations = np.asarray(gyro_bias_innovations, dtype=float)
        accel_rw = accel_bias_innovations / np.sqrt(dt)
        gyro_rw = gyro_bias_innovations / np.sqrt(dt)

        self.accel_bias = self.accel_bias.update_with_forgetting(accel_rw, self.gamma)
        self.gyro_bias = self.gyro_bias.update_with_forgetting(gyro_rw, self.gamma)

        self.history["accel_bias_trace"].append(float(np.trace(self.accel_bias.mean_covariance)))
        self.history["gyro_bias_trace"].append(float(np.trace(self.gyro_bias.mean_covariance)))

    def get_current_noise_params(self) -> dict:
        return {
            "accel_bias_cov": self.accel_bias.mean_covariance,
            "gyro_bias_cov": self.gyro_bias.mean_covariance,
            "accel_bias_confidence": float(self.accel_bias.n),
            "gyro_bias_confidence": float(self.gyro_bias.n),
        }

