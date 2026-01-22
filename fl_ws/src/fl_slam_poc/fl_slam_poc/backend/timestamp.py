"""
Probabilistic timestamp alignment model.

Replaces hard gates (e.g., max_alignment_dt_sec) with soft probabilistic weighting.

Generative model:
    dt ~ N(0, σ²)
    where σ is learned online from observed alignment offsets.
    
Weight function:
    w(dt) = exp(-0.5 * (dt/σ)²)
    
This is the likelihood of perfect alignment given observed offset dt.
"""

import math
from typing import Optional

from fl_slam_poc.backend.adaptive import AdaptiveParameter


class TimeAlignmentModel:
    """
    Probabilistic timestamp alignment using Gaussian likelihood.
    
    Instead of hard-gating (reject if |dt| > threshold), this model
    returns a soft weight based on how likely the alignment is good.
    
    The sigma parameter is learned online with a prior for stability.
    """
    def __init__(self, prior_sigma: float, prior_strength: float, sigma_floor: float) -> None:
        """
        Args:
            prior_sigma: Initial estimate of alignment std dev (seconds)
            prior_strength: Pseudo-count for prior (higher = slower adaptation)
            sigma_floor: Minimum sigma to prevent division by zero
        """
        self.sigma_param = AdaptiveParameter(prior_sigma, prior_strength, sigma_floor)

    def update(self, dt: Optional[float]) -> None:
        """Update sigma estimate with observed alignment offset."""
        if dt is None:
            return
        self.sigma_param.update(abs(float(dt)))

    def sigma(self) -> float:
        """Current estimate of alignment standard deviation."""
        return self.sigma_param.value()

    def weight(self, dt: Optional[float]) -> float:
        """
        Gaussian likelihood weight for timestamp offset.
        
        Returns 1.0 for perfect alignment, decreasing smoothly with |dt|.
        """
        if dt is None:
            return 1.0
        sigma = self.sigma()
        if sigma <= 0.0:
            return 1.0
        z = float(dt) / sigma
        return math.exp(-0.5 * z * z)
    
    def log_likelihood(self, dt: Optional[float]) -> float:
        """Log-likelihood of timestamp offset under Gaussian model."""
        if dt is None:
            return 0.0
        sigma = self.sigma()
        if sigma <= 0.0:
            return 0.0
        z = float(dt) / sigma
        return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2 * math.pi)

