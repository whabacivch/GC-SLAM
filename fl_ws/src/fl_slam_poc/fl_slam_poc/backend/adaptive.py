"""
Adaptive parameter estimation using online Bayesian updates.

These replace hardcoded constants with data-driven estimates that have
principled priors. All estimates are the posterior mean given observations.
"""

import math


class OnlineStats:
    """
    Welford's online algorithm for mean and variance.
    
    Numerically stable single-pass computation.
    Reference: Welford (1962)
    """
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / float(self.count)
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.M2 / float(self.count - 1)

    def stddev(self) -> float:
        return math.sqrt(self.variance())


class AdaptiveParameter:
    """
    Adaptive parameter with Bayesian online estimation.
    
    Uses Normal prior with strength parameter for regularization.
    The returned value is the posterior mean:
        
        posterior_mean = (prior_strength * prior_mean + n * sample_mean) / (prior_strength + n)
    
    This avoids fully hardcoded values while providing stable initialization.
    
    Generative model:
        θ ~ N(prior_mean, prior_variance / prior_strength)
        x_i | θ ~ N(θ, σ²)
    """
    def __init__(self, prior_mean: float, prior_strength: float, floor: float = 0.0):
        """
        Args:
            prior_mean: Initial estimate before any data
            prior_strength: Effective number of pseudo-observations (higher = more sticky to prior)
            floor: Minimum value (hard constraint for stability)
        """
        self.prior_mean = float(prior_mean)
        self.prior_strength = float(prior_strength)
        self.floor = float(floor)
        self.stats = OnlineStats()
    
    def update(self, value: float) -> None:
        """Update with new observation."""
        self.stats.update(value)
    
    def value(self) -> float:
        """Posterior mean with prior regularization."""
        if self.stats.count == 0:
            return max(self.prior_mean, self.floor)
        total_weight = self.prior_strength + self.stats.count
        posterior_mean = (self.prior_strength * self.prior_mean + 
                         self.stats.count * self.stats.mean) / total_weight
        return max(posterior_mean, self.floor)
    
    def confidence(self) -> float:
        """Confidence in estimate (0 = prior only, 1 = data-dominated)."""
        if self.stats.count == 0:
            return 0.0
        return self.stats.count / (self.prior_strength + self.stats.count)

