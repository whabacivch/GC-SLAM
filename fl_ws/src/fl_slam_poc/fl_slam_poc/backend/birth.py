"""
Stochastic birth model for anchor creation.

Replaces deterministic threshold (if r_new > threshold) with Poisson process.

Generative model:
    Birth events follow a Poisson process with intensity:
        λ(r) = λ₀ * r
    where r is the new-component responsibility.
    
    Probability of birth in time step dt:
        P(birth | r) = 1 - exp(-λ(r) * dt)
    
    For small dt: P(birth | r) ≈ λ₀ * r * dt

This makes anchor creation probabilistic, avoiding hard thresholds.
"""

import math

import numpy as np


class StochasticBirthModel:
    """
    Poisson process model for anchor birth.
    
    Higher responsibility r_new leads to higher probability of creating
    a new anchor, but the decision is stochastic.
    """
    def __init__(self, base_intensity: float, time_step: float):
        """
        Args:
            base_intensity: λ₀, base birth rate (higher = more anchors)
            time_step: dt, time between observations (e.g., scan period)
        """
        self.base_intensity = float(base_intensity)
        self.time_step = float(time_step)
        self.rng = np.random.default_rng()
    
    def sample_birth(self, r_new: float) -> bool:
        """
        Sample whether to create new anchor given r_new.
        
        Returns True with probability 1 - exp(-λ₀ * r_new * dt).
        """
        if r_new <= 0.0:
            return False
        intensity = self.base_intensity * float(r_new)
        prob_birth = 1.0 - math.exp(-intensity * self.time_step)
        return self.rng.random() < prob_birth
    
    def birth_probability(self, r_new: float) -> float:
        """Return probability of birth (for logging)."""
        if r_new <= 0.0:
            return 0.0
        intensity = self.base_intensity * float(r_new)
        return 1.0 - math.exp(-intensity * self.time_step)

