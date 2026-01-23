"""
Stochastic birth model for anchor creation.

Replaces deterministic threshold (if r_new > threshold) with Poisson process.

Generative model:
    Birth events follow a Poisson process with intensity:
        λ(r) = λ₀ * concentration_scale * r
    where r is the new-component responsibility and concentration_scale
    is adapted from system-wide entropy (per Self-Adaptive Systems Guide).
    
    Probability of birth in time step dt:
        P(birth | r) = 1 - exp(-λ(r) * dt)
    
    For small dt: P(birth | r) ≈ λ₀ * concentration_scale * r * dt

This makes anchor creation probabilistic, avoiding hard thresholds.

**Self-Adaptive Systems Invariants:**
- "Startup Is Not a Mode": behavior emerges from posterior uncertainty, not time-based logic
- concentration_scale adapts from system-wide entropy (Dirichlet concentration tracking)
- base_intensity is a prior hyperparameter, not a "reasonable default"
"""

import math
from typing import List, Optional

import numpy as np


# Prior hyperparameters for concentration adaptation
TARGET_ENTROPY_FRACTION = 0.5  # Prior: target entropy as fraction of max
CONCENTRATION_ADAPTATION_RATE = 0.1  # Prior: proportional gain (trust region step size)
CONCENTRATION_MIN = 0.1  # Prior: minimum concentration scale
CONCENTRATION_MAX = 10.0  # Prior: maximum concentration scale


class StochasticBirthModel:
    """
    Poisson process model for anchor birth with adaptive concentration.
    
    Higher responsibility r_new leads to higher probability of creating
    a new anchor, but the decision is stochastic.
    
    **Adaptive Concentration (Self-Adaptive Systems Compliant):**
    - concentration_scale adapts based on system-wide entropy of responsibilities
    - If entropy is too high (associations too uncertain) → increase scale → more births
    - If entropy is too low (over-confident) → decrease scale → fewer births
    - This replaces fixed birth_intensity with uncertainty-driven behavior
    """
    def __init__(self, base_intensity: float, time_step: float, rng_seed: Optional[int] = None):
        """
        Args:
            base_intensity: λ₀, base birth rate prior (higher = more anchors)
            time_step: dt, time between observations (e.g., scan period)
            rng_seed: Optional RNG seed for deterministic behavior
        """
        self.base_intensity = float(base_intensity)
        self.time_step = float(time_step)
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        
        # Adaptive concentration state
        self.concentration_scale = 1.0  # Start at neutral scale
        self._entropy_history: List[float] = []
    
    @property
    def effective_intensity(self) -> float:
        """Current effective intensity = base_intensity * concentration_scale."""
        return self.base_intensity * self.concentration_scale
    
    def sample_birth(self, r_new: float) -> bool:
        """
        Sample whether to create new anchor given r_new.
        
        Returns True with probability 1 - exp(-λ_eff * r_new * dt).
        """
        if r_new <= 0.0:
            return False
        intensity = self.effective_intensity * float(r_new)
        prob_birth = 1.0 - math.exp(-intensity * self.time_step)
        return self.rng.random() < prob_birth
    
    def birth_probability(self, r_new: float) -> float:
        """Return probability of birth (for logging)."""
        if r_new <= 0.0:
            return 0.0
        intensity = self.effective_intensity * float(r_new)
        return 1.0 - math.exp(-intensity * self.time_step)
    
    def update_concentration_from_responsibilities(
        self, 
        responsibilities: np.ndarray,
        n_anchors: int
    ) -> float:
        """
        Update concentration_scale based on entropy of responsibilities.
        
        **Adaptation Logic (from Self-Adaptive Guide Section 3):**
        - Compute entropy of responsibility distribution
        - Compare to target entropy (fraction of max entropy)
        - Adjust concentration_scale proportionally to entropy error
        
        Args:
            responsibilities: Array of association responsibilities (sums to 1)
            n_anchors: Number of existing anchors (for max entropy calculation)
            
        Returns:
            Updated concentration_scale value
        """
        if len(responsibilities) == 0 or n_anchors <= 0:
            return self.concentration_scale
        
        # Compute entropy of responsibilities
        # H = -Σ p_i log(p_i) with safe handling of zeros
        r = np.asarray(responsibilities, dtype=float)
        r = np.maximum(r, 1e-10)  # Avoid log(0)
        r = r / np.sum(r)  # Ensure normalization
        entropy = -float(np.sum(r * np.log(r)))
        
        # Max entropy for uniform distribution over (n_anchors + 1) components
        # (+1 for new component)
        max_entropy = math.log(max(n_anchors + 1, 2))
        target_entropy = TARGET_ENTROPY_FRACTION * max_entropy
        
        # Track entropy history
        self._entropy_history.append(entropy)
        if len(self._entropy_history) > 50:
            self._entropy_history.pop(0)
        
        # Compute entropy error
        entropy_error = entropy - target_entropy
        
        # PI controller adjustment (simplified to proportional only)
        # Positive error (high entropy) → increase scale → more births
        # Negative error (low entropy) → decrease scale → fewer births
        adjustment = CONCENTRATION_ADAPTATION_RATE * entropy_error
        
        # Apply adjustment (multiplicative for scale)
        self.concentration_scale *= math.exp(adjustment)
        
        # Clamp to bounds
        self.concentration_scale = max(CONCENTRATION_MIN, 
                                       min(CONCENTRATION_MAX, self.concentration_scale))
        
        return self.concentration_scale
    
    def get_diagnostics(self) -> dict:
        """Get diagnostics for monitoring."""
        return {
            "base_intensity": self.base_intensity,
            "concentration_scale": self.concentration_scale,
            "effective_intensity": self.effective_intensity,
            "recent_entropy": self._entropy_history[-1] if self._entropy_history else 0.0,
            "avg_entropy": float(np.mean(self._entropy_history)) if self._entropy_history else 0.0,
            "rng_seed": self.rng_seed,
        }

