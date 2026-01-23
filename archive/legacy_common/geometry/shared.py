"""
Shared geometry constants and utilities.

These are backend-agnostic (work with both NumPy and JAX).
"""

from __future__ import annotations

import math

# =============================================================================
# Numerical Constants (stability, not policy)
# =============================================================================

# For small-angle approximations: use when θ < ε to avoid division by ~0
# Choice: ~sqrt(machine_epsilon) ≈ 1e-8, use 1e-10 for safety margin
ROTATION_EPSILON: float = 1e-10

# For π-singularity handling: eigenvalue decomposition threshold
# Choice: 1e-6 provides stable numerics near θ = π
SINGULARITY_EPSILON: float = 1e-6

# Half of π for angle wrapping
HALF_PI: float = math.pi / 2.0

# Full rotation angle
TWO_PI: float = 2.0 * math.pi

# Machine epsilon for float64
FLOAT64_EPS: float = 2.220446049250313e-16

# Safe minimum for division
SAFE_MIN: float = 1e-12


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π] range.
    
    Args:
        angle: Input angle in radians
        
    Returns:
        Normalized angle in [-π, π]
    """
    while angle > math.pi:
        angle -= TWO_PI
    while angle < -math.pi:
        angle += TWO_PI
    return angle


def angle_distance(a: float, b: float) -> float:
    """
    Compute shortest angular distance between two angles.
    
    Args:
        a: First angle in radians
        b: Second angle in radians
        
    Returns:
        Shortest distance in [0, π]
    """
    diff = normalize_angle(a - b)
    return abs(diff)
