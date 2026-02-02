"""
Geometry package for Geometric Compositional SLAM v2.

Runtime backend uses JAX-only `se3_jax` functions.
NumPy helpers have been archived under `archive/legacy_common/geometry/`.

Modules:
- se3_numpy: NumPy-based SE(3) operations (CPU)
- se3_jax: JAX-based SE(3) operations (GPU-accelerated)

Usage:
    # NumPy SE(3) operations
    from fl_slam_poc.common.geometry.se3_numpy import (
        se3_compose,
        se3_inverse,
        rotvec_to_rotmat,
    )
    
    # JAX SE(3) operations
    from fl_slam_poc.common.geometry.se3_jax import (
        se3_compose_jax,
        se3_inverse_jax,
    )
"""

from __future__ import annotations

# NumPy SE(3) operations
from fl_slam_poc.common.geometry import se3_jax

__all__ = [
    "se3_jax",
]
