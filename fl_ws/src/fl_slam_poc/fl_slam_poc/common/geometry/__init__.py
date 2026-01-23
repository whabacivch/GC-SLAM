"""
Geometry package for Golden Child SLAM v2.

SE(3) and SO(3) operations with NumPy and JAX backends.

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
from fl_slam_poc.common.geometry.se3_numpy import (
    skew,
    unskew,
    rotvec_to_rotmat,
    rotmat_to_rotvec,
    quat_to_rotmat,
    rotmat_to_quat,
    quat_to_rotvec,
    se3_compose,
    se3_inverse,
    se3_relative,
    se3_apply,
    se3_adjoint,
    se3_cov_compose,
    se3_exp,
    se3_log,
)

__all__ = [
    # SO(3) operations
    "skew",
    "unskew",
    "rotvec_to_rotmat",
    "rotmat_to_rotvec",
    # Quaternion operations
    "quat_to_rotmat",
    "rotmat_to_quat",
    "quat_to_rotvec",
    # SE(3) operations
    "se3_compose",
    "se3_inverse",
    "se3_relative",
    "se3_apply",
    "se3_adjoint",
    "se3_cov_compose",
    "se3_exp",
    "se3_log",
]
