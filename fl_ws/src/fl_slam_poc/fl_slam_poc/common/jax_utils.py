"""
JAX utility helpers for backend-only conversions and SE(3) wrappers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common.geometry import se3_jax


def to_jax(value: Any, dtype=jnp.float64) -> jnp.ndarray:
    """Convert array-like input to JAX array with desired dtype."""
    return jnp.asarray(value, dtype=dtype)


def to_numpy(value: Any, dtype=float) -> np.ndarray:
    """Convert JAX/array-like input to NumPy array with desired dtype."""
    return np.asarray(value, dtype=dtype)


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to rotation vector using JAX ops.
    """
    q = to_jax(quat).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quat_to_rotvec: expected shape (4,), got {tuple(q.shape)}")
    norm = float(jnp.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError(f"quat_to_rotvec: quaternion norm too small ({norm:.3e})")
    q = q / norm
    x, y, z, w = q[0], q[1], q[2], q[3]
    R = jnp.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])
    rotvec = se3_jax.so3_log(R)
    return to_numpy(rotvec)


def se3_compose_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose SE(3) transforms via JAX and return NumPy."""
    return to_numpy(se3_jax.se3_compose(to_jax(a), to_jax(b)))


def se3_inverse_np(a: np.ndarray) -> np.ndarray:
    """Invert SE(3) transform via JAX and return NumPy."""
    return to_numpy(se3_jax.se3_inverse(to_jax(a)))


def se3_relative_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative transform via JAX: b^{-1} ∘ a, returned as NumPy."""
    return to_numpy(se3_jax.se3_relative(to_jax(a), to_jax(b)))


def se3_adjoint_np(a: np.ndarray) -> np.ndarray:
    """Adjoint via JAX, returned as NumPy."""
    return to_numpy(se3_jax.se3_adjoint(to_jax(a)))


def se3_cov_compose_np(cov_a: np.ndarray, cov_b: np.ndarray, pose_a: np.ndarray) -> np.ndarray:
    """Compose covariances via JAX adjoint transport, returned as NumPy."""
    return to_numpy(se3_jax.se3_cov_compose(to_jax(cov_a), to_jax(cov_b), to_jax(pose_a)))
