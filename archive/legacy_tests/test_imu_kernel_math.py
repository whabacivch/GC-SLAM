"""
NOTE (repo hygiene):
This test file originally referenced a legacy IMU kernel that lives under `archive/`
and is not part of the live package import graph. As part of the GC v2 spec upgrade,
we replace it with foundational Lie-primitive correctness tests.
"""

import numpy as np

from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common.geometry import se3_jax


def test_hat_vee_roundtrip():
    w = jnp.array([0.3, -0.2, 0.1], dtype=jnp.float64)
    W = se3_jax.skew(w)
    w2 = se3_jax.vee(W)
    np.testing.assert_allclose(np.asarray(w2), np.asarray(w), atol=1e-12, rtol=0.0)


def test_so3_right_jacobian_inverse_consistency():
    phi = jnp.array([0.2, -0.1, 0.05], dtype=jnp.float64)
    Jr = se3_jax.so3_right_jacobian(phi)
    Jr_inv = se3_jax.so3_right_jacobian_inv(phi)
    I = jnp.eye(3, dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(Jr_inv @ Jr), np.asarray(I), atol=1e-10, rtol=0.0)


def test_se3_exp_log_roundtrip_small():
    # xi = [rho, phi] in se(3)
    xi = jnp.array([0.1, -0.05, 0.02, 0.2, -0.1, 0.05], dtype=jnp.float64)
    T = se3_jax.se3_exp(xi)     # group element [t, rotvec]
    xi2 = se3_jax.se3_log(T)    # back to twist [rho, phi]
    np.testing.assert_allclose(np.asarray(xi2), np.asarray(xi), atol=1e-9, rtol=0.0)


def test_se3_V_matches_translation_part():
    xi = jnp.array([0.2, 0.1, -0.05, 0.15, -0.05, 0.02], dtype=jnp.float64)
    rho = xi[:3]
    phi = xi[3:6]
    T = se3_jax.se3_exp(xi)
    t = T[:3]
    t2 = se3_jax.se3_V(phi) @ rho
    np.testing.assert_allclose(np.asarray(t2), np.asarray(t), atol=1e-12, rtol=0.0)
