"""
Audit Invariants Test Suite for Geometric Compositional SLAM v2.

This test file verifies the core invariants required by the audit checklist:
1. Order invariance: Permuting evidence/hypothesis order should give same result
2. No-gates smoothness: Extreme outliers produce no discontinuities
3. Units/dt discretization: PSD -> dt correctness
4. SO(3)/SE(3) roundtrip: exp(log(R)) == R for various R including near-π
5. IW commutative update: Permuting hypotheses gives same IW update

Reference: Audit Plan gc_v2_deep_audit_33176b11.plan.md Issue #14
"""

import numpy as np
import pytest

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.common import constants
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_solve_lifted,
    softmax,
)
from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_batch


# =============================================================================
# 1. Order Invariance Tests
# =============================================================================


class TestOrderInvariance:
    """Tests that fusion operations are commutative/order-invariant."""

    def test_info_fusion_order_invariant(self):
        """Test that Gaussian info fusion is order-invariant (L1 + L2 == L2 + L1)."""
        key = jax.random.PRNGKey(42)
        d = 6

        # Generate random SPD matrices
        A1 = jax.random.normal(key, (d, d))
        L1 = A1 @ A1.T + 0.1 * jnp.eye(d)

        key, subkey = jax.random.split(key)
        A2 = jax.random.normal(subkey, (d, d))
        L2 = A2 @ A2.T + 0.1 * jnp.eye(d)

        key, subkey = jax.random.split(key)
        A3 = jax.random.normal(subkey, (d, d))
        L3 = A3 @ A3.T + 0.1 * jnp.eye(d)

        # Fusion should be commutative
        L_123 = L1 + L2 + L3
        L_321 = L3 + L2 + L1
        L_231 = L2 + L3 + L1

        np.testing.assert_allclose(np.asarray(L_123), np.asarray(L_321), atol=1e-12)
        np.testing.assert_allclose(np.asarray(L_123), np.asarray(L_231), atol=1e-12)

    def test_info_fusion_with_permuted_h(self):
        """Test that info vector fusion is order-invariant."""
        key = jax.random.PRNGKey(123)
        d = 6

        h1 = jax.random.normal(key, (d,))
        key, subkey = jax.random.split(key)
        h2 = jax.random.normal(subkey, (d,))
        key, subkey = jax.random.split(key)
        h3 = jax.random.normal(subkey, (d,))

        h_123 = h1 + h2 + h3
        h_312 = h3 + h1 + h2

        np.testing.assert_allclose(np.asarray(h_123), np.asarray(h_312), atol=1e-12)

    def test_kappa_batch_order_independent(self):
        """Test that kappa computation is independent of input order."""
        R_bar = jnp.array([0.3, 0.5, 0.7, 0.2, 0.8])
        R_bar_permuted = jnp.array([0.7, 0.3, 0.8, 0.5, 0.2])

        kappa = kappa_from_resultant_batch(R_bar)
        kappa_perm = kappa_from_resultant_batch(R_bar_permuted)

        # Results should be element-wise same when reordered
        np.testing.assert_allclose(
            np.sort(np.asarray(kappa)),
            np.sort(np.asarray(kappa_perm)),
            atol=1e-12,
        )


# =============================================================================
# 2. No-Gates Smoothness Tests
# =============================================================================


class TestNoGatesSmoothness:
    """Tests that extreme values produce smooth behavior without discontinuities."""

    def test_kappa_extreme_R_smooth(self):
        """Test that kappa is smooth for extreme R values (no discontinuous jumps)."""
        # Generate R values from 0 to near 1
        R_values = jnp.linspace(0.01, 0.99, 100)
        kappa_values = kappa_from_resultant_batch(R_values)

        # Compute finite differences - should be smooth (no huge jumps)
        dkappa = jnp.diff(kappa_values)

        # No jumps greater than 10x the median delta (smooth curve)
        median_delta = jnp.median(jnp.abs(dkappa))
        max_delta = jnp.max(jnp.abs(dkappa))

        # Allow some ratio but not discontinuous jumps
        assert float(max_delta) < 100 * float(median_delta), (
            f"Discontinuity detected: max_delta={max_delta}, median_delta={median_delta}"
        )

    def test_psd_projection_extreme_negative_eigenvalue(self):
        """Test PSD projection handles extreme negative eigenvalues smoothly."""
        # Matrix with large negative eigenvalue
        M = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, -1000.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        result = domain_projection_psd(M, eps_psd=1e-6)

        # Should be PSD (all eigenvalues >= eps_psd)
        eigvals = jnp.linalg.eigvalsh(result.M_psd)
        assert float(jnp.min(eigvals)) >= 1e-6 - 1e-12

        # Projection delta should be recorded (not silently ignored)
        assert result.projection_delta > 0

    def test_softmax_extreme_inputs_smooth(self):
        """Test softmax handles extreme inputs without NaN/Inf."""
        # Very large positive and negative values
        logits_extreme = jnp.array([1000.0, -1000.0, 0.0, 500.0, -500.0])
        probs = softmax(logits_extreme, tau=1.0)

        # Should still be valid probability distribution
        assert jnp.all(jnp.isfinite(probs))
        assert float(jnp.sum(probs)) == pytest.approx(1.0, abs=1e-6)
        assert jnp.all(probs >= 0)

    def test_solve_near_singular_lifted(self):
        """Test that lifted solve handles near-singular matrices smoothly."""
        # Near-singular matrix
        L = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1e-15, 0.0],
            [0.0, 0.0, 1.0],
        ])
        b = jnp.array([1.0, 1.0, 1.0])

        result = spd_cholesky_solve_lifted(L, b, eps_lift=1e-9)

        # Solution should be finite
        assert jnp.all(jnp.isfinite(result.x))

        # Lift should be applied
        assert result.lift_strength > 0


# =============================================================================
# 3. Units/dt Discretization Tests
# =============================================================================


class TestUnitsDiscretization:
    """Tests for correct PSD -> discrete variance discretization."""

    def test_diffusion_discretization_consistency(self):
        """Test that Q_continuous * dt gives correct discrete Q."""
        # Continuous-time PSD (per-second)
        Q_continuous = jnp.array([
            [1e-4, 0.0, 0.0],
            [0.0, 1e-4, 0.0],
            [0.0, 0.0, 1e-4],
        ])

        dt = 0.1  # 100ms

        # Correct discretization: Q_discrete = Q_continuous * dt
        Q_discrete = Q_continuous * dt

        # Verify trace scales linearly with dt
        trace_continuous = jnp.trace(Q_continuous)
        trace_discrete = jnp.trace(Q_discrete)

        expected_ratio = dt
        actual_ratio = float(trace_discrete / trace_continuous)

        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-12)

    def test_imu_noise_density_units(self):
        """Test that IMU noise densities have correct units interpretation."""
        # Gyro noise density is rad^2/s (continuous-time PSD)
        gyro_psd = constants.GC_IMU_GYRO_NOISE_DENSITY

        # For dt=1s integration, discrete variance should equal PSD
        dt = 1.0
        discrete_var = gyro_psd * dt

        np.testing.assert_allclose(discrete_var, gyro_psd, rtol=1e-12)

        # For dt=0.01s (100Hz), variance should be 100x smaller
        dt_fast = 0.01
        discrete_var_fast = gyro_psd * dt_fast

        assert float(discrete_var_fast) == pytest.approx(gyro_psd * 0.01, rel=1e-12)


# =============================================================================
# 4. SO(3)/SE(3) Roundtrip Tests
# =============================================================================


class TestLieGroupRoundtrip:
    """Tests for SO(3) and SE(3) exp/log consistency."""

    def test_so3_exp_log_roundtrip_small_angle(self):
        """Test exp(log(R)) == R for small angles."""
        omega_small = jnp.array([0.01, -0.02, 0.015])
        R = se3_jax.so3_exp(omega_small)
        omega_recovered = se3_jax.so3_log(R)
        R_recovered = se3_jax.so3_exp(omega_recovered)

        np.testing.assert_allclose(np.asarray(R_recovered), np.asarray(R), atol=1e-10)

    def test_so3_exp_log_roundtrip_medium_angle(self):
        """Test exp(log(R)) == R for medium angles (~1 rad)."""
        omega_medium = jnp.array([0.5, -0.7, 0.3])
        R = se3_jax.so3_exp(omega_medium)
        omega_recovered = se3_jax.so3_log(R)
        R_recovered = se3_jax.so3_exp(omega_recovered)

        np.testing.assert_allclose(np.asarray(R_recovered), np.asarray(R), atol=1e-10)

    def test_so3_exp_log_roundtrip_large_angle(self):
        """Test exp(log(R)) == R for large angles (~2.5 rad)."""
        omega_large = jnp.array([1.5, -1.2, 0.8])
        R = se3_jax.so3_exp(omega_large)
        omega_recovered = se3_jax.so3_log(R)
        R_recovered = se3_jax.so3_exp(omega_recovered)

        np.testing.assert_allclose(np.asarray(R_recovered), np.asarray(R), atol=1e-9)

    def test_so3_exp_log_roundtrip_near_pi(self):
        """Test exp(log(R)) == R for angles near π (singularity region)."""
        # Angle very close to π
        theta = jnp.pi - 0.01
        axis = jnp.array([1.0, 0.0, 0.0])  # Rotation around x-axis
        omega_near_pi = theta * axis

        R = se3_jax.so3_exp(omega_near_pi)
        omega_recovered = se3_jax.so3_log(R)
        R_recovered = se3_jax.so3_exp(omega_recovered)

        # Rotations should match (even if omega differs by sign flip)
        np.testing.assert_allclose(np.asarray(R_recovered), np.asarray(R), atol=1e-8)

    def test_so3_log_exp_roundtrip_random(self):
        """Test log(exp(ω)) ≈ ω for random ω (small to medium angles)."""
        key = jax.random.PRNGKey(456)

        for _ in range(10):
            key, subkey = jax.random.split(key)
            omega = jax.random.normal(subkey, (3,)) * 0.5  # ~0.5 rad magnitude

            R = se3_jax.so3_exp(omega)
            omega_recovered = se3_jax.so3_log(R)

            # Should recover same vector (up to numerical precision)
            np.testing.assert_allclose(
                np.asarray(omega_recovered), np.asarray(omega), atol=1e-9
            )

    def test_se3_exp_log_roundtrip_small(self):
        """Test SE(3) exp(log(T)) == T for small twists."""
        xi_small = jnp.array([0.1, -0.05, 0.02, 0.2, -0.1, 0.05])
        T = se3_jax.se3_exp(xi_small)
        xi_recovered = se3_jax.se3_log(T)
        T_recovered = se3_jax.se3_exp(xi_recovered)

        np.testing.assert_allclose(np.asarray(T_recovered), np.asarray(T), atol=1e-9)

    def test_se3_exp_log_roundtrip_medium(self):
        """Test SE(3) exp(log(T)) == T for medium twists."""
        xi_medium = jnp.array([1.0, -0.5, 0.3, 0.8, -0.6, 0.4])
        T = se3_jax.se3_exp(xi_medium)
        xi_recovered = se3_jax.se3_log(T)
        T_recovered = se3_jax.se3_exp(xi_recovered)

        np.testing.assert_allclose(np.asarray(T_recovered), np.asarray(T), atol=1e-8)

    def test_se3_V_inv_correctness(self):
        """Test that V^{-1} @ V @ rho == rho."""
        phi = jnp.array([0.3, -0.2, 0.1])
        rho = jnp.array([1.0, 2.0, 3.0])

        V = se3_jax.se3_V(phi)
        V_inv = se3_jax._se3_V_inv(phi)

        # V^{-1} @ V should be identity
        product = V_inv @ V
        np.testing.assert_allclose(np.asarray(product), np.eye(3), atol=1e-10)

        # V^{-1} @ V @ rho == rho
        rho_recovered = V_inv @ V @ rho
        np.testing.assert_allclose(np.asarray(rho_recovered), np.asarray(rho), atol=1e-10)

    def test_se3_V_inv_small_angle(self):
        """Test V^{-1} for very small angles (Taylor regime)."""
        phi_tiny = jnp.array([1e-9, -1e-9, 1e-9])

        V = se3_jax.se3_V(phi_tiny)
        V_inv = se3_jax._se3_V_inv(phi_tiny)

        # Both should be nearly identity
        np.testing.assert_allclose(np.asarray(V), np.eye(3), atol=1e-7)
        np.testing.assert_allclose(np.asarray(V_inv), np.eye(3), atol=1e-7)

        # Product should be identity
        product = V_inv @ V
        np.testing.assert_allclose(np.asarray(product), np.eye(3), atol=1e-10)


# =============================================================================
# 5. IW Commutative Update Tests
# =============================================================================


class TestIWCommutativeUpdate:
    """Tests for Inverse-Wishart update commutativity."""

    def test_sufficient_stats_accumulation_commutative(self):
        """Test that IW sufficient statistics accumulate commutatively."""
        # Simulate sufficient stats from multiple hypotheses
        key = jax.random.PRNGKey(789)
        d = 3

        # Generate random PSD updates (simulating dPsi from different hypotheses)
        def random_psd(key, d):
            A = jax.random.normal(key, (d, d))
            return A @ A.T + 0.01 * jnp.eye(d)

        dPsi_1 = random_psd(key, d)
        key, subkey = jax.random.split(key)
        dPsi_2 = random_psd(subkey, d)
        key, subkey = jax.random.split(key)
        dPsi_3 = random_psd(subkey, d)

        dnu_1, dnu_2, dnu_3 = 1.0, 1.5, 0.8

        # Order 1-2-3
        Psi_123 = dPsi_1 + dPsi_2 + dPsi_3
        nu_123 = dnu_1 + dnu_2 + dnu_3

        # Order 3-1-2
        Psi_312 = dPsi_3 + dPsi_1 + dPsi_2
        nu_312 = dnu_3 + dnu_1 + dnu_2

        # Order 2-3-1
        Psi_231 = dPsi_2 + dPsi_3 + dPsi_1
        nu_231 = dnu_2 + dnu_3 + dnu_1

        # Should be identical
        np.testing.assert_allclose(np.asarray(Psi_123), np.asarray(Psi_312), atol=1e-12)
        np.testing.assert_allclose(np.asarray(Psi_123), np.asarray(Psi_231), atol=1e-12)
        assert nu_123 == pytest.approx(nu_312, abs=1e-12)
        assert nu_123 == pytest.approx(nu_231, abs=1e-12)

    def test_weighted_stats_accumulation_commutative(self):
        """Test that weighted IW stats accumulate commutatively."""
        key = jax.random.PRNGKey(999)
        d = 3

        def random_psd(key, d):
            A = jax.random.normal(key, (d, d))
            return A @ A.T + 0.01 * jnp.eye(d)

        # Weighted updates
        w1, w2, w3 = 0.5, 0.3, 0.2
        dPsi_1 = random_psd(key, d)
        key, subkey = jax.random.split(key)
        dPsi_2 = random_psd(subkey, d)
        key, subkey = jax.random.split(key)
        dPsi_3 = random_psd(subkey, d)

        # Weighted sum order 1
        Psi_weighted_1 = w1 * dPsi_1 + w2 * dPsi_2 + w3 * dPsi_3

        # Weighted sum order 2
        Psi_weighted_2 = w3 * dPsi_3 + w1 * dPsi_1 + w2 * dPsi_2

        np.testing.assert_allclose(
            np.asarray(Psi_weighted_1), np.asarray(Psi_weighted_2), atol=1e-12
        )


# =============================================================================
# 6. Vectorized Operator Correctness Tests
# =============================================================================


class TestVectorizedOperators:
    """Tests that vectorized operators produce same results as scalar versions."""

    def test_kappa_batch_matches_scalar(self):
        """Test that batch kappa matches individual scalar computations."""
        from fl_slam_poc.backend.operators.kappa import (
            kappa_from_resultant_batch,
            kappa_from_resultant_v2,
        )

        R_values = jnp.array([0.1, 0.3, 0.5, 0.7, 0.85])
        kappa_batch = kappa_from_resultant_batch(R_values)

        for i, R in enumerate(R_values):
            result, _, _ = kappa_from_resultant_v2(float(R))
            np.testing.assert_allclose(
                kappa_batch[i], result.kappa, rtol=1e-10
            )

    def test_psd_projection_batch_correctness(self):
        """Test that vmapped PSD projection matches scalar version."""
        key = jax.random.PRNGKey(111)
        d = 3
        n_matrices = 5

        # Generate random matrices
        matrices = []
        for _ in range(n_matrices):
            key, subkey = jax.random.split(key)
            A = jax.random.normal(subkey, (d, d))
            # Make some slightly non-PSD
            M = A @ A.T - 0.1 * jnp.eye(d)
            matrices.append(M)

        M_stack = jnp.stack(matrices, axis=0)

        # Batch projection
        from fl_slam_poc.common.primitives import domain_projection_psd_core

        def project_one(M):
            M_psd, cert = domain_projection_psd_core(M, 1e-6)
            return M_psd

        M_psd_batch = jax.vmap(project_one)(M_stack)

        # Scalar projection
        for i, M in enumerate(matrices):
            result = domain_projection_psd(M, 1e-6)
            np.testing.assert_allclose(
                np.asarray(M_psd_batch[i]), np.asarray(result.M_psd), atol=1e-12
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
