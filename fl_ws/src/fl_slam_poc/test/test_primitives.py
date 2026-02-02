"""
Tests for Geometric Compositional SLAM v2 primitives.

Verifies that primitives are branch-free and always execute.

Reference: docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md Section 3
"""

import pytest
from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common.primitives import (
    symmetrize,
    domain_projection_psd,
    spd_cholesky_solve_lifted,
    spd_cholesky_inverse_lifted,
    inv_mass,
    clamp,
    clamp_array,
    safe_normalize,
    sigmoid,
    softmax,
)


class TestSymmetrize:
    """Tests for Symmetrize primitive."""

    def test_symmetric_input_unchanged(self):
        """Symmetric input should have zero delta."""
        M = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        result = symmetrize(M)
        assert jnp.allclose(result.M_sym, M)
        assert result.sym_delta < 1e-12

    def test_asymmetric_input_symmetrized(self):
        """Asymmetric input should be symmetrized."""
        M = jnp.array([[1.0, 0.3], [0.7, 1.0]])
        result = symmetrize(M)
        # Check symmetry
        assert jnp.allclose(result.M_sym, result.M_sym.T)
        # Delta should be nonzero
        assert result.sym_delta > 0

    def test_always_returns_result(self):
        """Symmetrize always returns a result, never fails."""
        # Test with various inputs
        for M in [jnp.zeros((3, 3)), jnp.eye(3), jnp.ones((3, 3)) * 1e10]:
            result = symmetrize(M)
            assert result.M_sym is not None
            assert result.sym_delta >= 0


class TestDomainProjectionPSD:
    """Tests for DomainProjectionPSD primitive."""

    def test_psd_input_unchanged(self):
        """PSD input should have small projection delta."""
        M = jnp.eye(3)
        result = domain_projection_psd(M)
        assert jnp.allclose(result.M_psd, M, atol=1e-10)
        assert result.projection_delta < 1e-10

    def test_negative_eigenvalue_clamped(self):
        """Matrix with negative eigenvalue should be projected."""
        # Create matrix with negative eigenvalue
        M = jnp.array([[1.0, 0.0], [0.0, -0.5]])
        result = domain_projection_psd(M)
        # Check all eigenvalues are positive
        eigvals = jnp.linalg.eigvalsh(result.M_psd)
        assert jnp.all(eigvals >= 1e-12)
        # Projection delta should be nonzero
        assert result.projection_delta > 0

    def test_conditioning_computed(self):
        """Conditioning info should be computed."""
        M = jnp.eye(3)
        result = domain_projection_psd(M)
        assert result.conditioning.eig_min > 0
        assert result.conditioning.eig_max >= result.conditioning.eig_min
        assert result.conditioning.cond >= 1.0

    def test_always_returns_psd(self):
        """Output is always PSD regardless of input."""
        # Test with various problematic inputs
        inputs = [
            jnp.array([[1.0, 0.0], [0.0, -1.0]]),  # Negative eigenvalue
            jnp.zeros((3, 3)),  # All zeros
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # Asymmetric
        ]
        for M in inputs:
            result = domain_projection_psd(M)
            eigvals = jnp.linalg.eigvalsh(result.M_psd)
            # Allow small floating point tolerance (result should be >= eps_psd - tolerance)
            assert jnp.all(eigvals >= 1e-12 - 1e-15)


class TestSPDCholeskySolveLift:
    """Tests for SPDCholeskySolveLifted primitive."""

    def test_basic_solve(self):
        """Basic SPD solve should work."""
        L = jnp.eye(3)
        b = jnp.array([1.0, 2.0, 3.0])
        result = spd_cholesky_solve_lifted(L, b)
        # With identity matrix, x should equal b (approximately)
        assert jnp.allclose(result.x, b, atol=1e-8)

    def test_lift_always_applied(self):
        """Lift is always applied, even for well-conditioned matrices."""
        L = jnp.eye(3)
        b = jnp.array([1.0, 2.0, 3.0])
        result = spd_cholesky_solve_lifted(L, b, eps_lift=1e-6)
        # Lift strength should be positive
        assert result.lift_strength > 0

    def test_singular_matrix_lifted(self):
        """Singular matrix should still be solvable with lift."""
        L = jnp.array([[1.0, 0.0], [0.0, 0.0]])  # Singular
        b = jnp.array([1.0, 1.0])
        result = spd_cholesky_solve_lifted(L, b, eps_lift=1e-6)
        # Should return a valid result
        assert result.x is not None
        assert jnp.all(jnp.isfinite(result.x))


class TestInvMass:
    """Tests for InvMass primitive."""

    def test_positive_mass(self):
        """Positive mass should give reasonable inverse."""
        result = inv_mass(1.0)
        assert abs(result.inv_mass - 1.0) < 1e-10
        assert result.mass_epsilon_ratio < 1e-10

    def test_zero_mass_regularized(self):
        """Zero mass should be regularized, not fail."""
        result = inv_mass(0.0)
        assert jnp.isfinite(result.inv_mass)
        assert result.inv_mass > 0
        # Epsilon ratio should indicate strong regularization
        assert result.mass_epsilon_ratio > 0.5

    def test_negative_mass_regularized(self):
        """Negative mass should be regularized."""
        result = inv_mass(-1.0, eps_mass=1.0)
        assert jnp.isfinite(result.inv_mass)


class TestClamp:
    """Tests for Clamp primitive."""

    def test_in_range_unchanged(self):
        """Value in range should be unchanged."""
        result = clamp(0.5, 0.0, 1.0)
        assert result.value == 0.5
        assert result.clamp_delta == 0.0

    def test_below_range_clamped(self):
        """Value below range should be clamped."""
        result = clamp(-0.5, 0.0, 1.0)
        assert result.value == 0.0
        assert result.clamp_delta == 0.5

    def test_above_range_clamped(self):
        """Value above range should be clamped."""
        result = clamp(1.5, 0.0, 1.0)
        assert result.value == 1.0
        assert result.clamp_delta == 0.5

    def test_always_returns_valid(self):
        """Clamp always returns valid result."""
        for x in [-1e10, 0.0, 1e10, float('inf')]:
            result = clamp(x, 0.0, 1.0)
            assert 0.0 <= result.value <= 1.0


class TestClampArray:
    """Tests for clamp_array primitive."""

    def test_array_clamping(self):
        """Array values should be clamped element-wise."""
        x = jnp.array([-1.0, 0.5, 2.0])
        result, delta = clamp_array(x, 0.0, 1.0)
        assert jnp.allclose(result, jnp.array([0.0, 0.5, 1.0]))
        assert delta > 0  # Some clamping occurred


class TestSafeNormalize:
    """Tests for safe_normalize primitive."""

    def test_nonzero_vector(self):
        """Nonzero vector should be normalized."""
        v = jnp.array([3.0, 4.0, 0.0])
        result, eps_ratio = safe_normalize(v)
        assert jnp.allclose(jnp.linalg.norm(result), 1.0, atol=1e-6)
        assert eps_ratio < 1e-10

    def test_zero_vector_regularized(self):
        """Zero vector should be regularized, not fail."""
        v = jnp.array([0.0, 0.0, 0.0])
        result, eps_ratio = safe_normalize(v)
        assert jnp.all(jnp.isfinite(result))
        # Epsilon ratio should be high for zero input
        assert eps_ratio > 0.5


class TestSigmoid:
    """Tests for sigmoid primitive."""

    def test_sigmoid_range(self):
        """Sigmoid output should be in (0, 1)."""
        for x in [-10.0, 0.0, 10.0]:
            y = sigmoid(x)
            assert 0.0 < y < 1.0

    def test_sigmoid_monotonic(self):
        """Sigmoid should be monotonically increasing."""
        assert sigmoid(-1.0) < sigmoid(0.0) < sigmoid(1.0)


class TestSoftmax:
    """Tests for softmax primitive."""

    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1."""
        logits = jnp.array([1.0, 2.0, 3.0])
        probs = softmax(logits)
        assert jnp.allclose(jnp.sum(probs), 1.0)

    def test_softmax_positive(self):
        """All softmax outputs should be positive."""
        logits = jnp.array([-10.0, 0.0, 10.0])
        probs = softmax(logits)
        assert jnp.all(probs > 0)

    def test_temperature_effect(self):
        """Higher temperature should produce more uniform distribution."""
        logits = jnp.array([0.0, 1.0, 2.0])
        probs_low_temp = softmax(logits, tau=0.1)
        probs_high_temp = softmax(logits, tau=10.0)
        # High temp should be more uniform (max prob closer to 1/3)
        assert probs_high_temp[2] < probs_low_temp[2]
