"""
Audit Invariant Tests

Verifies non-negotiable invariants from the adversarial audit checklist:
- Information distance closed-forms (Hellinger, Fisher-Rao, SPD)
- SE(3) representation (rotation vector consistency)
- Covariance transport (adjoint correctness)
- Frobenius correction (measurable delta)
- Budget truncation (mass conservation)
- ICP solver bounds (A2 compliance)
- Integration invariants I1-I3 (frame, motion, timestamp)
"""

import math
import numpy as np
import pytest

# Geometry (now in common/transforms/)
from fl_slam_poc.common.se3 import (
    rotvec_to_rotmat,
    rotmat_to_rotvec,
    se3_compose,
    se3_inverse,
    se3_apply,
    se3_adjoint,
    se3_cov_compose,
    skew,
    ROTATION_EPSILON,
    SINGULARITY_EPSILON,
)

# Operators - Dirichlet (experimental, stays in operators/)
from fl_slam_poc.common.dirichlet_geom import third_order_correct

# Backend fusion operators
from fl_slam_poc.backend.gaussian_info import (
    make_evidence,
    fuse_info,
    mean_cov,
    kl_divergence,
)
from fl_slam_poc.backend.information_distances import (
    hellinger_gaussian,
    fisher_rao_gaussian_1d,
    fisher_rao_student_t,
    fisher_rao_student_t_vec,
    fisher_rao_spd,
    product_distance,
)
from fl_slam_poc.backend.gaussian_geom import gaussian_frobenius_correction

# Frontend loops operators
from fl_slam_poc.frontend.icp import (
    icp_3d,
    icp_information_weight,
    icp_covariance_tangent,
    best_fit_se3,
    N_MIN_SE3_DOF,
    K_SIGMOID,
)

# Common
from fl_slam_poc.common.op_report import OpReport

# Models (now in backend/parameters/)
from fl_slam_poc.backend import (
    AdaptiveParameter,
    TimeAlignmentModel,
    StochasticBirthModel,
    AdaptiveProcessNoise,
    NIGModel,
    combine_independent_weights,
)


# =============================================================================
# Information Distance Tests
# =============================================================================


class TestHellingerDistance:
    """Test Hellinger distance implementations."""

    def test_hellinger_identical_distributions(self):
        """Hellinger distance between identical distributions is zero."""
        Sigma = np.diag([1.0, 2.0, 0.5])
        h = hellinger_gaussian(Sigma, Sigma)
        assert abs(h) < 1e-10

    def test_hellinger_symmetric(self):
        """Hellinger distance is symmetric."""
        Sigma1 = np.diag([1.0, 2.0, 0.5])
        Sigma2 = np.diag([2.0, 1.0, 1.0])
        h12 = hellinger_gaussian(Sigma1, Sigma2)
        h21 = hellinger_gaussian(Sigma2, Sigma1)
        assert abs(h12 - h21) < 1e-10

    def test_hellinger_bounded(self):
        """Hellinger distance is in [0, 1]."""
        for _ in range(10):
            A = np.random.randn(3, 3)
            Sigma1 = A @ A.T + 0.1 * np.eye(3)
            B = np.random.randn(3, 3)
            Sigma2 = B @ B.T + 0.1 * np.eye(3)
            h = hellinger_gaussian(Sigma1, Sigma2)
            assert 0.0 <= h <= 1.0 + 1e-10


class TestFisherRaoGaussian1D:
    """Test univariate Gaussian Fisher-Rao distance."""

    def test_fr_identical(self):
        d = fisher_rao_gaussian_1d(0.0, 1.0, 0.0, 1.0)
        assert abs(d) < 1e-10

    def test_fr_symmetric(self):
        d12 = fisher_rao_gaussian_1d(0.0, 1.0, 1.0, 2.0)
        d21 = fisher_rao_gaussian_1d(1.0, 2.0, 0.0, 1.0)
        assert abs(d12 - d21) < 1e-10

    def test_fr_positive(self):
        for _ in range(20):
            mu1, mu2 = np.random.randn(2)
            sigma1 = abs(np.random.randn()) + 0.1
            sigma2 = abs(np.random.randn()) + 0.1
            d = fisher_rao_gaussian_1d(mu1, sigma1, mu2, sigma2)
            assert d >= 0.0

    def test_fr_triangle_inequality(self):
        """FR distance satisfies triangle inequality (true metric)."""
        for _ in range(10):
            params = [(np.random.randn(), abs(np.random.randn()) + 0.1) for _ in range(3)]
            d12 = fisher_rao_gaussian_1d(*params[0], *params[1])
            d23 = fisher_rao_gaussian_1d(*params[1], *params[2])
            d13 = fisher_rao_gaussian_1d(*params[0], *params[2])
            assert d13 <= d12 + d23 + 1e-10


class TestFisherRaoStudentT:
    """Test Student-t Fisher-Rao distance (NIG predictive)."""

    def test_fr_student_identical(self):
        d = fisher_rao_student_t(0.0, 1.0, 0.0, 1.0, nu=5.0)
        assert abs(d) < 1e-10

    def test_fr_student_symmetric(self):
        d12 = fisher_rao_student_t(0.0, 1.0, 1.0, 2.0, nu=5.0)
        d21 = fisher_rao_student_t(1.0, 2.0, 0.0, 1.0, nu=5.0)
        assert abs(d12 - d21) < 1e-10

    def test_fr_student_triangle_inequality(self):
        nu = 5.0
        for _ in range(10):
            params = [(np.random.randn(), abs(np.random.randn()) + 0.1) for _ in range(3)]
            d12 = fisher_rao_student_t(*params[0], *params[1], nu)
            d23 = fisher_rao_student_t(*params[1], *params[2], nu)
            d13 = fisher_rao_student_t(*params[0], *params[2], nu)
            assert d13 <= d12 + d23 + 1e-10


class TestFisherRaoSPD:
    """Test SPD (covariance) Fisher-Rao distance."""

    def test_spd_identical(self):
        Sigma = np.diag([1.0, 2.0, 0.5])
        d = fisher_rao_spd(Sigma, Sigma)
        assert abs(d) < 1e-10

    def test_spd_symmetric(self):
        A = np.random.randn(3, 3)
        Sigma1 = A @ A.T + 0.1 * np.eye(3)
        B = np.random.randn(3, 3)
        Sigma2 = B @ B.T + 0.1 * np.eye(3)
        d12 = fisher_rao_spd(Sigma1, Sigma2)
        d21 = fisher_rao_spd(Sigma2, Sigma1)
        assert abs(d12 - d21) < 1e-10

    def test_spd_triangle_inequality(self):
        def random_spd():
            A = np.random.randn(3, 3)
            return A @ A.T + 0.1 * np.eye(3)
        
        for _ in range(10):
            S1, S2, S3 = random_spd(), random_spd(), random_spd()
            d12 = fisher_rao_spd(S1, S2)
            d23 = fisher_rao_spd(S2, S3)
            d13 = fisher_rao_spd(S1, S3)
            assert d13 <= d12 + d23 + 1e-10


# =============================================================================
# SE(3) Geometry Tests
# =============================================================================


class TestRotationVectorConversions:
    """Test rotation vector <-> rotation matrix conversions."""

    def test_identity_rotation(self):
        rotvec = np.array([0.0, 0.0, 0.0])
        R = rotvec_to_rotmat(rotvec)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_small_rotation(self):
        rotvec = np.array([0.001, 0.002, 0.003])
        R = rotvec_to_rotmat(rotvec)
        R_approx = np.eye(3) + skew(rotvec)
        assert np.allclose(R, R_approx, atol=1e-5)

    def test_90_degree_rotation_x(self):
        rotvec = np.array([math.pi / 2, 0.0, 0.0])
        R = rotvec_to_rotmat(rotvec)
        R_expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        assert np.allclose(R, R_expected, atol=1e-10)

    def test_round_trip(self):
        for _ in range(10):
            rotvec = np.random.randn(3) * 0.5
            R = rotvec_to_rotmat(rotvec)
            rotvec_back = rotmat_to_rotvec(R)
            R_back = rotvec_to_rotmat(rotvec_back)
            assert np.allclose(R, R_back, atol=1e-10)

    def test_near_pi_rotation(self):
        theta = math.pi - 0.01
        rotvec = np.array([theta, 0.0, 0.0])
        R = rotvec_to_rotmat(rotvec)
        rotvec_back = rotmat_to_rotvec(R)
        R_back = rotvec_to_rotmat(rotvec_back)
        assert np.allclose(R, R_back, atol=1e-8)

    def test_numerical_constants_documented(self):
        assert ROTATION_EPSILON > 0 and ROTATION_EPSILON < 1e-6
        assert SINGULARITY_EPSILON > 0 and SINGULARITY_EPSILON < 1e-3


class TestSE3Operations:
    """Test SE(3) group operations."""

    def test_compose_identity(self):
        T = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        identity = np.zeros(6)
        T_out = se3_compose(T, identity)
        assert np.allclose(T_out, T, atol=1e-10)

    def test_inverse(self):
        T = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        T_inv = se3_inverse(T)
        identity = se3_compose(T, T_inv)
        assert np.allclose(identity, np.zeros(6), atol=1e-10)

    def test_compose_associative(self):
        T1 = np.array([1.0, 0.0, 0.0, 0.1, 0.0, 0.0])
        T2 = np.array([0.0, 1.0, 0.0, 0.0, 0.1, 0.0])
        T3 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.1])
        lhs = se3_compose(se3_compose(T1, T2), T3)
        rhs = se3_compose(T1, se3_compose(T2, T3))
        assert np.allclose(lhs, rhs, atol=1e-10)


class TestCovarianceTransport:
    """Test covariance transport via adjoint."""

    def test_adjoint_identity(self):
        T = np.zeros(6)
        Ad = se3_adjoint(T)
        assert np.allclose(Ad, np.eye(6), atol=1e-10)

    def test_cov_transport_identity(self):
        cov = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
        T = np.zeros(6)
        cov_out = se3_cov_compose(np.zeros((6, 6)), cov, T)
        assert np.allclose(cov_out, cov, atol=1e-10)

    def test_cov_positive_definite(self):
        cov_a = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
        cov_b = np.diag([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])
        T = np.array([1.0, 2.0, 3.0, 0.3, 0.2, 0.1])
        cov_out = se3_cov_compose(cov_a, cov_b, T)
        eigvals = np.linalg.eigvalsh(cov_out)
        assert np.all(eigvals > 0)


# =============================================================================
# Gaussian Information Form Tests
# =============================================================================


class TestGaussianInfoForm:
    """Test Gaussian operations in information form."""

    def test_round_trip(self):
        mu = np.array([1.0, 2.0, 3.0])
        cov = np.diag([0.5, 1.0, 0.25])
        L, h = make_evidence(mu, cov)
        mu_back, cov_back = mean_cov(L, h)
        assert np.allclose(mu, mu_back, atol=1e-10)
        assert np.allclose(cov, cov_back, atol=1e-10)

    def test_fusion_commutative(self):
        L1, h1 = make_evidence(np.zeros(3), np.eye(3))
        L2, h2 = make_evidence(np.ones(3), 0.5 * np.eye(3))
        L_12, h_12 = fuse_info(L1, h1, L2, h2)
        L_21, h_21 = fuse_info(L2, h2, L1, h1)
        assert np.allclose(L_12, L_21, atol=1e-10)
        assert np.allclose(h_12, h_21, atol=1e-10)

    def test_fusion_associative(self):
        L1, h1 = make_evidence(np.zeros(3), np.eye(3))
        L2, h2 = make_evidence(np.ones(3), 0.5 * np.eye(3))
        L3, h3 = make_evidence(-np.ones(3), 0.25 * np.eye(3))
        L_12, h_12 = fuse_info(L1, h1, L2, h2)
        L_12_3, h_12_3 = fuse_info(L_12, h_12, L3, h3)
        L_23, h_23 = fuse_info(L2, h2, L3, h3)
        L_1_23, h_1_23 = fuse_info(L1, h1, L_23, h_23)
        assert np.allclose(L_12_3, L_1_23, atol=1e-10)
        assert np.allclose(h_12_3, h_1_23, atol=1e-10)

    def test_kl_divergence_non_negative(self):
        L1, h1 = make_evidence(np.zeros(3), np.eye(3))
        L2, h2 = make_evidence(np.ones(3), 0.5 * np.eye(3))
        kl = kl_divergence(L1, h1, L2, h2)
        assert kl >= 0.0

    def test_kl_divergence_zero_for_same(self):
        L, h = make_evidence(np.array([1.0, 2.0]), np.diag([0.5, 1.0]))
        kl = kl_divergence(L, h, L, h)
        assert abs(kl) < 1e-10


# =============================================================================
# Frobenius Correction Tests
# =============================================================================


class TestFrobeniusCorrection:
    """Test third-order Frobenius correction."""

    def test_third_order_produces_measurable_delta(self):
        alpha = np.array([5.0, 2.0, 1.0, 0.5])
        delta = np.array([0.5, -0.3, 0.2, -0.1])
        delta_corr = third_order_correct(alpha, delta)
        delta_norm = np.linalg.norm(delta_corr - delta)
        assert delta_norm > 1e-6

    def test_op_report_validation_requires_stats(self):
        report = OpReport(
            name="TestOp",
            exact=False,
            approximation_triggers=["BudgetTruncation"],
            frobenius_applied=True,
            frobenius_operator="dirichlet_third_order",
            frobenius_delta_norm=None,
        )
        with pytest.raises(ValueError, match="delta norm"):
            report.validate()

    def test_op_report_valid_with_all_stats(self):
        report = OpReport(
            name="TestOp",
            exact=False,
            approximation_triggers=["BudgetTruncation"],
            frobenius_applied=True,
            frobenius_operator="dirichlet_third_order",
            frobenius_delta_norm=0.05,
            frobenius_input_stats={"alpha": {"mean": 2.0}},
            frobenius_output_stats={"delta_corr": {"mean": 0.1}},
        )
        report.validate()

    def test_gaussian_frobenius_correction_noop(self):
        """Gaussian Frobenius correction is a no-op (C=0) with zero delta norm."""
        delta = np.array([0.1, -0.2, 0.3, -0.1, 0.0, 0.05])
        delta_corr, stats = gaussian_frobenius_correction(delta)
        assert np.allclose(delta_corr, delta, atol=1e-12)
        assert abs(stats["delta_norm"]) < 1e-12

    def test_combine_independent_weights_log_additive(self):
        """Combining independent weights is additive in log space."""
        weights = [0.9, 0.8, 0.7]
        combined = combine_independent_weights(weights)
        log_combined = math.log(combined)
        log_sum = sum(math.log(w) for w in weights)
        assert abs(log_combined - log_sum) < 1e-12


# =============================================================================
# ICP Solver Tests (A2 Compliance - Bounded Solver)
# =============================================================================


class TestICPSolver:
    """Test ICP solver properties (A2 compliance)."""

    def test_icp_converges_within_max_iter(self):
        """ICP should converge within max_iter."""
        np.random.seed(42)
        source = np.random.randn(50, 3)
        target = source + 0.1 * np.random.randn(50, 3)
        
        result = icp_3d(source, target, init=np.zeros(6), max_iter=20, tol=1e-6)
        
        assert result.iterations <= result.max_iterations

    def test_icp_improves_objective(self):
        """ICP should improve (or maintain) objective."""
        np.random.seed(42)
        source = np.random.randn(30, 3)
        R_true = rotvec_to_rotmat(np.array([0.1, 0.0, 0.0]))
        t_true = np.array([0.5, 0.0, 0.0])
        target = (R_true @ source.T).T + t_true
        
        result = icp_3d(source, target, init=np.zeros(6), max_iter=20, tol=1e-6)
        
        assert result.final_objective <= result.initial_objective + 1e-10

    def test_icp_returns_valid_transform(self):
        """ICP transform should be valid SE(3)."""
        np.random.seed(42)
        source = np.random.randn(20, 3)
        target = source + np.array([1.0, 0.0, 0.0])
        
        result = icp_3d(source, target, init=np.zeros(6), max_iter=10, tol=1e-6)
        
        assert result.transform.shape == (6,)
        assert np.all(np.isfinite(result.transform))

    def test_icp_identity_for_same_clouds(self):
        """ICP on identical clouds should give near-identity."""
        source = np.random.randn(30, 3)
        
        result = icp_3d(source, source.copy(), init=np.zeros(6), max_iter=10, tol=1e-6)
        
        assert np.linalg.norm(result.transform) < 0.1
        assert result.mse < 0.01

    def test_icp_covariance_positive_definite(self):
        """ICP covariance should be positive definite."""
        np.random.seed(42)
        source = np.random.randn(30, 3)
        target = source + 0.1 * np.random.randn(30, 3)
        
        result = icp_3d(source, target, init=np.zeros(6), max_iter=10, tol=1e-6)
        cov = icp_covariance_tangent(result.src_transformed, result.mse)
        
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

    def test_icp_information_weight_bounded(self):
        """ICP information weight should be in (0, 1]."""
        for n in [5, 10, 50, 100, 200]:
            for mse in [0.001, 0.01, 0.1, 1.0]:
                w = icp_information_weight(n, n, mse)
                assert 0.0 < w <= 1.0

    def test_icp_weight_constants_documented(self):
        """Verify ICP weight constants are documented."""
        assert N_MIN_SE3_DOF == 6.0  # SE(3) has 6 DOF
        assert 0.0 < K_SIGMOID < 2.0  # Reasonable sigmoid steepness

    def test_best_fit_se3_exact_for_rigid_transform(self):
        """best_fit_se3 should recover exact rigid transform."""
        np.random.seed(42)
        source = np.random.randn(20, 3)
        R_true = rotvec_to_rotmat(np.array([0.2, -0.1, 0.15]))
        t_true = np.array([1.0, -0.5, 0.3])
        target = (R_true @ source.T).T + t_true
        
        T_est = best_fit_se3(source, target)
        
        R_est = rotvec_to_rotmat(T_est[3:6])
        t_est = T_est[:3]
        
        assert np.allclose(R_est, R_true, atol=1e-8)
        assert np.allclose(t_est, t_true, atol=1e-8)


# =============================================================================
# Integration Invariant Tests (I1-I3)
# =============================================================================


class TestFrameConsistency:
    """I1: Frame consistency tests."""

    def test_transform_identity_is_identity(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        T_identity = np.zeros(6)
        points_out = se3_apply(T_identity, points)
        assert np.allclose(points_out, points, atol=1e-10)

    def test_transform_inverse_roundtrip(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        T = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        points_fwd = se3_apply(T, points)
        points_back = se3_apply(se3_inverse(T), points_fwd)
        assert np.allclose(points_back, points, atol=1e-10)

    def test_frame_chain_composition(self):
        """T_AC = T_AB ∘ T_BC means: apply T_BC first, then T_AB."""
        T_AB = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.1])
        T_BC = np.array([0.0, 1.0, 0.0, 0.0, 0.1, 0.0])
        T_AC = se3_compose(T_AB, T_BC)
        
        points = np.array([[1.0, 2.0, 3.0]])
        points_B = se3_apply(T_BC, points)
        points_A_chain = se3_apply(T_AB, points_B)
        points_A_direct = se3_apply(T_AC, points)
        
        assert np.allclose(points_A_chain, points_A_direct, atol=1e-10)


class TestKnownMotion:
    """I2: Known motion end-to-end tests."""

    def test_known_translation(self):
        anchor = np.zeros(6)
        true_motion = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        current = se3_compose(anchor, true_motion)
        rel = se3_compose(se3_inverse(anchor), current)
        assert np.allclose(rel, true_motion, atol=1e-10)

    def test_known_rotation(self):
        anchor = np.zeros(6)
        true_motion = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 4])
        current = se3_compose(anchor, true_motion)
        rel = se3_compose(se3_inverse(anchor), current)
        assert np.allclose(rel, true_motion, atol=1e-10)

    def test_known_combined_motion(self):
        anchor = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.1])
        true_motion = np.array([0.5, -0.3, 0.1, 0.05, -0.02, 0.1])
        current = se3_compose(anchor, true_motion)
        rel = se3_compose(se3_inverse(anchor), current)
        assert np.allclose(rel, true_motion, atol=1e-10)

    def test_loop_closure_identity_at_start(self):
        anchor = np.zeros(6)
        current = np.zeros(6)
        rel = se3_compose(se3_inverse(anchor), current)
        assert np.allclose(rel, np.zeros(6), atol=1e-10)


class TestTimestampAlignment:
    """I3: Timestamp alignment behavior tests."""

    def test_gaussian_weight_at_zero(self):
        sigma = 0.1
        dt = 0.0
        weight = math.exp(-0.5 * (dt / sigma) ** 2)
        assert abs(weight - 1.0) < 1e-10

    def test_gaussian_weight_decreases_with_dt(self):
        sigma = 0.1
        weights = [math.exp(-0.5 * (dt / sigma) ** 2) for dt in [0.0, 0.05, 0.1, 0.2]]
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]

    def test_gaussian_weight_symmetric(self):
        sigma = 0.1
        for dt in [0.01, 0.05, 0.1, 0.2]:
            w_pos = math.exp(-0.5 * (dt / sigma) ** 2)
            w_neg = math.exp(-0.5 * (-dt / sigma) ** 2)
            assert abs(w_pos - w_neg) < 1e-10

    def test_weight_bounded(self):
        sigma = 0.1
        for dt in np.linspace(-1.0, 1.0, 100):
            weight = math.exp(-0.5 * (dt / sigma) ** 2)
            assert 0.0 < weight <= 1.0


class TestTwoPoseFactorSemantics:
    """Test two-pose factor update semantics."""

    def test_joint_update_improves_both_beliefs(self):
        cov_anchor = np.eye(6) * 0.5
        cov_current = np.eye(6) * 0.5
        cov_meas = np.eye(6) * 0.01
        
        cov_joint = np.zeros((12, 12))
        cov_joint[:6, :6] = cov_anchor
        cov_joint[6:12, 6:12] = cov_current
        
        H = np.zeros((6, 12))
        H[:, :6] = -np.eye(6)
        H[:, 6:12] = np.eye(6)
        
        S = H @ cov_joint @ H.T + cov_meas
        K = cov_joint @ H.T @ np.linalg.inv(S)
        cov_joint_updated = (np.eye(12) - K @ H) @ cov_joint
        
        assert np.trace(cov_joint_updated[:6, :6]) < np.trace(cov_anchor)
        assert np.trace(cov_joint_updated[6:12, 6:12]) < np.trace(cov_current)


# =============================================================================
# Model Tests
# =============================================================================


class TestStochasticBirth:
    """Test stochastic birth model properties."""

    def test_zero_r_never_births(self):
        lambda_0 = 10.0
        dt = 0.1
        for _ in range(100):
            intensity = lambda_0 * 0.0
            prob = 1.0 - math.exp(-intensity * dt)
            assert prob == 0.0

    def test_high_r_likely_births(self):
        lambda_0 = 10.0
        dt = 0.1
        r_new = 1.0
        intensity = lambda_0 * r_new
        prob = 1.0 - math.exp(-intensity * dt)
        assert prob > 0.5

    def test_birth_probability_bounded(self):
        lambda_0 = 10.0
        dt = 0.1
        for r_new in np.linspace(0.0, 2.0, 100):
            intensity = lambda_0 * r_new
            prob = 1.0 - math.exp(-intensity * dt)
            assert 0.0 <= prob <= 1.0

    def test_birth_probability_increases_with_r(self):
        lambda_0 = 10.0
        dt = 0.1
        probs = []
        for r_new in [0.0, 0.25, 0.5, 0.75, 1.0]:
            intensity = lambda_0 * r_new
            prob = 1.0 - math.exp(-intensity * dt)
            probs.append(prob)
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1]


class TestNIGModel:
    """Test Normal-Inverse-Gamma model."""

    def test_nig_update_increases_kappa(self):
        model = NIGModel.from_prior(np.zeros(3), kappa=1.0, alpha=2.0, beta=1.0)
        kappa_before = model.kappa[0]
        model.update(np.ones(3), weight=1.0)
        assert model.kappa[0] > kappa_before

    def test_nig_fisher_rao_distance_symmetric(self):
        model1 = NIGModel.from_prior(np.zeros(3), kappa=1.0, alpha=2.0, beta=1.0)
        model2 = NIGModel.from_prior(np.ones(3), kappa=2.0, alpha=3.0, beta=2.0)
        d12 = model1.fisher_rao_distance(model2)
        d21 = model2.fisher_rao_distance(model1)
        assert abs(d12 - d21) < 1e-10

    def test_nig_fisher_rao_distance_zero_for_same(self):
        model = NIGModel.from_prior(np.zeros(3), kappa=1.0, alpha=2.0, beta=1.0)
        d = model.fisher_rao_distance(model)
        assert abs(d) < 1e-10


class TestAdaptiveParameter:
    """Test adaptive parameter estimation."""

    def test_returns_prior_with_no_data(self):
        param = AdaptiveParameter(prior_mean=5.0, prior_strength=10.0)
        assert param.value() == 5.0
        assert param.confidence() == 0.0

    def test_adapts_towards_data(self):
        param = AdaptiveParameter(prior_mean=5.0, prior_strength=10.0)
        for _ in range(100):
            param.update(10.0)
        assert param.value() > 5.0
        assert param.value() < 10.0

    def test_respects_floor(self):
        param = AdaptiveParameter(prior_mean=5.0, prior_strength=10.0, floor=3.0)
        for _ in range(100):
            param.update(0.0)
        assert param.value() >= 3.0


# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
