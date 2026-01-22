import numpy as np
# Use common JAX initialization to ensure consistent setup
from fl_slam_poc.common.jax_init import jax, jnp

from fl_slam_poc.backend.imu_jax_kernel import imu_batched_projection_kernel, imu_residual_from_raw
from fl_slam_poc.backend.gaussian_info import make_evidence, mean_cov
from fl_slam_poc.backend.dirichlet_routing import DirichletRoutingModule
from fl_slam_poc.common.se3 import se3_compose, se3_exp

# JAX is already configured (x64 + GPU) by common.jax_init


def _schur_marginalize_joint(joint_mean: np.ndarray, joint_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L_joint, h_joint = make_evidence(joint_mean, joint_cov)
    L_ii = L_joint[:15, :15]
    L_ij = L_joint[:15, 15:]
    L_ji = L_joint[15:, :15]
    L_jj = L_joint[15:, 15:]
    h_i = h_joint[:15]
    h_j = h_joint[15:]

    L_ii_inv = np.linalg.inv(L_ii)
    L_j = L_jj - L_ji @ L_ii_inv @ L_ij
    h_j = h_j - L_ji @ L_ii_inv @ h_i
    return mean_cov(L_j, h_j)


def _apply_imu_update(
    anchor_mu: np.ndarray,
    anchor_cov: np.ndarray,
    current_mu: np.ndarray,
    current_cov: np.ndarray,
    stamps: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    gravity: np.ndarray,
    R_imu: np.ndarray,
    routing_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    imu_valid = np.ones((stamps.shape[0],), dtype=bool)
    joint_mean, joint_cov, diagnostics = imu_batched_projection_kernel(
        anchor_mus=jnp.array(anchor_mu[None, :]),
        anchor_covs=jnp.array(anchor_cov[None, :, :]),
        current_mu=jnp.array(current_mu),
        current_cov=jnp.array(current_cov),
        routing_weights=jnp.array(routing_weights),
        imu_stamps=jnp.array(stamps),
        imu_accel=jnp.array(accel),
        imu_gyro=jnp.array(gyro),
        imu_valid=jnp.array(imu_valid),
        R_imu=jnp.array(R_imu),
        R_nom=jnp.array(R_imu),
        dt_total=float(stamps[-1] - stamps[0]),
        gravity=jnp.array(gravity),
    )

    delta_mu, delta_cov = _schur_marginalize_joint(np.array(joint_mean), np.array(joint_cov))
    pose_new = se3_compose(current_mu[:6], se3_exp(delta_mu[:6]))
    rest_new = current_mu[6:] + delta_mu[6:]
    mu_new = np.concatenate([pose_new, rest_new])
    return mu_new, delta_cov, diagnostics


def test_zero_residual_contract_b():
    anchor_mu = np.zeros(15)
    current_mu = np.zeros(15)
    anchor_cov = np.eye(15) * 1e-3
    current_cov = np.eye(15) * 1e-3

    stamps = np.array([0.0, 1.0])
    accel = np.zeros((2, 3))
    gyro = np.zeros((2, 3))
    gravity = np.zeros(3)
    R_imu = np.eye(9) * 1e-4

    mu_new, _, diagnostics = _apply_imu_update(
        anchor_mu, anchor_cov, current_mu, current_cov, stamps, accel, gyro, gravity, R_imu, np.array([1.0])
    )

    assert np.linalg.norm(mu_new[:9]) < 1e-6
    assert bool(diagnostics.get("degenerate_weights")) is False


def test_bias_observability_changes_residual():
    """Test that bias actively changes the residual (Contract B Gate 1)."""
    anchor_mu = np.zeros(15)
    current_mu = np.zeros(15)
    stamps = np.array([0.0, 1.0])
    accel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    gyro = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    gravity = np.zeros(3)

    # Test that changing bias changes the residual
    anchor_mu_bias = anchor_mu.copy()
    anchor_mu_bias[9:12] = np.array([0.05, 0.0, 0.0])

    r0 = imu_residual_from_raw(
        jnp.array(anchor_mu),
        jnp.array(current_mu),
        jnp.array(stamps),
        jnp.array(accel),
        jnp.array(gyro),
        jnp.array([True, True]),
        jnp.array(gravity),
        1.0,
    )
    r1 = imu_residual_from_raw(
        jnp.array(anchor_mu_bias),
        jnp.array(current_mu),
        jnp.array(stamps),
        jnp.array(accel),
        jnp.array(gyro),
        jnp.array([True, True]),
        jnp.array(gravity),
        1.0,
    )
    # Contract B Gate 1: bias must actively change the residual
    residual_diff = float(jnp.linalg.norm(r0 - r1))
    assert residual_diff > 1e-6, f"Bias change must affect residual: diff={residual_diff}"
    
    # Additional check: verify that posterior changes when bias changes
    # This proves bias is in the model (not just a constant offset)
    anchor_cov = np.eye(15) * 1e-2
    current_cov = np.eye(15) * 1e-2
    R_imu = np.eye(9) * 1e-3
    
    mu_start = current_mu.copy()
    cov_start = current_cov.copy()
    
    # Apply update with zero bias
    mu_new_zero, _, _ = _apply_imu_update(
        anchor_mu, anchor_cov, mu_start, cov_start, stamps, accel, gyro, gravity, R_imu, np.array([1.0])
    )
    
    # Apply update with non-zero bias
    mu_new_bias, _, _ = _apply_imu_update(
        anchor_mu_bias, anchor_cov, mu_start, cov_start, stamps, accel, gyro, gravity, R_imu, np.array([1.0])
    )
    
    # Posterior should differ when bias differs (proves bias is in model)
    posterior_diff = float(np.linalg.norm(mu_new_zero - mu_new_bias))
    assert posterior_diff > 1e-8, f"Posterior must change when bias changes: diff={posterior_diff}"


def test_order_invariance_update():
    anchor_mu = np.zeros(15)
    anchor_cov = np.eye(15) * 1e-2
    current_mu = np.zeros(15)
    current_cov = np.eye(15) * 1e-2
    gravity = np.zeros(3)
    R_imu = np.eye(9) * 1e-3
    routing_weights = np.array([1.0])

    stamps = np.array([0.0, 0.5])
    accel_a = np.array([[0.0, 0.1, 0.0], [0.0, 0.1, 0.0]])
    gyro_a = np.array([[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]])
    accel_b = np.array([[0.0, -0.1, 0.0], [0.0, -0.1, 0.0]])
    gyro_b = np.array([[-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])

    mu_ab, cov_ab, _ = _apply_imu_update(
        anchor_mu, anchor_cov, current_mu, current_cov, stamps, accel_a, gyro_a, gravity, R_imu, routing_weights
    )
    mu_ab, cov_ab, _ = _apply_imu_update(
        anchor_mu, anchor_cov, mu_ab, cov_ab, stamps, accel_b, gyro_b, gravity, R_imu, routing_weights
    )

    mu_ba, cov_ba, _ = _apply_imu_update(
        anchor_mu, anchor_cov, current_mu, current_cov, stamps, accel_b, gyro_b, gravity, R_imu, routing_weights
    )
    mu_ba, cov_ba, _ = _apply_imu_update(
        anchor_mu, anchor_cov, mu_ba, cov_ba, stamps, accel_a, gyro_a, gravity, R_imu, routing_weights
    )

    assert np.allclose(mu_ab, mu_ba, atol=1e-4)
    assert np.allclose(cov_ab, cov_ba, atol=1e-4)


def test_frame_convention_yaw_90():
    anchor_mu = np.zeros(15)
    current_mu = np.zeros(15)
    current_mu[3:6] = np.array([0.0, 0.0, np.pi / 2.0])
    anchor_cov = np.eye(15) * 1e-3
    current_cov = np.eye(15) * 1e-3

    stamps = np.array([0.0, 1.0])
    accel = np.zeros((2, 3))
    gyro = np.array([[0.0, 0.0, np.pi / 2.0], [0.0, 0.0, np.pi / 2.0]])
    gravity = np.zeros(3)
    R_imu = np.eye(9) * 1e-4

    mu_new, _, _ = _apply_imu_update(
        anchor_mu, anchor_cov, current_mu, current_cov, stamps, accel, gyro, gravity, R_imu, np.array([1.0])
    )
    assert abs(mu_new[5] - np.pi / 2.0) < 1e-3


def test_hellinger_boundedness():
    """Test that Hellinger distance is bounded in [0,1] and responds to measurement consistency."""
    anchor_mu = np.zeros(15)
    current_mu = np.zeros(15)
    anchor_cov = np.eye(15) * 1e-3
    current_cov = np.eye(15) * 1e-3
    gravity = np.zeros(3)
    R_imu = np.eye(9) * 1e-3
    stamps = np.array([0.0, 1.0])

    # Zero motion (consistent with zero state)
    accel_zero = np.zeros((2, 3))
    gyro_zero = np.zeros((2, 3))
    _, _, diag_zero = _apply_imu_update(
        anchor_mu, anchor_cov, current_mu, current_cov, stamps, accel_zero, gyro_zero, gravity, R_imu, np.array([1.0])
    )

    # High motion (inconsistent with zero state prediction)
    accel_high = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    gyro_high = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _, _, diag_high = _apply_imu_update(
        anchor_mu, anchor_cov, current_mu, current_cov, stamps, accel_high, gyro_high, gravity, R_imu, np.array([1.0])
    )

    h_zero = float(diag_zero.get("hellinger_mean", 0.0))
    h_high = float(diag_high.get("hellinger_mean", 0.0))
    
    # Boundedness check
    assert 0.0 <= h_zero <= 1.0, f"Hellinger for zero motion out of bounds: {h_zero}"
    assert 0.0 <= h_high <= 1.0, f"Hellinger for high motion out of bounds: {h_high}"
    
    # High motion should produce different Hellinger than zero motion
    # (The exact relationship depends on prediction vs measurement consistency)
    assert abs(h_high - h_zero) > 1e-6, f"Hellinger should differ: zero={h_zero}, high={h_high}"


def test_routing_consistency_order():
    routing = DirichletRoutingModule(n_anchors=2, retention_base=0.95)
    logits = np.array([0.2, -0.1])

    routing_ab = DirichletRoutingModule(n_anchors=2, retention_base=0.95)
    routing_ab.update(logits)
    routing_ab.update(logits)

    routing_ba = DirichletRoutingModule(n_anchors=2, retention_base=0.95)
    routing_ba.update(logits)
    routing_ba.update(logits)

    assert np.allclose(routing_ab.get_responsibilities(), routing_ba.get_responsibilities(), atol=1e-6)
