"""
IMU Batched Projection Kernel.

Implements the core computation for Hellinger-Dirichlet IMU integration:
1. Batched sigma-support sampling across all anchors
2. IMU residual computation (exact Forster et al. model)
3. Hellinger-tilted weight computation
4. Global moment matching (e-projection)

Key implementation notes:
- Uses manifold retraction for pose components (SE(3) ⊕), not raw vector addition.
- Uses spherical-radial cubature weights (non-negative sigma-support weights),
  because sigma points are later treated as explicit mixture support.

Reference: Forster et al. (2017), Hellinger hierarchical construction
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cholesky, solve_triangular
from typing import Tuple

# Configure JAX for GPU and enable x64 precision
# Must be set before any JAX operations
import os

# Force GPU-only execution
os.environ["JAX_PLATFORMS"] = "gpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Enable x64 for numerical stability (required for precision)
jax.config.update("jax_enable_x64", True)

# Hard require GPU; raise if unavailable
# Check platform (not device_kind) since device_kind is vendor-specific string
if not any(d.platform == "gpu" for d in jax.devices()):
    available_devices = [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    raise RuntimeError(
        f"JAX GPU backend is required but not available.\n"
        f"Available JAX devices: {available_devices}\n"
        f"To fix: Ensure CUDA is installed and JAX can detect GPU devices."
    )

from fl_slam_poc.backend.lie_jax import so3_exp, so3_log, se3_plus, se3_minus


# =============================================================================
# Numerical Constants
# =============================================================================

# Covariance regularization for Cholesky stability
COV_REGULARIZATION = 1e-8

# Minimum weight threshold for mixture components
MIN_MIXTURE_WEIGHT = 1e-15

# Hellinger tilt weight (from Hellinger hierarchical construction)
HELLINGER_TILT_WEIGHT = 2.0  # exp(-γ * D_H²) with γ = 2

def _apply_delta_state(xbar: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """
    Apply a tangent-space delta to a 15D state.

    Layout:
      [p(3), rotvec(3), v(3), b_g(3), b_a(3)]

    Pose uses SE(3) right-composition retraction; remaining terms are Euclidean.
    """
    pose = se3_plus(xbar[:6], delta[:6])
    rest = xbar[6:] + delta[6:]
    return jnp.concatenate([pose, rest], axis=0)


# =============================================================================
# IMU Residual Model (Contract B: raw IMU integration)
# =============================================================================

def _integrate_raw_imu(
    imu_stamps: jnp.ndarray,
    imu_accel: jnp.ndarray,
    imu_gyro: jnp.ndarray,
    imu_valid: jnp.ndarray,
    bias_g: jnp.ndarray,
    bias_a: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Integrate raw IMU measurements into (delta_p, delta_v, delta_rotvec).
    """
    def step(carry, inputs):
        delta_R, delta_v, delta_p, t_prev, prev_valid = carry
        t, a, g, valid = inputs

        dt = jnp.where(prev_valid & valid, t - t_prev, 0.0)
        dt = jnp.maximum(dt, 0.0)

        omega = (g - bias_g) * dt
        delta_R = delta_R @ so3_exp(omega)
        accel_corr = a - bias_a
        accel_rot = delta_R @ accel_corr
        delta_v = delta_v + accel_rot * dt
        delta_p = delta_p + delta_v * dt + 0.5 * accel_rot * dt * dt

        t_prev = jnp.where(valid, t, t_prev)
        prev_valid = valid
        return (delta_R, delta_v, delta_p, t_prev, prev_valid), None

    t0 = imu_stamps[0]
    carry0 = (jnp.eye(3, dtype=imu_stamps.dtype), jnp.zeros(3), jnp.zeros(3), t0, imu_valid[0])
    inputs = (imu_stamps[1:], imu_accel[1:], imu_gyro[1:], imu_valid[1:])
    carry_out, _ = lax.scan(step, carry0, inputs)
    delta_R, delta_v, delta_p, _, _ = carry_out
    delta_rotvec = so3_log(delta_R)
    return delta_p, delta_v, delta_rotvec


def imu_residual_from_raw(
    xi_anchor: jnp.ndarray,
    xi_current: jnp.ndarray,
    imu_stamps: jnp.ndarray,
    imu_accel: jnp.ndarray,
    imu_gyro: jnp.ndarray,
    imu_valid: jnp.ndarray,
    gravity: jnp.ndarray,
    dt_total: float,
) -> jnp.ndarray:
    """
    Compute IMU residual from raw IMU segment (Contract B).
    """
    p_i = xi_anchor[:3]
    rotvec_i = xi_anchor[3:6]
    v_i = xi_anchor[6:9]
    b_g_i = xi_anchor[9:12]
    b_a_i = xi_anchor[12:15]

    p_j = xi_current[:3]
    rotvec_j = xi_current[3:6]
    v_j = xi_current[6:9]

    delta_p_meas, delta_v_meas, delta_rotvec_meas = _integrate_raw_imu(
        imu_stamps, imu_accel, imu_gyro, imu_valid, b_g_i, b_a_i
    )

    R_i = so3_exp(rotvec_i)
    R_j = so3_exp(rotvec_j)

    delta_p_pred = R_i.T @ (p_j - p_i - v_i * dt_total - 0.5 * gravity * dt_total**2)
    delta_v_pred = R_i.T @ (v_j - v_i - gravity * dt_total)
    delta_R_pred = R_i.T @ R_j
    delta_rotvec_pred = so3_log(delta_R_pred)

    r_p = delta_p_meas - delta_p_pred
    r_v = delta_v_meas - delta_v_pred
    r_omega = delta_rotvec_meas - delta_rotvec_pred

    return jnp.concatenate([r_p, r_v, r_omega], axis=0)


# =============================================================================
# Hellinger Distance
# =============================================================================

def hellinger_squared_gaussian(
    mu1: jnp.ndarray,
    cov1: jnp.ndarray,
    mu2: jnp.ndarray,
    cov2: jnp.ndarray,
) -> jnp.ndarray:
    """
    Squared Hellinger distance H²(N₁, N₂).

    H²(p, q) = 1 - BC where BC = ∫√(p q) dx

    For Gaussians: BC = |Σ₁|^{1/4} |Σ₂|^{1/4} |Σ_avg|^{-1/2} exp(-⅛ Δμᵀ Σ_avg⁻¹ Δμ)

    Args:
        mu1, cov1: First Gaussian parameters
        mu2, cov2: Second Gaussian parameters

    Returns:
        H²: Squared Hellinger distance in [0, 1]
    """
    cov_avg = 0.5 * (cov1 + cov2)

    # Regularize for numerical stability
    n = cov1.shape[0]
    I = jnp.eye(n, dtype=cov1.dtype)
    cov_avg_reg = cov_avg + I * COV_REGULARIZATION
    cov1_reg = cov1 + I * COV_REGULARIZATION
    cov2_reg = cov2 + I * COV_REGULARIZATION

    # Cholesky factorizations
    L_avg = cholesky(cov_avg_reg, lower=True)
    L1 = cholesky(cov1_reg, lower=True)
    L2 = cholesky(cov2_reg, lower=True)

    # Log determinants
    logdet_avg = 2.0 * jnp.sum(jnp.log(jnp.diag(L_avg)))
    logdet1 = 2.0 * jnp.sum(jnp.log(jnp.diag(L1)))
    logdet2 = 2.0 * jnp.sum(jnp.log(jnp.diag(L2)))

    # Mahalanobis distance
    diff = mu1 - mu2
    y = solve_triangular(L_avg, diff, lower=True)
    mahal_sq = jnp.dot(y, y)

    # Bhattacharyya coefficient
    log_bc = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdet_avg - 0.125 * mahal_sq
    bc = jnp.exp(log_bc)
    return jnp.maximum(0.0, 1.0 - bc)


# =============================================================================
# Main IMU Projection Kernel
# =============================================================================

@jax.jit
def imu_batched_projection_kernel(
    anchor_mus: jnp.ndarray,
    anchor_covs: jnp.ndarray,
    current_mu: jnp.ndarray,
    current_cov: jnp.ndarray,
    routing_weights: jnp.ndarray,
    imu_stamps: jnp.ndarray,
    imu_accel: jnp.ndarray,
    imu_gyro: jnp.ndarray,
    imu_valid: jnp.ndarray,
    R_imu: jnp.ndarray,
    R_nom: jnp.ndarray,
    dt_total: float,
    gravity: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Batched IMU projection kernel with Hellinger-tilted likelihood.

    This implementation follows the plan exactly:
    - Batched sigma-support (cubature) residual computation
    - Hellinger-tilted weighting for robustness
    - Global moment matching over joint deltas (anchor, current)
    - Exact Schur marginalization in backend

    Algorithm:
    1. For each anchor i with routing weight w_i:
       a. Form joint prior (anchor + current) in 30D
       b. Generate sigma-support points via spherical-radial cubature
       c. Propagate through IMU residual model (using JAX Lie ops)
       d. Compute residual mean r̄_i and covariance S_i
       e. Compute Hellinger-tilted weight: w̃_i ∝ w_i · exp(-½r̄ᵀR⁻¹r̄) · exp(-γ D_H²)

    2. Global moment matching (e-projection):
       - Form global mixture over (anchor, sigma)
       - Moment-match onto ξ_j (tangent at current_mu)
       - Retract: current_mu ⊕ δ_mean

    Args:
        anchor_mus: Per-anchor state means (M, 15)
        anchor_covs: Per-anchor state covariances (M, 15, 15)
        current_mu: Current state mean (15,)
        current_cov: Current state covariance (15, 15)
        routing_weights: Dirichlet responsibilities (M,)
        imu_stamps: IMU stamps (N,)
        imu_accel: IMU accel samples (N,3)
        imu_gyro: IMU gyro samples (N,3)
        imu_valid: IMU valid mask (N,)
        R_imu: IMU covariance (9, 9)
        R_nom: Nominal residual covariance for Hellinger (9, 9)
        dt_total: Total integration time
        gravity: Gravity vector (3,)

    Returns:
        mu_new: Updated state mean (15,)
        cov_new: Updated state covariance (15, 15)
        diagnostics: Dict with computation statistics
    """
    M = anchor_mus.shape[0]
    state_dim = 15
    joint_dim = 2 * state_dim  # 30D joint: [anchor, current]

    # -------------------------------------------------------------------------
    # Sigma-support for block-diagonal joint prior (batched across anchors)
    #
    # We use spherical-radial cubature weights:
    #   - center point has 0 weight
    #   - 2n symmetric points each have weight 1/(2n)
    #
    # This is chosen because sigma points are treated as explicit mixture
    # support weights downstream (must be non-negative).
    # -------------------------------------------------------------------------
    scale = float(joint_dim)
    W = jnp.concatenate([
        jnp.array([0.0], dtype=anchor_mus.dtype),
        jnp.full((2 * joint_dim,), 1.0 / (2.0 * joint_dim), dtype=anchor_mus.dtype),
    ])  # (61,)

    I15 = jnp.eye(state_dim, dtype=anchor_mus.dtype)
    anchor_covs_reg = anchor_covs + COV_REGULARIZATION * I15[None, :, :]
    current_cov_reg = current_cov + COV_REGULARIZATION * I15

    L_anchor = jax.vmap(lambda C: cholesky(scale * C, lower=True))(anchor_covs_reg)  # (M,15,15)
    L_current = cholesky(scale * current_cov_reg, lower=True)  # (15,15)

    delta_anchor_cols = jnp.swapaxes(L_anchor, 1, 2)  # (M,15,15), columns as vectors
    delta_current_cols = jnp.swapaxes(L_current, 0, 1)  # (15,15), columns as vectors

    # Build anchor sigma states: [base, +δ_a, -δ_a, base, base]
    anchor_plus = jax.vmap(
        lambda xbar_a, cols: jax.vmap(_apply_delta_state, in_axes=(None, 0))(xbar_a, cols),
        in_axes=(0, 0),
    )(anchor_mus, delta_anchor_cols)  # (M,15,15)
    anchor_minus = jax.vmap(
        lambda xbar_a, cols: jax.vmap(_apply_delta_state, in_axes=(None, 0))(xbar_a, -cols),
        in_axes=(0, 0),
    )(anchor_mus, delta_anchor_cols)  # (M,15,15)
    anchor_base = jnp.broadcast_to(anchor_mus[:, None, :], (M, state_dim, state_dim))
    x_anchor = jnp.concatenate([anchor_mus[:, None, :], anchor_plus, anchor_minus, anchor_base, anchor_base], axis=1)  # (M,61,15)

    # Build current sigma states: [base, base, base, +δ_c, -δ_c]
    current_plus = jax.vmap(_apply_delta_state, in_axes=(None, 0))(current_mu, delta_current_cols)  # (15,15)
    current_minus = jax.vmap(_apply_delta_state, in_axes=(None, 0))(current_mu, -delta_current_cols)  # (15,15)
    current_base = jnp.broadcast_to(current_mu[None, :], (state_dim, state_dim))
    x_current_single = jnp.concatenate([current_mu[None, :], current_base, current_base, current_plus, current_minus], axis=0)  # (61,15)
    x_current = jnp.broadcast_to(x_current_single[None, :, :], (M, x_current_single.shape[0], state_dim))  # (M,61,15)

    # -------------------------------------------------------------------------
    # Residuals and per-(anchor,sigma) likelihood weights (no per-anchor loops)
    # -------------------------------------------------------------------------
    residuals = jax.vmap(
        lambda xa, xc: jax.vmap(
            imu_residual_from_raw,
            in_axes=(0, 0, None, None, None, None, None, None),
        )(xa, xc, imu_stamps, imu_accel, imu_gyro, imu_valid, gravity, dt_total),
        in_axes=(0, 0),
    )(x_anchor, x_current)  # (M,61,9)

    R_imu_reg = R_imu + jnp.eye(9, dtype=R_imu.dtype) * COV_REGULARIZATION
    L_R = cholesky(R_imu_reg, lower=True)

    residuals_flat = residuals.reshape((-1, 9))  # (M*61,9)
    y = solve_triangular(L_R, residuals_flat.T, lower=True).T
    mahal_sq = jnp.sum(y * y, axis=1).reshape((M, -1))  # (M,61)
    log_lik = -0.5 * mahal_sq

    # Per-anchor Hellinger tilt from predictive residual Gaussian (r̄, S)
    r_bar = jnp.einsum("s,msd->md", W, residuals)
    r_centered = residuals - r_bar[:, None, :]
    S = jnp.einsum("s,msi,msj->mij", W, r_centered, r_centered)
    S = 0.5 * (S + jnp.swapaxes(S, 1, 2)) + jnp.eye(9, dtype=S.dtype)[None, :, :] * COV_REGULARIZATION

    h_sq = jax.vmap(
        lambda rb, Si: hellinger_squared_gaussian(rb, Si, jnp.zeros(9, dtype=rb.dtype), R_nom)
    )(r_bar, S)  # (M,)
    hellinger_tilt = jnp.exp(-HELLINGER_TILT_WEIGHT * h_sq)  # (M,)
    hellinger_mean = jnp.sum(routing_weights * h_sq) / jnp.maximum(jnp.sum(routing_weights), MIN_MIXTURE_WEIGHT)

    w_base = routing_weights * hellinger_tilt
    w_tilde = (w_base[:, None] * W[None, :] * jnp.exp(log_lik))
    w_sum = jnp.sum(w_tilde)
    valid = w_sum >= MIN_MIXTURE_WEIGHT
    w_sum_safe = jnp.where(valid, w_sum, 1.0)
    w = jnp.where(valid, w_tilde / w_sum_safe, jnp.zeros_like(w_tilde))  # (M,61)

    # -------------------------------------------------------------------------
    # Global moment matching (Legendre e-projection) onto joint deltas (anchor, current)
    # -------------------------------------------------------------------------
    anchor_pose_base = anchor_mus[:, :6]
    anchor_rest_base = anchor_mus[:, 6:]
    current_pose_base = current_mu[:6]
    current_rest_base = current_mu[6:]

    anchor_pose_delta = jax.vmap(
        lambda xa, base: jax.vmap(lambda p: se3_minus(p, base))(xa[:, :6]),
        in_axes=(0, 0),
    )(x_anchor, anchor_pose_base)  # (M,61,6)
    anchor_rest_delta = x_anchor[:, :, 6:] - anchor_rest_base[:, None, :]  # (M,61,9)
    anchor_delta = jnp.concatenate([anchor_pose_delta, anchor_rest_delta], axis=2)  # (M,61,15)

    current_pose_sigma_flat = x_current[:, :, :6].reshape((-1, 6))
    current_pose_delta_flat = jax.vmap(lambda p: se3_minus(p, current_pose_base))(current_pose_sigma_flat)
    current_pose_delta = current_pose_delta_flat.reshape((M, -1, 6))  # (M,61,6)
    current_rest_delta = x_current[:, :, 6:] - current_rest_base[None, None, :]  # (M,61,9)
    current_delta = jnp.concatenate([current_pose_delta, current_rest_delta], axis=2)  # (M,61,15)

    joint_delta = jnp.concatenate([anchor_delta, current_delta], axis=2)  # (M,61,30)

    joint_mean_raw = jnp.einsum("ms,msd->d", w, joint_delta)
    joint_centered = joint_delta - joint_mean_raw[None, None, :]
    cov_joint_raw = jnp.einsum("ms,msi,msj->ij", w, joint_centered, joint_centered)
    cov_joint_raw = 0.5 * (cov_joint_raw + cov_joint_raw.T) + jnp.eye(joint_dim, dtype=cov_joint_raw.dtype) * COV_REGULARIZATION

    joint_mean = jnp.where(valid, joint_mean_raw, jnp.zeros_like(joint_mean_raw))
    cov_joint = jnp.where(valid, cov_joint_raw, jnp.eye(joint_dim, dtype=cov_joint_raw.dtype) * COV_REGULARIZATION)

    anchor_mass = jnp.sum(w_tilde, axis=1)
    valid_anchors = jnp.sum(anchor_mass > MIN_MIXTURE_WEIGHT).astype(jnp.int32)

    ess = jnp.where(valid, 1.0 / jnp.sum(jnp.square(w)), 0.0)
    degenerate_weights = jnp.logical_not(valid)
    weight_entropy = -jnp.sum(w * jnp.log(w + MIN_MIXTURE_WEIGHT))

    def _build_diag_ok():
        return {
            "n_anchors": jnp.asarray(M, dtype=jnp.int32),
            "valid_anchors": valid_anchors,
            "ess": ess,
            "degenerate_weights": degenerate_weights,
            "hellinger_mean": hellinger_mean,
            "weight_entropy": weight_entropy,
        }

    def _build_diag_bad():
        return {
            "n_anchors": jnp.asarray(M, dtype=jnp.int32),
            "valid_anchors": jnp.asarray(0, dtype=jnp.int32),
            "ess": jnp.asarray(0.0, dtype=current_mu.dtype),
            "degenerate_weights": jnp.asarray(True),
            "hellinger_mean": jnp.asarray(0.0, dtype=current_mu.dtype),
            "weight_entropy": jnp.asarray(0.0, dtype=current_mu.dtype),
        }

    diag_ok = _build_diag_ok()
    diag_bad = _build_diag_bad()

    joint_out, cov_out, diag_out = lax.cond(
        valid,
        lambda _: (joint_mean, cov_joint, diag_ok),
        lambda _: (jnp.zeros_like(joint_mean), cov_joint, diag_bad),
        operand=None,
    )

    return joint_out, cov_out, diag_out
