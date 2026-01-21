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

from fl_slam_poc.backend.fusion.lie_jax import so3_exp, so3_log, se3_plus, se3_minus


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
      [p(3), ω(3), v(3), b_g(3), b_a(3)]

    Pose uses SE(3) right-composition retraction; remaining terms are Euclidean.
    """
    pose = se3_plus(xbar[:6], delta[:6])
    rest = xbar[6:] + delta[6:]
    return jnp.concatenate([pose, rest], axis=0)


# =============================================================================
# IMU Residual Model (Forster et al. 2017)
# =============================================================================

def imu_prediction_residual(
    xi_anchor: jnp.ndarray,
    xi_current: jnp.ndarray,
    z_imu: jnp.ndarray,
    gravity: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Compute IMU prediction residual following Forster et al. (2017) Eq. 24-26.

    Note:
        Biases are part of the 15D state, but this residual currently does not
        include bias sensitivity because `IMUFactor.msg` does not carry
        preintegration Jacobians (e.g., ∂Δp/∂b, ∂Δv/∂b, ∂ΔR/∂b). The node treats
        this as an explicit approximation and applies Frobenius correction.

    Args:
        xi_anchor: Anchor state (15,): [p_i, ω_i, v_i, b_g_i, b_a_i]
        xi_current: Current state (15,): [p_j, ω_j, v_j, b_g_j, b_a_j]
        z_imu: Preintegrated measurement (9,): [Δp, Δv, Δθ]
        gravity: Gravity vector (3,) in world frame
        dt: Integration time interval

    Returns:
        residual: 9D residual [r_p, r_v, r_θ]
    """
    p_i = xi_anchor[:3]
    omega_i = xi_anchor[3:6]
    v_i = xi_anchor[6:9]

    p_j = xi_current[:3]
    omega_j = xi_current[3:6]
    v_j = xi_current[6:9]

    # Rotation matrices (using JAX Lie ops)
    R_i = so3_exp(omega_i)
    R_j = so3_exp(omega_j)

    # Predicted measurements (Forster Eq. 24-26)
    # Position increment in anchor frame
    delta_p_pred = R_i.T @ (p_j - p_i - v_i * dt - 0.5 * gravity * dt**2)

    # Velocity increment in anchor frame
    delta_v_pred = R_i.T @ (v_j - v_i - gravity * dt)

    # Rotation increment
    delta_R_pred = R_i.T @ R_j
    delta_omega_pred = so3_log(delta_R_pred)

    z_p = z_imu[:3]
    z_v = z_imu[3:6]
    z_omega = z_imu[6:9]

    r_p = z_p - delta_p_pred
    r_v = z_v - delta_v_pred
    r_omega = z_omega - delta_omega_pred

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

def imu_batched_projection_kernel(
    anchor_mus: jnp.ndarray,
    anchor_covs: jnp.ndarray,
    current_mu: jnp.ndarray,
    current_cov: jnp.ndarray,
    routing_weights: jnp.ndarray,
    z_imu: jnp.ndarray,
    R_imu: jnp.ndarray,
    R_nom: jnp.ndarray,
    dt: float,
    gravity: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Batched IMU projection kernel with Hellinger-tilted likelihood.

    This implementation follows the plan exactly:
    - Batched sigma-support (cubature) residual computation
    - Hellinger-tilted weighting for robustness
    - Global moment matching across all anchor posteriors

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
        z_imu: IMU measurement (9,)
        R_imu: IMU covariance (9, 9)
        R_nom: Nominal residual covariance for Hellinger (9, 9)
        dt: Integration time
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
        lambda xa, xc: jax.vmap(imu_prediction_residual, in_axes=(0, 0, None, None, None))(xa, xc, z_imu, gravity, dt),
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
    S = S + R_imu[None, :, :]
    S = 0.5 * (S + jnp.swapaxes(S, 1, 2)) + jnp.eye(9, dtype=S.dtype)[None, :, :] * COV_REGULARIZATION

    h_sq = jax.vmap(lambda rb, Si: hellinger_squared_gaussian(rb, Si, jnp.zeros(9, dtype=rb.dtype), R_nom))(r_bar, S)  # (M,)
    hellinger_tilt = jnp.exp(-HELLINGER_TILT_WEIGHT * h_sq)  # (M,)

    w_base = routing_weights * hellinger_tilt
    w_tilde = (w_base[:, None] * W[None, :] * jnp.exp(log_lik))
    w_sum = jnp.sum(w_tilde)
    valid = w_sum >= MIN_MIXTURE_WEIGHT
    w_sum_safe = jnp.where(valid, w_sum, 1.0)
    w = jnp.where(valid, w_tilde / w_sum_safe, jnp.zeros_like(w_tilde))  # (M,61)

    # -------------------------------------------------------------------------
    # Global moment matching (Legendre e-projection) onto ξ_j (tangent at current_mu)
    # -------------------------------------------------------------------------
    pose_base = current_mu[:6]
    rest_base = current_mu[6:]

    pose_sigma_flat = x_current[:, :, :6].reshape((-1, 6))
    pose_delta_flat = jax.vmap(lambda p: se3_minus(p, pose_base))(pose_sigma_flat)
    pose_delta = pose_delta_flat.reshape((M, -1, 6))  # (M,61,6)
    rest_delta = x_current[:, :, 6:] - rest_base[None, None, :]  # (M,61,9)
    delta = jnp.concatenate([pose_delta, rest_delta], axis=2)  # (M,61,15)

    delta_mean_raw = jnp.einsum("ms,msd->d", w, delta)
    delta_centered = delta - delta_mean_raw[None, None, :]
    cov_new_raw = jnp.einsum("ms,msi,msj->ij", w, delta_centered, delta_centered)
    cov_new_raw = 0.5 * (cov_new_raw + cov_new_raw.T) + jnp.eye(state_dim, dtype=cov_new_raw.dtype) * COV_REGULARIZATION

    delta_mean = jnp.where(valid, delta_mean_raw, jnp.zeros_like(delta_mean_raw))
    pose_new = se3_plus(pose_base, delta_mean[:6])
    rest_new = rest_base + delta_mean[6:]
    mu_new_raw = jnp.concatenate([pose_new, rest_new], axis=0)

    mu_new = jnp.where(valid, mu_new_raw, current_mu)
    cov_new = jnp.where(valid, cov_new_raw, current_cov)

    anchor_mass = jnp.sum(w_tilde, axis=1)
    valid_anchors = jnp.sum(anchor_mass > MIN_MIXTURE_WEIGHT).astype(jnp.int32)

    ess = jnp.where(valid, 1.0 / jnp.sum(jnp.square(w)), 0.0)
    degenerate_weights = jnp.logical_not(valid)

    def _build_diag_ok():
        return {
            "n_anchors": jnp.asarray(M, dtype=jnp.int32),
            "valid_anchors": valid_anchors,
            "ess": ess,
            "degenerate_weights": degenerate_weights,
        }

    def _build_diag_bad():
        return {
            "n_anchors": jnp.asarray(M, dtype=jnp.int32),
            "valid_anchors": jnp.asarray(0, dtype=jnp.int32),
            "ess": jnp.asarray(0.0, dtype=current_mu.dtype),
            "degenerate_weights": jnp.asarray(True),
        }

    diag_ok = _build_diag_ok()
    diag_bad = _build_diag_bad()

    mu_out, cov_out, diag_out = lax.cond(
        valid,
        lambda _: (mu_new, cov_new, diag_ok),
        lambda _: (current_mu, current_cov, diag_bad),
        operand=None,
    )

    return mu_out, cov_out, diag_out
