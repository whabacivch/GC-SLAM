"""
JAX Lie group operators for SE(3).

Provides manifold-correct operations for IMU fusion without global flattening.
All operations are compatible with JAX transformations (jit, vmap, grad).

Key functions:
- so3_exp: Exponential map SO(3) via Rodrigues formula
- so3_log: Logarithm map SO(3) (inverse Rodrigues)
- se3_plus: Manifold retraction x ⊕ δ
- se3_minus: Manifold difference x₁ ⊖ x₂

THIS MODULE REQUIRES JAX. There is no NumPy fallback.
If JAX is not installed, import will fail loudly.

Reference: Barfoot (2017), Forster et al. (2017)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple

# =============================================================================
# Numerical Constants (documented rationale)
# =============================================================================

# Small angle threshold for Taylor series approximation
# Below this, sin(θ)/θ ≈ 1 and (1-cos(θ))/θ² ≈ 0.5 to machine precision
# Derivation: |sin(θ) - θ| < θ³/6, so for θ < 1e-7, error < 1e-21
SMALL_ANGLE_THRESHOLD = 1e-7

# Near-π threshold for SO(3) logarithm singularity handling
# At θ = π, sin(θ) = 0, requiring alternative axis extraction
NEAR_PI_THRESHOLD = 1e-7


# =============================================================================
# Core JAX Implementations
# =============================================================================

@jit
def skew(v: jnp.ndarray) -> jnp.ndarray:
    """
    Skew-symmetric matrix from 3-vector.
    
    [v]× such that [v]× @ w = v × w (cross product)
    """
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


@jit
def so3_exp(omega: jnp.ndarray) -> jnp.ndarray:
    """
    SO(3) exponential map via Rodrigues formula.
    
    R = I + sin(θ)/θ · [ω]× + (1-cos(θ))/θ² · [ω]×²
    
    where θ = ||ω||.
    
    For small angles (θ < 1e-7), uses Taylor approximation:
    R ≈ I + [ω]×
    
    Args:
        omega: Rotation vector (3,) in radians
    
    Returns:
        R: Rotation matrix (3, 3) ∈ SO(3)
    """
    theta_sq = jnp.dot(omega, omega)
    theta = jnp.sqrt(theta_sq)
    
    # Compute skew matrix
    K = skew(omega)
    K_sq = K @ K
    
    # Small angle: R = I + K (Taylor series)
    # General: R = I + sin(θ)/θ * K + (1-cos(θ))/θ² * K²
    
    # Use jnp.where for branchless computation (JAX-friendly)
    # For small θ: sin(θ)/θ → 1, (1-cos(θ))/θ² → 0.5
    safe_theta = jnp.where(theta < SMALL_ANGLE_THRESHOLD, 1.0, theta)
    safe_theta_sq = jnp.where(theta_sq < SMALL_ANGLE_THRESHOLD**2, 1.0, theta_sq)
    
    sin_coeff = jnp.where(
        theta < SMALL_ANGLE_THRESHOLD,
        1.0,  # Taylor: sin(θ)/θ ≈ 1
        jnp.sin(safe_theta) / safe_theta
    )
    cos_coeff = jnp.where(
        theta < SMALL_ANGLE_THRESHOLD,
        0.5,  # Taylor: (1-cos(θ))/θ² ≈ 0.5
        (1.0 - jnp.cos(safe_theta)) / safe_theta_sq
    )
    
    R = jnp.eye(3) + sin_coeff * K + cos_coeff * K_sq
    return R


@jit
def so3_log(R: jnp.ndarray) -> jnp.ndarray:
    """
    SO(3) logarithm map (inverse Rodrigues).
    
    Returns rotation vector ω such that exp([ω]×) = R.
    
    Uses trace to recover angle: cos(θ) = (tr(R) - 1) / 2
    
    Special cases:
    - θ ≈ 0: ω ≈ vex(R - Rᵀ) / 2
    - θ ≈ π: Extract axis from diagonal
    
    Args:
        R: Rotation matrix (3, 3) ∈ SO(3)
    
    Returns:
        omega: Rotation vector (3,) in radians
    """
    # cos(θ) from trace
    cos_theta = 0.5 * (jnp.trace(R) - 1.0)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    
    # Extract skew-symmetric part: (R - Rᵀ) / 2
    skew_part = 0.5 * (R - R.T)
    vex_skew = jnp.array([skew_part[2, 1], skew_part[0, 2], skew_part[1, 0]])
    
    # Small angle approximation: ω ≈ vex((R - Rᵀ)/2)
    omega_small = vex_skew
    
    # General case: ω = θ / (2 sin(θ)) * vex(R - Rᵀ)
    sin_theta = jnp.sin(theta)
    safe_sin = jnp.where(jnp.abs(sin_theta) < SMALL_ANGLE_THRESHOLD, 1.0, sin_theta)
    omega_general = (theta / (2.0 * safe_sin)) * (2.0 * vex_skew)
    
    # Near-π case: extract axis from diagonal
    # R = I + 2aaᵀ - I = 2aaᵀ when θ = π
    # So diag(R) + 1 = 2a²
    diag_R = jnp.diag(R)
    diag_plus_1 = diag_R + 1.0
    idx = jnp.argmax(diag_plus_1)
    
    # Build axis from column
    axis_col = R[:, idx] + jnp.eye(3)[:, idx]
    axis_norm = jnp.linalg.norm(axis_col)
    safe_axis_norm = jnp.where(axis_norm < SMALL_ANGLE_THRESHOLD, 1.0, axis_norm)
    axis = axis_col / safe_axis_norm
    omega_pi = axis * theta
    
    # Select based on angle
    is_small = theta < SMALL_ANGLE_THRESHOLD
    is_near_pi = jnp.abs(theta - jnp.pi) < NEAR_PI_THRESHOLD
    
    omega = jnp.where(is_small, omega_small,
                      jnp.where(is_near_pi, omega_pi, omega_general))
    
    return omega


@jit
def se3_plus(x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """
    SE(3) retraction: x ⊕ δ.
    
    Composition: T(x) ∘ T(δ) in SE(3), where T is the pose represented
    by position p and rotation vector ω.
    
    x, delta are 6-vectors: [p(3), ω(3)]
    
    Args:
        x: Base pose [p(3), ω(3)]
        delta: Tangent vector [δp(3), δω(3)]
    
    Returns:
        x_new: Composed pose [p(3), ω(3)]
    """
    p_x = x[:3]
    omega_x = x[3:6]
    R_x = so3_exp(omega_x)
    
    p_delta = delta[:3]
    omega_delta = delta[3:6]
    R_delta = so3_exp(omega_delta)
    
    # Compose: T_new = T_x ∘ T_delta
    R_new = R_x @ R_delta
    p_new = p_x + R_x @ p_delta
    omega_new = so3_log(R_new)
    
    return jnp.concatenate([p_new, omega_new])


@jit
def se3_minus(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """
    SE(3) difference: x₁ ⊖ x₂.
    
    Returns δ such that x₂ ⊕ δ = x₁ (exactly on manifold).
    
    Computes: T(δ) = T(x₂)⁻¹ ∘ T(x₁)
    
    Args:
        x1: Target pose [p(3), ω(3)]
        x2: Reference pose [p(3), ω(3)]
    
    Returns:
        delta: Tangent vector [δp(3), δω(3)]
    """
    p1 = x1[:3]
    omega1 = x1[3:6]
    R1 = so3_exp(omega1)
    
    p2 = x2[:3]
    omega2 = x2[3:6]
    R2 = so3_exp(omega2)
    
    # Relative rotation: R_delta = R₂ᵀ @ R₁
    R_delta = R2.T @ R1
    omega_delta = so3_log(R_delta)
    
    # Relative position in x2 frame
    p_delta = R2.T @ (p1 - p2)
    
    return jnp.concatenate([p_delta, omega_delta])


@jit
def so3_adjoint(omega: jnp.ndarray) -> jnp.ndarray:
    """
    SO(3) adjoint representation.
    
    For rotation vector ω, Ad_R = R where R = exp([ω]×).
    
    The adjoint transforms tangent vectors under conjugation.
    """
    return so3_exp(omega)


@jit 
def se3_adjoint(xi: jnp.ndarray) -> jnp.ndarray:
    """
    SE(3) adjoint representation (6x6 matrix).
    
    For pose ξ = [p, ω], the adjoint is:
    
        Ad_T = | R    [p]× R |
               | 0    R      |
    
    where R = exp([ω]×) and [p]× is the skew matrix of p.
    
    Args:
        xi: Pose [p(3), ω(3)]
    
    Returns:
        Ad: Adjoint matrix (6, 6)
    """
    p = xi[:3]
    omega = xi[3:6]
    R = so3_exp(omega)
    p_skew = skew(p)
    
    Ad = jnp.zeros((6, 6))
    Ad = Ad.at[:3, :3].set(R)
    Ad = Ad.at[:3, 3:6].set(p_skew @ R)
    Ad = Ad.at[3:6, 3:6].set(R)
    
    return Ad


# =============================================================================
# Vectorized Operations (for batched processing)
# =============================================================================

# Batched versions using vmap
so3_exp_batch = jax.vmap(so3_exp)  # (N, 3) -> (N, 3, 3)
so3_log_batch = jax.vmap(so3_log)  # (N, 3, 3) -> (N, 3)
se3_plus_batch = jax.vmap(se3_plus)  # (N, 6), (N, 6) -> (N, 6)
se3_minus_batch = jax.vmap(se3_minus)  # (N, 6), (N, 6) -> (N, 6)
