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

from __future__ import annotations

from jax import jit

from fl_slam_poc.common.jax_init import jax, jnp

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
def vee(W: jnp.ndarray) -> jnp.ndarray:
    """
    Vee operator (inverse of skew/hat) for a 3x3 skew-symmetric matrix.

    Assumes W ≈ [w]× and returns w.
    """
    return jnp.array([W[2, 1], W[0, 2], W[1, 0]])


@jit
def so3_right_jacobian(phi: jnp.ndarray) -> jnp.ndarray:
    """
    SO(3) right Jacobian Jr(phi) (closed form, universal).

        Jr(phi) = I - B(theta) [phi]× + C2(theta) [phi]×^2

    where theta = ||phi|| and

        B(theta)  = (1 - cos(theta)) / theta^2
        C2(theta) = (theta - sin(theta)) / theta^3

    Uses analytic continuation near theta -> 0.
    """
    phi = jnp.asarray(phi, dtype=jnp.float64).reshape(-1)
    theta_sq = jnp.dot(phi, phi)
    theta = jnp.sqrt(theta_sq)
    K = skew(phi)
    K_sq = K @ K
    I = jnp.eye(3, dtype=jnp.float64)

    safe_theta = jnp.where(theta < SMALL_ANGLE_THRESHOLD, 1.0, theta)
    safe_theta_sq = jnp.where(theta_sq < SMALL_ANGLE_THRESHOLD**2, 1.0, theta_sq)
    safe_theta_cu = safe_theta_sq * safe_theta

    B = jnp.where(
        theta < SMALL_ANGLE_THRESHOLD,
        0.5 - theta_sq / 24.0,
        (1.0 - jnp.cos(safe_theta)) / safe_theta_sq,
    )
    C2 = jnp.where(
        theta < SMALL_ANGLE_THRESHOLD,
        1.0 / 6.0 - theta_sq / 120.0,
        (safe_theta - jnp.sin(safe_theta)) / safe_theta_cu,
    )

    return I - B * K + C2 * K_sq


@jit
def so3_right_jacobian_inv(phi: jnp.ndarray) -> jnp.ndarray:
    """
    SO(3) inverse right Jacobian Jr^{-1}(phi) (closed form, universal).

        Jr^{-1}(phi) = I + 1/2 [phi]× + D(theta) [phi]×^2

    where

        D(theta) = (1/theta^2) - (1 + cos(theta)) / (2*theta*sin(theta))

    Uses analytic continuation near theta -> 0.
    """
    phi = jnp.asarray(phi, dtype=jnp.float64).reshape(-1)
    theta_sq = jnp.dot(phi, phi)
    theta = jnp.sqrt(theta_sq)
    K = skew(phi)
    K_sq = K @ K
    I = jnp.eye(3, dtype=jnp.float64)

    eps = 1e-12
    denom = (2.0 * theta * jnp.sin(theta)) + eps
    D = jnp.where(
        theta < 1e-4,
        1.0 / 12.0 + theta_sq / 720.0,
        (1.0 / (theta_sq + eps)) - (1.0 + jnp.cos(theta)) / denom,
    )

    return I + 0.5 * K + D * K_sq


@jit
def se3_V(phi: jnp.ndarray) -> jnp.ndarray:
    """
    SE(3) V(phi) matrix mapping rho -> t in Exp([rho;phi]).

        Exp([rho;phi]) has translation t = V(phi) rho

    where

        V(phi) = I + B(theta)[phi]× + C(theta)[phi]×^2

    with

        B(theta) = (1 - cos(theta)) / theta^2
        C(theta) = (theta - sin(theta)) / theta^3
    """
    phi = jnp.asarray(phi, dtype=jnp.float64).reshape(-1)
    theta_sq = jnp.dot(phi, phi)
    theta = jnp.sqrt(theta_sq)

    K = skew(phi)
    K_sq = K @ K

    safe_theta = jnp.where(theta < SMALL_ANGLE_THRESHOLD, 1.0, theta)
    safe_theta_sq = jnp.where(theta_sq < SMALL_ANGLE_THRESHOLD**2, 1.0, theta_sq)
    safe_theta_cu = safe_theta_sq * safe_theta

    B = jnp.where(
        theta < SMALL_ANGLE_THRESHOLD,
        0.5 - theta_sq / 24.0,
        (1.0 - jnp.cos(safe_theta)) / safe_theta_sq,
    )
    C = jnp.where(
        theta < SMALL_ANGLE_THRESHOLD,
        1.0 / 6.0 - theta_sq / 120.0,
        (safe_theta - jnp.sin(safe_theta)) / safe_theta_cu,
    )

    return jnp.eye(3) + B * K + C * K_sq


@jit
def _se3_V_inv(phi: jnp.ndarray) -> jnp.ndarray:
    """
    SE(3) V(phi)^{-1} matrix for computing rho from t in Log.

    For small phi, V ≈ I so V^{-1} ≈ I.
    For larger phi, uses the closed-form inverse.

    The closed-form inverse is:
        V^{-1} = I - 0.5 * [phi]× + D(theta) * [phi]×²

    where D(theta) = (1/theta²) - (1 + cos(theta)) / (2 * theta * sin(theta))

    This is numerically stable and avoids linear solve.
    """
    phi = jnp.asarray(phi, dtype=jnp.float64).reshape(-1)
    theta_sq = jnp.dot(phi, phi)
    theta = jnp.sqrt(theta_sq)

    K = skew(phi)
    K_sq = K @ K
    I = jnp.eye(3, dtype=jnp.float64)

    # For small theta, use Taylor expansion: V^{-1} ≈ I - 0.5 * K + K²/12
    # D(theta) ≈ 1/12 + theta²/720 + ...
    eps = 1e-12
    safe_theta = jnp.where(theta < SMALL_ANGLE_THRESHOLD, 1.0, theta)
    safe_theta_sq = jnp.where(theta_sq < SMALL_ANGLE_THRESHOLD**2, 1.0, theta_sq)

    # Denominator: 2 * theta * sin(theta), regularized
    sin_theta = jnp.sin(safe_theta)
    denom = 2.0 * safe_theta * sin_theta + eps

    D = jnp.where(
        theta < SMALL_ANGLE_THRESHOLD,
        1.0 / 12.0 + theta_sq / 720.0,  # Taylor: 1/12 + θ²/720 + O(θ⁴)
        (1.0 / safe_theta_sq) - (1.0 + jnp.cos(safe_theta)) / denom,
    )

    return I - 0.5 * K + D * K_sq


@jit
def se3_log(T: jnp.ndarray) -> jnp.ndarray:
    """
    SE(3) logarithm map from group element to algebra (twist) in 6D representation.

    Input pose is represented as a 6-vector [t(3), rotvec(3)] where rotvec is an
    axis-angle rotation vector parameterization of R.

    Output is the se(3) twist [rho(3), phi(3)] such that:

        se3_exp([rho;phi]) == [t;rotvec]   (up to rotation-vector canonicalization)

    Critically, the translational twist component is:

        rho = V(phi)^{-1} t

    not just rho=t.

    NUMERICAL STABILITY NOTE:
    Uses closed-form V^{-1} instead of linear solve to avoid numerical
    instability when phi is small (where V is near-identity but solve
    can accumulate errors). The closed-form uses Taylor expansion for
    small angles and the exact inverse formula for larger angles.
    """
    T = jnp.asarray(T, dtype=jnp.float64).reshape(-1)
    t = T[:3]
    rotvec = T[3:6]

    # Canonicalize rotation via Log(Exp(rotvec)) for robustness near pi.
    R = so3_exp(rotvec)
    phi = so3_log(R)

    # Use closed-form V^{-1} instead of linear solve for numerical stability
    V_inv = _se3_V_inv(phi)
    rho = V_inv @ t

    return jnp.concatenate([rho, phi])


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
    diag_R = jnp.diag(R)
    diag_plus_1 = diag_R + 1.0
    # Avoid discrete argmax selection: softmax-mixture of the 3 candidate axes.
    # This removes a hard branch point while preserving the standard diagonal heuristic.
    k = jnp.array(50.0, dtype=jnp.float64)  # sharpness; larger -> closer to argmax
    w = jax.nn.softmax(k * diag_plus_1)  # (3,)

    I = jnp.eye(3, dtype=jnp.float64)
    axis_cols = jnp.stack(
        [R[:, 0] + I[:, 0], R[:, 1] + I[:, 1], R[:, 2] + I[:, 2]],
        axis=0,
    )  # (3,3)
    axis_col = w[0] * axis_cols[0] + w[1] * axis_cols[1] + w[2] * axis_cols[2]
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
def se3_compose(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compose two SE(3) transforms: T_a ∘ T_b."""
    a = jnp.asarray(a, dtype=jnp.float64).reshape(-1)
    b = jnp.asarray(b, dtype=jnp.float64).reshape(-1)

    t_a = a[:3]
    rotvec_a = a[3:6]
    t_b = b[:3]
    rotvec_b = b[3:6]

    R_a = so3_exp(rotvec_a)
    R_b = so3_exp(rotvec_b)

    t_out = t_a + R_a @ t_b
    R_out = R_a @ R_b
    rotvec_out = so3_log(R_out)

    return jnp.concatenate([t_out, rotvec_out])


@jit
def se3_inverse(a: jnp.ndarray) -> jnp.ndarray:
    """Compute inverse of SE(3) transform."""
    a = jnp.asarray(a, dtype=jnp.float64).reshape(-1)
    t = a[:3]
    rotvec = a[3:6]

    R = so3_exp(rotvec)
    R_inv = R.T
    t_inv = -R_inv @ t
    rotvec_inv = so3_log(R_inv)

    return jnp.concatenate([t_inv, rotvec_inv])


@jit
def se3_relative(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute group-consistent relative transform: a ⊖ b = b^{-1} ∘ a."""
    return se3_compose(se3_inverse(b), a)


@jit
def se3_cov_compose(cov_a: jnp.ndarray, cov_b: jnp.ndarray, T_a: jnp.ndarray) -> jnp.ndarray:
    """Compose covariances under SE(3) composition T_out = T_a ∘ T_b."""
    cov_a = jnp.asarray(cov_a, dtype=jnp.float64)
    cov_b = jnp.asarray(cov_b, dtype=jnp.float64)

    Ad = se3_adjoint(T_a)
    cov_b_transported = Ad @ cov_b @ Ad.T
    return cov_a + cov_b_transported


@jit
def se3_exp(xi: jnp.ndarray) -> jnp.ndarray:
    """Exponential map from se(3) to SE(3) in rotation-vector representation."""
    xi = jnp.asarray(xi, dtype=jnp.float64).reshape(-1)
    rho = xi[:3]
    phi = xi[3:6]

    theta_sq = jnp.dot(phi, phi)
    theta = jnp.sqrt(theta_sq)

    K = skew(phi)
    K_sq = K @ K

    safe_theta = jnp.where(theta < SMALL_ANGLE_THRESHOLD, 1.0, theta)
    safe_theta_sq = jnp.where(theta_sq < SMALL_ANGLE_THRESHOLD**2, 1.0, theta_sq)
    safe_theta_cu = safe_theta_sq * safe_theta

    B = jnp.where(
        theta < SMALL_ANGLE_THRESHOLD,
        0.5 - theta_sq / 24.0,
        (1.0 - jnp.cos(safe_theta)) / safe_theta_sq,
    )
    C = jnp.where(
        theta < SMALL_ANGLE_THRESHOLD,
        1.0 / 6.0 - theta_sq / 120.0,
        (safe_theta - jnp.sin(safe_theta)) / safe_theta_cu,
    )

    V = jnp.eye(3) + B * K + C * K_sq
    t = V @ rho

    return jnp.concatenate([t, phi])


@jit
def so3_adjoint(omega: jnp.ndarray) -> jnp.ndarray:
    """
    SO(3) adjoint representation.
    
    For rotation vector ω, Ad_R = R where R = exp([ω]×).
    """
    return so3_exp(omega)


@jit
def se3_adjoint(xi: jnp.ndarray) -> jnp.ndarray:
    """
    SE(3) adjoint representation (6x6 matrix).
    
    For pose ξ = [p, ω], the adjoint is:
    
        Ad_T = | R    0      |
               | [p]× R  R   |
    
    where R = exp([ω]×) and [p]× is the skew matrix of p.
    """
    p = xi[:3]
    omega = xi[3:6]
    R = so3_exp(omega)
    p_skew = skew(p)
    
    Ad = jnp.zeros((6, 6))
    Ad = Ad.at[:3, :3].set(R)
    Ad = Ad.at[3:6, :3].set(p_skew @ R)
    Ad = Ad.at[3:6, 3:6].set(R)
    
    return Ad


# =============================================================================
# Vectorized Operations (for batched processing)
# =============================================================================

so3_exp_batch = jax.vmap(so3_exp)  # (N, 3) -> (N, 3, 3)
so3_log_batch = jax.vmap(so3_log)  # (N, 3, 3) -> (N, 3)
se3_plus_batch = jax.vmap(se3_plus)  # (N, 6), (N, 6) -> (N, 6)
se3_minus_batch = jax.vmap(se3_minus)  # (N, 6), (N, 6) -> (N, 6)


__all__ = [
    "SMALL_ANGLE_THRESHOLD",
    "NEAR_PI_THRESHOLD",
    "skew",
    "vee",
    "so3_exp",
    "so3_log",
    "so3_right_jacobian",
    "so3_right_jacobian_inv",
    "so3_adjoint",
    "se3_plus",
    "se3_minus",
    "se3_compose",
    "se3_inverse",
    "se3_relative",
    "se3_cov_compose",
    "se3_exp",
    "se3_log",
    "se3_V",
    "se3_adjoint",
    "so3_exp_batch",
    "so3_log_batch",
    "se3_plus_batch",
    "se3_minus_batch",
]
