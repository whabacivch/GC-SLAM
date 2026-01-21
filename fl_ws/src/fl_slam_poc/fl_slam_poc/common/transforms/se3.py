"""
SE(3) geometry using Lie algebra (tangent space) representation.

State representation: (x, y, z, rx, ry, rz) where:
- (x, y, z): translation in R^3
- (rx, ry, rz): rotation vector (axis-angle) in so(3)

This avoids RPY singularities and enables proper covariance transport
via the adjoint representation. Following information geometry principles,
operations are closed-form in the tangent space.

Numerical Policy:
    Epsilon thresholds are chosen based on IEEE 754 double precision:
    - ROTATION_EPSILON = 1e-10: ~sqrt(machine_epsilon) for stable trig
    - SINGULARITY_EPSILON = 1e-6: threshold for π-singularity handling
    
    These are NUMERICAL STABILITY choices, not model parameters.
    They affect only the computational path, not the mathematical result.

References:
- Barfoot (2017): State Estimation for Robotics
- Sola et al. (2018): A micro Lie theory for state estimation
- Combe (2022-2025): Pre-Frobenius manifolds and information geometry
"""

import math
import numpy as np
from typing import Tuple


# =============================================================================
# Numerical Constants (stability, not policy)
# =============================================================================
# These are chosen based on IEEE 754 double precision (~15 decimal digits)
# They affect computational path only, not mathematical result.

# For small-angle approximations: use when θ < ε to avoid division by ~0
# Choice: ~sqrt(machine_epsilon) ≈ 1e-8, use 1e-10 for safety margin
ROTATION_EPSILON: float = 1e-10

# For π-singularity handling: eigenvalue decomposition threshold
# Choice: 1e-6 provides stable numerics near θ = π
SINGULARITY_EPSILON: float = 1e-6


# =============================================================================
# Rotation vector <-> Rotation matrix conversions (so(3) <-> SO(3))
# =============================================================================


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector (hat operator)."""
    v = np.asarray(v, dtype=float).reshape(-1)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)


def unskew(S: np.ndarray) -> np.ndarray:
    """Extract 3-vector from skew-symmetric matrix (vee operator)."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector (axis-angle) to rotation matrix.
    Uses Rodrigues' formula: R = I + sin(θ)[ω]_× + (1-cos(θ))[ω]_×²
    
    This is the exponential map exp: so(3) -> SO(3).
    
    Numerical note: For θ < ROTATION_EPSILON, uses Taylor expansion
    to avoid division by small numbers. This is numerically equivalent
    to the exact formula.
    """
    rotvec = np.asarray(rotvec, dtype=float).reshape(-1)
    theta = np.linalg.norm(rotvec)
    
    if theta < ROTATION_EPSILON:
        # Small angle: R ≈ I + [rotvec]_× (first-order Taylor)
        # Error is O(θ²), which is < 1e-20 for θ < 1e-10
        return np.eye(3, dtype=float) + skew(rotvec)
    
    axis = rotvec / theta
    K = skew(axis)
    
    # Rodrigues' formula
    R = np.eye(3, dtype=float) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    return R


def rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to rotation vector (axis-angle).
    This is the logarithmic map log: SO(3) -> so(3).
    
    Handles three cases:
    1. θ ≈ 0: Extract from skew-symmetric part
    2. θ ≈ π: Use eigenvalue decomposition (singularity)
    3. General: Standard formula
    """
    R = np.asarray(R, dtype=float)
    
    # Compute angle from trace: tr(R) = 1 + 2*cos(θ)
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)
    
    if theta < ROTATION_EPSILON:
        # Small angle: extract from skew-symmetric part
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float) / 2.0
    
    if abs(theta - math.pi) < SINGULARITY_EPSILON:
        # Near π: use deterministic axis extraction from diagonal
        axis = np.zeros(3, dtype=float)
        axis[0] = math.sqrt(max((R[0, 0] + 1.0) * 0.5, 0.0))
        axis[1] = math.sqrt(max((R[1, 1] + 1.0) * 0.5, 0.0))
        axis[2] = math.sqrt(max((R[2, 2] + 1.0) * 0.5, 0.0))

        # Resolve sign ambiguity using off-diagonal elements
        if axis[0] > 1e-6:
            axis[1] = math.copysign(axis[1], R[0, 1])
            axis[2] = math.copysign(axis[2], R[0, 2])
        elif axis[1] > 1e-6:
            axis[0] = math.copysign(axis[0], R[0, 1])
            axis[2] = math.copysign(axis[2], R[1, 2])
        else:
            axis[0] = math.copysign(axis[0], R[0, 2])
            axis[1] = math.copysign(axis[1], R[1, 2])

        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return np.zeros(3, dtype=float)
        return axis / axis_norm * theta
    
    # General case: axis from skew-symmetric part
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float)
    axis = axis / (2.0 * math.sin(theta))
    return axis * theta


# =============================================================================
# Quaternion conversions
# =============================================================================


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to rotation matrix."""
    # Normalize
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3, dtype=float)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    
    # Rotation matrix from quaternion
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    
    return np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz), 2.0*(xz + wy)],
        [2.0*(xy + wz), 1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy), 2.0*(yz + wx), 1.0 - 2.0*(xx + yy)]
    ], dtype=float)


def rotmat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert rotation matrix to quaternion (x, y, z, w)."""
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    # Normalize
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    return (qx/n, qy/n, qz/n, qw/n)


def quat_to_rotvec(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert quaternion directly to rotation vector (axis-angle).
    
    OPTIMIZATION: This is more efficient than quat → rotmat → rotvec for ROS message conversion.
    Avoids intermediate rotation matrix construction (saves ~30% computation).
    
    Args:
        qx, qy, qz, qw: Quaternion components (x, y, z, w)
        
    Returns:
        Rotation vector (rx, ry, rz) representing axis-angle
        
    Mathematical Note:
        For quaternion q = (v, w) where v = [qx, qy, qz],
        the rotation vector is: θ * axis where:
        - θ = 2 * atan2(||v||, w)
        - axis = v / ||v||
    """
    # Normalize quaternion
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.zeros(3, dtype=float)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    
    # Ensure w >= 0 (shortest rotation)
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    
    # Handle small angles (avoid division by zero)
    if abs(qw) > 0.9999:  # θ ≈ 0
        # For small angles: rotvec ≈ 2 * [qx, qy, qz]
        return 2.0 * np.array([qx, qy, qz], dtype=float)
    
    # General case: rotvec = 2 * atan2(||v||, w) * v/||v||
    # where v = [qx, qy, qz]
    v_norm = math.sqrt(qx*qx + qy*qy + qz*qz)
    if v_norm < 1e-12:
        return np.zeros(3, dtype=float)
    
    angle = 2.0 * math.atan2(v_norm, qw)
    axis = np.array([qx/v_norm, qy/v_norm, qz/v_norm], dtype=float)
    
    return angle * axis


def se3_relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute group-consistent relative transform: a ⊖ b = b^{-1} ∘ a.
    """
    return se3_compose(se3_inverse(b), a)


# =============================================================================
# SE(3) operations in tangent space (rotation vector representation)
# =============================================================================


def se3_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compose two SE(3) transforms: T_a ∘ T_b.
    
    Each transform is (x, y, z, rx, ry, rz) where (rx, ry, rz) is a rotation vector.
    Result: T_out where T_out = T_a * T_b (right multiplication).
    
    This is exact group composition, not an approximation.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    
    t_a = a[:3]
    rotvec_a = a[3:6]
    t_b = b[:3]
    rotvec_b = b[3:6]
    
    R_a = rotvec_to_rotmat(rotvec_a)
    R_b = rotvec_to_rotmat(rotvec_b)
    
    # Composition: t_out = t_a + R_a @ t_b, R_out = R_a @ R_b
    t_out = t_a + R_a @ t_b
    R_out = R_a @ R_b
    rotvec_out = rotmat_to_rotvec(R_out)
    
    return np.array([t_out[0], t_out[1], t_out[2], 
                     rotvec_out[0], rotvec_out[1], rotvec_out[2]], dtype=float)


def se3_inverse(a: np.ndarray) -> np.ndarray:
    """
    Compute inverse of SE(3) transform.
    
    For T = (R, t), T^{-1} = (R^T, -R^T t).
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    t = a[:3]
    rotvec = a[3:6]
    
    R = rotvec_to_rotmat(rotvec)
    R_inv = R.T
    t_inv = -R_inv @ t
    rotvec_inv = rotmat_to_rotvec(R_inv)
    
    return np.array([t_inv[0], t_inv[1], t_inv[2],
                     rotvec_inv[0], rotvec_inv[1], rotvec_inv[2]], dtype=float)


def se3_apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply SE(3) transform to point(s).
    
    T: (x, y, z, rx, ry, rz)
    points: (N, 3) or (3,)
    Returns: transformed points (same shape as input)
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    points = np.asarray(points, dtype=float)
    
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 3)
    
    t = T[:3]
    R = rotvec_to_rotmat(T[3:6])
    
    result = (R @ points.T).T + t
    
    if single:
        return result.reshape(-1)
    return result


# =============================================================================
# SE(3) Lie algebra operations for covariance transport
# =============================================================================


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """
    Compute the adjoint representation Ad_T for SE(3).
    
    Used for covariance transport: if Σ is covariance at identity,
    then Σ' = Ad_T @ Σ @ Ad_T.T is covariance at T.
    
    Ad_T = | R    0 |
           | [t]_× R   R |
    
    where [t]_× is the skew-symmetric matrix of translation.
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    t = T[:3]
    R = rotvec_to_rotmat(T[3:6])
    t_skew = skew(t)
    
    Ad = np.zeros((6, 6), dtype=float)
    Ad[:3, :3] = R
    Ad[3:6, :3] = t_skew @ R
    Ad[3:6, 3:6] = R
    
    return Ad


def se3_cov_compose(cov_a: np.ndarray, cov_b: np.ndarray, T_a: np.ndarray) -> np.ndarray:
    """
    Compose covariances under SE(3) composition T_out = T_a ∘ T_b.
    
    For independent uncertainties, the composed covariance is:
    Σ_out = Σ_a + Ad_{T_a} @ Σ_b @ Ad_{T_a}.T
    
    This properly transports the covariance from frame B to frame A.
    
    Following information geometry: this is the exact transport in the
    tangent space, not an approximation.
    """
    cov_a = np.asarray(cov_a, dtype=float)
    cov_b = np.asarray(cov_b, dtype=float)
    
    Ad = se3_adjoint(T_a)
    cov_b_transported = Ad @ cov_b @ Ad.T
    
    return cov_a + cov_b_transported


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map from se(3) Lie algebra to SE(3) group.
    
    xi = (rho, phi) where rho is translational velocity and phi is angular velocity.
    Returns SE(3) element as (t, rotvec).
    
    Uses closed-form Rodriguez formula for SE(3).
    """
    xi = np.asarray(xi, dtype=float).reshape(-1)
    rho = xi[:3]  # translational part
    phi = xi[3:6]  # rotational part
    
    theta = np.linalg.norm(phi)
    
    if theta < ROTATION_EPSILON:
        # Small angle: t ≈ rho, R ≈ I
        return np.concatenate([rho, phi])
    
    axis = phi / theta
    K = skew(axis)
    
    # Rotation matrix via Rodrigues
    R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    
    # Translation via left Jacobian: t = J_l * rho
    # J_l = I + (1-cos)/θ² [φ]_× + (θ-sin)/θ³ [φ]_×²
    J_l = (np.eye(3) + 
           (1.0 - math.cos(theta)) / (theta * theta) * skew(phi) +
           (theta - math.sin(theta)) / (theta * theta * theta) * (skew(phi) @ skew(phi)))
    t = J_l @ rho
    
    rotvec = phi
    return np.array([t[0], t[1], t[2], rotvec[0], rotvec[1], rotvec[2]], dtype=float)


def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithmic map from SE(3) group to se(3) Lie algebra.
    
    T = (t, rotvec) -> xi = (rho, phi)
    
    Uses closed-form inverse of Rodriguez formula.
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    t = T[:3]
    rotvec = T[3:6]
    
    theta = np.linalg.norm(rotvec)
    
    if theta < ROTATION_EPSILON:
        # Small angle: rho ≈ t, phi ≈ rotvec
        return np.concatenate([t, rotvec])
    
    # Inverse left Jacobian: J_l^{-1}
    # J_l^{-1} = I - 0.5 [φ]_× + (1/θ² - (1+cos)/(2θ sin)) [φ]_×²
    half_theta = theta / 2.0
    cot_half = 1.0 / math.tan(half_theta) if abs(half_theta) > ROTATION_EPSILON else 1.0 / half_theta
    
    K = skew(rotvec / theta)
    J_l_inv = (np.eye(3) - 
               0.5 * skew(rotvec) +
               (1.0 - half_theta * cot_half) * (K @ K))
    
    rho = J_l_inv @ t
    phi = rotvec
    
    return np.array([rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]], dtype=float)


# =============================================================================
# Jacobians for ICP covariance (sensor extraction layer only)
# =============================================================================


def icp_jacobian_point(point: np.ndarray) -> np.ndarray:
    """
    Jacobian of transformed point w.r.t. SE(3) perturbation in tangent space.
    
    For p' = T ⊕ δξ applied to point p:
    ∂p'/∂δξ = [I | -[p']_×]
    
    where [p']_× is the skew-symmetric matrix of the transformed point.
    
    Basis convention: [δx, δy, δz, δωx, δωy, δωz] (translation first, rotation second)
    """
    point = np.asarray(point, dtype=float).reshape(-1)
    J = np.zeros((3, 6), dtype=float)
    J[:3, :3] = np.eye(3)
    J[:3, 3:6] = -skew(point)
    return J


def icp_covariance_tangent(
    transform: np.ndarray,
    src_transformed: np.ndarray,
    mse: float
) -> np.ndarray:
    """
    Compute ICP covariance in se(3) tangent space coordinates.
    
    Uses the normal-equation approximation: Σ = σ² (J^T J)^{-1}
    where J is the Jacobian of residuals w.r.t. se(3) perturbation.
    
    Basis convention: [δx, δy, δz, δωx, δωy, δωz] (translation first, rotation second)
    
    This is valid for sensor-to-evidence extraction (per Jacobian policy).
    """
    src_transformed = np.asarray(src_transformed, dtype=float)
    if src_transformed.size == 0:
        return np.eye(6, dtype=float) * 1e3
    
    JtJ = np.zeros((6, 6), dtype=float)
    for pt in src_transformed:
        J = icp_jacobian_point(pt)
        JtJ += J.T @ J
    
    sigma2 = max(float(mse), 1e-12)
    
    try:
        cov = sigma2 * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(JtJ)
    
    return cov

