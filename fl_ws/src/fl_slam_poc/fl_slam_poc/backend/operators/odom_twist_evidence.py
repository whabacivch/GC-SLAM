"""
Odometry twist evidence operators for Golden Child SLAM v2.

These operators add kinematic constraints from wheel odometry twist (velocity):
1. Body velocity factor - constrains linear velocity
2. Yaw rate factor - constrains angular velocity about z

The twist from wheel odometry provides strong kinematic coupling that was
previously unused (only pose was read from odom messages).

Reference: Plan Phase 2 (odom twist factors)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import D_Z
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import domain_projection_psd, spd_cholesky_inverse_lifted
from fl_slam_poc.common.geometry import se3_jax


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class OdomVelocityEvidenceResult:
    """Result of odometry velocity evidence operator."""
    L_vel: jnp.ndarray  # (22, 22) information matrix contribution
    h_vel: jnp.ndarray  # (22,) information vector contribution
    r_vel: jnp.ndarray  # (3,) velocity residual in body frame


@dataclass
class OdomYawRateEvidenceResult:
    """Result of odometry yaw rate evidence operator."""
    L_wz: jnp.ndarray  # (22, 22) information matrix contribution
    h_wz: jnp.ndarray  # (22,) information vector contribution
    r_wz: float  # yaw rate residual


# =============================================================================
# Body Velocity Evidence
# =============================================================================


def odom_velocity_evidence(
    v_pred_world: jnp.ndarray,  # (3,) predicted velocity in world frame
    R_world_body: jnp.ndarray,  # (3,3) rotation from body to world
    v_odom_body: jnp.ndarray,  # (3,) odometry velocity in body frame
    Sigma_v: jnp.ndarray,  # (3,3) velocity covariance from odom
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "",
) -> Tuple[OdomVelocityEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Build Gaussian evidence from odometry velocity measurement.

    Computes the velocity residual between predicted velocity (transformed to
    body frame) and the odometry velocity measurement.

    The evidence constrains the velocity block [6:9] of the 22D state.

    Residual uses measurement - prediction:
        r_v = v_odom_body - (R_body_world @ v_pred_world)

    State ordering: [trans(0:3), rot(3:6), vel(6:9), bg(9:12), ba(12:15), dt(15:16), ex(16:22)]
    Velocity is at indices 6:9.

    Args:
        v_pred_world: Predicted velocity in world frame (3,)
        R_world_body: Rotation matrix from body to world (3,3)
        v_odom_body: Odometry velocity in body frame (3,)
        Sigma_v: Velocity covariance (3,3)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        chart_id: Chart identifier
        anchor_id: Anchor identifier

    Returns:
        Tuple of (OdomVelocityEvidenceResult, CertBundle, ExpectedEffect)
    """
    v_pred_world = jnp.asarray(v_pred_world, dtype=jnp.float64).reshape(-1)
    R_world_body = jnp.asarray(R_world_body, dtype=jnp.float64).reshape(3, 3)
    v_odom_body = jnp.asarray(v_odom_body, dtype=jnp.float64).reshape(-1)
    Sigma_v = jnp.asarray(Sigma_v, dtype=jnp.float64).reshape(3, 3)

    # Transform predicted velocity to body frame
    R_body_world = R_world_body.T
    v_pred_body = R_body_world @ v_pred_world

    # Velocity residual in body frame: measurement - prediction
    r_vel = v_odom_body - v_pred_body

    # Get precision matrix from covariance
    Sigma_v_psd = domain_projection_psd(Sigma_v, eps_psd).M_psd
    L_vel_3x3, lift_strength = spd_cholesky_inverse_lifted(Sigma_v_psd, eps_lift)

    # Build 22D information matrix (velocity at indices 6:9)
    L_vel = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L_vel = L_vel.at[6:9, 6:9].set(L_vel_3x3)

    # Information vector
    h_vel = jnp.zeros((D_Z,), dtype=jnp.float64)
    h_vel = h_vel.at[6:9].set(L_vel_3x3 @ r_vel)

    # NLL proxy
    nll_proxy = 0.5 * float(r_vel @ L_vel_3x3 @ r_vel)

    # Conditioning certificate
    eigvals = jnp.linalg.eigvalsh(Sigma_v_psd)
    eig_min = float(jnp.min(eigvals))
    eig_max = float(jnp.max(eigvals))
    cond = eig_max / max(eig_min, 1e-18)

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["OdomVelocityEvidence"],
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=int(jnp.sum(eigvals < 1e-12)),
        ),
        influence=InfluenceCert(
            lift_strength=float(lift_strength),
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    effect = ExpectedEffect(
        objective_name="odom_velocity_nll",
        predicted=nll_proxy,
        realized=None,
    )

    return OdomVelocityEvidenceResult(L_vel=L_vel, h_vel=h_vel, r_vel=r_vel), cert, effect


# =============================================================================
# Yaw Rate Evidence
# =============================================================================


def odom_yawrate_evidence(
    omega_z_pred: float,  # Predicted yaw rate (from state or gyro-bias corrected)
    omega_z_odom: float,  # Odometry yaw rate
    sigma_wz: float,  # Yaw rate std dev
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "",
) -> Tuple[OdomYawRateEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Build Gaussian evidence from odometry yaw rate measurement.

    Constrains the yaw rate component of angular velocity.
    This provides a direct comparison between wheel-derived yaw rate and
    the gyro-predicted yaw rate.

    For a ground robot, the yaw rate from wheels is typically very accurate.

    Note: This is a scalar constraint, not a full SO(3) angular velocity constraint.
    The yaw component is chosen because it's the dominant rotation for ground robots.

    Args:
        omega_z_pred: Predicted yaw rate (rad/s)
        omega_z_odom: Odometry yaw rate (rad/s)
        sigma_wz: Yaw rate std dev (rad/s)
        chart_id: Chart identifier
        anchor_id: Anchor identifier

    Returns:
        Tuple of (OdomYawRateEvidenceResult, CertBundle, ExpectedEffect)

    Note:
        This primarily affects the rotation block. For simplicity, we inject
        a soft constraint into the rotation increment. A more rigorous approach
        would couple this with the gyro bias estimate.
    """
    omega_z_pred = float(omega_z_pred)
    omega_z_odom = float(omega_z_odom)
    sigma_wz = float(sigma_wz)

    # Yaw rate residual: measurement - prediction
    r_wz = omega_z_odom - omega_z_pred

    # Precision
    precision_wz = 1.0 / (sigma_wz ** 2)

    # Build 22D information matrix
    # For yaw rate, we constrain the z-component of the rotation block
    # Rotation is at indices 3:6, and z (yaw) is the third component (index 5)
    L_wz = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L_wz = L_wz.at[5, 5].set(precision_wz)

    # Information vector
    h_wz = jnp.zeros((D_Z,), dtype=jnp.float64)
    h_wz = h_wz.at[5].set(precision_wz * r_wz)

    # NLL proxy
    nll_proxy = 0.5 * r_wz ** 2 * precision_wz

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["OdomYawRateEvidence"],
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    effect = ExpectedEffect(
        objective_name="odom_yawrate_nll",
        predicted=nll_proxy,
        realized=None,
    )

    return OdomYawRateEvidenceResult(L_wz=L_wz, h_wz=h_wz, r_wz=r_wz), cert, effect


# =============================================================================
# Pose-Twist Kinematic Consistency Factor (6.1.2 #3)
# =============================================================================


@dataclass
class PoseTwistConsistencyResult:
    """Result of pose-twist kinematic consistency factor."""
    L_consistency: jnp.ndarray  # (22, 22) information matrix contribution
    h_consistency: jnp.ndarray  # (22,) information vector contribution
    r_trans: jnp.ndarray  # (3,) translation residual
    r_rot: jnp.ndarray  # (3,) rotation residual


def pose_twist_kinematic_consistency(
    pose_prev: jnp.ndarray,  # (6,) previous pose [trans, rotvec] in world frame
    pose_curr: jnp.ndarray,  # (6,) current pose [trans, rotvec] in world frame
    v_body: jnp.ndarray,  # (3,) body-frame linear velocity from odom
    omega_body: jnp.ndarray,  # (3,) body-frame angular velocity from odom
    dt: float,  # Time between poses (seconds)
    Sigma_v: jnp.ndarray,  # (3,3) linear velocity covariance
    Sigma_omega: jnp.ndarray,  # (3,3) angular velocity covariance
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "",
) -> Tuple[PoseTwistConsistencyResult, CertBundle, ExpectedEffect]:
    """
    Pose-twist kinematic consistency factor.

    Enforces that pose change is consistent with reported twist:
        Log(X_prev^{-1} @ X_curr) ≈ [R_prev @ v_body * dt; omega_body * dt]

    This directly repairs the "pose snapshots with no dynamic linkage" issue
    identified in PIPELINE_DESIGN_GAPS.md section 6.1.2 #3.

    Key insight: If odom reports (pose, twist), they are NOT independent.
    The pose change should be consistent with integrated twist. This factor
    penalizes inconsistency, providing a soft kinematic constraint.

    Residuals (measurement - prediction):
        r_trans = (R_prev @ v_body * dt) - (t_curr - t_prev)
        r_rot = (omega_body * dt) - Log(R_prev^T @ R_curr)

    Information is scaled by 1/dt² (covariance scales with dt for integrated
    quantities) and by the twist covariances.

    Args:
        pose_prev: Previous pose [trans(3), rotvec(3)] in world frame
        pose_curr: Current pose [trans(3), rotvec(3)] in world frame
        v_body: Body-frame linear velocity from odom (3,)
        omega_body: Body-frame angular velocity from odom (3,)
        dt: Time between poses (seconds)
        Sigma_v: Linear velocity covariance (3,3)
        Sigma_omega: Angular velocity covariance (3,3)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        chart_id: Chart identifier
        anchor_id: Anchor identifier

    Returns:
        Tuple of (PoseTwistConsistencyResult, CertBundle, ExpectedEffect)
    """
    pose_prev = jnp.asarray(pose_prev, dtype=jnp.float64).reshape(-1)
    pose_curr = jnp.asarray(pose_curr, dtype=jnp.float64).reshape(-1)
    v_body = jnp.asarray(v_body, dtype=jnp.float64).reshape(-1)
    omega_body = jnp.asarray(omega_body, dtype=jnp.float64).reshape(-1)
    Sigma_v = jnp.asarray(Sigma_v, dtype=jnp.float64).reshape(3, 3)
    Sigma_omega = jnp.asarray(Sigma_omega, dtype=jnp.float64).reshape(3, 3)
    dt = float(dt)

    # Extract translation and rotation
    t_prev = pose_prev[:3]
    t_curr = pose_curr[:3]
    R_prev = se3_jax.so3_exp(pose_prev[3:6])
    R_curr = se3_jax.so3_exp(pose_curr[3:6])

    # Predicted pose change from twist integration
    # Translation: dp_pred = R_prev @ v_body * dt (body vel rotated to world, integrated)
    dp_pred = R_prev @ v_body * dt

    # Rotation: dR_pred = Exp(omega_body * dt), giving R_curr_pred = R_prev @ dR_pred
    dtheta_pred = omega_body * dt

    # Actual pose change
    dp_actual = t_curr - t_prev
    # Rotation change: R_prev^T @ R_curr = Exp(dtheta_actual)
    R_rel = R_prev.T @ R_curr
    dtheta_actual = se3_jax.so3_log(R_rel)

    # Residuals: measurement - actual (so MAP increment reduces the residual)
    r_trans = dp_pred - dp_actual  # (3,)
    r_rot = dtheta_pred - dtheta_actual  # (3,)

    # ==========================================================================
    # Covariance scaling: integrated twist covariance scales with dt²
    # ==========================================================================
    # If v ~ N(v_true, Sigma_v), then v*dt ~ N(v_true*dt, dt² * Sigma_v)
    # So the covariance of the integrated position/rotation is dt² * Sigma
    dt2 = dt * dt + eps_psd  # Add eps to avoid singularity at dt=0

    Sigma_trans = dt2 * Sigma_v
    Sigma_rot = dt2 * Sigma_omega

    # Get precision matrices
    Sigma_trans_psd = domain_projection_psd(Sigma_trans, eps_psd).M_psd
    Sigma_rot_psd = domain_projection_psd(Sigma_rot, eps_psd).M_psd

    L_trans_3x3, lift_trans = spd_cholesky_inverse_lifted(Sigma_trans_psd, eps_lift)
    L_rot_3x3, lift_rot = spd_cholesky_inverse_lifted(Sigma_rot_psd, eps_lift)

    # Build 22D information matrix
    # Translation at indices 0:3, rotation at indices 3:6
    L_consistency = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L_consistency = L_consistency.at[0:3, 0:3].set(L_trans_3x3)
    L_consistency = L_consistency.at[3:6, 3:6].set(L_rot_3x3)

    # Information vector
    h_consistency = jnp.zeros((D_Z,), dtype=jnp.float64)
    h_consistency = h_consistency.at[0:3].set(L_trans_3x3 @ r_trans)
    h_consistency = h_consistency.at[3:6].set(L_rot_3x3 @ r_rot)

    # NLL proxy
    nll_trans = 0.5 * float(r_trans @ L_trans_3x3 @ r_trans)
    nll_rot = 0.5 * float(r_rot @ L_rot_3x3 @ r_rot)
    nll_total = nll_trans + nll_rot

    # Conditioning certificate
    eigvals_trans = jnp.linalg.eigvalsh(Sigma_trans_psd)
    eigvals_rot = jnp.linalg.eigvalsh(Sigma_rot_psd)
    eig_min = float(jnp.minimum(jnp.min(eigvals_trans), jnp.min(eigvals_rot)))
    eig_max = float(jnp.maximum(jnp.max(eigvals_trans), jnp.max(eigvals_rot)))
    cond = eig_max / max(eig_min, 1e-18)

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["PoseTwistKinematicConsistency"],
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=0,
        ),
        influence=InfluenceCert(
            lift_strength=float(lift_trans + lift_rot),
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    effect = ExpectedEffect(
        objective_name="pose_twist_consistency_nll",
        predicted=nll_total,
        realized=None,
    )

    return PoseTwistConsistencyResult(
        L_consistency=L_consistency,
        h_consistency=h_consistency,
        r_trans=r_trans,
        r_rot=r_rot,
    ), cert, effect
