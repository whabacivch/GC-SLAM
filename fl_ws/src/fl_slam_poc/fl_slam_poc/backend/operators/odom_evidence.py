"""
Odometry evidence operator (Gaussian) for GC v2.

Constructs a quadratic evidence term on the 22D tangent by comparing the belief's
current world pose to an odometry pose measurement with covariance.

This is a pure Gaussian factor in SE(3) tangent coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import D_Z, pose_se3_to_z_delta
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    InfluenceCert,
    MismatchCert,
)
from fl_slam_poc.common.primitives import domain_projection_psd, spd_cholesky_inverse_lifted
from fl_slam_poc.common.geometry import se3_jax


@dataclass
class OdomEvidenceResult:
    L_odom: jnp.ndarray  # (22,22)
    h_odom: jnp.ndarray  # (22,)
    delta_z_star: jnp.ndarray  # (22,)


def odom_quadratic_evidence(
    belief_pred_pose: jnp.ndarray,   # (6,) [trans, rotvec]
    odom_pose: jnp.ndarray,          # (6,) [trans, rotvec]
    odom_cov_se3: jnp.ndarray,       # (6,6) covariance in ROS [x,y,z,roll,pitch,yaw] order
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "",
) -> Tuple[OdomEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Build Gaussian evidence from an odometry pose observation.

    We compute the right-invariant-ish pose error in se(3) via:
      T_err = T_pred^{-1} ∘ T_odom
      xi_err = Log(T_err)   (6,) [rho, phi] in se3_jax ordering

    Then embed into the 22D chart pose slice ([trans, rot]) and build:
      L = Σ^{-1}
      h = L * delta_z_star

    NOTE: ROS odom covariance is in [x,y,z,roll,pitch,yaw] = [trans,rot] order,
    which now matches GC tangent ordering. No permutation needed!
    """
    belief_pred_pose = jnp.asarray(belief_pred_pose, dtype=jnp.float64).reshape(-1)
    odom_pose = jnp.asarray(odom_pose, dtype=jnp.float64).reshape(-1)
    cov_ros = jnp.asarray(odom_cov_se3, dtype=jnp.float64)

    # Pose error as a twist in se3 ordering
    T_err = se3_jax.se3_relative(odom_pose, belief_pred_pose)  # belief^{-1} ∘ odom
    xi_err = se3_jax.se3_log(T_err)  # [rho, phi]

    # Map to 22D pose slice ordering [trans, rot] - same as se3_jax, no conversion needed!
    delta_pose_z = pose_se3_to_z_delta(xi_err)  # identity - [trans, rot]
    delta_z_star = jnp.zeros((D_Z,), dtype=jnp.float64)
    delta_z_star = delta_z_star.at[0:6].set(delta_pose_z)

    # ROS pose covariance: [x, y, z, roll, pitch, yaw] = [trans(0:3), rot(3:6)]
    # GC pose ordering:    [tx, ty, tz, rx, ry, rz]    = [trans(0:3), rot(3:6)]
    # No permutation needed - orderings now match!
    cov = cov_ros

    cov_psd = domain_projection_psd(cov, eps_psd).M_psd
    L_pose, lift_strength = spd_cholesky_inverse_lifted(cov_psd, eps_lift)

    L = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L = L.at[0:6, 0:6].set(L_pose)
    h = L @ delta_z_star

    nll_proxy = 0.5 * float(delta_pose_z @ L_pose @ delta_pose_z)

    eigvals = jnp.linalg.eigvalsh(domain_projection_psd(L_pose, eps_psd).M_psd)
    eig_min = float(jnp.min(eigvals))
    eig_max = float(jnp.max(eigvals))
    cond = eig_max / max(eig_min, 1e-18)

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["OdomEvidenceGaussian"],
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=int(jnp.sum(eigvals < 1e-12)),
        ),
        mismatch=MismatchCert(nll_per_ess=nll_proxy, directional_score=0.0),
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
        objective_name="odom_quadratic_nll_proxy",
        predicted=nll_proxy,
        realized=None,
    )

    return OdomEvidenceResult(L_odom=L, h_odom=h, delta_z_star=delta_z_star), cert, effect

