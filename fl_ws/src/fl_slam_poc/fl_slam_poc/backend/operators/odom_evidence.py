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
from fl_slam_poc.common.primitives import (
    domain_projection_psd_core,
    spd_cholesky_inverse_lifted_core,
)
from fl_slam_poc.common.geometry import se3_jax


@dataclass
class OdomEvidenceResult:
    L_odom: jnp.ndarray  # (22,22)
    h_odom: jnp.ndarray  # (22,)
    delta_z_star: jnp.ndarray  # (22,)


@jax.jit
def _odom_quadratic_evidence_core(
    belief_pred_pose: jnp.ndarray,
    odom_pose: jnp.ndarray,
    cov_ros: jnp.ndarray,
    eps_psd: jnp.ndarray,
    eps_lift: jnp.ndarray,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
]:
    """
    JIT'd core: returns (L_odom, h_odom, delta_z_star, lift_strength, nll_proxy,
    eig_min, eig_max, cond, near_null_count) for wrapper to build cert.
    """
    T_err = se3_jax.se3_relative(odom_pose, belief_pred_pose)
    xi_err = se3_jax.se3_log(T_err)
    delta_pose_z = pose_se3_to_z_delta(xi_err)
    delta_z_star = jnp.zeros((D_Z,), dtype=jnp.float64).at[0:6].set(delta_pose_z)

    cov_psd, _ = domain_projection_psd_core(cov_ros, eps_psd)
    L_pose, lift_strength = spd_cholesky_inverse_lifted_core(cov_psd, eps_lift)

    L = jnp.zeros((D_Z, D_Z), dtype=jnp.float64).at[0:6, 0:6].set(L_pose)
    h = L @ delta_z_star

    nll_proxy = 0.5 * (delta_pose_z @ L_pose @ delta_pose_z)

    L_pose_psd, cert_vec = domain_projection_psd_core(L_pose, eps_psd)
    eigvals = jnp.linalg.eigvalsh(L_pose_psd)
    eig_min = jnp.min(eigvals)
    eig_max = jnp.max(eigvals)
    cond = eig_max / jnp.maximum(eig_min, 1e-18)
    near_null_count = jnp.sum(eigvals < 1e-12).astype(jnp.int32)

    return (
        L,
        h,
        delta_z_star,
        lift_strength,
        nll_proxy,
        eig_min,
        eig_max,
        cond,
        near_null_count,
    )


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

    (
        L,
        h,
        delta_z_star,
        lift_strength,
        nll_proxy,
        eig_min,
        eig_max,
        cond,
        near_null_count,
    ) = _odom_quadratic_evidence_core(
        belief_pred_pose,
        odom_pose,
        cov_ros,
        jnp.array(eps_psd, dtype=jnp.float64),
        jnp.array(eps_lift, dtype=jnp.float64),
    )

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["OdomEvidenceGaussian"],
        conditioning=ConditioningCert(
            eig_min=float(eig_min),
            eig_max=float(eig_max),
            cond=float(cond),
            near_null_count=int(near_null_count),
        ),
        mismatch=MismatchCert(nll_per_ess=float(nll_proxy), directional_score=0.0),
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
        predicted=float(nll_proxy),
        realized=None,
    )

    return OdomEvidenceResult(L_odom=L, h_odom=h, delta_z_star=delta_z_star), cert, effect

