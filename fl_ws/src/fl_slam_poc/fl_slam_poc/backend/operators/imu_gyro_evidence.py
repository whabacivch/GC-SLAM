"""
Gyro rotation evidence operator (Gaussian on SO(3)) for GC v2.

This is a unary factor on the scan-end orientation, using the scan-start
orientation (from the previous belief) as a known anchor plus IMU preintegrated
relative rotation over the scan window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import D_Z
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
class ImuGyroEvidenceResult:
    L_gyro: jnp.ndarray  # (22,22)
    h_gyro: jnp.ndarray  # (22,)
    r_rot: jnp.ndarray   # (3,) rotation residual


def imu_gyro_rotation_evidence(
    rotvec_start_WB: jnp.ndarray,       # (3,) scan-start orientation
    rotvec_end_pred_WB: jnp.ndarray,    # (3,) scan-end predicted orientation
    delta_rotvec_meas: jnp.ndarray,     # (3,) IMU-preintegrated relative rotvec over scan
    Sigma_g: jnp.ndarray,               # (3,3) gyro noise covariance proxy
    dt_int: float,                      # Sum of actual IMU sample intervals (bag-agnostic)
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[ImuGyroEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Build a Gaussian evidence term on the 22D tangent rotation block from gyro preintegration.

    We form a predicted scan-end orientation from the scan-start orientation and IMU delta:
      R_end_imu = R_start * Exp(delta_rotvec_meas)

    Residual on SO(3) (right-perturbation, measurement-target form):
      r = Log( R_end_pred^T * R_end_imu )

    Covariance on r is approximated as:
      Sigma_rot ≈ Sigma_g * dt_int
    where dt_int = Σ_i Δt_i over actual IMU sample intervals (bag-agnostic definition).
    
    Continuous mass check: if dt_int ≈ 0 (no samples), evidence is ~0 without boolean gates.
    """
    rotvec_start_WB = jnp.asarray(rotvec_start_WB, dtype=jnp.float64).reshape(-1)
    rotvec_end_pred_WB = jnp.asarray(rotvec_end_pred_WB, dtype=jnp.float64).reshape(-1)
    delta_rotvec_meas = jnp.asarray(delta_rotvec_meas, dtype=jnp.float64).reshape(-1)
    Sigma_g = jnp.asarray(Sigma_g, dtype=jnp.float64)

    R_start = se3_jax.so3_exp(rotvec_start_WB)
    R_delta = se3_jax.so3_exp(delta_rotvec_meas)
    R_end_imu = R_start @ R_delta
    R_end_pred = se3_jax.so3_exp(rotvec_end_pred_WB)

    # Residual must be pred^{-1} ∘ meas so that the canonical quadratic term
    # 0.5 (r - 0)^T Σ^{-1} (r - 0) drives R_end_pred toward R_end_imu.
    R_diff = R_end_pred.T @ R_end_imu
    r_rot = se3_jax.so3_log(R_diff)

    # dt_int is bag-agnostic: sum of actual IMU sample intervals.
    # Continuous mass check: when dt_int -> 0, evidence weight -> 0 (no boolean gates).
    eps_mass = constants.GC_EPS_MASS
    dt_pos = jnp.maximum(jnp.array(dt_int, dtype=jnp.float64), 0.0)
    dt_eff = dt_pos + eps_mass  # strictly-positive for PSD/inversion stability
    mass_scale = dt_pos / dt_eff

    Sigma_rot = Sigma_g * dt_eff
    Sigma_rot_psd = domain_projection_psd(Sigma_rot, eps_psd).M_psd
    L_rot, lift_strength = spd_cholesky_inverse_lifted(Sigma_rot_psd, eps_lift)
    
    # Diagnostic: check for NaN at each step
    import numpy as _np
    if not _np.all(_np.isfinite(_np.array(R_diff))):
        raise ValueError(f"R_diff has NaN: {_np.array(R_diff)}")
    if not _np.all(_np.isfinite(_np.array(r_rot))):
        raise ValueError(f"r_rot has NaN from so3_log. R_diff trace={float(jnp.trace(R_diff)):.6f}")
    if not _np.all(_np.isfinite(_np.array(L_rot))):
        raise ValueError(f"L_rot has NaN. Sigma_rot diag={_np.diag(_np.array(Sigma_rot))}, dt_eff={float(dt_eff)}")
    
    # Scale evidence by continuous mass (branch-free)
    L_rot_scaled = mass_scale * L_rot

    L = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    # GC ordering: [trans(0:3), rot(3:6)] - rotation evidence goes to [3:6] block
    L = L.at[3:6, 3:6].set(L_rot_scaled)
    h = jnp.zeros((D_Z,), dtype=jnp.float64)
    h = h.at[3:6].set(L_rot_scaled @ r_rot)

    nll_proxy = 0.5 * float(r_rot @ L_rot @ r_rot)

    eigvals = jnp.linalg.eigvalsh(domain_projection_psd(L_rot, eps_psd).M_psd)
    eig_min = float(jnp.min(eigvals))
    eig_max = float(jnp.max(eigvals))
    cond = eig_max / max(eig_min, 1e-18)

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ImuGyroRotationGaussian"],
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=int(jnp.sum(eigvals < eps_psd)),
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
        objective_name="imu_gyro_rotation_nll_proxy",
        predicted=nll_proxy,
        realized=None,
    )

    return ImuGyroEvidenceResult(L_gyro=L, h_gyro=h, r_rot=r_rot), cert, effect
