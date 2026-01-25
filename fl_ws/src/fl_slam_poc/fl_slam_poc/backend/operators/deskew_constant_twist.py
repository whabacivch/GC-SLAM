"""
Constant-twist deskew operator for GC v2.

Given a body twist xi over the scan interval, deskew points to the scan start frame:
  T(alpha) = Exp(alpha * xi)
  p0 = T(alpha)^{-1} ⊙ p

No sigma points / moment matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import CertBundle, ExpectedEffect, InfluenceCert, SupportCert
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.backend.operators.imu_preintegration import smooth_window_weights


@dataclass
class DeskewConstantTwistResult:
    points: jnp.ndarray       # (N,3)
    timestamps: jnp.ndarray   # (N,)
    weights: jnp.ndarray      # (N,)
    ess_imu: float            # IMU ESS proxy used to form xi


@jax.jit
def _deskew_points_constant_twist_jax(
    points: jnp.ndarray,        # (N,3)
    timestamps: jnp.ndarray,    # (N,)
    weights: jnp.ndarray,       # (N,)
    scan_start_time: float,
    scan_end_time: float,
    xi_body: jnp.ndarray,       # (6,) [trans, rotvec] over full interval
) -> tuple[jnp.ndarray, jnp.ndarray]:
    points = jnp.asarray(points, dtype=jnp.float64)
    timestamps = jnp.asarray(timestamps, dtype=jnp.float64).reshape(-1)
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)
    xi_body = jnp.asarray(xi_body, dtype=jnp.float64).reshape(-1)

    t0 = jnp.array(scan_start_time, dtype=jnp.float64)
    t1 = jnp.array(scan_end_time, dtype=jnp.float64)
    denom = jnp.maximum(t1 - t0, 1e-12)
    alpha = (timestamps - t0) / denom  # no hard [0,1] clipping; weights handle membership

    def one_point(p, a):
        T_a = se3_jax.se3_exp(a * xi_body)  # (6,) [t, rotvec]
        t = T_a[:3]
        rotvec = T_a[3:6]
        R = se3_jax.so3_exp(rotvec)
        # Apply inverse: p0 = R^T (p - t)
        return R.T @ (p - t)

    points_out = jax.vmap(one_point)(points, alpha)

    # Soft membership kernel in time (no hard window boundaries).
    sigma = jnp.array(constants.GC_TIME_WARP_SIGMA_FRAC, dtype=jnp.float64) * denom
    w_time = smooth_window_weights(
        imu_stamps=timestamps,  # same functional form for points
        scan_start_time=scan_start_time,
        scan_end_time=scan_end_time,
        sigma=sigma,
    )
    weights_out = jnp.asarray(weights, dtype=jnp.float64).reshape(-1) * w_time
    return points_out, weights_out


def deskew_constant_twist(
    points: jnp.ndarray,
    timestamps: jnp.ndarray,
    weights: jnp.ndarray,
    scan_start_time: float,
    scan_end_time: float,
    xi_body: jnp.ndarray,
    ess_imu: float,
    chart_id: str,
    anchor_id: str,
) -> Tuple[DeskewConstantTwistResult, CertBundle, ExpectedEffect]:
    """
    Deskew points with a constant body twist model (fixed cost).
    """
    points_out, weights_out = _deskew_points_constant_twist_jax(
        points=points,
        timestamps=timestamps,
        weights=weights,
        scan_start_time=scan_start_time,
        scan_end_time=scan_end_time,
        xi_body=xi_body,
    )

    result = DeskewConstantTwistResult(
        points=points_out,
        timestamps=jnp.asarray(timestamps, dtype=jnp.float64),
        weights=weights_out,
        ess_imu=float(ess_imu),
    )

    w_in = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)
    retained = float(jnp.sum(weights_out) / (jnp.sum(w_in) + constants.GC_EPS_MASS))
    cert = CertBundle.create_exact(
        chart_id=chart_id,
        anchor_id=anchor_id,
        support=SupportCert(ess_total=float(ess_imu), support_frac=retained),
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
        objective_name="deskew_variance_reduction_proxy",
        predicted=0.0,
        realized=None,
    )

    return result, cert, effect
