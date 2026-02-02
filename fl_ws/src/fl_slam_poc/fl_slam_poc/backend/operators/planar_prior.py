"""
Planar robot priors for Geometric Compositional SLAM v2.

Soft constraints for ground-hugging robots:
1. z = z_ref (robot height stays constant)
2. vel_z = 0 (robot doesn't fly)

These priors address the z-drift problem where LiDAR+map feedback causes
the z estimate to diverge to -50 to -80m on planar robots.

Reference: Plan Phase 1 (z fix via soft prior)
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
    InfluenceCert,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PlanarPriorResult:
    """Result of planar z prior operator."""
    L_planar: jnp.ndarray  # (22, 22) information matrix contribution
    h_planar: jnp.ndarray  # (22,) information vector contribution
    r_z: float  # z residual (belief_z - z_ref)


@dataclass
class VelocityZPriorResult:
    """Result of velocity z prior operator."""
    L_vz: jnp.ndarray  # (22, 22) information matrix contribution
    h_vz: jnp.ndarray  # (22,) information vector contribution
    v_z: float  # z velocity value


# =============================================================================
# Planar Z Prior
# =============================================================================


def planar_z_prior(
    belief_pred_pose: jnp.ndarray,  # (6,) [trans, rotvec]
    z_ref: float,  # Reference world Z for base/body frame (e.g., 0.0m for base_footprint)
    sigma_z: float,  # Soft constraint std dev (e.g., 0.1m)
    eps_psd: float = constants.GC_EPS_PSD,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "",
) -> Tuple[PlanarPriorResult, CertBundle, ExpectedEffect]:
    """
    Soft planar constraint: z = z_ref with variance sigma_z^2.

    This injects information into L[2,2] (z translation block) to
    pull the estimate toward the reference height.

    For a planar robot, z_ref should be the known robot height.
    sigma_z controls how strongly to enforce (smaller = stronger).

    The evidence form is:
        -log p(z | z_ref) = (z - z_ref)^2 / (2 * sigma_z^2)

    In information form:
        L[2,2] = 1 / sigma_z^2  (precision on z)
        h[2] = z_ref / sigma_z^2  (pulling toward z_ref)

    Args:
        belief_pred_pose: Predicted pose (6,) [tx, ty, tz, rx, ry, rz]
        z_ref: Reference z height (meters)
        sigma_z: Soft constraint std dev (meters)
        eps_psd: PSD epsilon (unused, for interface consistency)
        chart_id: Chart identifier
        anchor_id: Anchor identifier

    Returns:
        Tuple of (PlanarPriorResult, CertBundle, ExpectedEffect)
    """
    belief_pred_pose = jnp.asarray(belief_pred_pose, dtype=jnp.float64).reshape(-1)

    # Extract z from pose (index 2 in [tx, ty, tz, rx, ry, rz]).
    z_pred = belief_pred_pose[2]
    # Residual is measurement - prediction so the MAP increment moves toward z_ref.
    r_z = float(z_ref - z_pred)

    # Information precision on z
    precision_z = 1.0 / (sigma_z ** 2)

    # Build 22D information matrix with z constraint
    # State ordering: [trans(0:3), rot(3:6), vel(6:9), bg(9:12), ba(12:15), dt(15:16), ex(16:22)]
    # z is at index 2
    L_planar = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L_planar = L_planar.at[2, 2].set(precision_z)

    # Information vector: h = L @ delta, where delta is the desired increment.
    # For this unary prior, the desired pose increment is delta_z = (z_ref - z_pred),
    # so h[2] = precision_z * (z_ref - z_pred).
    h_planar = jnp.zeros((D_Z,), dtype=jnp.float64)
    h_planar = h_planar.at[2].set(precision_z * r_z)

    # NLL proxy for diagnostics
    nll_proxy = 0.5 * r_z ** 2 * precision_z

    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["PlanarZPrior"],
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
        objective_name="planar_z_prior_nll",
        predicted=nll_proxy,
        realized=None,
    )

    return PlanarPriorResult(L_planar=L_planar, h_planar=h_planar, r_z=r_z), cert, effect


# =============================================================================
# Velocity Z Prior
# =============================================================================


def velocity_z_prior(
    v_z_pred: float,  # Predicted z velocity
    sigma_vz: float = 0.01,  # Very tight - robot doesn't fly (m/s)
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "",
) -> Tuple[VelocityZPriorResult, CertBundle, ExpectedEffect]:
    """
    Soft constraint: vel_z = 0 with variance sigma_vz^2.

    Injects into L[8,8] (z velocity block) to prevent vertical velocity drift.
    For a ground robot, vertical velocity should be essentially zero.

    State ordering: [trans(0:3), rot(3:6), vel(6:9), bg(9:12), ba(12:15), dt(15:16), ex(16:22)]
    vel_z is at index 8 (vel block starts at 6, z is third component)

    Args:
        v_z_pred: Predicted z velocity (m/s)
        sigma_vz: Soft constraint std dev (m/s), default 0.01 = 1cm/s
        chart_id: Chart identifier
        anchor_id: Anchor identifier

    Returns:
        Tuple of (VelocityZPriorResult, CertBundle, ExpectedEffect)
    """
    v_z_pred = float(v_z_pred)
    # Residual is measurement - prediction so the MAP increment moves toward 0.
    r_vz = -v_z_pred

    # Information precision on vel_z
    precision_vz = 1.0 / (sigma_vz ** 2)

    # Build 22D information matrix with vel_z constraint
    L_vz = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L_vz = L_vz.at[8, 8].set(precision_vz)

    # Information vector: h[8] = precision * (0 - v_z_pred) (pulling toward 0)
    h_vz = jnp.zeros((D_Z,), dtype=jnp.float64)
    h_vz = h_vz.at[8].set(precision_vz * r_vz)

    # NLL proxy
    nll_proxy = 0.5 * r_vz ** 2 * precision_vz

    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["VelocityZPrior"],
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
        objective_name="velocity_z_prior_nll",
        predicted=nll_proxy,
        realized=None,
    )

    return VelocityZPriorResult(L_vz=L_vz, h_vz=h_vz, v_z=v_z_pred), cert, effect
