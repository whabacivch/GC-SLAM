"""
AnchorDriftUpdate operator for Geometric Compositional SLAM v2.

Continuous local chart maintenance using smooth rho function.
Replaces threshold-based anchor promotion with continuous blending.

Reference: docs/GC_SLAM.md Section 5.14
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    D_Z,
    SLICE_POSE,
    pose_z_to_se3_delta,
)
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_solve_lifted,
)
from fl_slam_poc.common.geometry import se3_jax


# =============================================================================
# Constants
# =============================================================================

# Anchor drift parameters (from spec)
M0 = constants.GC_ANCHOR_DRIFT_M0  # 0.5 meters
R0 = constants.GC_ANCHOR_DRIFT_R0  # 0.2 radians


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AnchorDriftResult:
    """Result of AnchorDriftUpdate operator."""
    rho: float  # Continuous drift blend factor in [0, 1]
    drift_m: float  # Position drift magnitude
    drift_r: float  # Rotation drift magnitude
    new_anchor_id: str  # Updated anchor id (may be same or new)


# =============================================================================
# Smooth Drift Function
# =============================================================================


def _compute_rho(drift_m: jnp.ndarray, drift_r: jnp.ndarray, M0: float, R0: float) -> jnp.ndarray:
    """
    Compute continuous drift blend factor.
    
    rho = max(drift_m / M0, drift_r / R0)
    
    Saturates at 1.0. Smoothly transitions based on drift magnitude.
    
    Args:
        drift_m: Position drift magnitude (meters)
        drift_r: Rotation drift magnitude (radians)
        M0: Position threshold
        R0: Rotation threshold
        
    Returns:
        rho in [0, 1]
    """
    rho_m = drift_m / M0
    rho_r = drift_r / R0
    rho_raw = jnp.maximum(rho_m, rho_r)
    rho = jnp.clip(rho_raw, 0.0, 1.0)
    return rho


# =============================================================================
# Main Operator
# =============================================================================


def anchor_drift_update(
    belief: BeliefGaussianInfo,
    eps_lift: float = constants.GC_EPS_LIFT,
    eps_psd: float = constants.GC_EPS_PSD,
) -> Tuple[AnchorDriftResult, BeliefGaussianInfo, CertBundle, ExpectedEffect]:
    """
    Continuous anchor drift update.
    
    Replaces threshold-based "if drift > M0: promote anchor" with
    continuous rho function that smoothly blends the update.
    
    When rho > 0, the anchor is partially updated toward the current
    pose estimate, with the degree controlled by rho.
    
    Args:
        belief: Current belief
        eps_lift: Solve lift epsilon
        eps_psd: PSD projection epsilon
        
    Returns:
        Tuple of (AnchorDriftResult, updated_belief, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.14
    """
    # Step 1: Compute current pose increment
    delta_z = belief.mean_increment(eps_lift)
    delta_pose = delta_z[SLICE_POSE]  # (6,)
    
    # Step 2: Compute drift magnitudes
    # GC ordering: [trans(0:3), rot(3:6)]
    # Position drift: ||delta_t||
    delta_trans = delta_pose[0:3]
    drift_m = float(jnp.linalg.norm(delta_trans))

    # Rotation drift: ||delta_omega||
    delta_rot = delta_pose[3:6]
    drift_r = float(jnp.linalg.norm(delta_rot))
    
    # Step 3: Compute rho (continuous)
    rho = _compute_rho(drift_m, drift_r, M0, R0)
    
    # Step 4: Blend anchor update
    # new_anchor = current_anchor ⊕ Exp(rho * delta_pose)
    # delta_pose is GC-ordered [trans, rot]; se3_jax uses same ordering!
    scaled_delta_z = rho * delta_pose
    scaled_delta_se3 = pose_z_to_se3_delta(scaled_delta_z)  # identity now
    delta_SE3 = se3_jax.se3_exp(scaled_delta_se3)
    X_anchor_new = se3_jax.se3_compose(belief.X_anchor, delta_SE3)
    
    # Step 5: Update linearization point
    # z_lin_new = (1 - rho) * delta_z
    # When rho = 1, we've absorbed all the increment into the anchor
    # When rho = 0, we keep the increment in z_lin
    z_lin_new = (1.0 - rho) * delta_z
    
    # Step 6: Update h to match new linearization
    h_new = belief.L @ z_lin_new
    
    # Deterministic anchor id update (branch-free)
    anchor_id_suffix = int(belief.stamp_sec * 1000) % 10000
    new_anchor_id = f"anchor_{anchor_id_suffix}"
    
    # Build result
    result = AnchorDriftResult(
        rho=rho,
        drift_m=drift_m,
        drift_r=drift_r,
        new_anchor_id=new_anchor_id,
    )
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=belief.chart_id,
        anchor_id=new_anchor_id,
        triggers=["AnchorDriftUpdate"],
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=rho,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    
    # Build updated belief
    belief_updated = BeliefGaussianInfo(
        chart_id=belief.chart_id,
        anchor_id=new_anchor_id,
        X_anchor=X_anchor_new,
        stamp_sec=belief.stamp_sec,
        z_lin=z_lin_new,
        L=belief.L,
        h=h_new,
        cert=cert,
    )
    
    expected_effect = ExpectedEffect(
        objective_name="anchor_drift_rho",
        predicted=rho,
        realized=None,
    )
    
    return result, belief_updated, cert, expected_effect
