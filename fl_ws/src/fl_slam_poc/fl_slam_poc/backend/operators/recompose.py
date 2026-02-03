"""
PoseUpdateFrobeniusRecompose operator for Geometric Compositional SLAM v2.

Recompose pose with continuous Frobenius strength-blended BCH3 correction.

Reference: docs/GC_SLAM.md Section 5.12
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
    InfluenceCert,
)
from fl_slam_poc.common.primitives import spd_cholesky_solve_lifted
from fl_slam_poc.common.geometry import se3_jax


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class RecomposeResult:
    """Result of PoseUpdateFrobeniusRecompose operator."""
    delta_pose: jnp.ndarray  # (6,) pose increment
    X_new: jnp.ndarray  # (6,) new world pose
    frobenius_strength: float  # Continuous Frobenius correction strength
    bch_correction: jnp.ndarray  # (6,) BCH3 correction term


# =============================================================================
# BCH3 Correction
# =============================================================================


@jax.jit
def _bch3_correction(xi1: jnp.ndarray, xi2: jnp.ndarray) -> jnp.ndarray:
    """
    Baker-Campbell-Hausdorff third-order correction.
    
    BCH formula: log(exp(xi1) * exp(xi2)) ≈ xi1 + xi2 + 0.5*[xi1, xi2] + ...
    
    For SE(3), the Lie bracket [xi1, xi2] = ad(xi1) @ xi2
    
    Args:
        xi1: First tangent vector (6,)
        xi2: Second tangent vector (6,)
        
    Returns:
        BCH3 correction term (6,)
    """
    # Extract translation and rotation parts
    # xi = [translation(3), rotation(3)] (GC ordering matches se3_jax)
    v1 = xi1[:3]
    omega1 = xi1[3:6]
    v2 = xi2[:3]
    omega2 = xi2[3:6]
    
    # Compute adjoint action ad(xi1) @ xi2
    # For SE(3) with xi = [v, omega]: ad([v, omega]) in [omega, v] ordering is
    # [[omega]x, 0; [v]x, [omega]x]. We compute in [v, omega] and return [v, omega].
    
    # Cross products for rotation part
    omega_cross = jnp.cross(omega1, omega2)
    
    # Cross products for translation part
    v_cross = jnp.cross(omega1, v2) + jnp.cross(v1, omega2)
    
    # BCH3 second-order term: 0.5 * [xi1, xi2] in [trans, rot] ordering
    correction = 0.5 * jnp.concatenate([v_cross, omega_cross])
    
    return correction


# =============================================================================
# Main Operator
# =============================================================================


def pose_update_frobenius_recompose(
    belief_post: BeliefGaussianInfo,
    total_trigger_magnitude: float,
    c_frob: float = constants.GC_C_FROB,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> Tuple[RecomposeResult, BeliefGaussianInfo, CertBundle, ExpectedEffect]:
    """
    Recompose pose with continuous Frobenius strength-blended BCH3 correction.
    
    The Frobenius correction strength is:
        s = total_trigger_magnitude / (total_trigger_magnitude + c_frob)
    
    When triggers are large, correction is strong.
    When triggers are small/zero, correction is weak/zero.
    
    Args:
        belief_post: Posterior belief after fusion
        total_trigger_magnitude: Sum of all trigger magnitudes
        c_frob: Frobenius correction coupling constant
        eps_lift: Solve lift epsilon
        
    Returns:
        Tuple of (RecomposeResult, updated_belief, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.12
    """
    total_trigger_magnitude = float(total_trigger_magnitude)
    
    # Step 1: Compute MAP increment
    delta_z = belief_post.mean_increment(eps_lift)
    delta_pose_z = delta_z[SLICE_POSE]  # (6,) GC ordering: [trans, rot]
    
    # Step 2: Compute Frobenius strength (continuous)
    frobenius_strength = total_trigger_magnitude / (total_trigger_magnitude + c_frob)
    
    # Step 3: Compute BCH correction in GC pose ordering [trans, rot].
    #
    # NOTE: belief_post.X_anchor is an SE(3) element encoded as [trans, rot] for se3_jax,
    # which now matches GC ordering.
    #
    # We use the current pose linearization offset (in-chart) as the second term.
    xi_lin_z = belief_post.z_lin[SLICE_POSE]
    bch_correction = _bch3_correction(xi_lin_z, delta_pose_z)
    
    # Step 4: Apply blended correction
    # delta_pose_corrected = delta_pose + s * bch_correction
    delta_pose_corrected_z = delta_pose_z + frobenius_strength * bch_correction
    
    # Step 5: Compose with anchor to get new world pose
    # X_new = X_anchor ∘ Exp(delta_pose_corrected)
    # GC ordering now matches se3_jax - no conversion needed (identity).
    delta_pose_corrected_se3 = pose_z_to_se3_delta(delta_pose_corrected_z)  # identity
    delta_SE3 = se3_jax.se3_exp(delta_pose_corrected_se3)
    X_new = se3_jax.se3_compose(belief_post.X_anchor, delta_SE3)
    
    # Build result
    result = RecomposeResult(
        delta_pose=delta_pose_corrected_z,
        X_new=X_new,
        frobenius_strength=frobenius_strength,
        bch_correction=bch_correction,
    )
    
    # Update belief by shifting the chart origin in tangent space.
    #
    # Treat the recompose as a change-of-variables:
    #   z' = z - shift, where shift applies only to the pose slice.
    #
    # For a Gaussian in information form (L, h) with mean mu = L^{-1} h:
    #   mu' = mu - shift
    #   h'  = L @ mu' = h - L @ shift
    #
    # This preserves non-pose state components (v/bias/dt/extrinsic) instead of
    # implicitly zeroing them.
    shift_z = jnp.zeros(D_Z, dtype=jnp.float64).at[SLICE_POSE].set(delta_pose_corrected_z)
    z_lin_new = belief_post.z_lin - shift_z
    h_new = belief_post.h - belief_post.L @ shift_z
    
    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=belief_post.chart_id,
        anchor_id=belief_post.anchor_id,
        triggers=["PoseUpdateFrobeniusRecompose"],
        frobenius_applied=frobenius_strength > 0.0,
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
    
    # Update the certificate's frobenius field based on strength
    eps_frob = float(jnp.finfo(jnp.float64).eps)
    cert.frobenius_applied = frobenius_strength > eps_frob
    
    belief_updated = BeliefGaussianInfo(
        chart_id=belief_post.chart_id,
        anchor_id=belief_post.anchor_id,
        X_anchor=X_new,  # Updated anchor
        stamp_sec=belief_post.stamp_sec,
        z_lin=z_lin_new,
        L=belief_post.L,
        h=h_new,
        cert=cert,
    )
    
    # Expected effect: pose increment magnitude
    pose_magnitude = float(jnp.linalg.norm(delta_pose_corrected_z))
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_pose_increment_magnitude",
        predicted=pose_magnitude,
        realized=None,
    )
    
    return result, belief_updated, cert, expected_effect
