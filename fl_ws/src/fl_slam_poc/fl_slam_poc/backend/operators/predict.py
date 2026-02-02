"""
PredictDiffusion operator for Geometric Compositional SLAM v2.

OU-style bounded propagation with continuous DomainProjectionPSD.

Replaces pure diffusion (Σ ← Σ + Q*dt) with Ornstein-Uhlenbeck mean-reverting
propagation to prevent unbounded uncertainty growth during missing-data gaps.

For A = -λI, the closed-form OU propagation is:
  Σ(t+Δt) = e^(-2λΔt) Σ(t) + (1 - e^(-2λΔt))/(2λ) Q

This is continuous, smooth, and bounded: as Δt → ∞, Σ → Q/(2λ) (not ∞).

Reference: docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md Section 5.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import BeliefGaussianInfo, D_Z
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd_core,
    spd_cholesky_inverse_lifted_core,
    spd_cholesky_solve_lifted_core,
)


# =============================================================================
# JIT'd core (arrays in/out; wrapper builds BeliefGaussianInfo and CertBundle)
# =============================================================================


@jax.jit
def _predict_diffusion_core(
    L_prev: jnp.ndarray,
    h_prev: jnp.ndarray,
    z_lin: jnp.ndarray,
    Q: jnp.ndarray,
    dt_sec: jnp.ndarray,
    eps_psd: jnp.ndarray,
    eps_lift: jnp.ndarray,
    lambda_ou: jnp.ndarray,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray,
]:
    """
    OU-style bounded propagation: (L_prev, h_prev) -> (L_pred, h_pred).
    Returns (L_pred, h_pred, z_lin, lift_prev, lift_inv, total_projection_delta,
             eig_min, eig_max, cond, near_null_count, trace_cov_pred) for wrapper.
    """
    # Mean and covariance from previous belief (lifted solve/inverse)
    mean_prev, _ = spd_cholesky_solve_lifted_core(L_prev, h_prev, eps_lift)
    cov_prev, lift_prev = spd_cholesky_inverse_lifted_core(L_prev, eps_lift)

    # OU propagation: Σ' = e^(-2λΔt) Σ + (1 - e^(-2λΔt))/(2λ) Q
    exp_factor = jnp.exp(-2.0 * lambda_ou * dt_sec)
    diffusion_coeff = (1.0 - exp_factor) / (2.0 * lambda_ou + jnp.finfo(jnp.float64).eps)
    cov_pred_raw = exp_factor * cov_prev + diffusion_coeff * Q

    # PSD project predicted covariance
    cov_pred_psd, cert_cov = domain_projection_psd_core(cov_pred_raw, eps_psd)

    # Back to information form: L_pred = inv(cov_pred_psd)
    L_pred, lift_inv = spd_cholesky_inverse_lifted_core(cov_pred_psd, eps_lift)

    # PSD project L_pred
    L_pred_psd, cert_L = domain_projection_psd_core(L_pred, eps_psd)

    h_pred = L_pred_psd @ mean_prev
    total_projection_delta = cert_cov[0] + cert_L[0]
    trace_cov_pred = jnp.trace(cov_pred_psd)

    return (
        L_pred_psd,
        h_pred,
        z_lin,
        lift_prev,
        lift_inv,
        total_projection_delta,
        cert_L[2],   # eig_min
        cert_L[3],   # eig_max
        cert_L[4],   # cond
        cert_L[5],   # near_null_count
        trace_cov_pred,
    )


# =============================================================================
# Main Operator
# =============================================================================


def predict_diffusion(
    belief_prev: BeliefGaussianInfo,
    Q: jnp.ndarray,
    dt_sec: float,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    lambda_ou: float = constants.GC_OU_DAMPING_LAMBDA,
) -> Tuple[BeliefGaussianInfo, CertBundle, ExpectedEffect]:
    """
    Predict belief forward with OU-style bounded propagation.
    
    Uses Ornstein-Uhlenbeck mean-reverting diffusion instead of pure diffusion
    to prevent unbounded uncertainty growth during missing-data gaps.
    
    For A = -λI, the closed-form propagation is:
      Σ(t+Δt) = e^(-2λΔt) Σ(t) + (1 - e^(-2λΔt))/(2λ) Q
    
    This is continuous, smooth, and bounded: as Δt → ∞, Σ → Q/(2λ).
    For small Δt, it approximates pure diffusion: Σ ≈ Σ + Q*Δt.
    
    Always applies DomainProjectionPSD (even if Q is zero).
    
    Prediction in information form:
    1. Convert to moment form
    2. Apply OU propagation (bounded, continuous)
    3. Convert back to information form
    4. Apply DomainProjectionPSD
    
    Args:
        belief_prev: Previous belief
        Q: Process noise matrix (D_Z, D_Z)
        dt_sec: Time delta in seconds (may be large for missing-data gaps)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        lambda_ou: OU damping rate (1/s); larger = faster saturation
        
    Returns:
        Tuple of (predicted_belief, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.2
    """
    Q = jnp.asarray(Q, dtype=jnp.float64)
    dt_sec_val = float(dt_sec)
    lambda_ou_val = float(lambda_ou)

    # JIT'd core: arrays in, arrays out; cert scalars pulled once below
    L_prev = belief_prev.L
    h_prev = belief_prev.h
    z_lin = belief_prev.z_lin
    (
        L_pred_psd,
        h_pred,
        z_lin_out,
        lift_prev,
        lift_inv,
        total_projection_delta,
        eig_min,
        eig_max,
        cond,
        near_null_count,
        trace_cov_pred,
    ) = _predict_diffusion_core(
        L_prev,
        h_prev,
        z_lin,
        Q,
        jnp.array(dt_sec_val, dtype=jnp.float64),
        jnp.array(eps_psd, dtype=jnp.float64),
        jnp.array(eps_lift, dtype=jnp.float64),
        jnp.array(lambda_ou_val, dtype=jnp.float64),
    )

    # Build certificate from scalars (single host pull after JIT return)
    cert = CertBundle.create_approx(
        chart_id=belief_prev.chart_id,
        anchor_id=belief_prev.anchor_id,
        triggers=["PredictDiffusion"],
        conditioning=ConditioningCert(
            eig_min=float(eig_min),
            eig_max=float(eig_max),
            cond=float(cond),
            near_null_count=int(near_null_count),
        ),
        influence=InfluenceCert(
            lift_strength=float(lift_prev) + float(lift_inv),
            psd_projection_delta=float(total_projection_delta),
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=dt_sec_val,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    belief_pred = BeliefGaussianInfo(
        chart_id=belief_prev.chart_id,
        anchor_id=belief_prev.anchor_id,
        X_anchor=belief_prev.X_anchor,
        stamp_sec=belief_prev.stamp_sec + dt_sec_val,
        z_lin=z_lin_out,
        L=L_pred_psd,
        h=h_pred,
        cert=cert,
    )

    # Expected effect: predicted cov trace (from core; no inv in wrapper)
    expected_effect = ExpectedEffect(
        objective_name="predicted_cov_trace",
        predicted=float(trace_cov_pred),
        realized=None,
    )

    return belief_pred, cert, expected_effect
