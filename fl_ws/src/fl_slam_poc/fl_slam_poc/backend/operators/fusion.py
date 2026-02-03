"""
Fusion operators for Geometric Compositional SLAM v2.

FusionScaleFromCertificates and InfoFusionAdditive.

Reference: docs/GC_SLAM.md Sections 5.10, 5.11
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
    OverconfidenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    clamp,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class FusionScaleResult:
    """Result of FusionScaleFromCertificates operator."""
    alpha: float  # Continuous fusion scale in [alpha_min, alpha_max]


# =============================================================================
# FusionScaleFromCertificates Operator
# =============================================================================


def fusion_scale_from_certificates(
    cert_evidence: CertBundle,
    cert_belief: CertBundle,
    alpha_min: float = constants.GC_ALPHA_MIN,
    alpha_max: float = constants.GC_ALPHA_MAX,
    kappa_scale: float = constants.GC_KAPPA_SCALE,
    c0_cond: float = constants.GC_C0_COND,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[FusionScaleResult, CertBundle, ExpectedEffect]:
    """
    Compute continuous fusion scale from certificates.
    
    alpha = alpha_min + (alpha_max - alpha_min) * f(quality)
    
    where f combines conditioning, support, and mismatch metrics.
    
    Args:
        cert_evidence: Certificate from evidence extraction
        cert_belief: Certificate from belief prediction
        alpha_min: Minimum fusion scale
        alpha_max: Maximum fusion scale
        kappa_scale: Scale for conditioning contribution
        c0_cond: Conditioning baseline
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (FusionScaleResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.10
    """
    # Extract quality metrics from certificates
    cond_evidence = cert_evidence.conditioning.cond
    ess_evidence = cert_evidence.support.ess_total
    support_frac = cert_evidence.support.support_frac
    excitation_total = cert_evidence.excitation.dt_effect + cert_evidence.excitation.extrinsic_effect
    dt_asymmetry = cert_evidence.overconfidence.dt_asymmetry
    z_to_xy_ratio = cert_evidence.overconfidence.z_to_xy_ratio
    power_beta = cert_evidence.influence.power_beta
    nll_per_ess = cert_evidence.mismatch.nll_per_ess
    
    # Conditioning quality: lower condition number is better
    cond_quality = c0_cond / (cond_evidence + c0_cond)
    
    # Support quality: higher ESS is better (normalize by expected max)
    support_quality = ess_evidence / (ess_evidence + 1.0)

    # Mismatch quality: smaller nll_per_ess => closer to 1.0. Default nll_per_ess=0 => no penalty.
    mismatch_quality = jnp.exp(-jnp.asarray(nll_per_ess, dtype=jnp.float64))

    # Observability qualities (bounded in [0,1] when present):
    dt_quality = jnp.clip(jnp.asarray(dt_asymmetry, dtype=jnp.float64), 0.0, 1.0)
    # z_to_xy_ratio is unbounded; map to [0,1] via saturation.
    z_quality = jnp.asarray(z_to_xy_ratio, dtype=jnp.float64) / (jnp.asarray(z_to_xy_ratio, dtype=jnp.float64) + 1.0)
    z_quality = jnp.clip(z_quality, 0.0, 1.0)
    # Excitation proxy: more excitation_total => safer. Map to [0,1) via saturation.
    exc_quality = jnp.asarray(excitation_total, dtype=jnp.float64) / (jnp.asarray(excitation_total, dtype=jnp.float64) + 1.0)
    exc_quality = jnp.clip(exc_quality, 0.0, 1.0)

    # Combined quality (geometric mean for smoothness; multiplicative so it is monotone in each risk proxy)
    base = jnp.sqrt(cond_quality * support_quality)
    quality = base * mismatch_quality * dt_quality * z_quality * exc_quality * jnp.clip(jnp.asarray(power_beta, dtype=jnp.float64), 0.0, 1.0)
    
    # Map to alpha range (continuous)
    alpha_raw = alpha_min + (alpha_max - alpha_min) * quality
    
    # Clamp to valid range (always applied)
    alpha_result = clamp(alpha_raw, alpha_min, alpha_max)
    alpha = alpha_result.value
    
    # Build result
    result = FusionScaleResult(alpha=alpha)
    
    # This is an exact operation (closed-form formula)
    cert = CertBundle.create_exact(
        chart_id=chart_id,
        anchor_id=anchor_id,
        overconfidence=OverconfidenceCert(
            excitation_total=float(excitation_total),
            ess_to_excitation=float(ess_evidence) / (float(excitation_total) + float(constants.GC_EPS_MASS)),
            cond_to_support=float(cond_evidence) / (float(support_frac) + float(constants.GC_EPS_MASS)),
            dt_asymmetry=float(dt_asymmetry),
            z_to_xy_ratio=float(z_to_xy_ratio),
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=alpha,
        ),
    )
    
    expected_effect = ExpectedEffect(
        objective_name="fusion_alpha",
        predicted=alpha,
        realized=None,
    )
    
    return result, cert, expected_effect


# =============================================================================
# InfoFusionAdditive Operator
# =============================================================================


def info_fusion_additive(
    belief_pred: BeliefGaussianInfo,
    L_evidence: jnp.ndarray,
    h_evidence: jnp.ndarray,
    alpha: float,
    eps_psd: float = constants.GC_EPS_PSD,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[BeliefGaussianInfo, CertBundle, ExpectedEffect]:
    """
    Additive information fusion with continuous scaling.
    
    L_post = L_pred + alpha * L_evidence
    h_post = h_pred + alpha * h_evidence
    
    Always applies DomainProjectionPSD to result.
    
    Args:
        belief_pred: Predicted belief
        L_evidence: Evidence information matrix (D_Z, D_Z)
        h_evidence: Evidence information vector (D_Z,)
        alpha: Fusion scale from FusionScaleFromCertificates
        eps_psd: PSD projection epsilon
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (fused_belief, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.11
    """
    L_evidence = jnp.asarray(L_evidence, dtype=jnp.float64)
    h_evidence = jnp.asarray(h_evidence, dtype=jnp.float64)
    alpha = float(alpha)
    
    # Additive fusion (always computed)
    L_post_raw = belief_pred.L + alpha * L_evidence
    h_post = belief_pred.h + alpha * h_evidence
    
    # Always apply DomainProjectionPSD
    L_post_result = domain_projection_psd(L_post_raw, eps_psd)
    L_post = L_post_result.M_psd
    
    # Build fused belief
    cert_out = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["InfoFusionAdditive"],
        conditioning=ConditioningCert(
            eig_min=L_post_result.conditioning.eig_min,
            eig_max=L_post_result.conditioning.eig_max,
            cond=L_post_result.conditioning.cond,
            near_null_count=L_post_result.conditioning.near_null_count,
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=L_post_result.projection_delta,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=alpha,
        ),
    )
    
    belief_post = BeliefGaussianInfo(
        chart_id=chart_id,
        anchor_id=anchor_id,
        X_anchor=belief_pred.X_anchor,
        stamp_sec=belief_pred.stamp_sec,
        z_lin=belief_pred.z_lin,
        L=L_post,
        h=h_post,
        cert=cert_out,
    )
    
    # Expected effect: trace increase
    trace_increase = float(jnp.trace(L_post) - jnp.trace(belief_pred.L))
    
    expected_effect = ExpectedEffect(
        objective_name="predicted_info_trace_increase",
        predicted=trace_increase,
        realized=None,
    )
    
    return belief_post, cert_out, expected_effect
