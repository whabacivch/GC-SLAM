"""
Certificate structures for Geometric Compositional SLAM v2.

Certificates provide the audit trail for all approximations
and numerical stabilizations.

Reference: docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md Section 2.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# =============================================================================
# Component Certificates
# =============================================================================


@dataclass
class ConditioningCert:
    """Conditioning information from eigenvalue analysis."""
    eig_min: float = 1.0
    eig_max: float = 1.0
    cond: float = 1.0
    near_null_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eig_min": self.eig_min,
            "eig_max": self.eig_max,
            "cond": self.cond,
            "near_null_count": self.near_null_count,
        }


@dataclass
class SupportCert:
    """Support/coverage information."""
    ess_total: float = 0.0  # Effective sample size
    support_frac: float = 1.0  # Fraction of bins/points with support

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ess_total": self.ess_total,
            "support_frac": self.support_frac,
        }


@dataclass
class MismatchCert:
    """Model-observation mismatch information."""
    nll_per_ess: float = 0.0  # Negative log-likelihood per ESS
    directional_score: float = 1.0  # Directional alignment quality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nll_per_ess": self.nll_per_ess,
            "directional_score": self.directional_score,
        }


@dataclass
class ExcitationCert:
    """Excitation information from uncertain states."""
    dt_effect: float = 0.0  # Effect from time offset uncertainty
    extrinsic_effect: float = 0.0  # Effect from extrinsic uncertainty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dt_effect": self.dt_effect,
            "extrinsic_effect": self.extrinsic_effect,
        }


@dataclass
class InfluenceCert:
    """Influence/scaling information."""
    lift_strength: float = 0.0  # Total lift applied
    psd_projection_delta: float = 0.0  # Total PSD projection magnitude
    nu_projection_delta: float = 0.0  # Total ν / dof projection magnitude (domain projection)
    mass_epsilon_ratio: float = 0.0  # eps_mass / (mass + eps_mass)
    anchor_drift_rho: float = 0.0  # Anchor drift blend factor
    dt_scale: float = 1.0  # Time scaling factor
    extrinsic_scale: float = 1.0  # Extrinsic coupling factor
    trust_alpha: float = 1.0  # Fusion trust factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lift_strength": self.lift_strength,
            "psd_projection_delta": self.psd_projection_delta,
            "nu_projection_delta": self.nu_projection_delta,
            "mass_epsilon_ratio": self.mass_epsilon_ratio,
            "anchor_drift_rho": self.anchor_drift_rho,
            "dt_scale": self.dt_scale,
            "extrinsic_scale": self.extrinsic_scale,
            "trust_alpha": self.trust_alpha,
        }


# =============================================================================
# Main Certificate Bundle
# =============================================================================


@dataclass
class CertBundle:
    """
    Certificate bundle for an operator.
    
    Tracks all approximations and their magnitudes.
    """
    chart_id: str
    anchor_id: str
    exact: bool
    approximation_triggers: List[str] = field(default_factory=list)
    frobenius_applied: bool = False
    conditioning: ConditioningCert = field(default_factory=ConditioningCert)
    support: SupportCert = field(default_factory=SupportCert)
    mismatch: MismatchCert = field(default_factory=MismatchCert)
    excitation: ExcitationCert = field(default_factory=ExcitationCert)
    influence: InfluenceCert = field(default_factory=InfluenceCert)

    @classmethod
    def create_exact(
        cls,
        chart_id: str,
        anchor_id: str,
        conditioning: Optional[ConditioningCert] = None,
        support: Optional[SupportCert] = None,
        mismatch: Optional[MismatchCert] = None,
        excitation: Optional[ExcitationCert] = None,
        influence: Optional[InfluenceCert] = None,
    ) -> "CertBundle":
        """Create certificate for exact (ExactOp) operation."""
        return cls(
            chart_id=chart_id,
            anchor_id=anchor_id,
            exact=True,
            approximation_triggers=[],
            frobenius_applied=False,
            conditioning=conditioning or ConditioningCert(),
            support=support or SupportCert(),
            mismatch=mismatch or MismatchCert(),
            excitation=excitation or ExcitationCert(),
            influence=influence or InfluenceCert(),
        )

    @classmethod
    def create_approx(
        cls,
        chart_id: str,
        anchor_id: str,
        triggers: List[str],
        frobenius_applied: bool = False,
        conditioning: Optional[ConditioningCert] = None,
        support: Optional[SupportCert] = None,
        mismatch: Optional[MismatchCert] = None,
        excitation: Optional[ExcitationCert] = None,
        influence: Optional[InfluenceCert] = None,
    ) -> "CertBundle":
        """Create certificate for approximate (ApproxOp) operation."""
        return cls(
            chart_id=chart_id,
            anchor_id=anchor_id,
            exact=False,
            approximation_triggers=triggers,
            frobenius_applied=frobenius_applied,
            conditioning=conditioning or ConditioningCert(),
            support=support or SupportCert(),
            mismatch=mismatch or MismatchCert(),
            excitation=excitation or ExcitationCert(),
            influence=influence or InfluenceCert(),
        )

    def total_trigger_magnitude(self) -> float:
        """
        Compute total trigger magnitude for Frobenius correction.
        
        Sum of all influence magnitudes that indicate approximation.
        """
        return (
            self.influence.lift_strength
            + self.influence.psd_projection_delta
            + self.influence.nu_projection_delta
            + self.influence.mass_epsilon_ratio
            + self.influence.anchor_drift_rho
            + abs(1.0 - self.influence.dt_scale)
            + abs(1.0 - self.influence.extrinsic_scale)
            + abs(1.0 - self.influence.trust_alpha)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/publishing."""
        return {
            "chart_id": self.chart_id,
            "anchor_id": self.anchor_id,
            "exact": self.exact,
            "approximation_triggers": self.approximation_triggers,
            "frobenius_applied": self.frobenius_applied,
            "conditioning": self.conditioning.to_dict(),
            "support": self.support.to_dict(),
            "mismatch": self.mismatch.to_dict(),
            "excitation": self.excitation.to_dict(),
            "influence": self.influence.to_dict(),
            "total_trigger_magnitude": self.total_trigger_magnitude(),
        }


# =============================================================================
# Expected Effect
# =============================================================================


@dataclass
class ExpectedEffect:
    """
    Expected vs realized effect for an operator.
    
    Per spec: every operator must return this for audit.
    """
    objective_name: str
    predicted: float
    realized: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_name": self.objective_name,
            "predicted": self.predicted,
            "realized": self.realized,
        }


# =============================================================================
# Certificate Aggregation
# =============================================================================


def aggregate_certificates(certs: List[CertBundle]) -> CertBundle:
    """
    Aggregate multiple certificates into a single summary.
    
    Used for pipeline-level reporting.
    
    Args:
        certs: List of certificates from operators
        
    Returns:
        Aggregated CertBundle
    """
    if not certs:
        return CertBundle.create_exact(
            chart_id="unknown",
            anchor_id="unknown",
        )
    
    # Use first cert as template
    template = certs[0]
    
    # Collect all triggers
    all_triggers = []
    for c in certs:
        all_triggers.extend(c.approximation_triggers)
    
    # Any exact = False means result is approximate
    any_approx = any(not c.exact for c in certs)
    
    # Any frobenius applied
    any_frobenius = any(c.frobenius_applied for c in certs)
    
    # Aggregate conditioning (worst case)
    agg_cond = ConditioningCert(
        eig_min=min(c.conditioning.eig_min for c in certs),
        eig_max=max(c.conditioning.eig_max for c in certs),
        cond=max(c.conditioning.cond for c in certs),
        near_null_count=sum(c.conditioning.near_null_count for c in certs),
    )
    
    # Aggregate support (average)
    agg_support = SupportCert(
        ess_total=sum(c.support.ess_total for c in certs) / len(certs),
        support_frac=sum(c.support.support_frac for c in certs) / len(certs),
    )
    
    # Aggregate mismatch (sum)
    agg_mismatch = MismatchCert(
        nll_per_ess=sum(c.mismatch.nll_per_ess for c in certs),
        directional_score=sum(c.mismatch.directional_score for c in certs) / len(certs),
    )
    
    # Aggregate excitation (max)
    agg_excitation = ExcitationCert(
        dt_effect=max(c.excitation.dt_effect for c in certs),
        extrinsic_effect=max(c.excitation.extrinsic_effect for c in certs),
    )
    
    # Aggregate influence (sum magnitudes)
    agg_influence = InfluenceCert(
        lift_strength=sum(c.influence.lift_strength for c in certs),
        psd_projection_delta=sum(c.influence.psd_projection_delta for c in certs),
        nu_projection_delta=sum(c.influence.nu_projection_delta for c in certs),
        mass_epsilon_ratio=max(c.influence.mass_epsilon_ratio for c in certs),
        anchor_drift_rho=max(c.influence.anchor_drift_rho for c in certs),
        dt_scale=min(c.influence.dt_scale for c in certs),
        extrinsic_scale=min(c.influence.extrinsic_scale for c in certs),
        trust_alpha=min(c.influence.trust_alpha for c in certs),
    )
    
    return CertBundle(
        chart_id=template.chart_id,
        anchor_id=template.anchor_id,
        exact=not any_approx,
        approximation_triggers=all_triggers,
        frobenius_applied=any_frobenius,
        conditioning=agg_cond,
        support=agg_support,
        mismatch=agg_mismatch,
        excitation=agg_excitation,
        influence=agg_influence,
    )
