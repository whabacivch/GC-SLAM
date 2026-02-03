"""
Certificate structures for Geometric Compositional SLAM v2.

Certificates provide the audit trail for all approximations
and numerical stabilizations.

Reference: docs/GC_SLAM.md Section 2.3
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
    support_frac: float = 1.0  # Fraction of points/primitives with support

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
    power_beta: float = 1.0  # Tempered posterior / power EP scaling of evidence in (0,1]

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
            "power_beta": self.power_beta,
        }


@dataclass
class OverconfidenceCert:
    """
    Continuous sentinels for detecting overconfidence risk from dependent evidence.

    These fields are diagnostic-only; they must not trigger gating. Downstream logic may
    respond only by continuous conservatism (e.g., scaling information / trust).

    Convention:
    - 0.0 means "not computed / not available" for that field.

    Growth sentinels (spec §5.8):
    - ess_growth_rate: time-derivative proxy of ESS (requires state history)
    - excitation_growth_rate: time-derivative proxy of excitation_total
    - nullspace_energy_ratio: energy in near-null eigenspace / total energy
    """

    excitation_total: float = 0.0  # dt_effect + extrinsic_effect (or other agreed excitation proxy)
    ess_to_excitation: float = 0.0  # ess_total / (excitation_total + eps)
    cond_to_support: float = 0.0  # conditioning proxy divided by support proxy (higher => riskier)
    dt_asymmetry: float = 0.0  # |dt_vel - dt_pose| / (dt_vel + dt_pose + eps) in [0,1] when computed
    z_to_xy_ratio: float = 0.0  # L_zz / mean(L_xx, L_yy) (or analogous) when computed
    # Growth sentinels (time-series aware; require state history to compute)
    ess_growth_rate: float = 0.0  # d(ess)/dt proxy; positive = ESS increasing over scans
    excitation_growth_rate: float = 0.0  # d(excitation)/dt proxy; positive = excitation increasing
    nullspace_energy_ratio: float = 0.0  # energy in near-null eigenspace / total; 0 = well-constrained

    def to_dict(self) -> Dict[str, Any]:
        return {
            "excitation_total": self.excitation_total,
            "ess_to_excitation": self.ess_to_excitation,
            "cond_to_support": self.cond_to_support,
            "dt_asymmetry": self.dt_asymmetry,
            "z_to_xy_ratio": self.z_to_xy_ratio,
            "ess_growth_rate": self.ess_growth_rate,
            "excitation_growth_rate": self.excitation_growth_rate,
            "nullspace_energy_ratio": self.nullspace_energy_ratio,
        }


@dataclass
class OTCert:
    """
    Optimal Transport association certificate (spec §5.7.4).

    Diagnostic fields for OT marginal defects, transport mass, and mass policies.
    These enable verification of OT conformance without inspecting raw π.
    """
    # Marginal defects: ||π·1 - a|| and ||π^T·1 - b||
    marginal_defect_a: float = 0.0  # Row marginal defect
    marginal_defect_b: float = 0.0  # Column marginal defect
    transport_mass_total: float = 0.0  # sum(π)
    dual_gap_proxy: float = 0.0  # Optional dual gap estimate (0 if not computed)
    # Mass budget statistics
    sum_a: float = 0.0  # Total declared measurement mass
    sum_b: float = 0.0  # Total declared map mass
    p95_a: float = 0.0  # 95th percentile of a distribution
    p95_b: float = 0.0  # 95th percentile of b distribution
    nonzero_a: int = 0  # Count of nonzero entries in a
    nonzero_b: int = 0  # Count of nonzero entries in b
    # OT parameters used (for reproducibility)
    epsilon: float = 0.0  # Entropic regularization
    tau_a: float = 0.0  # Unbalanced KL for a
    tau_b: float = 0.0  # Unbalanced KL for b
    n_iters: int = 0  # Sinkhorn iterations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marginal_defect_a": self.marginal_defect_a,
            "marginal_defect_b": self.marginal_defect_b,
            "transport_mass_total": self.transport_mass_total,
            "dual_gap_proxy": self.dual_gap_proxy,
            "sum_a": self.sum_a,
            "sum_b": self.sum_b,
            "p95_a": self.p95_a,
            "p95_b": self.p95_b,
            "nonzero_a": self.nonzero_a,
            "nonzero_b": self.nonzero_b,
            "epsilon": self.epsilon,
            "tau_a": self.tau_a,
            "tau_b": self.tau_b,
            "n_iters": self.n_iters,
        }


@dataclass
class MapUpdateCert:
    """
    Map update certificate for budget verification (spec §3.2, §5.8).

    Tracks tile activity, insertion/eviction, and primitive counts for
    fixed-cost verification.
    """
    # Tile activity
    n_active_tiles: int = 0  # Number of tiles touched this scan
    tile_ids_active: List[int] = field(default_factory=list)  # IDs of active tiles
    # Candidate statistics
    candidate_tiles_per_meas_mean: float = 0.0  # Mean tiles per measurement
    candidate_primitives_per_meas_mean: float = 0.0  # Mean primitives per measurement
    candidate_primitives_per_meas_p95: float = 0.0  # 95th percentile
    # Insertion statistics
    insert_count_total: int = 0  # Primitives inserted this scan
    insert_mass_total: float = 0.0  # Total mass of inserted primitives
    insert_mass_p95: float = 0.0  # 95th percentile insertion mass
    # Eviction/cull statistics
    evicted_count: int = 0  # Primitives culled this scan
    evicted_mass_total: float = 0.0  # Total mass of evicted primitives
    # Fusion statistics
    fused_count: int = 0  # Primitives fused this scan
    fused_mass_total: float = 0.0  # Total mass fused

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_active_tiles": self.n_active_tiles,
            "tile_ids_active": list(self.tile_ids_active),
            "candidate_tiles_per_meas_mean": self.candidate_tiles_per_meas_mean,
            "candidate_primitives_per_meas_mean": self.candidate_primitives_per_meas_mean,
            "candidate_primitives_per_meas_p95": self.candidate_primitives_per_meas_p95,
            "insert_count_total": self.insert_count_total,
            "insert_mass_total": self.insert_mass_total,
            "insert_mass_p95": self.insert_mass_p95,
            "evicted_count": self.evicted_count,
            "evicted_mass_total": self.evicted_mass_total,
            "fused_count": self.fused_count,
            "fused_mass_total": self.fused_mass_total,
        }


# =============================================================================
# Compute / Runtime Certificates
# =============================================================================


@dataclass
class ScanIOCert:
    """
    Scan-clock and per-stream I/O accounting.

    All fields are deterministic and always populated (defaults are zeros/empty).
    """

    scan_seq: int = 0
    scan_stamp_sec: float = 0.0
    scan_window_start_sec: float = 0.0
    scan_window_end_sec: float = 0.0
    streams: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_seq": int(self.scan_seq),
            "scan_stamp_sec": float(self.scan_stamp_sec),
            "scan_window_start_sec": float(self.scan_window_start_sec),
            "scan_window_end_sec": float(self.scan_window_end_sec),
            "streams": dict(self.streams),
        }


@dataclass
class DeviceRuntimeCert:
    """
    Host/device transfer and runtime accounting (best-effort estimates).
    """

    host_sync_count_est: int = 0
    device_to_host_bytes_est: int = 0
    host_to_device_bytes_est: int = 0
    jit_recompile_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_sync_count_est": int(self.host_sync_count_est),
            "device_to_host_bytes_est": int(self.device_to_host_bytes_est),
            "host_to_device_bytes_est": int(self.host_to_device_bytes_est),
            "jit_recompile_count": int(self.jit_recompile_count),
        }


@dataclass
class ComputeCert:
    """
    Runtime compute/shape accounting for fixed-cost verification.
    """

    alloc_bytes_est: int = 0
    largest_tensor_shape: tuple = (0, 0)
    segment_sum_k: int = 0
    psd_projection_count: int = 0
    chol_solve_count: int = 0
    scan_io: ScanIOCert = field(default_factory=ScanIOCert)
    device_runtime: DeviceRuntimeCert = field(default_factory=DeviceRuntimeCert)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alloc_bytes_est": int(self.alloc_bytes_est),
            "largest_tensor_shape": self.largest_tensor_shape,
            "segment_sum_k": int(self.segment_sum_k),
            "psd_projection_count": int(self.psd_projection_count),
            "chol_solve_count": int(self.chol_solve_count),
            "scan_io": self.scan_io.to_dict(),
            "device_runtime": self.device_runtime.to_dict(),
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
    overconfidence: OverconfidenceCert = field(default_factory=OverconfidenceCert)
    compute: ComputeCert = field(default_factory=ComputeCert)
    # Optional domain-specific certs (populated by relevant operators)
    ot: Optional[OTCert] = None  # Populated by association operators
    map_update: Optional[MapUpdateCert] = None  # Populated by map update operators

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
        overconfidence: Optional[OverconfidenceCert] = None,
        compute: Optional[ComputeCert] = None,
        ot: Optional["OTCert"] = None,
        map_update: Optional["MapUpdateCert"] = None,
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
            overconfidence=overconfidence or OverconfidenceCert(),
            compute=compute or ComputeCert(),
            ot=ot,
            map_update=map_update,
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
        overconfidence: Optional[OverconfidenceCert] = None,
        compute: Optional[ComputeCert] = None,
        ot: Optional["OTCert"] = None,
        map_update: Optional["MapUpdateCert"] = None,
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
            overconfidence=overconfidence or OverconfidenceCert(),
            compute=compute or ComputeCert(),
            ot=ot,
            map_update=map_update,
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
            + abs(1.0 - self.influence.power_beta)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/publishing."""
        d = {
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
            "overconfidence": self.overconfidence.to_dict(),
            "compute": self.compute.to_dict(),
            "total_trigger_magnitude": self.total_trigger_magnitude(),
        }
        # Optional domain-specific certs (only include if populated)
        if self.ot is not None:
            d["ot"] = self.ot.to_dict()
        if self.map_update is not None:
            d["map_update"] = self.map_update.to_dict()
        return d


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
        power_beta=min(c.influence.power_beta for c in certs),
    )

    # Aggregate overconfidence (max / worst-case)
    agg_overconfidence = OverconfidenceCert(
        excitation_total=max(c.overconfidence.excitation_total for c in certs),
        ess_to_excitation=max(c.overconfidence.ess_to_excitation for c in certs),
        cond_to_support=max(c.overconfidence.cond_to_support for c in certs),
        dt_asymmetry=max(c.overconfidence.dt_asymmetry for c in certs),
        z_to_xy_ratio=max(c.overconfidence.z_to_xy_ratio for c in certs),
        ess_growth_rate=max(c.overconfidence.ess_growth_rate for c in certs),
        excitation_growth_rate=max(c.overconfidence.excitation_growth_rate for c in certs),
        nullspace_energy_ratio=max(c.overconfidence.nullspace_energy_ratio for c in certs),
    )

    def _shape_score(shape: Any) -> int:
        if isinstance(shape, (tuple, list)) and shape:
            prod = 1
            for v in shape:
                try:
                    prod *= int(v)
                except Exception:
                    return 0
            return int(prod)
        return 0

    # Aggregate compute (max for counts/bytes; latest scan_io by scan_seq)
    max_alloc = max(c.compute.alloc_bytes_est for c in certs)
    max_seg = max(c.compute.segment_sum_k for c in certs)
    max_psd = max(c.compute.psd_projection_count for c in certs)
    max_chol = max(c.compute.chol_solve_count for c in certs)
    largest_shape = max((c.compute.largest_tensor_shape for c in certs), key=_shape_score, default=(0, 0))
    max_host_sync = max(c.compute.device_runtime.host_sync_count_est for c in certs)
    max_d2h = max(c.compute.device_runtime.device_to_host_bytes_est for c in certs)
    max_h2d = max(c.compute.device_runtime.host_to_device_bytes_est for c in certs)
    max_recomp = max(c.compute.device_runtime.jit_recompile_count for c in certs)

    # Pick scan_io from the highest scan_seq (deterministic); fallback to first.
    scan_io_choice = max((c.compute.scan_io for c in certs), key=lambda s: s.scan_seq, default=certs[0].compute.scan_io)
    agg_compute = ComputeCert(
        alloc_bytes_est=max_alloc,
        largest_tensor_shape=largest_shape,
        segment_sum_k=max_seg,
        psd_projection_count=max_psd,
        chol_solve_count=max_chol,
        scan_io=scan_io_choice,
        device_runtime=DeviceRuntimeCert(
            host_sync_count_est=max_host_sync,
            device_to_host_bytes_est=max_d2h,
            host_to_device_bytes_est=max_h2d,
            jit_recompile_count=max_recomp,
        ),
    )
    
    # Aggregate OTCert (pick first non-None; sum transport mass)
    ot_certs = [c.ot for c in certs if c.ot is not None]
    agg_ot = None
    if ot_certs:
        agg_ot = OTCert(
            marginal_defect_a=max(c.marginal_defect_a for c in ot_certs),
            marginal_defect_b=max(c.marginal_defect_b for c in ot_certs),
            transport_mass_total=sum(c.transport_mass_total for c in ot_certs),
            dual_gap_proxy=max(c.dual_gap_proxy for c in ot_certs),
            sum_a=sum(c.sum_a for c in ot_certs),
            sum_b=sum(c.sum_b for c in ot_certs),
            p95_a=max(c.p95_a for c in ot_certs),
            p95_b=max(c.p95_b for c in ot_certs),
            nonzero_a=sum(c.nonzero_a for c in ot_certs),
            nonzero_b=sum(c.nonzero_b for c in ot_certs),
            epsilon=ot_certs[0].epsilon,  # Use first OT params
            tau_a=ot_certs[0].tau_a,
            tau_b=ot_certs[0].tau_b,
            n_iters=ot_certs[0].n_iters,
        )

    # Aggregate MapUpdateCert (sum counts; union tile IDs)
    map_certs = [c.map_update for c in certs if c.map_update is not None]
    agg_map = None
    if map_certs:
        all_tile_ids = []
        for mc in map_certs:
            all_tile_ids.extend(mc.tile_ids_active)
        agg_map = MapUpdateCert(
            n_active_tiles=len(set(all_tile_ids)),
            tile_ids_active=list(set(all_tile_ids)),
            candidate_tiles_per_meas_mean=max(c.candidate_tiles_per_meas_mean for c in map_certs),
            candidate_primitives_per_meas_mean=max(c.candidate_primitives_per_meas_mean for c in map_certs),
            candidate_primitives_per_meas_p95=max(c.candidate_primitives_per_meas_p95 for c in map_certs),
            insert_count_total=sum(c.insert_count_total for c in map_certs),
            insert_mass_total=sum(c.insert_mass_total for c in map_certs),
            insert_mass_p95=max(c.insert_mass_p95 for c in map_certs),
            evicted_count=sum(c.evicted_count for c in map_certs),
            evicted_mass_total=sum(c.evicted_mass_total for c in map_certs),
            fused_count=sum(c.fused_count for c in map_certs),
            fused_mass_total=sum(c.fused_mass_total for c in map_certs),
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
        overconfidence=agg_overconfidence,
        compute=agg_compute,
        ot=agg_ot,
        map_update=agg_map,
    )
