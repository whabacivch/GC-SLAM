"""
Stage-0 Per-Scan Diagnostics for Geometric Compositional SLAM v2.

This module provides data structures and utilities for capturing per-scan
minimal tape diagnostics (canonical). Full ScanDiagnostics has been removed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np


@dataclass
class MinimalScanTape:
    """
    Minimal per-scan tape for crash-tolerant, low-overhead diagnostics.
    Hot path stores only this; full ScanDiagnostics is removed.
    """
    scan_number: int
    timestamp: float
    dt_sec: float
    n_points_raw: int
    n_points_budget: int
    fusion_alpha: float
    cond_pose6: float
    conditioning_number: float
    eigmin_pose6: float
    L_pose6: np.ndarray  # (6, 6) pose block of L_evidence
    total_trigger_magnitude: float
    # Certificate summary (aggregated)
    cert_exact: bool
    cert_frobenius_applied: bool
    cert_n_triggers: int
    support_ess_total: float
    support_frac: float
    mismatch_nll_per_ess: float
    mismatch_directional_score: float
    excitation_dt_effect: float
    excitation_extrinsic_effect: float
    influence_psd_projection_delta: float
    influence_mass_epsilon_ratio: float
    influence_anchor_drift_rho: float
    influence_dt_scale: float
    influence_extrinsic_scale: float
    influence_trust_alpha: float
    influence_power_beta: float
    overconfidence_excitation_total: float
    overconfidence_ess_to_excitation: float
    overconfidence_cond_to_support: float
    overconfidence_dt_asymmetry: float
    overconfidence_z_to_xy_ratio: float
    # Optional timing (when enable_timing)
    t_total_ms: float = 0.0
    t_point_budget_ms: float = 0.0
    t_deskew_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_number": self.scan_number,
            "timestamp": self.timestamp,
            "dt_sec": self.dt_sec,
            "n_points_raw": self.n_points_raw,
            "n_points_budget": self.n_points_budget,
            "fusion_alpha": self.fusion_alpha,
            "cond_pose6": self.cond_pose6,
            "conditioning_number": self.conditioning_number,
            "eigmin_pose6": self.eigmin_pose6,
            "L_pose6": self.L_pose6.tolist(),
            "total_trigger_magnitude": self.total_trigger_magnitude,
            "cert_exact": self.cert_exact,
            "cert_frobenius_applied": self.cert_frobenius_applied,
            "cert_n_triggers": self.cert_n_triggers,
            "support_ess_total": self.support_ess_total,
            "support_frac": self.support_frac,
            "mismatch_nll_per_ess": self.mismatch_nll_per_ess,
            "mismatch_directional_score": self.mismatch_directional_score,
            "excitation_dt_effect": self.excitation_dt_effect,
            "excitation_extrinsic_effect": self.excitation_extrinsic_effect,
            "influence_psd_projection_delta": self.influence_psd_projection_delta,
            "influence_mass_epsilon_ratio": self.influence_mass_epsilon_ratio,
            "influence_anchor_drift_rho": self.influence_anchor_drift_rho,
            "influence_dt_scale": self.influence_dt_scale,
            "influence_extrinsic_scale": self.influence_extrinsic_scale,
            "influence_trust_alpha": self.influence_trust_alpha,
            "influence_power_beta": self.influence_power_beta,
            "overconfidence_excitation_total": self.overconfidence_excitation_total,
            "overconfidence_ess_to_excitation": self.overconfidence_ess_to_excitation,
            "overconfidence_cond_to_support": self.overconfidence_cond_to_support,
            "overconfidence_dt_asymmetry": self.overconfidence_dt_asymmetry,
            "overconfidence_z_to_xy_ratio": self.overconfidence_z_to_xy_ratio,
            "t_total_ms": self.t_total_ms,
            "t_point_budget_ms": self.t_point_budget_ms,
            "t_deskew_ms": self.t_deskew_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MinimalScanTape":
        return cls(
            scan_number=int(d["scan_number"]),
            timestamp=float(d["timestamp"]),
            dt_sec=float(d["dt_sec"]),
            n_points_raw=int(d["n_points_raw"]),
            n_points_budget=int(d["n_points_budget"]),
            fusion_alpha=float(d["fusion_alpha"]),
            cond_pose6=float(d["cond_pose6"]),
            conditioning_number=float(d["conditioning_number"]),
            eigmin_pose6=float(d["eigmin_pose6"]),
            L_pose6=np.array(d["L_pose6"], dtype=np.float64),
            total_trigger_magnitude=float(d["total_trigger_magnitude"]),
            cert_exact=bool(d.get("cert_exact", True)),
            cert_frobenius_applied=bool(d.get("cert_frobenius_applied", False)),
            cert_n_triggers=int(d.get("cert_n_triggers", 0)),
            support_ess_total=float(d.get("support_ess_total", 0.0)),
            support_frac=float(d.get("support_frac", 0.0)),
            mismatch_nll_per_ess=float(d.get("mismatch_nll_per_ess", 0.0)),
            mismatch_directional_score=float(d.get("mismatch_directional_score", 0.0)),
            excitation_dt_effect=float(d.get("excitation_dt_effect", 0.0)),
            excitation_extrinsic_effect=float(d.get("excitation_extrinsic_effect", 0.0)),
            influence_psd_projection_delta=float(d.get("influence_psd_projection_delta", 0.0)),
            influence_mass_epsilon_ratio=float(d.get("influence_mass_epsilon_ratio", 0.0)),
            influence_anchor_drift_rho=float(d.get("influence_anchor_drift_rho", 0.0)),
            influence_dt_scale=float(d.get("influence_dt_scale", 1.0)),
            influence_extrinsic_scale=float(d.get("influence_extrinsic_scale", 1.0)),
            influence_trust_alpha=float(d.get("influence_trust_alpha", 1.0)),
            influence_power_beta=float(d.get("influence_power_beta", 1.0)),
            overconfidence_excitation_total=float(d.get("overconfidence_excitation_total", 0.0)),
            overconfidence_ess_to_excitation=float(d.get("overconfidence_ess_to_excitation", 0.0)),
            overconfidence_cond_to_support=float(d.get("overconfidence_cond_to_support", 0.0)),
            overconfidence_dt_asymmetry=float(d.get("overconfidence_dt_asymmetry", 0.0)),
            overconfidence_z_to_xy_ratio=float(d.get("overconfidence_z_to_xy_ratio", 0.0)),
            t_total_ms=float(d.get("t_total_ms", 0.0)),
            t_point_budget_ms=float(d.get("t_point_budget_ms", 0.0)),
            t_deskew_ms=float(d.get("t_deskew_ms", 0.0)),
        )


@dataclass
class DiagnosticsLog:
    """Container for minimal per-scan tape diagnostics."""

    tape: List[MinimalScanTape] = field(default_factory=list)

    # Run metadata
    run_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_scans: int = 0

    def append_tape(self, entry: MinimalScanTape) -> None:
        """Add a scan's minimal tape entry (hot path)."""
        self.tape.append(entry)
        self.total_scans = len(self.tape)

    def save_jsonl(self, path: str) -> None:
        """Save as JSON Lines (one JSON object per line for streaming)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            header = {
                "_type": "header",
                "run_id": self.run_id,
                "start_time": self.start_time,
                "total_scans": self.total_scans,
            }
            f.write(json.dumps(header) + "\n")
            for entry in self.tape:
                f.write(json.dumps(entry.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: str) -> "DiagnosticsLog":
        """Load from JSON Lines file."""
        log = cls()
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("_type") == "header":
                    log.run_id = d.get("run_id", "")
                    log.start_time = d.get("start_time", 0.0)
                else:
                    log.tape.append(MinimalScanTape.from_dict(d))
        log.total_scans = len(log.tape)
        return log

    def save_npz(self, path: str) -> None:
        """Save minimal tape format (crash-tolerant, low overhead)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n = len(self.tape)
        if n == 0:
            np.savez_compressed(path, format="minimal_tape", n_scans=0)
            return
        data = {
            "format": "minimal_tape",
            "n_scans": n,
            "run_id": self.run_id,
            "start_time": self.start_time,
            "scan_numbers": np.array([t.scan_number for t in self.tape]),
            "timestamps": np.array([t.timestamp for t in self.tape]),
            "dt_secs": np.array([t.dt_sec for t in self.tape]),
            "n_points_raw": np.array([t.n_points_raw for t in self.tape]),
            "n_points_budget": np.array([t.n_points_budget for t in self.tape]),
            "fusion_alpha": np.array([t.fusion_alpha for t in self.tape]),
            "cond_pose6": np.array([t.cond_pose6 for t in self.tape]),
            "conditioning_number": np.array([t.conditioning_number for t in self.tape]),
            "eigmin_pose6": np.array([t.eigmin_pose6 for t in self.tape]),
            "L_pose6": np.stack([t.L_pose6 for t in self.tape]),
            "total_trigger_magnitude": np.array([t.total_trigger_magnitude for t in self.tape]),
            "cert_exact": np.array([t.cert_exact for t in self.tape]),
            "cert_frobenius_applied": np.array([t.cert_frobenius_applied for t in self.tape]),
            "cert_n_triggers": np.array([t.cert_n_triggers for t in self.tape]),
            "support_ess_total": np.array([t.support_ess_total for t in self.tape]),
            "support_frac": np.array([t.support_frac for t in self.tape]),
            "mismatch_nll_per_ess": np.array([t.mismatch_nll_per_ess for t in self.tape]),
            "mismatch_directional_score": np.array([t.mismatch_directional_score for t in self.tape]),
            "excitation_dt_effect": np.array([t.excitation_dt_effect for t in self.tape]),
            "excitation_extrinsic_effect": np.array([t.excitation_extrinsic_effect for t in self.tape]),
            "influence_psd_projection_delta": np.array([t.influence_psd_projection_delta for t in self.tape]),
            "influence_mass_epsilon_ratio": np.array([t.influence_mass_epsilon_ratio for t in self.tape]),
            "influence_anchor_drift_rho": np.array([t.influence_anchor_drift_rho for t in self.tape]),
            "influence_dt_scale": np.array([t.influence_dt_scale for t in self.tape]),
            "influence_extrinsic_scale": np.array([t.influence_extrinsic_scale for t in self.tape]),
            "influence_trust_alpha": np.array([t.influence_trust_alpha for t in self.tape]),
            "influence_power_beta": np.array([t.influence_power_beta for t in self.tape]),
            "overconfidence_excitation_total": np.array([t.overconfidence_excitation_total for t in self.tape]),
            "overconfidence_ess_to_excitation": np.array([t.overconfidence_ess_to_excitation for t in self.tape]),
            "overconfidence_cond_to_support": np.array([t.overconfidence_cond_to_support for t in self.tape]),
            "overconfidence_dt_asymmetry": np.array([t.overconfidence_dt_asymmetry for t in self.tape]),
            "overconfidence_z_to_xy_ratio": np.array([t.overconfidence_z_to_xy_ratio for t in self.tape]),
            "t_total_ms": np.array([t.t_total_ms for t in self.tape]),
            "t_point_budget_ms": np.array([t.t_point_budget_ms for t in self.tape]),
            "t_deskew_ms": np.array([t.t_deskew_ms for t in self.tape]),
        }
        np.savez_compressed(path, **data)

    @classmethod
    def load_npz(cls, path: str) -> "DiagnosticsLog":
        """Load minimal tape from NumPy archive."""
        data = np.load(path, allow_pickle=True)
        if str(data.get("format", "")) != "minimal_tape":
            raise ValueError("Unsupported diagnostics format; only minimal_tape is supported.")

        n = int(data["n_scans"])
        if n == 0:
            return cls()
        log = cls()
        log.run_id = str(data.get("run_id", ""))
        log.start_time = float(data.get("start_time", 0.0))
        for i in range(n):
            tape_entry = MinimalScanTape(
                scan_number=int(data["scan_numbers"][i]),
                timestamp=float(data["timestamps"][i]),
                dt_sec=float(data["dt_secs"][i]),
                n_points_raw=int(data["n_points_raw"][i]),
                n_points_budget=int(data["n_points_budget"][i]),
                fusion_alpha=float(data["fusion_alpha"][i]),
                cond_pose6=float(data["cond_pose6"][i]),
                conditioning_number=float(data["conditioning_number"][i]),
                eigmin_pose6=float(data["eigmin_pose6"][i]),
                L_pose6=data["L_pose6"][i],
                total_trigger_magnitude=float(data["total_trigger_magnitude"][i]),
                cert_exact=bool(data["cert_exact"][i]) if "cert_exact" in data else True,
                cert_frobenius_applied=bool(data["cert_frobenius_applied"][i]) if "cert_frobenius_applied" in data else False,
                cert_n_triggers=int(data["cert_n_triggers"][i]) if "cert_n_triggers" in data else 0,
                support_ess_total=float(data["support_ess_total"][i]) if "support_ess_total" in data else 0.0,
                support_frac=float(data["support_frac"][i]) if "support_frac" in data else 0.0,
                mismatch_nll_per_ess=float(data["mismatch_nll_per_ess"][i]) if "mismatch_nll_per_ess" in data else 0.0,
                mismatch_directional_score=float(data["mismatch_directional_score"][i]) if "mismatch_directional_score" in data else 0.0,
                excitation_dt_effect=float(data["excitation_dt_effect"][i]) if "excitation_dt_effect" in data else 0.0,
                excitation_extrinsic_effect=float(data["excitation_extrinsic_effect"][i]) if "excitation_extrinsic_effect" in data else 0.0,
                influence_psd_projection_delta=float(data["influence_psd_projection_delta"][i]) if "influence_psd_projection_delta" in data else 0.0,
                influence_mass_epsilon_ratio=float(data["influence_mass_epsilon_ratio"][i]) if "influence_mass_epsilon_ratio" in data else 0.0,
                influence_anchor_drift_rho=float(data["influence_anchor_drift_rho"][i]) if "influence_anchor_drift_rho" in data else 0.0,
                influence_dt_scale=float(data["influence_dt_scale"][i]) if "influence_dt_scale" in data else 1.0,
                influence_extrinsic_scale=float(data["influence_extrinsic_scale"][i]) if "influence_extrinsic_scale" in data else 1.0,
                influence_trust_alpha=float(data["influence_trust_alpha"][i]) if "influence_trust_alpha" in data else 1.0,
                influence_power_beta=float(data["influence_power_beta"][i]) if "influence_power_beta" in data else 1.0,
                overconfidence_excitation_total=float(data["overconfidence_excitation_total"][i]) if "overconfidence_excitation_total" in data else 0.0,
                overconfidence_ess_to_excitation=float(data["overconfidence_ess_to_excitation"][i]) if "overconfidence_ess_to_excitation" in data else 0.0,
                overconfidence_cond_to_support=float(data["overconfidence_cond_to_support"][i]) if "overconfidence_cond_to_support" in data else 0.0,
                overconfidence_dt_asymmetry=float(data["overconfidence_dt_asymmetry"][i]) if "overconfidence_dt_asymmetry" in data else 0.0,
                overconfidence_z_to_xy_ratio=float(data["overconfidence_z_to_xy_ratio"][i]) if "overconfidence_z_to_xy_ratio" in data else 0.0,
                t_total_ms=float(data["t_total_ms"][i]) if "t_total_ms" in data else 0.0,
                t_point_budget_ms=float(data["t_point_budget_ms"][i]) if "t_point_budget_ms" in data else 0.0,
                t_deskew_ms=float(data["t_deskew_ms"][i]) if "t_deskew_ms" in data else 0.0,
            )
            log.tape.append(tape_entry)
        log.total_scans = len(log.tape)
        return log
