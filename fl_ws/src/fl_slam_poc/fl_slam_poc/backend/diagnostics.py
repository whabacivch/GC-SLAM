"""
Stage-0 Per-Scan Diagnostics for Geometric Compositional SLAM v2.

This module provides data structures and utilities for capturing per-scan
diagnostic data needed for the debugging dashboard.

Stage-0 Schema (no map atoms):
- Pose estimates (position + orientation)
- 48-bin reducer outputs (N, S, kappa)
- Evidence matrices (L_total 22x22, h_total 22)
- Excitation scalars (s_dt, s_ex)
- PSD projection diagnostics
- Noise trace summaries
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np


@dataclass
class ScanDiagnostics:
    """Per-scan diagnostic data for the debugging dashboard."""

    # Scan metadata
    scan_number: int
    timestamp: float
    dt_sec: float
    n_points_raw: int
    n_points_budget: int

    # Pose (in world frame)
    p_W: np.ndarray  # (3,) position
    R_WL: np.ndarray  # (3,3) rotation matrix (world <- lidar)

    # 48-bin statistics
    N_bins: np.ndarray  # (48,) mass per bin
    S_bins: np.ndarray  # (48, 3) direction resultant vectors
    kappa_bins: np.ndarray  # (48,) concentration per bin

    # Evidence matrices
    L_total: np.ndarray  # (22, 22) total information matrix
    h_total: np.ndarray  # (22,) total information vector

    # Individual evidence components (for decomposition)
    L_lidar: Optional[np.ndarray] = None  # (22, 22)
    L_odom: Optional[np.ndarray] = None   # (22, 22)
    L_imu: Optional[np.ndarray] = None    # (22, 22)
    L_gyro: Optional[np.ndarray] = None   # (22, 22)
    L_imu_preint: Optional[np.ndarray] = None  # (22, 22) IMU preintegration factor

    # Diagnostic scalars
    logdet_L_total: float = 0.0
    trace_L_total: float = 0.0
    L_dt: float = 0.0  # L[15,15] - dt observability
    trace_L_ex: float = 0.0  # trace(L[16:22, 16:22]) - extrinsic observability

    # Excitation scalars
    s_dt: float = 0.0  # dt excitation scale
    s_ex: float = 0.0  # extrinsic excitation scale

    # PSD projection diagnostics
    psd_delta_fro: float = 0.0  # Frobenius norm of projection delta
    psd_min_eig_before: float = 0.0
    psd_min_eig_after: float = 0.0

    # Noise trace summaries
    trace_Q_mode: float = 0.0  # trace of process noise Q
    trace_Sigma_lidar_mode: float = 0.0  # trace of LiDAR measurement noise
    trace_Sigma_g_mode: float = 0.0  # trace of gyro measurement noise
    trace_Sigma_a_mode: float = 0.0  # trace of accel measurement noise

    # Wahba and translation diagnostics
    wahba_cost: float = 0.0
    translation_residual_norm: float = 0.0

    # Matrix Fisher rotation evidence diagnostics
    # Singular values of the (weighted) scatter matrix used by the MF rotation evidence.
    # These are the "MF health" sentinels: s3 -> 0 indicates a near-degenerate rotation subspace.
    mf_svd: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))

    # Directional scatter diagnostics (per bin)
    # Eigenvalues (ascending) of directional scatter tensors for scan and map.
    # Used by the dashboard to compute anisotropy/planarity observability proxies.
    scan_scatter_eigs: np.ndarray = field(default_factory=lambda: np.zeros((48, 3), dtype=np.float64))
    map_scatter_eigs: np.ndarray = field(default_factory=lambda: np.zeros((48, 3), dtype=np.float64))

    # Map kappa per bin (for top-K bin comparisons; may differ from scan kappa)
    kappa_map_bins: np.ndarray = field(default_factory=lambda: np.zeros((48,), dtype=np.float64))

    # Rotation binding diagnostics (degrees)
    rot_err_lidar_deg_pred: float = 0.0
    rot_err_lidar_deg_post: float = 0.0
    rot_err_odom_deg_pred: float = 0.0
    rot_err_odom_deg_post: float = 0.0

    # Fusion diagnostics
    fusion_alpha: float = 1.0

    # Certificate summaries
    total_trigger_magnitude: float = 0.0
    conditioning_number: float = 1.0
    conditioning_pose6: float = 1.0
    
    # IMU discretization diagnostics
    dt_scan: float = 0.0  # LiDAR scan duration
    dt_int: float = 0.0  # IMU-covered time (Σ_i Δt_i over actual sample intervals)
    num_imu_samples: int = 0  # Number of IMU samples used

    # Frame coherence probes (base-frame IMU sanity)
    accel_dir_dot_mu0: float = 0.0  # xbar · mu0 in body frame (should be near +1 when consistent)
    accel_mag_mean: float = 0.0     # mean ||a|| (m/s^2), should be near 9.81 when stationary

    # IMU propagation probes (weighted by dt_eff over scan-to-scan window)
    imu_a_body_mean: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))
    imu_a_world_nog_mean: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))
    imu_a_world_mean: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))
    imu_dt_eff_sum: float = 0.0

    # IMU dt diagnostics (scan-to-scan window; raw + weighted)
    imu_dt_valid_min: float = 0.0
    imu_dt_valid_max: float = 0.0
    imu_dt_valid_mean: float = 0.0
    imu_dt_valid_median: float = 0.0
    imu_dt_valid_std: float = 0.0
    imu_dt_valid_nonpos: int = 0
    imu_dt_weighted_mean: float = 0.0
    imu_dt_weighted_std: float = 0.0
    imu_dt_weighted_sum: float = 0.0

    # IMU preintegration factor residuals
    preint_r_vel: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))
    preint_r_pos: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))

    # Invariant test: yaw increments from different sources (degrees)
    dyaw_gyro: float = 0.0   # Gyro-integrated yaw change
    dyaw_odom: float = 0.0   # Odom yaw change
    dyaw_wahba: float = 0.0  # Wahba (LiDAR) yaw change

    # Timing diagnostics (milliseconds)
    t_total_ms: float = 0.0
    t_point_budget_ms: float = 0.0
    t_imu_preint_scan_ms: float = 0.0
    t_imu_preint_int_ms: float = 0.0
    t_deskew_ms: float = 0.0
    t_bin_assign_ms: float = 0.0
    t_bin_moment_ms: float = 0.0
    t_matrix_fisher_ms: float = 0.0
    t_planar_translation_ms: float = 0.0
    t_lidar_bucket_iw_ms: float = 0.0
    t_map_update_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scan_number": self.scan_number,
            "timestamp": self.timestamp,
            "dt_sec": self.dt_sec,
            "n_points_raw": self.n_points_raw,
            "n_points_budget": self.n_points_budget,
            "p_W": self.p_W.tolist(),
            "R_WL": self.R_WL.tolist(),
            "N_bins": self.N_bins.tolist(),
            "S_bins": self.S_bins.tolist(),
            "kappa_bins": self.kappa_bins.tolist(),
            "L_total": self.L_total.tolist(),
            "h_total": self.h_total.tolist(),
            "L_lidar": self.L_lidar.tolist() if self.L_lidar is not None else None,
            "L_odom": self.L_odom.tolist() if self.L_odom is not None else None,
            "L_imu": self.L_imu.tolist() if self.L_imu is not None else None,
            "L_gyro": self.L_gyro.tolist() if self.L_gyro is not None else None,
            "L_imu_preint": self.L_imu_preint.tolist() if self.L_imu_preint is not None else None,
            "logdet_L_total": self.logdet_L_total,
            "trace_L_total": self.trace_L_total,
            "L_dt": self.L_dt,
            "trace_L_ex": self.trace_L_ex,
            "s_dt": self.s_dt,
            "s_ex": self.s_ex,
            "psd_delta_fro": self.psd_delta_fro,
            "psd_min_eig_before": self.psd_min_eig_before,
            "psd_min_eig_after": self.psd_min_eig_after,
            "trace_Q_mode": self.trace_Q_mode,
            "trace_Sigma_lidar_mode": self.trace_Sigma_lidar_mode,
            "trace_Sigma_g_mode": self.trace_Sigma_g_mode,
            "trace_Sigma_a_mode": self.trace_Sigma_a_mode,
            "wahba_cost": self.wahba_cost,
            "translation_residual_norm": self.translation_residual_norm,
            "mf_svd": self.mf_svd.tolist(),
            "scan_scatter_eigs": self.scan_scatter_eigs.tolist(),
            "map_scatter_eigs": self.map_scatter_eigs.tolist(),
            "kappa_map_bins": self.kappa_map_bins.tolist(),
            "rot_err_lidar_deg_pred": self.rot_err_lidar_deg_pred,
            "rot_err_lidar_deg_post": self.rot_err_lidar_deg_post,
            "rot_err_odom_deg_pred": self.rot_err_odom_deg_pred,
            "rot_err_odom_deg_post": self.rot_err_odom_deg_post,
            "fusion_alpha": self.fusion_alpha,
            "total_trigger_magnitude": self.total_trigger_magnitude,
            "conditioning_number": self.conditioning_number,
            "conditioning_pose6": self.conditioning_pose6,
            "dt_scan": self.dt_scan,
            "dt_int": self.dt_int,
            "num_imu_samples": self.num_imu_samples,
            "accel_dir_dot_mu0": self.accel_dir_dot_mu0,
            "accel_mag_mean": self.accel_mag_mean,
            "imu_a_body_mean": self.imu_a_body_mean.tolist(),
            "imu_a_world_nog_mean": self.imu_a_world_nog_mean.tolist(),
            "imu_a_world_mean": self.imu_a_world_mean.tolist(),
            "imu_dt_eff_sum": self.imu_dt_eff_sum,
            "imu_dt_valid_min": self.imu_dt_valid_min,
            "imu_dt_valid_max": self.imu_dt_valid_max,
            "imu_dt_valid_mean": self.imu_dt_valid_mean,
            "imu_dt_valid_median": self.imu_dt_valid_median,
            "imu_dt_valid_std": self.imu_dt_valid_std,
            "imu_dt_valid_nonpos": self.imu_dt_valid_nonpos,
            "imu_dt_weighted_mean": self.imu_dt_weighted_mean,
            "imu_dt_weighted_std": self.imu_dt_weighted_std,
            "imu_dt_weighted_sum": self.imu_dt_weighted_sum,
            "preint_r_vel": self.preint_r_vel.tolist(),
            "preint_r_pos": self.preint_r_pos.tolist(),
            "dyaw_gyro": self.dyaw_gyro,
            "dyaw_odom": self.dyaw_odom,
            "dyaw_wahba": self.dyaw_wahba,
            "t_total_ms": self.t_total_ms,
            "t_point_budget_ms": self.t_point_budget_ms,
            "t_imu_preint_scan_ms": self.t_imu_preint_scan_ms,
            "t_imu_preint_int_ms": self.t_imu_preint_int_ms,
            "t_deskew_ms": self.t_deskew_ms,
            "t_bin_assign_ms": self.t_bin_assign_ms,
            "t_bin_moment_ms": self.t_bin_moment_ms,
            "t_matrix_fisher_ms": self.t_matrix_fisher_ms,
            "t_planar_translation_ms": self.t_planar_translation_ms,
            "t_lidar_bucket_iw_ms": self.t_lidar_bucket_iw_ms,
            "t_map_update_ms": self.t_map_update_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScanDiagnostics":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            scan_number=d["scan_number"],
            timestamp=d["timestamp"],
            dt_sec=d["dt_sec"],
            n_points_raw=d["n_points_raw"],
            n_points_budget=d["n_points_budget"],
            p_W=np.array(d["p_W"]),
            R_WL=np.array(d["R_WL"]),
            N_bins=np.array(d["N_bins"]),
            S_bins=np.array(d["S_bins"]),
            kappa_bins=np.array(d["kappa_bins"]),
            L_total=np.array(d["L_total"]),
            h_total=np.array(d["h_total"]),
            L_lidar=np.array(d["L_lidar"]) if d.get("L_lidar") is not None else None,
            L_odom=np.array(d["L_odom"]) if d.get("L_odom") is not None else None,
            L_imu=np.array(d["L_imu"]) if d.get("L_imu") is not None else None,
            L_gyro=np.array(d["L_gyro"]) if d.get("L_gyro") is not None else None,
            L_imu_preint=np.array(d["L_imu_preint"]) if d.get("L_imu_preint") is not None else None,
            logdet_L_total=d.get("logdet_L_total", 0.0),
            trace_L_total=d.get("trace_L_total", 0.0),
            L_dt=d.get("L_dt", 0.0),
            trace_L_ex=d.get("trace_L_ex", 0.0),
            s_dt=d.get("s_dt", 0.0),
            s_ex=d.get("s_ex", 0.0),
            psd_delta_fro=d.get("psd_delta_fro", 0.0),
            psd_min_eig_before=d.get("psd_min_eig_before", 0.0),
            psd_min_eig_after=d.get("psd_min_eig_after", 0.0),
            trace_Q_mode=d.get("trace_Q_mode", 0.0),
            trace_Sigma_lidar_mode=d.get("trace_Sigma_lidar_mode", 0.0),
            trace_Sigma_g_mode=d.get("trace_Sigma_g_mode", 0.0),
            trace_Sigma_a_mode=d.get("trace_Sigma_a_mode", 0.0),
            wahba_cost=d.get("wahba_cost", 0.0),
            translation_residual_norm=d.get("translation_residual_norm", 0.0),
            mf_svd=np.array(d.get("mf_svd", [0.0, 0.0, 0.0]), dtype=np.float64),
            scan_scatter_eigs=np.array(d.get("scan_scatter_eigs", np.zeros((48, 3))), dtype=np.float64),
            map_scatter_eigs=np.array(d.get("map_scatter_eigs", np.zeros((48, 3))), dtype=np.float64),
            kappa_map_bins=np.array(d.get("kappa_map_bins", np.zeros((48,))), dtype=np.float64),
            rot_err_lidar_deg_pred=d.get("rot_err_lidar_deg_pred", 0.0),
            rot_err_lidar_deg_post=d.get("rot_err_lidar_deg_post", 0.0),
            rot_err_odom_deg_pred=d.get("rot_err_odom_deg_pred", 0.0),
            rot_err_odom_deg_post=d.get("rot_err_odom_deg_post", 0.0),
            fusion_alpha=d.get("fusion_alpha", 1.0),
            total_trigger_magnitude=d.get("total_trigger_magnitude", 0.0),
            conditioning_number=d.get("conditioning_number", 1.0),
            conditioning_pose6=d.get("conditioning_pose6", 1.0),
            dt_scan=d.get("dt_scan", 0.0),
            dt_int=d.get("dt_int", 0.0),
            num_imu_samples=d.get("num_imu_samples", 0),
            accel_dir_dot_mu0=d.get("accel_dir_dot_mu0", 0.0),
            accel_mag_mean=d.get("accel_mag_mean", 0.0),
            imu_a_body_mean=np.array(d.get("imu_a_body_mean", [0.0, 0.0, 0.0])),
            imu_a_world_nog_mean=np.array(d.get("imu_a_world_nog_mean", [0.0, 0.0, 0.0])),
            imu_a_world_mean=np.array(d.get("imu_a_world_mean", [0.0, 0.0, 0.0])),
            imu_dt_eff_sum=d.get("imu_dt_eff_sum", 0.0),
            imu_dt_valid_min=d.get("imu_dt_valid_min", 0.0),
            imu_dt_valid_max=d.get("imu_dt_valid_max", 0.0),
            imu_dt_valid_mean=d.get("imu_dt_valid_mean", 0.0),
            imu_dt_valid_median=d.get("imu_dt_valid_median", 0.0),
            imu_dt_valid_std=d.get("imu_dt_valid_std", 0.0),
            imu_dt_valid_nonpos=d.get("imu_dt_valid_nonpos", 0),
            imu_dt_weighted_mean=d.get("imu_dt_weighted_mean", 0.0),
            imu_dt_weighted_std=d.get("imu_dt_weighted_std", 0.0),
            imu_dt_weighted_sum=d.get("imu_dt_weighted_sum", 0.0),
            preint_r_vel=np.array(d.get("preint_r_vel", [0.0, 0.0, 0.0])),
            preint_r_pos=np.array(d.get("preint_r_pos", [0.0, 0.0, 0.0])),
            dyaw_gyro=float(d.get("dyaw_gyro", 0.0)),
            dyaw_odom=float(d.get("dyaw_odom", 0.0)),
            dyaw_wahba=float(d.get("dyaw_wahba", 0.0)),
            t_total_ms=float(d.get("t_total_ms", 0.0)),
            t_point_budget_ms=float(d.get("t_point_budget_ms", 0.0)),
            t_imu_preint_scan_ms=float(d.get("t_imu_preint_scan_ms", 0.0)),
            t_imu_preint_int_ms=float(d.get("t_imu_preint_int_ms", 0.0)),
            t_deskew_ms=float(d.get("t_deskew_ms", 0.0)),
            t_bin_assign_ms=float(d.get("t_bin_assign_ms", 0.0)),
            t_bin_moment_ms=float(d.get("t_bin_moment_ms", 0.0)),
            t_matrix_fisher_ms=float(d.get("t_matrix_fisher_ms", 0.0)),
            t_planar_translation_ms=float(d.get("t_planar_translation_ms", 0.0)),
            t_lidar_bucket_iw_ms=float(d.get("t_lidar_bucket_iw_ms", 0.0)),
            t_map_update_ms=float(d.get("t_map_update_ms", 0.0)),
        )


@dataclass
class MinimalScanTape:
    """
    Minimal per-scan tape for crash-tolerant, low-overhead diagnostics.
    Hot path stores only this; full ScanDiagnostics is optional at config.
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
    # Optional timing (when enable_timing)
    t_total_ms: float = 0.0
    t_point_budget_ms: float = 0.0
    t_deskew_ms: float = 0.0
    t_bin_moment_ms: float = 0.0
    t_map_update_ms: float = 0.0


@dataclass
class DiagnosticsLog:
    """Container for all per-scan diagnostics from a run."""

    scans: List[ScanDiagnostics] = field(default_factory=list)
    tape: List[MinimalScanTape] = field(default_factory=list)

    # Run metadata
    run_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_scans: int = 0

    def append(self, diag: ScanDiagnostics):
        """Add a scan's full diagnostics."""
        self.scans.append(diag)
        self.total_scans = len(self.scans)

    def append_tape(self, entry: MinimalScanTape):
        """Add a scan's minimal tape entry (hot path)."""
        self.tape.append(entry)
        self.total_scans = len(self.tape)

    def save_jsonl(self, path: str):
        """Save as JSON Lines (one JSON object per line for streaming)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            # Write header with metadata
            header = {
                "_type": "header",
                "run_id": self.run_id,
                "start_time": self.start_time,
                "total_scans": self.total_scans,
            }
            f.write(json.dumps(header) + "\n")

            # Write each scan
            for diag in self.scans:
                f.write(json.dumps(diag.to_dict()) + "\n")

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
                    log.scans.append(ScanDiagnostics.from_dict(d))
        log.total_scans = len(log.scans)
        return log

    def save_npz(self, path: str):
        """Save as compressed NumPy archive (more efficient for large runs).
        If only tape entries exist, saves minimal format; otherwise full ScanDiagnostics format.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        n_full = len(self.scans)
        n_tape = len(self.tape)
        if n_full > 0:
            self._save_npz_full(path, n_full)
        elif n_tape > 0:
            self._save_npz_tape(path, n_tape)
        else:
            np.savez_compressed(path, n_scans=0)

    def _save_npz_tape(self, path: str, n: int):
        """Save minimal tape format (crash-tolerant, low overhead)."""
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
            "t_total_ms": np.array([t.t_total_ms for t in self.tape]),
            "t_point_budget_ms": np.array([t.t_point_budget_ms for t in self.tape]),
            "t_deskew_ms": np.array([t.t_deskew_ms for t in self.tape]),
            "t_bin_moment_ms": np.array([t.t_bin_moment_ms for t in self.tape]),
            "t_map_update_ms": np.array([t.t_map_update_ms for t in self.tape]),
        }
        np.savez_compressed(path, **data)

    def _save_npz_full(self, path: str, n: int):
        """Save full ScanDiagnostics format (legacy)."""
        data = {
            "n_scans": n,
            "run_id": self.run_id,
            "start_time": self.start_time,
            "scan_numbers": np.array([s.scan_number for s in self.scans]),
            "timestamps": np.array([s.timestamp for s in self.scans]),
            "dt_secs": np.array([s.dt_sec for s in self.scans]),
            "dt_scan": np.array([s.dt_scan for s in self.scans]),
            "dt_int": np.array([s.dt_int for s in self.scans]),
            "num_imu_samples": np.array([s.num_imu_samples for s in self.scans]),
            "n_points_raw": np.array([s.n_points_raw for s in self.scans]),
            "n_points_budget": np.array([s.n_points_budget for s in self.scans]),
            "p_W": np.stack([s.p_W for s in self.scans]),  # (n, 3)
            "R_WL": np.stack([s.R_WL for s in self.scans]),  # (n, 3, 3)
            "N_bins": np.stack([s.N_bins for s in self.scans]),  # (n, 48)
            "S_bins": np.stack([s.S_bins for s in self.scans]),  # (n, 48, 3)
            "kappa_bins": np.stack([s.kappa_bins for s in self.scans]),  # (n, 48)
            "kappa_map_bins": np.stack([s.kappa_map_bins for s in self.scans]),  # (n, 48)
            "L_total": np.stack([s.L_total for s in self.scans]),  # (n, 22, 22)
            "h_total": np.stack([s.h_total for s in self.scans]),  # (n, 22)
            # Scalar diagnostics
            "logdet_L_total": np.array([s.logdet_L_total for s in self.scans]),
            "trace_L_total": np.array([s.trace_L_total for s in self.scans]),
            "L_dt": np.array([s.L_dt for s in self.scans]),
            "trace_L_ex": np.array([s.trace_L_ex for s in self.scans]),
            "s_dt": np.array([s.s_dt for s in self.scans]),
            "s_ex": np.array([s.s_ex for s in self.scans]),
            "psd_delta_fro": np.array([s.psd_delta_fro for s in self.scans]),
            "psd_min_eig_before": np.array([s.psd_min_eig_before for s in self.scans]),
            "psd_min_eig_after": np.array([s.psd_min_eig_after for s in self.scans]),
            "trace_Q_mode": np.array([s.trace_Q_mode for s in self.scans]),
            "trace_Sigma_lidar_mode": np.array([s.trace_Sigma_lidar_mode for s in self.scans]),
            "trace_Sigma_g_mode": np.array([s.trace_Sigma_g_mode for s in self.scans]),
            "trace_Sigma_a_mode": np.array([s.trace_Sigma_a_mode for s in self.scans]),
            "wahba_cost": np.array([s.wahba_cost for s in self.scans]),
            "mf_svd": np.stack([s.mf_svd for s in self.scans]),  # (n, 3)
            "scan_scatter_eigs": np.stack([s.scan_scatter_eigs for s in self.scans]),  # (n, 48, 3)
            "map_scatter_eigs": np.stack([s.map_scatter_eigs for s in self.scans]),  # (n, 48, 3)
            "translation_residual_norm": np.array([s.translation_residual_norm for s in self.scans]),
            "rot_err_lidar_deg_pred": np.array([s.rot_err_lidar_deg_pred for s in self.scans]),
            "rot_err_lidar_deg_post": np.array([s.rot_err_lidar_deg_post for s in self.scans]),
            "rot_err_odom_deg_pred": np.array([s.rot_err_odom_deg_pred for s in self.scans]),
            "rot_err_odom_deg_post": np.array([s.rot_err_odom_deg_post for s in self.scans]),
            "fusion_alpha": np.array([s.fusion_alpha for s in self.scans]),
            "total_trigger_magnitude": np.array([s.total_trigger_magnitude for s in self.scans]),
            "conditioning_number": np.array([s.conditioning_number for s in self.scans]),
            "conditioning_pose6": np.array([s.conditioning_pose6 for s in self.scans]),
            "accel_dir_dot_mu0": np.array([s.accel_dir_dot_mu0 for s in self.scans]),
            "accel_mag_mean": np.array([s.accel_mag_mean for s in self.scans]),
            "imu_a_body_mean": np.stack([s.imu_a_body_mean for s in self.scans]),
            "imu_a_world_nog_mean": np.stack([s.imu_a_world_nog_mean for s in self.scans]),
            "imu_a_world_mean": np.stack([s.imu_a_world_mean for s in self.scans]),
            "imu_dt_eff_sum": np.array([s.imu_dt_eff_sum for s in self.scans]),
            "imu_dt_valid_min": np.array([s.imu_dt_valid_min for s in self.scans]),
            "imu_dt_valid_max": np.array([s.imu_dt_valid_max for s in self.scans]),
            "imu_dt_valid_mean": np.array([s.imu_dt_valid_mean for s in self.scans]),
            "imu_dt_valid_median": np.array([s.imu_dt_valid_median for s in self.scans]),
            "imu_dt_valid_std": np.array([s.imu_dt_valid_std for s in self.scans]),
            "imu_dt_valid_nonpos": np.array([s.imu_dt_valid_nonpos for s in self.scans]),
            "imu_dt_weighted_mean": np.array([s.imu_dt_weighted_mean for s in self.scans]),
            "imu_dt_weighted_std": np.array([s.imu_dt_weighted_std for s in self.scans]),
            "imu_dt_weighted_sum": np.array([s.imu_dt_weighted_sum for s in self.scans]),
            "dyaw_gyro": np.array([s.dyaw_gyro for s in self.scans]),
            "dyaw_odom": np.array([s.dyaw_odom for s in self.scans]),
            "dyaw_wahba": np.array([s.dyaw_wahba for s in self.scans]),
            "t_total_ms": np.array([s.t_total_ms for s in self.scans]),
            "t_point_budget_ms": np.array([s.t_point_budget_ms for s in self.scans]),
            "t_imu_preint_scan_ms": np.array([s.t_imu_preint_scan_ms for s in self.scans]),
            "t_imu_preint_int_ms": np.array([s.t_imu_preint_int_ms for s in self.scans]),
            "t_deskew_ms": np.array([s.t_deskew_ms for s in self.scans]),
            "t_bin_assign_ms": np.array([s.t_bin_assign_ms for s in self.scans]),
            "t_bin_moment_ms": np.array([s.t_bin_moment_ms for s in self.scans]),
            "t_matrix_fisher_ms": np.array([s.t_matrix_fisher_ms for s in self.scans]),
            "t_planar_translation_ms": np.array([s.t_planar_translation_ms for s in self.scans]),
            "t_lidar_bucket_iw_ms": np.array([s.t_lidar_bucket_iw_ms for s in self.scans]),
            "t_map_update_ms": np.array([s.t_map_update_ms for s in self.scans]),
        }

        # Optional individual evidence components
        if self.scans[0].L_lidar is not None:
            data["L_lidar"] = np.stack([s.L_lidar for s in self.scans])
        if self.scans[0].L_odom is not None:
            data["L_odom"] = np.stack([s.L_odom for s in self.scans])
        if self.scans[0].L_imu is not None:
            data["L_imu"] = np.stack([s.L_imu for s in self.scans])
        if self.scans[0].L_gyro is not None:
            data["L_gyro"] = np.stack([s.L_gyro for s in self.scans])

        np.savez_compressed(path, **data)

    @classmethod
    def load_npz(cls, path: str) -> "DiagnosticsLog":
        """Load from NumPy archive."""
        data = np.load(path, allow_pickle=True)

        n = int(data["n_scans"])
        if n == 0:
            return cls()

        log = cls()
        log.run_id = str(data.get("run_id", ""))
        log.start_time = float(data.get("start_time", 0.0))

        for i in range(n):
            diag = ScanDiagnostics(
                scan_number=int(data["scan_numbers"][i]),
                timestamp=float(data["timestamps"][i]),
                dt_sec=float(data["dt_secs"][i]),
                dt_scan=float(data["dt_scan"][i]) if "dt_scan" in data else 0.0,
                dt_int=float(data["dt_int"][i]) if "dt_int" in data else 0.0,
                num_imu_samples=int(data["num_imu_samples"][i]) if "num_imu_samples" in data else 0,
                n_points_raw=int(data["n_points_raw"][i]),
                n_points_budget=int(data["n_points_budget"][i]),
                p_W=data["p_W"][i],
                R_WL=data["R_WL"][i],
                N_bins=data["N_bins"][i],
                S_bins=data["S_bins"][i],
                kappa_bins=data["kappa_bins"][i],
                kappa_map_bins=data["kappa_map_bins"][i] if "kappa_map_bins" in data else np.zeros((48,), dtype=np.float64),
                L_total=data["L_total"][i],
                h_total=data["h_total"][i],
                L_lidar=data["L_lidar"][i] if "L_lidar" in data else None,
                L_odom=data["L_odom"][i] if "L_odom" in data else None,
                L_imu=data["L_imu"][i] if "L_imu" in data else None,
                L_gyro=data["L_gyro"][i] if "L_gyro" in data else None,
                logdet_L_total=float(data["logdet_L_total"][i]),
                trace_L_total=float(data["trace_L_total"][i]),
                L_dt=float(data["L_dt"][i]),
                trace_L_ex=float(data["trace_L_ex"][i]),
                s_dt=float(data["s_dt"][i]),
                s_ex=float(data["s_ex"][i]),
                psd_delta_fro=float(data["psd_delta_fro"][i]),
                psd_min_eig_before=float(data["psd_min_eig_before"][i]),
                psd_min_eig_after=float(data["psd_min_eig_after"][i]),
                trace_Q_mode=float(data["trace_Q_mode"][i]),
                trace_Sigma_lidar_mode=float(data["trace_Sigma_lidar_mode"][i]),
                trace_Sigma_g_mode=float(data["trace_Sigma_g_mode"][i]),
                trace_Sigma_a_mode=float(data["trace_Sigma_a_mode"][i]),
                wahba_cost=float(data["wahba_cost"][i]),
                translation_residual_norm=float(data["translation_residual_norm"][i]),
                mf_svd=data["mf_svd"][i] if "mf_svd" in data else np.zeros((3,), dtype=np.float64),
                scan_scatter_eigs=data["scan_scatter_eigs"][i] if "scan_scatter_eigs" in data else np.zeros((48, 3), dtype=np.float64),
                map_scatter_eigs=data["map_scatter_eigs"][i] if "map_scatter_eigs" in data else np.zeros((48, 3), dtype=np.float64),
                rot_err_lidar_deg_pred=float(data["rot_err_lidar_deg_pred"][i]) if "rot_err_lidar_deg_pred" in data else 0.0,
                rot_err_lidar_deg_post=float(data["rot_err_lidar_deg_post"][i]) if "rot_err_lidar_deg_post" in data else 0.0,
                rot_err_odom_deg_pred=float(data["rot_err_odom_deg_pred"][i]) if "rot_err_odom_deg_pred" in data else 0.0,
                rot_err_odom_deg_post=float(data["rot_err_odom_deg_post"][i]) if "rot_err_odom_deg_post" in data else 0.0,
                fusion_alpha=float(data["fusion_alpha"][i]),
                total_trigger_magnitude=float(data["total_trigger_magnitude"][i]),
                conditioning_number=float(data["conditioning_number"][i]),
                conditioning_pose6=float(data["conditioning_pose6"][i]) if "conditioning_pose6" in data else 1.0,
                accel_dir_dot_mu0=float(data["accel_dir_dot_mu0"][i]) if "accel_dir_dot_mu0" in data else 0.0,
                accel_mag_mean=float(data["accel_mag_mean"][i]) if "accel_mag_mean" in data else 0.0,
                imu_a_body_mean=data["imu_a_body_mean"][i] if "imu_a_body_mean" in data else np.zeros((3,), dtype=np.float64),
                imu_a_world_nog_mean=data["imu_a_world_nog_mean"][i] if "imu_a_world_nog_mean" in data else np.zeros((3,), dtype=np.float64),
                imu_a_world_mean=data["imu_a_world_mean"][i] if "imu_a_world_mean" in data else np.zeros((3,), dtype=np.float64),
                imu_dt_eff_sum=float(data["imu_dt_eff_sum"][i]) if "imu_dt_eff_sum" in data else 0.0,
                imu_dt_valid_min=float(data["imu_dt_valid_min"][i]) if "imu_dt_valid_min" in data else 0.0,
                imu_dt_valid_max=float(data["imu_dt_valid_max"][i]) if "imu_dt_valid_max" in data else 0.0,
                imu_dt_valid_mean=float(data["imu_dt_valid_mean"][i]) if "imu_dt_valid_mean" in data else 0.0,
                imu_dt_valid_median=float(data["imu_dt_valid_median"][i]) if "imu_dt_valid_median" in data else 0.0,
                imu_dt_valid_std=float(data["imu_dt_valid_std"][i]) if "imu_dt_valid_std" in data else 0.0,
                imu_dt_valid_nonpos=int(data["imu_dt_valid_nonpos"][i]) if "imu_dt_valid_nonpos" in data else 0,
                imu_dt_weighted_mean=float(data["imu_dt_weighted_mean"][i]) if "imu_dt_weighted_mean" in data else 0.0,
                imu_dt_weighted_std=float(data["imu_dt_weighted_std"][i]) if "imu_dt_weighted_std" in data else 0.0,
                imu_dt_weighted_sum=float(data["imu_dt_weighted_sum"][i]) if "imu_dt_weighted_sum" in data else 0.0,
                dyaw_gyro=float(data["dyaw_gyro"][i]) if "dyaw_gyro" in data else 0.0,
                dyaw_odom=float(data["dyaw_odom"][i]) if "dyaw_odom" in data else 0.0,
                dyaw_wahba=float(data["dyaw_wahba"][i]) if "dyaw_wahba" in data else 0.0,
                t_total_ms=float(data["t_total_ms"][i]) if "t_total_ms" in data else 0.0,
                t_point_budget_ms=float(data["t_point_budget_ms"][i]) if "t_point_budget_ms" in data else 0.0,
                t_imu_preint_scan_ms=float(data["t_imu_preint_scan_ms"][i]) if "t_imu_preint_scan_ms" in data else 0.0,
                t_imu_preint_int_ms=float(data["t_imu_preint_int_ms"][i]) if "t_imu_preint_int_ms" in data else 0.0,
                t_deskew_ms=float(data["t_deskew_ms"][i]) if "t_deskew_ms" in data else 0.0,
                t_bin_assign_ms=float(data["t_bin_assign_ms"][i]) if "t_bin_assign_ms" in data else 0.0,
                t_bin_moment_ms=float(data["t_bin_moment_ms"][i]) if "t_bin_moment_ms" in data else 0.0,
                t_matrix_fisher_ms=float(data["t_matrix_fisher_ms"][i]) if "t_matrix_fisher_ms" in data else 0.0,
                t_planar_translation_ms=float(data["t_planar_translation_ms"][i]) if "t_planar_translation_ms" in data else 0.0,
                t_lidar_bucket_iw_ms=float(data["t_lidar_bucket_iw_ms"][i]) if "t_lidar_bucket_iw_ms" in data else 0.0,
                t_map_update_ms=float(data["t_map_update_ms"][i]) if "t_map_update_ms" in data else 0.0,
            )
            log.scans.append(diag)

        log.total_scans = n
        return log
