"""
Stage-0 Per-Scan Diagnostics for Golden Child SLAM v2.

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
    
    # IMU discretization diagnostics
    dt_scan: float = 0.0  # LiDAR scan duration
    dt_int: float = 0.0  # IMU-covered time (Σ_i Δt_i over actual sample intervals)
    num_imu_samples: int = 0  # Number of IMU samples used

    # Frame coherence probes (base-frame IMU sanity)
    accel_dir_dot_mu0: float = 0.0  # xbar · mu0 in body frame (should be near +1 when consistent)
    accel_mag_mean: float = 0.0     # mean ||a|| (m/s^2), should be near 9.81 when stationary

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
            "rot_err_lidar_deg_pred": self.rot_err_lidar_deg_pred,
            "rot_err_lidar_deg_post": self.rot_err_lidar_deg_post,
            "rot_err_odom_deg_pred": self.rot_err_odom_deg_pred,
            "rot_err_odom_deg_post": self.rot_err_odom_deg_post,
            "fusion_alpha": self.fusion_alpha,
            "total_trigger_magnitude": self.total_trigger_magnitude,
            "conditioning_number": self.conditioning_number,
            "dt_scan": self.dt_scan,
            "dt_int": self.dt_int,
            "num_imu_samples": self.num_imu_samples,
            "accel_dir_dot_mu0": self.accel_dir_dot_mu0,
            "accel_mag_mean": self.accel_mag_mean,
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
            rot_err_lidar_deg_pred=d.get("rot_err_lidar_deg_pred", 0.0),
            rot_err_lidar_deg_post=d.get("rot_err_lidar_deg_post", 0.0),
            rot_err_odom_deg_pred=d.get("rot_err_odom_deg_pred", 0.0),
            rot_err_odom_deg_post=d.get("rot_err_odom_deg_post", 0.0),
            fusion_alpha=d.get("fusion_alpha", 1.0),
            total_trigger_magnitude=d.get("total_trigger_magnitude", 0.0),
            conditioning_number=d.get("conditioning_number", 1.0),
            dt_scan=d.get("dt_scan", 0.0),
            dt_int=d.get("dt_int", 0.0),
            num_imu_samples=d.get("num_imu_samples", 0),
            accel_dir_dot_mu0=d.get("accel_dir_dot_mu0", 0.0),
            accel_mag_mean=d.get("accel_mag_mean", 0.0),
        )


@dataclass
class DiagnosticsLog:
    """Container for all per-scan diagnostics from a run."""

    scans: List[ScanDiagnostics] = field(default_factory=list)

    # Run metadata
    run_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_scans: int = 0

    def append(self, diag: ScanDiagnostics):
        """Add a scan's diagnostics."""
        self.scans.append(diag)
        self.total_scans = len(self.scans)

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
        """Save as compressed NumPy archive (more efficient for large runs)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        n = len(self.scans)
        if n == 0:
            np.savez_compressed(path, n_scans=0)
            return

        # Stack arrays for efficient storage
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
            "translation_residual_norm": np.array([s.translation_residual_norm for s in self.scans]),
            "rot_err_lidar_deg_pred": np.array([s.rot_err_lidar_deg_pred for s in self.scans]),
            "rot_err_lidar_deg_post": np.array([s.rot_err_lidar_deg_post for s in self.scans]),
            "rot_err_odom_deg_pred": np.array([s.rot_err_odom_deg_pred for s in self.scans]),
            "rot_err_odom_deg_post": np.array([s.rot_err_odom_deg_post for s in self.scans]),
            "fusion_alpha": np.array([s.fusion_alpha for s in self.scans]),
            "total_trigger_magnitude": np.array([s.total_trigger_magnitude for s in self.scans]),
            "conditioning_number": np.array([s.conditioning_number for s in self.scans]),
            "accel_dir_dot_mu0": np.array([s.accel_dir_dot_mu0 for s in self.scans]),
            "accel_mag_mean": np.array([s.accel_mag_mean for s in self.scans]),
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
                rot_err_lidar_deg_pred=float(data["rot_err_lidar_deg_pred"][i]) if "rot_err_lidar_deg_pred" in data else 0.0,
                rot_err_lidar_deg_post=float(data["rot_err_lidar_deg_post"][i]) if "rot_err_lidar_deg_post" in data else 0.0,
                rot_err_odom_deg_pred=float(data["rot_err_odom_deg_pred"][i]) if "rot_err_odom_deg_pred" in data else 0.0,
                rot_err_odom_deg_post=float(data["rot_err_odom_deg_post"][i]) if "rot_err_odom_deg_post" in data else 0.0,
                fusion_alpha=float(data["fusion_alpha"][i]),
                total_trigger_magnitude=float(data["total_trigger_magnitude"][i]),
                conditioning_number=float(data["conditioning_number"][i]),
                accel_dir_dot_mu0=float(data["accel_dir_dot_mu0"][i]) if "accel_dir_dot_mu0" in data else 0.0,
                accel_mag_mean=float(data["accel_mag_mean"][i]) if "accel_mag_mean" in data else 0.0,
            )
            log.scans.append(diag)

        log.total_scans = n
        return log
