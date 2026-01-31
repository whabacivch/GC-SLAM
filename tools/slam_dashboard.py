#!/usr/bin/env python3
"""
Golden Child SLAM v2 diagnostics dashboard.

Loads an NPZ from a SLAM run and builds an interactive HTML dashboard (Plotly.js from CDN).
Panels: A = timeline (conditioning, MF, posterior health), B = L_pose6 heatmap, C = 3D trajectory, D = factor influence + top-K bins.

Trajectory: The pipeline always writes an estimated trajectory to a TUM file (trajectory_export_path).
Full-diagnostics NPZ stores p_W per scan; minimal-tape NPZ does not (to keep the hot path cheap).
When loading a minimal-tape NPZ from a results directory, the dashboard loads estimated_trajectory.tum
from the same dir and interpolates at scan timestamps so Panel C can still show the trajectory.

Usage:
  tools/slam_dashboard.py <diagnostics.npz> [--output dashboard.html] [--scan N] [--ground-truth path.tum]
  Ground truth is auto-detected from results/gc_*/ground_truth_aligned.tum when present.
  Ground truth is cropped to the run's time window (first to last scan timestamp) so only
  the segment corresponding to the actual run is shown (e.g. first minute of bag => first minute of GT).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
FL_WS_SRC = PROJECT_ROOT / "fl_ws" / "src" / "fl_slam_poc"
sys.path.insert(0, str(FL_WS_SRC))


def load_diagnostics_npz(path: str) -> dict:
    """Load diagnostics from NPZ file into a dictionary."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_tum_positions(path: str):
    """
    Load x, y, z positions and timestamps from a TUM trajectory file.
    TUM format: timestamp x y z qx qy qz qw (space-separated, # for comments).
    Returns (x, y, z, timestamps) as 1D numpy arrays, or None if file missing/unreadable.
    """
    try:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        if not rows:
            return None
        arr = np.array(rows, dtype=np.float64)
        return arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 0]  # x, y, z, timestamps
    except (OSError, ValueError) as e:
        print(f"Warning: Could not load ground truth from {path}: {e}")
        return None


def interpolate_gt_at_times(gt_x: np.ndarray, gt_y: np.ndarray, gt_z: np.ndarray, gt_ts: np.ndarray, query_ts: np.ndarray):
    """
    Linear interpolation of GT trajectory at query timestamps.
    gt_* and gt_ts are 1D arrays (same length); query_ts is 1D.
    Returns (x, y, z) at query_ts, clipped to GT time range.
    """
    if gt_ts.size < 2 or query_ts.size == 0:
        if gt_ts.size == 1 and query_ts.size > 0:
            return np.full_like(query_ts, gt_x[0]), np.full_like(query_ts, gt_y[0]), np.full_like(query_ts, gt_z[0])
        return np.array([]), np.array([]), np.array([])
    x_out = np.interp(query_ts, gt_ts, gt_x)
    y_out = np.interp(query_ts, gt_ts, gt_y)
    z_out = np.interp(query_ts, gt_ts, gt_z)
    return x_out, y_out, z_out


def numpy_to_json(obj):
    """Convert numpy arrays to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json(v) for v in obj]
    return obj


def open_browser_wayland_compatible(file_path: str) -> bool:
    """
    Open a file in the default browser, compatible with Wayland and X11.
    
    Tries multiple methods:
    1. xdg-open (works on both Wayland and X11)
    2. $BROWSER environment variable
    3. webbrowser module (fallback)
    
    Returns True if successful, False otherwise.
    """
    file_path = os.path.abspath(file_path)
    file_url = f"file://{urllib.parse.quote(file_path, safe='/')}"
    
    # Method 1: Try xdg-open (works on Wayland and X11)
    try:
        subprocess.Popen(
            ["xdg-open", file_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        return True
    except (FileNotFoundError, OSError):
        pass
    
    # Method 2: Try $BROWSER environment variable
    browser = os.environ.get("BROWSER")
    if browser:
        try:
            # Handle browsers that need the URL as an argument
            if "%s" in browser or "%u" in browser:
                cmd = browser.replace("%s", file_url).replace("%u", file_url).split()
            else:
                cmd = [browser, file_url]
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            return True
        except (FileNotFoundError, OSError):
            pass
    
    # Method 3: Fallback to webbrowser module
    try:
        import webbrowser
        webbrowser.open(file_url)
        return True
    except Exception:
        pass
    
    return False


def create_full_dashboard(
    data: dict,
    selected_scan: int = 0,
    output_path: str = None,
    ground_truth_path: str = None,
    gt_y_flip: bool = False,
) -> str:
    """
    Create and display the fully interactive dashboard.

    All panels update dynamically when the slider is moved.

    Args:
        data: Diagnostics data dictionary
        selected_scan: Initial scan index to display
        output_path: Optional path to save HTML file. If None, uses temp file.
        ground_truth_path: Optional path to ground truth TUM file. If provided,
            loads GT and adds it to the 3D trajectory plot. GT is aligned to the
            estimate frame (same coordinate system; arbitrary start = 0,0,0) so
            we compare internal estimates vs external truth.
        gt_y_flip: If True, negate GT Y when loading (for frame-convention mismatch:
            e.g. if per-scan residual Y is all one sign, try this).
    Returns:
        Path to the created HTML file
    """
    n_scans = int(data.get("n_scans", 0))
    if n_scans == 0:
        print("No scan data found in file")
        return None

    # Detect minimal-tape format (from pipeline optimization: save_full_diagnostics=False)
    is_minimal_tape = (
        data.get("format") == "minimal_tape"
        or ("L_pose6" in data and "L_total" not in data)
    )
    if is_minimal_tape:
        # Alias: minimal tape uses cond_pose6; dashboard expects conditioning_pose6
        if "cond_pose6" in data and "conditioning_pose6" not in data:
            data = dict(data)
            data["conditioning_pose6"] = np.asarray(data["cond_pose6"], dtype=np.float64)
        # Build L_total (22x22) from L_pose6 (6x6) so heatmap and pose6-derived timeline work
        L_pose6_arr = np.asarray(data["L_pose6"], dtype=np.float64)
        if L_pose6_arr.ndim == 2:
            L_pose6_arr = L_pose6_arr[np.newaxis, :, :]
        n_tape = L_pose6_arr.shape[0]
        L_total_full = np.zeros((n_tape, 22, 22), dtype=np.float64)
        L_total_full[:, 0:6, 0:6] = L_pose6_arr
        data = dict(data)
        data["L_total"] = L_total_full

    # Prepare all data for JavaScript
    scan_idx = list(range(n_scans))

    # Timeline data - handle missing keys gracefully
    def safe_list(key, default_val=0.0):
        if key in data:
            arr = np.array(data[key])
            return arr.tolist() if hasattr(arr, 'tolist') else list(arr)
        return [default_val] * n_scans

    timeline_data = {
        "n_scans": n_scans,
        "scan_idx": scan_idx,
        "logdet_L_total": safe_list("logdet_L_total"),
        "trace_L_total": safe_list("trace_L_total"),
        "L_dt": safe_list("L_dt"),
        "trace_L_ex": safe_list("trace_L_ex"),
        "psd_delta_fro": safe_list("psd_delta_fro"),
        "psd_min_eig_after": safe_list("psd_min_eig_after"),
        "trace_Q_mode": safe_list("trace_Q_mode"),
        "trace_Sigma_lidar_mode": safe_list("trace_Sigma_lidar_mode"),
        "s_dt": safe_list("s_dt"),
        "s_ex": safe_list("s_ex"),
        "fusion_alpha": safe_list("fusion_alpha"),
        "dt_secs": safe_list("dt_secs", 0.1),
        "dt_scan": safe_list("dt_scan", 0.1),
        "dt_int": safe_list("dt_int", 0.1),
        "num_imu_samples": safe_list("num_imu_samples", 0),
        "wahba_cost": safe_list("wahba_cost"),
        "translation_residual_norm": safe_list("translation_residual_norm"),
        # Conditioning (prefer pose6 observable subspace; fall back to full if missing)
        "conditioning_number": safe_list("conditioning_number", 1.0),  # full 22x22 (may be dominated by null dirs)
        "conditioning_pose6": safe_list("conditioning_pose6", 1.0),
        # Rotation binding diagnostics (if present)
        "rot_err_lidar_deg_pred": safe_list("rot_err_lidar_deg_pred", 0.0),
        "rot_err_lidar_deg_post": safe_list("rot_err_lidar_deg_post", 0.0),
        "rot_err_odom_deg_pred": safe_list("rot_err_odom_deg_pred", 0.0),
        "rot_err_odom_deg_post": safe_list("rot_err_odom_deg_post", 0.0),
        # Yaw increment diagnostics (degrees)
        "dyaw_gyro": safe_list("dyaw_gyro", 0.0),
        "dyaw_odom": safe_list("dyaw_odom", 0.0),
        "dyaw_wahba": safe_list("dyaw_wahba", 0.0),
    }

    # Precompute log10 conditioning for visualization stability
    cond_src = "conditioning_pose6" if ("conditioning_pose6" in data) else "conditioning_number"
    cond = np.asarray(timeline_data.get(cond_src, timeline_data["conditioning_number"]), dtype=float)
    cond = np.where(np.isfinite(cond) & (cond > 1.0), cond, 1.0)
    timeline_data["log10_cond_pose6"] = np.log10(cond).tolist()

    # Bin statistics
    N_bins = data.get("N_bins", np.zeros((n_scans, 48)))
    kappa_bins = data.get("kappa_bins", np.zeros((n_scans, 48)))
    kappa_map_bins = data.get("kappa_map_bins", np.zeros((n_scans, 48)))
    if hasattr(N_bins, 'tolist'):
        N_bins = np.array(N_bins)
    if hasattr(kappa_bins, 'tolist'):
        kappa_bins = np.array(kappa_bins)
    if hasattr(kappa_map_bins, 'tolist'):
        kappa_map_bins = np.array(kappa_map_bins)

    # Optional: Matrix Fisher singular values and scatter eigenvalues (newer schema)
    mf_svd = np.array(data.get("mf_svd", np.zeros((n_scans, 3))), dtype=np.float64)
    scan_scatter_eigs = np.array(data.get("scan_scatter_eigs", np.zeros((n_scans, 48, 3))), dtype=np.float64)
    map_scatter_eigs = np.array(data.get("map_scatter_eigs", np.zeros((n_scans, 48, 3))), dtype=np.float64)

    timeline_data["sum_N"] = np.sum(N_bins, axis=1).tolist()
    timeline_data["mean_kappa"] = np.mean(kappa_bins, axis=1).tolist()

    # =========================================================================
    # MF health + scatter sentinels + posterior subspace health (Panel A)
    # =========================================================================
    L_total_arr = np.array(data.get("L_total", np.zeros((n_scans, 22, 22))))
    logdet_L_pose6_list = []
    pose6_eigmin_list, pose6_eig2_list, pose6_eig3_list = [], [], []
    xy_yaw_eigmin_list, xy_yaw_cond_log10_list = [], []
    for i in range(n_scans):
        L_pose6 = L_total_arr[i, 0:6, 0:6]
        eigvals = np.sort(np.linalg.eigvalsh(L_pose6))  # ascending
        eigvals_pos = np.maximum(eigvals, 1e-12)
        logdet_L_pose6_list.append(float(np.sum(np.log(eigvals_pos))))
        pose6_eigmin_list.append(float(eigvals[0]) if len(eigvals) > 0 else 0.0)
        pose6_eig2_list.append(float(eigvals[1]) if len(eigvals) > 1 else 0.0)
        pose6_eig3_list.append(float(eigvals[2]) if len(eigvals) > 2 else 0.0)

        # Observable ground-robot-ish subspace: (x, y, yaw)
        idx = [0, 1, 5]
        L_xy_yaw = L_total_arr[i][np.ix_(idx, idx)]
        eig_xy_yaw = np.sort(np.linalg.eigvalsh(L_xy_yaw))
        xy_yaw_eigmin_list.append(float(eig_xy_yaw[0]) if len(eig_xy_yaw) > 0 else 0.0)
        lam_min = float(max(eig_xy_yaw[0], 1e-18)) if len(eig_xy_yaw) > 0 else 1e-18
        lam_max = float(max(eig_xy_yaw[-1], lam_min)) if len(eig_xy_yaw) > 0 else lam_min
        xy_yaw_cond_log10_list.append(float(np.log10(lam_max / lam_min)) if lam_min > 0 else 0.0)

    timeline_data["logdet_L_pose6"] = logdet_L_pose6_list
    timeline_data["eigmin_L_pose6"] = pose6_eigmin_list
    timeline_data["eig2_L_pose6"] = pose6_eig2_list
    timeline_data["eig3_L_pose6"] = pose6_eig3_list
    timeline_data["eigmin_L_xy_yaw"] = xy_yaw_eigmin_list
    timeline_data["log10_cond_xy_yaw"] = xy_yaw_cond_log10_list

    # Z-leak sentinel ratios (total and LiDAR factor): L[z,z] / mean(L[x,x],L[y,y])
    xy_mean_total = 0.5 * (L_total_arr[:, 0, 0] + L_total_arr[:, 1, 1]) + 1e-12
    timeline_data["z_leak_ratio_total"] = (L_total_arr[:, 2, 2] / xy_mean_total).tolist()

    # Matrix Fisher health signals (singular values + condition proxy)
    mf_svd = np.asarray(mf_svd, dtype=np.float64).reshape(n_scans, 3)
    timeline_data["mf_s1"] = mf_svd[:, 0].tolist()
    timeline_data["mf_s2"] = mf_svd[:, 1].tolist()
    timeline_data["mf_s3"] = mf_svd[:, 2].tolist()
    timeline_data["mf_s1_f"] = np.maximum(mf_svd[:, 0], 1e-12).tolist()
    timeline_data["mf_s2_f"] = np.maximum(mf_svd[:, 1], 1e-12).tolist()
    timeline_data["mf_s3_f"] = np.maximum(mf_svd[:, 2], 1e-12).tolist()
    mf_kappa = np.sum(mf_svd, axis=1)
    # Fallback: older logs stored only wahba_cost as sum(svd); preserve visibility.
    if np.all(mf_kappa == 0.0) and ("wahba_cost" in data):
        mf_kappa = np.asarray(data["wahba_cost"], dtype=np.float64)
    timeline_data["mf_kappa"] = mf_kappa.tolist()
    eps = 1e-12
    mf_cond = (mf_svd[:, 0] + eps) / (mf_svd[:, 2] + eps)
    timeline_data["log10_mf_cond"] = np.log10(mf_cond).tolist()

    # Log-scaled posterior spectrum sentinels
    timeline_data["log10_eigmin_L_pose6"] = np.log10(np.maximum(np.asarray(pose6_eigmin_list), 1e-12)).tolist()
    timeline_data["log10_eigmin_L_xy_yaw"] = np.log10(np.maximum(np.asarray(xy_yaw_eigmin_list), 1e-12)).tolist()

    # MF row (Panel A row 1) log-scale range so all three singular values are visible and nothing clips
    mf_vals = np.concatenate([
        np.asarray(timeline_data["mf_s1_f"]),
        np.asarray(timeline_data["mf_s2_f"]),
        np.asarray(timeline_data["mf_s3_f"]),
    ])
    mf_finite = mf_vals[np.isfinite(mf_vals) & (mf_vals > 0)]
    if mf_finite.size > 0:
        mf_lo = max(1e-12, float(np.percentile(mf_finite, 0.5)))
        mf_hi = float(np.percentile(mf_finite, 99.5))
        mf_hi = max(mf_hi, mf_lo * 10)
        timeline_data["mf_svd_y_range"] = [mf_lo, mf_hi]
    else:
        timeline_data["mf_svd_y_range"] = [1e-12, 1.0]

    # Scatter health signals (anisotropy/planarity proxies)
    # Eigenvalues are expected in ascending order from eigvalsh: [λ1<=λ2<=λ3].
    # For metrics we use l1=max, l2=mid, l3=min.
    def _scatter_to_metrics(eigs_asc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        l3 = eigs_asc[..., 0]
        l2 = eigs_asc[..., 1]
        l1 = eigs_asc[..., 2]
        denom = np.maximum(l1, 1e-18)
        anisotropy = (l1 - l2) / denom
        planarity = (l2 - l3) / denom
        return anisotropy, planarity

    scan_aniso, scan_planarity = _scatter_to_metrics(scan_scatter_eigs)
    map_aniso, map_planarity = _scatter_to_metrics(map_scatter_eigs)
    timeline_data["median_scan_aniso"] = np.median(scan_aniso, axis=1).tolist()
    timeline_data["median_scan_planarity"] = np.median(scan_planarity, axis=1).tolist()
    timeline_data["median_map_aniso"] = np.median(map_aniso, axis=1).tolist()
    timeline_data["median_map_planarity"] = np.median(map_planarity, axis=1).tolist()

    # Simple support proxies for the dashboard (relative, visualization-only thresholds)
    N_max = np.max(N_bins, axis=1, keepdims=True) + 1e-12
    timeline_data["bins_strong_frac"] = np.mean(N_bins > (0.1 * N_max), axis=1).tolist()
    timeline_data["bins_planar_frac"] = np.mean(map_planarity > 0.6, axis=1).tolist()

    # Factor influence ledger (subspace traces, alpha-scaled; visualization proxy)
    alpha = np.asarray(timeline_data["fusion_alpha"], dtype=np.float64)
    alpha = np.where(np.isfinite(alpha), alpha, 1.0)

    def _subspace_traces(L_fac: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # rot: [3:6], xy: [0:2], z: [2,2]
        rot = np.trace(L_fac[:, 3:6, 3:6], axis1=1, axis2=2)
        xy = np.trace(L_fac[:, 0:2, 0:2], axis1=1, axis2=2)
        z = L_fac[:, 2, 2]
        return rot, xy, z

    for key, prefix in [
        ("L_lidar", "lidar"),
        ("L_odom", "odom"),
        ("L_imu", "imu"),
        ("L_gyro", "gyro"),
        ("L_imu_preint", "preint"),
    ]:
        if key in data:
            L_fac = np.array(data[key], dtype=np.float64)  # (n_scans, 22, 22)
            rot, xy, z = _subspace_traces(L_fac)
            timeline_data[f"I_rot_{prefix}"] = (alpha * rot).tolist()
            timeline_data[f"I_xy_{prefix}"] = (alpha * xy).tolist()
            timeline_data[f"I_z_{prefix}"] = (alpha * z).tolist()
            xy_mean = 0.5 * (L_fac[:, 0, 0] + L_fac[:, 1, 1]) + 1e-12
            timeline_data[f"z_leak_ratio_{prefix}"] = (L_fac[:, 2, 2] / xy_mean).tolist()
        else:
            timeline_data[f"I_rot_{prefix}"] = [0.0] * n_scans
            timeline_data[f"I_xy_{prefix}"] = [0.0] * n_scans
            timeline_data[f"I_z_{prefix}"] = [0.0] * n_scans
            timeline_data[f"z_leak_ratio_{prefix}"] = [0.0] * n_scans

    # Trajectory: one coordinate system = estimate frame (robot's arbitrary start = 0,0,0).
    # We align GT to that start so we compare apples to apples: how good are our estimates vs external truth.
    p_W = data.get("p_W", np.zeros((n_scans, 3)))
    if hasattr(p_W, 'tolist'):
        p_W = np.array(p_W)
    origin_est = np.array(p_W[0], dtype=np.float64)  # estimate frame origin (arbitrary start)
    est_x = (p_W[:, 0] - origin_est[0]).tolist()
    est_y = (p_W[:, 1] - origin_est[1]).tolist()
    est_z = (p_W[:, 2] - origin_est[2]).tolist()
    has_ground_truth = False
    scan_timestamps = np.array(data.get("timestamps", np.zeros(n_scans)), dtype=np.float64)
    if scan_timestamps.size != n_scans:
        scan_timestamps = np.zeros(n_scans)
    if ground_truth_path:
        gt_xyz = load_tum_positions(ground_truth_path)
        if gt_xyz is not None:
            gt_x_raw = np.array(gt_xyz[0], dtype=np.float64)
            gt_y_raw = np.array(gt_xyz[1], dtype=np.float64)
            gt_z_raw = np.array(gt_xyz[2], dtype=np.float64)
            if gt_y_flip:
                gt_y_raw = -gt_y_raw
            gt_ts_arr = np.array(gt_xyz[3], dtype=np.float64)
            # Put GT in the same coordinate system as the estimate: align so at first scan GT = estimate (both 0,0,0).
            # GT_in_est_frame = GT_world + (origin_est - GT_at_t0); then plot GT_in_est_frame - origin_est = GT_world - GT_at_t0.
            if scan_timestamps.size > 0 and gt_ts_arr.size >= 2:
                t0 = float(scan_timestamps[0])
                gt_at_t0 = np.array([
                    np.interp(t0, gt_ts_arr, gt_x_raw),
                    np.interp(t0, gt_ts_arr, gt_y_raw),
                    np.interp(t0, gt_ts_arr, gt_z_raw),
                ], dtype=np.float64)
            else:
                gt_at_t0 = np.array([gt_x_raw[0], gt_y_raw[0], gt_z_raw[0]], dtype=np.float64)
            gt_x_arr = gt_x_raw - gt_at_t0[0]
            gt_y_arr = gt_y_raw - gt_at_t0[1]
            gt_z_arr = gt_z_raw - gt_at_t0[2]
            has_ground_truth = True
            # GT positions at scan timestamps (for markers); GT already in estimate frame (GT - gt_at_t0)
            gt_at_scan_x, gt_at_scan_y, gt_at_scan_z = interpolate_gt_at_times(
                gt_x_arr, gt_y_arr, gt_z_arr,
                gt_ts_arr,
                scan_timestamps,
            )
            # Crop GT to run time window so dashboard only shows ground truth for the actual run
            # (e.g. first minute of bag => only first minute of GT)
            if scan_timestamps.size > 0:
                t_start = float(np.min(scan_timestamps))
                t_end = float(np.max(scan_timestamps))
                run_mask = (gt_ts_arr >= t_start) & (gt_ts_arr <= t_end)
                gt_x_arr = gt_x_arr[run_mask]
                gt_y_arr = gt_y_arr[run_mask]
                gt_z_arr = gt_z_arr[run_mask]
                gt_ts_arr = gt_ts_arr[run_mask]
    # For minimal tape, logdet_L_total is not saved; use logdet_L_pose6 for trajectory color
    logdet_for_trajectory = (
        timeline_data["logdet_L_pose6"]
        if is_minimal_tape
        else timeline_data["logdet_L_total"]
    )
    trajectory_data = {
        "x": est_x,
        "y": est_y,
        "z": est_z,
        "logdet": logdet_for_trajectory,
        "has_ground_truth": has_ground_truth,
        "scan_timestamps": scan_timestamps.tolist(),
        "is_minimal_tape": is_minimal_tape,
    }
    if has_ground_truth:
        trajectory_data["gt_x"] = gt_x_arr.tolist()
        trajectory_data["gt_y"] = gt_y_arr.tolist()
        trajectory_data["gt_z"] = gt_z_arr.tolist()
        trajectory_data["gt_timestamps"] = gt_ts_arr.tolist()
        trajectory_data["gt_at_scan_x"] = gt_at_scan_x.tolist()
        trajectory_data["gt_at_scan_y"] = gt_at_scan_y.tolist()
        trajectory_data["gt_at_scan_z"] = gt_at_scan_z.tolist()
        # Per-scan residual (GT - estimate) in estimate frame: both in same coordinate system
        res_x = (gt_at_scan_x - (p_W[:, 0] - origin_est[0])).tolist()
        res_y = (gt_at_scan_y - (p_W[:, 1] - origin_est[1])).tolist()
        res_z = (gt_at_scan_z - (p_W[:, 2] - origin_est[2])).tolist()
        trajectory_data["res_x"] = res_x
        trajectory_data["res_y"] = res_y
        trajectory_data["res_z"] = res_z

    # L matrices for heatmap (all scans)
    L_total = data.get("L_total", np.zeros((n_scans, 22, 22)))
    if hasattr(L_total, 'tolist'):
        L_total = np.array(L_total)

    # S_bins and R_WL for direction glyphs
    S_bins = data.get("S_bins", np.zeros((n_scans, 48, 3)))
    R_WL = data.get("R_WL", np.tile(np.eye(3), (n_scans, 1, 1)))
    if hasattr(S_bins, 'tolist'):
        S_bins = np.array(S_bins)
    if hasattr(R_WL, 'tolist'):
        R_WL = np.array(R_WL)

    # Create HTML with embedded data and interactive JavaScript
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GC SLAM v2 Diagnostics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            text-align: center;
            color: #00d4ff;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            max-width: 1900px;
            margin: 0 auto;
        }}
        .panel {{
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        #offset-table th, #offset-table td {{
            padding: 6px 10px;
            text-align: right;
            border-bottom: 1px solid #2a2a4a;
        }}
        #offset-table th {{ background: #0f3460; position: sticky; top: 0; }}
        #offset-table tbody tr:nth-child(even) {{ background: rgba(15, 52, 96, 0.3); }}
        #offset-table td:first-child, #offset-table th:first-child {{ text-align: right; }}
        #offset-table td:nth-child(2) {{ text-align: right; }}
        .controls {{
            text-align: center;
            margin: 15px 0;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }}
        .controls input[type="range"] {{
            width: 400px;
            height: 8px;
            -webkit-appearance: none;
            background: #0f3460;
            border-radius: 4px;
            outline: none;
        }}
        .controls input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }}
        .controls label {{
            font-weight: bold;
            font-size: 16px;
        }}
        .scan-info {{
            background: #0f3460;
            padding: 8px 15px;
            border-radius: 5px;
            font-family: monospace;
        }}
        .scan-info span {{
            color: #00d4ff;
            font-weight: bold;
        }}
        #z-leak-display {{
            font-size: 1.1rem;
            font-weight: bold;
            padding: 8px 12px;
            margin-bottom: 8px;
            background: #0f3460;
            border-radius: 6px;
            color: #f7b731;
        }}
        #z-leak-display .value {{
            color: #4ecdc4;
        }}
        #scan-display {{
            font-size: 24px;
            color: #00d4ff;
            min-width: 60px;
            display: inline-block;
            text-align: center;
        }}
        .panel-title {{
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stats-row {{
            display: flex;
            gap: 15px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        .stat-box {{
            background: #0f3460;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 12px;
        }}
        .stat-box .label {{
            color: #888;
        }}
        .stat-box .value {{
            color: #00d4ff;
            font-weight: bold;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <h1>Golden Child SLAM v2 Diagnostics Dashboard</h1>
    <p class="subtitle">Interactive per-scan pipeline diagnostics | {n_scans} scans loaded</p>
    <p class="subtitle" id="minimal-tape-notice" style="display: none; color: #f7b731;">Minimal tape mode: trajectory and some panels (MF, bins, factor influence) not recorded. Use save_full_diagnostics=True for full dashboard.</p>

	    <div class="controls">
	        <label>Scan:</label>
	        <span id="scan-display">{selected_scan}</span>
	        <input type="range" id="scan-slider" min="0" max="{n_scans - 1}" value="{selected_scan}">
	        <div class="scan-info">
	            dt: <span id="info-dt">--</span>s |
	            α: <span id="info-alpha">--</span> |
	            log|L|: <span id="info-logdet">--</span> |
	            log10 κ_pose6: <span id="info-cond">--</span> |
	            rot_lidar_post: <span id="info-rot-lidar">--</span>°
	        </div>
	    </div>

    <div class="dashboard-grid">
        <div class="panel full-width">
            <div class="panel-title">Panel A: MF + degeneracy + posterior subspace health</div>
            <div id="timeline"></div>
        </div>
        <div class="panel">
            <div class="panel-title">Panel B: L_pose6 / L_xy_yaw (toggle) + Z leak</div>
            <div id="z-leak-display"><span id="z-leak-value"></span></div>
            <div id="heatmap"></div>
        </div>
        <div class="panel">
            <div class="panel-title">Panel C: 3D Trajectory</div>
            <div id="trajectory-note" style="font-size: 12px; color: #aaa; margin-bottom: 6px;"></div>
            <div id="trajectory"></div>
        </div>
        <div class="panel full-width">
            <div class="panel-title">Panel D: Factor influence ledger + top-K bin anatomy</div>
            <div id="factor-stacked"></div>
            <div id="topk-bins"></div>
        </div>
        <div class="panel full-width" id="offset-table-panel" style="display: none;">
            <div class="panel-title">Per-scan offset (estimate, GT, residual) — same coordinate system (estimate frame, m)</div>
            <div style="overflow-x: auto; max-height: 400px; overflow-y: auto;">
                <table id="offset-table" style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead><tr id="offset-table-head"></tr></thead>
                    <tbody id="offset-table-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
    // Embedded data
    const timelineData = {json.dumps(numpy_to_json(timeline_data))};
    const trajectoryData = {json.dumps(numpy_to_json(trajectory_data))};
	    const L_matrices = {json.dumps(L_total.tolist())};
	    const S_bins = {json.dumps(S_bins.tolist())};
	    const R_WL = {json.dumps(R_WL.tolist())};
	    const kappa_bins = {json.dumps(kappa_bins.tolist())};
	    const kappa_map_bins = {json.dumps(kappa_map_bins.tolist())};
	    const N_bins = {json.dumps(N_bins.tolist())};
	    const scan_aniso = {json.dumps(scan_aniso.tolist())};
	    const scan_planarity = {json.dumps(scan_planarity.tolist())};
	    const map_aniso = {json.dumps(map_aniso.tolist())};
	    const map_planarity = {json.dumps(map_planarity.tolist())};

    let currentScan = {selected_scan};
    const nScans = {n_scans};

    // Block labels for pose heatmap (6x6: translation + rotation)
    const poseBlockBoundaries = [0, 3, 6];
    const poseBlockLabels = ['trans', 'rot'];
    let heatmapMode = 'pose6';  // 'pose6' | 'xy_yaw'

    // Dark theme layout
    const darkLayout = {{
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#0f3460',
        font: {{ color: '#eee' }},
        xaxis: {{ gridcolor: '#1a1a2e', zerolinecolor: '#1a1a2e' }},
        yaxis: {{ gridcolor: '#1a1a2e', zerolinecolor: '#1a1a2e' }},
    }};

	    // =====================================================================
	    // Panel A: MF + degeneracy + posterior subspace health
	    // =====================================================================
	    const TIMELINE_ROWS = 5;
	    const PANEL_A_COLORS = [
	        '#00d4ff', '#4ecdc4', '#45aaf2', '#f7b731', '#26de81', '#a55eea', '#ff6b6b',
	        '#fd9644', '#2bcbba', '#a29bfe', '#55efc4', '#ffeaa7', '#fab1a0', '#81ecec', '#74b9ff', '#fd79a8', '#00b894'
	    ];
		    function createTimeline() {{
		        const traces = [
            // Row 1: Matrix Fisher singular values (log scale; MF health)
            {{ x: timelineData.scan_idx, y: timelineData.mf_s1_f, name: 'MF s1', line: {{color: PANEL_A_COLORS[0]}}, xaxis: 'x', yaxis: 'y' }},
            {{ x: timelineData.scan_idx, y: timelineData.mf_s2_f, name: 'MF s2', line: {{color: PANEL_A_COLORS[1]}}, xaxis: 'x', yaxis: 'y' }},
            {{ x: timelineData.scan_idx, y: timelineData.mf_s3_f, name: 'MF s3', line: {{color: PANEL_A_COLORS[2]}}, xaxis: 'x', yaxis: 'y' }},

            // Row 2: Scatter observability (0..1 proxies)
            {{ x: timelineData.scan_idx, y: timelineData.median_map_aniso, name: 'median(map aniso)', line: {{color: PANEL_A_COLORS[3]}}, xaxis: 'x2', yaxis: 'y2' }},
            {{ x: timelineData.scan_idx, y: timelineData.median_map_planarity, name: 'median(map planarity)', line: {{color: PANEL_A_COLORS[4]}}, xaxis: 'x2', yaxis: 'y2' }},
            {{ x: timelineData.scan_idx, y: timelineData.bins_strong_frac, name: '%bins strong (mass)', line: {{color: PANEL_A_COLORS[5]}}, xaxis: 'x2', yaxis: 'y2' }},
            {{ x: timelineData.scan_idx, y: timelineData.bins_planar_frac, name: '%bins planar (map)', line: {{color: PANEL_A_COLORS[6]}}, xaxis: 'x2', yaxis: 'y2' }},

            // Row 3: Conditioning sentinels (log10)
            {{ x: timelineData.scan_idx, y: timelineData.log10_mf_cond, name: 'log10(MF cond)', line: {{color: PANEL_A_COLORS[7]}}, xaxis: 'x3', yaxis: 'y3' }},
            {{ x: timelineData.scan_idx, y: timelineData.log10_cond_pose6, name: 'log10(cond pose6)', line: {{color: PANEL_A_COLORS[8]}}, xaxis: 'x3', yaxis: 'y3' }},
            {{ x: timelineData.scan_idx, y: timelineData.log10_cond_xy_yaw, name: 'log10(cond xy_yaw)', line: {{color: PANEL_A_COLORS[9]}}, xaxis: 'x3', yaxis: 'y3' }},

            // Row 4: Rotation / yaw agreement (degrees)
            {{ x: timelineData.scan_idx, y: timelineData.rot_err_lidar_deg_pred, name: '||Log(R_predᵀ R_MF)|| (deg)', line: {{color: PANEL_A_COLORS[10]}}, xaxis: 'x4', yaxis: 'y4' }},
            {{ x: timelineData.scan_idx, y: timelineData.dyaw_gyro, name: 'Δyaw gyro', line: {{color: PANEL_A_COLORS[11]}}, xaxis: 'x4', yaxis: 'y4' }},
            {{ x: timelineData.scan_idx, y: timelineData.dyaw_odom, name: 'Δyaw odom', line: {{color: PANEL_A_COLORS[12]}}, xaxis: 'x4', yaxis: 'y4' }},
            {{ x: timelineData.scan_idx, y: timelineData.dyaw_wahba, name: 'Δyaw MF', line: {{color: PANEL_A_COLORS[13]}}, xaxis: 'x4', yaxis: 'y4' }},

            // Row 5: Posterior health (log sentinels)
            {{ x: timelineData.scan_idx, y: timelineData.log10_eigmin_L_pose6, name: 'log10(λmin pose6)', line: {{color: PANEL_A_COLORS[14]}}, xaxis: 'x5', yaxis: 'y5' }},
            {{ x: timelineData.scan_idx, y: timelineData.log10_eigmin_L_xy_yaw, name: 'log10(λmin xy_yaw)', line: {{color: PANEL_A_COLORS[15]}}, xaxis: 'x5', yaxis: 'y5' }},
            {{ x: timelineData.scan_idx, y: timelineData.logdet_L_pose6, name: 'log|L_pose6|', line: {{color: PANEL_A_COLORS[16]}}, xaxis: 'x5', yaxis: 'y5' }},
        ];

        const mfRange = timelineData.mf_svd_y_range || [1e-12, 1];
        const layout = {{
            ...darkLayout,
            height: 720,
            showlegend: true,
            legend: {{ orientation: 'v', x: 1.02, y: 1, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(22, 33, 62, 0.9)', font: {{ size: 10 }} }},
            grid: {{ rows: TIMELINE_ROWS, columns: 1, pattern: 'independent', roworder: 'top to bottom' }},
            xaxis: {{ ...darkLayout.xaxis, anchor: 'y', domain: [0, 1], showticklabels: false }},
            xaxis2: {{ ...darkLayout.xaxis, anchor: 'y2', domain: [0, 1], showticklabels: false }},
            xaxis3: {{ ...darkLayout.xaxis, anchor: 'y3', domain: [0, 1], showticklabels: false }},
            xaxis4: {{ ...darkLayout.xaxis, anchor: 'y4', domain: [0, 1], showticklabels: false }},
            xaxis5: {{ ...darkLayout.xaxis, anchor: 'y5', domain: [0, 1], title: 'Scan Index' }},
            yaxis: {{ ...darkLayout.yaxis, title: 'MF svd (log)', type: 'log', range: mfRange }},
            yaxis2: {{ ...darkLayout.yaxis, title: 'scatter proxies', range: [0, 1.05] }},
            yaxis3: {{ ...darkLayout.yaxis, title: 'log10(cond)' }},
            yaxis4: {{ ...darkLayout.yaxis, title: 'deg' }},
            yaxis5: {{ ...darkLayout.yaxis, title: 'log sentinels' }},
            margin: {{ t: 50, b: 40, l: 70, r: 220 }},
            shapes: createVerticalLines(currentScan, TIMELINE_ROWS),
        }};

        Plotly.newPlot('timeline', traces, layout, {{responsive: true}});
    }}

    function createVerticalLines(scanIdx, numRows) {{
        const shapes = [];
        for (let i = 0; i < numRows; i++) {{
            shapes.push({{
                type: 'line',
                x0: scanIdx, x1: scanIdx,
                y0: 0, y1: 1,
                xref: 'x' + (i === 0 ? '' : (i + 1)),
                yref: 'paper',
                line: {{ color: '#ff6b6b', width: 2, dash: 'dash' }}
            }});
        }}
        return shapes;
    }}

    // =====================================================================
    // Panel B: L_pose6 (6x6) / L_xy_yaw (3x3) toggle + Z leak indicator
    // =====================================================================
	    function updateZLeakDisplay(scanIdx) {{
	        const L_full = L_matrices[scanIdx];
	        const L_pose = [];
	        for (let i = 0; i < 6; i++) L_pose.push(L_full[i].slice(0, 6));
	        const Lzz = L_pose[2][2];
	        const Lxx = L_pose[0][0], Lyy = L_pose[1][1];
	        const xy_mean = (Lxx + Lyy) / 2 + 1e-10;
	        const zLeakRatio = Lzz / xy_mean;
	        const zLeakTotal = timelineData.z_leak_ratio_total ? timelineData.z_leak_ratio_total[scanIdx] : zLeakRatio;
	        const fmt = (v) => (v != null && v !== undefined) ? Number(v).toFixed(4) : '--';
	        const zLeakLidar = timelineData.z_leak_ratio_lidar ? timelineData.z_leak_ratio_lidar[scanIdx] : 0.0;
	        const zLeakOdom = timelineData.z_leak_ratio_odom ? timelineData.z_leak_ratio_odom[scanIdx] : null;
	        const zLeakImu = timelineData.z_leak_ratio_imu ? timelineData.z_leak_ratio_imu[scanIdx] : null;
	        const zLeakGyro = timelineData.z_leak_ratio_gyro ? timelineData.z_leak_ratio_gyro[scanIdx] : null;
	        const zLeakPreint = timelineData.z_leak_ratio_preint ? timelineData.z_leak_ratio_preint[scanIdx] : null;
	        const el = document.getElementById('z-leak-value');
	        if (el) {{
	            el.innerHTML =
	                'Z leak total: <span class="value">' + zLeakTotal.toFixed(4) + '</span> &nbsp;|&nbsp; L_zz=' + Lzz.toFixed(4) + '<br>' +
	                'LiDAR: <span class="value">' + fmt(zLeakLidar) + '</span> (planar=0) &nbsp; ' +
	                'odom: <span class="value">' + fmt(zLeakOdom) + '</span> &nbsp; imu: <span class="value">' + fmt(zLeakImu) + '</span> &nbsp; ' +
	                'gyro: <span class="value">' + fmt(zLeakGyro) + '</span> &nbsp; preint: <span class="value">' + fmt(zLeakPreint) + '</span>';
	        }}
	    }}

    function createHeatmap(scanIdx) {{
        const L_full = L_matrices[scanIdx];
        let L_plot, axisLabels, title, rows, cols;
        const idx_xy_yaw = [0, 1, 5];  // x, y, yaw

        if (heatmapMode === 'xy_yaw') {{
            rows = 3; cols = 3;
            L_plot = [];
            for (let i = 0; i < 3; i++) {{
                const row = [];
                for (let j = 0; j < 3; j++) row.push(L_full[idx_xy_yaw[i]][idx_xy_yaw[j]]);
                L_plot.push(row);
            }}
            axisLabels = ['x', 'y', 'yaw'];
            title = 'L_xy_yaw (3×3) - Scan ' + scanIdx;
        }} else {{
            rows = 6; cols = 6;
            L_plot = [];
            for (let i = 0; i < 6; i++) L_plot.push(L_full[i].slice(0, 6));
            axisLabels = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'];
            title = 'L_pose6 (6×6) - Scan ' + scanIdx;
        }}

        const annotations = [];
        for (let i = 0; i < rows; i++) {{
            annotations.push({{ x: i, y: -0.8, text: axisLabels[i], showarrow: false, font: {{size: 9, color: '#aaa'}} }});
            annotations.push({{ x: -0.8, y: i, text: axisLabels[i], showarrow: false, font: {{size: 9, color: '#aaa'}} }});
        }}

        const trace = {{
            z: L_plot,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0,
            colorbar: {{ title: 'Info', tickfont: {{color: '#eee'}} }}
        }};

        const layout = {{
            ...darkLayout,
            height: 420,
            title: {{ text: title, font: {{color: '#00d4ff'}} }},
            xaxis: {{ ...darkLayout.xaxis, title: 'Column', scaleanchor: 'y', constrain: 'domain' }},
            yaxis: {{ ...darkLayout.yaxis, title: 'Row', autorange: 'reversed', constrain: 'domain' }},
            annotations: annotations,
            margin: {{ t: 50, b: 50, l: 50, r: 30 }},
        }};

        Plotly.react('heatmap', [trace], layout, {{responsive: true}});
        updateZLeakDisplay(scanIdx);
    }}

    function setupHeatmapToggle() {{
        const container = document.getElementById('z-leak-display');
        if (!container) return;
        const btn = document.createElement('button');
        btn.textContent = 'Toggle L_pose6 / L_xy_yaw';
        btn.style.marginLeft = '12px';
        btn.style.padding = '4px 8px';
        btn.style.cursor = 'pointer';
        btn.style.background = '#0f3460';
        btn.style.color = '#00d4ff';
        btn.style.border = '1px solid #00d4ff';
        btn.style.borderRadius = '4px';
        btn.onclick = function() {{
            heatmapMode = heatmapMode === 'pose6' ? 'xy_yaw' : 'pose6';
            createHeatmap(currentScan);
        }};
        container.appendChild(btn);
    }}

    // =====================================================================
    // Panel C: 3D Trajectory
    // =====================================================================
	    function createTrajectory(scanIdx) {{
	        const noteEl = document.getElementById('trajectory-note');
	        if (noteEl) {{
	            const geod = (timelineData.rot_err_lidar_deg_pred && timelineData.rot_err_lidar_deg_pred[scanIdx] !== undefined)
	                ? timelineData.rot_err_lidar_deg_pred[scanIdx].toFixed(2) : '--';
	            const zLeakL = (timelineData.z_leak_ratio_lidar && timelineData.z_leak_ratio_lidar[scanIdx] !== undefined)
	                ? timelineData.z_leak_ratio_lidar[scanIdx].toFixed(4) : '--';
	            const IzL = (timelineData.I_z_lidar && timelineData.I_z_lidar[scanIdx] !== undefined)
	                ? timelineData.I_z_lidar[scanIdx].toFixed(2) : '--';
	            const prefix = trajectoryData.has_ground_truth
	                ? 'GT aligned to estimate frame (arbitrary start = 0,0,0). Both in same coordinate system.'
	                : 'Estimate only (add --ground-truth to overlay).';
	            noteEl.textContent = `${{prefix}} Selected scan=${{scanIdx}} | geodesic_pred→MF=${{geod}}° | z_leak_lidar=${{zLeakL}} | ΔI_z_lidar=${{IzL}}`;
	        }}
        const traces = [];
        // Ground truth (if present; draw first so it appears behind estimated)
        if (trajectoryData.gt_x && trajectoryData.gt_y && trajectoryData.gt_z) {{
            const gtHover = trajectoryData.gt_timestamps
                ? 'Ground truth<br>t=%{{customdata:.3f}} s<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
                : 'Ground truth<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>';
            traces.push({{
                x: trajectoryData.gt_x,
                y: trajectoryData.gt_y,
                z: trajectoryData.gt_z,
                customdata: trajectoryData.gt_timestamps || trajectoryData.gt_x.map((_, i) => i),
                mode: 'lines',
                type: 'scatter3d',
                line: {{ color: '#4ecdc4', width: 5 }},
                name: 'Ground truth',
                hovertemplate: gtHover
            }});
            // GT at scan timestamps (markers aligned with our scans)
            if (trajectoryData.gt_at_scan_x && trajectoryData.gt_at_scan_x.length > 0) {{
                const scanTs = trajectoryData.scan_timestamps || trajectoryData.gt_at_scan_x.map((_, i) => i);
                const scanLabels = trajectoryData.gt_at_scan_x.map((_, i) =>
                    'scan ' + i + ' | t=' + (typeof scanTs[i] === 'number' ? scanTs[i].toFixed(3) : i) + ' s');
                traces.push({{
                    x: trajectoryData.gt_at_scan_x,
                    y: trajectoryData.gt_at_scan_y,
                    z: trajectoryData.gt_at_scan_z,
                    mode: 'markers',
                    type: 'scatter3d',
                    marker: {{ size: 5, color: '#26de81', symbol: 'circle' }},
                    name: 'GT at scan times',
                    text: scanLabels,
                    hovertemplate: '%{{text}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
                }});
            }}
        }}
        // Estimated trajectory (bright color so visible next to ground truth on dark background)
        traces.push({{
            x: trajectoryData.x, y: trajectoryData.y, z: trajectoryData.z,
            mode: 'lines+markers',
            type: 'scatter3d',
            marker: {{
                size: 4,
                color: trajectoryData.logdet,
                colorscale: 'Viridis',
                colorbar: {{ title: 'log|L|', x: 1.02, tickfont: {{color: '#eee'}} }},
                showscale: true
            }},
            line: {{ color: '#f7b731', width: 4 }},
            name: 'Estimated',
            hovertemplate: 'Estimated (scan %{{text}})<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>',
            text: timelineData.scan_idx.map(i => i.toString())
        }});
        // Selected point marker
        traces.push({{
            x: [trajectoryData.x[scanIdx]],
            y: [trajectoryData.y[scanIdx]],
            z: [trajectoryData.z[scanIdx]],
            mode: 'markers',
            type: 'scatter3d',
            marker: {{ size: 10, color: '#ff6b6b', symbol: 'diamond' }},
            name: `Scan ${{scanIdx}}`
        }});

        // Add direction glyphs for top-6 bins by kappa
        const kappaArr = kappa_bins[scanIdx];
        const indices = [...Array(48).keys()].sort((a, b) => kappaArr[b] - kappaArr[a]).slice(0, 6);
        const S_scan = S_bins[scanIdx];
        const R_scan = R_WL[scanIdx];
        const p_scan = [trajectoryData.x[scanIdx], trajectoryData.y[scanIdx], trajectoryData.z[scanIdx]];

        for (const b of indices) {{
            const S_b = S_scan[b];
            const norm = Math.sqrt(S_b[0]*S_b[0] + S_b[1]*S_b[1] + S_b[2]*S_b[2]);
            if (norm < 1e-6) continue;

            const d_body = [S_b[0]/norm, S_b[1]/norm, S_b[2]/norm];
            // Rotate to world frame: d_world = R_scan @ d_body
            const d_world = [
                R_scan[0][0]*d_body[0] + R_scan[0][1]*d_body[1] + R_scan[0][2]*d_body[2],
                R_scan[1][0]*d_body[0] + R_scan[1][1]*d_body[1] + R_scan[1][2]*d_body[2],
                R_scan[2][0]*d_body[0] + R_scan[2][1]*d_body[1] + R_scan[2][2]*d_body[2]
            ];

            const length = 0.3 * Math.log(1 + kappaArr[b]);
            const endX = p_scan[0] + length * d_world[0];
            const endY = p_scan[1] + length * d_world[1];
            const endZ = p_scan[2] + length * d_world[2];

            traces.push({{
                x: [p_scan[0], endX],
                y: [p_scan[1], endY],
                z: [p_scan[2], endZ],
                mode: 'lines',
                type: 'scatter3d',
                line: {{ color: '#f7b731', width: 4 }},
                showlegend: false,
                hoverinfo: 'skip'
            }});
        }}

        const layout = {{
            ...darkLayout,
            height: 450,
            title: {{ text: `3D Trajectory (Scan ${{scanIdx}} selected)`, font: {{color: '#00d4ff'}} }},
            scene: {{
                xaxis: {{ title: 'X (m)', gridcolor: '#1a1a2e', backgroundcolor: '#0f3460' }},
                yaxis: {{ title: 'Y (m)', gridcolor: '#1a1a2e', backgroundcolor: '#0f3460' }},
                zaxis: {{ title: 'Z (m)', gridcolor: '#1a1a2e', backgroundcolor: '#0f3460' }},
                bgcolor: '#0f3460',
                aspectmode: 'data'
            }},
            margin: {{ t: 50, b: 30, l: 30, r: 30 }},
            legend: {{ x: 0, y: 1, bgcolor: 'rgba(22, 33, 62, 0.8)' }}
        }};

        Plotly.react('trajectory', traces, layout, {{responsive: true}});
    }}

	    // =====================================================================
	    // Panel D: Factor influence (stacked area) + top-K bins bar chart
	    // =====================================================================
	    const TOP_K_BINS = 12;
		    function createFactorStacked() {{
		        const factors = [
		            {{ id: 'lidar', label: 'LiDAR', color: '#00d4ff' }},
		            {{ id: 'odom', label: 'Odom', color: '#4ecdc4' }},
		            {{ id: 'imu', label: 'Accel(vMF)', color: '#f7b731' }},
		            {{ id: 'gyro', label: 'Gyro', color: '#ff6b6b' }},
		            {{ id: 'preint', label: 'Preint', color: '#a55eea' }},
		        ];

		        const traces = [];
		        const hasAny = (arr) => arr && arr.some(v => v !== 0);

		        // Row 1: Rotation info (stacked)
		        let anyRot = false;
		        factors.forEach((f, idx) => {{
		            const y = timelineData['I_rot_' + f.id];
		            if (hasAny(y)) {{
		                anyRot = true;
		                traces.push({{
		                    x: timelineData.scan_idx, y: y, name: f.label + ' (rot)',
		                    stackgroup: 'rot', fill: (idx === 0 ? 'tozeroy' : 'tonexty'),
		                    line: {{color: f.color, width: 1}},
		                    mode: 'lines', xaxis: 'x', yaxis: 'y'
		                }});
		            }}
		        }});

		        // Row 2: XY translation info (stacked)
		        let anyXy = false;
		        factors.forEach((f, idx) => {{
		            const y = timelineData['I_xy_' + f.id];
		            if (hasAny(y)) {{
		                anyXy = true;
		                traces.push({{
		                    x: timelineData.scan_idx, y: y, name: f.label + ' (xy)',
		                    stackgroup: 'xy', fill: (idx === 0 ? 'tozeroy' : 'tonexty'),
		                    line: {{color: f.color, width: 1}},
		                    mode: 'lines', xaxis: 'x2', yaxis: 'y2'
		                }});
		            }}
		        }});

		        // Row 3: Z info leak (lines; should be ~0 for LiDAR on planar robots)
		        const zTotal = timelineData.scan_idx.map((_, i) =>
		            (timelineData.I_z_lidar[i] || 0) +
		            (timelineData.I_z_odom[i] || 0) +
		            (timelineData.I_z_imu[i] || 0) +
		            (timelineData.I_z_gyro[i] || 0) +
		            (timelineData.I_z_preint[i] || 0)
		        );
		        traces.push({{
		            x: timelineData.scan_idx, y: zTotal, name: 'Z total (α·Lzz sum)',
		            mode: 'lines', line: {{color: '#ffffff', width: 2}},
		            xaxis: 'x3', yaxis: 'y3'
		        }});
		        factors.forEach((f) => {{
		            const y = timelineData['I_z_' + f.id];
		            if (hasAny(y)) {{
		                traces.push({{
		                    x: timelineData.scan_idx, y: y, name: f.label + ' (z)',
		                    mode: 'lines', line: {{color: f.color, width: 1, dash: 'dot'}},
		                    xaxis: 'x3', yaxis: 'y3'
		                }});
		            }}
		        }});

		        if (!anyRot && !anyXy) {{
		            traces.push({{ x: timelineData.scan_idx, y: timelineData.scan_idx.map(() => 0), name: 'No factor data', line: {{color: '#888'}}, xaxis: 'x', yaxis: 'y' }});
		        }}

		        const layout = {{
		            ...darkLayout,
		            height: 420,
		            showlegend: true,
		            legend: {{ orientation: 'h', y: 1.06, x: 0.5, xanchor: 'center' }},
		            grid: {{ rows: 3, columns: 1, pattern: 'independent', roworder: 'top to bottom' }},
		            xaxis: {{ ...darkLayout.xaxis, anchor: 'y', domain: [0, 1], showticklabels: false }},
		            xaxis2: {{ ...darkLayout.xaxis, anchor: 'y2', domain: [0, 1], showticklabels: false }},
		            xaxis3: {{ ...darkLayout.xaxis, anchor: 'y3', domain: [0, 1], title: 'Scan Index' }},
		            yaxis: {{ ...darkLayout.yaxis, title: 'ΔI_rot (α·tr L_rot)' }},
		            yaxis2: {{ ...darkLayout.yaxis, title: 'ΔI_xy (α·tr L_xy)' }},
		            yaxis3: {{ ...darkLayout.yaxis, title: 'ΔI_z (α·Lzz)' }},
		            margin: {{ t: 50, b: 40, l: 70, r: 40 }},
		            shapes: createVerticalLines(currentScan, 3),
		        }};
		        Plotly.react('factor-stacked', traces, layout, {{responsive: true}});
		    }}

		    function createTopKBins(scanIdx) {{
		        const N = N_bins[scanIdx] || Array(48).fill(0);
		        const kS = kappa_bins[scanIdx] || Array(48).fill(0);
		        const kM = (kappa_map_bins && kappa_map_bins[scanIdx]) ? kappa_map_bins[scanIdx] : Array(48).fill(0);
		        const planM = (map_planarity && map_planarity[scanIdx]) ? map_planarity[scanIdx] : Array(48).fill(0);
		        const anisoM = (map_aniso && map_aniso[scanIdx]) ? map_aniso[scanIdx] : Array(48).fill(0);

		        const w_b = N.map((n, i) => n * kS[i]);
		        const indices = [...Array(48).keys()].sort((a, b) => w_b[b] - w_b[a]).slice(0, TOP_K_BINS);
		        const binLabels = indices.map(i => 'b' + i);

		        // Left axis: mass-like quantities
		        const traceW = {{
		            x: binLabels, y: indices.map(i => w_b[i]),
		            name: 'w_b = N×κ_scan', type: 'bar',
		            marker: {{ color: '#00d4ff' }},
		            yaxis: 'y'
		        }};
		        const traceN = {{
		            x: binLabels, y: indices.map(i => N[i]),
		            name: 'N', type: 'bar',
		            marker: {{ color: '#4ecdc4' }},
		            yaxis: 'y'
		        }};

		        // Right axis: normalized 0..1 sentinels so κ/aniso/planarity are visible
		        const kMax = Math.max(...indices.map(i => kS[i])) + 1e-12;
		        const kMaxM = Math.max(...indices.map(i => kM[i])) + 1e-12;
		        const kS_norm = indices.map(i => kS[i] / kMax);
		        const kM_norm = indices.map(i => kM[i] / kMaxM);

		        const traceKs = {{
		            x: binLabels, y: kS_norm,
		            name: 'κ_scan (norm)', type: 'scatter', mode: 'lines+markers',
		            marker: {{ color: '#f7b731' }},
		            line: {{ color: '#f7b731' }},
		            yaxis: 'y2',
		            hovertemplate: 'κ_scan=%{{customdata:.3f}}<extra></extra>',
		            customdata: indices.map(i => kS[i]),
		        }};
		        const traceKm = {{
		            x: binLabels, y: kM_norm,
		            name: 'κ_map (norm)', type: 'scatter', mode: 'lines+markers',
		            marker: {{ color: '#a55eea' }},
		            line: {{ color: '#a55eea', dash: 'dot' }},
		            yaxis: 'y2',
		            hovertemplate: 'κ_map=%{{customdata:.3f}}<extra></extra>',
		            customdata: indices.map(i => kM[i]),
		        }};
		        const tracePlan = {{
		            x: binLabels, y: indices.map(i => planM[i]),
		            name: 'planarity(map)', type: 'scatter', mode: 'lines+markers',
		            marker: {{ color: '#26de81' }},
		            line: {{ color: '#26de81', dash: 'dot' }},
		            yaxis: 'y2'
		        }};
		        const traceAniso = {{
		            x: binLabels, y: indices.map(i => anisoM[i]),
		            name: 'aniso(map)', type: 'scatter', mode: 'lines+markers',
		            marker: {{ color: '#ff6b6b' }},
		            line: {{ color: '#ff6b6b', dash: 'dot' }},
		            yaxis: 'y2'
		        }};

		        const layout = {{
		            ...darkLayout,
		            height: 320,
		            title: {{ text: 'Top-' + TOP_K_BINS + ' bins (scan ' + scanIdx + ') — mass vs (κ/aniso/planarity)', font: {{color: '#00d4ff'}} }},
		            xaxis: {{ ...darkLayout.xaxis, title: 'Bin' }},
		            yaxis: {{ ...darkLayout.yaxis, title: 'w_b / N' }},
		            yaxis2: {{ ...darkLayout.yaxis, title: 'normalized / 0..1', overlaying: 'y', side: 'right', range: [0, 1.05] }},
		            barmode: 'group',
		            showlegend: true,
		            legend: {{ orientation: 'h', y: 1.05 }},
		            margin: {{ t: 50, b: 40, l: 60, r: 60 }},
		        }};
		        Plotly.react('topk-bins', [traceW, traceN, traceKs, traceKm, tracePlan, traceAniso], layout, {{responsive: true}});
		    }}

    // =====================================================================
    // Update all panels when slider changes
    // =====================================================================
	    function updateAllPanels(scanIdx) {{
        currentScan = scanIdx;

        // Update info display
        document.getElementById('scan-display').textContent = scanIdx;
	        const logdetVal = trajectoryData.is_minimal_tape ? timelineData.logdet_L_pose6[scanIdx] : timelineData.logdet_L_total[scanIdx];
	        document.getElementById('info-dt').textContent = timelineData.dt_secs[scanIdx].toFixed(3);
	        document.getElementById('info-alpha').textContent = timelineData.fusion_alpha[scanIdx].toFixed(3);
	        document.getElementById('info-logdet').textContent = (typeof logdetVal === 'number' ? logdetVal : 0).toFixed(1);
	        document.getElementById('info-cond').textContent = timelineData.log10_cond_pose6[scanIdx].toFixed(2);
	        document.getElementById('info-rot-lidar').textContent = (timelineData.rot_err_lidar_deg_post && timelineData.rot_err_lidar_deg_post[scanIdx] !== undefined) ? timelineData.rot_err_lidar_deg_post[scanIdx].toFixed(2) : '--';

        // Update timeline vertical lines
        Plotly.relayout('timeline', {{ shapes: createVerticalLines(scanIdx, TIMELINE_ROWS) }});

        // Update heatmap and Z leak
        createHeatmap(scanIdx);

        // Update trajectory
        createTrajectory(scanIdx);

        // Update factor ledger vertical lines
        Plotly.relayout('factor-stacked', {{ shapes: createVerticalLines(scanIdx, 3) }});

        // Update top-K bins bar chart
        createTopKBins(scanIdx);
    }}

    // =====================================================================
    // Initialize
    // =====================================================================
	    document.addEventListener('DOMContentLoaded', function() {{
        if (trajectoryData.is_minimal_tape) {{
            const el = document.getElementById('minimal-tape-notice');
            if (el) el.style.display = 'block';
        }}
        createTimeline();
        setupHeatmapToggle();
        createHeatmap(currentScan);
        createTrajectory(currentScan);
        createFactorStacked();
        createTopKBins(currentScan);
        if (trajectoryData.has_ground_truth && trajectoryData.res_x && trajectoryData.res_x.length > 0) {{
            document.getElementById('offset-table-panel').style.display = 'block';
            const thead = document.getElementById('offset-table-head');
            thead.innerHTML = '<th>Scan</th><th>t (s)</th><th>est_x</th><th>est_y</th><th>est_z</th><th>gt_x</th><th>gt_y</th><th>gt_z</th><th>res_x</th><th>res_y</th><th>res_z</th>';
            const tbody = document.getElementById('offset-table-body');
            const ts = trajectoryData.scan_timestamps || [];
            const fmt = (v) => (typeof v === 'number' && !isNaN(v) ? v.toFixed(4) : '—');
            for (let i = 0; i < trajectoryData.res_x.length; i++) {{
                const row = document.createElement('tr');
                row.innerHTML = `<td>${{i}}</td><td>${{fmt(ts[i])}}</td><td>${{fmt(trajectoryData.x[i])}}</td><td>${{fmt(trajectoryData.y[i])}}</td><td>${{fmt(trajectoryData.z[i])}}</td><td>${{fmt(trajectoryData.gt_at_scan_x[i])}}</td><td>${{fmt(trajectoryData.gt_at_scan_y[i])}}</td><td>${{fmt(trajectoryData.gt_at_scan_z[i])}}</td><td>${{fmt(trajectoryData.res_x[i])}}</td><td>${{fmt(trajectoryData.res_y[i])}}</td><td>${{fmt(trajectoryData.res_z[i])}}</td>`;
                tbody.appendChild(row);
            }}
        }}

        // Update info display (use logdet_L_pose6 when minimal tape)
        const logdetVal = trajectoryData.is_minimal_tape ? timelineData.logdet_L_pose6[currentScan] : timelineData.logdet_L_total[currentScan];
        document.getElementById('info-dt').textContent = timelineData.dt_secs[currentScan].toFixed(3);
        document.getElementById('info-alpha').textContent = timelineData.fusion_alpha[currentScan].toFixed(3);
        document.getElementById('info-logdet').textContent = (typeof logdetVal === 'number' ? logdetVal : 0).toFixed(1);
        document.getElementById('info-cond').textContent = timelineData.log10_cond_pose6[currentScan].toFixed(2);
        document.getElementById('info-rot-lidar').textContent = (timelineData.rot_err_lidar_deg_post && timelineData.rot_err_lidar_deg_post[currentScan] !== undefined) ? timelineData.rot_err_lidar_deg_post[currentScan].toFixed(2) : '--';

        // Slider event
        const slider = document.getElementById('scan-slider');
        slider.addEventListener('input', function(e) {{
            updateAllPanels(parseInt(e.target.value));
        }});

        // Click on timeline to select scan
        document.getElementById('timeline').on('plotly_click', function(data) {{
            if (data.points && data.points.length > 0) {{
                const scanIdx = data.points[0].x;
                slider.value = scanIdx;
                updateAllPanels(scanIdx);
            }}
        }});

        // Click on timeline or factor-stacked to select scan
        document.getElementById('factor-stacked').on('plotly_click', function(data) {{
            if (data.points && data.points.length > 0) {{
                const scanIdx = data.points[0].x;
                slider.value = scanIdx;
                updateAllPanels(scanIdx);
            }}
        }});
    }});
    </script>
</body>
</html>
"""

    # Write HTML file
    if output_path:
        html_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(html_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # Use temp file
        temp_fd, html_path = tempfile.mkstemp(suffix=".html", text=True)
        os.close(temp_fd)  # Close the file descriptor, we'll open it for writing below
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Dashboard saved to: {html_path}")
    
    # Try to open in browser (only if not explicitly saving to file)
    if not output_path:
        if open_browser_wayland_compatible(html_path):
            print("Dashboard opened in browser")
        else:
            print("")
            print("Could not automatically open browser.")
            print(f"Please manually open: {html_path}")
            print("")
            print("Or use --output to save to a specific location:")
            print(f"  .venv/bin/python tools/slam_dashboard.py <diagnostics.npz> --output dashboard.html")
    
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description="Golden Child SLAM v2 Debugging Dashboard"
    )
    parser.add_argument(
        "diagnostics_file",
        type=str,
        help="Path to diagnostics NPZ file (e.g., /tmp/gc_slam_diagnostics.npz)",
    )
    parser.add_argument(
        "--scan",
        type=int,
        default=0,
        help="Initial selected scan index (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save HTML file to specified path (does not auto-open browser)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth TUM file. If omitted and diagnostics are in a results dir, uses ground_truth_aligned.tum in the same directory.",
    )
    parser.add_argument(
        "--gt-y-flip",
        action="store_true",
        help="Negate GT Y when loading (if per-scan residual Y is all one sign, may be frame convention; try this to test).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.diagnostics_file):
        print(f"Error: File not found: {args.diagnostics_file}")
        print("")
        print("Expected file: diagnostics NPZ file from a SLAM run")
        print("This file is created by the backend node during operation.")
        print("")
        print("To generate this file, run the SLAM pipeline:")
        print("  ./tools/run_and_evaluate_gc.sh")
        print("")
        print("Or check if a previous run created it at:")
        print("  /tmp/gc_slam_diagnostics.npz")
        print("  results/gc_*/diagnostics.npz")
        sys.exit(1)

    print(f"Loading diagnostics from: {args.diagnostics_file}")
    data = load_diagnostics_npz(args.diagnostics_file)

    n_scans = int(data.get("n_scans", 0))
    print(f"Loaded {n_scans} scans")

    # Minimal tape: NPZ does not store p_W (pipeline still writes trajectory to TUM file).
    # Load estimated_trajectory.tum from same dir when present so Panel C can show the trajectory.
    is_minimal = data.get("format") == "minimal_tape" or ("L_pose6" in data and "L_total" not in data)
    if is_minimal and n_scans > 0:
        results_dir = Path(args.diagnostics_file).resolve().parent
        est_tum = results_dir / "estimated_trajectory.tum"
        if est_tum.exists():
            xyz_ts = load_tum_positions(str(est_tum))
            if xyz_ts is not None:
                x_est, y_est, z_est, ts_est = xyz_ts
                ts_est = np.asarray(ts_est, dtype=np.float64)
                scan_ts = np.asarray(data.get("timestamps", np.zeros(n_scans)), dtype=np.float64)
                if scan_ts.size == n_scans and ts_est.size >= 2:
                    x_at_scan = np.interp(scan_ts, ts_est, np.asarray(x_est, dtype=np.float64))
                    y_at_scan = np.interp(scan_ts, ts_est, np.asarray(y_est, dtype=np.float64))
                    z_at_scan = np.interp(scan_ts, ts_est, np.asarray(z_est, dtype=np.float64))
                    data = dict(data)
                    data["p_W"] = np.stack([x_at_scan, y_at_scan, z_at_scan], axis=1)
                    print(f"Trajectory: loaded {est_tum.name} (interpolated at {n_scans} scan timestamps)")

    # Print available keys for debugging
    print(f"Available data keys: {sorted(data.keys())}")

    if n_scans == 0:
        print("")
        print("ERROR: No scan data found in file!")
        print("The diagnostics file exists but contains 0 scans.")
        print("")
        print("This could mean:")
        print("  1. The SLAM run didn't complete successfully")
        print("  2. No LiDAR scans were processed")
        print("  3. The diagnostics weren't saved properly")
        print("")
        print("Check the SLAM log for errors.")
        sys.exit(1)

    # Validate selected scan
    if args.scan >= n_scans:
        print(f"Warning: Requested scan {args.scan} >= n_scans {n_scans}, using scan 0")
        args.scan = 0

    # Resolve ground truth path (explicit, or auto from same dir as diagnostics)
    ground_truth_path = args.ground_truth
    if not ground_truth_path:
        results_dir = Path(args.diagnostics_file).resolve().parent
        auto_gt = results_dir / "ground_truth_aligned.tum"
        if auto_gt.exists():
            ground_truth_path = str(auto_gt)
            print(f"Using ground truth: {ground_truth_path}")
    elif not os.path.exists(ground_truth_path):
        print(f"Warning: Ground truth file not found: {ground_truth_path}")
        ground_truth_path = None

    # Create dashboard
    html_path = create_full_dashboard(
        data, args.scan, output_path=args.output, ground_truth_path=ground_truth_path, gt_y_flip=args.gt_y_flip
    )
    
    if html_path:
        print(f"\n✓ Dashboard ready at: {html_path}")
        if args.output:
            print(f"  Saved to: {os.path.abspath(args.output)}")
            print(f"  Open manually in your browser or use: xdg-open {html_path}")


if __name__ == "__main__":
    main()
