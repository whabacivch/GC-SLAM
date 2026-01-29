#!/usr/bin/env python3
"""
FL-SLAM Evaluation Script - Enhanced Publication Quality

Compares estimated trajectory against ground truth using:
1. ATE (Absolute Trajectory Error) - Global consistency (translation + rotation)
   - Full trajectory metrics with percentiles (P75, P90, P95, P99)
   - Per-axis breakdown (X/Y/Z translation, roll/pitch/yaw rotation)
   - Distribution statistics (skewness, kurtosis, IQR, normality tests)
2. RPE (Relative Pose Error) - Local drift at multiple scales (1m, 5m, 10m)
   - Enhanced with percentiles and distribution statistics
3. Generates publication-quality plots:
   - Trajectory comparison (4-view)
   - Error heatmap
   - Error over time + histogram
   - Per-axis error breakdown (6 subplots)
   - Cumulative error analysis
   - Pose graph visualization
4. Exports metrics in multiple formats:
   - metrics.txt: Human-readable text format
   - metrics.csv: Spreadsheet-ready with all statistics
   - metrics.json: Structured JSON for automation/CI integration

Uses evo library for standard SLAM metrics, enhanced with comprehensive statistics.

Alignment: SLAM is relative — the robot arbitrarily says "start here (0,0,0)" and estimates from there.
To evaluate meaningfully, ground truth must be in the same coordinate system as the estimate.
Default is initial-pose alignment: GT is transformed into the estimate frame (at first pose, GT = estimate).
Then ATE/RPE measure how the estimate diverges from reality. Use --align umeyama for full
SE(3) best-fit (hides initial error; not valid for relative SLAM).
"""
import argparse
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
# Import Axes3D only when needed (3D plots)
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    # Fallback: Axes3D may not be needed if matplotlib version is incompatible
    Axes3D = None
from evo.core import sync
from evo.tools import file_interface
from evo.core import metrics
from evo.core import trajectory
import numpy as np
import copy
import json
from scipy import stats
from scipy.spatial.transform import Rotation


def load_trajectory(file_path):
    """Load trajectory from TUM format."""
    return file_interface.read_tum_trajectory_file(file_path)


# Signed permutation from Umeyama (est → GT): GT x ≈ -EST z, GT y ≈ EST x, GT z ≈ -EST y.
# Apply to EST positions/orientations to express EST in GT axis convention for same-plane comparison.
R_EST_TO_GT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
], dtype=np.float64)


def apply_est_to_gt_convention(traj: trajectory.PoseTrajectory3D) -> trajectory.PoseTrajectory3D:
    """
    Return a new trajectory with each pose transformed by R_EST_TO_GT (position and rotation).
    Expresses estimate in GT axis convention so 2D projections are same-plane.
    """
    new_poses = []
    for pose in traj.poses_se3:
        R = np.array(pose[:3, :3], dtype=np.float64)
        t = np.array(pose[:3, 3], dtype=np.float64)
        R_new = R_EST_TO_GT @ R
        t_new = R_EST_TO_GT @ t
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_new
        T[:3, 3] = t_new
        new_poses.append(T)
    return trajectory.PoseTrajectory3D(
        poses_se3=new_poses,
        timestamps=np.array(traj.timestamps, dtype=np.float64),
        meta=dict(traj.meta) if traj.meta else None,
    )


def swap_gt_yz(traj: trajectory.PoseTrajectory3D) -> trajectory.PoseTrajectory3D:
    """
    Return a new trajectory with Y and Z axes swapped (position and rotation).
    Diagnostic for axis-convention mismatch when "our XZ looks like GT XY".
    """
    new_poses = []
    for pose in traj.poses_se3:
        R = np.array(pose[:3, :3], dtype=np.float64)
        t = np.array(pose[:3, 3], dtype=np.float64)
        R_new = R[:, [0, 2, 1]]
        t_new = np.array([t[0], t[2], t[1]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_new
        T[:3, 3] = t_new
        new_poses.append(T)
    return trajectory.PoseTrajectory3D(
        poses_se3=new_poses,
        timestamps=np.array(traj.timestamps, dtype=np.float64),
        meta=dict(traj.meta) if traj.meta else None,
    )


def load_op_reports(file_path):
    """Load OpReport JSON lines (strict)."""
    reports = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                raise ValueError(f"Empty OpReport line at {idx} (expected JSON object).")
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid OpReport JSON at line {idx}: {exc}") from exc
    if not reports:
        raise ValueError(f"No OpReports found in {file_path}.")
    return reports


def validate_op_reports(op_report_path, require_imu=True):
    """Validate OpReports for runtime health and audit compliance."""
    if not op_report_path or not os.path.exists(op_report_path):
        raise FileNotFoundError(f"OpReport file not found: {op_report_path}")

    reports = load_op_reports(op_report_path)

    required_reports = [
        "GaussianPredictSE3",
        "AnchorCreate",
        "LoopFactorPublished",
        "LoopFactorRecomposition",
    ]
    if require_imu:
        required_reports.append("IMUFactorUpdate")

    forbidden_reports = [
        "PointCloudTransformMissing",
        "PointCloudExtrinsicInvalid",
        "PointCloudConversionFailed",
        "ScanPointsMissing",
        "LoopFactorUnavailable",
        "DenseModuleKeepFractionProjection",
    ]

    warning_reports = [
        "ICPEvidenceUnavailable",
        "LoopResponsibilityDomainProjection",
        "DenseAssociationDomainProjection",
        "LoopBudgetProjection",
        "AnchorBudgetProjection",
    ]

    required_metrics = {
        "GaussianPredictSE3": ["state_dim", "linearization_point", "process_noise_trace"],
        "AnchorCreate": ["anchor_id", "dt_sec", "timestamp_weight"],
        "LoopFactorPublished": ["anchor_id", "weight", "mse", "iterations", "converged", "point_source"],
        "LoopFactorRecomposition": ["anchor_id", "weight", "innovation_norm"],
        "IMUFactorUpdate": [
            "dt_header",
            "dt_stamps",
            "dt_gap_start",
            "dt_gap_end",
            "bias_rw_cov_adaptive",
            "bias_rw_cov_trace_gyro",
            "bias_rw_cov_trace_accel",
        ],
    }

    def _require_keys(obj, keys, label):
        missing = [k for k in keys if k not in obj or obj[k] is None]
        if missing:
            raise ValueError(f"Missing {label} keys: {missing}")

    # Base schema validation
    for report in reports:
        _require_keys(
            report,
            ["name", "exact", "approximation_triggers", "family_in", "family_out",
             "closed_form", "domain_projection", "metrics", "timestamp"],
            "op_report",
        )

    # Required report presence
    for name in required_reports:
        if not any(r.get("name") == name for r in reports):
            raise ValueError(f"Missing required OpReport: {name}")

    # Required metric keys
    for report in reports:
        name = report.get("name")
        if name in required_metrics:
            metrics = report.get("metrics", {})
            _require_keys(metrics, required_metrics[name], f"{name}.metrics")

    # Forbidden report detection
    forbidden_hits = [r for r in reports if r.get("name") in forbidden_reports]
    if forbidden_hits:
        names = sorted({r.get("name") for r in forbidden_hits})
        raise ValueError(f"Forbidden OpReports present: {names}")

    # Warning summary
    warning_hits = [r for r in reports if r.get("name") in warning_reports]
    if warning_hits:
        counts = {}
        for r in warning_hits:
            counts[r.get("name")] = counts.get(r.get("name"), 0) + 1
        print(f"  WARNING: Degraded OpReports detected: {counts}")

    print(f"  OpReport checks passed: {len(reports)} total reports.")


def validate_trajectory(traj: trajectory.PoseTrajectory3D, name: str, strict: bool = True):
    """
    Validate trajectory has proper timestamps and coordinates.
    
    Checks:
    - Monotonic timestamps (no duplicates, increasing)
    - Reasonable coordinate ranges
    - Sufficient poses
    
    Args:
        traj: Trajectory to validate
        name: Name for logging
        strict: If True, raise ValueError on failure. If False, only warn.
    
    Returns True if valid, prints warnings for issues.
    """
    timestamps = np.array(traj.timestamps)
    xyz = traj.positions_xyz
    
    print(f"\n  Validating {name}:")
    print(f"    Poses: {len(timestamps)}")
    
    valid = True
    critical_errors = []
    
    # Check monotonic timestamps
    diffs = np.diff(timestamps)
    if not np.all(diffs >= 0):
        non_mono = np.sum(diffs < 0)
        print(f"    WARNING: {non_mono} non-monotonic timestamp gaps!")
        valid = False
        critical_errors.append("non-monotonic timestamps")
    
    # Check for duplicate timestamps (strict: must be > 0, not >= 0)
    dups = np.sum(diffs == 0)
    if dups > 0:
        print(f"    WARNING: {dups} duplicate timestamps ({100*dups/len(timestamps):.1f}%)")
        print("    FAILED: Timestamps must be strictly monotonic (no duplicates)")
        valid = False
        critical_errors.append("duplicate timestamps")
    
    # Check coordinate ranges
    print(f"    Coordinate ranges:")
    print(f"      X: [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}] m")
    print(f"      Y: [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}] m")
    print(f"      Z: [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}] m")
    
    # Check for unreasonable values (warn but don't fail for estimated trajectory)
    max_coord = 1000.0  # 1km max reasonable range
    if np.any(np.abs(xyz) > max_coord):
        print(f"    WARNING: Coordinates exceed {max_coord}m - possible data corruption or coordinate frame mismatch!")
        if strict:
            valid = False
            critical_errors.append("coordinates exceed reasonable range")
        else:
            print(f"    (Continuing with evaluation despite large coordinate values)")
    
    # Check timestamp range
    duration = timestamps[-1] - timestamps[0]
    print(f"    Duration: {duration:.2f}s")
    print(f"    Avg rate: {len(timestamps)/max(duration,0.001):.1f} Hz")
    
    if valid:
        print("    Status: VALID")
        return
    
    # Only raise for critical errors (timestamp issues) if strict mode
    if critical_errors:
        error_msg = f"{name} trajectory validation failed: {', '.join(critical_errors)}"
        if strict:
            raise ValueError(error_msg)
        else:
            print(f"    WARNING: {error_msg} (continuing anyway)")
    else:
        print(f"    Status: WARNINGS (continuing with evaluation)")


def align_trajectories_initial(gt_sync: trajectory.PoseTrajectory3D, est_sync: trajectory.PoseTrajectory3D) -> None:
    """
    Put GT in the same coordinate system as the estimate (estimate frame).
    Transforms GT so that at the first associated pose, GT = estimate (robot's arbitrary start).
    Then we compare apples to apples: how good are our internal estimates vs external truth.
    Modifies gt_sync in place.
    """
    if len(gt_sync.poses_se3) == 0 or len(est_sync.poses_se3) == 0:
        return
    T_est0 = np.array(est_sync.poses_se3[0], dtype=np.float64)
    T_gt0_inv = np.linalg.inv(gt_sync.poses_se3[0])
    T_gt_to_est = T_est0 @ T_gt0_inv
    gt_sync.transform(T_gt_to_est)


def compute_ate_full(
    gt_traj: trajectory.PoseTrajectory3D,
    est_traj: trajectory.PoseTrajectory3D,
    *,
    align_mode: str = "initial",
):
    """
    Compute ATE for both translation and rotation.

    align_mode: "initial" = align first pose only (GT → estimate frame; valid for relative SLAM).
                "umeyama" = full SE(3) Umeyama over whole trajectory (best-fit; hides initial error).

    Returns: (ate_trans, ate_rot, gt_aligned, est_aligned)
    """
    # Deep copy to avoid modifying originals
    gt_copy = copy.deepcopy(gt_traj)
    est_copy = copy.deepcopy(est_traj)

    # Associate by timestamp
    gt_sync, est_sync = sync.associate_trajectories(gt_copy, est_copy)

    # Put both in same frame: initial pose (estimate frame) or full Umeyama
    if align_mode == "initial":
        align_trajectories_initial(gt_sync, est_sync)
    else:
        est_sync.align(gt_sync, correct_scale=False)

    # Translation ATE
    ate_trans = metrics.APE(metrics.PoseRelation.translation_part)
    ate_trans.process_data((gt_sync, est_sync))
    
    # Rotation ATE (angle in degrees)
    ate_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ate_rot.process_data((gt_sync, est_sync))
    
    return ate_trans, ate_rot, gt_sync, est_sync


def diagnose_constant_rotation_offset(
    gt_aligned: trajectory.PoseTrajectory3D,
    est_aligned: trajectory.PoseTrajectory3D,
    *,
    mean_deg_threshold: float = 150.0,
    std_deg_threshold: float = 10.0,
) -> dict | None:
    """
    Detect the common failure mode where translation aligns well but rotation ATE is ~180deg
    with low variance, indicating a near-constant frame offset (e.g., exporting LiDAR/IMU
    orientation while GT is base frame, or axis convention mismatch).
    """
    if len(gt_aligned.poses_se3) == 0 or len(est_aligned.poses_se3) == 0:
        return None

    angles = []
    rel_rots = []
    for i in range(min(len(gt_aligned.poses_se3), len(est_aligned.poses_se3))):
        Rg = gt_aligned.poses_se3[i][:3, :3]
        Re = est_aligned.poses_se3[i][:3, :3]
        R_rel = Re.T @ Rg
        rel_rots.append(R_rel)
        angles.append(Rotation.from_matrix(R_rel).magnitude() * 180.0 / np.pi)

    angles = np.asarray(angles, dtype=float)
    mean_deg = float(np.mean(angles))
    std_deg = float(np.std(angles))
    if not (mean_deg >= mean_deg_threshold and std_deg <= std_deg_threshold):
        return None

    # Markley quaternion average for a stable "best" constant rotation estimate.
    rots = Rotation.from_matrix(np.stack(rel_rots, axis=0))
    quats = rots.as_quat()  # x,y,z,w
    A = np.zeros((4, 4), dtype=float)
    for q in quats:
        q = q / (np.linalg.norm(q) + 1e-12)
        A += np.outer(q, q)
    A /= max(len(quats), 1)
    vals, vecs = np.linalg.eigh(A)
    q_avg = vecs[:, int(np.argmax(vals))]
    rot_avg = Rotation.from_quat(q_avg)
    angle_avg_deg = float(rot_avg.magnitude() * 180.0 / np.pi)
    rotvec = rot_avg.as_rotvec().astype(float)
    axis = (rotvec / (np.linalg.norm(rotvec) + 1e-12)).tolist()

    return {
        "kind": "constant_rotation_offset_suspected",
        "mean_rel_angle_deg": mean_deg,
        "std_rel_angle_deg": std_deg,
        "avg_rel_angle_deg": angle_avg_deg,
        "avg_rel_rotvec": rotvec.tolist(),
        "avg_rel_axis": axis,
        "note": (
            "Likely frame mismatch (e.g., base vs IMU/LiDAR) or axis convention flip. "
            "This is not necessarily a SLAM divergence; fix/export the intended frame."
        ),
    }


def compute_rpe_multi_scale(
    gt_traj: trajectory.PoseTrajectory3D,
    est_traj: trajectory.PoseTrajectory3D,
    *,
    align_mode: str = "initial",
):
    """
    Compute RPE at multiple scales: 1m, 5m, 10m.
    Uses same align_mode as ATE so trajectories are in the same frame.
    Returns dict: scale -> {'trans': metric, 'rot': metric}
    """
    scales = [1.0, 5.0, 10.0]
    results = {}

    for delta in scales:
        # Deep copy for each scale
        gt_copy = copy.deepcopy(gt_traj)
        est_copy = copy.deepcopy(est_traj)
        gt_sync, est_sync = sync.associate_trajectories(gt_copy, est_copy)
        if align_mode == "initial":
            align_trajectories_initial(gt_sync, est_sync)
        else:
            est_sync.align(gt_sync, correct_scale=False)

        try:
            # Translation RPE
            rpe_trans = metrics.RPE(
                metrics.PoseRelation.translation_part,
                delta=delta,
                delta_unit=metrics.Unit.meters,
                all_pairs=False
            )
            rpe_trans.process_data((gt_sync, est_sync))
            
            # Rotation RPE
            rpe_rot = metrics.RPE(
                metrics.PoseRelation.rotation_angle_deg,
                delta=delta,
                delta_unit=metrics.Unit.meters,
                all_pairs=False
            )
            rpe_rot.process_data((gt_sync, est_sync))
            
            results[f'{delta:.0f}m'] = {'trans': rpe_trans, 'rot': rpe_rot}
        except Exception as e:
            print(f"    WARNING: RPE at {delta}m failed: {e}")
    
    return results


def compute_per_axis_errors(
    gt_traj: trajectory.PoseTrajectory3D,
    est_traj: trajectory.PoseTrajectory3D,
    *,
    align_mode: str = "initial",
):
    """
    Compute per-axis translation and rotation errors after alignment.

    align_mode: "initial" or "umeyama" (same as ATE).

    Translation: per-axis absolute difference |gt - est| in meters (X, Y, Z).
    Rotation: at each pose, relative rotation est_inv * gt; Euler angles (XYZ order)
    in degrees; per-axis RMSE is over roll, pitch, yaw components (absolute value).

    Returns:
        dict with keys:
        - 'translation': {'x': errors (m), 'y': ..., 'z': ...}
        - 'rotation': {'roll': errors (deg), 'pitch': ..., 'yaw': ...}
        - 'timestamps': aligned timestamps
    """
    # Deep copy to avoid modifying originals
    gt_copy = copy.deepcopy(gt_traj)
    est_copy = copy.deepcopy(est_traj)

    # Associate and align (same mode as ATE)
    gt_sync, est_sync = sync.associate_trajectories(gt_copy, est_copy)
    if align_mode == "initial":
        align_trajectories_initial(gt_sync, est_sync)
    else:
        est_sync.align(gt_sync, correct_scale=False)
    
    # Translation errors per axis
    trans_diff = gt_sync.positions_xyz - est_sync.positions_xyz
    trans_errors = {
        'x': np.abs(trans_diff[:, 0]),
        'y': np.abs(trans_diff[:, 1]),
        'z': np.abs(trans_diff[:, 2])
    }
    
    # Rotation errors per axis (Euler angles)
    # Compute relative rotation at each pose
    rot_errors = {'roll': [], 'pitch': [], 'yaw': []}
    
    for i in range(len(gt_sync.poses_se3)):
        gt_rot = Rotation.from_matrix(gt_sync.poses_se3[i][:3, :3])
        est_rot = Rotation.from_matrix(est_sync.poses_se3[i][:3, :3])
        
        # Relative rotation error
        rel_rot = est_rot.inv() * gt_rot
        euler = rel_rot.as_euler('xyz', degrees=True)
        
        rot_errors['roll'].append(abs(euler[0]))
        rot_errors['pitch'].append(abs(euler[1]))
        rot_errors['yaw'].append(abs(euler[2]))
    
    rot_errors = {k: np.array(v) for k, v in rot_errors.items()}
    
    return {
        'translation': trans_errors,
        'rotation': rot_errors,
        'timestamps': np.array(est_sync.timestamps)
    }


def compute_percentiles(errors, percentiles=[50, 75, 90, 95, 99]):
    """Compute percentile statistics for error array."""
    errors = np.asarray(errors)
    results = {}
    for p in percentiles:
        results[f'p{p}'] = np.percentile(errors, p)
    return results


def compute_distribution_stats(errors):
    """Compute distribution statistics: skewness, kurtosis, IQR."""
    errors = np.asarray(errors)
    stats_dict = {
        'skewness': stats.skew(errors),
        'kurtosis': stats.kurtosis(errors),
        'iqr': np.percentile(errors, 75) - np.percentile(errors, 25)
    }
    
    # Normality test (Shapiro-Wilk for samples < 5000, otherwise use Anderson-Darling)
    if len(errors) < 5000:
        try:
            _, p_value = stats.shapiro(errors)
            stats_dict['normality_test'] = 'shapiro_wilk'
            stats_dict['normality_p_value'] = p_value
            stats_dict['is_normal'] = p_value > 0.05
        except:
            stats_dict['normality_test'] = 'failed'
            stats_dict['normality_p_value'] = None
            stats_dict['is_normal'] = None
    else:
        try:
            result = stats.anderson(errors, dist='norm')
            stats_dict['normality_test'] = 'anderson_darling'
            stats_dict['normality_statistic'] = result.statistic
            stats_dict['is_normal'] = result.statistic < result.critical_values[2]  # 5% level
        except:
            stats_dict['normality_test'] = 'failed'
            stats_dict['normality_statistic'] = None
            stats_dict['is_normal'] = None
    
    return stats_dict


def plot_trajectories(gt_traj: trajectory.PoseTrajectory3D, est_traj: trajectory.PoseTrajectory3D, output_path):
    """Plot ground truth vs estimated trajectory (4-view)."""
    fig = plt.figure(figsize=(14, 12))
    
    # Get positions
    gt_xyz = gt_traj.positions_xyz
    est_xyz = est_traj.positions_xyz

    # Sanity-print axis ranges (TUM order: col0=x, col1=y, col2=z). GT Z should be ~constant for planar robot.
    ptp_gt = np.ptp(gt_xyz, axis=0)
    ptp_est = np.ptp(est_xyz, axis=0)
    print(f"   [plot] GT  ptp(x,y,z) (m): {ptp_gt[0]:.4f}, {ptp_gt[1]:.4f}, {ptp_gt[2]:.4f}")
    print(f"   [plot] EST ptp(x,y,z) (m): {ptp_est[0]:.4f}, {ptp_est[1]:.4f}, {ptp_est[2]:.4f}")
    if ptp_gt[2] > 0.5:
        print("   [plot] WARNING: GT ptp(z) is large — expected ~0.04 m for planar robot; check axis convention or column order.")
    
    # XY view
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'b-', label='Ground Truth', linewidth=2)
    ax1.plot(est_xyz[:, 0], est_xyz[:, 1], 'r--', label='Estimated', linewidth=1.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectory (Top View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # XZ view
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'b-', label='Ground Truth', linewidth=2)
    ax2.plot(est_xyz[:, 0], est_xyz[:, 2], 'r--', label='Estimated', linewidth=1.5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('XZ Trajectory (Side View)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # YZ view
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(gt_xyz[:, 1], gt_xyz[:, 2], 'b-', label='Ground Truth', linewidth=2)
    ax3.plot(est_xyz[:, 1], est_xyz[:, 2], 'r--', label='Estimated', linewidth=1.5)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('YZ Trajectory (Front View)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3D trajectory
    try:
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    except Exception:
        # Fallback to 2D if 3D projection fails
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.text(0.5, 0.5, '3D plot unavailable\n(use trajectory_comparison XY/XZ/YZ views)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('3D Trajectory (unavailable)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Trajectory plot saved: {output_path} (3D view unavailable)")
        return
    ax4.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], 'b-', label='Ground Truth', linewidth=2)
    ax4.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], 'r--', label='Estimated', linewidth=1.5)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('3D Trajectory')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Trajectory plot saved: {output_path}")


def plot_trajectory_heatmap(gt_traj, est_traj, ate_metric, output_path):
    """Plot trajectory colored by error magnitude (heatmap)."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    gt_xyz = gt_traj.positions_xyz
    est_xyz = est_traj.positions_xyz
    errors = ate_metric.error
    
    # Ensure arrays match (evo may have synced them)
    n_points = min(len(est_xyz), len(errors))
    est_xyz = est_xyz[:n_points]
    errors = errors[:n_points]
    gt_xyz = gt_xyz[:n_points]
    
    # Ground truth as reference
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.7, zorder=1)
    
    # Estimated trajectory colored by error
    vmax = np.percentile(errors, 95)
    scatter = ax.scatter(est_xyz[:, 0], est_xyz[:, 1], 
                        c=errors, cmap='hot_r', s=30, 
                        vmin=0, vmax=vmax,
                        label='Estimated (colored by error)', zorder=2)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Translation Error (m)', rotation=270, labelpad=20)
    
    # Start/end markers
    ax.scatter([est_xyz[0, 0]], [est_xyz[0, 1]], c='green', s=200, marker='^', 
               label='Start', zorder=3, edgecolors='black')
    ax.scatter([est_xyz[-1, 0]], [est_xyz[-1, 1]], c='purple', s=200, marker='s', 
               label='End', zorder=3, edgecolors='black')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison with Error Heatmap')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Error heatmap saved: {output_path}")


def plot_error_over_time(ate_metric, timestamps, output_path):
    """Plot translation error over time + histogram."""
    errors = ate_metric.error
    timestamps = np.asarray(timestamps)
    if timestamps.shape[0] != errors.shape[0]:
        n = min(timestamps.shape[0], errors.shape[0])
        timestamps = timestamps[:n]
        errors = errors[:n]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Error over time
    axes[0].plot(timestamps, errors, 'r-', linewidth=1.5, alpha=0.7)
    axes[0].fill_between(timestamps, 0, errors, alpha=0.3, color='red')
    axes[0].axhline(y=np.mean(errors), color='blue', linestyle='--', linewidth=1.5, 
                    label=f'Mean: {np.mean(errors):.3f}m')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Translation Error (m)')
    axes[0].set_title('Absolute Trajectory Error Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(errors):.4f}m')
    axes[1].axvline(np.median(errors), color='green', linestyle='--', linewidth=2, 
                    label=f'Median: {np.median(errors):.4f}m')
    rmse = np.sqrt(np.mean(errors**2))
    axes[1].axvline(rmse, color='orange', linestyle='--', linewidth=2, 
                    label=f'RMSE: {rmse:.4f}m')
    axes[1].set_xlabel('Translation Error (m)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Error analysis saved: {output_path}")


def plot_pose_graph(est_traj, output_path):
    """
    Visualize pose graph with odometry edges.
    
    Shows trajectory nodes with odometry connectivity.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    xyz = est_traj.positions_xyz
    
    # Plot nodes (subsample for clarity)
    step = max(1, len(xyz) // 500)
    ax.scatter(xyz[::step, 0], xyz[::step, 1], c='blue', s=15, alpha=0.6, label='Poses')
    
    # Plot odometry edges (every Nth connection to avoid clutter)
    edge_step = max(1, len(xyz) // 100)
    for i in range(0, len(xyz) - edge_step, edge_step):
        ax.plot([xyz[i, 0], xyz[i+edge_step, 0]], 
               [xyz[i, 1], xyz[i+edge_step, 1]], 
               'gray', alpha=0.3, linewidth=0.5)
    
    # Highlight trajectory path
    ax.plot(xyz[:, 0], xyz[:, 1], 'b-', alpha=0.3, linewidth=1, label='Trajectory')
    
    # Start/end markers
    ax.scatter([xyz[0, 0]], [xyz[0, 1]], c='green', s=200, marker='^', 
               label='Start', zorder=3, edgecolors='black')
    ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], c='purple', s=200, marker='s', 
               label='End', zorder=3, edgecolors='black')
    
    ax.set_title('Pose Graph (Odometry Constraints)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Pose graph saved: {output_path}")


def plot_per_axis_errors(per_axis_data, output_path):
    """Plot error for each translation and rotation axis separately."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    timestamps = per_axis_data['timestamps']
    trans_errors = per_axis_data['translation']
    rot_errors = per_axis_data['rotation']
    
    # Translation errors
    axes[0, 0].plot(timestamps, trans_errors['x'], 'r-', linewidth=1.5, alpha=0.7, label='X error')
    axes[0, 0].axhline(y=np.mean(trans_errors['x']), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(trans_errors["x"]):.4f}m')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Error (m)')
    axes[0, 0].set_title('Translation Error: X-axis')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(timestamps, trans_errors['y'], 'g-', linewidth=1.5, alpha=0.7, label='Y error')
    axes[0, 1].axhline(y=np.mean(trans_errors['y']), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(trans_errors["y"]):.4f}m')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Error (m)')
    axes[0, 1].set_title('Translation Error: Y-axis')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(timestamps, trans_errors['z'], 'b-', linewidth=1.5, alpha=0.7, label='Z error')
    axes[0, 2].axhline(y=np.mean(trans_errors['z']), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(trans_errors["z"]):.4f}m')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Error (m)')
    axes[0, 2].set_title('Translation Error: Z-axis')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Rotation errors
    axes[1, 0].plot(timestamps, rot_errors['roll'], 'r-', linewidth=1.5, alpha=0.7, label='Roll error')
    axes[1, 0].axhline(y=np.mean(rot_errors['roll']), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(rot_errors["roll"]):.4f}°')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Error (deg)')
    axes[1, 0].set_title('Rotation Error: Roll')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(timestamps, rot_errors['pitch'], 'g-', linewidth=1.5, alpha=0.7, label='Pitch error')
    axes[1, 1].axhline(y=np.mean(rot_errors['pitch']), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(rot_errors["pitch"]):.4f}°')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (deg)')
    axes[1, 1].set_title('Rotation Error: Pitch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(timestamps, rot_errors['yaw'], 'b-', linewidth=1.5, alpha=0.7, label='Yaw error')
    axes[1, 2].axhline(y=np.mean(rot_errors['yaw']), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(rot_errors["yaw"]):.4f}°')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Error (deg)')
    axes[1, 2].set_title('Rotation Error: Yaw')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Per-axis error plot saved: {output_path}")


def plot_cumulative_error(ate_metric, timestamps, output_path):
    """Plot cumulative error over time."""
    errors = ate_metric.error
    timestamps = np.asarray(timestamps)
    
    if timestamps.shape[0] != errors.shape[0]:
        n = min(timestamps.shape[0], errors.shape[0])
        timestamps = timestamps[:n]
        errors = errors[:n]
    
    # Compute cumulative error (integral of absolute error)
    dt = np.diff(timestamps)
    dt = np.concatenate([[dt[0] if len(dt) > 0 else 0.0], dt])  # First element
    cumulative = np.cumsum(errors * dt)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Cumulative error over time
    axes[0].plot(timestamps, cumulative, 'purple', linewidth=2, alpha=0.8)
    axes[0].fill_between(timestamps, 0, cumulative, alpha=0.3, color='purple')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Cumulative Error (m·s)')
    axes[0].set_title('Cumulative Translation Error Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Error rate (derivative of cumulative)
    error_rate = np.gradient(cumulative, timestamps)
    axes[1].plot(timestamps, error_rate, 'orange', linewidth=1.5, alpha=0.7)
    axes[1].axhline(y=np.mean(error_rate), color='red', linestyle='--', 
                    label=f'Mean rate: {np.mean(error_rate):.4f} m/s')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error Rate (m/s)')
    axes[1].set_title('Error Accumulation Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Cumulative error plot saved: {output_path}")


def save_metrics_txt(ate_trans, ate_rot, rpe_results, per_axis_data, output_path, eval_diagnostics: dict | None = None, align_mode: str = "initial"):
    """Save metrics in human-readable text format."""
    ate_t_stats = ate_trans.get_all_statistics()
    ate_r_stats = ate_rot.get_all_statistics()
    rpe_1m = rpe_results.get('1m', {}).get('trans')
    rpe_1m_rmse = rpe_1m.get_all_statistics()['rmse'] if rpe_1m else float('nan')
    align_note = "initial-pose alignment (GT → estimate frame)" if align_mode == "initial" else "SE(3) Umeyama alignment (scale fixed)"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("FL-SLAM Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write("SUMMARY (parsable)\n")
        f.write("-" * 40 + "\n")
        f.write(f"ATE translation RMSE (m):     {ate_t_stats['rmse']:.6f}\n")
        f.write(f"ATE rotation RMSE (deg):       {ate_r_stats['rmse']:.6f}\n")
        f.write(f"RPE translation @ 1m (m/m):    {rpe_1m_rmse:.6f}\n")
        f.write("\n")

        f.write("ABSOLUTE TRAJECTORY ERROR (ATE)\n")
        f.write(f"  (after {align_note})\n")
        f.write("-" * 40 + "\n")
        
        f.write("Translation (m):\n")
        for key, val in ate_t_stats.items():
            f.write(f"  {key:12s}: {val:.6f} m\n")
        
        # Percentiles for translation
        trans_percentiles = compute_percentiles(ate_trans.error)
        f.write("\n  Percentiles:\n")
        for key, val in trans_percentiles.items():
            f.write(f"    {key:12s}: {val:.6f} m\n")
        
        # Distribution stats for translation
        trans_dist = compute_distribution_stats(ate_trans.error)
        f.write("\n  Distribution:\n")
        f.write(f"    Skewness:     {trans_dist['skewness']:.6f}\n")
        f.write(f"    Kurtosis:     {trans_dist['kurtosis']:.6f}\n")
        f.write(f"    IQR:          {trans_dist['iqr']:.6f} m\n")
        if 'normality_p_value' in trans_dist and trans_dist['normality_p_value'] is not None:
            f.write(f"    Normality:    {trans_dist['normality_test']} (p={trans_dist['normality_p_value']:.6f})\n")
        
        f.write("\nRotation (deg, angular error):\n")
        for key, val in ate_r_stats.items():
            f.write(f"  {key:12s}: {val:.6f} deg\n")
        
        # Percentiles for rotation
        rot_percentiles = compute_percentiles(ate_rot.error)
        f.write("\n  Percentiles:\n")
        for key, val in rot_percentiles.items():
            f.write(f"    {key:12s}: {val:.6f} deg\n")
        
        # Distribution stats for rotation
        rot_dist = compute_distribution_stats(ate_rot.error)
        f.write("\n  Distribution:\n")
        f.write(f"    Skewness:     {rot_dist['skewness']:.6f}\n")
        f.write(f"    Kurtosis:     {rot_dist['kurtosis']:.6f}\n")
        f.write(f"    IQR:          {rot_dist['iqr']:.6f} deg\n")
        if 'normality_p_value' in rot_dist and rot_dist['normality_p_value'] is not None:
            f.write(f"    Normality:    {rot_dist['normality_test']} (p={rot_dist['normality_p_value']:.6f})\n")

        if eval_diagnostics:
            f.write("\nEVALUATION DIAGNOSTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  kind: {eval_diagnostics.get('kind', 'unknown')}\n")
            f.write(f"  mean_rel_angle_deg: {eval_diagnostics.get('mean_rel_angle_deg', 0.0):.6f}\n")
            f.write(f"  std_rel_angle_deg:  {eval_diagnostics.get('std_rel_angle_deg', 0.0):.6f}\n")
            f.write(f"  avg_rel_angle_deg:  {eval_diagnostics.get('avg_rel_angle_deg', 0.0):.6f}\n")
            f.write(f"  avg_rel_rotvec:     {eval_diagnostics.get('avg_rel_rotvec')}\n")
            f.write(f"  avg_rel_axis:       {eval_diagnostics.get('avg_rel_axis')}\n")
            note = eval_diagnostics.get("note")
            if note:
                f.write(f"  note: {note}\n")

        f.write(f"\n\nPER-AXIS ATE (after {align_note})\n")
        f.write("-" * 40 + "\n")
        
        trans_errors = per_axis_data['translation']
        f.write("Translation RMSE per axis (X, Y, Z in meters):\n")
        for axis in ['x', 'y', 'z']:
            errors = trans_errors[axis]
            f.write(f"  {axis.upper()}-axis (m):\n")
            f.write(f"    RMSE:        {np.sqrt(np.mean(errors**2)):.6f} m\n")
            f.write(f"    Mean:        {np.mean(errors):.6f} m\n")
            f.write(f"    Median:      {np.median(errors):.6f} m\n")
            f.write(f"    Max:         {np.max(errors):.6f} m\n")
        
        rot_errors = per_axis_data['rotation']
        f.write("\nRotation RMSE per axis (roll, pitch, yaw in deg; Euler XYZ from relative rotation est_inv*gt):\n")
        for axis in ['roll', 'pitch', 'yaw']:
            errors = rot_errors[axis]
            f.write(f"  {axis.capitalize()} (deg):\n")
            f.write(f"    RMSE:        {np.sqrt(np.mean(errors**2)):.6f} deg\n")
            f.write(f"    Mean:        {np.mean(errors):.6f} deg\n")
            f.write(f"    Median:      {np.median(errors):.6f} deg\n")
            f.write(f"    Max:         {np.max(errors):.6f} deg\n")
        
        f.write("\n\nRELATIVE POSE ERROR (RPE)\n")
        f.write("-" * 40 + "\n")
        
        for scale, metrics_dict in rpe_results.items():
            f.write(f"\nScale: {scale}\n")
            
            rpe_t_stats = metrics_dict['trans'].get_all_statistics()
            f.write("  Translation:\n")
            for key, val in rpe_t_stats.items():
                f.write(f"    {key:12s}: {val:.6f} m/{scale}\n")
            
            rpe_r_stats = metrics_dict['rot'].get_all_statistics()
            f.write("  Rotation:\n")
            for key, val in rpe_r_stats.items():
                f.write(f"    {key:12s}: {val:.6f} deg/{scale}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"  Metrics (txt) saved: {output_path}")


def save_metrics_csv(ate_trans, ate_rot, rpe_results, per_axis_data, output_path):
    """Save all metrics to CSV for spreadsheet analysis."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header with new columns
        f.write("Metric,Type,Scale,Axis,RMSE,Mean,Median,Std,Min,Max,P75,P90,P95,P99,Skewness,Kurtosis,IQR\n")
        
        # ATE Translation (full)
        ate_t_stats = ate_trans.get_all_statistics()
        trans_percentiles = compute_percentiles(ate_trans.error)
        trans_dist = compute_distribution_stats(ate_trans.error)
        f.write(f"ATE,translation,full,all,"
                f"{ate_t_stats['rmse']:.6f},"
                f"{ate_t_stats['mean']:.6f},"
                f"{ate_t_stats['median']:.6f},"
                f"{ate_t_stats['std']:.6f},"
                f"{ate_t_stats['min']:.6f},"
                f"{ate_t_stats['max']:.6f},"
                f"{trans_percentiles['p75']:.6f},"
                f"{trans_percentiles['p90']:.6f},"
                f"{trans_percentiles['p95']:.6f},"
                f"{trans_percentiles['p99']:.6f},"
                f"{trans_dist['skewness']:.6f},"
                f"{trans_dist['kurtosis']:.6f},"
                f"{trans_dist['iqr']:.6f}\n")
        
        # ATE Translation per axis
        trans_errors = per_axis_data['translation']
        for axis in ['x', 'y', 'z']:
            errors = trans_errors[axis]
            axis_percentiles = compute_percentiles(errors)
            axis_dist = compute_distribution_stats(errors)
            f.write(f"ATE,translation,full,{axis},"
                    f"{np.sqrt(np.mean(errors**2)):.6f},"
                    f"{np.mean(errors):.6f},"
                    f"{np.median(errors):.6f},"
                    f"{np.std(errors):.6f},"
                    f"{np.min(errors):.6f},"
                    f"{np.max(errors):.6f},"
                    f"{axis_percentiles['p75']:.6f},"
                    f"{axis_percentiles['p90']:.6f},"
                    f"{axis_percentiles['p95']:.6f},"
                    f"{axis_percentiles['p99']:.6f},"
                    f"{axis_dist['skewness']:.6f},"
                    f"{axis_dist['kurtosis']:.6f},"
                    f"{axis_dist['iqr']:.6f}\n")
        
        # ATE Rotation (full)
        ate_r_stats = ate_rot.get_all_statistics()
        rot_percentiles = compute_percentiles(ate_rot.error)
        rot_dist = compute_distribution_stats(ate_rot.error)
        f.write(f"ATE,rotation,full,all,"
                f"{ate_r_stats['rmse']:.6f},"
                f"{ate_r_stats['mean']:.6f},"
                f"{ate_r_stats['median']:.6f},"
                f"{ate_r_stats['std']:.6f},"
                f"{ate_r_stats['min']:.6f},"
                f"{ate_r_stats['max']:.6f},"
                f"{rot_percentiles['p75']:.6f},"
                f"{rot_percentiles['p90']:.6f},"
                f"{rot_percentiles['p95']:.6f},"
                f"{rot_percentiles['p99']:.6f},"
                f"{rot_dist['skewness']:.6f},"
                f"{rot_dist['kurtosis']:.6f},"
                f"{rot_dist['iqr']:.6f}\n")
        
        # ATE Rotation per axis
        rot_errors = per_axis_data['rotation']
        for axis in ['roll', 'pitch', 'yaw']:
            errors = rot_errors[axis]
            axis_percentiles = compute_percentiles(errors)
            axis_dist = compute_distribution_stats(errors)
            f.write(f"ATE,rotation,full,{axis},"
                    f"{np.sqrt(np.mean(errors**2)):.6f},"
                    f"{np.mean(errors):.6f},"
                    f"{np.median(errors):.6f},"
                    f"{np.std(errors):.6f},"
                    f"{np.min(errors):.6f},"
                    f"{np.max(errors):.6f},"
                    f"{axis_percentiles['p75']:.6f},"
                    f"{axis_percentiles['p90']:.6f},"
                    f"{axis_percentiles['p95']:.6f},"
                    f"{axis_percentiles['p99']:.6f},"
                    f"{axis_dist['skewness']:.6f},"
                    f"{axis_dist['kurtosis']:.6f},"
                    f"{axis_dist['iqr']:.6f}\n")
        
        # RPE at multiple scales
        for scale, metrics_dict in rpe_results.items():
            rpe_t_stats = metrics_dict['trans'].get_all_statistics()
            rpe_t_percentiles = compute_percentiles(metrics_dict['trans'].error)
            rpe_t_dist = compute_distribution_stats(metrics_dict['trans'].error)
            f.write(f"RPE,translation,{scale},all,"
                    f"{rpe_t_stats['rmse']:.6f},"
                    f"{rpe_t_stats['mean']:.6f},"
                    f"{rpe_t_stats['median']:.6f},"
                    f"{rpe_t_stats['std']:.6f},"
                    f"{rpe_t_stats['min']:.6f},"
                    f"{rpe_t_stats['max']:.6f},"
                    f"{rpe_t_percentiles['p75']:.6f},"
                    f"{rpe_t_percentiles['p90']:.6f},"
                    f"{rpe_t_percentiles['p95']:.6f},"
                    f"{rpe_t_percentiles['p99']:.6f},"
                    f"{rpe_t_dist['skewness']:.6f},"
                    f"{rpe_t_dist['kurtosis']:.6f},"
                    f"{rpe_t_dist['iqr']:.6f}\n")
            
            rpe_r_stats = metrics_dict['rot'].get_all_statistics()
            rpe_r_percentiles = compute_percentiles(metrics_dict['rot'].error)
            rpe_r_dist = compute_distribution_stats(metrics_dict['rot'].error)
            f.write(f"RPE,rotation,{scale},all,"
                    f"{rpe_r_stats['rmse']:.6f},"
                    f"{rpe_r_stats['mean']:.6f},"
                    f"{rpe_r_stats['median']:.6f},"
                    f"{rpe_r_stats['std']:.6f},"
                    f"{rpe_r_stats['min']:.6f},"
                    f"{rpe_r_stats['max']:.6f},"
                    f"{rpe_r_percentiles['p75']:.6f},"
                    f"{rpe_r_percentiles['p90']:.6f},"
                    f"{rpe_r_percentiles['p95']:.6f},"
                    f"{rpe_r_percentiles['p99']:.6f},"
                    f"{rpe_r_dist['skewness']:.6f},"
                    f"{rpe_r_dist['kurtosis']:.6f},"
                    f"{rpe_r_dist['iqr']:.6f}\n")
    
    print(f"  Metrics (csv) saved: {output_path}")


def save_metrics_json(ate_trans, ate_rot, rpe_results, per_axis_data, output_path, eval_diagnostics: dict | None = None, align_mode: str = "initial"):
    """Save all metrics to structured JSON format."""
    from datetime import datetime

    # Build comprehensive metrics dictionary
    metrics_dict = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'evaluation_tool': 'evaluate_slam.py',
            'align_mode': align_mode,
            'diagnostics': eval_diagnostics or None,
        },
        'ate': {
            'translation': {
                'full': {
                    **ate_trans.get_all_statistics(),
                    'percentiles': compute_percentiles(ate_trans.error),
                    'distribution': compute_distribution_stats(ate_trans.error)
                },
                'per_axis': {}
            },
            'rotation': {
                'full': {
                    **ate_rot.get_all_statistics(),
                    'percentiles': compute_percentiles(ate_rot.error),
                    'distribution': compute_distribution_stats(ate_rot.error)
                },
                'per_axis': {}
            }
        },
        'rpe': {}
    }
    
    # Per-axis translation
    trans_errors = per_axis_data['translation']
    for axis in ['x', 'y', 'z']:
        errors = trans_errors[axis]
        metrics_dict['ate']['translation']['per_axis'][axis] = {
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'percentiles': compute_percentiles(errors),
            'distribution': compute_distribution_stats(errors)
        }
    
    # Per-axis rotation
    rot_errors = per_axis_data['rotation']
    for axis in ['roll', 'pitch', 'yaw']:
        errors = rot_errors[axis]
        metrics_dict['ate']['rotation']['per_axis'][axis] = {
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'percentiles': compute_percentiles(errors),
            'distribution': compute_distribution_stats(errors)
        }
    
    # RPE at multiple scales
    for scale, metrics_dict_scale in rpe_results.items():
        metrics_dict['rpe'][scale] = {
            'translation': {
                **metrics_dict_scale['trans'].get_all_statistics(),
                'percentiles': compute_percentiles(metrics_dict_scale['trans'].error),
                'distribution': compute_distribution_stats(metrics_dict_scale['trans'].error)
            },
            'rotation': {
                **metrics_dict_scale['rot'].get_all_statistics(),
                'percentiles': compute_percentiles(metrics_dict_scale['rot'].error),
                'distribution': compute_distribution_stats(metrics_dict_scale['rot'].error)
            }
        }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)
    
    metrics_dict = convert_to_native(metrics_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  Metrics (json) saved: {output_path}")


def main(gt_file, est_file, output_dir, op_report_path, require_imu=True, align_mode="initial", gt_swap_yz=False, apply_est_to_gt_axes=False, verbose=False):
    """Run full evaluation pipeline.

    align_mode: "initial" = align first pose only (GT → estimate frame; valid for relative SLAM).
                "umeyama" = full SE(3) Umeyama over whole trajectory (best-fit).
    gt_swap_yz: if True, swap GT Y and Z axes before evaluation (diagnostic for axis-convention mismatch).
    apply_est_to_gt_axes: if True, transform EST by R_EST_TO_GT so EST is in GT axis convention (same-plane plots).
    verbose: if True, print first 5 rows of positions (TUM: t, x, y, z, qx, qy, qz, qw) to verify column order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FL-SLAM Trajectory Evaluation")
    print("=" * 60)

    # Load trajectories
    print("\n1. Loading trajectories...")
    gt_traj = load_trajectory(gt_file)
    est_traj = load_trajectory(est_file)
    if gt_swap_yz:
        gt_traj = swap_gt_yz(gt_traj)
        print("   Applied --gt-swap-yz: GT Y and Z axes swapped (diagnostic for axis-convention mismatch).")
    if apply_est_to_gt_axes:
        est_traj = apply_est_to_gt_convention(est_traj)
        print("   Applied --apply-est-to-gt-convention: EST expressed in GT axis convention (R_EST_TO_GT).")
    # Raw ptp (before alignment): GT Z should be ~0.04 m for planar robot
    ptp_gt_raw = np.ptp(gt_traj.positions_xyz, axis=0)
    ptp_est_raw = np.ptp(est_traj.positions_xyz, axis=0)
    print(f"   [raw] GT  ptp(x,y,z) (m): {ptp_gt_raw[0]:.4f}, {ptp_gt_raw[1]:.4f}, {ptp_gt_raw[2]:.4f}")
    print(f"   [raw] EST ptp(x,y,z) (m): {ptp_est_raw[0]:.4f}, {ptp_est_raw[1]:.4f}, {ptp_est_raw[2]:.4f}")
    if ptp_gt_raw[2] > 0.5:
        print("   [raw] WARNING: GT ptp(z) large — expected ~0.04 m for planar; check TUM column order (t x y z qx qy qz qw).")
    if verbose:
        print("   First 5 positions (x,y,z) — GT:")
        for i in range(min(5, len(gt_traj.positions_xyz))):
            print(f"     {gt_traj.positions_xyz[i, 0]:.4f}, {gt_traj.positions_xyz[i, 1]:.4f}, {gt_traj.positions_xyz[i, 2]:.4f}")
        print("   First 5 positions (x,y,z) — EST:")
        for i in range(min(5, len(est_traj.positions_xyz))):
            print(f"     {est_traj.positions_xyz[i, 0]:.4f}, {est_traj.positions_xyz[i, 1]:.4f}, {est_traj.positions_xyz[i, 2]:.4f}")
    print(f"   Ground truth: {len(gt_traj.timestamps)} poses")
    print(f"   Estimated: {len(est_traj.timestamps)} poses")

    # Validate trajectories
    print("\n2. Validating trajectories...")
    validate_trajectory(gt_traj, "Ground Truth", strict=True)
    # Use non-strict validation for estimated trajectory to allow evaluation even with extreme values
    validate_trajectory(est_traj, "Estimated", strict=False)

    # OpReport validation (IMU + Frobenius diagnostics)
    print("\n3. Validating OpReports (runtime health + audit)...")
    validate_op_reports(op_report_path, require_imu=require_imu)

    # Compute ATE (translation + rotation)
    print("\n4. Computing Absolute Trajectory Error (ATE)...")
    print(f"   Alignment: {align_mode} (GT in estimate frame)")
    ate_trans, ate_rot, gt_aligned, est_aligned = compute_ate_full(gt_traj, est_traj, align_mode=align_mode)
    eval_diag = diagnose_constant_rotation_offset(gt_aligned, est_aligned)
    
    ate_t_stats = ate_trans.get_all_statistics()
    print("   Translation ATE:")
    print(f"     RMSE: {ate_t_stats['rmse']:.4f} m")
    print(f"     Mean: {ate_t_stats['mean']:.4f} m")
    print(f"     Max:  {ate_t_stats['max']:.4f} m")
    
    # Percentiles for translation
    trans_percentiles = compute_percentiles(ate_trans.error)
    print(f"     P95:  {trans_percentiles['p95']:.4f} m")
    print(f"     P99:  {trans_percentiles['p99']:.4f} m")
    
    ate_r_stats = ate_rot.get_all_statistics()
    print("   Rotation ATE:")
    print(f"     RMSE: {ate_r_stats['rmse']:.4f} deg")
    print(f"     Mean: {ate_r_stats['mean']:.4f} deg")
    print(f"     Max:  {ate_r_stats['max']:.4f} deg")
    
    # Percentiles for rotation
    rot_percentiles = compute_percentiles(ate_rot.error)
    print(f"     P95:  {rot_percentiles['p95']:.4f} deg")
    print(f"     P99:  {rot_percentiles['p99']:.4f} deg")

    if eval_diag:
        print("\n   WARNING: Rotation looks like a near-constant frame offset:")
        print(f"     mean_rel_angle: {eval_diag['mean_rel_angle_deg']:.2f} deg (std: {eval_diag['std_rel_angle_deg']:.2f})")
        print(f"     avg_rel_angle:  {eval_diag['avg_rel_angle_deg']:.2f} deg")
        print(f"     avg_rel_rotvec: {eval_diag['avg_rel_rotvec']}")
        print(f"     note: {eval_diag['note']}")
    
    # Compute per-axis errors
    print("\n5. Computing per-axis errors...")
    per_axis_data = compute_per_axis_errors(gt_traj, est_traj, align_mode=align_mode)
    
    print("   Translation per axis:")
    for axis in ['x', 'y', 'z']:
        errors = per_axis_data['translation'][axis]
        print(f"     {axis.upper()}-axis RMSE: {np.sqrt(np.mean(errors**2)):.4f} m")
    
    print("   Rotation per axis:")
    for axis in ['roll', 'pitch', 'yaw']:
        errors = per_axis_data['rotation'][axis]
        print(f"     {axis.capitalize()} RMSE: {np.sqrt(np.mean(errors**2)):.4f} deg")
    
    # Compute multi-scale RPE
    print("\n6. Computing Relative Pose Error (RPE) at multiple scales...")
    rpe_results = compute_rpe_multi_scale(gt_traj, est_traj, align_mode=align_mode)
    
    for scale, rpe_dict in rpe_results.items():
        rpe_t_stats = rpe_dict['trans'].get_all_statistics()
        rpe_r_stats = rpe_dict['rot'].get_all_statistics()
        print(f"   RPE @ {scale}:")
        print(f"     Translation RMSE: {rpe_t_stats['rmse']:.4f} m/{scale}")
        print(f"     Rotation RMSE:    {rpe_r_stats['rmse']:.4f} deg/{scale}")
    
    # Generate plots
    print("\n7. Generating plots...")
    plot_trajectories(gt_aligned, est_aligned, output_dir / "trajectory_comparison.png")
    plot_trajectory_heatmap(gt_aligned, est_aligned, ate_trans, output_dir / "trajectory_heatmap.png")
    plot_error_over_time(ate_trans, est_aligned.timestamps, output_dir / "error_analysis.png")
    plot_per_axis_errors(per_axis_data, output_dir / "error_per_axis.png")
    plot_cumulative_error(ate_trans, est_aligned.timestamps, output_dir / "cumulative_error.png")
    plot_pose_graph(est_aligned, output_dir / "pose_graph.png")
    
    # Save metrics
    print("\n8. Saving metrics...")
    save_metrics_txt(ate_trans, ate_rot, rpe_results, per_axis_data, output_dir / "metrics.txt", eval_diagnostics=eval_diag, align_mode=align_mode)
    save_metrics_csv(ate_trans, ate_rot, rpe_results, per_axis_data, output_dir / "metrics.csv")
    save_metrics_json(ate_trans, ate_rot, rpe_results, per_axis_data, output_dir / "metrics.json", eval_diagnostics=eval_diag, align_mode=align_mode)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  ATE translation RMSE (m):        {ate_t_stats['rmse']:.4f} m  (P95: {trans_percentiles['p95']:.4f} m)")
    print(f"  ATE rotation RMSE (deg):         {ate_r_stats['rmse']:.4f} deg  (P95: {rot_percentiles['p95']:.4f} deg)")
    if '1m' in rpe_results:
        print(f"  RPE translation @ 1 m (m/m):    {rpe_results['1m']['trans'].get_all_statistics()['rmse']:.4f} m/m")
    
    # Per-axis summary
    print("\n  Per-axis translation RMSE (X, Y, Z in meters):")
    for axis in ['x', 'y', 'z']:
        errors = per_axis_data['translation'][axis]
        print(f"    {axis.upper()}: {np.sqrt(np.mean(errors**2)):.4f} m")
    
    print("\n  Per-axis rotation RMSE (roll, pitch, yaw in degrees; Euler XYZ from relative rotation est_inv*gt):")
    for axis in ['roll', 'pitch', 'yaw']:
        errors = per_axis_data['rotation'][axis]
        print(f"    {axis.capitalize()}: {np.sqrt(np.mean(errors**2)):.4f} deg")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - metrics.txt, metrics.csv, metrics.json (comprehensive statistics)")
    print(f"  - trajectory_comparison.png, trajectory_heatmap.png")
    print(f"  - error_analysis.png, error_per_axis.png, cumulative_error.png")
    print(f"  - pose_graph.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth", help="Aligned ground truth TUM file (time-aligned)")
    ap.add_argument("estimated", help="Estimated TUM file")
    ap.add_argument("output_dir", help="Output directory")
    ap.add_argument("op_report", help="OpReport JSONL file")
    ap.add_argument("--no-imu", action="store_true", help="Disable IMU OpReport requirements")
    ap.add_argument(
        "--align",
        choices=("initial", "umeyama"),
        default="initial",
        help="Alignment for metrics: initial = first pose only (GT → estimate frame; valid for relative SLAM). umeyama = full SE(3) best-fit (default: initial)",
    )
    ap.add_argument(
        "--gt-swap-yz",
        action="store_true",
        help="Swap GT Y and Z axes before evaluation (diagnostic when 'our XZ looks like GT XY' — axis-convention mismatch).",
    )
    ap.add_argument(
        "--apply-est-to-gt-convention",
        action="store_true",
        help="Transform EST by R_EST_TO_GT (GT x≈-EST z, GT y≈EST x, GT z≈-EST y) so plots are same-plane in GT axes.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print first 5 rows of positions (x,y,z) and ptp ranges to verify TUM column order.",
    )
    args = ap.parse_args()

    try:
        main(
            args.ground_truth,
            args.estimated,
            args.output_dir,
            args.op_report,
            require_imu=not args.no_imu,
            align_mode=args.align,
            gt_swap_yz=args.gt_swap_yz,
            apply_est_to_gt_axes=args.apply_est_to_gt_convention,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nERROR: Evaluation failed with exception: {type(e).__name__}")
        print(f"Message: {e}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)
