#!/usr/bin/env python3
"""
Diagnose trajectory alignment between estimated trajectory and ground truth.

This script:
1. Loads both trajectories (estimate from odom, GT from VRPN)
2. Analyzes the coordinate systems and origins
3. Computes the transform needed to align them
4. Validates the alignment by checking overlap
5. Outputs the exact transforms needed for evaluation

The goal is to determine HOW to transform the estimate so it can be
compared to GT in a meaningful way.
"""

import argparse
import sqlite3
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tools"))

from rosbag_sqlite_utils import resolve_db3_path, topic_id


def load_odom_trajectory(cur, topic: str, max_poses: int = 5000) -> Optional[np.ndarray]:
    """Load trajectory from odometry messages. Returns Nx7 array [t, x, y, z, qx, qy, qz, qw]."""
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry

    tid = topic_id(cur, topic)
    if tid is None:
        return None

    poses = []
    for row in cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
        (tid, max_poses),
    ):
        ts_ns, data = row
        try:
            msg = deserialize_message(data, Odometry)
            t = ts_ns / 1e9
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            poses.append([t, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        except Exception:
            continue

    return np.array(poses) if poses else None


def load_gt_trajectory(cur, topic: str, max_poses: int = 50000) -> Optional[np.ndarray]:
    """Load trajectory from PoseStamped messages. Returns Nx7 array [t, x, y, z, qx, qy, qz, qw]."""
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseStamped

    tid = topic_id(cur, topic)
    if tid is None:
        return None

    poses = []
    for row in cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
        (tid, max_poses),
    ):
        ts_ns, data = row
        try:
            msg = deserialize_message(data, PoseStamped)
            t = ts_ns / 1e9
            pos = msg.pose.position
            ori = msg.pose.orientation
            poses.append([t, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        except Exception:
            continue

    return np.array(poses) if poses else None


def compute_trajectory_stats(traj: np.ndarray, name: str) -> dict:
    """Compute statistics about a trajectory."""
    pos = traj[:, 1:4]

    # Position stats
    pos_mean = np.mean(pos, axis=0)
    pos_std = np.std(pos, axis=0)
    pos_min = np.min(pos, axis=0)
    pos_max = np.max(pos, axis=0)
    pos_ptp = np.ptp(pos, axis=0)

    # Time stats
    t = traj[:, 0]
    duration = t[-1] - t[0]

    # Distance traveled
    deltas = np.diff(pos, axis=0)
    dist_traveled = np.sum(np.linalg.norm(deltas, axis=1))

    return {
        'name': name,
        'n_poses': len(traj),
        'duration_sec': duration,
        'pos_start': pos[0],
        'pos_end': pos[-1],
        'pos_mean': pos_mean,
        'pos_std': pos_std,
        'pos_min': pos_min,
        'pos_max': pos_max,
        'pos_ptp': pos_ptp,
        'dist_traveled': dist_traveled,
    }


def find_initial_alignment(est: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute initial alignment: transform that makes EST[0] = GT[0].

    Returns: (R_align, t_align) such that:
        p_aligned = R_align @ p_est + t_align
    """
    # Get first poses
    est_pos0 = est[0, 1:4]
    gt_pos0 = gt[0, 1:4]

    est_quat0 = est[0, 4:8]  # [qx, qy, qz, qw]
    gt_quat0 = gt[0, 4:8]

    # Rotation alignment: R_align = R_gt0 @ R_est0^{-1}
    R_est0 = R.from_quat(est_quat0)
    R_gt0 = R.from_quat(gt_quat0)
    R_align = R_gt0 * R_est0.inv()

    # Translation alignment: t_align = gt_pos0 - R_align @ est_pos0
    t_align = gt_pos0 - R_align.apply(est_pos0)

    return R_align.as_matrix(), t_align


def apply_alignment(traj: np.ndarray, R_align: np.ndarray, t_align: np.ndarray) -> np.ndarray:
    """Apply alignment transform to trajectory."""
    aligned = traj.copy()

    # Transform positions
    pos = traj[:, 1:4]
    pos_aligned = (R_align @ pos.T).T + t_align
    aligned[:, 1:4] = pos_aligned

    # Transform orientations
    R_align_scipy = R.from_matrix(R_align)
    for i in range(len(traj)):
        quat = traj[i, 4:8]
        R_orig = R.from_quat(quat)
        R_new = R_align_scipy * R_orig
        aligned[i, 4:8] = R_new.as_quat()

    return aligned


def compute_umeyama_alignment(est: np.ndarray, gt: np.ndarray,
                               max_pairs: int = 1000) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Umeyama alignment (best-fit SE(3) without scale).

    Returns: (R, t, rmse_before) where p_aligned = R @ p_est + t
    """
    # Subsample for efficiency
    step = max(1, len(est) // max_pairs)
    est_sub = est[::step, 1:4]

    # Find corresponding GT poses by time
    gt_times = gt[:, 0]
    est_times = est[::step, 0]

    pairs_est = []
    pairs_gt = []

    for i, t_est in enumerate(est_times):
        # Find closest GT time
        idx = np.argmin(np.abs(gt_times - t_est))
        if np.abs(gt_times[idx] - t_est) < 0.1:  # Within 100ms
            pairs_est.append(est_sub[i])
            pairs_gt.append(gt[idx, 1:4])

    if len(pairs_est) < 3:
        return np.eye(3), np.zeros(3), float('inf')

    pairs_est = np.array(pairs_est)
    pairs_gt = np.array(pairs_gt)

    # Umeyama algorithm (no scale)
    mu_est = np.mean(pairs_est, axis=0)
    mu_gt = np.mean(pairs_gt, axis=0)

    est_centered = pairs_est - mu_est
    gt_centered = pairs_gt - mu_gt

    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)

    R_align = Vt.T @ U.T

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T

    t_align = mu_gt - R_align @ mu_est

    # Compute RMSE before alignment
    errors_before = np.linalg.norm(pairs_est - pairs_gt, axis=1)
    rmse_before = np.sqrt(np.mean(errors_before**2))

    return R_align, t_align, rmse_before


def compute_ate(est: np.ndarray, gt: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute Absolute Trajectory Error by matching timestamps."""
    gt_times = gt[:, 0]

    errors = []
    for i in range(len(est)):
        t_est = est[i, 0]
        pos_est = est[i, 1:4]

        # Find closest GT time
        idx = np.argmin(np.abs(gt_times - t_est))
        if np.abs(gt_times[idx] - t_est) < 0.1:
            pos_gt = gt[idx, 1:4]
            error = np.linalg.norm(pos_est - pos_gt)
            errors.append(error)

    errors = np.array(errors)
    if len(errors) == 0:
        return float('inf'), np.array([])

    rmse = np.sqrt(np.mean(errors**2))
    return rmse, errors


def analyze_axis_correspondence(est: np.ndarray, gt: np.ndarray) -> dict:
    """
    Analyze which EST axis corresponds to which GT axis.

    Uses correlation between motion in each axis.
    """
    # Use motion (deltas) rather than absolute positions
    est_delta = np.diff(est[:, 1:4], axis=0)
    gt_times = gt[:, 0]
    est_times = est[:-1, 0]

    # Find matching GT deltas
    gt_deltas = []
    est_deltas = []

    for i in range(len(est_times)):
        t = est_times[i]
        idx = np.argmin(np.abs(gt_times[:-1] - t))
        if np.abs(gt_times[idx] - t) < 0.1:
            gt_delta = gt[idx+1, 1:4] - gt[idx, 1:4]
            gt_deltas.append(gt_delta)
            est_deltas.append(est_delta[i])

    if len(gt_deltas) < 10:
        return {'error': 'Insufficient matching deltas'}

    est_deltas = np.array(est_deltas)
    gt_deltas = np.array(gt_deltas)

    # Compute correlation matrix between axes
    corr_matrix = np.zeros((3, 3))
    axis_names = ['X', 'Y', 'Z']

    for i in range(3):  # EST axes
        for j in range(3):  # GT axes
            # Correlation between EST axis i and GT axis j
            if np.std(est_deltas[:, i]) > 1e-6 and np.std(gt_deltas[:, j]) > 1e-6:
                corr = np.corrcoef(est_deltas[:, i], gt_deltas[:, j])[0, 1]
                corr_matrix[i, j] = corr

    # Find best axis mapping
    mapping = {}
    for i in range(3):
        best_j = np.argmax(np.abs(corr_matrix[i, :]))
        sign = '+' if corr_matrix[i, best_j] > 0 else '-'
        mapping[f'EST_{axis_names[i]}'] = f'{sign}GT_{axis_names[best_j]} (corr={corr_matrix[i, best_j]:.3f})'

    return {
        'correlation_matrix': corr_matrix,
        'axis_mapping': mapping,
    }


def main():
    ap = argparse.ArgumentParser(description="Diagnose trajectory alignment for evaluation")
    ap.add_argument("bag_path", help="Bag directory containing *.db3")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--gt-topic", default="/vrpn_client_node/UGV/pose")
    ap.add_argument("--output-aligned", help="Output aligned estimate trajectory (TUM format)")
    args = ap.parse_args()

    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        print(f"ERROR: Could not locate *.db3 under '{args.bag_path}'", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        import rclpy
        rclpy.init()

        print("=" * 100)
        print("TRAJECTORY ALIGNMENT DIAGNOSTIC")
        print("=" * 100)
        print(f"Bag: {db_path}")
        print()

        # Load trajectories
        print("Loading trajectories...")
        odom_traj = load_odom_trajectory(cur, args.odom_topic)
        gt_traj = load_gt_trajectory(cur, args.gt_topic)

        if odom_traj is None:
            print(f"ERROR: Could not load odom from {args.odom_topic}")
            return 1
        if gt_traj is None:
            print(f"ERROR: Could not load GT from {args.gt_topic}")
            return 1

        print(f"  Odom: {len(odom_traj)} poses")
        print(f"  GT:   {len(gt_traj)} poses")
        print()

        # =====================================================================
        # 1. RAW TRAJECTORY STATISTICS
        # =====================================================================
        print("=" * 100)
        print("1. RAW TRAJECTORY STATISTICS (before any alignment)")
        print("=" * 100)

        odom_stats = compute_trajectory_stats(odom_traj, "Odometry")
        gt_stats = compute_trajectory_stats(gt_traj, "Ground Truth")

        for stats in [odom_stats, gt_stats]:
            print(f"\n{stats['name']}:")
            print(f"  Poses: {stats['n_poses']}, Duration: {stats['duration_sec']:.1f}s")
            print(f"  Start position: [{stats['pos_start'][0]:.3f}, {stats['pos_start'][1]:.3f}, {stats['pos_start'][2]:.3f}]")
            print(f"  End position:   [{stats['pos_end'][0]:.3f}, {stats['pos_end'][1]:.3f}, {stats['pos_end'][2]:.3f}]")
            print(f"  Position range:")
            print(f"    X: [{stats['pos_min'][0]:.3f}, {stats['pos_max'][0]:.3f}] (extent: {stats['pos_ptp'][0]:.3f}m)")
            print(f"    Y: [{stats['pos_min'][1]:.3f}, {stats['pos_max'][1]:.3f}] (extent: {stats['pos_ptp'][1]:.3f}m)")
            print(f"    Z: [{stats['pos_min'][2]:.3f}, {stats['pos_max'][2]:.3f}] (extent: {stats['pos_ptp'][2]:.3f}m)")
            print(f"  Distance traveled: {stats['dist_traveled']:.3f}m")

        print()

        # =====================================================================
        # 2. AXIS CORRESPONDENCE ANALYSIS
        # =====================================================================
        print("=" * 100)
        print("2. AXIS CORRESPONDENCE ANALYSIS")
        print("=" * 100)
        print("(Which EST axis corresponds to which GT axis?)")
        print()

        axis_analysis = analyze_axis_correspondence(odom_traj, gt_traj)

        if 'error' in axis_analysis:
            print(f"  {axis_analysis['error']}")
        else:
            print("  Correlation matrix (EST rows, GT cols):")
            corr = axis_analysis['correlation_matrix']
            print(f"           GT_X     GT_Y     GT_Z")
            for i, name in enumerate(['EST_X', 'EST_Y', 'EST_Z']):
                print(f"  {name}:  {corr[i,0]:7.3f}  {corr[i,1]:7.3f}  {corr[i,2]:7.3f}")

            print("\n  Best axis mapping:")
            for k, v in axis_analysis['axis_mapping'].items():
                print(f"    {k} → {v}")

            # Derive R_EST_TO_GT from correlations
            print("\n  Derived rotation matrix (EST → GT):")
            R_derived = np.zeros((3, 3))
            for i in range(3):
                best_j = np.argmax(np.abs(corr[i, :]))
                R_derived[best_j, i] = np.sign(corr[i, best_j])
            print(f"  R_EST_TO_GT =")
            for row in R_derived:
                print(f"    [{row[0]:5.1f}, {row[1]:5.1f}, {row[2]:5.1f}]")

        print()

        # =====================================================================
        # 3. INITIAL ALIGNMENT (first pose matching)
        # =====================================================================
        print("=" * 100)
        print("3. INITIAL ALIGNMENT (match first pose)")
        print("=" * 100)

        R_init, t_init = find_initial_alignment(odom_traj, gt_traj)

        print(f"\n  Alignment transform (EST → GT at t=0):")
        print(f"  Translation: [{t_init[0]:.4f}, {t_init[1]:.4f}, {t_init[2]:.4f}]")
        print(f"  Rotation (rotvec): {R.from_matrix(R_init).as_rotvec()}")
        print(f"  Rotation (euler deg): {R.from_matrix(R_init).as_euler('xyz', degrees=True)}")

        # Apply initial alignment
        odom_aligned_init = apply_alignment(odom_traj, R_init, t_init)

        # Compute ATE after initial alignment
        ate_init, errors_init = compute_ate(odom_aligned_init, gt_traj)
        print(f"\n  ATE after initial alignment: {ate_init:.4f}m RMSE")
        if len(errors_init) > 0:
            print(f"    Mean: {np.mean(errors_init):.4f}m, Max: {np.max(errors_init):.4f}m")

        print()

        # =====================================================================
        # 4. UMEYAMA ALIGNMENT (best-fit SE(3))
        # =====================================================================
        print("=" * 100)
        print("4. UMEYAMA ALIGNMENT (best-fit SE(3), no scale)")
        print("=" * 100)

        R_umeyama, t_umeyama, rmse_before = compute_umeyama_alignment(odom_traj, gt_traj)

        print(f"\n  Alignment transform (EST → GT best-fit):")
        print(f"  Translation: [{t_umeyama[0]:.4f}, {t_umeyama[1]:.4f}, {t_umeyama[2]:.4f}]")
        print(f"  Rotation matrix:")
        for row in R_umeyama:
            print(f"    [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
        print(f"  Rotation (rotvec): {R.from_matrix(R_umeyama).as_rotvec()}")
        print(f"  Rotation (euler deg): {R.from_matrix(R_umeyama).as_euler('xyz', degrees=True)}")

        # Apply Umeyama alignment
        odom_aligned_umeyama = apply_alignment(odom_traj, R_umeyama, t_umeyama)

        # Compute ATE after Umeyama alignment
        ate_umeyama, errors_umeyama = compute_ate(odom_aligned_umeyama, gt_traj)
        print(f"\n  ATE before alignment: {rmse_before:.4f}m RMSE")
        print(f"  ATE after Umeyama alignment: {ate_umeyama:.4f}m RMSE")
        if len(errors_umeyama) > 0:
            print(f"    Mean: {np.mean(errors_umeyama):.4f}m, Max: {np.max(errors_umeyama):.4f}m")

        print()

        # =====================================================================
        # 5. RECOMMENDATIONS
        # =====================================================================
        print("=" * 100)
        print("5. RECOMMENDATIONS FOR EVALUATION")
        print("=" * 100)

        # Check if there's a major axis permutation needed
        if 'correlation_matrix' in axis_analysis:
            corr = axis_analysis['correlation_matrix']
            # Check if axes are scrambled (off-diagonal correlations are strongest)
            diag_corr = np.abs([corr[0,0], corr[1,1], corr[2,2]])
            max_offdiag = max(np.abs(corr[0,1]), np.abs(corr[0,2]),
                             np.abs(corr[1,0]), np.abs(corr[1,2]),
                             np.abs(corr[2,0]), np.abs(corr[2,1]))

            if max_offdiag > np.max(diag_corr):
                print("\n  ⚠️  AXIS PERMUTATION DETECTED")
                print("  The estimate and GT use DIFFERENT axis conventions.")
                print("  Apply R_EST_TO_GT before comparing:")
                print(f"  R_EST_TO_GT = {R_derived.tolist()}")
            else:
                print("\n  ✓ Axes appear to be aligned (same convention)")

        # Check Z behavior
        odom_z_ptp = odom_stats['pos_ptp'][2]
        gt_z_ptp = gt_stats['pos_ptp'][2]

        if odom_z_ptp > 1.0 and gt_z_ptp < 0.1:
            print("\n  ⚠️  ODOM Z IS DRIFTING")
            print(f"  Odom Z extent: {odom_z_ptp:.2f}m, GT Z extent: {gt_z_ptp:.2f}m")
            print("  For planar robot, estimate Z should be nearly constant.")
            print("  This may indicate Z is being estimated incorrectly.")

        # Recommend alignment method
        print("\n  RECOMMENDED ALIGNMENT METHOD:")
        if ate_init < ate_umeyama * 1.5:
            print("  → Use INITIAL alignment (simpler, first-pose matching)")
            print(f"    R = {R_init.tolist()}")
            print(f"    t = {t_init.tolist()}")
            best_aligned = odom_aligned_init
            best_name = "initial"
        else:
            print("  → Use UMEYAMA alignment (better fit)")
            print(f"    R = {R_umeyama.tolist()}")
            print(f"    t = {t_umeyama.tolist()}")
            best_aligned = odom_aligned_umeyama
            best_name = "umeyama"

        # Output aligned trajectory
        if args.output_aligned:
            print(f"\n  Saving aligned trajectory to: {args.output_aligned}")
            with open(args.output_aligned, 'w') as f:
                for row in best_aligned:
                    # TUM format: timestamp x y z qx qy qz qw
                    f.write(f"{row[0]:.9f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} "
                           f"{row[4]:.6f} {row[5]:.6f} {row[6]:.6f} {row[7]:.6f}\n")
            print(f"  Saved {len(best_aligned)} poses ({best_name} alignment)")

        print()
        print("=" * 100)

        rclpy.shutdown()
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
