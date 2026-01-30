#!/usr/bin/env python3
"""
2D Trajectory Evaluation for Planar Robots.

For planar robots like the M3DGR UGV:
- Z coordinate from wheel odometry is MEANINGLESS (not measured)
- GT Z is constant (robot height ~0.85m)
- Fair comparison requires 2D (XY plane) alignment and evaluation

This tool:
1. Projects trajectories to XY plane
2. Computes 2D alignment (SE(2): rotation + translation in plane)
3. Evaluates 2D ATE and RPE
4. Generates comparison plots
"""

import argparse
import sqlite3
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tools"))

from rosbag_sqlite_utils import resolve_db3_path, topic_id


def load_trajectory_from_bag(cur, topic: str, msg_type: str, max_poses: int = 50000) -> Optional[np.ndarray]:
    """Load trajectory from bag. Returns Nx8 array [t, x, y, z, qx, qy, qz, qw]."""
    from rclpy.serialization import deserialize_message

    tid = topic_id(cur, topic)
    if tid is None:
        return None

    if msg_type == "Odometry":
        from nav_msgs.msg import Odometry as MsgType
        get_pose = lambda msg: (msg.pose.pose.position, msg.pose.pose.orientation)
    elif msg_type == "PoseStamped":
        from geometry_msgs.msg import PoseStamped as MsgType
        get_pose = lambda msg: (msg.pose.position, msg.pose.orientation)
    else:
        return None

    poses = []
    for row in cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
        (tid, max_poses),
    ):
        ts_ns, data = row
        try:
            msg = deserialize_message(data, MsgType)
            t = ts_ns / 1e9
            pos, ori = get_pose(msg)
            poses.append([t, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        except Exception:
            continue

    return np.array(poses) if poses else None


def load_trajectory_from_tum(filepath: str) -> Optional[np.ndarray]:
    """Load trajectory from TUM format file."""
    if not os.path.exists(filepath):
        return None

    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                poses.append([float(p) for p in parts[:8]])

    return np.array(poses) if poses else None


def extract_yaw_from_quaternion(quat: np.ndarray) -> float:
    """Extract yaw angle from quaternion [qx, qy, qz, qw]."""
    r = R.from_quat(quat)
    # Get euler angles (assuming Z-up convention)
    euler = r.as_euler('xyz')  # [roll, pitch, yaw]
    return euler[2]


def compute_se2_alignment(est_xy: np.ndarray, gt_xy: np.ndarray,
                          est_yaw: np.ndarray, gt_yaw: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute SE(2) alignment: rotation + translation in XY plane.

    Uses first pose alignment.

    Returns: (theta_align, t_align) such that:
        p_aligned = R(theta) @ p_est + t
        yaw_aligned = yaw_est + theta
    """
    # Align first poses
    theta_align = gt_yaw[0] - est_yaw[0]

    # Rotation matrix for 2D
    c, s = np.cos(theta_align), np.sin(theta_align)
    R_align = np.array([[c, -s], [s, c]])

    # Translation
    t_align = gt_xy[0] - R_align @ est_xy[0]

    return theta_align, t_align


def apply_se2_alignment(est_xy: np.ndarray, est_yaw: np.ndarray,
                        theta: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SE(2) alignment to trajectory."""
    c, s = np.cos(theta), np.sin(theta)
    R_align = np.array([[c, -s], [s, c]])

    xy_aligned = (R_align @ est_xy.T).T + t
    yaw_aligned = est_yaw + theta

    return xy_aligned, yaw_aligned


def compute_ate_2d(est_xy: np.ndarray, gt_xy: np.ndarray,
                   est_times: np.ndarray, gt_times: np.ndarray,
                   max_dt: float = 0.1) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute 2D Absolute Trajectory Error.

    Returns: (rmse, errors, matched_indices)
    """
    errors = []
    matched_idx = []

    for i in range(len(est_times)):
        # Find closest GT by time
        dt = np.abs(gt_times - est_times[i])
        j = np.argmin(dt)

        if dt[j] < max_dt:
            error = np.linalg.norm(est_xy[i] - gt_xy[j])
            errors.append(error)
            matched_idx.append((i, j))

    errors = np.array(errors)
    if len(errors) == 0:
        return float('inf'), np.array([]), []

    rmse = np.sqrt(np.mean(errors**2))
    return rmse, errors, matched_idx


def compute_rpe_2d(est_xy: np.ndarray, gt_xy: np.ndarray,
                   est_times: np.ndarray, gt_times: np.ndarray,
                   delta_m: float = 1.0, max_dt: float = 0.1) -> Tuple[float, np.ndarray]:
    """
    Compute 2D Relative Pose Error for segments of given length.

    Returns: (rmse, errors)
    """
    # First match timestamps
    matched = []
    for i in range(len(est_times)):
        dt = np.abs(gt_times - est_times[i])
        j = np.argmin(dt)
        if dt[j] < max_dt:
            matched.append((i, j))

    if len(matched) < 2:
        return float('inf'), np.array([])

    # Compute segment errors
    errors = []

    for k in range(len(matched) - 1):
        i1, j1 = matched[k]
        # Find segment endpoint at ~delta_m distance
        dist = 0
        for l in range(k + 1, len(matched)):
            i2, j2 = matched[l]
            dist = np.linalg.norm(gt_xy[j2] - gt_xy[j1])
            if dist >= delta_m * 0.9:  # Allow some tolerance
                # Compute relative motion
                est_delta = est_xy[i2] - est_xy[i1]
                gt_delta = gt_xy[j2] - gt_xy[j1]

                # Error is difference in relative motion
                error = np.linalg.norm(est_delta - gt_delta)
                errors.append(error / max(dist, 0.1))  # Normalize by distance
                break

    errors = np.array(errors)
    if len(errors) == 0:
        return float('inf'), np.array([])

    rmse = np.sqrt(np.mean(errors**2))
    return rmse, errors


def plot_trajectory_comparison(est_xy: np.ndarray, gt_xy: np.ndarray,
                               est_xy_aligned: np.ndarray,
                               errors: np.ndarray, matched_idx: List,
                               output_path: str):
    """Generate trajectory comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Raw trajectories
    ax = axes[0, 0]
    ax.plot(est_xy[:, 0], est_xy[:, 1], 'b-', linewidth=1, label='Estimate (raw)', alpha=0.7)
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'g-', linewidth=1, label='Ground Truth', alpha=0.7)
    ax.plot(est_xy[0, 0], est_xy[0, 1], 'bo', markersize=10, label='Est start')
    ax.plot(gt_xy[0, 0], gt_xy[0, 1], 'go', markersize=10, label='GT start')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Raw Trajectories (before alignment)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # 2. Aligned trajectories
    ax = axes[0, 1]
    ax.plot(est_xy_aligned[:, 0], est_xy_aligned[:, 1], 'b-', linewidth=1, label='Estimate (aligned)', alpha=0.7)
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'g-', linewidth=1, label='Ground Truth', alpha=0.7)
    ax.plot(est_xy_aligned[0, 0], est_xy_aligned[0, 1], 'bo', markersize=10)
    ax.plot(gt_xy[0, 0], gt_xy[0, 1], 'go', markersize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Aligned Trajectories (SE(2) first-pose alignment)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # 3. Error over trajectory
    ax = axes[1, 0]
    if len(errors) > 0:
        ax.plot(errors, 'r-', linewidth=1)
        ax.axhline(np.mean(errors), color='k', linestyle='--', label=f'Mean: {np.mean(errors):.3f}m')
        ax.fill_between(range(len(errors)), 0, errors, alpha=0.3, color='red')
    ax.set_xlabel('Matched pose index')
    ax.set_ylabel('Position error (m)')
    ax.set_title('2D Position Error Over Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Error histogram
    ax = axes[1, 1]
    if len(errors) > 0:
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}m')
        ax.axvline(np.sqrt(np.mean(errors**2)), color='g', linestyle='--', label=f'RMSE: {np.sqrt(np.mean(errors**2)):.3f}m')
    ax.set_xlabel('Position error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved plot to: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="2D trajectory evaluation for planar robots")
    ap.add_argument("--bag", help="Bag directory containing *.db3")
    ap.add_argument("--est-tum", help="Estimated trajectory in TUM format")
    ap.add_argument("--gt-tum", help="Ground truth trajectory in TUM format")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--gt-topic", default="/vrpn_client_node/UGV/pose")
    ap.add_argument("--output-dir", default=".", help="Output directory for results")
    ap.add_argument("--rpe-delta", type=float, default=1.0, help="RPE segment length (m)")
    args = ap.parse_args()

    # Load trajectories
    est_traj = None
    gt_traj = None

    if args.bag:
        db_path = resolve_db3_path(args.bag)
        if not db_path:
            print(f"ERROR: Could not locate *.db3 under '{args.bag}'", file=sys.stderr)
            return 1

        import rclpy
        rclpy.init()

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        print("Loading trajectories from bag...")
        est_traj = load_trajectory_from_bag(cur, args.odom_topic, "Odometry")
        gt_traj = load_trajectory_from_bag(cur, args.gt_topic, "PoseStamped")

        conn.close()
        rclpy.shutdown()

    if args.est_tum:
        print(f"Loading estimate from: {args.est_tum}")
        est_traj = load_trajectory_from_tum(args.est_tum)

    if args.gt_tum:
        print(f"Loading GT from: {args.gt_tum}")
        gt_traj = load_trajectory_from_tum(args.gt_tum)

    if est_traj is None:
        print("ERROR: Could not load estimated trajectory")
        return 1
    if gt_traj is None:
        print("ERROR: Could not load ground truth trajectory")
        return 1

    print(f"  Estimate: {len(est_traj)} poses")
    print(f"  GT:       {len(gt_traj)} poses")
    print()

    # Extract 2D data
    est_times = est_traj[:, 0]
    est_xy = est_traj[:, 1:3]  # X, Y only
    est_yaw = np.array([extract_yaw_from_quaternion(est_traj[i, 4:8]) for i in range(len(est_traj))])

    gt_times = gt_traj[:, 0]
    gt_xy = gt_traj[:, 1:3]
    gt_yaw = np.array([extract_yaw_from_quaternion(gt_traj[i, 4:8]) for i in range(len(gt_traj))])

    # Align timestamps (make both start at t=0 relative to their first pose)
    est_times = est_times - est_times[0]
    gt_times = gt_times - gt_times[0]

    print("=" * 80)
    print("2D TRAJECTORY EVALUATION (Planar Robot)")
    print("=" * 80)
    print()

    # Statistics before alignment
    print("RAW STATISTICS:")
    print(f"  Estimate XY extent: {np.ptp(est_xy[:, 0]):.3f}m x {np.ptp(est_xy[:, 1]):.3f}m")
    print(f"  GT XY extent:       {np.ptp(gt_xy[:, 0]):.3f}m x {np.ptp(gt_xy[:, 1]):.3f}m")
    print(f"  Estimate Z extent:  {np.ptp(est_traj[:, 3]):.3f}m (should be ~0 for planar)")
    print(f"  GT Z extent:        {np.ptp(gt_traj[:, 3]):.3f}m")
    print()

    # SE(2) Alignment
    theta_align, t_align = compute_se2_alignment(est_xy, gt_xy, est_yaw, gt_yaw)
    est_xy_aligned, est_yaw_aligned = apply_se2_alignment(est_xy, est_yaw, theta_align, t_align)

    print("SE(2) ALIGNMENT (first-pose matching):")
    print(f"  Rotation: {np.degrees(theta_align):.2f}°")
    print(f"  Translation: [{t_align[0]:.4f}, {t_align[1]:.4f}]")
    print()

    # Compute 2D ATE
    ate_rmse, ate_errors, matched_idx = compute_ate_2d(est_xy_aligned, gt_xy, est_times, gt_times)

    print("2D ABSOLUTE TRAJECTORY ERROR (ATE):")
    print(f"  RMSE:   {ate_rmse:.4f} m")
    if len(ate_errors) > 0:
        print(f"  Mean:   {np.mean(ate_errors):.4f} m")
        print(f"  Median: {np.median(ate_errors):.4f} m")
        print(f"  Max:    {np.max(ate_errors):.4f} m")
        print(f"  Matched poses: {len(ate_errors)}")
    print()

    # Compute 2D RPE
    rpe_rmse, rpe_errors = compute_rpe_2d(est_xy_aligned, gt_xy, est_times, gt_times, args.rpe_delta)

    print(f"2D RELATIVE POSE ERROR (RPE @ {args.rpe_delta}m):")
    print(f"  RMSE:   {rpe_rmse:.4f} m/m ({rpe_rmse*100:.2f}%)")
    if len(rpe_errors) > 0:
        print(f"  Mean:   {np.mean(rpe_errors):.4f} m/m")
        print(f"  Segments evaluated: {len(rpe_errors)}")
    print()

    # Generate plot
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "trajectory_2d_comparison.png")
    plot_trajectory_comparison(est_xy, gt_xy, est_xy_aligned, ate_errors, matched_idx, plot_path)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics_2d.txt")
    with open(metrics_path, 'w') as f:
        f.write("2D Trajectory Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Estimate poses: {len(est_traj)}\n")
        f.write(f"GT poses: {len(gt_traj)}\n")
        f.write(f"Matched poses: {len(ate_errors)}\n\n")
        f.write("2D ATE (Absolute Trajectory Error):\n")
        f.write(f"  RMSE:   {ate_rmse:.4f} m\n")
        if len(ate_errors) > 0:
            f.write(f"  Mean:   {np.mean(ate_errors):.4f} m\n")
            f.write(f"  Median: {np.median(ate_errors):.4f} m\n")
            f.write(f"  Max:    {np.max(ate_errors):.4f} m\n")
        f.write(f"\n2D RPE (Relative Pose Error @ {args.rpe_delta}m):\n")
        f.write(f"  RMSE:   {rpe_rmse:.4f} m/m ({rpe_rmse*100:.2f}%)\n")
    print(f"  Saved metrics to: {metrics_path}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  2D ATE RMSE: {ate_rmse:.4f} m")
    print(f"  2D RPE RMSE: {rpe_rmse:.4f} m/m ({rpe_rmse*100:.2f}%)")

    # Interpret results
    if ate_rmse < 0.5:
        print("\n  ✓ Good accuracy (<0.5m ATE)")
    elif ate_rmse < 1.0:
        print("\n  ~ Moderate accuracy (0.5-1.0m ATE)")
    else:
        print(f"\n  ✗ Poor accuracy (>{ate_rmse:.1f}m ATE)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
