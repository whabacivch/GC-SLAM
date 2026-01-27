#!/usr/bin/env python3
"""
Diagnostic tool to analyze the frame offset between GT and SLAM trajectories.

This helps identify:
1. What rotation transform converts SLAM orientation to GT orientation
2. Whether the offset is constant (frame mismatch) or varying (SLAM error)
3. Specific axis/angle to understand the convention difference
"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation

def load_tum(path: str, max_lines: int = 100):
    """Load TUM trajectory: timestamp x y z qx qy qz qw"""
    poses = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                t = float(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                poses.append({
                    'time': t,
                    'pos': np.array([x, y, z]),
                    'quat': np.array([qx, qy, qz, qw])  # xyzw
                })
            if len(poses) >= max_lines:
                break
    return poses


def find_closest_gt(gt_poses, t_est, max_dt=0.1):
    """Find GT pose closest to timestamp t_est."""
    best_idx = None
    best_dt = float('inf')
    for i, p in enumerate(gt_poses):
        dt = abs(p['time'] - t_est)
        if dt < best_dt:
            best_dt = dt
            best_idx = i
    if best_dt > max_dt:
        return None
    return gt_poses[best_idx]


def analyze_rotation_offset(gt_poses, est_poses, match_count=50, use_time_sync=True):
    """Analyze rotation offset between GT and estimated poses."""

    if use_time_sync:
        # Match by timestamp
        matched_pairs = []
        for est in est_poses:
            gt_match = find_closest_gt(gt_poses, est['time'])
            if gt_match is not None:
                matched_pairs.append((gt_match, est))
            if len(matched_pairs) >= match_count:
                break
        n = len(matched_pairs)
        print(f"\n=== Analyzing {n} time-synced pose pairs ===\n")
    else:
        # Match by index (assumes time-aligned or using aligned files)
        n = min(len(gt_poses), len(est_poses), match_count)
        matched_pairs = [(gt_poses[i], est_poses[i]) for i in range(n)]
        print(f"\n=== Analyzing first {n} pose pairs (by index) ===\n")

    rel_rots = []
    angles = []
    axes = []

    for i, (gt, est) in enumerate(matched_pairs):

        # Get rotation matrices
        R_gt = Rotation.from_quat(gt['quat']).as_matrix()
        R_est = Rotation.from_quat(est['quat']).as_matrix()

        # Relative rotation: R_rel = R_est^T @ R_gt
        # This is the rotation that transforms EST orientation to GT orientation
        R_rel = R_est.T @ R_gt

        rot = Rotation.from_matrix(R_rel)
        angle_deg = rot.magnitude() * 180.0 / np.pi
        rotvec = rot.as_rotvec()
        axis = rotvec / (np.linalg.norm(rotvec) + 1e-12)

        rel_rots.append(rot)
        angles.append(angle_deg)
        axes.append(axis)

        if i < 5:  # Print first few
            print(f"Pose {i}:")
            print(f"  GT pos:  ({gt['pos'][0]:.3f}, {gt['pos'][1]:.3f}, {gt['pos'][2]:.3f})")
            print(f"  EST pos: ({est['pos'][0]:.3f}, {est['pos'][1]:.3f}, {est['pos'][2]:.3f})")
            print(f"  GT quat:  ({gt['quat'][0]:.4f}, {gt['quat'][1]:.4f}, {gt['quat'][2]:.4f}, {gt['quat'][3]:.4f})")
            print(f"  EST quat: ({est['quat'][0]:.4f}, {est['quat'][1]:.4f}, {est['quat'][2]:.4f}, {est['quat'][3]:.4f})")
            print(f"  Rel angle: {angle_deg:.2f}°")
            print(f"  Rel axis:  ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
            print()

    angles = np.array(angles)
    axes = np.stack(axes)

    print("=== Statistics ===")
    print(f"Angle: mean={np.mean(angles):.2f}°, std={np.std(angles):.2f}°, min={np.min(angles):.2f}°, max={np.max(angles):.2f}°")
    print(f"Axis mean: ({np.mean(axes[:,0]):.3f}, {np.mean(axes[:,1]):.3f}, {np.mean(axes[:,2]):.3f})")
    print(f"Axis std:  ({np.std(axes[:,0]):.3f}, {np.std(axes[:,1]):.3f}, {np.std(axes[:,2]):.3f})")

    # Compute average rotation using Markley quaternion average
    quats = np.array([r.as_quat() for r in rel_rots])
    A = np.zeros((4, 4))
    for q in quats:
        q = q / (np.linalg.norm(q) + 1e-12)
        A += np.outer(q, q)
    A /= len(quats)
    vals, vecs = np.linalg.eigh(A)
    q_avg = vecs[:, np.argmax(vals)]
    rot_avg = Rotation.from_quat(q_avg)

    avg_angle = rot_avg.magnitude() * 180.0 / np.pi
    avg_rotvec = rot_avg.as_rotvec()
    avg_axis = avg_rotvec / (np.linalg.norm(avg_rotvec) + 1e-12)
    avg_euler = rot_avg.as_euler('xyz', degrees=True)

    print(f"\n=== Average Relative Rotation (EST→GT) ===")
    print(f"Angle: {avg_angle:.2f}°")
    print(f"Axis:  ({avg_axis[0]:.4f}, {avg_axis[1]:.4f}, {avg_axis[2]:.4f})")
    print(f"Rotvec: ({avg_rotvec[0]:.6f}, {avg_rotvec[1]:.6f}, {avg_rotvec[2]:.6f})")
    print(f"Euler (XYZ, deg): roll={avg_euler[0]:.2f}°, pitch={avg_euler[1]:.2f}°, yaw={avg_euler[2]:.2f}°")

    # Decompose axis into cardinal components
    print(f"\n=== Axis Interpretation ===")
    print(f"X component: {avg_axis[0]:.3f} ({abs(avg_axis[0])*100:.1f}% of unit)")
    print(f"Y component: {avg_axis[1]:.3f} ({abs(avg_axis[1])*100:.1f}% of unit)")
    print(f"Z component: {avg_axis[2]:.3f} ({abs(avg_axis[2])*100:.1f}% of unit)")

    # Check if close to cardinal axes
    cardinal_names = ['X', '-X', 'Y', '-Y', 'Z', '-Z']
    cardinal_vecs = [
        np.array([1, 0, 0]), np.array([-1, 0, 0]),
        np.array([0, 1, 0]), np.array([0, -1, 0]),
        np.array([0, 0, 1]), np.array([0, 0, -1])
    ]

    print(f"\n=== Closest Cardinal Axis ===")
    for name, vec in zip(cardinal_names, cardinal_vecs):
        dot = np.dot(avg_axis, vec)
        angle_to_cardinal = np.arccos(np.clip(abs(dot), 0, 1)) * 180 / np.pi
        print(f"  {name}: dot={dot:.3f}, angle_from_axis={angle_to_cardinal:.1f}°")

    # Suggest correction
    print(f"\n=== Suggested Correction ===")
    print(f"To correct EST poses to match GT frame, apply:")
    print(f"  R_correction @ R_est")
    print(f"  where R_correction.as_rotvec() = {avg_rotvec.tolist()}")

    # Check if this looks like a Z-flip (180° about X)
    if avg_angle > 160:
        z_flip_axis = np.array([1, 0, 0])  # π about X
        angle_from_z_flip = np.arccos(np.clip(np.dot(avg_axis, z_flip_axis), -1, 1)) * 180 / np.pi
        angle_from_neg_z_flip = np.arccos(np.clip(np.dot(avg_axis, -z_flip_axis), -1, 1)) * 180 / np.pi

        print(f"\n=== Z-flip Analysis ===")
        print(f"If avg rotation were π about +X: axis would be [1,0,0]")
        print(f"  Actual axis is {angle_from_z_flip:.1f}° away from +X")
        print(f"  Actual axis is {angle_from_neg_z_flip:.1f}° away from -X")

        if angle_from_z_flip < 30 or angle_from_neg_z_flip < 30:
            print(f"  → This looks like a Z-flip (Z-up ↔ Z-down convention mismatch)")
            print(f"  → But it's not pure X-axis rotation, suggesting additional misalignment")

    return {
        'avg_angle': avg_angle,
        'avg_axis': avg_axis,
        'avg_rotvec': avg_rotvec,
        'avg_euler': avg_euler,
    }


def check_gravity_direction(poses, label=""):
    """Check the average 'up' direction in the trajectory poses."""
    # In a properly calibrated system at rest, the gravity direction (Z in base frame)
    # should be consistent across poses
    z_dirs = []
    for p in poses[:50]:
        R = Rotation.from_quat(p['quat']).as_matrix()
        z_world = R @ np.array([0, 0, 1])  # Where does local Z point in world?
        z_dirs.append(z_world)

    z_dirs = np.stack(z_dirs)
    mean_z = np.mean(z_dirs, axis=0)
    mean_z = mean_z / (np.linalg.norm(mean_z) + 1e-12)

    print(f"\n=== {label} Gravity Check ===")
    print(f"Mean 'up' direction in world frame: ({mean_z[0]:.3f}, {mean_z[1]:.3f}, {mean_z[2]:.3f})")
    print(f"  If Z-up world: expect (0, 0, 1)")
    print(f"  Dot with +Z: {np.dot(mean_z, [0,0,1]):.3f}")
    print(f"  Dot with -Z: {np.dot(mean_z, [0,0,-1]):.3f}")


def main():
    if len(sys.argv) < 3:
        print("Usage: diagnose_frame_offset.py <gt_tum> <est_tum>")
        print("\nExample:")
        print("  .venv/bin/python tools/diagnose_frame_offset.py results/gc_20260126_173510/ground_truth_aligned.tum results/gc_20260126_173510/estimated_trajectory.tum")
        return 1

    gt_path = sys.argv[1]
    est_path = sys.argv[2]

    print(f"Loading GT: {gt_path}")
    print(f"Loading EST: {est_path}")

    gt_poses = load_tum(gt_path, max_lines=200)
    est_poses = load_tum(est_path, max_lines=200)

    print(f"Loaded {len(gt_poses)} GT poses, {len(est_poses)} EST poses")

    # Analyze rotation offset
    result = analyze_rotation_offset(gt_poses, est_poses)

    # Check gravity directions
    check_gravity_direction(gt_poses, "GT")
    check_gravity_direction(est_poses, "EST")

    return 0


if __name__ == "__main__":
    sys.exit(main())
