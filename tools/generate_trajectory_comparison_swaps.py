#!/usr/bin/env python3
"""
Generate multiple trajectory_comparison PNGs with various GT axis permutations.

Uses the last run's ground_truth_aligned.tum and estimated_trajectory.tum.
Permutes GT axes (position + rotation) then aligns GT to estimate at first pose
and plots. Use to test which axis convention matches (e.g. "our Z = GT X").

Usage:
  .venv/bin/python tools/generate_trajectory_comparison_swaps.py [results_dir]
  # Default results_dir: results/gc_20260129_112502
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evo.core import sync, trajectory

# Reuse evaluate_slam helpers without full validation
from evo.tools import file_interface


def load_trajectory(file_path: Path):
    """Load trajectory from TUM format."""
    return file_interface.read_tum_trajectory_file(str(file_path))


def permute_gt_axes(
    traj: trajectory.PoseTrajectory3D,
    perm: tuple[int, int, int],
) -> trajectory.PoseTrajectory3D:
    """
    Return a new trajectory with GT axes permuted: new_xyz[i] = old_xyz[perm[i]].

    perm: (ix, iy, iz) indices 0,1,2. E.g. (2,1,0) => new_x=old_z, new_y=old_y, new_z=old_x
          ("our Z = GT X" => use (2,1,0) so GT's X becomes our Z axis in the plot).
    """
    ix, iy, iz = perm
    new_poses = []
    for pose in traj.poses_se3:
        R = np.array(pose[:3, :3], dtype=np.float64)
        t = np.array(pose[:3, 3], dtype=np.float64)
        R_new = R[:, [ix, iy, iz]]
        t_new = np.array([t[ix], t[iy], t[iz]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_new
        T[:3, 3] = t_new
        new_poses.append(T)
    return trajectory.PoseTrajectory3D(
        poses_se3=new_poses,
        timestamps=np.array(traj.timestamps, dtype=np.float64),
        meta=dict(traj.meta) if traj.meta else None,
    )


def align_trajectories_initial(
    gt_sync: trajectory.PoseTrajectory3D,
    est_sync: trajectory.PoseTrajectory3D,
) -> None:
    """Put GT in estimate frame (first pose match). Modifies gt_sync in place."""
    if len(gt_sync.poses_se3) == 0 or len(est_sync.poses_se3) == 0:
        return
    T_est0 = np.array(est_sync.poses_se3[0], dtype=np.float64)
    T_gt0_inv = np.linalg.inv(gt_sync.poses_se3[0])
    T_gt_to_est = T_est0 @ T_gt0_inv
    gt_sync.transform(T_gt_to_est)


def main() -> int:
    default_results = PROJECT_ROOT / "results" / "gc_20260129_112502"
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_results
    gt_path = results_dir / "ground_truth_aligned.tum"
    est_path = results_dir / "estimated_trajectory.tum"
    out_dir = results_dir / "trajectory_swap_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gt_path.exists():
        print(f"ERROR: Ground truth not found: {gt_path}", file=sys.stderr)
        return 1
    if not est_path.exists():
        print(f"ERROR: Estimated trajectory not found: {est_path}", file=sys.stderr)
        return 1

    print("Loading trajectories...")
    gt_traj = load_trajectory(gt_path)
    est_traj = load_trajectory(est_path)

    # Permutations: (name, (ix, iy, iz)) so new_x = old[ix], new_y = old[iy], new_z = old[iz]
    # "our Z = GT X" => we want GT's X to appear on our Z axis => perm (2,1,0): new_x=old_z, new_y=old_y, new_z=old_x
    # So in the plot, "Z" will be GT's original X. So we're relabeling GT: GT_x -> our Z.
    permutations = [
        ("identity", (0, 1, 2), "GT (x,y,z) unchanged"),
        ("swap_yz", (0, 2, 1), "GT: new_y=z, new_z=y"),
        ("swap_xz", (2, 1, 0), "GT: new_x=z, new_z=x (our Z = GT X)"),
        ("swap_xy", (1, 0, 2), "GT: new_x=y, new_y=x"),
        ("swap_xz_yx", (2, 0, 1), "GT: new_x=z, new_y=x, new_z=y"),
        ("swap_xy_zx", (1, 2, 0), "GT: new_x=y, new_y=z, new_z=x"),
    ]

    # Lazy import so matplotlib doesn't need to be set before we check paths
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot_trajectories(gt_traj, est_traj, output_path, title_suffix=""):
        fig = plt.figure(figsize=(14, 12))
        gt_xyz = gt_traj.positions_xyz
        est_xyz = est_traj.positions_xyz
        # XY
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(gt_xyz[:, 0], gt_xyz[:, 1], "b-", label="Ground Truth", linewidth=2)
        ax1.plot(est_xyz[:, 0], est_xyz[:, 1], "r--", label="Estimated", linewidth=1.5)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title("XY (Top View)" + title_suffix)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis("equal")
        # XZ
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(gt_xyz[:, 0], gt_xyz[:, 2], "b-", label="Ground Truth", linewidth=2)
        ax2.plot(est_xyz[:, 0], est_xyz[:, 2], "r--", label="Estimated", linewidth=1.5)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Z (m)")
        ax2.set_title("XZ (Side View)" + title_suffix)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # YZ
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(gt_xyz[:, 1], gt_xyz[:, 2], "b-", label="Ground Truth", linewidth=2)
        ax3.plot(est_xyz[:, 1], est_xyz[:, 2], "r--", label="Estimated", linewidth=1.5)
        ax3.set_xlabel("Y (m)")
        ax3.set_ylabel("Z (m)")
        ax3.set_title("YZ (Front View)" + title_suffix)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        # 3D or placeholder
        try:
            ax4 = fig.add_subplot(2, 2, 4, projection="3d")
            ax4.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], "b-", label="Ground Truth", linewidth=2)
            ax4.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], "r--", label="Estimated", linewidth=1.5)
            ax4.set_xlabel("X (m)")
            ax4.set_ylabel("Y (m)")
            ax4.set_zlabel("Z (m)")
        except Exception:
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.text(
                0.5, 0.5, "3D unavailable\n(use XY/XZ/YZ views)",
                ha="center", va="center", transform=ax4.transAxes,
            )
        if title_suffix:
            fig.suptitle("Trajectory comparison — " + title_suffix.strip(), fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

    for name, perm, desc in permutations:
        print(f"  {name}: {desc}")
        gt_perm = permute_gt_axes(gt_traj, perm)
        gt_copy = __import__("copy").deepcopy(gt_perm)
        est_copy = __import__("copy").deepcopy(est_traj)
        gt_sync, est_sync = sync.associate_trajectories(gt_copy, est_copy)
        align_trajectories_initial(gt_sync, est_sync)
        out_path = out_dir / f"trajectory_comparison_{name}.png"
        plot_trajectories(gt_sync, est_sync, out_path, title_suffix=f" GT perm: {name}")
        print(f"    -> {out_path}")

    print(f"\nDone. All comparison images saved under: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
