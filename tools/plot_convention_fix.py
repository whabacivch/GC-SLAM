#!/usr/bin/env python3
"""
Plot trajectory comparison with correct convention fix (no GT scrambling).

Rule: Apply R_EST_TO_GT to EST positions only; do not apply to GT; then shift
both by the same origin (gt[0]). No evo alignment — minimal flow so GT axes
stay intact (GT z ptp should be ~0.04 m for planar robot).

  GT x ≈ −EST z,  GT y ≈ EST x,  GT z ≈ −EST y  (R_EST_TO_GT on EST only)

TUM format: t x y z qx qy qz qw  → we use columns 1,2,3 as [x,y,z] only.

Usage:
  .venv/bin/python tools/plot_convention_fix.py [results_dir]
  # Output: results_dir/trajectory_comparison_convention_fix.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# R_EST_TO_GT: GT x ≈ −EST z, GT y ≈ EST x, GT z ≈ −EST y (apply to EST only)
R_EST_TO_GT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
], dtype=np.float64)


def load_tum_xyz(file_path: Path) -> np.ndarray:
    """Load TUM file; return (N, 3) positions [x, y, z] from columns 1,2,3 only."""
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                # TUM: t x y z qx qy qz qw  -> cols 1,2,3 = x,y,z
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                rows.append([x, y, z])
    return np.array(rows, dtype=np.float64)


def ptp(v: np.ndarray) -> np.ndarray:
    """Peak-to-peak (max - min) per column."""
    return np.array([v[:, i].max() - v[:, i].min() for i in range(v.shape[1])], dtype=np.float64)


def main() -> int:
    default_results = PROJECT_ROOT / "results" / "gc_20260129_112502"
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_results
    gt_path = results_dir / "ground_truth_aligned.tum"
    est_path = results_dir / "estimated_trajectory.tum"
    out_path = results_dir / "trajectory_comparison_convention_fix.png"

    if not gt_path.exists():
        print(f"ERROR: Ground truth not found: {gt_path}", file=sys.stderr)
        return 1
    if not est_path.exists():
        print(f"ERROR: Estimated trajectory not found: {est_path}", file=sys.stderr)
        return 1

    # Load [x,y,z] only (columns 1,2,3) — never touch quat
    gt = load_tum_xyz(gt_path)
    est = load_tum_xyz(est_path)

    # Apply R_EST_TO_GT to EST only; GT stays as-is
    est_gt = (R_EST_TO_GT @ est.T).T

    # Same origin for both: shift by first GT position
    gt0 = gt[0].copy()
    gt_plot = gt - gt0
    est_plot = est_gt - gt0

    # Sanity checks before plotting
    print("GT  ptp xyz (m):", ptp(gt))
    print("EST ptp xyz (m):", ptp(est))
    print("EST_gt ptp xyz (m):", ptp(est_gt))
    print("GT z ptp (m):", ptp(gt)[2], "  (expected ~0.04 for planar robot)")
    print("First row GT [x,y,z]:", gt[0])
    if ptp(gt)[2] > 0.5:
        print("WARNING: GT z ptp is large — plotted GT 'Z' may not be GT z column.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 12))
    # XY
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(gt_plot[:, 0], gt_plot[:, 1], "b-", label="Ground Truth", linewidth=1.5, alpha=0.8)
    ax1.plot(est_plot[:, 0], est_plot[:, 1], "r--", label="Estimated (R_EST_TO_GT, EST only)", linewidth=1.5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("XY (Top View)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    # XZ
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_plot[:, 0], gt_plot[:, 2], "b-", label="Ground Truth", linewidth=1.5, alpha=0.8)
    ax2.plot(est_plot[:, 0], est_plot[:, 2], "r--", label="Estimated (R_EST_TO_GT, EST only)", linewidth=1.5)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("XZ (Side View)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # YZ
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(gt_plot[:, 1], gt_plot[:, 2], "b-", label="Ground Truth", linewidth=1.5, alpha=0.8)
    ax3.plot(est_plot[:, 1], est_plot[:, 2], "r--", label="Estimated (R_EST_TO_GT, EST only)", linewidth=1.5)
    ax3.set_xlabel("Y (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("YZ (Front View)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # 3D or placeholder
    try:
        ax4 = fig.add_subplot(2, 2, 4, projection="3d")
        ax4.plot(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], "b-", label="Ground Truth", linewidth=1.5, alpha=0.8)
        ax4.plot(est_plot[:, 0], est_plot[:, 1], est_plot[:, 2], "r--", label="Estimated (R_EST_TO_GT, EST only)", linewidth=1.5)
        ax4.set_xlabel("X (m)")
        ax4.set_ylabel("Y (m)")
        ax4.set_zlabel("Z (m)")
    except Exception:
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.text(0.5, 0.5, "3D unavailable", ha="center", va="center", transform=ax4.transAxes)
    fig.suptitle("Convention fix: R_EST_TO_GT on EST only; both shifted by gt[0]; GT unchanged", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
