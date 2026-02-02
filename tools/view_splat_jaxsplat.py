#!/usr/bin/env python3
"""
Post-run JAXsplat visualization: load splat_export.npz and render view(s).

Renders the primitive map (splats) from a GC run using jaxsplat.render_v2,
using the last trajectory pose as the camera. Saves an image and optionally
opens it.

Usage:
  python tools/view_splat_jaxsplat.py results/gc_YYYYMMDD_HHMMSS/splat_export.npz [--trajectory path.tum] [--output splat_render.png]
  If --trajectory is omitted, looks for estimated_trajectory.tum in the same dir as the npz.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _covariance_to_packed_6(Sigma: np.ndarray) -> np.ndarray:
    """Pack (N, 3, 3) covariance into (N, 6) lower-triangular row-major [00, 10, 11, 20, 21, 22]."""
    N = Sigma.shape[0]
    out = np.empty((N, 6), dtype=np.float32)
    out[:, 0] = Sigma[:, 0, 0]
    out[:, 1] = Sigma[:, 1, 0]
    out[:, 2] = Sigma[:, 1, 1]
    out[:, 3] = Sigma[:, 2, 0]
    out[:, 4] = Sigma[:, 2, 1]
    out[:, 5] = Sigma[:, 2, 2]
    return out


def _tum_last_pose_as_4x4(tum_path: str) -> np.ndarray:
    """Read last pose from TUM file; return 4x4 world-from-body (or body pose in world)."""
    poses = []
    with open(tum_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                from scipy.spatial.transform import Rotation
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R
                T[:3, 3] = [x, y, z]
                poses.append(T)
    if not poses:
        return np.eye(4, dtype=np.float32)
    return poses[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Render GC splat export with JAXsplat")
    ap.add_argument("splat_npz", help="Path to splat_export.npz from a GC run")
    ap.add_argument("--trajectory", default=None, help="TUM trajectory (default: same dir as npz / estimated_trajectory.tum)")
    ap.add_argument("--output", default=None, help="Output image path (default: same dir as npz / splat_render.png)")
    ap.add_argument("--open-image", dest="open_image", action="store_true", help="Open the image after saving")
    ap.add_argument("--height", type=int, default=480, help="Render height")
    ap.add_argument("--width", type=int, default=640, help="Render width")
    args = ap.parse_args()

    npz_path = Path(args.splat_npz)
    if not npz_path.is_file():
        print(f"ERROR: Not found: {npz_path}", file=sys.stderr)
        return 1

    data = np.load(npz_path, allow_pickle=True)
    n = int(data.get("n", data["positions"].shape[0]))
    positions = np.asarray(data["positions"], dtype=np.float32)[:n]
    covariances = np.asarray(data["covariances"], dtype=np.float32)[:n]
    colors = np.asarray(data["colors"], dtype=np.float32)[:n]
    weights = np.asarray(data["weights"], dtype=np.float32)[:n]
    directions = np.asarray(data["directions"], dtype=np.float32)[:n]
    kappas = np.asarray(data["kappas"], dtype=np.float32)[:n]

    if n == 0:
        print("No primitives in export; nothing to render.", file=sys.stderr)
        return 0

    # Trajectory: camera pose (world-from-body). T_CW = inv(T_WB) for camera-at-body.
    traj_path = args.trajectory
    if traj_path is None:
        traj_path = npz_path.parent / "estimated_trajectory.tum"
    if not os.path.isfile(traj_path):
        traj_path = npz_path.parent / "estimated_trajectory_wheel.tum"
    if not os.path.isfile(traj_path):
        print("WARN: No trajectory file; using identity camera.", file=sys.stderr)
        T_WB = np.eye(4, dtype=np.float32)
    else:
        T_WB = _tum_last_pose_as_4x4(str(traj_path))
    # T_CW: camera-from-world (render_v2 convention: transform world points to camera frame)
    T_CW = np.linalg.inv(T_WB).astype(np.float32)

    # Pack covariance to (N, 6)
    Sigma_packed = _covariance_to_packed_6(covariances)

    # Alpha from weights (normalize to [0, 1] range for visibility)
    w_max = float(np.max(weights)) if np.max(weights) > 0 else 1.0
    alpha_base = (weights / w_max).astype(np.float32)
    alpha_base = np.clip(alpha_base, 0.01, 1.0)

    # Eta = kappa * direction (vMF natural params)
    eta = (kappas[:, None] * directions).astype(np.float32)

    out_path = args.output
    if out_path is None:
        out_path = npz_path.parent / "splat_render.png"
    out_path = Path(out_path)

    try:
        import jax.numpy as jnp
        from jaxsplat import render_v2, RenderV2Config
    except ImportError as e:
        print(f"JAXsplat not available: {e}. Install jaxsplat and re-run.", file=sys.stderr)
        return 1

    rcfg = RenderV2Config(H=args.height, W=args.width)
    mu_w = jnp.array(positions)
    Sigma_w = jnp.array(Sigma_packed)
    color = jnp.array(colors)
    alpha = jnp.array(alpha_base)
    eta_j = jnp.array(eta)
    T_CW_j = jnp.array(T_CW)

    out = render_v2(mu_w, Sigma_w, color, alpha, eta_j, T_CW_j, rcfg)
    rgb = np.asarray(out[0])
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imshow(rgb)
        plt.axis("off")
        plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
        plt.close()
    except Exception:
        try:
            from PIL import Image
            Image.fromarray(rgb).save(str(out_path))
        except Exception as e2:
            print(f"Could not save image: {e2}", file=sys.stderr)
            return 1

    print(f"Saved: {out_path}")

    if getattr(args, "open_image", False):
        try:
            import subprocess
            subprocess.run(["xdg-open", str(out_path)], check=False, timeout=2)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
