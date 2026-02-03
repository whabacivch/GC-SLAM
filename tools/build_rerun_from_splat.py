#!/usr/bin/env python3
"""
Post-run Rerun builder: build .rrd from splat_export.npz + trajectory.

Uses Ellipsoids3D (covariance → half_sizes + quaternions), BEV 15 shaded colors
(vMF + fBm from fl_slam_poc.backend.rendering), trajectory as LineStrips3D and
optional Transform3D per pose. No live recording; run after eval finishes.

Optional BEV15 post-run view layer: creates a second .rrd with 2D BEV projections
for each of 15 oblique views. This is a view-only artifact (not used in runtime).

Usage:
  python tools/build_rerun_from_splat.py RESULTS_DIR [--output RESULTS_DIR/gc_slam.rrd]
  python tools/build_rerun_from_splat.py --splat path.npz --trajectory path.tum --output out.rrd
  python tools/build_rerun_from_splat.py RESULTS_DIR --bev15-output RESULTS_DIR/gc_bev15.rrd
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# Allow importing fl_slam_poc.backend.rendering for vMF + fBm
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FL_SLAM_POC_ROOT = PROJECT_ROOT / "fl_ws" / "src" / "fl_slam_poc"
if str(FL_SLAM_POC_ROOT) not in sys.path:
    sys.path.insert(0, str(FL_SLAM_POC_ROOT))

try:
    import rerun as rr
except ImportError:
    rr = None

# BEV 15: view direction 15° off vertical (toward +Y), for shading
BEV_15_DEG = 15.0


def _covariance_to_ellipsoid(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From (3,3) covariance Σ, return (half_sizes, quat_xyzw) for Rerun Ellipsoids3D.

    Σ = R D R^T; half_sizes = sqrt(eigenvalues) for 1-sigma ellipsoid; quat from R.
    Same convention as 3D Gaussian principal axes (jaxsplat / EWA).
    """
    from scipy.spatial.transform import Rotation

    Sigma = np.asarray(Sigma, dtype=np.float64).reshape(3, 3)
    # Ensure symmetric
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-12)
    half_sizes = np.sqrt(eigvals)
    # R columns = eigenvectors; R gives rotation from principal to world
    R = eigvecs
    # Ensure right-handed coordinate frame (det = +1)
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]  # Flip first column to fix handedness
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    return half_sizes, quat_xyzw


def _covariances_to_ellipsoids(covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3, 3) -> half_sizes (N, 3), quaternions (N, 4) xyzw."""
    N = covariances.shape[0]
    half_sizes = np.zeros((N, 3), dtype=np.float64)
    quats = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        half_sizes[i], quats[i] = _covariance_to_ellipsoid(covariances[i])
    return half_sizes, quats


def _bev15_view_direction(deg: float = BEV_15_DEG) -> np.ndarray:
    """Unit view direction: 15° off vertical (e.g. toward +Y). v = (0, sin(deg), -cos(deg))."""
    rad = math.radians(deg)
    v = np.array([0.0, math.sin(rad), -math.cos(rad)], dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _build_bev15_rrd(
    splat_npz: str,
    output_rrd: str,
    n_views: int = 15,
    phi_center_deg: float = 10.0,
    phi_span_deg: float = 14.0,
    max_points: int | None = 20000,
) -> bool:
    """
    Build BEV15 view-layer .rrd with 2D projected points for each oblique view.

    This is FUTURE/EXPERIMENTAL: view-only, post-run, not wired into runtime.
    Uses exact linear pushforward for positions (no S¹ reduction for directions).
    """
    if rr is None:
        print("ERROR: rerun-sdk not installed. pip install rerun-sdk", file=sys.stderr)
        return False

    from fl_slam_poc.common.bev_pushforward import BEVPushforwardConfig, oblique_Ps_bev15

    data = np.load(splat_npz, allow_pickle=True)
    n = int(data.get("n", data["positions"].shape[0]))
    positions = np.asarray(data["positions"], dtype=np.float64)[:n]
    colors = np.asarray(data["colors"], dtype=np.float64)[:n]
    weights = np.asarray(data["weights"], dtype=np.float64)[:n]

    if max_points is not None and n > max_points:
        top = np.argsort(-weights)[:max_points]
        positions = positions[top]
        colors = colors[top]
        n = len(top)

    cfg = BEVPushforwardConfig(
        n_views=int(n_views),
        phi_center_deg=float(phi_center_deg),
        phi_span_deg=float(phi_span_deg),
    )
    Ps = oblique_Ps_bev15(cfg)  # (N,2,3)

    rr.init("fl_slam_poc_bev15", default_enabled=True, spawn=False)
    rr.save(output_rrd)
    rr.set_time("time", timestamp=0.0)

    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    for k, P in enumerate(Ps):
        mu_bev = (P @ positions.T).T  # (n,2)
        rr.log(
            f"gc/bev15/view_{k:02d}",
            rr.Points2D(positions=mu_bev.astype(np.float32), colors=colors_uint8),
        )

    try:
        rec = rr.get_global_data_recording()
        if rec is not None:
            rec.flush()
    except Exception:
        pass

    return True


def _shaded_colors(
    positions: np.ndarray,
    colors: np.ndarray,
    directions: np.ndarray,
    kappas: np.ndarray,
    view_deg: float = BEV_15_DEG,
    fbm_scale: float = 0.15,
    fbm_octaves: int = 5,
    fbm_gain: float = 0.5,
    fbm_seed: int = 0,
) -> np.ndarray:
    """Per-splat shaded color: base_color * vMF(view) * fBm(position)."""
    from fl_slam_poc.backend.rendering import (
        fbm_at_splat_positions,
        vmf_shading_multi_lobe,
    )

    N = positions.shape[0]
    v = _bev15_view_direction(view_deg)
    colors = np.asarray(colors, dtype=np.float64).reshape(N, 3)
    out = np.zeros_like(colors)
    for i in range(N):
        mu_app = directions[i : i + 1]
        kappa_app = kappas[i : i + 1]
        s = vmf_shading_multi_lobe(v, mu_app, kappa_app)
        out[i] = colors[i] * s
    if fbm_scale > 0 and N > 0:
        fbm_val = fbm_at_splat_positions(
            positions[:, :2], octaves=fbm_octaves, gain=fbm_gain, seed=fbm_seed
        )
        fbm_mod = (1.0 - fbm_scale) + fbm_scale * np.clip(fbm_val, 0.0, 1.0)
        out = out * fbm_mod[:, np.newaxis]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _load_trajectory_tum(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load TUM file; return (stamps, path_xyz (N,3), path_quat_xyzw (N,4) or None)."""
    stamps = []
    xyz_list = []
    quat_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                stamps.append(float(parts[0]))
                xyz_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(parts) >= 8:
                quat_list.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    if not stamps:
        return np.zeros(0), np.zeros((0, 3)), None
    path_xyz = np.array(xyz_list, dtype=np.float64)
    path_quat = np.array(quat_list, dtype=np.float64) if len(quat_list) == len(stamps) else None
    return np.array(stamps), path_xyz, path_quat


def _log_trajectory_transforms(
    rr_module,
    path_xyz: np.ndarray,
    stamps: np.ndarray,
    path_quat_xyzw: np.ndarray | None = None,
) -> None:
    """Log each pose as Transform3D under gc/cameras (translation + rotation when quat available)."""
    n = path_xyz.shape[0]
    if n == 0:
        return
    has_quat = path_quat_xyzw is not None and path_quat_xyzw.shape[0] == n
    for i in range(n):
        t_sec = float(stamps[i]) if i < stamps.shape[0] else float(i)
        rr_module.set_time("time", timestamp= t_sec)
        pos = path_xyz[i].tolist()
        if has_quat:
            q = path_quat_xyzw[i].tolist()
            rr_module.log(
                "gc/cameras",
                rr_module.Transform3D(translation=pos, rotation=rr_module.datatypes.Quaternion(xyzw=q)),
            )
        else:
            rr_module.log("gc/cameras", rr_module.Transform3D(translation=pos))


def build_rrd(
    splat_npz: str,
    trajectory_tum: str | None,
    output_rrd: str,
    view_deg: float = BEV_15_DEG,
    fbm_scale: float = 0.15,
    max_ellipsoids: int | None = 20000,
) -> bool:
    """Build .rrd from splat NPZ + optional trajectory TUM. Returns True on success."""
    if rr is None:
        print("ERROR: rerun-sdk not installed. pip install rerun-sdk", file=sys.stderr)
        return False

    data = np.load(splat_npz, allow_pickle=True)
    n = int(data.get("n", data["positions"].shape[0]))
    positions = np.asarray(data["positions"], dtype=np.float64)[:n]
    covariances = np.asarray(data["covariances"], dtype=np.float64)[:n]
    colors = np.asarray(data["colors"], dtype=np.float64)[:n]
    weights = np.asarray(data["weights"], dtype=np.float64)[:n]
    directions = np.asarray(data["directions"], dtype=np.float64)[:n]
    kappas = np.asarray(data["kappas"], dtype=np.float64)[:n]
    # Prefer creation timestamps for "splats appear when first observed".
    timestamps = None
    if "created_timestamps" in data:
        timestamps = np.asarray(data["created_timestamps"], dtype=np.float64)[:n]
    elif "timestamps" in data:
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)[:n]
    primitive_ids = np.asarray(data["primitive_ids"], dtype=np.int64)[:n] if "primitive_ids" in data else None

    if n == 0:
        print("WARN: No primitives in splat export; writing empty Rerun.", file=sys.stderr)

    if max_ellipsoids is not None and n > max_ellipsoids:
        # Keep top by weight
        top = np.argsort(-weights)[:max_ellipsoids]
        positions = positions[top]
        covariances = covariances[top]
        colors = colors[top]
        directions = directions[top]
        kappas = kappas[top]
        n = len(top)

    half_sizes, quats = _covariances_to_ellipsoids(covariances)
    colors_shaded = _shaded_colors(
        positions, colors, directions, kappas,
        view_deg=view_deg, fbm_scale=fbm_scale,
    )
    # Rerun expects uint8 0-255 or float 0-1 for colors
    colors_uint8 = (np.clip(colors_shaded, 0, 1) * 255).astype(np.uint8)

    if trajectory_tum and os.path.isfile(trajectory_tum):
        stamps, path_xyz, path_quat = _load_trajectory_tum(trajectory_tum)
    else:
        stamps = np.zeros(0)
        path_xyz = np.zeros((0, 3))
        path_quat = None

    rr.init("fl_slam_poc", default_enabled=True, spawn=False)
    rr.save(output_rrd)

    # If timestamps available, log per-primitive so playback matches acquisition rate.
    if timestamps is not None and timestamps.shape[0] == n:
        ids = primitive_ids if primitive_ids is not None and primitive_ids.shape[0] == n else np.arange(n)
        order = np.argsort(timestamps)
        for i in order:
            rr.set_time("time", timestamp=float(timestamps[i]))
            rr.log(
                f"gc/map/ellipsoids/{int(ids[i])}",
                rr.Ellipsoids3D(
                    centers=positions[i : i + 1].astype(np.float32),
                    half_sizes=half_sizes[i : i + 1].astype(np.float32),
                    quaternions=quats[i : i + 1].astype(np.float32),
                    colors=colors_uint8[i : i + 1],
                ),
            )
    else:
        rr.set_time("time", timestamp=0.0)
        if n > 0:
            rr.log(
                "gc/map/ellipsoids",
                rr.Ellipsoids3D(
                    centers=positions.astype(np.float32),
                    half_sizes=half_sizes.astype(np.float32),
                    quaternions=quats.astype(np.float32),
                    colors=colors_uint8,
                ),
            )
        else:
            rr.log(
                "gc/map/ellipsoids",
                rr.Ellipsoids3D(centers=np.zeros((0, 3)), half_sizes=np.zeros((0, 3))),
            )

    if path_xyz.shape[0] > 0:
        # Replay trajectory over time (prefix line strip), so the recording is useful for debugging.
        for i in range(path_xyz.shape[0]):
            t_sec = float(stamps[i]) if i < stamps.shape[0] else float(i)
            rr.set_time("time", timestamp=t_sec)
            rr.log("gc/trajectory", rr.LineStrips3D([path_xyz[: i + 1].astype(np.float32)]))
        _log_trajectory_transforms(rr, path_xyz, stamps, path_quat)

    try:
        rec = rr.get_global_data_recording()
        if rec is not None:
            rec.flush()
    except Exception:
        pass

    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build Rerun .rrd from splat_export.npz + trajectory (post-run)."
    )
    ap.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Results directory (splat_export.npz and estimated_trajectory.tum inside)",
    )
    ap.add_argument("--splat", default=None, help="Path to splat_export.npz")
    ap.add_argument("--trajectory", default=None, help="Path to TUM trajectory")
    ap.add_argument("--output", default=None, help="Output .rrd path")
    ap.add_argument("--view-deg", type=float, default=BEV_15_DEG, help="BEV view angle (deg off vertical)")
    ap.add_argument("--fbm-scale", type=float, default=0.15, help="fBm color modulation 0..1")
    ap.add_argument("--max-ellipsoids", type=int, default=20000, help="Cap ellipsoids by weight (0 = no cap)")
    ap.add_argument("--bev15-output", default=None, help="Optional BEV15 .rrd output path (post-run)")
    ap.add_argument("--bev15-n", type=int, default=15, help="BEV15 number of views")
    ap.add_argument("--bev15-center-deg", type=float, default=10.0, help="BEV15 center oblique angle (deg)")
    ap.add_argument("--bev15-span-deg", type=float, default=14.0, help="BEV15 total span across views (deg)")
    args = ap.parse_args()

    if args.results_dir is not None:
        res = Path(args.results_dir)
        splat = args.splat or str(res / "splat_export.npz")
        traj = args.trajectory
        if traj is None:
            for name in ("estimated_trajectory.tum", "estimated_trajectory_wheel.tum"):
                p = res / name
                if p.is_file():
                    traj = str(p)
                    break
        if traj is None:
            traj = str(res / "estimated_trajectory.tum")
        out = args.output or str(res / "gc_slam.rrd")
    else:
        if not args.splat:
            print("ERROR: Either results_dir or --splat required.", file=sys.stderr)
            return 1
        splat = args.splat
        traj = args.trajectory
        out = args.output or "gc_slam.rrd"

    if not os.path.isfile(splat):
        print(f"ERROR: Splat file not found: {splat}", file=sys.stderr)
        return 1
    if not os.path.isfile(traj):
        print(f"WARN: Trajectory not found: {traj}; logging map only.", file=sys.stderr)
        traj = None

    max_ell = None if args.max_ellipsoids <= 0 else args.max_ellipsoids
    ok = build_rrd(
        splat,
        traj,
        out,
        view_deg=args.view_deg,
        fbm_scale=args.fbm_scale,
        max_ellipsoids=max_ell,
    )
    if ok:
        print(f"Rerun recording written: {out}")
    if ok and args.bev15_output:
        bev_ok = _build_bev15_rrd(
            splat_npz=splat,
            output_rrd=args.bev15_output,
            n_views=args.bev15_n,
            phi_center_deg=args.bev15_center_deg,
            phi_span_deg=args.bev15_span_deg,
            max_points=max_ell,
        )
        if bev_ok:
            print(f"BEV15 Rerun recording written: {args.bev15_output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
