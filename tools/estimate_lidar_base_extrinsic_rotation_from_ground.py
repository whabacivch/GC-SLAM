#!/usr/bin/env python3
"""
Estimate a static LiDAR-to-base rotation in no-TF bags by fitting the ground plane.

Goal:
  Estimate roll/pitch alignment for T_{base<-lidar} so the dominant ground-plane normal
  maps to +Z in base_footprint (Z-up). Yaw about gravity is unobservable from a plane.

This is an *offline calibration helper* (not a runtime operator):
- Uses fixed-iteration RANSAC (no adaptive early exit).
- Reports both sign alternatives (+/- normal) because plane normals are sign-ambiguous.

Assumptions (M3DGR / Dynamic01 typically satisfies these):
- The ground plane is one of the largest planar structures in a scan.
- The base frame is Z-up.
- LiDAR topic is Livox CustomMsg (mid360): /livox/mid360/lidar
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass

import numpy as np

from rosbag_sqlite_utils import resolve_db3_path, topic_id, topic_type


def _iter_msgs(cur: sqlite3.Cursor, tid: int):
    for ts, data in cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (tid,),
    ):
        yield int(ts), data


def _rotvec_from_two_unit_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    v_from = np.asarray(v_from, dtype=float).reshape(3)
    v_to = np.asarray(v_to, dtype=float).reshape(3)
    v_from = v_from / (np.linalg.norm(v_from) + 1e-12)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-12)

    axis = np.cross(v_from, v_to)
    s = np.linalg.norm(axis)
    c = float(np.dot(v_from, v_to))
    angle = float(np.arctan2(s, c))
    if s < 1e-12:
        return np.zeros((3,), dtype=float)
    axis = axis / s
    return axis * angle


@dataclass
class _PlaneFit:
    normal: np.ndarray  # (3,)
    d: float  # plane offset in form n·x + d = 0 (with ||n||=1)
    inlier_frac: float


def _fit_plane_from_3pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, float] | None:
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    nn = np.linalg.norm(n)
    if not np.isfinite(nn) or nn < 1e-9:
        return None
    n = n / nn
    d = -float(np.dot(n, p1))
    return n, d


def _ransac_ground_plane(
    points: np.ndarray,
    *,
    iters: int,
    dist_thresh: float,
    sample_size: int,
    expected_height: float,
    height_sigma: float,
    rng: np.random.Generator,
) -> _PlaneFit | None:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 100:
        return None

    # Downsample for speed (deterministic given RNG seed).
    if pts.shape[0] > sample_size:
        idx = rng.choice(pts.shape[0], size=sample_size, replace=False)
        pts = pts[idx]

    best: _PlaneFit | None = None
    best_score = -1.0
    n_pts = pts.shape[0]
    for _ in range(int(iters)):
        i = rng.choice(n_pts, size=3, replace=False)
        fit = _fit_plane_from_3pts(pts[i[0]], pts[i[1]], pts[i[2]])
        if fit is None:
            continue
        n, d = fit
        # Fix plane sign so that d >= 0 (origin is on the +n side).
        # For a ground plane below the sensor, this makes d approximately the sensor height above ground.
        if d < 0.0:
            n = -n
            d = -d
        # Distances to plane
        dist = np.abs(pts @ n + d)
        inliers = dist < dist_thresh
        frac = float(np.mean(inliers))

        # Soft preference for planes at a plausible distance from the LiDAR origin.
        # This disambiguates "largest plane" vs "ground plane" without hard gates.
        h = float(expected_height)
        s = float(max(height_sigma, 1e-6))
        height_weight = float(np.exp(-0.5 * ((abs(d) - h) / s) ** 2))
        score = frac * height_weight

        if score > best_score:
            best_score = score
            best = _PlaneFit(normal=n, d=d, inlier_frac=frac)
    return best


def _pointcloud2_to_xyz(msg) -> np.ndarray:
    """Extract x,y,z from sensor_msgs/PointCloud2 (VLP-16 or any layout with x,y,z). Returns (N,3) float64."""
    n_points = msg.width * msg.height
    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float64)
    field_map = {f.name: (f.offset, f.datatype) for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in field_map:
            raise RuntimeError(f"PointCloud2 missing field '{name}'")
    def _size(ft):
        return 8 if ft == 8 else 4
    x_off, x_dt = field_map["x"]
    y_off, y_dt = field_map["y"]
    z_off, z_dt = field_map["z"]
    step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    x = np.zeros(n_points, dtype=np.float64)
    y = np.zeros(n_points, dtype=np.float64)
    z = np.zeros(n_points, dtype=np.float64)
    for i in range(n_points):
        base = i * step
        sz_x, sz_y, sz_z = _size(x_dt), _size(y_dt), _size(z_dt)
        x[i] = np.frombuffer(data[base + x_off : base + x_off + sz_x].tobytes(), dtype=np.float64 if sz_x == 8 else np.float32)[0]
        y[i] = np.frombuffer(data[base + y_off : base + y_off + sz_y].tobytes(), dtype=np.float64 if sz_y == 8 else np.float32)[0]
        z[i] = np.frombuffer(data[base + z_off : base + z_off + sz_z].tobytes(), dtype=np.float64 if sz_z == 8 else np.float32)[0]
    out = np.column_stack([x, y, z])
    return out[np.isfinite(out).all(axis=1)]


def _extract_pointcloud2_scans(cur: sqlite3.Cursor, tid: int, *, skip_scans: int, max_scans: int):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    n_used = 0
    for i, (_ts, data) in enumerate(_iter_msgs(cur, tid)):
        if i < skip_scans:
            continue
        msg = deserialize_message(data, PointCloud2)
        pts = _pointcloud2_to_xyz(msg)
        if pts.shape[0] >= 100:
            yield pts
            n_used += 1
            if max_scans > 0 and n_used >= max_scans:
                break


def _extract_livox_custom_points_m(cur: sqlite3.Cursor, tid: int, *, skip_scans: int, max_scans: int):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    # Avoid hard dependency on generated Python modules for custom messages.
    # This resolves the message type via rosidl runtime (works as long as the interface is installed).
    CustomMsg = get_message("livox_ros_driver2/msg/CustomMsg")

    n_used = 0
    for i, (_ts, data) in enumerate(_iter_msgs(cur, tid)):
        if i < skip_scans:
            continue
        msg = deserialize_message(data, CustomMsg)
        pts = []
        for p in msg.points:
            # livox_ros_driver2 defines CustomPoint.{x,y,z} as float32 (meters).
            x = float(p.x)
            y = float(p.y)
            z = float(p.z)
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                pts.append((x, y, z))
        if pts:
            yield np.asarray(pts, dtype=np.float64)
            n_used += 1
            if max_scans > 0 and n_used >= max_scans:
                break


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Estimate LiDAR->base rotation from dominant ground plane (roll/pitch only)."
    )
    ap.add_argument("bag_path", help="Bag directory containing *.db3 (or direct .db3 file).")
    ap.add_argument("--lidar-topic", default="/livox/mid360/lidar")
    ap.add_argument("--skip-scans", type=int, default=50)
    ap.add_argument("--max-scans", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200, help="RANSAC iterations per scan (fixed).")
    ap.add_argument("--dist-thresh", type=float, default=0.08, help="Plane inlier threshold (meters).")
    ap.add_argument("--sample-size", type=int, default=20000, help="Max points per scan used for fitting.")
    ap.add_argument(
        "--expected-height",
        type=float,
        default=0.778,
        help="Expected LiDAR height above the ground plane (meters). Used as a soft preference.",
    )
    ap.add_argument(
        "--height-sigma",
        type=float,
        default=0.30,
        help="Soft preference width for expected height (meters). Larger = weaker preference.",
    )
    ap.add_argument(
        "--base-up",
        default="0,0,1",
        help="Expected up direction in base frame (unit-ish), default +Z.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        raise SystemExit(f"Could not locate *.db3 under '{args.bag_path}'")

    base_up = np.array([float(x) for x in args.base_up.split(",")], dtype=float).reshape(3)
    base_up = base_up / (np.linalg.norm(base_up) + 1e-12)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        t = topic_type(cur, args.lidar_topic)
        print("=== Estimate LiDAR->base rotation (ground plane alignment) ===")
        print(f"db_path:     {db_path}")
        print(f"lidar_topic: {args.lidar_topic} ({t or 'MISSING'})")
        if not t:
            return 2

        tid = topic_id(cur, args.lidar_topic)
        if tid is None:
            raise RuntimeError(f"Topic not found: {args.lidar_topic}")

        use_pointcloud2 = t and "PointCloud2" in t
        rng = np.random.default_rng(int(args.seed))

        normals = []
        fracs = []
        ds = []
        n_scans = 0
        extractor = (
            _extract_pointcloud2_scans if use_pointcloud2 else _extract_livox_custom_points_m
        )
        for pts in extractor(
            cur, tid, skip_scans=int(args.skip_scans), max_scans=int(args.max_scans)
        ):
            fit = _ransac_ground_plane(
                pts,
                iters=int(args.iters),
                dist_thresh=float(args.dist_thresh),
                sample_size=int(args.sample_size),
                expected_height=float(args.expected_height),
                height_sigma=float(args.height_sigma),
                rng=rng,
            )
            if fit is None:
                continue
            normals.append(fit.normal)
            fracs.append(fit.inlier_frac)
            ds.append(fit.d)
            n_scans += 1

        if n_scans == 0:
            raise RuntimeError("No scans produced a valid plane fit (try lowering dist threshold or skipping fewer).")

        normals = np.asarray(normals, dtype=np.float64)
        fracs = np.asarray(fracs, dtype=np.float64)
        ds = np.asarray(ds, dtype=np.float64)
        n_mean = np.mean(normals, axis=0)
        n_mean = n_mean / (np.linalg.norm(n_mean) + 1e-12)

        rv = _rotvec_from_two_unit_vectors(n_mean, base_up)
        ang_deg = float(np.linalg.norm(rv) * 180.0 / np.pi)
        dot = float(np.dot(n_mean, base_up))

        rv_flip = _rotvec_from_two_unit_vectors(-n_mean, base_up)
        ang_flip_deg = float(np.linalg.norm(rv_flip) * 180.0 / np.pi)

        print(f"scans_used: {n_scans}")
        print(f"inlier_frac: mean={float(np.mean(fracs)):.3f} min={float(np.min(fracs)):.3f} max={float(np.max(fracs)):.3f}")
        print(f"plane_d (origin->plane distance): mean={float(np.mean(ds)):.3f} min={float(np.min(ds)):.3f} max={float(np.max(ds)):.3f} (expected ~{float(args.expected_height):.3f})")
        print(f"mean_plane_normal_lidar: {n_mean.tolist()}")
        print(f"expected_up_base:        {base_up.tolist()}")
        print(f"dot(mean, expected):     {dot:.6f}")
        print()

        print("Suggested T_base_lidar rotation (roll/pitch; yaw unobservable from a plane):")
        print(f"  rotvec_rad: {rv.tolist()}")
        print(f"  angle_deg:  {ang_deg:.3f}")
        print(f"  launch/yaml: T_base_lidar: [x, y, z, {rv[0]:.6f}, {rv[1]:.6f}, {rv[2]:.6f}]")
        print()
        print("If the fitted plane normal sign is flipped, alternative:")
        print(f"  rotvec_rad: {rv_flip.tolist()}")
        print(f"  angle_deg:  {ang_flip_deg:.3f}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
