#!/usr/bin/env python3
"""
Estimate a static IMU-to-base rotation in no-TF bags using the accelerometer gravity direction.

This tool is intentionally simple and conservative:
- Uses only the mean accelerometer *direction* (unit vectors), so scale is irrelevant.
- Solves the minimal rotation R such that:
    a_base_dir ≈ R_base<-imu @ a_imu_dir

Notes:
- With gravity alone, yaw about gravity is unobservable. This tool estimates roll/pitch alignment only.
- For Livox Mid-360 bags, /livox/mid360/imu is typically in "livox_frame" and reports acceleration in g.
- Geometric Compositional uses Z-up world gravity (0,0,-9.81), so the expected *specific force* direction at rest is +Z.
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
class _AccelStats:
    n: int
    mean_dir: np.ndarray  # (3,)
    mean_mag: float


def _estimate_mean_accel_dir(
    cur: sqlite3.Cursor,
    topic: str,
    max_msgs: int,
    skip_msgs: int,
) -> _AccelStats:
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    tid = topic_id(cur, topic)
    if tid is None:
        raise RuntimeError(f"Topic not found: {topic}")

    sum_dir = np.zeros((3,), dtype=float)
    sum_mag = 0.0
    n = 0

    for i, (_ts, data) in enumerate(_iter_msgs(cur, tid)):
        if i < skip_msgs:
            continue
        msg = deserialize_message(data, Imu)
        a = np.array(
            [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            dtype=float,
        )
        mag = float(np.linalg.norm(a))
        if not np.isfinite(mag) or mag < 1e-9:
            continue
        sum_mag += mag
        sum_dir += a / mag
        n += 1
        if max_msgs > 0 and n >= max_msgs:
            break

    if n == 0:
        raise RuntimeError("No valid IMU accel samples found.")

    mean_dir = sum_dir / (np.linalg.norm(sum_dir) + 1e-12)
    mean_mag = sum_mag / max(n, 1)
    return _AccelStats(n=n, mean_dir=mean_dir, mean_mag=mean_mag)


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate static IMU->base rotation from accel gravity direction.")
    ap.add_argument("bag_path", help="Bag directory containing *.db3 (or direct .db3 file).")
    ap.add_argument("--imu-topic", default="/livox/mid360/imu")
    ap.add_argument("--max-msgs", type=int, default=20000)
    ap.add_argument("--skip-msgs", type=int, default=500)
    ap.add_argument(
        "--base-accel-dir",
        default="0,0,1",
        help="Expected accel direction in base at rest (unit-ish), default +Z.",
    )
    args = ap.parse_args()

    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        raise SystemExit(f"Could not locate *.db3 under '{args.bag_path}'")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        t = topic_type(cur, args.imu_topic)
        print("=== Estimate IMU->base rotation (gravity alignment) ===")
        print(f"db_path:   {db_path}")
        print(f"imu_topic: {args.imu_topic} ({t or 'MISSING'})")
        if not t:
            return 2

        stats = _estimate_mean_accel_dir(cur, args.imu_topic, int(args.max_msgs), int(args.skip_msgs))
        base_dir = np.array([float(x) for x in args.base_accel_dir.split(",")], dtype=float).reshape(3)
        base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-12)

        rv = _rotvec_from_two_unit_vectors(stats.mean_dir, base_dir)
        ang_deg = float(np.linalg.norm(rv) * 180.0 / np.pi)
        dot = float(np.dot(stats.mean_dir, base_dir))

        # Also report the sign-flip alternative (common convention confusion).
        rv_flip = _rotvec_from_two_unit_vectors(stats.mean_dir, -base_dir)
        ang_flip_deg = float(np.linalg.norm(rv_flip) * 180.0 / np.pi)

        print(f"samples_used: {stats.n}")
        print(f"mean_accel_mag: {stats.mean_mag:.6f} (units are whatever the bag reports; Livox often uses g)")
        print(f"mean_accel_dir_imu: {stats.mean_dir.tolist()}")
        print(f"expected_accel_dir_base: {base_dir.tolist()}")
        print(f"dot(mean, expected): {dot:.6f}")
        print()

        print("Suggested T_base_imu rotation (roll/pitch alignment; yaw unobservable from gravity):")
        print(f"  rotvec_rad: {rv.tolist()}")
        print(f"  angle_deg:  {ang_deg:.3f}")
        print(f"  launch/yaml: T_base_imu: [0, 0, 0, {rv[0]:.6f}, {rv[1]:.6f}, {rv[2]:.6f}]")
        print()
        print("If your gravity/sign convention is flipped, alternative (+/-Z ambiguity):")
        print(f"  rotvec_rad: {rv_flip.tolist()}")
        print(f"  angle_deg:  {ang_flip_deg:.3f}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

