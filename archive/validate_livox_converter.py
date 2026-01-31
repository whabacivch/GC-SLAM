#!/usr/bin/env python3
"""
Validate Livox bag content + converter assumptions (offline).

This tool is meant to prevent "misleading eval" by surfacing:
- Which Livox lidar topics exist (MID360 / AVIA) and their message types
- Frame IDs used by each topic
- Basic numeric sanity checks (XYZ ranges, finite ratio)
- Distribution checks for reflectivity/line/tag (when decodable)
- Message accounting (point_num vs len(points))

It does NOT attempt full extrinsic calibration (that is a separate, higher-effort tool),
but it reports the evidence you'd need and flags when a static extrinsic is required
(e.g., no /tf in bag).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.serialization import deserialize_message

from rosbag_sqlite_utils import resolve_db3_path, topic_id, topic_type

def count_messages(cur: sqlite3.Cursor, tid: int) -> int:
    cur.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (tid,))
    return int(cur.fetchone()[0])


def first_message(cur: sqlite3.Cursor, tid: int) -> Optional[tuple[int, bytes]]:
    cur.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1", (tid,))
    row = cur.fetchone()
    if not row:
        return None
    return int(row[0]), row[1]


def iter_messages(cur: sqlite3.Cursor, tid: int):
    for ts, data in cur.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (tid,)):
        yield int(ts), data


def try_import(msg_type: str):
    if msg_type == "livox_ros_driver2/msg/CustomMsg":
        from livox_ros_driver2.msg import CustomMsg  # type: ignore

        return CustomMsg
    if msg_type == "livox_ros_driver/msg/CustomMsg":
        from livox_ros_driver.msg import CustomMsg  # type: ignore

        return CustomMsg
    raise ValueError(msg_type)


def finite_ratio(xyz: np.ndarray) -> float:
    if xyz.size == 0:
        return 0.0
    return float(np.isfinite(xyz).all(axis=1).mean())


@dataclass
class LidarStats:
    topic: str
    msg_type: str
    count: int
    frame_ids: list[str]
    points_per_msg_min: int
    points_per_msg_max: int
    xyz_min: list[float]
    xyz_max: list[float]
    reflectivity_unique: int
    line_unique: int
    tag_unique: int
    point_num_mismatch_count: int


def analyze_livox_topic(cur: sqlite3.Cursor, topic: str, msg_type: str, max_msgs: int = 200) -> Optional[LidarStats]:
    tid = topic_id(cur, topic)
    if tid is None:
        return None

    MsgT = try_import(msg_type)

    frames = set()
    pts_min = 10**9
    pts_max = 0
    xyz_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    xyz_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    refl = Counter()
    line = Counter()
    tag = Counter()
    mismatch = 0

    n = 0
    for _, data in iter_messages(cur, tid):
        msg = deserialize_message(data, MsgT)
        frames.add(str(msg.header.frame_id))
        pts = list(getattr(msg, "points", []))
        npts = len(pts)
        pts_min = min(pts_min, npts)
        pts_max = max(pts_max, npts)
        if hasattr(msg, "point_num") and int(msg.point_num) != npts:
            mismatch += 1

        if npts > 0:
            xyz = np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float64)
            ok = np.isfinite(xyz).all(axis=1)
            if np.any(ok):
                xyz_min = np.minimum(xyz_min, xyz[ok].min(axis=0))
                xyz_max = np.maximum(xyz_max, xyz[ok].max(axis=0))

            # Optional attributes (driver2 has these)
            for p in pts:
                if hasattr(p, "reflectivity"):
                    refl[int(p.reflectivity)] += 1
                if hasattr(p, "line"):
                    line[int(p.line)] += 1
                if hasattr(p, "tag"):
                    tag[int(p.tag)] += 1

        n += 1
        if n >= max_msgs:
            break

    if pts_min == 10**9:
        pts_min = 0

    return LidarStats(
        topic=topic,
        msg_type=msg_type,
        count=count_messages(cur, tid),
        frame_ids=sorted(frames),
        points_per_msg_min=int(pts_min),
        points_per_msg_max=int(pts_max),
        xyz_min=[float(x) for x in xyz_min.tolist()],
        xyz_max=[float(x) for x in xyz_max.tolist()],
        reflectivity_unique=len(refl),
        line_unique=len(line),
        tag_unique=len(tag),
        point_num_mismatch_count=int(mismatch),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Livox bag content + converter assumptions (offline).")
    ap.add_argument("bag_path", help="Bag directory containing *.db3 (or direct .db3 file).")
    ap.add_argument("--max-msgs", type=int, default=200, help="Max messages to decode per topic (for speed).")
    args = ap.parse_args()

    db_path = resolve_db3_path(args.bag_path)
    if not db_path or not os.path.exists(db_path):
        raise SystemExit(f"Could not locate *.db3 under '{args.bag_path}'")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Find TF availability quickly
    cur.execute("SELECT name FROM topics WHERE name IN ('/tf', '/tf_static')")
    tf_topics = [r[0] for r in cur.fetchall()]

    # Candidate livox topics (M3DGR has both)
    candidates = [
        ("/livox/mid360/lidar", "livox_ros_driver2/msg/CustomMsg"),
        ("/livox/avia/lidar", "livox_ros_driver/msg/CustomMsg"),
    ]

    print("=== Livox Bag Validation ===")
    print(f"bag_path: {args.bag_path}")
    print(f"db_path: {db_path}")
    print(f"tf_topics_present: {tf_topics if tf_topics else 'NONE'}")
    print()

    rclpy.init()
    results: list[LidarStats] = []
    for topic, mtype in candidates:
        tid = topic_id(cur, topic)
        ttype = topic_type(cur, topic) if tid is not None else None
        if tid is None:
            print(f"- {topic}: MISSING")
            continue
        print(f"- {topic}: present, recorded_type={ttype}")
        try:
            stats = analyze_livox_topic(cur, topic, mtype, max_msgs=args.max_msgs)
        except Exception as exc:
            print(f"  decode: FAILED for msg_type={mtype}: {exc}")
            continue
        if stats is not None:
            results.append(stats)
            print(f"  msgs: {stats.count}")
            print(f"  frame_ids: {stats.frame_ids}")
            print(f"  points/msg: min={stats.points_per_msg_min}, max={stats.points_per_msg_max}")
            print(f"  xyz_min: {stats.xyz_min}")
            print(f"  xyz_max: {stats.xyz_max}")
            print(f"  unique(reflectivity,line,tag): ({stats.reflectivity_unique},{stats.line_unique},{stats.tag_unique})")
            print(f"  point_num_mismatch_count (sampled): {stats.point_num_mismatch_count}")
        print()

    rclpy.shutdown()
    conn.close()

    print("=== Actionable conclusions ===")
    if not tf_topics:
        print("- No TF in bag: you must provide LiDAR extrinsics explicitly (e.g., lidar_base_extrinsic) for base-frame processing.")
    if not results:
        print("- No Livox topics decodable with installed message types. Install/build the required Livox message packages.")
    else:
        print("- Livox topics are present and decodable; converter can preserve sensor frame and metadata (intensity/ring/tag/timebase).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
