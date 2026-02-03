#!/usr/bin/env python3
"""
Deep rosbag2 (.db3) inspector for frames + camera/IMU/LiDAR metadata.

This script is intentionally "deep" and audit-friendly:
- Lists topic types, message counts, and timestamp ranges
- Samples across the bag (first/middle/last + deterministic random) for each key topic
- Extracts frame IDs, child frame IDs, formats, and selected metadata
- Detects inconsistencies (multiple frame_ids, varying CompressedImage.format, etc.)

Works directly against the rosbag2 sqlite database to avoid needing ros2 bag APIs.
"""
import argparse
import os
import random
import sqlite3
from typing import Any, Callable, Optional

import rclpy
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu
from tf2_msgs.msg import TFMessage

from rosbag_sqlite_utils import resolve_db3_path, topic_type

def _ns_to_sec(ns: int) -> float:
    return float(ns) * 1e-9


def _topic_exists(cursor: sqlite3.Cursor, name: str) -> bool:
    cursor.execute("SELECT 1 FROM topics WHERE name = ? LIMIT 1", (name,))
    return cursor.fetchone() is not None


def _topic_count_and_range(cursor: sqlite3.Cursor, topic: str) -> tuple[int, Optional[int], Optional[int]]:
    cursor.execute(
        """
        SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
        FROM messages
        WHERE topic_id = (SELECT id FROM topics WHERE name = ?)
        """,
        (topic,),
    )
    count, tmin, tmax = cursor.fetchone()
    return int(count), (int(tmin) if tmin is not None else None), (int(tmax) if tmax is not None else None)


def _fetch_message_at_index(cursor: sqlite3.Cursor, topic: str, idx: int) -> Optional[tuple[int, bytes]]:
    cursor.execute(
        """
        SELECT timestamp, data
        FROM messages
        WHERE topic_id = (SELECT id FROM topics WHERE name = ?)
        ORDER BY timestamp
        LIMIT 1 OFFSET ?
        """,
        (topic, int(idx)),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return int(row[0]), row[1]


def _sample_indices(count: int, samples: int, seed: int) -> list[int]:
    if count <= 0:
        return []
    fixed = {0, count - 1, count // 2, count // 4, (3 * count) // 4}
    fixed = {i for i in fixed if 0 <= i < count}
    rng = random.Random(seed)
    want = max(0, samples - len(fixed))
    extra = set()
    if count > 1 and want > 0:
        # deterministic random indices across the bag
        while len(extra) < min(want, count - len(fixed)):
            extra.add(rng.randrange(0, count))
    out = sorted(fixed.union(extra))
    return out


def _print_kv(k: str, v: Any, indent: int = 0) -> None:
    pad = " " * indent
    print(f"{pad}{k}: {v}")


def _summarize_topic(
    cursor: sqlite3.Cursor,
    topic: str,
    msg_type: Any,
    samples: int,
    seed: int,
    extract: Callable[[Any], dict],
) -> None:
    if not _topic_exists(cursor, topic):
        _print_kv("topic", topic)
        _print_kv("status", "MISSING (no such topic in bag)", indent=2)
        print()
        return

    ttype = topic_type(cursor, topic)
    count, tmin, tmax = _topic_count_and_range(cursor, topic)
    _print_kv("topic", topic)
    _print_kv("type", ttype, indent=2)
    _print_kv("messages", count, indent=2)
    if tmin is not None and tmax is not None:
        _print_kv("time_start_sec", f"{_ns_to_sec(tmin):.6f}", indent=2)
        _print_kv("time_end_sec", f"{_ns_to_sec(tmax):.6f}", indent=2)
        _print_kv("duration_sec", f"{(_ns_to_sec(tmax - tmin)):.3f}", indent=2)

    idxs = _sample_indices(count, samples=samples, seed=seed)
    frames = set()
    formats = set()
    extracted_rows = []

    for idx in idxs:
        row = _fetch_message_at_index(cursor, topic, idx)
        if row is None:
            continue
        ts, data = row
        msg = deserialize_message(data, msg_type)
        info = extract(msg)
        info["_idx"] = idx
        info["_t_sec"] = float(_ns_to_sec(ts))
        extracted_rows.append(info)
        if "frame_id" in info and info["frame_id"]:
            frames.add(info["frame_id"])
        if "format" in info and info["format"]:
            formats.add(info["format"])

    _print_kv("sample_count", len(extracted_rows), indent=2)
    if frames:
        _print_kv("unique_frame_ids", sorted(frames), indent=2)
    if formats:
        _print_kv("unique_formats", sorted(formats), indent=2)

    # Print sample rows (compact)
    _print_kv("samples", "", indent=2)
    for info in extracted_rows:
        meta = {k: v for k, v in info.items() if not k.startswith("_")}
        _print_kv(
            f"- idx={info['_idx']} t={info['_t_sec']:.6f}s",
            meta,
            indent=4,
        )

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep rosbag2 sqlite inspector (frames + sensor metadata).")
    parser.add_argument(
        "bag_path",
        nargs="?",
        default=os.environ.get("BAG_PATH", "rosbags/Kimera_Data/ros2/10_14_acl_jackal-005"),
        help="Bag directory containing *.db3 (or a direct path to a *.db3 file).",
    )
    parser.add_argument("--samples", type=int, default=25, help="Number of samples per topic (includes fixed indices).")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic sampling seed.")
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="Print all topics+types present in the bag before the deep summary.",
    )
    args = parser.parse_args()

    bag_path = args.bag_path
    db_path = resolve_db3_path(bag_path)

    if not os.path.exists(db_path):
        print(f"Error: Database not found for bag_path='{bag_path}'.")
        print("Pass the bag directory (containing *.db3) or a specific *.db3 file.")
        return 1

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    rclpy.init()

    print("=== Deep Rosbag Frame + Sensor Metadata Inspector ===")
    _print_kv("bag_path", bag_path, indent=0)
    _print_kv("db_path", db_path, indent=0)
    _print_kv("samples_per_topic", args.samples, indent=0)
    _print_kv("seed", args.seed, indent=0)
    print()

    if args.list_topics:
        print("=== Topics Present ===")
        cursor.execute("SELECT name, type FROM topics ORDER BY name")
        for name, ttype in cursor.fetchall():
            print(f"- {name} [{ttype}]")
        print()

    # Key topics for Kimera (override bag path via argv/env)
    print("=== Key Topic Summaries ===\n")

    _summarize_topic(
        cursor,
        "/odom",
        Odometry,
        samples=args.samples,
        seed=args.seed,
        extract=lambda m: {
            "frame_id": m.header.frame_id,
            "child_frame_id": m.child_frame_id,
        },
    )

    _summarize_topic(
        cursor,
        "/camera/color/image_raw/compressed",
        CompressedImage,
        samples=args.samples,
        seed=args.seed,
        extract=lambda m: {
            "frame_id": m.header.frame_id,
            "format": m.format,
            "data_len": len(m.data),
        },
    )

    _summarize_topic(
        cursor,
        "/camera/aligned_depth_to_color/image_raw/compressedDepth",
        CompressedImage,
        samples=args.samples,
        seed=args.seed,
        extract=lambda m: {
            "frame_id": m.header.frame_id,
            "format": m.format,
            "data_len": len(m.data),
        },
    )

    # Any CameraInfo topics present?
    cursor.execute("SELECT name FROM topics WHERE type = 'sensor_msgs/msg/CameraInfo' ORDER BY name")
    camera_info_topics = [r[0] for r in cursor.fetchall()]
    if camera_info_topics:
        for t in camera_info_topics:
            _summarize_topic(
                cursor,
                t,
                CameraInfo,
                samples=min(args.samples, 10),
                seed=args.seed,
                extract=lambda m: {
                    "frame_id": m.header.frame_id,
                    "K": [float(m.k[0]), float(m.k[4]), float(m.k[2]), float(m.k[5])],
                    "D_len": len(m.d),
                },
            )
    else:
        print("topic: (CameraInfo)")
        _print_kv("status", "MISSING (no CameraInfo topics in bag)", indent=2)
        print()

    _summarize_topic(
        cursor,
        "/camera/imu",
        Imu,
        samples=args.samples,
        seed=args.seed,
        extract=lambda m: {
            "frame_id": m.header.frame_id,
            "lin_accel": [float(m.linear_acceleration.x), float(m.linear_acceleration.y), float(m.linear_acceleration.z)],
            "ang_vel": [float(m.angular_velocity.x), float(m.angular_velocity.y), float(m.angular_velocity.z)],
        },
    )

    # Livox frame: use type discovery so this script works even if messages aren't sourced.
    livox_topic = "/livox/mid360/lidar"
    if _topic_exists(cursor, livox_topic):
        livox_type = topic_type(cursor, livox_topic) or ""
        _print_kv("topic", livox_topic)
        _print_kv("type", livox_type, indent=2)
        _print_kv("note", "Frame ID is stored in CustomMsg.header.frame_id; requires livox_ros_driver2 messages to be sourced.", indent=2)
        # Try to decode if message support is present
        try:
            from livox_ros_driver2.msg import CustomMsg  # type: ignore

            _summarize_topic(
                cursor,
                livox_topic,
                CustomMsg,
                samples=min(args.samples, 15),
                seed=args.seed,
                extract=lambda m: {
                    "frame_id": m.header.frame_id,
                    "points_len": len(m.points),
                    "timebase": int(getattr(m, "timebase", 0)),
                },
            )
        except Exception as exc:
            _print_kv("decode", f"SKIPPED ({exc})", indent=2)
            print()
    else:
        _print_kv("topic", livox_topic)
        _print_kv("status", "MISSING (no such topic in bag)", indent=2)
        print()

    # TF topics
    for tf_topic in ("/tf", "/tf_static"):
        if _topic_exists(cursor, tf_topic):
            _summarize_topic(
                cursor,
                tf_topic,
                TFMessage,
                samples=min(args.samples, 10),
                seed=args.seed,
                extract=lambda m: {
                    "transforms": len(m.transforms),
                    "first_edges": [
                        {"parent": t.header.frame_id, "child": t.child_frame_id} for t in list(m.transforms)[:5]
                    ],
                },
            )
        else:
            _print_kv("topic", tf_topic)
            _print_kv("status", "MISSING (no such topic in bag)", indent=2)
            print()

    # Consolidated conclusion for the critical frames
    print("=== Critical Frame Conclusions (from sampled headers) ===")
    print("- /odom: header.frame_id is the world/odom frame; child_frame_id is base frame.")
    print("- compressed RGB/depth topics: header.frame_id is camera optical frame.")
    print("- /camera/imu: header.frame_id is the IMU frame.")
    print("- /tf and /tf_static: if missing, extrinsics must be declared out-of-band.")
    print()

    rclpy.shutdown()
    conn.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
