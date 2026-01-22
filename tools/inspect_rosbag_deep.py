#!/usr/bin/env python3
"""
Deep rosbag2 inspector (sqlite .db3).

Goal: make evaluation robust by surfacing *everything that can bite us*:
- All topics, types, serialization formats, and (when available) offered QoS
- Message counts per topic + timestamp ranges + per-topic monotonicity/duplicates
- For common sensor types: full-scan unique frame_ids / encodings / formats
- TF presence, CameraInfo presence, frame consistency warnings
- Optional JSON summary export for CI/automation

Usage:
  python3 tools/inspect_rosbag_deep.py /path/to/bag_dir
  python3 tools/inspect_rosbag_deep.py /path/to/bag_dir --json /tmp/bag_summary.json
  python3 tools/inspect_rosbag_deep.py /path/to/bag_dir --full-scan-all

Notes:
  - Expects a ROS 2 bag directory containing at least one *.db3 file.
  - Uses rclpy.serialization to deserialize when message Python types are available.
  - Unknown message types are still reported, but only structurally inspected.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any, Optional

import rclpy
from rclpy.serialization import deserialize_message


def resolve_db3_path(bag_path: str) -> str:
    if os.path.isfile(bag_path) and bag_path.endswith(".db3"):
        return bag_path
    if not os.path.isdir(bag_path):
        return ""
    for name in sorted(os.listdir(bag_path)):
        if name.endswith(".db3"):
            return os.path.join(bag_path, name)
    return ""


def ns_to_sec(ns: int) -> float:
    return float(ns) * 1e-9


def safe_decode_qos(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    return str(value)


def import_ros_msg_type(type_str: str) -> Optional[type]:
    """
    Convert ROS interface name 'pkg/msg/Type' to Python message class pkg.msg.Type.
    Returns None if unavailable.
    """
    try:
        pkg, kind, name = type_str.split("/")
        if kind != "msg":
            return None
        mod = importlib.import_module(f"{pkg}.msg")
        return getattr(mod, name)
    except Exception:
        return None


def has_header(msg: Any) -> bool:
    return hasattr(msg, "header") and hasattr(msg.header, "frame_id") and hasattr(msg.header, "stamp")


def get_frame_id(msg: Any) -> Optional[str]:
    if not has_header(msg):
        return None
    try:
        return str(msg.header.frame_id)
    except Exception:
        return None


def is_truthy_string(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


@dataclass
class TopicInfo:
    id: int
    name: str
    type: str
    serialization_format: Optional[str] = None
    offered_qos_profiles: Optional[str] = None


@dataclass
class TopicStats:
    count: int
    t_min_ns: Optional[int]
    t_max_ns: Optional[int]
    duration_sec: Optional[float]


@dataclass
class DeepScanResult:
    unique_frame_ids: list[str] = dataclasses.field(default_factory=list)
    unique_child_frame_ids: list[str] = dataclasses.field(default_factory=list)
    unique_encodings: list[str] = dataclasses.field(default_factory=list)
    unique_formats: list[str] = dataclasses.field(default_factory=list)
    timestamp_monotonic: Optional[bool] = None
    timestamp_duplicate_count: Optional[int] = None
    notes: list[str] = dataclasses.field(default_factory=list)


def topic_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM topics WHERE name = ? LIMIT 1", (name,))
    return cur.fetchone() is not None


def read_topics(cur: sqlite3.Cursor) -> list[TopicInfo]:
    # Schema varies slightly between rosbag2 versions. Discover columns.
    cur.execute("PRAGMA table_info(topics)")
    cols = [r[1] for r in cur.fetchall()]

    select_cols = ["id", "name", "type"]
    if "serialization_format" in cols:
        select_cols.append("serialization_format")
    if "offered_qos_profiles" in cols:
        select_cols.append("offered_qos_profiles")

    cur.execute(f"SELECT {', '.join(select_cols)} FROM topics ORDER BY name")
    rows = cur.fetchall()

    out: list[TopicInfo] = []
    for row in rows:
        base = {"id": int(row[0]), "name": row[1], "type": row[2]}
        i = 3
        serialization_format = None
        offered_qos_profiles = None
        if "serialization_format" in select_cols:
            serialization_format = row[i]
            i += 1
        if "offered_qos_profiles" in select_cols:
            offered_qos_profiles = safe_decode_qos(row[i])
            i += 1
        out.append(
            TopicInfo(
                **base,
                serialization_format=serialization_format,
                offered_qos_profiles=offered_qos_profiles,
            )
        )
    return out


def topic_stats(cur: sqlite3.Cursor, topic_id: int) -> TopicStats:
    cur.execute(
        "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id = ?",
        (topic_id,),
    )
    count, tmin, tmax = cur.fetchone()
    count = int(count)
    tmin_i = int(tmin) if tmin is not None else None
    tmax_i = int(tmax) if tmax is not None else None
    dur = None
    if tmin_i is not None and tmax_i is not None:
        dur = ns_to_sec(tmax_i - tmin_i)
    return TopicStats(count=count, t_min_ns=tmin_i, t_max_ns=tmax_i, duration_sec=dur)


def iter_topic_messages(cur: sqlite3.Cursor, topic_id: int):
    # Streaming iterator, ordered by timestamp.
    for ts, data in cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,),
    ):
        yield int(ts), data


def scan_timestamps(cur: sqlite3.Cursor, topic_id: int) -> tuple[bool, int]:
    prev = None
    dup = 0
    monotonic = True
    for ts, _ in iter_topic_messages(cur, topic_id):
        if prev is not None:
            if ts < prev:
                monotonic = False
            if ts == prev:
                dup += 1
        prev = ts
    return monotonic, dup


def deep_scan_topic(
    cur: sqlite3.Cursor,
    topic: TopicInfo,
    full_scan: bool,
    max_unique: int = 50,
) -> DeepScanResult:
    res = DeepScanResult()
    msg_cls = import_ros_msg_type(topic.type)
    if msg_cls is None:
        res.notes.append("No Python message type available; skipping deep decode.")
        return res

    # Always scan timestamps (cheap).
    mono, dup = scan_timestamps(cur, topic.id)
    res.timestamp_monotonic = mono
    res.timestamp_duplicate_count = dup

    # Only deserialize the whole topic when requested or for critical sensor types.
    if not full_scan:
        res.notes.append("Deep scan skipped (sampling-only mode).")
        return res

    frames: set[str] = set()
    child_frames: set[str] = set()
    encodings: set[str] = set()
    formats: set[str] = set()

    # Full-scan deserialization.
    for _, data in iter_topic_messages(cur, topic.id):
        msg = deserialize_message(data, msg_cls)

        fid = get_frame_id(msg)
        if is_truthy_string(fid):
            frames.add(str(fid))
            if len(frames) > max_unique:
                res.notes.append(f"Frame ID cardinality > {max_unique}; truncating.")
                break

        # Special cases
        if hasattr(msg, "child_frame_id"):
            try:
                cf = str(msg.child_frame_id)
                if is_truthy_string(cf):
                    child_frames.add(cf)
            except Exception:
                pass

        if hasattr(msg, "encoding"):
            try:
                enc = str(msg.encoding)
                if is_truthy_string(enc):
                    encodings.add(enc)
            except Exception:
                pass

        if hasattr(msg, "format"):
            try:
                fmt = str(msg.format)
                if is_truthy_string(fmt):
                    formats.add(fmt)
            except Exception:
                pass

    res.unique_frame_ids = sorted(frames)
    res.unique_child_frame_ids = sorted(child_frames)
    res.unique_encodings = sorted(encodings)
    res.unique_formats = sorted(formats)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Deep rosbag2 sqlite inspector.")
    ap.add_argument(
        "bag_path",
        nargs="?",
        default=os.environ.get("BAG_PATH", ""),
        help="Bag directory containing *.db3 (or a direct *.db3 file).",
    )
    ap.add_argument("--json", dest="json_out", default="", help="Write JSON summary to this path.")
    ap.add_argument(
        "--full-scan-all",
        action="store_true",
        help="Deserialize and scan *all* topics that have Python message types.",
    )
    ap.add_argument(
        "--full-scan-types",
        default="nav_msgs/msg/Odometry,geometry_msgs/msg/PoseStamped,"
        "sensor_msgs/msg/Imu,sensor_msgs/msg/Image,sensor_msgs/msg/CompressedImage,"
        "sensor_msgs/msg/PointCloud2,tf2_msgs/msg/TFMessage,sensor_msgs/msg/LaserScan,"
        "sensor_msgs/msg/CameraInfo,livox_ros_driver2/msg/CustomMsg,livox_ros_driver/msg/CustomMsg",
        help="Comma-separated list of ROS msg types to full-scan (even if --full-scan-all is false).",
    )
    args = ap.parse_args()

    if not args.bag_path:
        print("ERROR: bag_path not provided and BAG_PATH env var not set.", file=sys.stderr)
        return 2

    db_path = resolve_db3_path(args.bag_path)
    if not db_path or not os.path.exists(db_path):
        print(f"ERROR: could not locate *.db3 under '{args.bag_path}'", file=sys.stderr)
        return 2

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Pull topics + global stats
    topics = read_topics(cur)
    stats_by_id: dict[int, TopicStats] = {t.id: topic_stats(cur, t.id) for t in topics}

    bag_start = min((s.t_min_ns for s in stats_by_id.values() if s.t_min_ns is not None), default=None)
    bag_end = max((s.t_max_ns for s in stats_by_id.values() if s.t_max_ns is not None), default=None)

    # Determine which topics to deep scan
    full_scan_types = {s.strip() for s in args.full_scan_types.split(",") if s.strip()}
    deep_topics: list[TopicInfo] = []
    for t in topics:
        if args.full_scan_all or (t.type in full_scan_types):
            deep_topics.append(t)

    rclpy.init()

    deep_results: dict[str, DeepScanResult] = {}
    for t in deep_topics:
        deep_results[t.name] = deep_scan_topic(cur, t, full_scan=True)

    rclpy.shutdown()

    # Warnings / actionable findings
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    # TF availability
    has_tf = any(t.name == "/tf" for t in topics)
    has_tf_static = any(t.name == "/tf_static" for t in topics)
    if not has_tf and not has_tf_static:
        warn("No /tf or /tf_static in bag. Any extrinsics must be declared out-of-band (static params/URDF).")

    # CameraInfo availability
    has_caminfo = any(t.type == "sensor_msgs/msg/CameraInfo" for t in topics)
    if not has_caminfo:
        warn("No CameraInfo topics found. Intrinsics must be provided via parameters and logged.")

    # Core sensor presence heuristics (non-fatal, but useful)
    if not any(t.type == "nav_msgs/msg/Odometry" for t in topics):
        warn("No nav_msgs/Odometry topics found (odometry missing).")
    if not any(t.type == "sensor_msgs/msg/Imu" for t in topics):
        warn("No sensor_msgs/Imu topics found (IMU missing).")
    if not any(t.type in ("sensor_msgs/msg/PointCloud2", "sensor_msgs/msg/LaserScan") for t in topics):
        warn("No PointCloud2 or LaserScan topics found (no primary geometry sensor).")

    # Frame inconsistency checks for scanned topics
    for topic_name, scan in deep_results.items():
        if scan.unique_frame_ids and len(scan.unique_frame_ids) > 1:
            warn(f"Topic '{topic_name}' has multiple frame_ids: {scan.unique_frame_ids}")
        if scan.unique_child_frame_ids and len(scan.unique_child_frame_ids) > 1:
            warn(f"Topic '{topic_name}' has multiple child_frame_ids: {scan.unique_child_frame_ids}")
        if scan.unique_encodings and len(scan.unique_encodings) > 1:
            warn(f"Topic '{topic_name}' has multiple encodings: {scan.unique_encodings}")
        if scan.unique_formats and len(scan.unique_formats) > 1:
            warn(f"Topic '{topic_name}' has multiple CompressedImage formats: {scan.unique_formats}")
        if scan.timestamp_monotonic is False:
            warn(f"Topic '{topic_name}' has non-monotonic timestamps.")
        if scan.timestamp_duplicate_count and scan.timestamp_duplicate_count > 0:
            warn(f"Topic '{topic_name}' has {scan.timestamp_duplicate_count} duplicate timestamps.")

    # Print human report
    print("=== Rosbag Deep Inspection Report ===")
    print(f"bag_path: {args.bag_path}")
    print(f"db_path:  {db_path}")
    if bag_start is not None and bag_end is not None:
        print(f"bag_start_sec: {ns_to_sec(bag_start):.6f}")
        print(f"bag_end_sec:   {ns_to_sec(bag_end):.6f}")
        print(f"bag_duration_sec: {ns_to_sec(bag_end - bag_start):.3f}")
    print(f"topics: {len(topics)}")
    print()

    print("=== Topics (name | type | count | duration) ===")
    for t in topics:
        s = stats_by_id[t.id]
        dur = f"{s.duration_sec:.3f}s" if s.duration_sec is not None else "n/a"
        print(f"- {t.name} | {t.type} | {s.count} | {dur}")
    print()

    if deep_results:
        print("=== Deep Scan (frames/encodings/formats) ===")
        for t in deep_topics:
            scan = deep_results.get(t.name)
            if scan is None:
                continue
            print(f"- {t.name} [{t.type}]")
            if scan.unique_frame_ids:
                print(f"    frame_ids: {scan.unique_frame_ids}")
            if scan.unique_child_frame_ids:
                print(f"    child_frame_ids: {scan.unique_child_frame_ids}")
            if scan.unique_encodings:
                print(f"    encodings: {scan.unique_encodings}")
            if scan.unique_formats:
                print(f"    formats: {scan.unique_formats}")
            if scan.timestamp_monotonic is not None:
                print(f"    timestamps_monotonic: {scan.timestamp_monotonic}")
            if scan.timestamp_duplicate_count is not None:
                print(f"    timestamps_duplicates: {scan.timestamp_duplicate_count}")
            for note in scan.notes:
                print(f"    note: {note}")
        print()

    print("=== Warnings (actionable) ===")
    if warnings:
        for w in warnings:
            print(f"- {w}")
    else:
        print("(none)")
    print()

    summary = {
        "bag_path": args.bag_path,
        "db_path": db_path,
        "bag_start_ns": bag_start,
        "bag_end_ns": bag_end,
        "topics": [dataclasses.asdict(t) for t in topics],
        "topic_stats": {t.name: dataclasses.asdict(stats_by_id[t.id]) for t in topics},
        "deep_scan": {name: dataclasses.asdict(scan) for name, scan in deep_results.items()},
        "warnings": list(warnings),
    }

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Wrote JSON summary to: {args.json_out}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

