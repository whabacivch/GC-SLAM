#!/usr/bin/env python3
"""
Inspect odom pose covariance in a rosbag: fixed or varies per message.

Reads the first N odom messages and reports whether msg.pose.covariance
is identical for every message (FIXED) or changes (VARIES). Optionally
dumps covariance rows to CSV for inspection.

Usage:
  .venv/bin/python tools/inspect_odom_covariance.py /path/to/bag_dir
  .venv/bin/python tools/inspect_odom_covariance.py /path/to/bag.db3 --max 500 --out-dir /tmp --dump-csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
from tools.rosbag_sqlite_utils import resolve_db3_path, topic_id


def ns_to_sec(ns: int) -> float:
    return float(ns) * 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inspect odom pose covariance: fixed or varies across messages."
    )
    ap.add_argument(
        "bag_path",
        nargs="?",
        default=os.environ.get("BAG_PATH", "rosbags/Kimera_Data/ros2/10_14_acl_jackal-005"),
        help="Bag directory or .db3 path",
    )
    ap.add_argument("--odom-topic", default="/odom", help="Odom topic")
    ap.add_argument("--max", type=int, default=500, help="Max odom messages to sample")
    ap.add_argument("--out-dir", default=None, help="Output directory for CSV (default: same as bag or cwd)")
    ap.add_argument("--dump-csv", action="store_true", help="Write CSV with stamp and 36 cov values")
    args = ap.parse_args()

    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        print(f"ERROR: No .db3 found in {args.bag_path}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or os.path.dirname(db_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    conn = __import__("sqlite3").connect(db_path)
    cur = conn.cursor()
    tid = topic_id(cur, args.odom_topic)
    if tid is None:
        print(f"ERROR: Odom topic '{args.odom_topic}' not found.", file=sys.stderr)
        conn.close()
        return 1

    rows = cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
        (tid, args.max),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No odom messages on '{args.odom_topic}'.", file=sys.stderr)
        return 1

    if not rclpy.ok():
        rclpy.init()

    covs: list[tuple[float, tuple[float, ...]]] = []
    for ts_ns, data in rows:
        msg = deserialize_message(data, Odometry)
        stamp = ns_to_sec(ts_ns)
        cov_flat = tuple(float(x) for x in msg.pose.covariance)
        covs.append((stamp, cov_flat))

    first = covs[0][1]
    all_same = all(c == first for _, c in covs)
    if all_same:
        print("RESULT: pose covariance is FIXED (identical for all sampled messages)")
        print(f"  Sampled {len(covs)} messages. First 6 diagonal elements (x,y,z,roll,pitch,yaw):")
        # ROS order row-major [x,y,z,roll,pitch,yaw]
        for i in (0, 7, 14, 21, 28, 35):
            print(f"    [{i//6}] = {first[i]}")
    else:
        print("RESULT: pose covariance VARIES across messages")
        for i, (stamp, c) in enumerate(covs):
            if c != first:
                print(f"  First differing message: index {i}, stamp={stamp:.6f}")
                print(f"  First 6 diagonal (reference): {[first[j] for j in (0,7,14,21,28,35)]}")
                print(f"  First 6 diagonal (this msg): {[c[j] for j in (0,7,14,21,28,35)]}")
                break

    if args.dump_csv:
        csv_path = os.path.join(out_dir, "odom_pose_covariance_first_{}.csv".format(len(covs)))
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stamp_sec"] + [f"cov_{i}" for i in range(36)])
            for stamp, c in covs:
                w.writerow([f"{stamp:.9f}"] + list(c))
        print(f"Wrote {csv_path}")

    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
