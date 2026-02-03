#!/usr/bin/env python3
"""
Dump raw IMU and odom messages from a rosbag for inspection (e.g. first 300).

Writes two CSV files: imu_raw_*.csv and odom_raw_*.csv.
Useful for checking tangent frame, gravity alignment, and covariance structure.

Usage:
  .venv/bin/python tools/dump_raw_imu_odom.py /path/to/bag_dir
  .venv/bin/python tools/dump_raw_imu_odom.py /path/to/bag_dir --max-imu 300 --max-odom 300 --out-dir /tmp
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
from tools.rosbag_sqlite_utils import resolve_db3_path, topic_id


def ns_to_sec(ns: int) -> float:
    return float(ns) * 1e-9


def dump_imu(cur, topic: str, max_messages: int, out_path: str) -> int:
    tid = topic_id(cur, topic)
    if tid is None:
        print(f"  IMU topic '{topic}' not found.", file=sys.stderr)
        return 0
    rows = cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
        (tid, max_messages),
    ).fetchall()
    if not rows:
        print(f"  No IMU messages on '{topic}'.", file=sys.stderr)
        return 0
    rclpy.init()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp_sec", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"])
        for ts_ns, data in rows:
            msg = deserialize_message(data, Imu)
            stamp = ns_to_sec(ts_ns)
            w.writerow([
                f"{stamp:.9f}",
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            ])
    return len(rows)


def dump_odom(cur, topic: str, max_messages: int, out_path: str) -> int:
    tid = topic_id(cur, topic)
    if tid is None:
        print(f"  Odom topic '{topic}' not found.", file=sys.stderr)
        return 0
    rows = cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
        (tid, max_messages),
    ).fetchall()
    if not rows:
        print(f"  No odom messages on '{topic}'.", file=sys.stderr)
        return 0
    if not rclpy.ok():
        rclpy.init()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "stamp_sec", "x", "y", "z",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz", "wx", "wy", "wz",
        ])
        for ts_ns, data in rows:
            msg = deserialize_message(data, Odometry)
            stamp = ns_to_sec(ts_ns)
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            t = msg.twist.twist
            w.writerow([
                f"{stamp:.9f}",
                p.x, p.y, p.z,
                o.x, o.y, o.z, o.w,
                t.linear.x, t.linear.y, t.linear.z,
                t.angular.x, t.angular.y, t.angular.z,
            ])
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump raw IMU and odom from rosbag (first N messages).")
    ap.add_argument("bag_path", nargs="?", default=os.environ.get("BAG_PATH", "rosbags/Kimera_Data/ros2/10_14_acl_jackal-005"), help="Bag directory or .db3 path")
    ap.add_argument("--imu-topic", default="/livox/mid360/imu", help="IMU topic (raw bag topic)")
    ap.add_argument("--odom-topic", default="/odom", help="Odom topic")
    ap.add_argument("--max-imu", type=int, default=300, help="Max IMU messages to dump")
    ap.add_argument("--max-odom", type=int, default=300, help="Max odom messages to dump")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: same as bag or cwd)")
    args = ap.parse_args()

    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        print(f"ERROR: No .db3 found in {args.bag_path}", file=sys.stderr)
        return 1
    out_dir = args.out_dir or os.path.dirname(db_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    imu_csv = os.path.join(out_dir, f"imu_raw_first_{args.max_imu}.csv")
    odom_csv = os.path.join(out_dir, f"odom_raw_first_{args.max_odom}.csv")

    conn = __import__("sqlite3").connect(db_path)
    cur = conn.cursor()

    n_imu = dump_imu(cur, args.imu_topic, args.max_imu, imu_csv)
    n_odom = dump_odom(cur, args.odom_topic, args.max_odom, odom_csv)
    conn.close()

    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass

    print(f"IMU:  {n_imu} rows -> {imu_csv}")
    print(f"Odom: {n_odom} rows -> {odom_csv}")
    return 0 if (n_imu or n_odom) else 1


if __name__ == "__main__":
    sys.exit(main())
