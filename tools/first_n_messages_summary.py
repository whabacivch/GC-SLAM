#!/usr/bin/env python3
"""
Emit "first N messages" summary for PointCloud2, Imu, Odometry (or given topics).

Output: markdown or JSON per bag. Use to check field names, frame_id, and sample
values across bags (e.g. Kimera vs M3DGR). See docs/POINTCLOUD2_LAYOUTS.md.

Usage:
  python tools/first_n_messages_summary.py /path/to/bag_dir
  python tools/first_n_messages_summary.py /path/to/bag_dir --n 5 --json /tmp/summary.json
  python tools/first_n_messages_summary.py /path/to/bag_dir --topics /odom /gc/sensors/imu
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rosbag_sqlite_utils import resolve_db3_path, topic_id, topic_type


def _sample_pointcloud2(msg) -> dict:
    """Extract field names, frame_id, and sample from first PointCloud2."""
    out = {
        "frame_id": getattr(msg.header, "frame_id", None),
        "stamp_sec": getattr(msg.header.stamp, "sec", None),
        "stamp_nanosec": getattr(msg.header.stamp, "nanosec", None),
        "width": msg.width,
        "height": msg.height,
        "point_step": msg.point_step,
        "row_step": msg.row_step,
        "fields": [{"name": f.name, "offset": f.offset, "datatype": f.datatype} for f in msg.fields],
    }
    return out


def _sample_imu(msg) -> dict:
    """Extract frame_id and sample from first Imu."""
    out = {
        "frame_id": getattr(msg.header, "frame_id", None),
        "orientation": [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
        "angular_velocity": [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
        "linear_acceleration": [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
        "orientation_covariance": list(msg.orientation_covariance) if hasattr(msg, "orientation_covariance") else [],
        "angular_velocity_covariance": list(msg.angular_velocity_covariance) if hasattr(msg, "angular_velocity_covariance") else [],
        "linear_acceleration_covariance": list(msg.linear_acceleration_covariance) if hasattr(msg, "linear_acceleration_covariance") else [],
    }
    return out


def _sample_odometry(msg) -> dict:
    """Extract frame_id, child_frame_id and sample from first Odometry."""
    out = {
        "frame_id": getattr(msg.header, "frame_id", None),
        "child_frame_id": getattr(msg, "child_frame_id", None),
        "pose_position": [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
        "pose_orientation": [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],
        "twist_linear": [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
        "twist_angular": [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z],
        "pose_covariance_diag": list(msg.pose.covariance)[::7] if len(msg.pose.covariance) >= 36 else [],
        "twist_covariance_diag": list(msg.twist.covariance)[::7] if len(msg.twist.covariance) >= 36 else [],
    }
    return out


def import_ros_msg_type(type_str: str):
    """Return Python message class for 'pkg/msg/Type' or None."""
    try:
        pkg, kind, name = type_str.split("/")
        if kind != "msg":
            return None
        mod = __import__(f"{pkg}.msg", fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="First N messages summary for key topics")
    ap.add_argument("bag_path", help="Bag directory or .db3 path")
    ap.add_argument("--n", type=int, default=25, help="Number of messages to sample per topic (default 25)")
    ap.add_argument("--topics", nargs="*", default=None, help="Topic names; default: auto-detect PointCloud2, Imu, Odometry")
    ap.add_argument("--json", default="", help="If set, write JSON to this path")
    ap.add_argument("--md", default="", help="If set, write markdown to this path")
    args = ap.parse_args()

    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        print(f"ERROR: No .db3 found under {args.bag_path}", file=sys.stderr)
        return 1

    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Resolve topics: if not given, find first of each type
    if args.topics:
        topic_names = list(args.topics)
    else:
        cur.execute("SELECT name, type FROM topics ORDER BY name")
        rows = cur.fetchall()
        topic_names = []
        seen = set()
        for name, t in rows:
            if "PointCloud2" in (t or "") and "pointcloud2" not in seen:
                topic_names.append(name)
                seen.add("pointcloud2")
            if "Imu" in (t or "") and "imu" not in seen:
                topic_names.append(name)
                seen.add("imu")
            if "Odometry" in (t or "") and "odom" not in seen:
                topic_names.append(name)
                seen.add("odom")
        if not topic_names:
            topic_names = [r[0] for r in rows[:10]]

    try:
        import rclpy
        from rclpy.serialization import deserialize_message
        rclpy.init()
    except Exception as e:
        print(f"ERROR: rclpy init: {e}", file=sys.stderr)
        return 1

    result = {"bag": db_path, "n": args.n, "topics": {}}

    for topic_name in topic_names:
        tid = topic_id(cur, topic_name)
        if tid is None:
            result["topics"][topic_name] = {"error": "topic not found"}
            continue
        ttype = topic_type(cur, topic_name)
        msg_cls = import_ros_msg_type(ttype) if ttype else None
        if msg_cls is None:
            result["topics"][topic_name] = {"type": ttype, "error": "message type not loadable"}
            continue

        samples = []
        for row in cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
            (tid, args.n),
        ):
            ts, data = row
            try:
                msg = deserialize_message(data, msg_cls)
                if len(samples) == 0:
                    if "PointCloud2" in (ttype or ""):
                        samples.append(_sample_pointcloud2(msg))
                    elif "Imu" in (ttype or ""):
                        samples.append(_sample_imu(msg))
                    elif "Odometry" in (ttype or ""):
                        samples.append(_sample_odometry(msg))
                    else:
                        samples.append({"frame_id": getattr(msg.header, "frame_id", None)})
            except Exception as e:
                samples.append({"deserialize_error": str(e)})
        result["topics"][topic_name] = {"type": ttype, "first_message": samples[0] if samples else None, "count": len(samples)}

    rclpy.shutdown()
    conn.close()

    # Output
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.json}")

    if args.md or not args.json:
        lines = [f"# First {args.n} messages summary", f"Bag: {db_path}", ""]
        for topic_name, data in result["topics"].items():
            lines.append(f"## {topic_name}")
            lines.append(f"Type: {data.get('type', '')}")
            if "error" in data:
                lines.append(f"Error: {data['error']}")
            elif data.get("first_message"):
                lines.append("```json")
                lines.append(json.dumps(data["first_message"], indent=2))
                lines.append("```")
            lines.append("")
        md_text = "\n".join(lines)
        if args.md:
            with open(args.md, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"Wrote {args.md}")
        else:
            print(md_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
