#!/usr/bin/env python3
"""
Diagnose "all motion in Z" vs "motion in XY" by comparing bag odom to exported TUM.

If the bag's odometry has motion mainly in X,Y (planar) but the exported trajectory
has motion mainly in Z, there is an axis or frame convention mismatch in the pipeline.

Usage (canonical bag = same as run_and_evaluate_gc.sh):
  python tools/diagnose_trajectory_axes.py rosbags/Kimera_Data/ros2/10_14_acl_jackal-005
  python tools/diagnose_trajectory_axes.py rosbags/Kimera_Data/ros2/10_14_acl_jackal-005 /tmp/gc_slam_trajectory.tum
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rosbag_sqlite_utils import resolve_db3_path, topic_id


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare odom axes vs TUM trajectory axes")
    ap.add_argument("bag_path", help="ROS2 bag directory or db3 file")
    ap.add_argument("tum_path", nargs="?", default=None, help="Exported TUM trajectory (optional)")
    ap.add_argument("--odom-topic", default="/acl_jackal/jackal_velocity_controller/odom", help="Odometry topic")
    ap.add_argument("--max-odom", type=int, default=500, help="Max odom messages to sample")
    args = ap.parse_args()

    db_path = resolve_db3_path(args.bag_path)
    if not db_path or not Path(db_path).is_file():
        print(f"ERROR: Bag not found: {args.bag_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    odom_tid = topic_id(cur, args.odom_topic)
    if odom_tid is None:
        print(f"ERROR: Topic not found: {args.odom_topic}", file=sys.stderr)
        conn.close()
        return 1

    try:
        from rclpy.serialization import deserialize_message
        from nav_msgs.msg import Odometry
    except ImportError as e:
        print(f"ERROR: Need rclpy and nav_msgs: {e}", file=sys.stderr)
        conn.close()
        return 1

    odom_xyzs = []
    for row in cur.execute(
        "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
        (odom_tid, args.max_odom),
    ):
        (data,) = row
        try:
            msg = deserialize_message(data, Odometry)
            p = msg.pose.pose.position
            odom_xyzs.append((float(p.x), float(p.y), float(p.z)))
        except Exception:
            continue
    conn.close()

    if not odom_xyzs:
        print(f"ERROR: No odom messages on {args.odom_topic}", file=sys.stderr)
        return 1

    odom_xyzs = list(zip(*odom_xyzs))  # [(x...), (y...), (z...)]
    odom_range_x = max(odom_xyzs[0]) - min(odom_xyzs[0])
    odom_range_y = max(odom_xyzs[1]) - min(odom_xyzs[1])
    odom_range_z = max(odom_xyzs[2]) - min(odom_xyzs[2])

    print("Odom (from bag) position ranges (m):")
    print(f"  X: [{min(odom_xyzs[0]):.3f}, {max(odom_xyzs[0]):.3f}]  range = {odom_range_x:.3f}")
    print(f"  Y: [{min(odom_xyzs[1]):.3f}, {max(odom_xyzs[1]):.3f}]  range = {odom_range_y:.3f}")
    print(f"  Z: [{min(odom_xyzs[2]):.3f}, {max(odom_xyzs[2]):.3f}]  range = {odom_range_z:.3f}")
    odom_main = "XY" if odom_range_z < 0.5 * max(odom_range_x, odom_range_y) else ("Z" if odom_range_z > max(odom_range_x, odom_range_y) else "mixed")
    print(f"  → Motion mainly in: {odom_main}")

    if args.tum_path and Path(args.tum_path).is_file():
        tum_xyzs = []
        with open(args.tum_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    tum_xyzs.append((float(parts[1]), float(parts[2]), float(parts[3])))
        if tum_xyzs:
            tum_xyzs = list(zip(*tum_xyzs))
            tum_range_x = max(tum_xyzs[0]) - min(tum_xyzs[0])
            tum_range_y = max(tum_xyzs[1]) - min(tum_xyzs[1])
            tum_range_z = max(tum_xyzs[2]) - min(tum_xyzs[2])
            print("")
            print("TUM (exported trajectory) position ranges (m):")
            print(f"  X: [{min(tum_xyzs[0]):.3f}, {max(tum_xyzs[0]):.3f}]  range = {tum_range_x:.3f}")
            print(f"  Y: [{min(tum_xyzs[1]):.3f}, {max(tum_xyzs[1]):.3f}]  range = {tum_range_y:.3f}")
            print(f"  Z: [{min(tum_xyzs[2]):.3f}, {max(tum_xyzs[2]):.3f}]  range = {tum_range_z:.3f}")
            tum_main = "Z" if tum_range_z > max(tum_range_x, tum_range_y) else ("XY" if max(tum_range_x, tum_range_y) > tum_range_z else "mixed")
            print(f"  → Motion mainly in: {tum_main}")
            if odom_main == "XY" and tum_main == "Z":
                print("")
                print("MISMATCH: Odom has motion in XY but exported trajectory has motion in Z.")
                print("This usually means an axis or frame convention error (e.g. odom frame")
                print("uses different axes than assumed, or state/export swaps X/Y with Z).")
                print("See docs/FRAME_AND_QUATERNION_CONVENTIONS.md and docs/KIMERA_DATASET_AND_PIPELINE.md.")
            elif odom_main == "Z" and tum_main == "XY":
                print("")
                print("MISMATCH: Odom has motion in Z but exported trajectory has motion in XY.")
                print("Check whether the bag's odom frame uses Z as forward; you may need to")
                print("remap odom pose (e.g. swap axes) when ingesting.")
        else:
            print(f"No valid lines in TUM file: {args.tum_path}")
    else:
        print("")
        print("No TUM file provided. Run again with a trajectory file to compare axes:")
        print(f"  python {__file__} {args.bag_path} /tmp/gc_slam_trajectory.tum")

    return 0


if __name__ == "__main__":
    sys.exit(main())
