#!/usr/bin/env python3
"""
Convert Kimera-Multi-Data ground truth CSV to TUM format.

GT CSV format: #timestamp_kf,x,y,z,qw,qx,qy,qz (timestamp in nanoseconds).
TUM format:    timestamp x y z qx qy qz qw (timestamp in seconds).

Usage:
  python tools/kimera_gt_to_tum.py <input_csv> <output.tum>
  python tools/kimera_gt_to_tum.py ground_truth/1014/acl_jackal_gt_odom.csv ground_truth/1014/acl_jackal_gt.tum
"""

import csv
import sys


def convert_csv_to_tum(csv_path: str, tum_path: str) -> bool:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            # timestamp_kf (ns), x, y, z, qw, qx, qy, qz
            if len(row) < 8:
                continue
            ts_ns = int(row[0])
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            ts_sec = ts_ns * 1e-9
            rows.append((ts_sec, x, y, z, qx, qy, qz, qw))

    if not rows:
        print(f"No data rows in {csv_path}")
        return False

    with open(tum_path, "w", encoding="utf-8") as f:
        f.write("# timestamp x y z qx qy qz qw\n")
        for r in rows:
            f.write(f"{r[0]:.9f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {r[5]:.6f} {r[6]:.6f} {r[7]:.6f}\n")

    print(f"Wrote {len(rows)} poses to {tum_path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: kimera_gt_to_tum.py <input_csv> <output.tum>")
        sys.exit(1)
    ok = convert_csv_to_tum(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
