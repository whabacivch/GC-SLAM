#!/usr/bin/env bash
# Run GC v2 on a Kimera 10_14 bag with correct topics/extrinsics pulled from dataset_ready_manifest.yaml.
# Usage: ./tools/run_gc_kimera.sh [--robot <name>] [--manifest <path>] [--bag_duration <seconds>] [--bag_play_rate <rate>]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MANIFEST="$PROJECT_ROOT/rosbags/Kimera_Data/dataset_ready_manifest.yaml"
ROBOT="acl_jackal"
MANIFEST="$DEFAULT_MANIFEST"
BAG_DURATION="60"
BAG_PLAY_RATE="0.25"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot) ROBOT="$2"; shift 2;;
    --manifest) MANIFEST="$2"; shift 2;;
    --bag_duration) BAG_DURATION="$2"; shift 2;;
    --bag_play_rate) BAG_PLAY_RATE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

python3 - <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import yaml

manifest_path = Path(sys.argv[1])
robot = sys.argv[2]

with open(manifest_path, "r") as f:
    data = yaml.safe_load(f)

entries = [e for e in data.get("robots", []) if e.get("robot") == robot]
if not entries:
    sys.exit(f"No entry for robot '{robot}' in {manifest_path}")
entry = entries[0]

extr_path = Path(entry["extrinsics"])
if not extr_path.exists():
    sys.exit(f"Extrinsics file missing: {extr_path}")
extr = yaml.safe_load(extr_path.read_text())
Tdict = {t["name"]: np.array(t["T"], dtype=float) for t in extr.get("transforms", []) if "name" in t and "T" in t}

def mat_to_rt(mat: np.ndarray):
    from scipy.spatial.transform import Rotation as R
    t = mat[:3, 3].tolist()
    rotvec = R.from_matrix(mat[:3, :3]).as_rotvec().tolist()
    return t, rotvec

if "T_baselink_lidar" not in Tdict:
    sys.exit("T_baselink_lidar not found in extrinsics.")
t_lid, r_lid = mat_to_rt(Tdict["T_baselink_lidar"])

if "T_baselink_cameralink" in Tdict and "T_cameralink_gyro" in Tdict:
    T_base_imu = Tdict["T_baselink_cameralink"] @ Tdict["T_cameralink_gyro"]
    t_imu, r_imu = mat_to_rt(T_base_imu)
else:
    t_imu, r_imu = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

bag_path = Path(entry["ros2_bag"])
gt_tum = Path(entry.get("ground_truth_tum", ""))

topic_prefix = f"/{robot}"
imu_topic = f"{topic_prefix}/forward/imu"
lidar_topic = f"{topic_prefix}/lidar_points"
odom_topic = f"{topic_prefix}/jackal_velocity_controller/odom"

out = {
    "BAG": str(bag_path),
    "GT_TUM": str(gt_tum),
    "IMU_TOPIC": imu_topic,
    "LIDAR_TOPIC": lidar_topic,
    "ODOM_TOPIC": odom_topic,
    "T_BASE_LIDAR": json.dumps(t_lid + r_lid),
    "T_BASE_IMU": json.dumps(t_imu + r_imu),
}
print("\n".join(f"{k}={v}" for k, v in out.items()))
PY "$MANIFEST" "$ROBOT" >/tmp/gc_kimera_env.$$
source /tmp/gc_kimera_env.$$ && rm -f /tmp/gc_kimera_env.$$

echo "[GC Kimera] robot=$ROBOT bag=$BAG"
source "$PROJECT_ROOT/tools/common_venv.sh"

ros2 launch fl_slam_poc gc_rosbag.launch.py \
  bag:="$BAG" \
  bag_duration:="$BAG_DURATION" \
  bag_play_rate:="$BAG_PLAY_RATE" \
  config_path:="$PROJECT_ROOT/fl_ws/src/fl_slam_poc/config/gc_unified.yaml" \
  lidar_topic:="$LIDAR_TOPIC" \
  odom_topic:="$ODOM_TOPIC" \
  imu_topic:="$IMU_TOPIC" \
  base_frame:=base_link \
  T_base_lidar:="$T_BASE_LIDAR" \
  T_base_imu:="$T_BASE_IMU"
