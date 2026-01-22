#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST_DIR="${DEST_DIR:-${PROJECT_DIR}/rosbags}"
ROS1_BAG_NAME="tb3_slam3d_small_ros1.bag"
ROS2_DIR_NAME="tb3_slam3d_small_ros2"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  tools/download_tb3_rosbag.sh

Environment:
  DEST_DIR=/path/to/rosbags   (default: <project>/rosbags)

Notes:
  - Downloads the TurtleBot3 SLAM3D sample rosbag from ROBOTIS Japan.
  - Converts the ROS1 bag to a ROS2 bag directory (requires ros2 bag + rosbag2_bag_v2).
USAGE
  exit 0
fi

mkdir -p "${DEST_DIR}"

if [[ -d "${DEST_DIR}/${ROS2_DIR_NAME}" ]]; then
  echo "OK: ROS2 bag already present at ${DEST_DIR}/${ROS2_DIR_NAME}"
  exit 0
fi

if [[ ! -f "${DEST_DIR}/${ROS1_BAG_NAME}" ]]; then
  echo "Downloading ROS1 bag to ${DEST_DIR}/${ROS1_BAG_NAME} ..."
  # Source: https://github.com/ROBOTIS-JAPAN-GIT/turtlebot3_slam_3d
  curl -L \
    -o "${DEST_DIR}/${ROS1_BAG_NAME}" \
    "https://github.com/ROBOTIS-JAPAN-GIT/turtlebot3_slam_3d/raw/main/rosbag/tb3_slam3d_small.bag"
fi

echo "Converting ROS1 bag -> ROS2 bag directory..."
set +u
source /opt/ros/jazzy/setup.bash
set -u

if ! ros2 pkg prefix rosbag2_bag_v2 >/dev/null 2>&1; then
  echo "ERROR: missing rosbag2 v2 plugin (rosbag2_bag_v2). Install it, e.g.:" >&2
  echo "  sudo apt-get update && sudo apt-get install -y ros-jazzy-rosbag2-bag-v2" >&2
  exit 2
fi

rm -rf "${DEST_DIR:?}/${ROS2_DIR_NAME}"
ros2 bag convert \
  -i "${DEST_DIR}/${ROS1_BAG_NAME}" \
  -o "${DEST_DIR}/${ROS2_DIR_NAME}" \
  -s rosbag_v2 \
  -d sqlite3

echo "OK: converted bag at ${DEST_DIR}/${ROS2_DIR_NAME}"
