#!/usr/bin/env bash
# Download NVIDIA Isaac ROS2 Benchmark (r2b) Dataset
#
# This script downloads the r2b_storage dataset from NVIDIA NGC,
# which contains RealSense D455 data suitable for 3D FL-SLAM testing.
#
# Requirements:
#   - wget or curl
#   - ~5GB disk space
#
# Usage:
#   ./tools/download_r2b_dataset.sh
#   DEST_DIR=/path/to/dest ./tools/download_r2b_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST_DIR="${DEST_DIR:-${PROJECT_DIR}/rosbags}"
R2B_DIR_NAME="r2b_storage"

# Dataset info
# Source: https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark
# The r2b_storage dataset contains:
# - RealSense D455 RGB-D data (/camera/color/image_raw, /camera/depth/image_raw)
# - Point clouds (/camera/depth/points)
# - Camera info (/camera/color/camera_info, /camera/depth/camera_info)
# - Odometry (may vary)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  tools/download_r2b_dataset.sh

Environment:
  DEST_DIR=/path/to/rosbags   (default: <project>/rosbags)

Notes:
  - Downloads the NVIDIA Isaac ROS2 Benchmark (r2b) dataset
  - Contains RealSense D455 RGB-D and point cloud data
  - Suitable for 3D FL-SLAM testing with GPU acceleration

Dataset Contents:
  - /camera/color/image_raw      (RGB images)
  - /camera/depth/image_raw      (Depth images)
  - /camera/depth/points         (PointCloud2)
  - /camera/color/camera_info    (Camera intrinsics)

To use with FL-SLAM (note: may require additional configuration):
  ros2 launch fl_slam_poc gc_rosbag.launch.py \
    bag:=${DEST_DIR}/r2b_storage \
    play_bag:=true \
    pointcloud_topic:=/camera/depth/points \
    enable_livox_convert:=false
USAGE
  exit 0
fi

echo "============================================"
echo "NVIDIA r2b Dataset Downloader"
echo "============================================"
echo ""
echo "This will download the r2b_storage dataset (~2GB)"
echo "Destination: ${DEST_DIR}/${R2B_DIR_NAME}"
echo ""

mkdir -p "${DEST_DIR}"

if [[ -d "${DEST_DIR}/${R2B_DIR_NAME}" ]]; then
  echo "OK: r2b dataset already present at ${DEST_DIR}/${R2B_DIR_NAME}"
  echo ""
  echo "To re-download, remove the directory first:"
  echo "  rm -rf ${DEST_DIR}/${R2B_DIR_NAME}"
  exit 0
fi

# Create temp directory for download
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

echo "Downloading r2b dataset..."
echo ""

# The r2b dataset is available from NVIDIA NGC
# There are multiple ways to download:

# Option 1: Direct download (if available)
# The dataset is typically hosted on NGC with public access
R2B_URL="https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2023/versions/2/files/r2b_storage.tar.gz"

# Check if wget is available
if command -v wget &> /dev/null; then
  echo "Using wget..."
  wget --progress=bar:force -O "${TEMP_DIR}/r2b_storage.tar.gz" "${R2B_URL}" 2>&1 || {
    echo ""
    echo "Direct download failed. Trying alternative methods..."
    # Fallback: provide instructions for manual download
    cat <<'MANUAL'

========================================
Manual Download Instructions
========================================

The r2b dataset may require manual download from NVIDIA NGC.

1. Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/resources/r2bdataset2023

2. Download the r2b_storage sequence

3. Extract to: ${DEST_DIR}/r2b_storage

Alternatively, you can use the NGC CLI:
  ngc registry resource download-version "nvidia/isaac/r2bdataset2023:2" \
    --dest ${DEST_DIR}

For testing without the r2b dataset, you can:
- Use Gazebo simulation with TurtleBot3
- Use any rosbag with PointCloud2 data

MANUAL
    exit 1
  }
elif command -v curl &> /dev/null; then
  echo "Using curl..."
  curl -L --progress-bar -o "${TEMP_DIR}/r2b_storage.tar.gz" "${R2B_URL}" || {
    echo ""
    echo "Download failed. See manual instructions above."
    exit 1
  }
else
  echo "ERROR: Neither wget nor curl found. Please install one of them."
  exit 1
fi

echo ""
echo "Extracting dataset..."
tar -xzf "${TEMP_DIR}/r2b_storage.tar.gz" -C "${DEST_DIR}"

# Verify extraction
if [[ -d "${DEST_DIR}/${R2B_DIR_NAME}" ]]; then
  echo ""
  echo "============================================"
  echo "Download Complete!"
  echo "============================================"
  echo ""
  echo "Dataset location: ${DEST_DIR}/${R2B_DIR_NAME}"
  echo ""
  echo "To inspect the bag:"
  echo "  ros2 bag info ${DEST_DIR}/${R2B_DIR_NAME}"
  echo ""
  echo "To test with FL-SLAM (3D mode):"
  echo "  ros2 launch fl_slam_poc gc_rosbag.launch.py \\"
  echo "    bag:=${DEST_DIR}/${R2B_DIR_NAME} \\"
  echo "    play_bag:=true \\"
  echo "    enable_livox_convert:=false"
  echo ""
else
  echo "ERROR: Extraction failed or dataset not in expected location."
  exit 1
fi
