#!/usr/bin/env bash
# Inspect a rosbag and suggest FL-SLAM topic configuration
#
# Usage:
#   ./tools/inspect_rosbag_topics.sh /path/to/rosbag

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path-to-rosbag>"
    exit 1
fi

BAG_PATH="$1"

if [[ ! -d "$BAG_PATH" && ! -f "$BAG_PATH" ]]; then
    echo "ERROR: Rosbag not found at $BAG_PATH"
    exit 1
fi

# Source ROS
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
fi

echo "============================================"
echo "Rosbag Topic Inspector for FL-SLAM"
echo "============================================"
echo ""
echo "Inspecting: $BAG_PATH"
echo ""

# Get bag info
ros2 bag info "$BAG_PATH"

echo ""
echo "============================================"
echo "Topic Analysis for FL-SLAM Configuration"
echo "============================================"
echo ""

# Get all topics
TOPICS=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep "Topic:" | awk '{print $2}')

# Find point cloud topics
echo "Point Cloud Topics (PointCloud2):"
PC_TOPICS=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep -E "sensor_msgs/msg/PointCloud2" | awk '{print $2}' || echo "")
if [[ -n "$PC_TOPICS" ]]; then
    echo "$PC_TOPICS" | while read topic; do
        echo "  - $topic"
    done
    SUGGESTED_PC=$(echo "$PC_TOPICS" | head -1)
else
    echo "  (none found)"
    SUGGESTED_PC=""
fi

echo ""
echo "Odometry Topics (Odometry):"
ODOM_TOPICS=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep -E "nav_msgs/msg/Odometry" | awk '{print $2}' || echo "")
if [[ -n "$ODOM_TOPICS" ]]; then
    echo "$ODOM_TOPICS" | while read topic; do
        echo "  - $topic"
    done
    SUGGESTED_ODOM=$(echo "$ODOM_TOPICS" | head -1)
else
    echo "  (none found)"
    SUGGESTED_ODOM=""
fi

echo ""
echo "LaserScan Topics (LaserScan):"
SCAN_TOPICS=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep -E "sensor_msgs/msg/LaserScan" | awk '{print $2}' || echo "")
if [[ -n "$SCAN_TOPICS" ]]; then
    echo "$SCAN_TOPICS" | while read topic; do
        echo "  - $topic"
    done
else
    echo "  (none found)"
fi

echo ""
echo "Image Topics (Image/CompressedImage):"
IMG_TOPICS=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep -E "sensor_msgs/msg/(Image|CompressedImage)" | awk '{print $2}' || echo "")
if [[ -n "$IMG_TOPICS" ]]; then
    echo "$IMG_TOPICS" | while read topic; do
        echo "  - $topic"
    done
else
    echo "  (none found)"
fi

echo ""
echo "CameraInfo Topics:"
INFO_TOPICS=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep -E "sensor_msgs/msg/CameraInfo" | awk '{print $2}' || echo "")
if [[ -n "$INFO_TOPICS" ]]; then
    echo "$INFO_TOPICS" | while read topic; do
        echo "  - $topic"
    done
else
    echo "  (none found)"
fi

echo ""
echo "============================================"
echo "Suggested Configuration"
echo "============================================"
echo ""

if [[ -n "$SUGGESTED_PC" || -n "$SUGGESTED_ODOM" ]]; then
    echo "For 3D point cloud mode, try:"
    echo ""
    echo "With ros2 launch:"
    echo ""
    echo "  ros2 launch fl_slam_poc gc_rosbag.launch.py \\"
    echo "    bag:=$BAG_PATH \\"
    echo "    play_bag:=true \\"
    if [[ -n "$SUGGESTED_PC" ]]; then
        echo "    pointcloud_topic:=$SUGGESTED_PC \\"
    fi
    if [[ -n "$SUGGESTED_ODOM" ]]; then
        echo "    odom_topic:=$SUGGESTED_ODOM"
    fi
else
    echo "WARNING: Could not find suitable topics for 3D FL-SLAM."
    echo "This rosbag may not contain the required sensor data."
    echo ""
    echo "Required:"
    echo "  - PointCloud2 (for 3D mode) OR LaserScan (for 2D mode)"
    echo "  - Odometry"
fi
