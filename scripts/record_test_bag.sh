#!/usr/bin/env bash
# Record a test rosbag from Gazebo with loop closures
# This creates a proper test bag with the robot completing a circular trajectory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
BAG_NAME="${BAG_NAME:-fl_slam_loop_test}"
BAG_DIR="${BAG_DIR:-${PROJECT_DIR}/rosbags/${BAG_NAME}}"
DURATION="${DURATION:-60}"  # Recording duration in seconds
ROBOT_MODEL="${ROBOT_MODEL:-waffle}"  # TurtleBot3 model

echo "=========================================="
echo "FL-SLAM Test Bag Recording"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Robot model: ${ROBOT_MODEL}"
echo "  Duration:    ${DURATION}s"
echo "  Output:      ${BAG_DIR}"
echo ""
echo "This will:"
echo "  1. Launch Gazebo with TurtleBot3"
echo "  2. Drive the robot in a circular pattern"
echo "  3. Record all sensor data"
echo ""

# Source ROS
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    echo "ERROR: ROS 2 Jazzy not found" >&2
    exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
set -u

# Create output directory
mkdir -p "$(dirname "$BAG_DIR")"
if [ -d "$BAG_DIR" ]; then
    echo "WARNING: Bag directory already exists: $BAG_DIR"
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    rm -rf "$BAG_DIR"
fi

echo "=========================================="
echo "Step 1: Launching Gazebo"
echo "=========================================="
echo ""

# Set TurtleBot3 model
export TURTLEBOT3_MODEL="${ROBOT_MODEL}"

# Launch Gazebo in background
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py &
GAZEBO_PID=$!

echo "Waiting for Gazebo to start (10s)..."
sleep 10

# Check if Gazebo is running
if ! kill -0 $GAZEBO_PID 2>/dev/null; then
    echo "ERROR: Gazebo failed to start" >&2
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Starting Recording"
echo "=========================================="
echo ""

# Topics to record (all sensor data)
TOPICS=(
    /scan
    /odom
    /camera/image_raw
    /camera/depth/image_raw
    /camera/camera_info
    /tf
    /tf_static
    /joint_states
)

echo "Recording topics:"
for topic in "${TOPICS[@]}"; do
    echo "  - $topic"
done
echo ""

# Start recording
ros2 bag record -o "$BAG_DIR" "${TOPICS[@]}" &
RECORD_PID=$!

sleep 2

echo ""
echo "=========================================="
echo "Step 3: Driving Robot in Circle"
echo "=========================================="
echo ""
echo "The robot will drive in a circle to create loop closures."
echo "Duration: ${DURATION}s"
echo ""

# Drive robot in circle using simple twist commands
timeout ${DURATION}s bash -c '
source /opt/ros/jazzy/setup.bash

# Calculate circular motion (0.2 m/s linear, 0.5 rad/s angular for ~2.5m radius)
# This should complete a loop in ~30-40 seconds
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist "{
  linear: {x: 0.2, y: 0.0, z: 0.0},
  angular: {x: 0.0, y: 0.0, z: 0.5}
}"
' || true

echo ""
echo "Stopping robot..."
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" >/dev/null 2>&1 || true

echo ""
echo "=========================================="
echo "Step 4: Stopping Recording"
echo "=========================================="
echo ""

# Stop recording gracefully
kill -INT $RECORD_PID 2>/dev/null || true
sleep 3

# Kill recording if still running
kill -KILL $RECORD_PID 2>/dev/null || true

# Stop Gazebo
echo "Stopping Gazebo..."
kill -INT $GAZEBO_PID 2>/dev/null || true
sleep 2
kill -KILL $GAZEBO_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Recording Complete"
echo "=========================================="
echo ""
echo "Bag saved to: ${BAG_DIR}"
echo ""

# Show bag info
if [ -d "$BAG_DIR" ]; then
    echo "Bag info:"
    ros2 bag info "$BAG_DIR" 2>/dev/null || echo "  (run 'ros2 bag info $BAG_DIR' to see details)"
    echo ""
    echo "To use this bag for testing:"
    echo "  BAG_PATH=${BAG_DIR} ./scripts/test-integration.sh"
    echo ""
else
    echo "WARNING: Bag directory not found. Recording may have failed."
    exit 1
fi
