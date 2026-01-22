#!/bin/bash
# FL-SLAM M3DGR: Run SLAM + Evaluation
# Note: Rerun visualization temporarily disabled due to build complexity
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
BAG_PATH="$PROJECT_ROOT/rosbags/m3dgr/Dynamic01_ros2"
GT_FILE="$PROJECT_ROOT/rosbags/m3dgr/Dynamic01.txt"
EST_FILE="/tmp/fl_slam_trajectory.tum"
GT_ALIGNED="/tmp/m3dgr_ground_truth_aligned.tum"
RESULTS_DIR="$PROJECT_ROOT/results/m3dgr_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/slam_run.log"
OP_REPORT_FILE="$RESULTS_DIR/op_report.jsonl"

echo "========================================"
echo "FL-SLAM M3DGR Pipeline"
echo "========================================"
echo ""
echo "Bag: $BAG_PATH"
echo "Results: $RESULTS_DIR"
echo ""

# Clean previous run
rm -f "$EST_FILE" "$GT_ALIGNED"
mkdir -p "$RESULTS_DIR"

# Source ROS2 (suppress rerun_bridge not found warning)
source /opt/ros/jazzy/setup.bash
source "$PROJECT_ROOT/fl_ws/install/setup.bash" 2>/dev/null || source "$PROJECT_ROOT/fl_ws/install/setup.bash" 2>&1 | grep -v "not found:"

# Set ROS environment variables to writable locations (fixes permission issues)
export ROS_HOME="${ROS_HOME:-/tmp/ros_home}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_log}"
export RMW_FASTRTPS_USE_SHM="${RMW_FASTRTPS_USE_SHM:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://${PROJECT_ROOT}/config/cyclonedds.xml}"
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"

# Run SLAM with rosbag
echo "[1/3] Running SLAM system..."

# Get bag info for duration estimate
BAG_DURATION=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep "Duration" | awk '{print $2}' | cut -d'.' -f1 || echo "180")
TIMEOUT_SEC=$((BAG_DURATION + 30))  # Add 30s buffer for processing
echo "  Bag duration: ~${BAG_DURATION}s (timeout: ${TIMEOUT_SEC}s)"
echo "  Full log: $LOG_FILE"
echo "  Starting playback with progress monitoring..."
echo ""

# Launch SLAM in background with full logging
ros2 launch fl_slam_poc poc_m3dgr_rosbag.launch.py bag:="$BAG_PATH" > "$LOG_FILE" 2>&1 &
LAUNCH_PID=$!

# Capture OpReports for evaluation diagnostics
ros2 topic echo /cdwm/op_report --field data --full-length > "$OP_REPORT_FILE" 2>/dev/null &
OP_REPORT_PID=$!

# Monitor progress by tailing log with filtered display
tail -f "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    # Filter out noisy warnings for display (but keep in log file)
    if [[ "$line" == *"not found:"* ]] || \
       [[ "$line" == *"camera_color_optical_frame"* ]] || \
       [[ "$line" == *"SENSOR STALE"* ]] || \
       [[ "$line" == *"SENSOR MISSING"* ]]; then
      continue
    fi
    # Show progress updates (backend status)
    if [[ "$line" == *"Backend status:"* ]]; then
      status=$(echo "$line" | sed 's/.*Backend status: //')
      echo -ne "\r  Progress: $status                              "
    elif [[ "$line" == *"ERROR"* ]] || [[ "$line" == *"Traceback"* ]]; then
      echo ""
      echo "$line"
    elif [[ "$line" == *"Created anchor"* ]]; then
      echo -ne "\r  $line                              "
    fi
done &
TAIL_PID=$!

# Wait for timeout, then kill launch
sleep "$TIMEOUT_SEC"
echo ""
echo "  Timeout reached (${TIMEOUT_SEC}s), stopping SLAM..."

# Kill the launch process and all its children
pkill -P $LAUNCH_PID 2>/dev/null || true
kill $LAUNCH_PID 2>/dev/null || true
kill $TAIL_PID 2>/dev/null || true
kill $OP_REPORT_PID 2>/dev/null || true

# Give processes time to cleanup
sleep 2

echo "  SLAM complete!"
echo ""

# Evaluation
echo ""
echo "[2/3] Evaluating trajectory..."
echo ""

# Check if trajectory was exported
if [ ! -f "$EST_FILE" ]; then
    echo "ERROR: Estimated trajectory not found at $EST_FILE"
    exit 1
fi

# Activate venv if it exists (for evo package)
if [ -d "$HOME/.venv" ]; then
  source "$HOME/.venv/bin/activate" 2>/dev/null || true
fi

# Align ground truth timestamps
echo "Aligning ground truth timestamps..."
python3 "$PROJECT_ROOT/tools/align_ground_truth.py" \
  "$GT_FILE" \
  "$EST_FILE" \
  "$GT_ALIGNED"

# Run evaluation
echo ""
echo "Computing metrics and generating plots..."
python3 "$PROJECT_ROOT/tools/evaluate_slam.py" \
  "$GT_ALIGNED" \
  "$EST_FILE" \
  "$RESULTS_DIR" \
  "$OP_REPORT_FILE"

# Copy trajectory files to results
cp "$EST_FILE" "$RESULTS_DIR/estimated_trajectory.tum"
cp "$GT_ALIGNED" "$RESULTS_DIR/ground_truth_aligned.tum"

echo ""
echo "[3/3] Complete!"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
echo ""
echo "Contents:"
ls -lh "$RESULTS_DIR"
echo ""
echo "View results:"
echo "  Trajectory Plots:"
echo "    - trajectory_comparison.png (4-view overlay: XY, XZ, YZ, 3D)"
echo "    - trajectory_heatmap.png (error-colored trajectory)"
echo "    - pose_graph.png (pose nodes with odometry edges)"
echo ""
echo "  Error Analysis:"
echo "    - error_analysis.png (error over time + histogram)"
echo ""
echo "  Metrics:"
echo "    - metrics.txt (human-readable summary)"
echo "    - metrics.csv (spreadsheet-ready with all statistics)"
echo ""
echo "  Trajectories:"
echo "    - estimated_trajectory.tum"
echo "    - ground_truth_aligned.tum"
