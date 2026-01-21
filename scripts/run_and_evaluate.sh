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

# Run SLAM with rosbag
echo "[1/3] Running SLAM system..."

# Get bag info for duration estimate
BAG_DURATION=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep "Duration" | awk '{print $2}' | cut -d'.' -f1 || echo "unknown")
echo "  Bag duration: ~${BAG_DURATION}s"
echo "  Starting playback with progress monitoring..."
echo ""

# Run SLAM and filter output to show progress
ros2 launch fl_slam_poc poc_m3dgr_rosbag.launch.py bag:="$BAG_PATH" 2>&1 | \
  while IFS= read -r line; do
    # Filter out noisy warnings
    if [[ "$line" == *"not found:"* ]]; then
      continue
    fi
    if [[ "$line" == *"camera_color_optical_frame"* ]]; then
      continue
    fi
    if [[ "$line" == *"SENSOR STALE"* ]]; then
      continue
    fi
    if [[ "$line" == *"SENSOR MISSING"* ]]; then
      continue
    fi
    # Show progress updates (backend status)
    if [[ "$line" == *"Backend status:"* ]]; then
      # Extract just the status info
      status=$(echo "$line" | sed 's/.*Backend status: //')
      echo -ne "\r  Progress: $status                              "
    elif [[ "$line" == *"ERROR"* ]] || [[ "$line" == *"Traceback"* ]]; then
      echo ""
      echo "$line"
    elif [[ "$line" == *"Created anchor"* ]]; then
      # Show anchor creation
      echo -ne "\r  $line                              "
    fi
  done

echo ""
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

# Align ground truth timestamps
echo "Aligning ground truth timestamps..."
python3 "$PROJECT_ROOT/scripts/align_ground_truth.py" \
  "$GT_FILE" \
  "$EST_FILE" \
  "$GT_ALIGNED"

# Run evaluation
echo ""
echo "Computing metrics and generating plots..."
python3 "$PROJECT_ROOT/scripts/evaluate_slam.py" \
  "$GT_ALIGNED" \
  "$EST_FILE" \
  "$RESULTS_DIR"

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
