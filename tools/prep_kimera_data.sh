#!/bin/bash
# Prep Kimera_Data for potential GC SLAM usage:
#   1. Extract ground truth from GT zip and convert CSVs to TUM format.
#   2. Optionally decompress ROS 1 bags (if they were compressed).
#   3. Document ROS 1 → ROS 2 bag conversion (requires rosbag2_bag_v2 plugin or play+record).
#
# Usage: bash tools/prep_kimera_data.sh [path_to_Kimera_Data]
# Default path: PROJECT_ROOT/rosbags/Kimera_Data

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KIMERA_DIR="${1:-$PROJECT_ROOT/rosbags/Kimera_Data}"
cd "$PROJECT_ROOT"

if [ ! -d "$KIMERA_DIR" ]; then
    echo "Kimera_Data directory not found: $KIMERA_DIR"
    exit 1
fi

GT_ZIP=""
for f in "$KIMERA_DIR"/ground_truth-*.zip; do
    [ -f "$f" ] && GT_ZIP="$f" && break
done

# ---------------------------------------------------------------------------
# 1. Extract ground truth and convert to TUM
# ---------------------------------------------------------------------------
if [ -n "$GT_ZIP" ]; then
    echo "Extracting ground truth from $(basename "$GT_ZIP")..."
    (cd "$KIMERA_DIR" && unzip -o -q "$GT_ZIP")
    GT_DIR="$KIMERA_DIR/ground_truth"
    if [ ! -d "$GT_DIR" ]; then
        echo "Expected ground_truth/ after unzip; not found."
        exit 1
    fi
    echo "Converting *_gt_odom.csv → *_gt.tum..."
    for csv in "$GT_DIR"/1014/*_gt_odom.csv "$GT_DIR"/1207/*_gt_odom.csv "$GT_DIR"/1208/*_gt_odom.csv; do
        [ -f "$csv" ] || continue
        base=$(basename "$csv" _gt_odom.csv)
        seq=$(basename "$(dirname "$csv")")
        tum="$GT_DIR/$seq/${base}_gt.tum"
        python3 "$PROJECT_ROOT/tools/kimera_gt_to_tum.py" "$csv" "$tum"
    done
    echo "Ground truth TUM files written under $GT_DIR (per sequence and robot)."
else
    echo "No ground_truth-*.zip found in $KIMERA_DIR; skipping GT extraction."
fi

# ---------------------------------------------------------------------------
# 2. Decompress ROS 1 bags (if present and compressed)
# ---------------------------------------------------------------------------
for bag in "$KIMERA_DIR"/*.bag; do
    [ -f "$bag" ] || continue
    if [ -f "${bag}.active" ]; then
        echo "Decompressing $(basename "$bag")..."
        (cd "$KIMERA_DIR" && rosbag decompress "$(basename "$bag")" 2>/dev/null) || true
    fi
done
# If rosbag is not installed (ROS 1), decompress is skipped; bags may already be decompressed.

# ---------------------------------------------------------------------------
# 3. ROS 1 → ROS 2 bag conversion
# ---------------------------------------------------------------------------
# Jazzy: rosbag2_bag_v2_plugins is UNRELEASED for Jazzy (see index.ros.org).
# Options:
#   A) Install ROS 1 (Melodic/Noetic) + build rosbag2_bag_v2 from source for Jazzy; then:
#        source /opt/ros/noetic/setup.bash
#        source /opt/ros/jazzy/setup.bash
#        ros2 bag play -s rosbag_v2 /path/to/file.bag
#      In another terminal: ros2 bag record -o /path/to/output_ros2 -a
#   B) Use Humble (or another distro with released bag_v2): install ros-humble-rosbag2-bag-v2-plugins,
#      then play+record as above.
# We do not run conversion here to avoid requiring ROS 1; we write a helper script that users
# can run when the plugin is available.
CONVERT_SCRIPT="$KIMERA_DIR/convert_ros1_to_ros2.sh"
cat > "$CONVERT_SCRIPT" << 'CONVERT_EOF'
#!/bin/bash
# Convert one ROS 1 .bag to ROS 2 (sqlite3) by play+record.
# Requires: ROS 1 sourced first, then ROS 2, and rosbag2 with rosbag_v2 plugin.
#   e.g. source /opt/ros/noetic/setup.bash && source /opt/ros/jazzy/setup.bash
# Usage: ./convert_ros1_to_ros2.sh <path_to_ros1.bag> [output_ros2_dir]
# Example: ./convert_ros1_to_ros2.sh 10_14_acl_jackal-005.bag 10_14_acl_jackal-005_ros2

BAG="$1"
OUT="${2:-${BAG%.bag}_ros2}"
if [ -z "$BAG" ] || [ ! -f "$BAG" ]; then
    echo "Usage: $0 <path_to_ros1.bag> [output_ros2_dir]"
    exit 1
fi
echo "Play ROS 1 bag (with rosbag_v2) and record to $OUT"
echo "Terminal 1: ros2 bag play -s rosbag_v2 \"$BAG\""
echo "Terminal 2: ros2 bag record -o \"$OUT\" -a"
echo "Run record first, then play. Stop record when play finishes."
CONVERT_EOF
chmod +x "$CONVERT_SCRIPT"
echo "ROS 1→2 conversion helper written: $CONVERT_SCRIPT"

echo ""
echo "Prep done. Next steps for GC SLAM:"
echo "  - Use ground_truth/<seq>/<robot>_gt.tum with tools/align_ground_truth.py for evaluation."
echo "  - To use bags with ros2 bag play, convert ROS 1→2 using $CONVERT_SCRIPT when the bag_v2 plugin is available."
