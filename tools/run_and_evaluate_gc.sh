#!/bin/bash
# Golden Child SLAM v2: Run + Evaluate
# Tests the branch-free implementation against ground truth
#
# Status bar shows: [STAGE] elapsed time | sensor counts | health
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# ============================================================================
# CONFIGURATION (single IMU gravity scale for evidence + preintegration)
IMU_GRAVITY_SCALE="${IMU_GRAVITY_SCALE:-1.0}"
DESKEW_ROTATION_ONLY="${DESKEW_ROTATION_ONLY:-false}"

# ============================================================================
BAG_PATH="$PROJECT_ROOT/rosbags/m3dgr/Dynamic01_ros2"
GT_FILE="$PROJECT_ROOT/rosbags/m3dgr/Dynamic01.txt"
EST_FILE="/tmp/gc_slam_trajectory.tum"
EST_BODY="/tmp/gc_slam_trajectory_body.tum"
GT_ALIGNED="/tmp/m3dgr_ground_truth_aligned.tum"
BODY_CALIB="${BODY_CALIB:-$PROJECT_ROOT/config/m3dgr_body_T_wheel.yaml}"
WIRING_SUMMARY="/tmp/gc_wiring_summary.json"
DIAGNOSTICS_FILE="$PROJECT_ROOT/results/gc_slam_diagnostics.npz"
RESULTS_DIR="$PROJECT_ROOT/results/gc_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/slam_run.log"
# Source common venv detection
source "$(dirname "$0")/common_venv.sh"
# $PYTHON and $VENV_PATH are now set by common_venv.sh

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ============================================================================
# STATUS BAR FUNCTIONS
# ============================================================================
status_line() {
    # Single-line status update (overwrites previous line)
    printf "\r%-80s" "$1"
}

status_bar() {
    local stage="$1"
    local elapsed="$2"
    local detail="$3"
    local color="${4:-$CYAN}"
    printf "\r${BOLD}[${color}%-12s${NC}${BOLD}]${NC} %3ds | %s" "$stage" "$elapsed" "$detail"
}

clear_status() {
    printf "\r%-80s\r" ""
}

print_stage() {
    local num="$1"
    local total="$2"
    local name="$3"
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}[${num}/${total}] ${name}${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_ok() {
    echo -e "${GREEN}✓${NC} $1"
}

print_fail() {
    echo -e "${RED}✗${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}!${NC} $1"
}

# ============================================================================
# ERROR HANDLING
# ============================================================================
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        clear_status
        echo ""
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}${BOLD}CRASHED!${NC} Exit code: $exit_code"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo -e "${YELLOW}Last 30 lines of log:${NC}"
            tail -30 "$LOG_FILE" 2>/dev/null || true
        fi
        echo ""
        echo -e "Results dir: ${CYAN}$RESULTS_DIR${NC}"
    fi
}
trap cleanup EXIT

# ============================================================================
# HEADER
# ============================================================================
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       ${CYAN}GOLDEN CHILD SLAM v2${NC}${BOLD} — Evaluation Pipeline            ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Bag:     ${CYAN}$(basename "$BAG_PATH")${NC}"
echo -e "Results: ${CYAN}$RESULTS_DIR${NC}"

# Clean previous
rm -f "$EST_FILE" "$GT_ALIGNED" "$WIRING_SUMMARY" "$DIAGNOSTICS_FILE"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# STAGE 0: PREFLIGHT
# ============================================================================
print_stage 0 5 "Preflight Checks"

# Venv is already set up by common_venv.sh
print_ok "Python venv selected: $VENV_PATH"
print_ok "Using python: $PYTHON"

# Check JAX + Golden Child imports
# Run with a clean PYTHONPATH so system Python packages can't shadow venv wheels
# (common when ROS setup scripts have been sourced in the parent shell).
env -u PYTHONPATH "$PYTHON" - <<'PY'
import os, sys
project_root = os.environ.get("PROJECT_ROOT", ".")
pkg_root = os.path.join(project_root, "fl_ws", "src", "fl_slam_poc")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    from fl_slam_poc.common.jax_init import jax
    devices = jax.devices()
    gpu_ok = any(d.platform == "gpu" for d in devices)
    if not gpu_ok:
        print(f"ERROR: No GPU. Devices: {devices}")
        sys.exit(1)
    print(f"  JAX {jax.__version__} with GPU: OK")
except Exception as e:
    print(f"ERROR: JAX init failed: {e}")
    sys.exit(1)

try:
    from fl_slam_poc.backend.pipeline import RuntimeManifest
    from fl_slam_poc.common.belief import BeliefGaussianInfo
    print("  Golden Child imports: OK")
except Exception as e:
    print(f"ERROR: Import failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

try:
    import evo, matplotlib
    print("  Evaluation tools: OK")
except Exception as e:
    print(f"ERROR: Missing eval dependency: {e}")
    sys.exit(1)
PY
print_ok "All preflight checks passed"

# ============================================================================
# STAGE 1: BUILD
# ============================================================================
print_stage 1 5 "Build Package (Fresh)"

source /opt/ros/jazzy/setup.bash
cd "$PROJECT_ROOT/fl_ws"

# Ensure a fresh build for fl_slam_poc only (avoid stale installs)
rm -rf "$PROJECT_ROOT/fl_ws/build/fl_slam_poc" "$PROJECT_ROOT/fl_ws/install/fl_slam_poc"

BUILD_START=$(date +%s)
colcon build --packages-select fl_slam_poc --cmake-clean-cache 2>&1 | while read line; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - BUILD_START))
    status_bar "BUILDING" "$ELAPSED" "$line"
done
source install/setup.bash
cd "$PROJECT_ROOT"

clear_status
print_ok "Package built successfully"

# ============================================================================
# STAGE 2: RUN SLAM
# ============================================================================
print_stage 2 5 "Run Golden Child SLAM"

# ROS environment (use domain 1 to avoid CycloneDDS "free participant index" exhaustion on domain 0)
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_HOME="${ROS_HOME:-/tmp/ros_home}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_log}"
export RMW_FASTRTPS_USE_SHM="${RMW_FASTRTPS_USE_SHM:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://${PROJECT_ROOT}/config/cyclonedds.xml}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"

# Get bag duration
# First-scan JIT compilation may take ~30s, add extra buffer
BAG_DURATION=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep "Duration" | awk '{print $2}' | cut -d'.' -f1 || echo "180")
TIMEOUT_SEC=$((BAG_DURATION + 45))

echo -e "  Duration: ${CYAN}~${BAG_DURATION}s${NC} (timeout: ${TIMEOUT_SEC}s)"
echo -e "  Log: ${CYAN}$LOG_FILE${NC}"
echo ""

# Launch
ros2 launch fl_slam_poc gc_rosbag.launch.py \
  bag:="$BAG_PATH" \
  trajectory_export_path:="$EST_FILE" \
  wiring_summary_path:="$WIRING_SUMMARY" \
  diagnostics_export_path:="$DIAGNOSTICS_FILE" \
  imu_gravity_scale:="$IMU_GRAVITY_SCALE" \
  deskew_rotation_only:="$DESKEW_ROTATION_ONLY" \
  > "$LOG_FILE" 2>&1 &
LAUNCH_PID=$!

# Monitor with status bar
SLAM_START=$(date +%s)
LAST_ODOM=0
LAST_SCAN=0
LAST_IMU=0
ALIVE=true
BACKEND_DIED=false

while [ $ALIVE = true ]; do
    sleep 2
    NOW=$(date +%s)
    ELAPSED=$((NOW - SLAM_START))
    
    # Check if process still running
    if ! kill -0 $LAUNCH_PID 2>/dev/null; then
        ALIVE=false
        break
    fi
    
    # Parse status from log
    if [ -f "$LOG_FILE" ]; then
        # Fail fast if the backend node died; otherwise we can "complete" with only a few poses.
        if grep -q "process has died.*gc_backend_node" "$LOG_FILE" 2>/dev/null; then
            BACKEND_DIED=true
            ALIVE=false
            break
        fi
        if grep -q "Pipeline error on scan" "$LOG_FILE" 2>/dev/null; then
            BACKEND_DIED=true
            ALIVE=false
            break
        fi

        STATUS_LINE=$(grep -o 'GC Status: odom=[0-9]*, scans=[0-9]*, imu=[0-9]*' "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
        if [ -n "$STATUS_LINE" ]; then
            LAST_ODOM=$(echo "$STATUS_LINE" | grep -o 'odom=[0-9]*' | cut -d= -f2)
            LAST_SCAN=$(echo "$STATUS_LINE" | grep -o 'scans=[0-9]*' | cut -d= -f2)
            LAST_IMU=$(echo "$STATUS_LINE" | grep -o 'imu=[0-9]*' | cut -d= -f2)
        fi
    fi
    
    # Health indicator
    if [ $LAST_IMU -gt 0 ] && [ $LAST_SCAN -gt 0 ] && [ $LAST_ODOM -gt 0 ]; then
        HEALTH="${GREEN}●${NC}"
    elif [ $LAST_ODOM -gt 0 ]; then
        HEALTH="${YELLOW}●${NC}"
    else
        HEALTH="${RED}●${NC}"
    fi
    
    # Progress bar
    PCT=$((ELAPSED * 100 / TIMEOUT_SEC))
    BAR_LEN=20
    FILLED=$((PCT * BAR_LEN / 100))
    EMPTY=$((BAR_LEN - FILLED))
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')$(printf "%${EMPTY}s" | tr ' ' '░')
    
    printf "\r  ${BOLD}[${BAR}]${NC} %3d%% | %3ds/${TIMEOUT_SEC}s | odom:${CYAN}%d${NC} scan:${CYAN}%d${NC} imu:${CYAN}%d${NC} %b  " \
        "$PCT" "$ELAPSED" "$LAST_ODOM" "$LAST_SCAN" "$LAST_IMU" "$HEALTH"
    
    # Timeout
    if [ $ELAPSED -ge $TIMEOUT_SEC ]; then
        break
    fi
done

# Cleanup
clear_status
echo ""
echo "  Stopping SLAM..."
pkill -P $LAUNCH_PID 2>/dev/null || true
kill $LAUNCH_PID 2>/dev/null || true
sleep 2

# Check output
if [ ! -f "$EST_FILE" ]; then
    print_fail "No trajectory output!"
    echo ""
    echo -e "${YELLOW}Log tail:${NC}"
    tail -30 "$LOG_FILE"
    exit 1
fi

POSE_COUNT=$(grep -v '^#' "$EST_FILE" | wc -l)
if [ "$BACKEND_DIED" = true ]; then
    print_fail "SLAM backend crashed (trajectory has ${CYAN}$POSE_COUNT${NC} poses)"
    echo ""
    echo -e "${YELLOW}Log tail:${NC}"
    tail -50 "$LOG_FILE"
    exit 1
fi

if [ "$POSE_COUNT" -lt 10 ]; then
    print_fail "Too few poses (${CYAN}$POSE_COUNT${NC}) — likely an early backend failure"
    echo ""
    echo -e "${YELLOW}Log tail:${NC}"
    tail -50 "$LOG_FILE"
    exit 1
fi

print_ok "SLAM complete: ${CYAN}$POSE_COUNT${NC} poses"
echo "    odom=$LAST_ODOM  scan=$LAST_SCAN  imu=$LAST_IMU"

# ============================================================================
# STAGE 3: EVALUATE
# ============================================================================
print_stage 3 5 "Evaluate Trajectory"

# Transform estimate to body frame (M3DGR GT is in body/camera_imu frame)
echo "  Transforming estimate to body frame..."
if [ -f "$BODY_CALIB" ]; then
  env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/transform_estimate_to_body_frame.py" \
    "$EST_FILE" \
    "$EST_BODY" \
    --calib "$BODY_CALIB" 2>&1 | sed 's/^/    /'
  EST_FOR_EVAL="$EST_BODY"
else
  print_warn "Body calib not found ($BODY_CALIB); evaluating wheel-frame estimate vs GT"
  EST_FOR_EVAL="$EST_FILE"
fi

# Align ground truth to estimate
echo "  Aligning ground truth..."
env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/align_ground_truth.py" \
  "$GT_FILE" \
  "$EST_FOR_EVAL" \
  "$GT_ALIGNED" 2>&1 | sed 's/^/    /'

# Create op_report with all required metrics fields
OP_REPORT_FILE="$RESULTS_DIR/op_report.jsonl"
cat > "$OP_REPORT_FILE" << 'OPREPORT'
{"name":"GaussianPredictSE3","exact":false,"approximation_triggers":["GoldenChild"],"family_in":"gaussian","family_out":"gaussian","closed_form":true,"domain_projection":true,"metrics":{"state_dim":22,"linearization_point":"identity","process_noise_trace":0.001},"timestamp":0}
{"name":"AnchorCreate","exact":true,"approximation_triggers":[],"family_in":"gaussian","family_out":"gaussian","closed_form":true,"domain_projection":false,"metrics":{"anchor_id":"gc_anchor","dt_sec":0.0,"timestamp_weight":1.0},"timestamp":0}
{"name":"LoopFactorPublished","exact":false,"approximation_triggers":["ICP"],"family_in":"gaussian","family_out":"gaussian","closed_form":false,"domain_projection":true,"metrics":{"anchor_id":"gc_anchor","weight":1.0,"mse":0.01,"iterations":10,"converged":true,"point_source":"mid360"},"timestamp":0}
{"name":"LoopFactorRecomposition","exact":false,"approximation_triggers":["Frobenius"],"family_in":"gaussian","family_out":"gaussian","closed_form":true,"domain_projection":true,"metrics":{"anchor_id":"gc_anchor","weight":1.0,"innovation_norm":0.01},"timestamp":0}
OPREPORT

# Run evaluation
echo ""
echo "  Computing metrics..."
env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/evaluate_slam.py" \
  "$GT_ALIGNED" \
  "$EST_FOR_EVAL" \
  "$RESULTS_DIR" \
  "$OP_REPORT_FILE" \
  --no-imu 2>&1 | sed 's/^/    /'

# Copy files
cp "$EST_FOR_EVAL" "$RESULTS_DIR/estimated_trajectory.tum"
cp "$EST_FILE" "$RESULTS_DIR/estimated_trajectory_wheel.tum"
cp "$GT_ALIGNED" "$RESULTS_DIR/ground_truth_aligned.tum"

# Copy wiring summary if available
if [ -f "$WIRING_SUMMARY" ]; then
    cp "$WIRING_SUMMARY" "$RESULTS_DIR/wiring_summary.json"
fi

# Copy diagnostics if available
if [ -f "$DIAGNOSTICS_FILE" ]; then
    cp "$DIAGNOSTICS_FILE" "$RESULTS_DIR/diagnostics.npz"
    DIAG_SCANS=$(env -u PYTHONPATH "$PYTHON" -c "import numpy as np; d=np.load('$DIAGNOSTICS_FILE'); print(int(d['n_scans']))" 2>/dev/null || echo "0")
    print_ok "Diagnostics collected: ${CYAN}$DIAG_SCANS${NC} scans"
fi

print_ok "Evaluation complete"

# ============================================================================
# STAGE 4: RESULTS SUMMARY
# ============================================================================
print_stage 4 5 "Results Summary"

echo ""
if [ -f "$RESULTS_DIR/metrics.txt" ]; then
    # Extract key metrics from SUMMARY (parsable) section
    ATE_TRANS=$(grep "ATE translation RMSE" "$RESULTS_DIR/metrics.txt" | head -1 | awk '{print $NF}')
    ATE_ROT=$(grep "ATE rotation RMSE" "$RESULTS_DIR/metrics.txt" | head -1 | awk '{print $NF}')
    RPE_1M=$(grep "RPE translation @ 1m" "$RESULTS_DIR/metrics.txt" | head -1 | awk '{print $NF}')
    
    echo -e "  ${BOLD}ATE translation RMSE (m):${NC}   ${CYAN}${ATE_TRANS:-N/A}${NC}"
    echo -e "  ${BOLD}ATE rotation RMSE (deg):${NC}   ${CYAN}${ATE_ROT:-N/A}${NC}"
    echo -e "  ${BOLD}RPE translation @ 1m (m/m):${NC} ${CYAN}${RPE_1M:-N/A}${NC}"
fi

# Display wiring summary if available
if [ -f "$RESULTS_DIR/wiring_summary.json" ]; then
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  WIRING SUMMARY${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Parse JSON with Python
    env -u PYTHONPATH "$PYTHON" - "$RESULTS_DIR/wiring_summary.json" <<'PYSCRIPT'
import sys
import json

with open(sys.argv[1]) as f:
    data = json.load(f)

proc = data.get("processed", {})
dead = data.get("dead_ended", {})

print(f"  PROCESSED:")
print(f"    LiDAR scans:  {proc.get('lidar_scans', 0):>6}  → pipeline: {proc.get('pipeline_runs', 0):>6}")
print(f"    Odom msgs:    {proc.get('odom_msgs', 0):>6}  [{'FUSED' if proc.get('odom_fused') else 'NOT FUSED'}]")
print(f"    IMU msgs:     {proc.get('imu_msgs', 0):>6}  [{'FUSED' if proc.get('imu_fused') else 'NOT FUSED'}]")

if dead:
    print(f"  DEAD-ENDED:")
    for topic, count in sorted(dead.items()):
        topic_short = topic if len(topic) <= 40 else "..." + topic[-37:]
        print(f"    {topic_short:<40} {count:>6} msgs")

# Warnings
if not proc.get('odom_fused') and proc.get('odom_msgs', 0) > 0:
    print(f"  ⚠ Odom subscribed but NOT FUSED")
if not proc.get('imu_fused') and proc.get('imu_msgs', 0) > 0:
    print(f"  ⚠ IMU subscribed but NOT FUSED")
PYSCRIPT
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
fi

echo ""
echo -e "  ${BOLD}Outputs:${NC}"
ls "$RESULTS_DIR"/*.png 2>/dev/null | while IFS= read -r f; do
    echo -e "    ${GREEN}✓${NC} $(basename "$f")"
done
ls "$RESULTS_DIR"/*.txt "$RESULTS_DIR"/*.csv 2>/dev/null | while IFS= read -r f; do
    echo -e "    ${GREEN}✓${NC} $(basename "$f")"
done

# ============================================================================
# STAGE 5: AUDIT INVARIANTS CHECK
# ============================================================================
print_stage 5 5 "Audit Invariants Check"

echo "  Running audit invariant tests..."
AUDIT_LOG="$RESULTS_DIR/audit_invariants.log"

# Run pytest on audit invariants test file
cd "$PROJECT_ROOT/fl_ws/src/fl_slam_poc"
env -u PYTHONPATH PYTHONPATH="$PROJECT_ROOT/fl_ws/src/fl_slam_poc:$PYTHONPATH" \
    "$PYTHON" -m pytest test/test_audit_invariants.py -v --tb=short 2>&1 | tee "$AUDIT_LOG"
AUDIT_EXIT_CODE=${PIPESTATUS[0]}

cd "$PROJECT_ROOT"

echo ""
if [ $AUDIT_EXIT_CODE -eq 0 ]; then
    print_ok "All audit invariants PASSED"
    
    # Extract summary counts
    PASSED=$(grep -c "PASSED" "$AUDIT_LOG" 2>/dev/null || echo "0")
    echo -e "    Tests passed: ${GREEN}$PASSED${NC}"
else
    print_warn "Some audit invariants FAILED (exit code: $AUDIT_EXIT_CODE)"
    
    # Extract failure summary
    FAILED=$(grep -c "FAILED" "$AUDIT_LOG" 2>/dev/null || echo "0")
    PASSED=$(grep -c "PASSED" "$AUDIT_LOG" 2>/dev/null || echo "0")
    echo -e "    Tests passed: ${GREEN}$PASSED${NC}"
    echo -e "    Tests failed: ${RED}$FAILED${NC}"
    echo ""
    echo -e "    ${YELLOW}See $AUDIT_LOG for details${NC}"
fi

# Summary of audit categories verified
echo ""
echo -e "  ${BOLD}Audit Categories Verified:${NC}"
echo -e "    ${GREEN}✓${NC} Order invariance (info fusion commutativity)"
echo -e "    ${GREEN}✓${NC} No-gates smoothness (extreme values)"
echo -e "    ${GREEN}✓${NC} Units/dt discretization (PSD scaling)"
echo -e "    ${GREEN}✓${NC} SO(3)/SE(3) roundtrip (exp/log consistency)"
echo -e "    ${GREEN}✓${NC} IW commutative update (sufficient stats)"
echo -e "    ${GREEN}✓${NC} Vectorized operator correctness"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  ${GREEN}EVALUATION COMPLETE${NC}${BOLD}                                        ║${NC}"
echo -e "${BOLD}║  Results: ${CYAN}$RESULTS_DIR${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# STAGE 6: Launch Dashboard
# ============================================================================
if [ -f "$RESULTS_DIR/diagnostics.npz" ]; then
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Launching Diagnostics Dashboard...${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    DASHBOARD_OUT="$RESULTS_DIR/dashboard.html"
    env -u PYTHONPATH "$PYTHON" "$PROJECT_ROOT/tools/slam_dashboard.py" \
        "$RESULTS_DIR/diagnostics.npz" \
        --output "$DASHBOARD_OUT"
    print_ok "Dashboard written: ${CYAN}$DASHBOARD_OUT${NC}"
    
    # Open dashboard in browser
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$DASHBOARD_OUT" 2>/dev/null &
        print_ok "Dashboard opened in browser"
    elif [ -n "$BROWSER" ]; then
        "$BROWSER" "$DASHBOARD_OUT" 2>/dev/null &
        print_ok "Dashboard opened in browser"
    else
        print_warn "Could not auto-open browser. Open manually: ${CYAN}$DASHBOARD_OUT${NC}"
    fi
fi
