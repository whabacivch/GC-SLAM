#!/usr/bin/env bash
# FL-SLAM full integration test (rosbag + basic health checks).
#
# Defaults to M3DGR Dynamic01 rosbag (Livox + RGB-D).
#
# Overrides:
#   BAG_PATH=/path/to/bag ./tools/test-integration.sh
#   TIMEOUT_SEC=120 STARTUP_SEC=25 ./tools/test-integration.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BAG_PATH="${BAG_PATH:-${PROJECT_DIR}/rosbags/m3dgr/Dynamic01_ros2}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
STARTUP_SEC="${STARTUP_SEC:-25}"
REQUIRE_LOOP="${REQUIRE_LOOP:-1}"
REQUIRE_SLAM_ACTIVE="${REQUIRE_SLAM_ACTIVE:-1}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
echo "=========================================="
echo "FL-SLAM Integration Test Suite"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Bag:          ${BAG_PATH}"
echo "  Timeout:      ${TIMEOUT_SEC}s"
echo "  Startup wait: ${STARTUP_SEC}s"
echo "  Require loop: ${REQUIRE_LOOP}"
echo "  SLAM active:  ${REQUIRE_SLAM_ACTIVE}"
echo ""

if [[ -d "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "ERROR: Python venv not found at ${VENV_PATH}" >&2
  echo "Create it with: python3 -m venv \"${VENV_PATH}\"" >&2
  exit 1
fi

echo "Running preflight checks..."
python3 - <<PY
import os
import sys
import traceback

project_root = r"""${PROJECT_DIR}"""
pkg_root = os.path.join(project_root, "fl_ws", "src", "fl_slam_poc")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

try:
    from fl_slam_poc.common.jax_init import jax
except Exception as exc:
    print(f"ERROR: Failed to initialize JAX: {exc}")
    traceback.print_exc()
    sys.exit(1)

devices = jax.devices()
if not any(d.platform == "gpu" for d in devices):
    print(f"ERROR: JAX GPU backend not available. Detected devices: {devices}")
    sys.exit(1)

print(f"Preflight OK: jax {jax.__version__}, devices={devices}")
PY
ROS_SETUP="/opt/ros/jazzy/setup.bash"
WS_ROOT="${PROJECT_DIR}/fl_ws"
if [[ ! -d "${BAG_PATH}" && ! -f "${BAG_PATH}" ]]; then
  echo "ERROR: Rosbag not found at ${BAG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ERROR: ROS 2 Jazzy not found at ${ROS_SETUP}" >&2
  exit 1
fi
set +u
source "${ROS_SETUP}"
set -u
INSTALL_SETUP="${WS_ROOT}/install/setup.bash"
if [[ ! -f "${INSTALL_SETUP}" ]]; then
  echo "Building workspace..."
  (cd "${WS_ROOT}" && colcon build --symlink-install)
fi
set +u
source "${INSTALL_SETUP}"
set -u
export ROS_HOME="${ROS_HOME:-${PROJECT_DIR}/.ros}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${PROJECT_DIR}/.ros/log}"
export RMW_FASTRTPS_USE_SHM="${RMW_FASTRTPS_USE_SHM:-0}"
mkdir -p "${ROS_LOG_DIR}"
LOG_DIR="${PROJECT_DIR}/diagnostic_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/integration_test_${TS}.log"
cleanup() {
  if [[ -n "${LAUNCH_PID:-}" ]]; then
    kill -INT -- "-${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
echo "=========================================="
echo "Starting Integration Test"
echo "=========================================="
echo "Log: ${RUN_LOG}"
echo ""
# MVP integration test targets the M3DGR rosbag launch only.
LAUNCH_FILE="poc_m3dgr_rosbag.launch.py"
setsid timeout "${TIMEOUT_SEC}" ros2 launch fl_slam_poc "${LAUNCH_FILE}" \
  play_bag:=true \
  bag:="${BAG_PATH}" \
  use_sim_time:=true \
  > >(tee "${RUN_LOG}") 2>&1 &
LAUNCH_PID=$!
echo "Waiting ${STARTUP_SEC}s for system startup..."
sleep "${STARTUP_SEC}"
echo ""
echo "=========================================="
echo "System Validation Checks"
echo "=========================================="
echo ""
anchor_ok=0
imu_ok=0
loop_ok=0
backend_ok=0
backend_mode=""
echo "Check 1: Anchor creation"
if ros2 topic echo --once --timeout 5 /sim/anchor_create >/dev/null 2>&1; then
  anchor_ok=1
  echo "  ✓ Detected /sim/anchor_create topic"
elif rg -qE "Created anchor|Backend received anchor" "${RUN_LOG}"; then
  anchor_ok=1
  echo "  ✓ Detected anchor creation in logs"
else
  echo "  ✗ No anchor creation detected"
fi
echo "Check 2: IMU segment processing"
if ros2 topic echo --once --timeout 5 /sim/imu_segment >/dev/null 2>&1; then
  imu_ok=1
  echo "  ✓ Detected /sim/imu_segment topic"
elif rg -qE "IMU segment.*applied|Published IMU segment" "${RUN_LOG}"; then
  imu_ok=1
  echo "  ✓ Detected IMU segment processing in logs"
else
  echo "  ⊘ No IMU segment processing detected (IMU fusion may be disabled)"
fi
echo "Check 3: Loop closure detection"
if ros2 topic echo --once --timeout 5 /sim/loop_factor >/dev/null 2>&1; then
  loop_ok=1
  echo "  ✓ Detected /sim/loop_factor topic"
elif rg -qE "Published loop factor|Backend received loop factor" "${RUN_LOG}"; then
  loop_ok=1
  echo "  ✓ Detected loop factor in logs"
else
  echo "  ✗ No loop factor detected"
fi
echo "Check 3: Backend status"
for _ in $(seq 1 10); do
  backend_json="$(
    ros2 topic echo --once --timeout 2 /cdwm/backend_status --field data --full-length 2>/dev/null \
      | head -n 1 || true
  )"
  if [[ -z "${backend_json:-}" ]]; then
    continue
  fi
  backend_mode="$(
    python3 - <<'PY' "${backend_json}" 2>/dev/null || true
import json, sys
raw = (sys.argv[1] if len(sys.argv) > 1 else "").strip().splitlines()[0].strip()
try:
    obj = json.loads(raw)
    print(obj.get("mode", ""))
except Exception:
    print("")
PY
  )"
  if [[ -n "${backend_mode}" ]]; then
    backend_ok=1
  fi
  if [[ "${backend_mode}" == "SLAM_ACTIVE" ]]; then
    break
  fi
  sleep 1
done
if [[ "${backend_ok}" -eq 1 ]]; then
  echo "  ✓ Backend status: ${backend_mode}"
elif rg -q "Backend status: mode=" "${RUN_LOG}"; then
  backend_ok=1
  backend_mode="$(rg "Backend status: mode=" "${RUN_LOG}" | tail -n 1 | sed -E 's/.*mode=([^, ]+).*/\1/')"
  echo "  ✓ Backend status from log: ${backend_mode}"
else
  echo "  ✗ No backend status detected"
fi
cleanup
LAUNCH_PID=""
echo ""
echo "=========================================="
echo "Test Results"
echo "=========================================="
echo ""
fail=0
if [[ "${anchor_ok}" -ne 1 ]]; then
  echo "✗ FAIL: No anchor creation observed"
  fail=1
else
  echo "✓ PASS: Anchor creation detected"
fi
if [[ "${imu_ok}" -eq 1 ]]; then
  echo "✓ PASS: IMU segment processing detected"
else
  echo "⊘ SKIP: IMU segment processing not detected (IMU fusion may be disabled)"
fi
if [[ "${REQUIRE_LOOP}" -eq 1 && "${loop_ok}" -ne 1 ]]; then
  echo "✗ FAIL: No loop closure observed (required)"
  fail=1
elif [[ "${loop_ok}" -eq 1 ]]; then
  echo "✓ PASS: Loop closure detected"
else
  echo "⊘ SKIP: Loop closure not required"
fi
if [[ "${backend_ok}" -ne 1 ]]; then
  echo "✗ FAIL: No backend status observed"
  fail=1
else
  echo "✓ PASS: Backend running (mode: ${backend_mode})"
fi
if [[ "${REQUIRE_SLAM_ACTIVE}" -eq 1 && "${backend_mode}" != "SLAM_ACTIVE" ]]; then
  echo "✗ FAIL: Backend mode is not SLAM_ACTIVE (got '${backend_mode}')"
  fail=1
elif [[ "${backend_mode}" == "SLAM_ACTIVE" ]]; then
  echo "✓ PASS: Backend in SLAM_ACTIVE mode"
fi
echo ""
if [[ "${fail}" -eq 0 ]]; then
  echo "=========================================="
  echo "✓ ALL INTEGRATION TESTS PASSED"
  echo "=========================================="
  echo ""
  echo "Log saved to: ${RUN_LOG}"
  echo ""
else
  echo "=========================================="
  echo "✗ INTEGRATION TESTS FAILED"
  echo "=========================================="
  echo ""
  echo "Review the log for details: ${RUN_LOG}"
  echo ""
  exit 1
fi
