# FL-SLAM Testing Guide

This document describes the consolidated testing framework for the FL-SLAM project.

## Testing Philosophy

The FL-SLAM testing framework is organized into two tiers:

1. **MVP Evaluation (M3DGR)** - End-to-end SLAM + metrics/plots via `run_and_evaluate.sh`
2. **Integration Tests (Alternative)** - End-to-end system validation with rosbag data via `test-integration.sh`

**Note:** The previous Docker-based harness and minimal-test scripts were removed from the active workflow. Historical versions live under `archive/` for reference.

## Test Scripts Overview

### MVP Evaluation (M3DGR)

**Purpose:** Run the current MVP pipeline (rosbag SLAM + evaluation) end-to-end.

**When to use:**
-- During algorithm/debug iterations (drift, timestamps, loop closures)
-- Before merging changes that affect the runtime pipeline

**Execution:**
```bash
bash tools/run_and_evaluate.sh
```

### Integration Testing (Alternative)

**Purpose:** Validate the complete SLAM pipeline with real sensor data, including loop closure detection and backend optimization.

**Native execution:**
```bash
./tools/test-integration.sh
```

**What it tests:**
- ✓ Full ROS 2 node launch and communication
- ✓ Rosbag replay with TurtleBot3 data
- ✓ Anchor creation by frontend
- ✓ Loop closure detection
- ✓ Backend state estimation (SLAM_ACTIVE mode)
- ✓ Foxglove visualization bridge (optional)
- ✓ End-to-end SLAM pipeline

**When to use:**
- Before releasing new versions
- After major architectural changes
- Weekly regression testing
- When debugging integration issues

## Configuration

Both test scripts support environment variables for customization:

### Integration Test Configuration (Alternative)

```bash
# Rosbag path (default: rosbags/tb3_slam3d_small_ros2)
export BAG_PATH=/path/to/custom/bag

# Timeout for full test run (default: 90 seconds)
export TIMEOUT_SEC=120

# Startup wait before validation (default: 20 seconds)
export STARTUP_SEC=30

# Require loop closure detection (default: 1)
export REQUIRE_LOOP=0  # Set to 0 to allow anchor-only tests

# Require SLAM_ACTIVE backend mode (default: 1)
export REQUIRE_SLAM_ACTIVE=0  # Set to 0 to allow other modes

# Enable Foxglove visualization (default: 1)
export ENABLE_FOXGLOVE=0  # Set to 0 to disable

./tools/test-integration.sh
```

## Test Data

The integration tests require the TurtleBot3 SLAM rosbag dataset.

**Download test data:**
```bash
./tools/download_tb3_rosbag.sh
```

This script:
1. Downloads the ROS1 bag from ROBOTIS Japan GitHub
2. Converts it to ROS2 format using `rosbag2_bag_v2`
3. Saves to `rosbags/tb3_slam3d_small_ros2/`

**Inspect bag contents:**
```bash
./tools/inspect_rosbag_topics.sh [path/to/bag]
```

## Native Testing Workflow

Requirements:
- ROS 2 Jazzy
- Built workspace (`colcon build`)

```bash
# Unit tests (operators/models wiring + invariants)
cd fl_ws/src/fl_slam_poc
pytest -q

# Full validation (alternative integration)
cd -
./tools/test-integration.sh

# MVP evaluation (M3DGR)
bash tools/run_and_evaluate.sh
```

## Understanding Test Results

### Integration Test Output

```
Check 1: Anchor creation
  ✓ Detected /sim/anchor_create topic

Check 2: Loop closure detection
  ✓ Detected loop factor in logs

Check 3: Backend status
  ✓ Backend status: SLAM_ACTIVE

✓ PASS: Anchor creation detected
✓ PASS: Loop closure detected
✓ PASS: Backend running (mode: SLAM_ACTIVE)
✓ PASS: Backend in SLAM_ACTIVE mode

✓ ALL INTEGRATION TESTS PASSED
```

## Debugging Failed Tests

### Minimal Tests Fail

1. Check import errors - ensure workspace is built:
   ```bash
   cd fl_ws
   colcon build --symlink-install --packages-select fl_slam_poc
   source install/setup.bash
   ```

2. Run individual test files:
   ```bash
   cd fl_ws/src/fl_slam_poc
   .venv/bin/python -m pytest test/test_audit_invariants.py -v
   .venv/bin/python -m pytest test/test_rgbd_multimodal.py -v
   ```

### Integration Tests Fail

1. Check the detailed log:
   ```bash
   cat diagnostic_logs/integration_test_*.log
   ```

2. Review specific failures:
   - **No anchor creation**: Frontend not processing odometry correctly
   - **No loop closure**: Loop detection threshold too strict or insufficient motion
   - **Backend not SLAM_ACTIVE**: Check for errors in backend node startup

3. Run with extended timeout:
   ```bash
   TIMEOUT_SEC=180 STARTUP_SEC=30 ./tools/test-integration.sh
   ```

4. Disable Foxglove if it's causing issues:
   ```bash
   ENABLE_FOXGLOVE=0 ./tools/test-integration.sh
   ```

## CI/CD Integration (Conceptual)

This repo is primarily validated in a ROS 2 environment. A practical CI pipeline (when set up) should:
- Build the ROS 2 workspace (`colcon build`)
- Run unit tests (`pytest` under `fl_ws/src/fl_slam_poc/`)
- Optionally run `tools/test-integration.sh` if rosbag assets are available in CI

## Test Coverage

### Currently Tested

- ✓ Information geometry operators
- ✓ SE(3) operations
- ✓ ICP solver
- ✓ Adaptive models (NIG, process noise)
- ✓ Frontend anchor creation
- ✓ Loop closure detection
- ✓ Backend optimization
- ✓ RGB-D processing
- ✓ Multimodal fusion

### Not Yet Tested

- ⊘ Gazebo simulation integration
- ⊘ Live sensor streams (non-rosbag)
- ⊘ Performance benchmarks
- ⊘ Memory leak detection
- ⊘ Multi-robot scenarios

## Contributing

When adding new features:

1. Add unit tests to `test/test_audit_invariants.py` or `test/test_rgbd_multimodal.py`
2. Ensure `pytest -q` passes under `fl_ws/src/fl_slam_poc/`
3. Verify `tools/test-integration.sh` and/or `tools/run_and_evaluate.sh` still pass when applicable
4. Update this document if adding new test workflows

## Summary

| Workflow | Typical Duration | Use Case | Requires Bag |
|----------|------------------|----------|--------------|
| `pytest -q` (under `fl_ws/src/fl_slam_poc/`) | ~seconds | Unit/invariant checks | No |
| `tools/test-integration.sh` | ~minutes | Alternative end-to-end check | Yes |
| `tools/run_and_evaluate.sh` | ~minutes | MVP M3DGR evaluation | Yes |
