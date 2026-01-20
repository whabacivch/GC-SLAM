# Rosbag Workflow (Canonical)

This is the single, current workflow for running FL-SLAM against recorded data.

This file is the source of truth. If other rosbag docs exist, treat them as historical notes.

For complete testing documentation, see **[TESTING.md](TESTING.md)**.

## Quick Start (Local)

1. Build:

`cd fl_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fl_slam_poc && source install/setup.bash`

If you don't have the bag yet:

`./scripts/download_tb3_rosbag.sh`

2. Run integration test (asserts anchors + loop factors + backend mode; writes logs to `diagnostic_logs/`):

`./scripts/test-integration.sh`

Override bag path if needed:

`BAG_PATH=/absolute/path/to/bag_dir ./scripts/test-integration.sh`

## Quick Start (Docker)

`./scripts/docker-test-integration.sh`

Foxglove: connect to `ws://localhost:8765`.

## Pass/Fail Meaning

The integration test is considered a true "SLAM is working" check when all of these are observed:
- `/sim/anchor_create` published at least once
- `/sim/loop_factor` published at least once
- `/cdwm/backend_status` reports `mode: "SLAM_ACTIVE"`

**Note:** Use `scripts/record_test_bag.sh` to record a proper test bag with loop closures from Gazebo.

If you want a weaker check (anchors-only), run:

`REQUIRE_LOOP=0 REQUIRE_SLAM_ACTIVE=0 ./scripts/test-integration.sh`

## Notes

- The default TB3 3D SLAM sample bag includes `/scan`, `/odom`, `/tf`, `/tf_static` and stereo *compressed* image topics (not raw `/camera/*`). The rosbag launch defaults `enable_image:=false enable_depth:=false enable_camera_info:=false` to avoid noisy "sensor missing" warnings.
- The rosbag launch uses a packaged QoS override file at `fl_ws/src/fl_slam_poc/config/qos_override.yaml` so `/tf_static` behaves like a true static transform channel for late-joining subscribers.

## Launch Entry Point

Rosbag launch file:
- `fl_ws/src/fl_slam_poc/launch/poc_tb3_rosbag.launch.py`

## Troubleshooting

- If you see TF errors or missing transforms, first verify bag frames with:
  - `python3 scripts/inspect_bag_direct.py /path/to/bag_dir`
- If backend stays `DEAD_RECKONING`, it is not receiving loop factors from the frontend.
