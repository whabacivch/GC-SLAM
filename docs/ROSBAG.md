# Rosbag Workflow (Canonical)

This is the single, current workflow for running FL-SLAM against recorded data.

This file is the source of truth. If other rosbag docs exist, treat them as historical notes.

For complete testing documentation, see **[TESTING.md](TESTING.md)**.

## Quick Start (M3DGR MVP)

1. Build:

`cd fl_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fl_slam_poc && source install/setup.bash`

2. Run MVP pipeline (SLAM + metrics + plots):

`bash scripts/run_and_evaluate.sh`

Results are saved to `results/m3dgr_YYYYMMDD_HHMMSS/`.

## Quick Start (TB3 / Alternative Integration)

If you don't have the bag yet:

`./scripts/download_tb3_rosbag.sh`

Run the integration test (writes logs to `diagnostic_logs/`):

`./scripts/test-integration.sh`

Override bag path if needed:

`BAG_PATH=/absolute/path/to/bag_dir ./scripts/test-integration.sh`

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
- MVP: `fl_ws/src/fl_slam_poc/launch/poc_m3dgr_rosbag.launch.py`
- Alternative (Phase 2): `phase2/fl_ws/src/fl_slam_poc/launch/poc_tb3_rosbag.launch.py`

## Troubleshooting

- If you see TF errors or missing transforms, first verify bag frames with:
  - `bash scripts/inspect_rosbag_topics.sh /path/to/bag_dir`
- If backend stays `DEAD_RECKONING`, it is not receiving loop factors from the frontend.
