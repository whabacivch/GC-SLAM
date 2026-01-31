# M3DGR Dynamic01 — Archived Default Dataset

**Status:** Archived. The default evaluation profile is now **Kimera** (`PROFILE=kimera`). M3DGR Dynamic01 is no longer the default; it remains supported via `PROFILE=m3dgr` if you have the bag.

## Why archived

- Project primary evaluation moved to Kimera datasets (see `docs/KIMERA_FRAME_MAPPING.md`).
- Dynamic01 is no longer used by default in `tools/run_and_evaluate_gc.sh`.

## How to run with Dynamic01 (if you have the bag)

1. Place the bag and ground truth as follows:
   - Bag: `rosbags/m3dgr/Dynamic01_ros2/` (ROS 2, from converted `Dynamic01.bag`)
   - Ground truth: `rosbags/m3dgr/Dynamic01.txt` (TUM format)

2. Run evaluation with the m3dgr profile:
   ```bash
   bash tools/run_and_evaluate_gc.sh PROFILE=m3dgr
   ```
   Or set bag/GT explicitly:
   ```bash
   BAG_PATH=rosbags/m3dgr/Dynamic01_ros2 GT_FILE=rosbags/m3dgr/Dynamic01.txt bash tools/run_and_evaluate_gc.sh
   ```

## Profile parameters (m3dgr)

When `PROFILE=m3dgr`, the script sets:

| Variable        | Default value |
|----------------|----------------|
| BAG_PATH       | `$PROJECT_ROOT/rosbags/m3dgr/Dynamic01_ros2` |
| GT_FILE        | `$PROJECT_ROOT/rosbags/m3dgr/Dynamic01.txt` |
| CONFIG_PATH    | (empty → launch uses gc_unified.yaml) |
| BODY_CALIB     | `config/m3dgr_body_T_wheel.yaml` |
| ODOM_FRAME     | `odom` |
| BASE_FRAME     | `base_footprint` |
| POINTCLOUD_LAYOUT | `livox` |
| LIDAR_SIGMA_MEAS | `0.01` |
| IMU_ACCEL_SCALE | `9.81` |

## References

- Dataset: `docs/datasets/M3DGR_STATUS.md`, `docs/datasets/DATASET_DOWNLOAD_GUIDE.md`
- Frames/conventions: `docs/FRAME_AND_QUATERNION_CONVENTIONS.md` (CONFIRMED Dynamic01_ros2 items)
- Topics: `docs/BAG_TOPICS_AND_USAGE.md` (M3DGR Dynamic01 section)
