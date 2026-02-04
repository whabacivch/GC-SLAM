# SLAM Dataset Download Guide

This project uses the **Kimera** dataset for evaluation. Other datasets are archived (see `archive/docs/`).

---

## Kimera (current dataset)

- **Source:** [MIT-SPARK/Kimera-Multi-Data](https://github.com/MIT-SPARK/Kimera-Multi-Data)
- **Location:** `rosbags/Kimera_Data/`
- **Config:** `fl_ws/src/fl_slam_poc/config/gc_kimera.yaml`
- **Dataset, frames, calibration, topics:** [KIMERA_DATASET_AND_PIPELINE.md](../KIMERA_DATASET_AND_PIPELINE.md)

### Canonical bag (only one we use; same as run_and_evaluate_gc.sh)

- **Bag:** `rosbags/Kimera_Data/ros2/10_14_acl_jackal-005`
- **Ground truth:** `rosbags/Kimera_Data/ground_truth/1014/acl_jackal_gt.tum`
- **Extrinsics:** `rosbags/Kimera_Data/calibration/robots/acl_jackal/extrinsics.yaml` (convert with `tools/kimera_calibration_to_gc.py`)

### Sensors (Kimera acl_jackal)

- Velodyne VLP-16 LiDAR @ ~10 Hz (`/acl_jackal/lidar_points`)
- Wheel odometry (`/acl_jackal/jackal_velocity_controller/odom`)
- Forward IMU (`/acl_jackal/forward/imu`)
- Forward RGB/depth (optional; see launch camera args)

### Download / prep

1. Follow the Kimera-Multi-Data README to obtain rosbags and calibration.
2. Place ros2 bags under `rosbags/Kimera_Data/ros2/` and ground truth under `rosbags/Kimera_Data/ground_truth/<seq>/`.
3. Run `python tools/kimera_calibration_to_gc.py rosbags/Kimera_Data/calibration/robots/acl_jackal/extrinsics.yaml --apply` to refresh `gc_kimera.yaml` if needed.
4. Run evaluation: `bash tools/run_and_evaluate_gc.sh`

---

## Other datasets (archived)

Other archived datasets and related docs are in `archive/docs/`. They are not used by the current evaluation script.
