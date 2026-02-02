# Kimera Calibration and Frame (GC v2)

This document summarizes **where frame and extrinsics info live** in the Kimera bag directory and how GC v2 uses them. Use it to fix evaluation alignment and to confirm we are reading the right calibration.

**Reference:** [KIMERA_FRAME_MAPPING.md](KIMERA_FRAME_MAPPING.md), [FRAME_AND_QUATERNION_CONVENTIONS.md](FRAME_AND_QUATERNION_CONVENTIONS.md).

---

## 1. Directory layout (Kimera_Data)

All paths below are under `rosbags/Kimera_Data/`.

| Path | Purpose |
|------|--------|
| **calibration/README.md** | Conventions: `T_a_b` maps frame b → frame a; `T_BS` = sensor w.r.t. body. |
| **calibration/robots/<robot>/extrinsics.yaml** | Per-robot 4×4 transforms: `T_baselink_lidar`, `T_cameralink_gyro`, etc. |
| **calibration/extrinsics_manifest.yaml** | Index of all robots and which transform names each provides. |
| **calibration/sensor_params_manifest.yaml** | Summary of T_BS from Kimera-style YAMLs. |
| **dataset_ready_manifest.yaml** | Ready-to-use mapping: `ros2_bag`, `ground_truth_tum`, `extrinsics` (path to robots/<robot>/extrinsics.yaml). |
| **PREP_README.md** | GT format (TUM: timestamp x y z qx qy qz qw); sequences 1014, 1207, 1208. |
| **ground_truth/<seq>/<robot>_gt.tum** | Ground truth trajectory (e.g. `1014/acl_jackal_gt.tum`). |

For **acl_jackal** (bag `10_14_acl_jackal-005`):

- **Extrinsics:** `calibration/robots/acl_jackal/extrinsics.yaml`
- **GT:** `ground_truth/1014/acl_jackal_gt.tum`

---

## 2. How GC gets extrinsics

- **Tool:** `tools/kimera_calibration_to_gc.py` reads `robots/acl_jackal/extrinsics.yaml` and outputs GC format `[x, y, z, rx, ry, rz]` (translation m, rotvec rad).
  - `T_baselink_lidar` → **T_base_lidar** (GC: T_{base←lidar}).
  - `T_cameralink_gyro` → **T_base_imu** (translation from calib; rotation can be overridden with `--imu-rotation` for bag-estimated IMU frame).
- **Config:** `fl_ws/src/fl_slam_poc/config/gc_kimera.yaml` already contains these values (filled from that tool or from the dataset):
  - `T_base_lidar: [-0.039685, -0.067961, 0.147155, -0.006787, -0.097694, 0.001931]`
  - `T_base_imu:   [-0.016020, -0.030220, 0.007400, -1.602693, 0.002604, 0.000000]`
- **Eval script:** `tools/run_and_evaluate_gc.sh` (PROFILE=kimera) passes the same T_base_lidar and T_base_imu via launch args (from the script); they match gc_kimera.yaml and thus the Kimera calibration files.

So we **are** reading extrinsics from the Kimera directory (via gc_kimera.yaml / kimera_calibration_to_gc.py). No change needed for extrinsics source.

---

## 3. Ground truth frame vs estimate frame (why ATE can be wrong)

- **Our estimate:** Trajectory is in the **anchor frame** (first odom as origin), Z-up planar, exported in TUM as (timestamp, x, y, z, qx, qy, qz, qw).
- **Kimera GT:** Same TUM format; dataset_ready_manifest says *"Align frames to GC v2 conventions before eval."* The GT file may be in a **different world frame** (e.g. different “up” axis or forward axis). If the two frames differ, **initial-pose alignment** (GT → estimate at first pose) still leaves axis semantics mismatched (e.g. our Z ≈ const, GT Y = forward), giving huge ATE/RPE.
- **What to do:**  
  1. Confirm GT frame: check dataset docs or Kimera-Multi-Data repo for the world frame of `<robot>_gt.tum` (axes and handedness).  
  2. If GT uses a different convention, add a **GT → estimate frame** transform (e.g. rotation + optional flip) in the eval pipeline (e.g. in `align_ground_truth.py` or `evaluate_slam.py`) so both trajectories are in the same convention before alignment and ATE/RPE.  
  3. Until then, treat very large ATE/RPE as a **frame/convention mismatch**; see the note printed by `run_and_evaluate_gc.sh` when ATE > 10 m or RPE @ 1 m > 1.

---

## 4. Quick reference: files we read

| What | File(s) |
|------|--------|
| LiDAR/IMU extrinsics (acl_jackal) | `rosbags/Kimera_Data/calibration/robots/acl_jackal/extrinsics.yaml` |
| GC config (Kimera profile) | `fl_ws/src/fl_slam_poc/config/gc_kimera.yaml` |
| Convert calib → GC | `python tools/kimera_calibration_to_gc.py rosbags/Kimera_Data/calibration/robots/acl_jackal/extrinsics.yaml` |
| Bag ↔ GT ↔ extrinsics | `rosbags/Kimera_Data/dataset_ready_manifest.yaml` |
| GT trajectory (10_14 acl_jackal) | `rosbags/Kimera_Data/ground_truth/1014/acl_jackal_gt.tum` |
