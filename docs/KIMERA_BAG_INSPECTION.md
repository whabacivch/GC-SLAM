# Kimera Bag Inspection Checklist

Use this when you have a Kimera ROS 2 bag and need to fill in frame conventions, extrinsics, covariances, and timing. All scripts assume bag path is the directory containing `*.db3` (or path to a single `.db3`).

**Reference:** [KIMERA_FRAME_MAPPING.md](KIMERA_FRAME_MAPPING.md), [VELODYNE_VLP16.md](VELODYNE_VLP16.md), [POINTCLOUD2_LAYOUTS.md](POINTCLOUD2_LAYOUTS.md).

---

## All-in-one: inspect and apply extrinsics

From the project root, with a Kimera bag at `BAG_PATH` and ROS 2 sourced (e.g. `source /opt/ros/jazzy/setup.bash`):

```bash
# Inspect only (report to stdout; no config changes)
python tools/inspect_kimera_bag.py /path/to/Kimera_Data/ros2/10_14_acl_jackal-005_ros2

# Inspect and write estimated T_base_lidar / T_base_imu into gc_kimera.yaml (no longer placeholders)
python tools/inspect_kimera_bag.py /path/to/bag_dir --apply
```

This runs: first-N messages summary, diagnose_coordinate_frames, estimate_lidar_base_extrinsic_rotation_from_ground (PointCloud2), estimate_imu_base_extrinsic_rotation. Parsed extrinsics use **rotation only** (translation 0,0,0). To get translation for LiDAR (e.g. height above base), use dataset/community calibration or run `estimate_lidar_base_extrinsic.py` if adapted for PointCloud2.

---

## 1. Frame names and topic layout

**Script:** `tools/inspect_rosbag_deep.py` or `tools/validate_frame_conventions.py`

- Confirm `/odom` (or Kimera odom topic) `header.frame_id` and `child_frame_id`.
- Confirm LiDAR PointCloud2 topic name and `header.frame_id`.
- Confirm IMU topic name and `header.frame_id`.

**Record in:** `docs/KIMERA_FRAME_MAPPING.md` “Frame names” table and `config/gc_kimera.yaml` (odom_frame, base_frame; hub input topics).

---

## 2. First-N messages (fields, frame_id, sample values)

**Script:** `tools/first_n_messages_summary.py`

```bash
python tools/first_n_messages_summary.py /path/to/Kimera_Data/ros2/<seq>_ros2 --n 25 --md /tmp/kimera_first25.md
```

- **PointCloud2:** Check field names (x, y, z, ring; optional t/time). Confirm frame_id. Use to validate `pointcloud_layout: vlp16` and docs.
- **Imu:** Check orientation_covariance, angular_velocity_covariance, linear_acceleration_covariance (all-zero vs meaningful). Record in KIMERA_FRAME_MAPPING “IMU and odom covariances”.
- **Odometry:** Check pose.covariance and twist.covariance (diagonal or full). Record whether ROS [x,y,z,roll,pitch,yaw] ordering makes sense.

**Record in:** `docs/KIMERA_FRAME_MAPPING.md` Discovery section (IMU/odom covariances); optionally keep `/tmp/kimera_first25.md` or commit a sample.

---

## 3. LiDAR Z-up/Z-down and odom covariance ordering

**Script:** `tools/diagnose_coordinate_frames.py`

```bash
python tools/diagnose_coordinate_frames.py /path/to/Kimera_Data/ros2/<seq>_ros2 \
  --lidar-topic /acl_jackal/lidar_points \
  --imu-topic /acl_jackal/forward/imu \
  --odom-topic /acl_jackal/jackal_velocity_controller/odom \
  --n-scans 20
```

- **LiDAR:** Note “Z-UP” vs “Z-DOWN” (and suggested T_base_lidar rotation if Z-down).
- **IMU:** Note gravity/specific-force direction and any suggested T_base_imu correction.
- **Odom:** Note “ROS [x,y,z,roll,pitch,yaw]” vs “LEGACY PERMUTED”.

**Record in:** `docs/KIMERA_FRAME_MAPPING.md` “Diagnostic results” subsection (paste interpretation and, if needed, update axis conventions).

---

## 4. Extrinsics (T_base_lidar, T_base_imu)

**Current state:** Kimera config uses **placeholder** extrinsics (identity): `T_base_lidar: [0,0,0,0,0,0]`, `T_base_imu: [0,0,0,0,0,0]`. These are **not** the real sensor poses.

**Options:**

1. **Dataset/community:** [plusk01/Kimera-Multi-Data branch `parker/kmd_tools`](https://github.com/plusk01/Kimera-Multi-Data/tree/parker/kmd_tools) may provide LiDAR/IMU–base extrinsics. If format differs (quat, T_sensor_base), convert to GC format `[x,y,z,rx,ry,rz]` (m, rad, T_base_sensor) and either:
   - Put values in `config/gc_kimera.yaml` under `T_base_lidar` and `T_base_imu`, or
   - Write YAML files and set `extrinsics_source: file`, `T_base_lidar_file`, `T_base_imu_file`.
2. **Estimation:** Use `tools/estimate_lidar_base_extrinsic_rotation_from_ground.py` and `tools/estimate_imu_base_extrinsic_rotation.py` (or `estimate_lidar_base_extrinsic.py`, `estimate_imu_base_extrinsic.py`) on the bag to get rotation/translation, then convert to 6D and put in config.

After updating extrinsics, run:

```bash
python tools/check_extrinsics.py fl_ws/src/fl_slam_poc/config/gc_kimera.yaml
```

to print loaded T_base_lidar, T_base_imu and frame names.

---

## 5. Timing (optional)

- **Per-point timestamps:** From first-N summary, check if PointCloud2 has `t` or `time` field and units (s vs ns).
- **Deskew:** If running pipeline, confirm `t_scan` (header.stamp) and scan bounds (~0.1 s at 10 Hz for VLP-16) are consistent. Document in KIMERA_FRAME_MAPPING “Timing” if needed.

---

## Quick command summary

| Goal                         | Command |
|-----------------------------|---------|
| Frame names + topic layout  | `python tools/inspect_rosbag_deep.py <bag_dir>` |
| First N (fields, frame, cov)| `python tools/first_n_messages_summary.py <bag_dir> --n 25 [--md out.md]` |
| Z-up/Z-down + odom ordering| `python tools/diagnose_coordinate_frames.py <bag_dir> --lidar-topic /acl_jackal/lidar_points --imu-topic /acl_jackal/forward/imu --odom-topic /acl_jackal/jackal_velocity_controller/odom` |
| Print extrinsics from config| `python tools/check_extrinsics.py <config_path>` |
| **All-in-one (inspect + optional apply)** | `python tools/inspect_kimera_bag.py <bag_dir> [--apply]` (requires ROS 2 sourced) |
