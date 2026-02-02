# Kimera Dataset Frame Mapping (GC v2)

This document is the **single source of truth** for mapping Kimera dataset frames to GC v2 parameters. For **where calibration and frame info live** in the Kimera bag directory and how to fix evaluation alignment, see [KIMERA_CALIBRATION_AND_FRAME.md](KIMERA_CALIBRATION_AND_FRAME.md). Frame names and conventions were taken from the Kimera bag sample summary `rosbags/Kimera_Data/ros2/10_14_acl_jackal-005_sample25.md` (first-25-messages inspection). When a physical Kimera ROS 2 bag is available, use **[KIMERA_BAG_INSPECTION.md](KIMERA_BAG_INSPECTION.md)** for the full inspection checklist (frames, first-N messages, Z-up/Z-down, extrinsics, timing); run `tools/validate_frame_conventions.py` or `tools/inspect_rosbag_deep.py` to confirm or update frame values.

**Reference:** [FRAME_AND_QUATERNION_CONVENTIONS.md](FRAME_AND_QUATERNION_CONVENTIONS.md) for GC internal conventions (Z-up world, base planar, T_{parent<-child}).

---

## Frame names (from Kimera bag sample)

| GC parameter / role | Kimera value | Topic (for reference) |
|--------------------|--------------|------------------------|
| **odom_frame** (world/parent) | `acl_jackal2/odom` | `/acl_jackal/jackal_velocity_controller/odom` `header.frame_id` |
| **base_frame** (body/child) | `acl_jackal2/base` | `/acl_jackal/jackal_velocity_controller/odom` `child_frame_id` |
| **LiDAR / pointcloud frame** | `acl_jackal2/velodyne_link` | `/acl_jackal/lidar_points` `header.frame_id` |
| **IMU frame** | `acl_jackal2/forward_imu_optical_frame` | `/acl_jackal/forward/imu` `header.frame_id` |

- **Odometry:** Pose is `T_{odom_frame <- base_frame}` = `T_{acl_jackal2/odom <- acl_jackal2/base}`. Do not invert at ingest.
- **LiDAR:** Points are in `acl_jackal2/velodyne_link`; extrinsics `T_base_lidar` = T_{base <- velodyne_link}.
- **IMU:** Measurements are in `acl_jackal2/forward_imu_optical_frame`; extrinsics `T_base_imu` = T_{base <- imu_frame}.

---

## Axis conventions

- **World/odom (`acl_jackal2/odom`):** Assumed Z-up, planar motion (odom sample has z=0, pose_cov marks z/roll/pitch with 1e6). Aligns with GC Z-up world.
- **Base (`acl_jackal2/base`):** Child of odom; planar base. Same as GC base_footprint semantics.
- **LiDAR (`acl_jackal2/velodyne_link`):** Velodyne VLP-16; see [VELODYNE_VLP16.md](VELODYNE_VLP16.md). Convention (Z-up vs Z-down) must be verified on a real bag with `tools/diagnose_coordinate_frames.py` (script supports PointCloud2; use `--lidar-topic /acl_jackal/lidar_points` for Kimera).
- **IMU (`acl_jackal2/forward_imu_optical_frame`):** D435i IMU in optical frame. Stationary accelerometer reads -Y (gravity in +Y). T_base_imu uses Rx(-90°) = rotvec `[-1.57, 0, 0]` to map IMU -Y to base +Z (cancels gravity_W = [0,0,-9.81]).

When a Kimera bag is available, run `tools/diagnose_coordinate_frames.py` and update this section with CONFIRMED/TO-CONFIRM per frame.

### "All motion in Z" symptom

If the exported trajectory shows **motion mainly along Z** while the robot actually moves in the XY plane, there is an axis or frame convention mismatch. Common causes:

1. **IMU extrinsic rotation** — If T_base_imu rotation is wrong, gravity doesn't cancel properly during IMU preintegration, causing Z drift at ~10 m/s (gravity rate). The D435i IMU reads -Y when stationary (gravity in +Y in optical frame); Rx(-90°) = rotvec `[-1.57, 0, 0]` maps this to base +Z to cancel gravity_W. **Wrong sign causes ~20 m/s² net Z acceleration.**
2. **Odom frame axis convention** — The bag's odom may express position in a frame where "forward" is Z. We now use actual `pos.x, pos.y, pos.z` with z-variance capped by `GC_ODOM_Z_VARIANCE_PRIOR` (default 1e6 m² = effectively don't trust odom z).
3. **State/export swap** — Less likely if code follows [trans(0:3), rot(3:6)] consistently.

**Diagnostic:** Run `tools/diagnose_trajectory_axes.py` on the bag and (optionally) the exported TUM file. It reports position ranges for X,Y,Z from the bag's odom and from the TUM file. If odom has motion in XY but TUM has motion in Z at ~10 m/s, suspect IMU extrinsic rotation. If TUM matches odom axes but is offset, suspect odom or visual evidence issues.

### Diagnostic results (run on Kimera bag)

Run with Kimera topics and paste the script output here to lock in Z-up/Z-down and odom ordering:

```bash
python tools/diagnose_coordinate_frames.py rosbags/Kimera_Data/ros2/10_14_acl_jackal-005 \
  --lidar-topic /acl_jackal/lidar_points \
  --imu-topic /acl_jackal/forward/imu \
  --odom-topic /acl_jackal/jackal_velocity_controller/odom \
  --n-scans 20
```

- **LiDAR Z-up/Z-down:** **CONFIRMED (10_14_acl_jackal-005):** LiDAR is Z-UP. Ground normal · Z ≈ 0.996; T_base_lidar rotation from ground-plane fit: small roll/pitch (~1.3°) applied in gc_kimera.yaml.
- **Odom twist direction:** For Jackal 2D base, twist is planar: `v_x`, `ω_z` in child frame. **CONFIRMED:** Odom covariance ordering is ROS [x,y,z,roll,pitch,yaw] (xy_yaw small, z_rp large).

---

## Ground truth frame vs GC anchor

- **Kimera GT:** From `ground_truth/<seq>/<robot>_gt.tum` (e.g. world frame “p” or dataset-specific). Timestamps in seconds (from CSV ns).
- **GC anchor:** Trajectory is exported in anchor frame (first odom as origin).
- **Evaluation:** We align GT timestamps to the estimate timeline with `align_ground_truth.py`, then use **initial-pose alignment** in `evaluate_slam.py` (GT → estimate frame at first pose). No extra transform needed unless GT is in a different body frame than GC base; then a body calib (e.g. body_T_wheel) can be used if provided.

---

## Config overlay (gc_kimera.yaml)

Set the following when running Kimera profile:

- `odom_frame`: `acl_jackal2/odom`
- `base_frame`: `acl_jackal2/base`
- LiDAR/pointcloud frame: `acl_jackal2/velodyne_link` (for hub/converter if needed)

Sensor hub input topics (Kimera):

- Odom: `/acl_jackal/jackal_velocity_controller/odom`
- LiDAR: `/acl_jackal/lidar_points`
- IMU: `/acl_jackal/forward/imu`

**Canonical bag:** We use only the bag in `run_and_evaluate_gc.sh`: **10_14_acl_jackal-005**. Topics are under `/acl_jackal/`; frame IDs in messages are `acl_jackal2/*`; extrinsics from `calibration/robots/acl_jackal/extrinsics.yaml`.

---

## Discovery (what the dataset provides)

### Extrinsics

- **Source:** [MIT-SPARK/Kimera-Multi-Data](https://github.com/MIT-SPARK/Kimera-Multi-Data) README: camera calibration (intrinsics + extrinsics) from Google Drive, Kimera-VIO format. LiDAR/IMU–base extrinsics: [plusk01/Kimera-Multi-Data branch `parker/kmd_tools`](https://github.com/plusk01/Kimera-Multi-Data/tree/parker/kmd_tools) (community-contributed; “not perfect” per README).
- **Format / direction / units:** When using that repo or converted files: document in this section whether transforms are T_base_sensor or T_sensor_base, quat order (xyzw vs wxyz), and units (m, rad). GC expects **T_base_sensor** (T_{base<-sensor}), rotvec in rad, translation in m; see [FRAME_AND_QUATERNION_CONVENTIONS.md](FRAME_AND_QUATERNION_CONVENTIONS.md).
- **Config:** Use `extrinsics_source: file` and `T_base_lidar_file` / `T_base_imu_file` (or single `extrinsics_file`) once GC-format YAML is produced; see config template and §Extrinsics below.

### Per-axis LiDAR variance

- **Discovery:** Dataset and Kimera-Multi-Data repo do not publish per-axis LiDAR variance (σ²_x, σ²_y, σ²_z or range/azimuth/elevation). Use isotropic prior (e.g. `lidar_sigma_meas: 1e-3` for VLP-16) unless a calibration source is added later.

### IMU and odom covariances

- **From 10_14_acl_jackal-005:** IMU: orientation_covariance = -1 (invalid); angular_velocity_covariance and linear_acceleration_covariance = 0.01 (diagonal). Odom: pose_covariance diag [0.001, 0.001, 1e6, 1e6, 1e6, 0.03]; twist_covariance diag [0.001, 0.001, 0.001, 1e6, 1e6, 0.03]. So odom already has meaningful covariances (backend uses them). IMU message covariances are 0.01; optional to add `use_imu_message_covariance: true` path to derive Sigma_g/Sigma_a from these.

### Timing (per-point timestamps, IMU–LiDAR offset)

- **PointCloud2:** VLP-16 layout may include `t` or `time`; see [POINTCLOUD2_LAYOUTS.md](POINTCLOUD2_LAYOUTS.md). Dataset docs do not specify IMU–LiDAR time offset; assume consistent header.stamp and optional per-point t when present.
- **Deskew:** Validate `t_scan` and scan bounds (scan_start_time, scan_end_time) on a Kimera bag and document here. Spinning LiDAR at ~10 Hz implies scan window ~0.1 s.

---

## Extrinsics (loading from file)

- **Current Kimera config:** Extrinsics in `gc_unified.yaml` (single source of truth). T_base_lidar from **T_baselink_lidar** (full 6D); T_base_imu from **T_cameralink_gyro** (translation) + Rx(-90°) rotation to map D435i optical frame (gravity +Y) to base frame (gravity -Z). See [rosbags/Kimera_Data/calibration/README.md](../rosbags/Kimera_Data/calibration/README.md). Convention: calibration uses T_a_b = "frame b into frame a" → same as GC T_base_sensor.
- **Config:** `extrinsics_source: inline | file`. When `file`: `T_base_lidar_file`, `T_base_imu_file` (paths to YAML; fail-fast if missing). Backend loads at startup. Run `python tools/check_extrinsics.py <config_path>` to print loaded T_base_lidar, T_base_imu and frame names.
- **GC format:** T_base_sensor: `[x, y, z, rx, ry, rz]` (translation m, rotvec rad). If dataset uses quat or T_sensor_base, add a conversion script to produce GC-format YAML.

## Config options (dataset / noise)

- **lidar_sigma_meas:** Scalar (m²) isotropic LiDAR measurement prior. Kimera VLP-16: `1e-3`.
- **extrinsics_source:** `inline` (use T_base_lidar, T_base_imu from config) | `file` (load from T_base_lidar_file, T_base_imu_file).
- **use_imu_message_covariance:** (Optional) When `true`, derive Sigma_g / Sigma_a from first N IMU messages (units and fallback documented in pipeline). When `false`, use configured/datasheet priors only. No heuristic auto-switch.

## Camera (when enable_camera:=true)

Kimera bag topics and frame_ids (from dataset README / sample):

- **RGB:** `/acl_jackal/forward/color/image_raw/compressed` (sensor_msgs/CompressedImage); frame from camera_info.
- **Depth:** `/acl_jackal/forward/depth/image_rect_raw` (sensor_msgs/Image, raw); frame from camera_info.
- **Frame:** Forward camera optical frame (e.g. `acl_jackal2/forward_color_optical_frame`, `acl_jackal2/forward_depth_optical_frame`).

GC launch defaults for Kimera: `camera_rgb_compressed_topic:=/acl_jackal/forward/color/image_raw/compressed`, `camera_depth_raw_topic:=/acl_jackal/forward/depth/image_rect_raw`. See `gc_rosbag.launch.py` camera args.
