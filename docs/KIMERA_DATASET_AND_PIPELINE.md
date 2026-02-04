# Kimera Dataset and Pipeline (GC v2)

Single source of truth for the Kimera dataset: hardware, bag, topics, frames, calibration, PointCloud2 layout, inspection, message schemas, and config.

**Reference:** [FRAME_AND_QUATERNION_CONVENTIONS.md](FRAME_AND_QUATERNION_CONVENTIONS.md) for GC-wide conventions (quat, SE(3), gravity).

Use `tools/inspect_rosbag_deep.py` to regenerate counts/frames when bags change.

---

## 1. Overview and canonical bag

The project uses a single bag for all testing (referenced in `tools/run_and_evaluate_gc.sh`):

- **Bag:** `rosbags/Kimera_Data/ros2/10_14_acl_jackal-005`
- **Ground truth:** `rosbags/Kimera_Data/ground_truth/1014/acl_jackal_gt.tum`
- **Config:** `fl_ws/src/fl_slam_poc/config/gc_kimera.yaml`

All configs, docs, and diagnostic examples refer to this bag. We do not run or support other bags; the eval script is the only test path.

**Conventions:** Bag truth = values from rosbag (`*.db3`). Pipeline truth = values from code (topic names, types, field usage). Frame IDs: `header.frame_id` = data frame; `child_frame_id` (Odometry) = body frame. No-TF mode: extrinsics from config (`T_base_lidar`, `T_base_imu`). Covariance: 6×6 = se(3) tangent `[δx, δy, δz, δωx, δωy, δωz]`; transported via adjoint.

---

## 2. Hardware

All sensors used on the Kimera acl_jackal rig:

| Sensor | Model | Topic (bag) | Frame |
|--------|--------|-------------|--------|
| **LiDAR** | Velodyne VLP-16 (Puck) | `/acl_jackal/lidar_points` | `acl_jackal2/velodyne_link` |
| **IMU** | Intel RealSense D435i (forward) | `/acl_jackal/forward/imu` | `acl_jackal2/forward_imu_optical_frame` |
| **Odometry** | Jackal wheel odometry | `/acl_jackal/jackal_velocity_controller/odom` | `acl_jackal2/odom` → `acl_jackal2/base` |
| **Camera** | Intel RealSense D435i RGB-D (forward) | color/depth (see §4) | `acl_jackal2/forward_*_optical_frame` |

Details below.

### 2.1 LiDAR: Velodyne VLP-16 (Puck)

Compact 16-channel mechanical spinning LiDAR; time-of-flight; 5–20 Hz; data via Ethernet (UDP). Used on Kimera acl_jackal; topic `/acl_jackal/lidar_points`, frame `acl_jackal2/velodyne_link`.

| Property | Value |
|----------|--------|
| **Channels** | 16 lasers |
| **Range** | Up to 100 m |
| **Range Accuracy** | ±3 cm (typical) |
| **Field of View** | Horizontal: 360°; Vertical: 30° (±15°) |
| **Angular Resolution** | Horizontal: 0.1°–0.4° (e.g. ~0.2° at 10 Hz); Vertical: ~2° (non-uniform) |
| **Rotation Rate** | 5–20 Hz (e.g. 10 Hz) |
| **Point Rate** | ~300k pts/s (single); ~600k (dual return) |
| **Laser** | 905 nm, Class 1 |
| **Data Output** | UDP; distance (2 mm res), intensity (0–255), azimuth, timestamps |

**Vertical beam angles (VLP-16):** 16 lasers, interleaved ±15° FOV; lower beams (even IDs) bottom half, upper (odd IDs) top half.

| Laser ID | Vertical Angle (°) | Vertical Offset Correction (mm) |
|----------|--------------------|----------------------------------|
| 0 | -15 | 11.2 |
| 1 | 1 | -0.7 |
| 2 | -13 | 9.7 |
| 3 | 3 | -2.2 |
| 4 | -11 | 8.1 |
| 5 | 5 | -3.7 |
| 6 | -9 | 6.6 |
| 7 | 7 | -5.1 |
| 8 | -7 | 5.1 |
| 9 | 9 | -6.6 |
| 10 | -5 | 3.7 |
| 11 | 11 | -8.1 |
| 12 | -3 | 2.2 |
| 13 | 13 | -9.7 |
| 14 | -1 | 0.7 |
| 15 | 15 | -11.2 |

Point coordinates: X = distance × cos(ω) × sin(α), Y = distance × cos(ω) × cos(α), Z = distance × sin(ω); α = azimuth, ω = vertical angle.

**SLAM notes:** Sparse vertical resolution; preserve **ring** (laser ID) in pipeline. Motion distortion over ~0.1 s scan; deskew with IMU or constant twist. GC config: `lidar_sigma_meas: 1e-3` (m² isotropic); range ±3 cm → ~9e-4 m²; angular ~0.15° → order 1e-3 m²/axis.

### 2.2 IMU: Intel RealSense D435i (forward)

Integrated in forward camera unit. Topic `/acl_jackal/forward/imu`, frame `acl_jackal2/forward_imu_optical_frame`. Stationary accelerometer reads -Y (gravity in +Y in optical frame). Gyro: rad/s; accel: scaled to m/s² (`imu_accel_scale`). T_base_imu maps IMU -Y to base +Z (Rx(-90°) = rotvec `[-1.57, 0, 0]` for D435i optical → base).

### 2.3 Odometry: Wheel odometry (Jackal)

2D planar base. Topic `/acl_jackal/jackal_velocity_controller/odom`. Pose = T_{odom<-base}; twist in child frame (v_x, ω_z planar). Frame: `acl_jackal2/odom` → `acl_jackal2/base`.

### 2.4 Camera: Intel RealSense D435i (forward) RGB-D

Optional. RGB: `/acl_jackal/forward/color/image_raw/compressed`; depth: `/acl_jackal/forward/depth/image_rect_raw`. Frames: `acl_jackal2/forward_color_optical_frame`, `acl_jackal2/forward_depth_optical_frame`. GC: `camera_rgbd_node` → `/gc/sensors/camera_rgbd` (RGBDImage).

---

## 3. Directory layout (Kimera_Data)

All under `rosbags/Kimera_Data/`:

| Path | Purpose |
|------|--------|
| **calibration/README.md** | Conventions: T_a_b = frame b → frame a; T_BS = sensor w.r.t. body. |
| **calibration/robots/<robot>/extrinsics.yaml** | Per-robot 4×4: T_baselink_lidar, T_cameralink_gyro, etc. |
| **calibration/extrinsics_manifest.yaml** | Index of robots and transform names. |
| **dataset_ready_manifest.yaml** | ros2_bag, ground_truth_tum, extrinsics. |
| **PREP_README.md** | GT format (TUM: timestamp x y z qx qy qz qw); sequences 1014, 1207, 1208. |
| **ground_truth/<seq>/<robot>_gt.tum** | Ground truth trajectory. |

For **acl_jackal** (bag `10_14_acl_jackal-005`): extrinsics at `calibration/robots/acl_jackal/extrinsics.yaml`; GT at `ground_truth/1014/acl_jackal_gt.tum`.

---

## 4. Topics and pipeline

### 4.1 Bag topics (Kimera acl_jackal)

| Topic | Message type | Used | How |
|-------|--------------|------|-----|
| `/acl_jackal/jackal_velocity_controller/odom` | `nav_msgs/msg/Odometry` | **Yes** | Backend subscribes (odom_topic); fused in pipeline. Frame: acl_jackal2/odom → acl_jackal2/base. |
| `/acl_jackal/lidar_points` | `sensor_msgs/msg/PointCloud2` | **Yes** | pointcloud_passthrough → `/gc/sensors/lidar_points` → backend. Frame: acl_jackal2/velodyne_link. |
| `/acl_jackal/forward/imu` | `sensor_msgs/msg/Imu` | **Yes** | Backend subscribes (imu_topic); preintegration + evidence. |
| `/acl_jackal/forward/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Optional | camera_rgbd_node; visual features when enabled. |
| `/acl_jackal/forward/depth/image_rect_raw` | `sensor_msgs/msg/Image` | Optional | camera_rgbd_node. |

**Summary:** TF may be absent; extrinsics from config. Odom frames: header.frame_id = acl_jackal2/odom, child_frame_id = acl_jackal2/base. LiDAR frame: lidar_points.header.frame_id = acl_jackal2/velodyne_link. IMU frame: forward/imu.header.frame_id = acl_jackal2/forward_imu_optical_frame.

### 4.2 Pipeline topic graph (MVP)

- **LiDAR:** Bag `/acl_jackal/lidar_points` → pointcloud_passthrough (gc_sensor_hub) → `/gc/sensors/lidar_points` → backend. Backend parses with `pointcloud_layout: vlp16`.
- **Odom:** Bag odom topic (launch `odom_topic`) → backend.
- **IMU:** Bag IMU topic (launch `imu_topic`) → backend.
- **Camera:** camera_rgbd_node → `/gc/sensors/camera_rgbd`; visual_feature_node → `/gc/sensors/visual_features`; backend subscribes when enabled.
- **Backend outputs:** `/gc/state`, `/gc/trajectory`, `/gc/map`, status, TF (odom_frame → base_link).

### 4.3 LiDAR path (Kimera)

- **Hub:** pointcloud_passthrough subscribes to `/acl_jackal/lidar_points`, republishes to `/gc/sensors/lidar_points`.
- **Backend:** Subscribes to `/gc/sensors/lidar_points`; parses with `parse_pointcloud2_vlp16` when `pointcloud_layout: vlp16`. Fields used: x, y, z, ring; optional t/time for per-point timestamps; optional intensity (not yet consumed).

---

## 5. Frame names and axis conventions

| GC parameter / role | Kimera value | Topic (reference) |
|---------------------|--------------|-------------------|
| **odom_frame** (world/parent) | `acl_jackal2/odom` | odom header.frame_id |
| **base_frame** (body/child) | `acl_jackal2/base` | odom child_frame_id |
| **LiDAR frame** | `acl_jackal2/velodyne_link` | lidar_points header.frame_id |
| **IMU frame** | `acl_jackal2/forward_imu_optical_frame` | forward/imu header.frame_id |

- **Odometry:** Pose = T_{odom_frame <- base_frame}. Do not invert at ingest.
- **LiDAR:** Points in velodyne_link; T_base_lidar = T_{base <- velodyne_link}.
- **IMU:** Measurements in forward_imu_optical_frame; T_base_imu = T_{base <- imu_frame}.

**Axis conventions:** World/odom: Z-up, planar motion (odom z=0, cov z/roll/pitch large). Base: planar base. LiDAR: VLP-16; verify Z-up vs Z-down with `tools/diagnose_coordinate_frames.py` (use `--lidar-topic /acl_jackal/lidar_points`). IMU: D435i optical; stationary accel -Y; T_base_imu maps to base +Z.

**"All motion in Z" symptom:** If trajectory moves mainly along Z while robot moves in XY, check (1) IMU extrinsic T_base_imu, (2) odom axis convention, (3) state/export ordering. Run `tools/diagnose_trajectory_axes.py`.

**Diagnostic results (10_14_acl_jackal-005):** LiDAR Z-UP (ground normal · Z ≈ 0.996); T_base_lidar small roll/pitch ~1.3° in gc_kimera.yaml. Odom twist: planar v_x, ω_z; covariance order ROS [x,y,z,roll,pitch,yaw].

---

## 6. PointCloud2 layout (VLP-16)

Backend selects parser by config: `pointcloud_layout: vlp16`. Fail-fast if message fields do not match.

**Required fields:** x, y, z (float); ring (uint8, 0–15). **Optional:** intensity (float/uint8); t or time (per-point timestamp, s or ns). Per-point timestamp: use t/time if present (convert to s), else header.stamp for whole scan.

**Pipeline contract:** Parser outputs `(points, timestamps, weights, ring, tag)`. VLP-16: tag=0; timebase = header.stamp; time_offset from per-point t/time if present, else 0. Parser: `parse_pointcloud2_vlp16`; fail-fast if x, y, z, or ring missing.

**Config:** `pointcloud_layout` (string): `vlp16`. YAML: `pointcloud_layout: vlp16` for Kimera.

---

## 7. Calibration and extrinsics

**How GC gets extrinsics:** `tools/kimera_calibration_to_gc.py` reads `robots/acl_jackal/extrinsics.yaml`, outputs GC format `[x, y, z, rx, ry, rz]` (m, rotvec rad). T_baselink_lidar → T_base_lidar; T_cameralink_gyro → T_base_imu (rotation overridable with `--imu-rotation`). Config: gc_kimera.yaml; backend loads at startup. Run `python tools/check_extrinsics.py <config_path>` to print loaded T_base_lidar, T_base_imu.

**Current Kimera:** Extrinsics in gc_unified.yaml. T_base_lidar from T_baselink_lidar (6D); T_base_imu from T_cameralink_gyro (translation) + Rx(-90°) for D435i optical → base. GC format: T_base_sensor `[x, y, z, rx, ry, rz]` (m, rad). Convention: T_a_b = frame b into frame a = GC T_base_sensor.

**Config options:** `extrinsics_source`: inline | file. `lidar_sigma_meas`: 1e-3 (VLP-16). `use_imu_message_covariance`: optional.

---

## 8. Ground truth and evaluation

**Kimera GT:** `ground_truth/<seq>/<robot>_gt.tum`; timestamps in seconds. Dataset says "Align frames to GC v2 conventions before eval."

**GC anchor:** Trajectory exported in anchor frame (first odom as origin), Z-up planar, TUM format.

**Evaluation:** `align_ground_truth.py` then initial-pose alignment in `evaluate_slam.py` (GT → estimate at first pose). If GT world frame differs (up/forward axis), ATE/RPE can be huge. **What to do:** (1) Confirm GT frame in dataset docs. (2) Add GT→estimate transform in align/eval if needed. (3) Treat large ATE/RPE as frame mismatch; see note from `run_and_evaluate_gc.sh` when ATE > 10 m or RPE @ 1 m > 1.

---

## 9. Config (gc_kimera.yaml)

- **odom_frame:** acl_jackal2/odom. **base_frame:** acl_jackal2/base.
- **Hub input topics:** Odom `/acl_jackal/jackal_velocity_controller/odom`, LiDAR `/acl_jackal/lidar_points`, IMU `/acl_jackal/forward/imu`.
- **T_base_lidar, T_base_imu:** From Kimera calibration (kimera_calibration_to_gc.py).
- **lidar_sigma_meas:** 1e-3 (VLP-16).
- **pointcloud_layout:** vlp16.

**Canonical bag:** 10_14_acl_jackal-005. Topics under `/acl_jackal/`; frame IDs `acl_jackal2/*`.

---

## 10. Discovery (what the dataset provides)

**Extrinsics:** MIT-SPARK/Kimera-Multi-Data README; LiDAR/IMU–base from plusk01/Kimera-Multi-Data branch parker/kmd_tools (community). GC expects T_base_sensor (rotvec rad, trans m).

**LiDAR variance:** No per-axis variance published; use isotropic prior lidar_sigma_meas: 1e-3.

**IMU/odom covariances (10_14_acl_jackal-005):** IMU: orientation_cov = -1; angular_velocity_cov, linear_acceleration_cov = 0.01. Odom: pose_cov diag [0.001, 0.001, 1e6, 1e6, 1e6, 0.03]; twist_cov diag [0.001, 0.001, 0.001, 1e6, 1e6, 0.03]. Backend uses them.

**Timing:** PointCloud2 may have t/time; deskew: t_scan = header.stamp; ~0.1 s scan at 10 Hz.

---

## 11. Bag inspection checklist

When you have a Kimera ROS 2 bag and need to fill in frame conventions, extrinsics, covariances, timing. Bag path = directory containing `*.db3`.

**All-in-one:** With ROS 2 sourced:
```bash
python tools/inspect_kimera_bag.py /path/to/bag_dir          # report only
python tools/inspect_kimera_bag.py /path/to/bag_dir --apply  # write T_base_lidar/T_base_imu to gc_kimera.yaml
```
Runs: first-N summary, diagnose_coordinate_frames, estimate_lidar_base_extrinsic_rotation_from_ground, estimate_imu_base_extrinsic_rotation. Extrinsics from script are rotation-only; use dataset calib for full 6D.

**1. Frame names and topic layout:** `tools/inspect_rosbag_deep.py` or `tools/validate_frame_conventions.py`. Confirm odom frame_id/child_frame_id, LiDAR topic and frame_id, IMU topic and frame_id. Record in this doc (§5) and gc_kimera.yaml.

**2. First-N messages:** `tools/first_n_messages_summary.py <bag_dir> --n 25 [--md out.md]`. Check PointCloud2 fields (x,y,z,ring; t/time); Imu covariances; Odometry pose/twist cov. Record in this doc (§10).

**3. LiDAR Z-up/Z-down, odom ordering:** `tools/diagnose_coordinate_frames.py <bag_dir> --lidar-topic /acl_jackal/lidar_points --imu-topic /acl_jackal/forward/imu --odom-topic /acl_jackal/jackal_velocity_controller/odom --n-scans 20`. Record result in this doc (§5 Diagnostic results).

**4. Extrinsics:** Use dataset/community calib or estimation tools; put values in gc_kimera.yaml or set extrinsics_source: file. Then `python tools/check_extrinsics.py fl_ws/src/fl_slam_poc/config/gc_kimera.yaml`.

**5. Timing:** Check PointCloud2 t/time field and units; confirm t_scan and scan bounds.

**Quick commands:** Frame/topics: `inspect_rosbag_deep.py <bag_dir>`. First N: `first_n_messages_summary.py <bag_dir> --n 25`. Z-up/odom: `diagnose_coordinate_frames.py <bag_dir> --lidar-topic ... --imu-topic ... --odom-topic ...`. Extrinsics: `check_extrinsics.py <config>`.

---

## 12. Standard message usage (summary)

**nav_msgs/Odometry:** Consumed: header.stamp, frame_id, child_frame_id, pose.pose, pose.covariance (6×6). Backend uses pose and covariance. Twist consumed when available (velocity/yaw-rate factors). Produced (/gc/state): pose.pose, pose.covariance (6×6), child_frame_id base_link.

**sensor_msgs/Imu:** Consumed: header.stamp, frame_id, angular_velocity.{x,y,z}, linear_acceleration.{x,y,z}. Ignored: orientation, covariance arrays. Units: gyro rad/s; accel × imu_accel_scale → m/s².

**sensor_msgs/PointCloud2:** VLP-16 layout; backend parse_pointcloud2_vlp16. Fields: x, y, z, ring; optional t/time, intensity.

**Camera:** RGBDImage from camera_rgbd_node (rgb8 + 32FC1 depth). VisualFeatureBatch from visual_feature_node when enabled.

---

## 13. Custom message schemas (summary)

Defined in `fl_ws/src/fl_slam_poc/msg/`. **AnchorCreate:** header, anchor_id, points (geometry_msgs/Point[]; in base frame at anchor time). **LoopFactor:** anchor_id, weight, rel_pose (T_anchor^{-1} ∘ T_current), covariance[36] (6×6 se(3)), approximation_triggers. **IMUSegment:** keyframe ids, t_i/t_j, bias_ref, gravity_world, stamp[], accel[], gyro[]. **RGBDImage:** header, rgb (Image), depth (Image). **VisualFeatureBatch:** (see msg definition). **RGB-D evidence:** JSON string with position_L/h, color_L/h, normal_theta, alpha_mean/var.

---

## 14. QoS and critical assumptions

**QoS (typical):** Backend subscriptions RELIABLE, depth 100 (or from config). IMU often best_effort. Camera/LiDAR from launch.

**Assumptions:** No TF in bag → extrinsics from config (T_base_lidar, T_base_imu). No CameraInfo → intrinsics via params. IMU extrinsic maps to base frame; frame IDs logged for audit. Bias evolution: adaptive Wishart, not user imu_*_random_walk.

---

## 15. Quick reference: files we read

| What | File(s) |
|------|--------|
| LiDAR/IMU extrinsics (acl_jackal) | `rosbags/Kimera_Data/calibration/robots/acl_jackal/extrinsics.yaml` |
| GC config (Kimera) | `fl_ws/src/fl_slam_poc/config/gc_kimera.yaml` |
| Convert calib → GC | `python tools/kimera_calibration_to_gc.py rosbags/Kimera_Data/calibration/robots/acl_jackal/extrinsics.yaml` |
| Bag ↔ GT ↔ extrinsics | `rosbags/Kimera_Data/dataset_ready_manifest.yaml` |
| GT trajectory (10_14 acl_jackal) | `rosbags/Kimera_Data/ground_truth/1014/acl_jackal_gt.tum` |
