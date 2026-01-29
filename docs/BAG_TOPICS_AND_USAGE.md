# Bag Topics and Usage Map (Canonical)

This document is the single source of truth for:

- **What topics/messages exist** in supported datasets/bags
- **Which ones FL-SLAM consumes today** (MVP)
- **Which ones are present but not yet used** (roadmap)
- **Non-negotiable assumptions** (frames, TF presence, intrinsics, extrinsics)
- **Exact message schemas** for FL-SLAM custom messages and how standard ROS messages are interpreted (including covariance handling)

Use `tools/inspect_rosbag_deep.py` to regenerate counts/frames when bags change.

## Conventions

- **Bag truth**: values observed directly from the rosbag database (`*.db3`). No guessing.
- **Pipeline truth**: values observed directly from the current code (topic names, message types, and field usage).
- **Frame IDs**:
  - `header.frame_id` is the *data frame* for the message.
  - `child_frame_id` (Odometry) is the *body frame* being tracked.
- **No-TF mode**:
  - If `/tf` and `/tf_static` are absent, extrinsics must be provided as parameters.
  - For LiDAR pointcloud processing, use `lidar_base_extrinsic = T_base_lidar` (6DOF) to map points into the base frame.
- **Covariance conventions** (project-wide):
  - When we say “6×6 covariance”, we mean a covariance on an se(3) tangent vector ordered as `[δx, δy, δz, δωx, δωy, δωz]` (translation then rotation).
  - Covariances are treated as living in the tangent space at identity and transported via adjoint when composition/inversion is applied.

## M3DGR Dynamic01 (ROS 2 bag)

### Hardware Platform

The M3DGR dataset uses a ground robot platform with the following sensor configuration:

**Reference Diagram:** See `docs/car.jpg` for a detailed schematic diagram showing the robot platform, sensor mounting positions, coordinate frames, and physical dimensions.

**Platform Overview:**
- Lower main platform: 35 cm × 84 cm, 26 cm from ground
- Upper platform: 35 cm × 35 cm, 45 cm above lower platform
- Top platform (antennas): 15 cm above upper platform
- Overall height: ~133 cm to top of omnidirectional camera mast

**Sensor Mounting:**
- Front-view LiDAR (Livox Avia) and VI-sensor (RealSense D435i) mounted ~9 cm forward from front edge of lower platform
- 360° LiDAR (Livox MID-360) mounted on upper platform
- Omnidirectional camera (Insta360 X4) on tall mast extending upward
- GPS/RTK receivers (CUAV C-RTK9Ps, C-RTK2HP) on lower platform front-center
- Pixhawk flight controller on lower platform
- On-board computer on lower platform rear-center
- Wheel odometer (WHEELTEC) on rear wheel

*Note: See M3DGR dataset documentation for detailed platform schematic diagrams and photographs showing sensor placement, dimensions, coordinate frames, and physical mounting angles. The platform schematic shows the 360° LiDAR (MID-360) mounted on the upper platform with coordinate systems indicated for each sensor.*

### Sensor Specifications

All sensors and tracking devices with their most important parameters:

#### LiDAR Sensors

1. **LiDAR1 - Livox Avia**
   - Type: Non-repetitive scanning
   - Horizontal FOV: 70.4°
   - Vertical FOV: 77.2°
   - Frequency: 10 Hz
   - Max Range: 450 m
   - Range Precision: 2 cm
   - Angular Precision: 0.05°
   - IMU: 6-axis, 200 Hz
   - ROS Topic: `/livox/avia/lidar`

2. **LiDAR2 - Livox MID-360** (Primary sensor used by GC v2)
   - Type: Non-repetitive scanning
   - Horizontal FOV: 360°
   - Vertical FOV: -7° to +52° (scans from 7° below to 52° above horizontal)
   - Frequency: 10 Hz
   - Max Range: 40 m
   - Range Resolution: 3 cm
   - Angular Resolution: 0.15°
   - IMU: 6-axis, 200 Hz
   - ROS Topic: `/livox/mid360/lidar`
   - **Frame Convention**: For **M3DGR Dynamic01**, see `docs/FRAME_AND_QUATERNION_CONVENTIONS.md` (canonical). That doc states `livox_frame` is **Z-up** for this dataset (ground plane analysis). Generic Livox MID-360 datasheet may use Z-down; if so, conversion to Z-up `base_footprint` would require 180° rotation about X in `T_base_lidar`. Current code uses identity rotation; if evaluation shows ~180° roll offset vs ground truth, run `tools/diagnose_coordinate_frames.py` and set `T_base_lidar` rotation to `[π, 0, 0]` if the diagnostic reports Z-down.

#### Visual-Inertial Sensor

3. **V-I Sensor - RealSense D435i**
   - RGB/Depth Resolution: 640×480
   - Horizontal FOV: 69°
   - Vertical FOV: 42.5°
   - Frequency: 15 Hz
   - IMU: 6-axis, 200 Hz
   - ROS Topics:
     - `/camera/color/image_raw/compressed` (RGB)
     - `/camera/aligned_depth_to_color/image_raw/compressedDepth` (Depth)
     - `/camera/imu` (IMU)
   - *Note: Insta360 X4 driver ROS package available for real-time streaming (modify device ID in launch file)*

#### Omnidirectional Camera

4. **Insta360 X4**
   - RGB Resolution: 2880×1440
   - Horizontal FOV: 360°
   - Vertical FOV: 360°
   - Frequency: 15 Hz
   - ROS Topic: `/cv_camera/image_raw/compressed`

#### Odometry

5. **Wheel Odometer - WHEELTEC**
   - Type: 2D odometry
   - Frequency: 20 Hz
   - ROS Topic: `/odom`
   - Frame: `odom_combined` → `base_footprint`

#### GNSS/RTK

6. **GNSS Receiver - CUAV C-RTK9Ps**
   - Systems: BDS/GPS/GLONASS/Galileo
   - Frequency: 10 Hz
   - ROS Topics:
     - `/ublox_driver/ephem`
     - `/ublox_driver/glo_ephem`
     - `/ublox_driver/iono_params`
     - `/ublox_driver/range_meas`
     - `/ublox_driver/receiver_lla`
     - `/ublox_driver/receiver_pvt`
     - `/ublox_driver/time_pulse_info`

7. **RTK Receiver - CUAV C-RTK2HP**
   - Localization Accuracy: 0.8 cm (H) / 1.5 cm (V)
   - Frequency: 15 Hz

#### Motion Capture

8. **Motion-capture System - OptiTrack**
   - Localization Accuracy: 1 mm
   - Frequency: 360 Hz
   - ROS Topic: `/vrpn_client_node/UGV/pose` (ground truth for evaluation only)

### Summary (bag truth)

- **TF**: `/tf` and `/tf_static` are **absent**
- **CameraInfo**: `sensor_msgs/msg/CameraInfo` topics are **absent**
- **Odom frames**:
  - `/odom.header.frame_id = odom_combined`
  - `/odom.child_frame_id = base_footprint`
- **Camera frames**:
  - `/camera/color/image_raw/compressed.header.frame_id = camera_color_optical_frame`
  - `/camera/aligned_depth_to_color/image_raw/compressedDepth.header.frame_id = camera_color_optical_frame`
  - `/camera/imu.header.frame_id = camera_imu_optical_frame`
- **Livox frames**:
  - `/livox/mid360/lidar.header.frame_id = livox_frame`
  - `/livox/mid360/imu.header.frame_id = livox_frame`
  - `/livox/avia/lidar.header.frame_id = livox_frame` (if present)
  - `/livox/avia/imu.header.frame_id = livox_frame` (if present)
  - `/livox/avia/*` topics may be present, but require `livox_ros_driver` message package to decode.

### Adopted calibration + launch defaults (MVP)

These are the **explicit parameters we run with** for M3DGR (to compensate for missing `/tf(_static)` and missing `CameraInfo`):

- **Base/body frame policy**: `base_frame` is treated as the dataset **body/IMU frame \(b\)** (we keep the name `base_footprint` for M3DGR because it matches bag truth).
- **LiDAR mounting** (MID-360, no-TF): `T_base_lidar = [-0.011, 0.0, 0.778, 0.0, 0.0, 0.0]` interpreted as \(T_{b\leftarrow \text{mid360}}\) with format `[x,y,z,rx,ry,rz]` (rotvec in radians). **Canonical:** `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`.
  - Translation: `[-0.011, 0.0, 0.778]` meters (LiDAR offset from base)
  - Rotation: `[0.0, 0.0, 0.0]` (identity). For M3DGR Dynamic01, `tools/diagnose_coordinate_frames.py` reports LiDAR Z-up (ground normal · Z > 0.7), so no 180° X rotation. If a different bag reports Z-down, use rotation `[3.141593, 0.0, 0.0]`.
- **IMU mounting** (MID-360, no-TF): `T_base_imu = [0.0, 0.0, 0.0, -0.015586, 0.489293, 0.0]` (rotvec in radians).
  - Translation: `[0.0, 0.0, 0.0]` (IMU co-located with LiDAR in MID-360 unit)
  - Rotation: `[-0.015586, 0.489293, 0.0]` radians (~28° misalignment, estimated via gravity alignment)
- **IMU source**: `/livox/mid360/imu`
- **RealSense intrinsics** (640×480, bag has no `CameraInfo`):
  - `camera_fx = 610.16`
  - `camera_fy = 610.45`
  - `camera_cx = 326.35`
  - `camera_cy = 244.68`
- **IMU noise densities** (used by preintegration; random-walk terms are intentionally not used):
  - `imu_gyro_noise_density = 1.7e-4` (rad/s/√Hz)
  - `imu_accel_noise_density = 1.9e-4` (m/s²/√Hz)

### ROS Topics Reference

Complete list of ROS topics available in M3DGR rosbag sequences:

#### LiDAR Topics
- **LiDAR1**: `/livox/avia/lidar` (Livox Avia, `livox_ros_driver/msg/CustomMsg`)
- **LiDAR2**: `/livox/mid360/lidar` (Livox MID-360, `livox_ros_driver2/msg/CustomMsg`)

#### Odometry
- **Wheel Odometer**: `/odom` (`nav_msgs/msg/Odometry`, 20 Hz)

#### Camera Topics
- **RGB Camera**: `/camera/color/image_raw/compressed` (`sensor_msgs/msg/CompressedImage`)
- **Omnidirectional Camera**: `/cv_camera/image_raw/compressed` (`sensor_msgs/msg/CompressedImage`)
- **Depth Camera**: `/camera/aligned_depth_to_color/image_raw/compressedDepth` (`sensor_msgs/msg/CompressedImage`)

#### GNSS Topics
- `/ublox_driver/ephem` (ephemeris data)
- `/ublox_driver/glo_ephem` (GLONASS ephemeris)
- `/ublox_driver/iono_params` (ionospheric parameters)
- `/ublox_driver/range_meas` (range measurements)
- `/ublox_driver/receiver_lla` (receiver latitude/longitude/altitude)
- `/ublox_driver/receiver_pvt` (receiver position/velocity/time)
- `/ublox_driver/time_pulse_info` (time pulse information)

#### IMU Topics
- `/camera/imu` (`sensor_msgs/msg/Imu`, RealSense D435i IMU, 200 Hz)
- `/livox/avia/imu` (`sensor_msgs/msg/Imu`, Livox Avia IMU, 200 Hz)
- `/livox/mid360/imu` (`sensor_msgs/msg/Imu`, Livox MID-360 IMU, 200 Hz) - **Primary IMU used by GC v2**

#### Ground Truth
- `/vrpn_client_node/UGV/pose` (`geometry_msgs/msg/PoseStamped`, OptiTrack motion capture, 360 Hz) - **Evaluation only, must never be fused**

### Topic inventory (observed)

Run:

```bash
source /opt/ros/jazzy/setup.bash
source fl_ws/install/setup.bash
.venv/bin/python tools/inspect_rosbag_deep.py rosbags/m3dgr/Dynamic01_ros2 --json /tmp/m3dgr_summary.json
```

#### Key bag topics

These are the *bag inputs* used by the MVP pipeline (or present but explicitly not used yet):

| Topic | Message type | Used now? | How it is used |
| --- | --- | --- | --- |
| `/odom` | `nav_msgs/msg/Odometry` | **Yes** | Converted to delta odom by `odom_bridge` and fused in backend (`/sim/odom`). Frame truth: `odom_combined -> base_footprint`. |
| `/livox/mid360/lidar` | `livox_ros_driver2/msg/CustomMsg` | **Yes** | Converted to `PointCloud2` by `livox_converter` and consumed by frontend pointcloud path. Preserves `frame_id=livox_frame`. |
| `/livox/mid360/imu` | `sensor_msgs/msg/Imu` | **Yes** | Frontend buffers IMU and publishes `/sim/imu_segment` (Contract B); backend re-integrates. Frame: `livox_frame`. |
| `/camera/imu` | `sensor_msgs/msg/Imu` | Not used (MVP) | Present in bag but not used by default in the M3DGR pipeline. |
| `/camera/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Optional | Decompressed by `image_decompress_cpp` to `/camera/image_raw`. Downstream usage depends on `enable_image/publish_rgbd_evidence`. |
| `/camera/aligned_depth_to_color/image_raw/compressedDepth` | `sensor_msgs/msg/CompressedImage` | Optional | Decompressed by `image_decompress_cpp` to `/camera/depth/image_raw`. Depth is aligned to color (same frame). |
| `/vrpn_client_node/UGV/pose` | `geometry_msgs/msg/PoseStamped` | **Evaluation only** | Ground-truth source for offline evaluation. Must never be fused into inference. |
| `/livox/avia/lidar` | `livox_ros_driver/msg/CustomMsg` | Not yet | Present in bag; requires `livox_ros_driver` Python messages to decode and use. |
| `/livox/avia/imu` | `sensor_msgs/msg/Imu` | Not yet | Present for future work; currently unused. |
| `/cv_camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Not yet | Insta360 X4 omnidirectional camera (360° H/V FOV). |
| `/ublox_driver/*` | Various GNSS messages | Not yet | GNSS/RTK receiver topics (ephemeris, iono params, range measurements, receiver LLA/PVT, time pulse). |

#### Pipeline topic graph (MVP run)

This is the *runtime topic graph* the MVP launch creates (bag inputs → utility nodes → frontend → backend → outputs).

##### Utility / intermediate topics

| Topic | Message type | Producer | Consumer | Notes |
| --- | --- | --- | --- | --- |
| `/sim/odom` | `nav_msgs/msg/Odometry` | `odom_bridge` | `frontend_node` (SensorIO), `backend_node` | Delta-odom (absolute→delta). `pose.covariance` is propagated/approximated; `twist` is unused. |
| `/lidar/points` | `sensor_msgs/msg/PointCloud2` | `livox_converter` | `frontend_node` (SensorIO) | XYZ is consumed today; extra fields are preserved in the message but ignored by current frontend math. |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | `image_decompress_cpp` | `frontend_node` (SensorIO, optional) | Published as `rgb8`. |
| `/camera/depth/image_raw` | `sensor_msgs/msg/Image` | `image_decompress_cpp` | `frontend_node` (SensorIO, optional) | Published as `32FC1` depth in meters. |

##### Frontend → backend topics (estimation graph)

| Topic | Message type | Producer | Consumer | Covariance / weight semantics |
| --- | --- | --- | --- | --- |
| `/sim/anchor_create` | `fl_slam_poc/msg/AnchorCreate` | `frontend_node` | `backend_node` | No covariance. Carries an anchor id and an anchor-local point sample for mapping/visualization. |
| `/sim/loop_factor` | `fl_slam_poc/msg/LoopFactor` | `frontend_node` | `backend_node` | Contains `weight ∈ [0,1]` and `covariance[36]` (6×6, row-major, se(3) tangent). Backend treats `weight` as precision scaling (`Σ_eff = Σ / weight`). |
| `/sim/imu_segment` | `fl_slam_poc/msg/IMUSegment` | `frontend_node` | `backend_node` | Raw IMU samples (Contract B). No covariance in message; backend constructs a covariance from declared noise densities. |
| `/sim/rgbd_evidence` | `std_msgs/msg/String` (JSON) | `frontend_node` (optional) | `backend_node` | Dense evidence payload (position/color/normal). See “RGB-D evidence payload schema”. |

##### Backend outputs (observability + artifacts)

| Topic | Message type | Producer | Consumer | Notes |
| --- | --- | --- | --- | --- |
| `/cdwm/state` | `nav_msgs/msg/Odometry` | `backend_node` | Viz / evaluators | Posterior pose as `pose.pose` + `pose.covariance` (6×6). `child_frame_id` is currently `"base_link"`. |
| `/cdwm/trajectory` | `nav_msgs/msg/Path` | `backend_node` | Viz | Path built from `/cdwm/state` poses (no covariance). |
| `/cdwm/map` | `sensor_msgs/msg/PointCloud2` | `backend_node` | Viz | XYZRGB point cloud for Foxglove/RViz (see “/cdwm/map schema”). |
| `/cdwm/markers` | `visualization_msgs/msg/MarkerArray` | `backend_node` | Viz | Anchor markers (sparse layer). |
| `/cdwm/loop_markers` | `visualization_msgs/msg/MarkerArray` | `backend_node` | Viz | Loop closure visualization markers. |
| `/cdwm/backend_status` | `std_msgs/msg/String` (JSON) | `backend_node` | Viz | Backend health + counts + mode (DEAD_RECKONING vs SLAM). |
| `/cdwm/op_report` | `std_msgs/msg/String` (JSON) | Multiple nodes | Logging | OpReports (audit trail). Multiple publishers share this topic. |
| `/cdwm/frontend_status` | `std_msgs/msg/String` (JSON) | `frontend_node` | Viz | Frontend status snapshot (in addition to `/cdwm/status`). |
| `/cdwm/status` | `std_msgs/msg/String` (JSON) | `StatusMonitor` | Viz | Periodic sensor connectivity status (warnings are emitted here). |
| `/cdwm/debug` | `std_msgs/msg/String` | `backend_node` | Logging | Human-readable debug messages (not a stable schema). |
| `/tf` | TF frames | `backend_node` | ROS TF consumers | Backend publishes `odom_frame -> base_link` TF consistent with `/cdwm/state`. Bags may not include TF. |

### Current LiDAR conversion contract (MVP)

- **Converter**: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/livox_converter.py`
- **Output**: `sensor_msgs/msg/PointCloud2` on `/lidar/points`
- **Fields** (current, deterministic schema):
  - `x,y,z` (float32)
  - `intensity` (uint8, from reflectivity)
  - `ring` (uint8, from line)
  - `tag` (uint8)
  - `time_offset` (uint32, **only if present** in the Livox point type; treated as raw units)
  - `timebase_low`, `timebase_high` (uint32, message-level timebase preserved per point)

**Frontend consumption note:** current frontend 3D path reads only `x/y/z` from `PointCloud2` and ignores the other fields (they are preserved for audit/debug and future use).

### Standard message usage (exact field-level behavior)

This section answers: “Which fields are read/written, and are covariances propagated?”

#### `nav_msgs/msg/Odometry` (bag `/odom`, internal `/sim/odom`, output `/cdwm/state`)

- **Fields consumed (odom_bridge input `/odom`)**
  - `header.stamp` (duplicate detection, timing)
  - `header.frame_id` (validation/metadata)
  - `child_frame_id` (metadata)
  - `pose.pose` (position + orientation quaternion)
  - `pose.covariance` (6×6, row-major): **used**
  - `twist.*` + `twist.covariance`: **ignored**
- **Fields produced (odom_bridge output `/sim/odom`)**
  - `pose.pose`: **delta pose** between successive odom messages
  - `pose.covariance`: **delta covariance** computed by approximation (see below)
  - `twist`: not used by consumers (left default/unchanged)
- **Delta covariance (odom_bridge)**
  - Uses a small-delta approximation: `Σ_delta ≈ Σ_prev + Σ_curr` (logged via `OpReport`).
  - This is explicitly marked as an approximation in the bridge’s OpReport (audit visibility).
- **Fields consumed (backend input `/sim/odom`)**
  - `pose.pose` (delta pose)
  - `pose.covariance` (6×6): **used as measurement covariance in fusion**
- **Fields produced (backend output `/cdwm/state`)**
  - `pose.pose`: posterior pose estimate
  - `pose.covariance`: posterior covariance **(6×6)**
  - `child_frame_id`: currently `"base_link"` (hardcoded in backend publish)

#### `sensor_msgs/msg/Imu` (bag `/livox/mid360/imu`)

- **Fields consumed (frontend SensorIO)**
  - `header.stamp`, `header.frame_id` (buffering + audit logs)
  - `angular_velocity.{x,y,z}` and `linear_acceleration.{x,y,z}`
- **Fields ignored (current MVP)**
  - `orientation` and all IMU covariance arrays (`orientation_covariance`, `angular_velocity_covariance`, `linear_acceleration_covariance`)
- **Units**
  - Gyro is treated as **rad/s**.
  - Accel is scaled by `imu_accel_scale` (default 9.81) to convert **g → m/s²** for Livox MID-360 bags.

#### `sensor_msgs/msg/Image` (internal `/camera/image_raw`, `/camera/depth/image_raw`)

- Produced by `image_decompress_cpp`.
- **RGB output**: `encoding="rgb8"` (converted from decoded BGR).
- **Depth output**: `encoding="32FC1"` (meters). If input is 16-bit depth in mm and `depth_scale_mm_to_m=true`, it is converted to meters.

#### `sensor_msgs/msg/PointCloud2` (internal `/lidar/points`)

- Produced by `livox_converter`.
- Frontend currently parses only `x/y/z` and filters invalid points; other fields are ignored by current inference.

### FL-SLAM custom message schemas (authoritative)

These are defined in `fl_ws/src/fl_slam_poc/msg/` and are the binding schema for `/sim/*` topics.

#### `fl_slam_poc/msg/AnchorCreate` (`/sim/anchor_create`)

- `header`
  - `stamp`: set to the triggering sensor stamp (scan/pointcloud stamp)
  - `frame_id`: currently set to `odom_frame` by the frontend
- `anchor_id`: integer anchor identifier
- `points`: `geometry_msgs/Point[]`
  - **Semantics:** points are taken from the frontend’s geometry source after transformation into the configured `base_frame` (i.e., anchor-local points at the keyframe time).
  - **Important:** header frame is `odom_frame`, but the points are *not* expressed in odom coordinates; they are expressed in the anchor/body frame at anchor creation time.

#### `fl_slam_poc/msg/LoopFactor` (`/sim/loop_factor`)

- `anchor_id`: which anchor this factor closes to
- `weight`: continuous soft weight `∈ [0,1]` (backend treats it as precision scaling)
- `rel_pose`: `geometry_msgs/Pose`
  - **Convention:** `Z = T_anchor^{-1} ∘ T_current`
  - Backend reconstruction uses `T_current ≈ T_anchor ∘ Z`
- `covariance[36]`:
  - 6×6 row-major covariance in se(3) tangent `[δx, δy, δz, δωx, δωy, δωz]`
  - Backend reads this covariance and applies `Σ_eff = Σ / max(weight, ε)` before composition and fusion
- `approximation_triggers[]`: audit trail (e.g., `Linearization`, `BudgetTruncation`)
- Solver metadata fields: `solver_*`, `information_weight`
  - Note: frontend currently sets `information_weight = icp_result.final_objective` as a placeholder.

#### `fl_slam_poc/msg/IMUSegment` (`/sim/imu_segment`)

Contract B: raw IMU segment between successive keyframes.

- `keyframe_i`, `keyframe_j`: keyframe identifiers
- `t_i`, `t_j`: segment endpoints (float seconds)
- `bias_ref_bg[3]`, `bias_ref_ba[3]`: bias references at segment start (frontend currently publishes zeros)
- `gravity_world[3]`: gravity vector (frontend sets from its `gravity` parameter)
- `stamp[]`: timestamps in seconds (monotonic)
- `accel[]`: flattened `N*3` accelerometer samples (m/s²)
- `gyro[]`: flattened `N*3` gyroscope samples (rad/s)

### RGB-D evidence payload schema (`/sim/rgbd_evidence`)

This topic is a `std_msgs/msg/String` containing JSON:

```json
{
  "evidence": [
    {
      "position_L": [[...],[...],[...]],
      "position_h": [...],
      "color_L": [[...],[...],[...]],
      "color_h": [...],
      "normal_theta": [...],
      "alpha_mean": 0.0,
      "alpha_var": 0.0
    }
  ]
}
```

Field meaning (current implementation):

- `position_L`, `position_h`: information-form Gaussian for 3D position (`L` is 3×3, `h` is length-3)
- `color_L`, `color_h`: information-form Gaussian for RGB color (3×3, length-3)
- `normal_theta`: vMF directional parameter (length-3)
- `alpha_mean`, `alpha_var`: scalar appearance/occupancy parameters (used by dense module layer)

### QoS defaults (exact, as implemented)

QoS is a common cause of “it runs but nothing arrives”. These are the current defaults in code/launch:

- **Frontend SensorIO subscriptions**
  - General sensors (odom/pointcloud/scan/image/depth/camera_info): `depth=QOS_DEPTH_SENSOR_MED_FREQ` (currently 100), reliability from `sensor_qos_reliability`
  - IMU: `depth=QOS_DEPTH_SENSOR_HIGH_FREQ` (currently 500), reliability from `imu_qos_reliability` (default `best_effort`)
- **`odom_bridge`**
  - Subscriptions: `depth=QOS_DEPTH_SENSOR_MED_FREQ` and reliability from `qos_reliability`
  - Publisher `/sim/odom`: `RELIABLE`, `depth=QOS_DEPTH_SENSOR_MED_FREQ`
- **Backend subscriptions**
  - `/sim/odom`, `/sim/loop_factor`, `/sim/anchor_create`, `/sim/imu_segment`, `/sim/rgbd_evidence`: `RELIABLE`, `depth=QOS_DEPTH_SENSOR_MED_FREQ`
- **Frontend publishers**
  - `/sim/loop_factor`, `/sim/anchor_create`, `/sim/imu_segment`, `/cdwm/op_report`, `/cdwm/frontend_status`: `RELIABLE`, `depth=10`
- **C++ `image_decompress_cpp`**
  - Subscriptions: `depth=10`, reliability from `qos_reliability` (`reliable`, `best_effort`, `system_default`, or `both`)
  - Publishers: `depth=10`

### Critical assumptions (must be explicit)

- With no TF in bag, **LiDAR extrinsics are unknown unless declared**.
  - Use launch arg `lidar_base_extrinsic` as `T_base_lidar = [x,y,z,rx,ry,rz]`.
  - If omitted, frontend will warn and pointcloud may remain in sensor frame.
- With no CameraInfo, camera intrinsics must be provided via parameters and logged.
- With no TF, **IMU measurements are assumed to be expressed in the body/base frame** unless an IMU extrinsic is explicitly introduced in the future. Frame IDs are still logged for audit.
- **Random-walk IMU params are intentionally not used**: bias evolution is governed by the adaptive Wishart model (seeded by a fixed prior), not user-provided `imu_*_random_walk` values.

## TurtleBot3 / other bags (placeholder)

Add sections here as datasets are added.

Each section should include:

- bag truth frames
- TF/camera_info presence
- topic inventory and what the pipeline consumes

## NVIDIA r2b / other bags (placeholder)

Add sections here as datasets are added.

Each section should include:

- bag truth frames
- TF/camera_info presence
- topic inventory and what the pipeline consumes
