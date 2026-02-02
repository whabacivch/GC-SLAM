# Bag Topics and Usage Map (Canonical)

**Canonical bag (only one we use):** The project uses a single bag for all testing. It is the one referenced in `tools/run_and_evaluate_gc.sh`:

- **Bag:** `rosbags/Kimera_Data/ros2/10_14_acl_jackal-005`
- **Ground truth:** `rosbags/Kimera_Data/ground_truth/1014/acl_jackal_gt.tum`
- **Config:** `fl_ws/src/fl_slam_poc/config/gc_kimera.yaml`

All configs, docs, and diagnostic examples refer to this bag. We do not run or support other bags; the eval script is the only test path.

---

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

## Kimera (current dataset)

The project uses **Kimera** rosbags (acl_jackal, Velodyne VLP-16). Frame names, topics, and calibration are in [KIMERA_FRAME_MAPPING.md](KIMERA_FRAME_MAPPING.md) and [KIMERA_BAG_INSPECTION.md](KIMERA_BAG_INSPECTION.md).

### Bag topics (Kimera acl_jackal)

- **Odom:** `/acl_jackal/jackal_velocity_controller/odom` → `odom_normalizer` → `/gc/sensors/odom` → Backend. Frame: `acl_jackal2/odom` → `acl_jackal2/base`.
- **LiDAR:** `/acl_jackal/lidar_points` (`sensor_msgs/PointCloud2`, VLP-16) → `pointcloud_passthrough` → `/gc/sensors/lidar_points` → Backend. Frame: `acl_jackal2/velodyne_link`.
- **IMU:** `/acl_jackal/forward/imu` → `imu_normalizer` → `/gc/sensors/imu` → Backend. Frame: `acl_jackal2/forward_imu_optical_frame`.
- **Camera (optional):** `/acl_jackal/forward/color/image_raw/compressed`, `/acl_jackal/forward/depth/image_rect_raw`; see launch camera args.

### Summary (bag truth)

- **TF**: May be absent; extrinsics from config (`T_base_lidar`, `T_base_imu`).
- **Odom frames**: `header.frame_id = acl_jackal2/odom`, `child_frame_id = acl_jackal2/base`.
- **LiDAR frame**: `/acl_jackal/lidar_points.header.frame_id = acl_jackal2/velodyne_link`.
- **IMU frame**: `/acl_jackal/forward/imu.header.frame_id = acl_jackal2/forward_imu_optical_frame`.

### Config (gc_kimera.yaml)

- **odom_frame**: `acl_jackal2/odom`, **base_frame**: `acl_jackal2/base`.
- **T_base_lidar**, **T_base_imu**: From Kimera calibration; see `tools/kimera_calibration_to_gc.py`.
- **lidar_sigma_meas**: `1e-3` (VLP-16).

### Topic inventory (Kimera)

Run:

```bash
.venv/bin/python tools/inspect_rosbag_deep.py rosbags/Kimera_Data/ros2/10_14_acl_jackal-005 --json /tmp/kimera_summary.json
```

#### Key bag topics

| Topic | Message type | Used now? | How it is used |
| --- | --- | --- | --- |
| `/acl_jackal/jackal_velocity_controller/odom` | `nav_msgs/msg/Odometry` | **Yes** | Normalized to `/gc/sensors/odom`; fused in backend. Frame: `acl_jackal2/odom` → `acl_jackal2/base`. |
| `/acl_jackal/lidar_points` | `sensor_msgs/msg/PointCloud2` | **Yes** | Passthrough to `/gc/sensors/lidar_points`; backend consumes. Frame: `acl_jackal2/velodyne_link`. |
| `/acl_jackal/forward/imu` | `sensor_msgs/msg/Imu` | **Yes** | Normalized to `/gc/sensors/imu`; backend preintegration + evidence. |
| `/acl_jackal/forward/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Optional | Camera RGB for primitive pose evidence when enabled. |
| `/acl_jackal/forward/depth/image_rect_raw` | `sensor_msgs/msg/Image` | Optional | Camera depth when enabled. |

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

#### `sensor_msgs/msg/Imu` (Kimera: `/acl_jackal/forward/imu`)

- **Fields consumed (frontend)**
  - `header.stamp`, `header.frame_id` (buffering + audit logs)
  - `angular_velocity.{x,y,z}` and `linear_acceleration.{x,y,z}`
- **Fields ignored**
  - `orientation` and all IMU covariance arrays
- **Units**
  - Gyro: **rad/s**. Accel: scaled by config `imu_accel_scale` (Kimera default 1.0) to m/s² when needed.

#### `sensor_msgs/msg/Image` (internal `/camera/image_raw`, `/camera/depth/image_raw`)

- Produced by `image_decompress_cpp`.
- **RGB output**: `encoding="rgb8"` (converted from decoded BGR).
- **Depth output**: `encoding="32FC1"` (meters). If input is 16-bit depth in mm and `depth_scale_mm_to_m=true`, it is converted to meters.

#### `sensor_msgs/msg/PointCloud2` (internal `/lidar/points`)

- Kimera: native `PointCloud2` from bag (VLP-16). See [POINTCLOUD2_LAYOUTS.md](POINTCLOUD2_LAYOUTS.md). Backend uses `pointcloud_layout: vlp16`.
- Frontend/backend parse x/y/z and (when layout matches) ring, tag, timestamps; other fields are preserved for audit.

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
