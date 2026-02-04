# Geometric Compositional SLAM v2 - Complete System Dataflow Diagram

This document describes the complete dataflow of the Geometric Compositional SLAM v2 system, including all nodes, topics, and dead-ended items.

## Interactive Diagram

The diagram is available as an interactive HTML file:

- **View:** [system_dataflow_d3.html](../archive/docs/system_dataflow_d3.html) — open in a browser for the full dataflow diagram (archived).

## Topic Flow Details

### Processed Topics (Active Pipeline)

1. **`/acl_jackal/jackal_velocity_controller/odom`** → `odom_normalizer` → **`/gc/sensors/odom`** → Backend
   - Status: ✅ **FUSED** (pose + velocity + yaw-rate + pose–twist consistency)
   - Message: `nav_msgs/Odometry` (absolute → normalized)
   - Frame: `acl_jackal2/odom` → `acl_jackal2/base`

2. **`/acl_jackal/lidar_points`** → `pointcloud_passthrough` → **`/gc/sensors/lidar_points`** → Backend
   - Status: ✅ **FUSED** (14-step pipeline: Matrix Fisher rotation + planar translation evidence)
   - Message: `sensor_msgs/PointCloud2` (VLP-16)
   - Frame: `acl_jackal2/velodyne_link`

3. **`/acl_jackal/forward/imu`** → `imu_normalizer` → **`/gc/sensors/imu`** → Backend
   - Status: ✅ **FUSED** (gravity evidence, gyro evidence, preintegration)
   - Message: `sensor_msgs/Imu`
   - Frame: `acl_jackal2/forward_imu_optical_frame`

### Dead-End Topics (Tracked but Not Processed)

Topics present in the bag but **explicitly not consumed** by the GC v2 pipeline (tracked by `dead_end_audit`):

1. **`/acl_jackal/forward/color/image_raw/compressed`** - Optional when camera enabled
2. **`/acl_jackal/forward/depth/image_rect_raw`** - Optional when camera enabled

### Output Topics

1. **`/gc/state`** - `nav_msgs/Odometry`
   - Posterior pose estimate with 6×6 covariance
   - Frame: `header.frame_id` = odom_frame (Kimera: `acl_jackal2/odom`), `child_frame_id` = base_frame (Kimera: `acl_jackal2/base`)

2. **`/gc/trajectory`** - `nav_msgs/Path`
   - Trajectory path built from state estimates

3. **`/gc/status`** - `std_msgs/String` (JSON)
   - Runtime status: odom_count, scan_count, imu_count, pipeline_runs, map_bins_active

4. **`/gc/runtime_manifest`** - `std_msgs/String` (JSON)
   - System configuration: enabled sensors, backends, operators, topic mappings

5. **`/gc/dead_end_status`** - `std_msgs/String` (JSON)
   - Dead-end topic counts and timestamps

6. **`/tf`** - TF frames
   - Published by backend: `odom_frame` → `base_link`

7. **Trajectory File** (TUM format)
   - Exported to disk for evaluation (`tools/evaluate_slam.py`)

## Node Responsibilities

### gc_sensor_hub (Single Process)
- **pointcloud_passthrough**: Republishes bag LiDAR topic to `/gc/sensors/lidar_points` (VLP-16 PointCloud2)

### gc_backend_node
- Subscribes **ONLY** to canonical topics (`/gc/sensors/*`)
- Runs 14-step fixed-cost pipeline per scan (LiDAR-triggered; IMU and odom consumed per scan)
- Manages state: BeliefGaussianInfo, MapStats, Hypotheses
- Publishes state, trajectory, status, and runtime manifest

### wiring_auditor
- Subscribes to `/gc/status`, `/gc/dead_end_status`, `/gc/runtime_manifest`
- Produces consolidated end-of-run summary
- Writes JSON summary for evaluation script integration

## Current Status Notes

✅ **All three sensor modalities are FUSED** into the belief state:

- **LiDAR**: Matrix Fisher rotation + planar translation evidence (closed‑form pose information)
- **IMU**: `ImuVMFGravityEvidenceTimeResolved` (vMF Laplace on rotation) + `ImuGyroRotationEvidence` (Gaussian SO(3))
- **Odom**: `OdomQuadraticEvidence` (Gaussian SE(3) pose factor)

The evidence terms are combined additively: `L_evidence = L_lidar + L_odom + L_imu + L_gyro + L_imu_preint + L_planar + L_vel + L_wz + L_consistency` (and same for h).

✅ **Dead-end topics are explicitly tracked** for accountability - no data is silently ignored.

## Configuration

Dead-end topics are configured in:
- `fl_ws/src/fl_slam_poc/config/gc_dead_end_audit.yaml`

Canonical topic mappings are configured in:
- `fl_ws/src/fl_slam_poc/config/gc_unified.yaml`

## References

- **Topic usage**: `docs/KIMERA_DATASET_AND_PIPELINE.md`
- **Pipeline reference** (raw topics → frontend → backend → fusion): `docs/IMU_BELIEF_MAP_AND_FUSION.md`
- **Sigma_g and fusion**: `docs/SIGMA_G_AND_FUSION_EXPLAINED.md`
- **Wiring audit**: wiring auditor produces end-of-run summary (see launch)
- **Fusion status**: All sensors (LiDAR + IMU + Odom) are fused; evidence sum includes L_imu_preint.
- **Geometric Compositional spec**: `docs/GC_SLAM.md`
