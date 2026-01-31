# Minimal operational files audit

Audit of the minimal set of files required for a fully operational Golden Child SLAM v2 system (Kimera bag → run + evaluate).

## Operational flow

1. **Rosbag** (Kimera-Multi-Data, PointCloud2 LiDAR + IMU + odom + RGB-D)
2. **Launch** `gc_rosbag.launch.py` with config (e.g. `gc_kimera.yaml`)
3. **gc_sensor_hub**: pointcloud_passthrough, odom_normalizer, imu_normalizer, dead_end_audit → `/gc/sensors/*`
4. **gc_backend_node**: subscribes to `/gc/sensors/*`, runs 14-step pipeline, publishes state/trajectory/map
5. **Eval**: `run_and_evaluate_gc.sh` → ATE/RPE, trajectory export

## Minimal file set

### Config (required at runtime)

| File | Purpose |
|------|--------|
| `fl_ws/src/fl_slam_poc/config/gc_kimera.yaml` | Kimera profile (hub + backend params) |
| `fl_ws/src/fl_slam_poc/config/gc_unified.yaml` | Fallback unified config (Kimera-style default) |
| `fl_ws/src/fl_slam_poc/config/README.md` | Config usage (use gc_unified / gc_kimera) |

### Launch

| File | Purpose |
|------|--------|
| `fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py` | Single launch: bag play, hub, backend, camera nodes |

### Frontend (sensor hub + nodes)

| File | Purpose |
|------|--------|
| `fl_slam_poc/frontend/hub/gc_sensor_hub.py` | Single-process hub entry |
| `fl_slam_poc/frontend/sensors/pointcloud_passthrough.py` | LiDAR: PointCloud2 → /gc/sensors/lidar_points |
| `fl_slam_poc/frontend/sensors/odom_normalizer.py` | Odom → /gc/sensors/odom |
| `fl_slam_poc/frontend/sensors/imu_normalizer.py` | IMU → /gc/sensors/imu |
| `fl_slam_poc/frontend/sensors/depth_passthrough.py` | Depth image → /gc/sensors/camera_depth |
| `fl_slam_poc/frontend/sensors/visual_feature_extractor.py` | RGB-D → 3D features |
| `fl_slam_poc/frontend/sensors/splat_prep.py` | Fused depth splats |
| `fl_slam_poc/frontend/sensors/lidar_camera_depth_fusion.py` | LiDAR–camera depth fusion |
| `fl_slam_poc/frontend/audit/dead_end_audit_node.py` | Unused-topic audit |
| `fl_slam_poc/frontend/audit/wiring_auditor.py` | End-of-run wiring summary |

### Backend

| File | Purpose |
|------|--------|
| `fl_slam_poc/backend/backend_node.py` | ROS node: subscriptions, pipeline invocation, state/trajectory publish |
| `fl_slam_poc/backend/pipeline.py` | 14-step per-scan pipeline |
| `fl_slam_poc/backend/camera_batch_utils.py` | Feature list → MeasurementBatch |
| `fl_slam_poc/backend/map_publisher.py` | PrimitiveMap → PointCloud2 |
| `fl_slam_poc/backend/rerun_visualizer.py` | Rerun logging |
| `fl_slam_poc/backend/diagnostics.py` | Scan diagnostics schema |
| `fl_slam_poc/backend/operators/*` | Predict, deskew, evidence, fusion, recompose, OT, visual_pose_evidence, etc. |
| `fl_slam_poc/backend/structures/primitive_map.py` | PrimitiveMap, fuse, insert, cull, forget |
| `fl_slam_poc/backend/structures/measurement_batch.py` | MeasurementBatch |

### Common

| File | Purpose |
|------|--------|
| `fl_slam_poc/common/belief.py` | BeliefGaussianInfo (22D state) |
| `fl_slam_poc/common/certificates.py` | CertBundle, ExpectedEffect |
| `fl_slam_poc/common/constants.py` | GC_* constants |
| `fl_slam_poc/common/geometry/se3_jax.py` | SE(3) / SO(3) helpers |
| `fl_slam_poc/common/primitives.py` | PSD projection, Cholesky solve |
| `fl_slam_poc/common/ma_hex_web.py` | MA hex candidate generation |

### Tools (eval + dev)

| File | Purpose |
|------|--------|
| `tools/run_and_evaluate_gc.sh` | Primary: run SLAM + eval (ATE/RPE) |
| `tools/common_venv.sh` | Venv detection for eval |
| `Makefile` | `make gc-eval`, `make build`, `make clean` |

### Docs (reference)

| File | Purpose |
|------|--------|
| `docs/GOLDEN_CHILD_INTERFACE_SPEC.md` | Spec anchor |
| `docs/IMU_BELIEF_MAP_AND_FUSION.md` | Pipeline and data flow |
| `docs/PIPELINE_ORDER_AND_EVIDENCE.md` | Corrected spine (z_lin, evidence before map update) |
| `docs/PIPELINE_DEPTH_CONTRACT.md` | Depth fusion contract |
| `docs/FRAME_AND_QUATERNION_CONVENTIONS.md` | Frames and quat order |
| `docs/KIMERA_FRAME_MAPPING.md` | Kimera bag frames |
| `AGENTS.md` | Agent instructions |

## Removed / archived (not required for operation)

- **Livox**: livox_converter, livox_ros_driver2 dependency, livox config paths → archived; LiDAR path is pointcloud_passthrough only (PointCloud2).
- **Bins**: bin_atlas, MapBinStats → archived; map is PrimitiveMap only.
- **Lidar bucket IW**: lidar_bucket_noise_iw_jax (structures + operators) → archived.
- **validate_livox_converter.py** → archive.
- **gc_backend.yaml**: deprecated; use gc_unified.yaml or gc_kimera.yaml.

## Cleanup

- `make clean`: removes `results/gc_*`, `results/gc_slam_diagnostics.npz`, and `fl_ws/build`, `fl_ws/install`, `fl_ws/log` to reduce bulk.
- `results/` is in `.gitignore`; run artifacts are not committed.

## Verification

- Build: `cd fl_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fl_slam_poc && source install/setup.bash`
- Eval: `make gc-eval` or `bash tools/run_and_evaluate_gc.sh` (default profile: Kimera).
