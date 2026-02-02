# FL-SLAM Roadmap (Impact Project v1)

This roadmap is organized around the **Geometric Compositional SLAM v2** implementation. **Code is the source of truth;** this doc reflects the repo as of **2026-02-02**.

- **Primary operational code**: Required to run `tools/run_and_evaluate_gc.sh` (GC v2 on Kimera). Config: `fl_ws/src/fl_slam_poc/config/gc_unified.yaml`.
- **Current pipeline**: LiDAR + camera (RGBD). Camera: single topic from `camera_rgbd_node`, ring buffer, visual pose evidence, camera RGB → map/splat colors. Rerun: post-hoc build from `splat_export.npz` + trajectory; eval script opens browser.
- **Immediate priorities**: Design gaps (see `docs/PIPELINE_DESIGN_GAPS.md`), IW tuning/diagnostics, optional bin-level motion smear.
- **Future work**: Camera-frame vMF splat map (§4), alternative datasets (TB3, r2b), visual loop factors, semantic/categorical layer.

**Where to look:** Current architecture and entry points are in **§1 (GC v2 Status)** and **Completed Milestones** above. Sections 2 (IMU) and 3–8 retain historical and future-work detail; implementation status is as of the date at the top.

---

## ✅ Completed Milestones

### 22D State & 14-Step Pipeline (current)
- **22D augmented state** (per `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py`): `[trans(3), rot(3), vel(3), bg(3), ba(3), dt(1), ex(6)]`
- **14-step pipeline per scan**: PointBudgetResample → PredictDiffusion → DeskewConstantTwist → BinSoftAssign → ScanBinMomentMatch → MatrixFisherRotation → PlanarTranslationEvidence → Evidence (odom pose + odom twist + planar priors + IMU + **visual pose** + LiDAR) → FusionScaleFromCertificates → InfoFusionAdditive → PoseUpdateFrobeniusRecompose → PoseCovInflationPushforward → AnchorDriftUpdate
- **LiDAR evidence**: Matrix Fisher rotation + planarized translation with self-adaptive z precision (`matrix_fisher_evidence.py`, `build_combined_lidar_evidence_22d`)
- **Camera evidence**: Single RGBD topic (`/gc/sensors/camera_rgbd` from `camera_rgbd_node`); ring buffer; pick frame by min |t_frame − t_scan|; VisualFeatureExtractor + splat_prep_fused → camera batch; **camera RGB sampled per feature (Feature3D.color)** and passed via `feature_list_to_camera_batch` → map fusion (responsibility-weighted colors). Visual pose evidence at z_lin (`visual_pose_evidence.py`). Pipeline merges camera batch with LiDAR surfels (`lidar_surfel_extraction.py` base_batch=camera_batch).
- **IMU**: Time-resolved vMF tilt, gyro evidence, preintegration factor; preintegration window sliced to 512 steps (`GC_MAX_IMU_PREINT_LEN`)
- **Odom**: Pose evidence + twist evidence (velocity, yaw-rate, pose–twist kinematic consistency); see `odom_evidence.py`, `odom_twist_evidence.py`
- **Planar**: Planar z and vel_z priors; map update sets `t_hat[2]=0` to break z feedback (`planar_prior.py`, `map_update.py`)
- **Adaptive noise**: Inverse-Wishart for process Q and measurement Σ (gyro, accel, LiDAR); readiness weights (no gates); updates every scan (`structures/inverse_wishart_jax.py`, `measurement_noise_iw_jax.py`, `lidar_bucket_noise_iw_jax.py`)
- **Smoothed initial anchor**: Provisional A0 on first odom; after K odom, A_smoothed from closed-form aggregate; export = anchor_correction ∘ pose_belief (`backend_node.py`, `gc_unified.yaml` init_window_odom_count)
- **Wahba removed**: Pipeline uses Matrix Fisher only; legacy Wahba in `archive/legacy_operators/`
- **Rerun**: No live recording during eval. Post-run `tools/build_rerun_from_splat.py` builds `.rrd` from `splat_export.npz` + trajectory (Ellipsoids3D, BEV 15 shading, trajectory); `run_and_evaluate_gc.sh` opens Rerun in browser when available.

### Package & Entry Points
- **Package**: `fl_ws/src/fl_slam_poc/fl_slam_poc/` — `common/`, `frontend/` (hub, sensors, audit), `backend/` (pipeline, operators, structures)
- **Launch**: `fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py`; config: `config/gc_unified.yaml` (single source; `gc_kimera.yaml` removed)
- **Nodes**: `gc_sensor_hub` (frontend), `camera_rgbd_node` (C++: compressed RGB + raw depth → RGBDImage), `gc_backend_node` (backend); sensors: pointcloud_passthrough, odom_normalizer, imu_normalizer
- **Reference**: `docs/Self-Adaptive Systems Guide.md`, `docs/PIPELINE_DESIGN_GAPS.md`, `docs/IMU_BELIEF_MAP_AND_FUSION.md`, `docs/CAMERA_OVERHAUL_AND_SCAN_PATHWAY_PLAN.md`

---

## Immediate Priority: Hardening & Design Gaps

**Goal:** Align with full spec (`docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md`), close remaining design gaps in `docs/PIPELINE_DESIGN_GAPS.md`, and improve robustness/metrics.

### Done (current pipeline)
- **IW process + measurement + LiDAR bucket**: Per-block IW states; readiness weights (no gates); datasheet priors; updates every scan. Files: `backend/structures/inverse_wishart_jax.py`, `operators/measurement_noise_iw_jax.py`, `operators/lidar_bucket_noise_iw_jax.py`, `operators/inverse_wishart_jax.py`.
- **Sensor evidence**: IMU (vMF tilt, gyro, preintegration), odom (pose + twist + kinematic consistency), planar priors, LiDAR (Matrix Fisher + planar translation). All in 22D info form.
- **Diagnostics**: MF SVD, scatter eigenvalues, factor ledger, conditioning, timing (`enable_timing`); dashboard and NPZ schema updated.

### Next: Design gaps & tuning

1. **Pose–twist as joint observation** (PIPELINE_DESIGN_GAPS §4): Model pose and twist with declared joint/cross-covariance; reflect kinematics in evidence.
2. **Cross-sensor consistency as evidence**: Use dyaw_gyro / dyaw_odom / dyaw_mf agreement in the model (currently diagnostics only).
3. **Use more raw message info**: IMU message covariances and orientation not consumed; LiDAR intensity not used. Optional: forward/lateral decomposition.
4. **Q trans block z**: Use `GC_PROCESS_Z_DIFFUSION` for z in process noise (PIPELINE_DESIGN_GAPS §5.5).
5. **Rotation/tilt**: Audit gravity/IMU alignment; optional tilt prior for planar robot; inflate odom roll/pitch variance if bag publishes tight tilt.

### Optional: Bin-level motion smear (PIPELINE_DESIGN_GAPS §5.8)
- **Goal:** Replace per-point deskew with bin-level covariance inflation `Σ_motion,b` (closed-form 2nd-order rotational smear).
- **Operator:** `BinMotionSmearCovariance` between BinSoftAssign and ScanBinMomentMatch; config `deskew_model: per_point | bin_smear_2nd_order`; emit Frobenius correction when approximation triggers.
- **Diagnostics:** Per-bin `Σ_motion,b`, time moments, leak sentinels in NPZ/dashboard. IMU preint already sliced to 512 steps; `enable_timing` and per-stage timers in pipeline/diagnostics.

### Key Invariants (Must Preserve)
- No gating: all adaptations via continuous scaling/weights
- No fixed constants in the loop: noise params from IW (priors are config)
- Branch-free: IW and evidence run every scan

---

## 0) Ground-Truth Facts (Kimera Bag) + “No Silent Assumptions”

**Dataset:** Project uses **Kimera only** (M3DGR/Dynamic01 removed). Default bag/topics/extrinsics are Kimera.

These are **observed facts from the Kimera bag** and should be treated as the default configuration unless explicitly overridden:

- **No `CameraInfo` topics in the bag** → intrinsics must be declared via parameters and logged.
- **No `/tf` or `/tf_static` topics in the bag** → camera extrinsics are not available online unless we provide them (URDF/static transform parameter).
- **Odom frames in the bag** (Kimera acl_jackal):
  - `world_frame` = `/odom.header.frame_id` = `acl_jackal2/odom`
  - `base_frame` = `/odom.child_frame_id` = `acl_jackal2/base`
- **LiDAR frame**: `/acl_jackal/lidar_points.header.frame_id` = `acl_jackal2/velodyne_link`
- **IMU frame**: `/acl_jackal/forward/imu.header.frame_id` = `acl_jackal2/forward_imu_optical_frame`. See [KIMERA_FRAME_MAPPING.md](docs/KIMERA_FRAME_MAPPING.md).
- **RGB + depth in the bag** (Kimera): Topics use `acl_jackal/forward/` prefix. Launch defaults: `camera_rgb_compressed_topic` = `/acl_jackal/forward/color/image_raw/compressed`, `camera_depth_raw_topic` = `/acl_jackal/forward/depth/image_rect_raw`. `camera_rgbd_node` pairs them and publishes `/gc/sensors/camera_rgbd` (RGBDImage). Depth scale mm→m applied when configured.

**Policy (by construction):**
- Do not “guess” frames or extrinsics silently.
- All frame assumptions (world/base/camera frames, intrinsics source, extrinsics source) must be explicit parameters and logged from the first N message headers.

---

## 1) GC v2 Status (Current Baseline)

**Primary entrypoint**
- `tools/run_and_evaluate_gc.sh`: runs GC v2 on Kimera end-to-end (SLAM + alignment + metrics + plots + dashboard + Rerun). Artifacts under `results/gc_YYYYMMDD_HHMMSS/`. Uses `config/gc_unified.yaml`.

**Launch**
- `fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py`; config: `fl_ws/src/fl_slam_poc/config/gc_unified.yaml`

**Nodes**
- Frontend: `gc_sensor_hub` (`fl_slam_poc/frontend/hub/gc_sensor_hub.py`) — publishes `/gc/sensors/lidar_points`, `/gc/sensors/odom`, `/gc/sensors/imu`
- Camera: `camera_rgbd_node` (C++: `src/camera_rgbd_node.cpp`) — subscribes to compressed RGB + raw depth, pairs by timestamp, publishes `/gc/sensors/camera_rgbd` (RGBDImage). Required for backend (backend skips scan if camera ring buffer empty).
- Backend: `gc_backend_node` (`fl_slam_poc/backend/backend_node.py`) — 14-step pipeline, IW updates, camera batch + LiDAR merge, visual pose evidence, map/splat export with camera colors
- Sensors (used by hub): `pointcloud_passthrough`, `odom_normalizer`, `imu_normalizer`; optional: `dead_end_audit_node`, `wiring_auditor`

**Evaluation**
- `tools/align_ground_truth.py`, `tools/evaluate_slam.py`, `tools/slam_dashboard.py`, `tools/build_rerun_from_splat.py` (post-run Rerun .rrd from splat_export + trajectory)

**State (22D per chart)**
```python
# fl_slam_poc/common/constants.py: GC_D_Z = 22
z = [trans(0:3), rot(3:6), vel(6:9), bg(9:12), ba(12:15), dt(15:16), ex(16:22)]
# cov = 22×22 in tangent space
```

---

## 2) IMU & Multi-Sensor Fusion (Completed)

### Status: ✅ Implemented

**Current flow**: Backend subscribes to `/gc/sensors/imu` (from gc_sensor_hub); buffers IMU in-node; runs preintegration over the scan window (sliced to 512 steps). State is **22D** (pose + vel + biases + dt + ex). Evidence: time-resolved vMF tilt, gyro rotation, preintegration factor; odom pose + twist + kinematic consistency; planar priors; LiDAR Matrix Fisher + planar translation. IW updates every scan (process, measurement, LiDAR bucket). See Completed Milestones and §1 above.

---

### Keyframe Policy (Required Before IMU Factors)

IMU preintegration and “two-pose factors” require **explicit keyframes**. Keyframes must be:
- Deterministic and auditable (time-based by default).
- Independent of “anchor birth” (which is a probabilistic structure event and not a reliable time axis).

**Keyframe rule (recommended v1):**
- Create keyframe every `keyframe_interval_sec` (default 1.0s).
- Optionally add a motion criterion as a second trigger, but it must be logged as a declared policy.

Everything that produces interval factors (wheel odom, IMU) should be scheduled between consecutive keyframes `(i → i+1)`.

---

### Phase 1: IMU Infrastructure & Preintegration

#### 1.1 IMU Sensor I/O

**Files:** `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/processing/sensor_io.py`

**Changes:**
- Add IMU subscription to `SensorIO` class with configurable topic parameter (`imu_topic`, default `/camera/imu`)
- Add `imu_buffer` to store `(timestamp, accel, gyro)` tuples (high-rate buffering, keep ~1s window)
- Implement `get_imu_measurements(start_sec, end_sec)` to retrieve IMU data for preintegration intervals
- Add duplicate detection for IMU messages (similar to existing `_on_odom`, `_on_image` handlers)
- Add status monitoring registration for IMU sensor

**Invariants preserved:** Pure I/O layer, no math/inference in `SensorIO`; buffering only, no gating.

#### 1.2 IMU Preintegration Operator ✅ COMPLETED (Removed)

**Implementation Status:**
- ✅ Contract B architecture uses raw IMU segments; backend re-integrates in-kernel
- ✅ Preintegration utilities removed to avoid conflicting sources of truth
- ✅ Reference: Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (TRO 2017)

**Note**: Preintegration utilities were removed from the repo to prevent duplicate IMU integration paths.

#### 1.3 IMU Segment Message ✅ COMPLETED

**Files:**
- ✅ `fl_ws/src/fl_slam_poc/msg/IMUSegment.msg` (replaces deprecated `IMUFactor.msg`)
- ✅ `fl_ws/src/fl_slam_poc/CMakeLists.txt` (already includes `IMUSegment.msg`)

**Message**: `IMUSegment.msg` contains raw IMU measurements between keyframes with explicit units/frames/timebase semantics. Backend re-integrates these segments with sigma-point propagation.

#### 1.4 Frontend IMU Segment Publishing ✅ COMPLETED

**Files:** `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`

**Status:**
- ✅ Publishes `/sim/imu_segment` raw IMU slices at keyframe creation
- ✅ Parameters: `imu_topic`, `imu_gyro_noise_density`, `imu_accel_noise_density`, `imu_gyro_random_walk`, `imu_accel_random_walk`, `enable_imu`
- ✅ Keyframe-based timing (anchor birth remains separate event)

---

### Phase 2: Backend 15D State Extension ✅ COMPLETED

#### 2.1 State Representation Update ✅ COMPLETED

**Files:** `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`

**State (15DOF):**
```python
mu = [x, y, z, rx, ry, rz, vx, vy, vz, bg_x, bg_y, bg_z, ba_x, ba_y, ba_z]
cov = 15×15 matrix  # [δp, δθ, δv, δb_g, δb_a] in tangent space
```

**Status:**
- ✅ `SparseAnchorModule` extended to 15D (velocity=0, biases=0 initialization)
- ✅ `cov` extended to 15×15 with high-variance priors for v, b_g, b_a
- ✅ `make_evidence(mu, cov)` handles 15D information form (L: 15×15, h: 15D)

**Compatibility:**
- ✅ Odometry factors remain 6DOF (update first 6 rows/cols of 15×15 L/h)
- ✅ Loop factors remain 6DOF (update first 6 rows/cols of 15×15 L/h)
- ✅ IMU segments integrated in-kernel with sigma-point propagation (Contract B)
- ✅ Biases [9:15] updated by IMU integration and random walk process noise

#### 2.2 IMU Segment Fusion in Backend ✅ COMPLETED

**Files:** `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`

**Status:**
- ✅ Subscription: `/sim/imu_segment` → `on_imu_segment(msg: IMUSegment)`
- ✅ Contract B architecture: raw IMU segments re-integrated with sigma-point propagation
- ✅ Two-state factor update: joint e-projection on [pose_i, pose_j] (30D) followed by exact Schur marginalization
- ✅ Emits `OpReport` with Contract B diagnostics: `approximation_triggers: {"Linearization"}`, `frobenius_applied: True`

#### 2.3 Information-Form Gaussian Fusion (15D) ✅ COMPLETED

**Files:** `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/gaussian_info.py` (flattened from `backend/fusion/gaussian_info.py`)

**Status:**
- ✅ `make_evidence`, `fuse_info`, `mean_cov` work correctly for 15D inputs
- ✅ `embed_info_form` utility for dimension embedding (6D → 15D)
- ✅ State ordering convention documented: `[p(3), θ(3), v(3), b_g(3), b_a(3)]`

---

### Phase 3: Wheel Odometry Separation & Factor Scheduling

#### 3.1 Wheel Odometry as Separate Factor

**Files:** `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`

**Current:** `/sim/odom` is treated as "pose prior" (6DOF delta)

**New approach:**
- Treat wheel odom as a partial observation: constrains `[x, y, yaw]` strongly, `[z, roll, pitch]` weakly (high variance)
- Construct information form using **declared parameters** (no hard-coded constants), with two safe modes:
  1. **Use odom-provided covariance** (preferred when trustworthy): read `nav_msgs/Odometry.pose.covariance`.
  2. **Fallback diagonal covariance** (explicit params): `wheel_sigma_xy`, `wheel_sigma_yaw`, `wheel_sigma_z`, `wheel_sigma_roll`, `wheel_sigma_pitch`.
- Do NOT pre-fuse wheel odom with IMU upstream; keep as independent factor stream

**Rationale:** Wheel odom measures what it measures (XY + yaw), not a full 6DOF pose. Allows IMU to dominate roll/pitch estimation.

**Self-Adaptive Systems Integration**: See `docs/Self-Adaptive Systems Guide.md` Section 2 (Adaptive Sensor Weighting via Fisher Information) for future enhancement: sensor weights should adapt based on Fisher information and reliability scores, replacing fixed covariance with online-estimated Wishart priors.

#### 3.2 Factor Scheduling & Keyframe Policy

**Files:**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`

**Changes:**
- **Keyframe creation policy (deterministic + auditable):**
  - Time-based: every `keyframe_interval_sec` (default 1.0s)
  - Motion-based (optional): when `||delta_pose|| > keyframe_motion_threshold`
  - Log each keyframe decision: `"Keyframe {id} created at t={t:.3f}s (policy: {reason})"`
- **Factor streams (explicit):**
  - Odometry factors: published between consecutive keyframes (6DOF delta)
  - IMU segments: published between consecutive keyframes (raw IMU measurements, Contract B architecture)
  - Loop factors: published when loop detected (6DOF relative pose)
  - RGB-D factors (optional): published when RGB-D pair available (3DOF position)
- **Backend factor ingestion logging:**
  - Track counts: `n_odom_factors`, `n_imu_factors`, `n_loop_factors`, `n_rgbd_factors`
  - Publish in `/cdwm/backend_status`: `{"factors": {"odom": n, "imu": m, "loop": k, "rgbd": r}}`

---

### Phase 4: Dense RGB-D in 3D Mode (Optional TF)

**Files:** `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`

**Current:** Dense RGB-D evidence (`_publish_rgbd_evidence`) is only called in 2D LaserScan mode

**Changes:**
- Enable `_publish_rgbd_evidence` in 3D pointcloud mode when `enable_depth=True` and `publish_rgbd_evidence=True`
- Add parameter: `camera_base_extrinsic` (6DOF SE(3) static transform, default None)
- **No-TF mode (documented assumption):**
  - If `camera_base_extrinsic` is provided: use declared static transform (no TF lookup)
  - If `camera_base_extrinsic` is None and TF lookup fails: log warning and skip dense evidence (don't crash)
  - Document in `docs/3D_POINTCLOUD.md`: "RGB-D evidence requires either (1) TF tree or (2) declared camera_base_extrinsic parameter"

---

### Phase 5: Validation & Evaluation Hardening

#### 5.1 Provenance & Contribution Summary

**Files:**
- New: `fl_ws/src/fl_slam_poc/fl_slam_poc/common/provenance.py` (or `backend/provenance.py` if backend-specific)
- Update: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`, `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`

**Implementation:**
- Create `provenance.py` module with functions:
  - `get_git_sha()` → current git commit SHA
  - `get_module_path(module_name)` → resolved import path (verifies correct source is loaded)
  - `get_factor_counts()` → dictionary of factor stream counts
- Add provenance logging to backend/frontend node startup (git SHA, module paths, factor streams enabled)
- Add periodic factor count summary (every 10s)

#### 5.2 Baseline Evaluation After IMU Integration

**Files:** `tools/run_and_evaluate_gc.sh`, `tools/evaluate_slam.py`

**Changes:**
- Add rotation metrics (RPE rotation, ATE rotation) to `tools/evaluate_slam.py` (already present, verify)
- Run baseline before/after IMU integration on primary dataset (Kimera):
  - Baseline (current, no IMU): record ATE/RPE translation + rotation
  - With IMU: record ATE/RPE translation + rotation
  - **Expected:** rotation RPE should improve ~30-50%
- Document results in `CHANGELOG.md` under milestone entry

#### 5.3 Focused Tests

**Files:**
- New: `fl_ws/src/fl_slam_poc/test/test_imu_preintegration.py`
- New: `fl_ws/src/fl_slam_poc/test/test_15d_state.py`

**Test coverage:**
1. **IMU preintegration:** Synthetic IMU data → verify delta_R, OpReport, covariance 9×9 PD
2. **15D state fusion:** Verify odometry factor (6DOF) updates only first 6 dimensions, IMU factor (9DOF) updates first 9 dimensions
3. **Ground truth non-ingestion:** Verify backend never subscribes to `/vrpn_client_node/*` or `/ground_truth` topics

---

### Phase 6: Launch File & Parameter Updates

#### 6.1 Launch Parameters (Kimera / GC)

**Files:** `fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py` (config: `config/gc_unified.yaml`)

**Changes:**
- Add IMU parameters (if not already in gc_unified):
  ```python
  DeclareLaunchArgument("enable_imu", default_value="true"),
  DeclareLaunchArgument("imu_topic", default_value="/camera/imu"),
  DeclareLaunchArgument("imu_gyro_noise_density", default_value="1.0e-3"),
  DeclareLaunchArgument("imu_accel_noise_density", default_value="1.0e-2"),
  DeclareLaunchArgument("imu_gyro_random_walk", default_value="1.0e-5"),
  DeclareLaunchArgument("imu_accel_random_walk", default_value="1.0e-4"),
  ```

#### 6.2 Alternative Dataset Support (Newer College, etc.)

**Files:** Add launch for 3D rosbag under `fl_ws/.../launch/` when needed

**Changes:**
- Add same IMU parameters as above
- Add override examples in `docs/datasets/DATASET_DOWNLOAD_GUIDE.md`:
  ```bash
  # Newer College (LiDAR IMU)
  ros2 launch fl_slam_poc poc_3d_rosbag.launch.py \
    bag:=rosbags/newer_college/01_short_experiment.bag \
    imu_topic:=/os1_cloud_node/imu \
    pointcloud_topic:=/os1_cloud_node/points
  ```

---

### Milestones & Ordering

**Milestone 1: IMU Sensor + Preintegration (Phase 1) ✅ COMPLETED**
- ✅ IMU I/O, message definition (`IMUSegment.msg`), frontend publishing
- ✅ Contract B architecture: raw IMU segments published to `/sim/imu_segment`
- ✅ **Validation:** Kimera (or primary dataset) runs with `enable_imu:=true`, `/sim/imu_segment` messages published
- ✅ **Logs:** OpReport shows `Linearization` trigger + Frobenius correction

**Milestone 2: 15D State + Backend Fusion (Phase 2) ✅ COMPLETED**
- ✅ State representation extended to 15D, Contract B IMU fusion implemented
- ✅ **Validation:** Kimera runs, backend ingests IMU segments without errors
- ✅ **Logs:** Backend status shows IMU segments processed, Contract B diagnostics present

**Milestone 3: Wheel Odom Separation (Phase 3.1)**
- Update wheel odom to be a partial factor (strong XY/yaw, weak z/roll/pitch)
- **Validation:** Compare trajectory before/after (expect similar XY, improved roll/pitch when fused with IMU)

**Milestone 4: Factor Scheduling Hardening (Phase 3.2)**
- Deterministic keyframe policy + explicit factor stream logging
- **Validation:** Verify factor counts in `/cdwm/backend_status` match expected rates

**Milestone 5: Dense RGB-D in 3D Mode (Phase 4)**
- Enable `_publish_rgbd_evidence` in 3D mode with optional TF
- **Validation:** Run with 3D mode + RGB-D evidence, verify no TF crashes (current pipeline already uses camera RGBD via camera_rgbd_node + visual evidence + map colors)

**Milestone 6: Evaluation Hardening (Phase 5)**
- Add provenance logging, run baseline comparison, add focused tests
- **Validation:** Rotation metrics improve ~30-50% with IMU vs. without

---

### Design Invariants Compliance Checklist

- **Frobenius correction:** IMU preintegration emits `approximation_triggers: {"Linearization"}` and applies third-order correction to covariance
- **Closed-form-first:** All Gaussian fusion remains closed-form (information addition), no iterative solvers
- **Soft association:** No changes to association (still responsibility-based, no gating)
- **One-shot loop correction:** Loop factors remain late evidence, fused via barycenter (no changes needed for IMU work)
- **Jacobian policy:** IMU preintegration uses Jacobians (front-end sensor-to-evidence extraction, explicitly logged)
- **No ground-truth ingestion:** Backend never subscribes to GT topics (verified in tests)
- **OpReport taxonomy:** Every new operator emits OpReport with required fields

#### 2.7 Adaptive IMU Noise Model (Future Enhancement)

**Reference**: `docs/Self-Adaptive Systems Guide.md` Section 1 (Adaptive Noise Covariance via Wishart Conjugate Updates)

**Future Work:**
- Integrate `AdaptiveIMUNoiseModel` from Self-Adaptive Systems Guide
- Replace fixed IMU noise parameters with online Wishart conjugate updates
- Track accelerometer/gyroscope noise and bias random walk separately
- Use adaptive forgetting factor inferred from changepoint/hazard model
- Emit diagnostics: noise covariance traces, anomaly detection

**Files (to be created):**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/adaptive_noise.py` (Wishart prior implementation)
- Integration with `imu_jax_kernel.py` for adaptive noise in sigma-point propagation

**Design Invariants:**
- No hard gates: noise adaptation via continuous Wishart updates, not threshold-based rejection
- Startup is not a mode: behavior emerges from prior effective sample size, not time-based branching
- Constants surfaced as priors: initial confidence (ESS), forgetting factor (hazard rate Beta hyperparameters)

---

## 3) Medium-Term: Alternative Datasets & Algorithm Fixes

**Current:** Kimera only. TB3, r2b, Newer College, etc. are future; launch/config additions when needed.

### A) SE(3) drift investigation

**Symptom:** Trajectory blows up (e.g., kilometers instead of meters).

**Primary files:**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/geometry/se3_jax.py`
- Odom: backend subscribes to `/gc/sensors/odom` (from gc_sensor_hub)

**Checklist:**
- Verify pose composition conventions and frame semantics (odom/base).
- Confirm odometry is delta (twist-integrated) where expected.
- Validate quaternion/rotvec conversions and covariance transport.

### B) Timestamp monotonicity

**Symptom:** Remaining non-monotonic gaps / duplicates impacting association and evaluation.

**Primary files:**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensor_io.py` (flattened from `frontend/processing/sensor_io.py`)
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/timestamp.py` (flattened from `backend/parameters/timestamp.py`)
- `tools/align_ground_truth.py`

### C) TurtleBot3 (2D) validation

**Files:** `tools/download_tb3_rosbag.sh`. Launch: add `fl_ws/.../launch/` entry for TB3 when needed.

### D) NVIDIA r2b (3D) validation / GPU

**Files:** `tools/download_r2b_dataset.sh`. Launch: add entry for 3D rosbag when needed.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/pointcloud_gpu.py` (flattened from `frontend/loops/pointcloud_gpu.py`)

---

## 4) Future Work: Camera-Frame Gaussian Splat Map with vMF Shading

### Overview

This section documents the full specification for a **camera-frame Gaussian splat map** (K=512 splats, J=2 vMF lobes per splat) with view-dependent appearance modeling. This approach replaces histogram-based RGB-D descriptors with a continuous, probabilistic 3D map representation suitable for dense appearance modeling and loop closure.

**Key design principle:** The splat map is a **declared generative model** with exact EM-style updates (E-step: soft responsibilities, M-step: closed-form projections). Outlier handling and spatial/photometric likelihoods are explicit model components, not heuristics.

---

### 4.1 System Mode & Assumptions

**Mode 1: Camera-Frame Map (Recommended)**

- **Map frame:** `camera_color_optical_frame` (avoids TF dependency, robust to missing extrinsics)
- **Camera center:** Origin (c_t = 0 in camera frame)
- **View direction for splat k:**
  ```
  v_k = μ_k / (||μ_k|| + ε)
  ```
  with clamping for near-origin splats (see §4.6.2)

**Inputs (synchronized):**
- RGB (decompressed): `/camera/image_raw` (sensor_msgs/Image, frame_id = camera_color_optical_frame)
- Depth (aligned, decompressed): `/camera/depth/image_raw` (sensor_msgs/Image, same frame_id)
- Odometry `/odom`: optional for logging/recording, not required for core camera-frame inference

---

### 4.2 Intrinsics & Depth Conversion

**Camera intrinsics (fallback if CameraInfo unavailable):**

```
K = [[383,   0, 320],
     [  0, 383, 240],
     [  0,   0,   1]]
```

**Depth scale parameter (mandatory):**
- Depth encoding varies by sensor:
  - `16UC1` often stores depth in **millimeters**
  - `32FC1` often stores depth in **meters**
- Define `depth_scale` such that `z = depth_scale * d` is in meters
- Default suggestions (must confirm message encoding):
  - If 16U mm: `depth_scale = 0.001`
  - If 32F m: `depth_scale = 1.0`

**Backprojection:**

For pixel (u, v) with depth d:

```
z = depth_scale * d
x_camera = z * [(u - cx)/fx, (v - cy)/fy, 1]^T
```

---

### 4.3 Noise Model & Pixel Weight (By-Construction Likelihood)

Add parameter `depth_noise_model` with:

**Linear-in-depth noise:**

```
σ_z(z) = a + b*z
w = 1 / σ_z(z)^2
```

**Interpretation:** `w` is the precision of spatial observation; this is strictly within the likelihood model (not a heuristic).

**Diagnostic:** Log mean `w` per frame. If it collapses to near-zero, likely indicates wrong depth scale or intrinsics.

---

### 4.4 Map Parameters (Per Splat)

Configuration: **K = 512 splats**, **J = 2 directional lobes per splat**

**Geometry (per splat k):**

- Position: `μ_k ∈ ℝ³`
- Covariance: `Σ_k = diag(σ²_{k,x}, σ²_{k,y}, σ²_{k,z})`
- Mixture weight: `π_k ∈ (0, 1)` with `Σ_k π_k = 1`

**Directional gating (vMF lobes):**

- Mean directions: `m_{k,1}, m_{k,2} ∈ S²`
- Concentrations: `κ_{k,1}, κ_{k,2} ≥ 0`
- Gating (softmax over lobes):
  ```
  r_{k,j}(v) = softmax_j(κ_{k,j} * m_{k,j}^T * v)
  ```

**Appearance (Gaussian in Lab color space, per lobe):**

- Color mean: `μ^(y)_{k,j} ∈ ℝ³` (Lab coordinates)
- Color covariance: `Σ^(y)_{k,j} = diag(σ²_{k,j,L}, σ²_{k,j,a}, σ²_{k,j,b})`

**Predicted appearance (view-dependent mixture):**

```
μ̂^(y)_k(v) = Σ_j r_{k,j}(v) * μ^(y)_{k,j}
Σ̂^(y)_k(v) = Σ_j r_{k,j}(v) * Σ^(y)_{k,j} + ε_y*I
p(y | k, v) = 𝒩(y; μ̂^(y)_k(v), Σ̂^(y)_k(v))
```

---

### 4.5 Outlier Model (Photometric-Only for Camera-Frame Mode)

**Outlier parameters:**

- Outlier mixture weight: `π_out`
- Outlier appearance distribution: `p_out(y) = 𝒩(y; μ_out, Σ_out)` with large variance (or uniform-ish approximation)

**Note:** Spatial outlier term omitted in v1 camera-frame mode (will be added in Mode 2 world-map mode).

**Responsibilities (E-step):**

For pixel p with observation (x, y) and view direction v_k:

```
γ_{p,k} = [π_k * p(x|k) * p(y|k,v_k)] / [Σ_{k'∈K_p} π_{k'} * p(x|k') * p(y|k',v_{k'}) + π_out * p_out(y)]

γ_{p,out} = [π_out * p_out(y)] / [Σ_{k'∈K_p} π_{k'} * p(x|k') * p(y|k',v_{k'}) + π_out * p_out(y)]
```

**Diagnostic:** If mean outlier mass per frame > 0.3–0.5, flag "intrinsics/depth mismatch likely."

**Important:** This is a diagnostic threshold only. If we want adaptive behavior (e.g., raising `π_out` or spawning dynamic splats), it must be an explicit, declared meta-operator with logged triggers (not heuristic gating).

---

### 4.6 Candidate Selection & Numeric Safety

**4.6.1 Pixel Subsampling (Performance)**

Prototype defaults (RTX 4050 friendly):

- Stride: 4–8 pixels
- Cap to `max_pixels = 10k–20k` per frame

Only consider pixels with:

- Valid depth: `z > z_min` (e.g., 0.1m)
- Optional depth range: `z < z_max` (e.g., 10m)

**4.6.2 View Direction Safety**

Compute `v_k` with clamping:

```
v_k = μ_k / max(||μ_k||, ρ)
```

with `ρ = 0.1m` (suggested).

If `||μ_k|| < ρ`, either:

- Skip shading update for that splat this frame, or
- Clamp as above and proceed

This avoids NaNs from near-origin splats.

**4.6.3 KNN Candidate Selection**

For each observation point x:

- Choose `M = 16` nearest neighbors among `μ_k` in Euclidean distance
- At K=512, CPU KNN is sufficient (use FAISS CPU, FLANN, or brute-force)
- GPU KNN is optional but supported

---

### 4.7 Per-Frame Inference & Accumulator Updates

**4.7.1 Frame Preprocessing**

1. Decompress RGB + depth images
2. Convert RGB → Lab color space (float32)
3. Subsample pixels (stride + optional random selection)
4. For each sampled pixel:

   - Compute `z`, backproject `x`
   - Compute reliability weight `w`
   - Store `(x, y, w)`

**4.7.2 Responsibilities & Sufficient-Stat Accumulation**

For each observation `(x, y, w)`:

1. **Candidate set K_p:** KNN to `μ_k`

2. **For each candidate k:**

   - Spatial log-likelihood:
     ```
     log p(x|k) = -½(x-μ_k)^T Σ_k^{-1} (x-μ_k) - ½log|Σ_k| + C
     ```

   - View direction: `v_k` (camera-frame)
   - Gating: `r_{k,j}(v_k)`
   - Predicted color mean/cov: `μ̂^(y)_k(v_k)`, `Σ̂^(y)_k(v_k)`
   - Color log-likelihood: `log p(y|k,v_k)`
   - Combined:
     ```
     ℓ_k = log π_k + log p(x|k) + log p(y|k,v_k)
     ```


3. **Outlier log-likelihood:**
   ```
   ℓ_out = log π_out + log p_out(y)
   ```

4. **Normalize via softmax** over `{ℓ_k} ∪ {ℓ_out}` to get `γ_k` and `γ_out`

5. **Accumulate geometry:**

   - `W_k += γ_k * w`
   - `M_k += γ_k * w * x`
   - `Q_k += γ_k * w * x*x^T`

6. **Accumulate vMF (per lobe j):**

   - `γ_{k,j} = γ_k * r_{k,j}(v_k)`
   - `W^(v)_{k,j} += γ_{k,j} * w`
   - `S_{k,j} += γ_{k,j} * w * v_k`

7. **Accumulate appearance (per lobe j):**

   - `W^(y)_{k,j} += γ_{k,j} * w`
   - `M^(y)_{k,j} += γ_{k,j} * w * y`
   - `Q^(y)_{k,j} += γ_{k,j} * w * y*y^T`

Also accumulate frame-level diagnostics:

- Mean outlier mass
- Mean residual

**4.7.3 Apply Updates Once Per Frame (M-Step Projection)**

For each splat k with `W_k > τ`:

**Geometry:**

```
μ_k ← M_k / W_k
Σ_k ← diag(Q_k/W_k - μ_k*μ_k^T) + ε_x*I
```

**Directional lobes (per lobe j with W^(v)_{k,j} > τ_v):**

```
m_{k,j} ← S_{k,j} / ||S_{k,j}||
R̄_{k,j} = ||S_{k,j}|| / W^(v)_{k,j}
κ_{k,j} ← clamp([R̄_{k,j}(3 - R̄_{k,j}²)] / [1 - R̄_{k,j}²], κ_min, κ_max)
```

**Appearance (per lobe j):**

```
μ^(y)_{k,j} ← M^(y)_{k,j} / W^(y)_{k,j}
Σ^(y)_{k,j} ← diag(Q^(y)_{k,j}/W^(y)_{k,j} - μ^(y)_{k,j}*μ^(y)_{k,j}^T) + ε_y*I
```

Reset accumulators for next frame (or use EMA if desired, but EMA must be documented as a prior/forgetting process).

---

### 4.8 Initialization (First Frame)

**4.8.1 Sampling for Diversity**

On first frame:

- Subsample with stride + random selection among valid depth pixels (avoids grid bias)
- Choose K points and set `μ_k = x`

**4.8.2 Covariance Initialization**

Depth noise dominates z; initialize:

- `σ_z` larger than `σ_x, σ_y` (based on noise model at that z)

**4.8.3 Mixture Mass (π_k)**

Initialize uniform, optionally boost nearer points:

```
π_k ∝ 1 / max(z_k, ε)
```

then normalize over k.

**4.8.4 Lobe Initialization (Avoid Perfect Opposition)**

Set:

- `m_{k,1} = μ_k / ||μ_k||`
- `m_{k,2} = normalize(m_{k,1} + δ)`, where `δ` is a small random vector orthogonal-ish to `m_{k,1}`

This avoids forcing a two-lobe symmetry the data may not support.

Set initial `κ` small (0.1–1.0).

**4.8.5 Appearance Initialization**

Use the pixel Lab color at the seed pixel:

- Set both lobes' means close to that color, with moderate variance

---

### 4.9 Diagnostics (Expanded)

Add the following diagnostics to validate splat map quality:

- **Observed vs rendered (predicted) image** at sampled pixels
- **Residual heatmap:** `|y - μ̂^(y)|`
- **Mean and histogram of w** (reliability weights)
- **Active splat count:** number with `π_k` or `W_k` above threshold
- **Mean outlier mass** per frame
- **Kappa histogram** (directional concentration distribution)

**Alert conditions:**

- Outlier mass > 0.3–0.5 persistently ⇒ intrinsics/depth_scale likely wrong
- Mean `w` unexpectedly tiny ⇒ depth scale wrong or noise model too pessimistic
- Many NaNs ⇒ view-direction clamp missing or invalid depth not filtered

---

### 4.10 World-Map Mode (Mode 2) - Future Extension

When enabling world/base map in future:

- Add spatial outlier term `p_out(x)` broad and centered on current camera center
- Add diagnostic: compare distribution of backprojected points in map frame across time; if it "breathes" incoherently, extrinsics are wrong

---

### 4.11 Implementation Files & Structure

**New files:**

- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/splat_map/splat_map.py` - Core splat map class
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/splat_map/vmf_lobe.py` - vMF directional lobe operations
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/splat_map/appearance_model.py` - Lab color Gaussian model
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/splat_map/splat_visualizer.py` - Diagnostics + rendering

**Integration points:**

- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py` - Add splat map instance, update loop
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensor_io.py` - Add Lab color conversion utilities

**Message definitions:**

- New: `fl_ws/src/fl_slam_poc/msg/SplatMapState.msg` - For publishing splat map state to backend
- New: `fl_ws/src/fl_slam_poc/msg/SplatDiagnostics.msg` - For diagnostics visualization

---

### 4.12 Immediate Configuration (Kimera / primary dataset)

Starting parameters for Kimera (or primary) rosbag:

```python
# Camera intrinsics (fallback if no CameraInfo)
camera_fx = 383.0
camera_fy = 383.0
camera_cx = 320.0
camera_cy = 240.0

# Depth configuration (MUST VERIFY encoding)
depth_scale = 0.001  # if 16UC1 in mm, else 1.0 if 32FC1 in m

# Splat map configuration
splat_K = 512          # number of splats
splat_J = 2            # directional lobes per splat
splat_stride = 6       # pixel sampling stride
splat_max_pixels = 15000
splat_candidate_M = 16 # KNN candidates per observation

# Noise model (linear in depth)
depth_noise_a = 0.01   # baseline noise (m)
depth_noise_b = 0.02   # depth-proportional noise (unitless)

# Depth range
depth_min = 0.1        # meters
depth_max = 10.0       # meters

# Regularization
epsilon_x = 1e-6       # spatial covariance floor
epsilon_y = 1e-4       # appearance covariance floor
kappa_min = 0.1        # minimum concentration
kappa_max = 100.0      # maximum concentration

# Outlier model
pi_outlier = 0.1       # outlier mixture weight
```

---

### 4.13 Design Invariants Compliance

**Compliance with Frobenius-Legendre invariants:**

1. **Closed-form-first:** EM updates (E-step: softmax responsibilities, M-step: weighted mean/covariance) are closed-form, exact within the declared Gaussian mixture model
2. **Soft association:** Responsibilities `γ_{p,k}` are derived from declared likelihood model, no hard gating
3. **Declared approximations:**

   - KNN candidate selection (BudgetedRecomposition): explicit approximation operator with declared objective (spatial proximity)
   - Emit `OpReport` with `approximation_triggers: {"BudgetedRecomposition"}`, `frobenius_applied: True` (if covariance updates involve linearization)

4. **Outlier model:** Explicit mixture component, not a heuristic
5. **Operator taxonomy:** All updates emit `OpReport` with family declarations (`family_in: "GaussianMixture"`, `family_out: "GaussianMixture"`, `closed_form: True`)

---

### 4.14 Testing & Validation

**Tests:**

- `fl_ws/src/fl_slam_poc/test/test_splat_map.py` - Unit tests for splat map operations
  - Synthetic RGB-D frame → verify responsibilities sum to 1
  - Verify vMF κ estimation matches expected concentration for synthetic data
  - Verify Lab color Gaussian updates are exact weighted means

**Validation criteria:**

- Mean outlier mass < 0.2 on primary (Kimera) sequences (indicates good intrinsics/depth_scale)
- Active splat count ≈ 400-500 after 10 frames (indicates diversity maintained)
- Residual heatmap shows spatially coherent errors (not salt-and-pepper noise)
- Rendered image qualitatively matches observed image at sampled pixels

---

## 5) Long-Term Future Work

### A) Visual Loop Factors (Beyond Dense Map)

**When implemented:**

- Extract keypoint descriptors (ORB/SIFT/SuperPoint) from RGB images
- Produce bearing factors (vMF on viewing directions) for sparse features
- Emit `OpReport` for feature extraction:
  - `approximation_triggers: {"FeatureDetection"}` if detector is not rotation-invariant
  - `family_in: "Image"`, `family_out: "vMF"`, `closed_form: False` (feature extraction is not closed-form)
- Update soft association to include visual descriptor likelihood (e.g., Hamming distance → Dirichlet responsibilities)

**Integration:**

- Keypoints detected in `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/loop_processor.py` (or new `keypoint_detector.py`)
- Bearing factors published to `/sim/bearing_factor` (new message type)
- Backend fuses bearing factors as vMF evidence on anchor directions

**Self-Adaptive Systems Integration**: See `docs/Self-Adaptive Systems Guide.md` Section 4 (Adaptive Loop Closure Confidence) for adaptive loop gating:
- Hellinger-based soft gating (no hard χ² thresholds)
- Adaptive λ threshold based on false positive rate tracking
- Trajectory consistency checking
- Certificate-based quality metrics for downstream scaling (no accept/reject branching)

---

### B) Additional Sensors

**GNSS (if available in dataset):**

- Add as 3DOF position factor with horizontal/vertical uncertainty
- Construct information form with anisotropic covariance:
  - `sigma_horizontal = 2-5m` (typical GPS)
  - `sigma_vertical = 5-10m` (GPS vertical is weak)
- Emit `OpReport`: `family_in: "Gaussian"`, `family_out: "Gaussian"`, `closed_form: True` (exact information fusion)

**Semantic observations (if available):**

- Add as Dirichlet factors on object categories
- Use soft responsibilities over semantic classes (no argmax)
- Emit `OpReport`: `family_in: "Categorical"`, `family_out: "Dirichlet"`, `closed_form: True` (conjugate update)

---

### C) Backend Optimizations

**Hierarchical recomposition for large loop scopes:**

- Use associativity of Bregman barycenters (exact, not an approximation)
- Recursively fuse loop evidence in clusters before global recomposition
- Emit `OpReport`: `approximation_triggers: {}` (exact), `closed_form: True`

**Budgeted recomposition for real-time constraints:**

- Declared approximation operator: `BudgetedRecomposition(e_loop, budget B)`
- Objective: maximize Bregman divergence reduction or mutual information (closed-form for Gaussian/Dirichlet)
- Emit `OpReport`:
  - `approximation_triggers: {"BudgetedRecomposition"}`
  - `frobenius_applied: True` (third-order correction to transport error)
  - Log: loop factor id, selected scope, objective value, diagnostics (predicted vs realized posterior change)

**Self-Adaptive Systems Integration**: See `docs/Self-Adaptive Systems Guide.md` Section 5 (Adaptive Frobenius Correction Strength):
- Adaptive β correction strength based on observed linearization error
- Certificate-based quality metrics (correction magnitude, expected error reduction)
- No hard gates: continuous scaling based on certificate quality

---

### D) Dirichlet Semantic SLAM Integration

**Files:**
- Semantic/Dirichlet nodes and common utilities to be added under `fl_ws/src/fl_slam_poc/` if semantic layer is implemented

**Self-Adaptive Systems Integration**: See `docs/Self-Adaptive Systems Guide.md` Section 3 (Adaptive Association via Dirichlet Concentration Tracking):
- Adaptive concentration regulation based on system-wide entropy
- Entropy-based confidence tracking (no hard gating)
- Hellinger-based entity similarity for robust association
- Temporal decay for dynamic scenes

---

### E) Gazebo live testing

**Files:** To be added under `fl_ws/.../launch/` and frontend when needed.

---

### F) Visualization

**RViz:** `config/fl_slam_rviz.rviz` (local/optional)
**Rerun bridge:** removed from MVP; revisit later if needed

### G) Self-Adaptive System Integration

**Reference**: `docs/Self-Adaptive Systems Guide.md`

**System-Wide Adaptive Coordinator** (Section 6):
- Health monitoring across all subsystems (IMU, odom, loops, association)
- Cross-system adaptation via expected-utility maximization (myopic, single-step)
- Adaptation budget: maximize benefit (divergence reduction) subject to frame budget
- Graceful degradation: when one subsystem degrades, others compensate via continuous scaling

**Monge-Ampère Transport for Dynamic Maps** (Section 7):
- Optimal transport for dynamic scene handling (moving objects, scene changes)
- Hellinger trigger for transport initiation
- Scene flow adapter for filtering dynamic points in SLAM
- Equivariant Wishart transport for covariance adaptation

**Integration Points:**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/adaptive_coordinator.py` (to be created)
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/adaptive_transport.py` (to be created)
- Diagnostics publisher: `/cdwm/adaptive_diagnostics` topic

**Design Invariants Compliance:**
- No hard gates: all adaptations via continuous scaling, not accept/reject branching
- Startup is not a mode: behavior emerges from priors, not time-based logic
- Expected vs realized benefit: internal objectives only (divergence reduction, ELBO increase), not external metrics (ATE/RPE)
- Constants surfaced as priors: effective sample size, hazard rates, certificate risk levels, frame budgets

---

## Roadmap Summary

### Current + Immediate (Priority 1)
1. ✅ 22D state + 14-step pipeline (IW, MF+planar LiDAR, odom+twist, IMU, planar priors) - **COMPLETED**
2. ⏳ Design gaps: pose–twist joint; cross-sensor consistency; IMU cov / LiDAR intensity; Q trans z; rotation/tilt audit
3. ⏳ Optional: bin-level motion smear (PIPELINE_DESIGN_GAPS §5.8)

### Near-Term Future Work (Priority 2) — Phases 6-8
7. ⏳ Phase 6: Dense RGB-D in 3D mode
8. ⏳ Phase 7: Provenance + evaluation hardening
9. ⏳ Phase 8: Camera-frame Gaussian splat map with vMF shading (K=512, J=2)

### Medium-Term (Priority 3) — Phases 9-11
10. ⏳ Phase 9: Alternative datasets (TurtleBot3, NVIDIA r2b)
11. ⏳ Phase 10: GPU acceleration
12. ⏳ Phase 11: Gazebo live testing

### Long-Term (Priority 4) — Phases 12-17
13. ⏳ Phase 12: Visual loop closures (bearing-based) with adaptive gating
14. ⏳ Phase 13: GNSS integration with adaptive sensor weighting
15. ⏳ Phase 14: Semantic observations (Dirichlet)
16. ⏳ Phase 15: Backend optimizations (hierarchical/budgeted recomposition)
17. ⏳ Phase 16: Monge-Ampère transport for dynamic maps
18. ⏳ Phase 17: System-wide adaptive coordinator

---
## Future: Semantic / Categorical Layer (Reference)


Adding **categorical (and hierarchical categorical)** layers is a principled way to reach Hydra-DSG–level semantic usability while staying within exponential-family, additive stats, branch-free doctrine. Below: strict, implementable outline (first principles, JAX/static-shape friendly, auditable).

---

## 1) The right semantic primitive: Categorical with conjugate Dirichlet

For each map atom (j) (splat / voxel / bin-cell), maintain a categorical distribution over (K) labels:

[
\pi_j \in \Delta^{K-1},\qquad y \sim \mathrm{Cat}(\pi_j)
]

Use a Dirichlet prior/posterior:
[
\pi_j \sim \mathrm{Dir}(\alpha_j), \quad \alpha_j \in \mathbb R_{>0}^K
]

### Why this matches your architecture

* **Exponential family**: Cat/Dir are canonical.
* **Additive sufficient statistics**: Dirichlet posterior is updated by adding counts (or soft counts).
* **No gates**: every semantic observation contributes soft mass; uncertainty is in (\alpha), not thresholds.
* **Auditability**: (\alpha) is literally “evidence mass per class.”

---

## 2) The crucial move: semantics update uses *soft assignments* and *continuous reliability*

You must never do “take argmax class” or “only accept if confidence > 0.7”.

For a semantic observation at time (t), you should always produce a **soft responsibility vector**:
[
r_t \in \Delta^{K-1}
]
(e.g., from a perception model, a foundation model, or a learned classifier).

Then update the atom’s Dirichlet counts additively:
[
\alpha_j \leftarrow \alpha_j + w_{t\to j}, r_t
]

Where (w_{t\to j}) is a **continuous** association weight (see §3).

This is exactly analogous to how you already treat LiDAR bins and vMF resultants—just in the categorical simplex instead of the sphere.

---

## 3) How semantic observations attach to map atoms (no gating)

You already have the machinery: soft assignment.

Define a fixed-size set of map atoms (splats/voxels/bins). For each observation, compute **continuous association weights**:

* **Geometric association**: based on distance in (\mathbb R^3) under the atom covariance (\Sigma_j)
* **Directional consistency**: based on vMF alignment (\mu_j^\top x) and (\kappa_j)
* **Optional appearance**: intensity/tag/other metadata as additional likelihood terms

Then:
[
w_{t\to j} \propto \exp\Big(-\tfrac12 d_M(x_t,\mu_j,\Sigma_j)\Big)
]
normalized with a softmax temperature that you can also infer (same pattern as `tau_soft_assign` but now principled across modalities).

No “nearest neighbor” decisions. No “if distance < threshold”.

---

## 4) Hierarchical categorical for expressiveness (Hydra-comparable)

A flat categorical is useful but limited. The principled next step is a **hierarchical Dirichlet model** so you get:

* global priors,
* region priors (room/place),
* object priors,
* and local atom posteriors,

all with conjugate updates.

### 4.1 Three-level hierarchy (minimal and powerful)

Let:

* Global label frequency: (\pi_0 \sim \mathrm{Dir}(\alpha_0))
* Region (r): (\pi_r \sim \mathrm{Dir}(\lambda,\pi_0))
* Atom (j) in region (r(j)): (\pi_j \sim \mathrm{Dir}(\gamma,\pi_{r(j)}))

Observations update only (\alpha_j) (local counts). Periodically (or per scan) you do fixed-cost VB updates for (\pi_r, \pi_0) from aggregated counts. This is branch-free and static-shape if you fix the number of regions or treat regions as a fixed atlas (see §6).

### 4.2 Why this matters

* In Hydra DSG terms: regions/places become semantic containers.
* Your system remains fully probabilistic and continuous.
* Semantics become *smoothly transferrable*: if a room is “kitchen-like,” atoms inside inherit that prior.

---

## 5) Semantics + geometry coupling without breaking your purity

You have two correct options:

### Option A (recommended initially): “Semantics is a read-only layer”

* Semantics does **not** influence pose optimization.
* You only visualize and audit it.
* This avoids feedback loops and preserves estimator stability.

### Option B (later): semantics contributes as a factor through learned likelihoods

If you have a semantic sensor model (p(r_t \mid \pi_j)), you can include it as an additive negative log likelihood and project via Laplace along with the rest. Still no heuristics.

Most teams get Option A correct first, then move to B.

---

## 6) JAX/static-shape implementation strategy

To keep `jit` happy, you cannot create/destroy objects dynamically in the inner loop.

So you define:

* a fixed map atlas (e.g., hashed voxel table of fixed capacity, or fixed splat budget),
* each slot contains:

  * Gaussian params (mean/cov or info)
  * vMF stats (S,N → μ,κ)
  * Dirichlet α vector (`float32[K]`)

Updates are pure adds + renormalizations.

### Memory footprint

If (K=40) classes and you store `alpha[K]` per atom:

* 40 floats = 160 bytes/atom
* 100k atoms = 16 MB (very manageable)

---

## 7) Visualization: “semantic splats” that are auditable

Once each Gaussian splat atom has:

* geometric (\mu,\Sigma)
* directional vMF (\mu_d,\kappa)
* semantic Dirichlet (\alpha)

you can render:

* **Most likely class**: (\arg\max_k \mathbb E[\pi_k]=\alpha_k/\sum\alpha)
* **Semantic uncertainty**: entropy of (\mathbb E[\pi])
* **Evidence mass**: (\sum\alpha) (how much semantic evidence is there)
* **Geometry uncertainty**: (\mathrm{tr}(\Sigma)) or (\log\det\Sigma)

This yields a map humans can *audit*:

* “this region is labeled ‘chair’ but entropy is high”
* “that wall has high geometric certainty but semantic ambiguity”
* “this object hypothesis is strong because alpha mass is high”

That is Hydra-like usability, but with far better uncertainty grounding.

---

## 8) What I recommend as the next concrete step

Define a **MapAtom v2 schema**:

* `mu_xyz: float32[3]`
* `Lambda_xyz: float32[3,3]` (or Sigma)
* `S_dir: float32[3]`, `N_dir: float32`
* `alpha_sem: float32[K]`

Then define two operators:

1. `SemanticSoftAssign(observation → r_t, association weights w_{t→j})`
2. `DirichletAccumulate(alpha_j += w*r_t)` (pure add)


**End of Roadmap**
