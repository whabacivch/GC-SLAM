# Pipeline Reference: From Rosbag to State

This document is the **pipeline reference** for Golden Child SLAM v2. It describes how raw topics from the rosbag flow through the system end-to-end, how each sensor is handled (frontend and backend), the fixed-cost scan pipeline (steps 1–14), evidence types (Gaussians and vMF), fusion, and how beliefs, bins, and the map interact.

**Use this doc when you need to:** trace a topic from bag to output, see which pipeline step uses which data, or understand how IMU/odom/LiDAR evidence is combined.

---

## 1. From Rosbag to Output: End-to-End Data Flow

### 1.1 Raw Topics (Rosbag)

For the primary evaluation setup (e.g. M3DGR Dynamic01), the bag provides:

| Raw topic | Message type | Role |
|-----------|--------------|------|
| `/livox/mid360/lidar` | `livox_ros_driver2/msg/CustomMsg` | LiDAR point clouds (Livox proprietary). |
| `/livox/mid360/imu` | `sensor_msgs/msg/Imu` | 6-axis IMU (gyro + accel) at ~200 Hz, frame `livox_frame`. |
| `/odom` | `nav_msgs/msg/Odometry` | Wheel odometry pose + covariance, ~20 Hz, frames `odom_combined` → `base_footprint`. |

**Note:** `/tf` and `/tf_static` are **absent** in the bag. Extrinsics are applied numerically via parameters (`T_base_lidar`, `T_base_imu`), not TF.

### 1.2 Frontend: gc_sensor_hub (Single Process)

The **gc_sensor_hub** runs in one process with a MultiThreadedExecutor. It subscribes to **raw** topics and publishes to **canonical** topics that the backend consumes.

| Raw topic | Frontend node | Canonical topic | What the node does |
|-----------|---------------|-----------------|---------------------|
| `/livox/mid360/lidar` | **livox_converter** | `/gc/sensors/lidar_points` | Converts CustomMsg → `sensor_msgs/PointCloud2`. Schema: x,y,z (float32), intensity, ring, tag, time_offset (if present), timebase. No frame or unit change. |
| `/odom` | **odom_normalizer** | `/gc/sensors/odom` | Pass-through with optional frame id override. Validates covariance present; no numeric transform. |
| `/livox/mid360/imu` | **imu_normalizer** | `/gc/sensors/imu` | Pass-through with finite-value check and optional output_frame. No g→m/s² or extrinsics. |

**Dead-end audit** (same process): subscribes to unused bag topics and publishes `/gc/dead_end_status` for wiring validation.

### 1.3 Backend: Subscriptions and Triggers

The **gc_backend_node** subscribes **only** to canonical topics:

| Canonical topic | Backend callback | Effect |
|-----------------|------------------|--------|
| `/gc/sensors/lidar_points` | `on_lidar()` | **Triggers the pipeline.** Parses PointCloud2, applies `T_base_lidar` to points, pulls latest odom and IMU buffer, runs `process_scan_single_hypothesis` (or multi-hypothesis), updates map and state, publishes outputs. |
| `/gc/sensors/imu` | `on_imu()` | Applies accel scale (g→m/s²) and `R_base_imu` to gyro/accel, appends `(stamp, gyro, accel)` to `imu_buffer`. No pipeline run. |
| `/gc/sensors/odom` | `on_odom()` | Converts pose to SE(3), stores first pose as reference, keeps `last_odom_pose` = first_odom^{-1} ∘ odom_absolute, and `last_odom_cov_se3`. No pipeline run. |

So: **LiDAR drives the pipeline**; IMU and odom are consumed when each scan is processed.

### 1.4 Backend: Outputs

After each scan (and hypothesis combination), the backend:

| Output | Type | Description |
|--------|------|-------------|
| `/gc/state` | `nav_msgs/Odometry` | Current pose estimate (posterior mean) and 6×6 covariance. |
| `/gc/trajectory` | `nav_msgs/Path` | Path of poses (for visualization). |
| `/gc/runtime_manifest` | `std_msgs/String` (JSON) | Published once at startup: resolved topics, extrinsics, scales, etc. |
| `/gc/certificate` | `std_msgs/String` (JSON) | Per-scan or aggregated certificates (approximation triggers, conditioning). |
| `/gc/status` | `std_msgs/String` (JSON) | Periodic status (scan count, odom/IMU counts, etc.). |
| Trajectory file | TUM format | Written to `trajectory_export_path` if set (e.g. for eval). |
| Diagnostics | NPZ | Per-scan diagnostics (L/h blocks, alpha, poses, etc.) to `diagnostics_export_path` for the dashboard. |

---

## 2. Pipeline Step Reference (Per Scan)

Each **LiDAR** message triggers one pipeline run. The pipeline is **fixed-cost** and **branch-free**: all 14 steps run every time. Steps are executed in order; later steps use outputs of earlier ones.

| Step | Name | Main inputs | Main outputs |
|------|------|-------------|--------------|
| 1 | **PointBudgetResample** | raw_points, timestamps, weights, ring, tag | points, timestamps, weights (capped N_POINTS_CAP). |
| 2 | **PredictDiffusion** | belief_prev, Q, dt_sec | belief_pred. |
| 3 | **DeskewConstantTwist** | points, timestamps, **IMU** (w_imu_scan, preintegration → xi_body) | deskewed_points, deskewed_weights. |
| 4 | **BinSoftAssign** | point_directions (from deskewed_points, lidar_origin_base), bin_atlas | responsibilities. |
| 5 | **ScanBinMomentMatch** | deskewed_points, weights, responsibilities, point_lambda | scan_bins (s_dir, N, kappa_scan, p_bar, Sigma_p, etc.). |
| 6 | **KappaFromResultant** | map_stats, scan_bins | mu_map, kappa_map, c_map, Sigma_c_map; mu_scan. |
| 7 | **WahbaSVD** | mu_map, mu_scan, wahba_weights (N×kappa_map×kappa_scan) | R_hat. |
| 8 | **TranslationWLS** | c_map, Sigma_c_map, scan_bins, R_hat, Sigma_meas | t_hat, t_cov. |
| 9 | **Odom + IMU + LiDAR evidence** | belief_pred, **odom_pose**, **imu** (gyro preint, accel vMF, preint factor), scan_bins, map_bins, R_hat, t_hat | L_evidence, h_evidence (sum of L_odom, L_imu, L_gyro, L_imu_preint, L_lidar; same for h). |
| *(after 9)* | **Excitation scaling** | L_evidence, belief_pred.L | belief_pred with scaled prior (dt index 15, extrinsic 16:22). |
| 10 | **FusionScaleFromCertificates** | L_evidence (pose 6×6 conditioning), certs | α. |
| 11 | **InfoFusionAdditive** | belief_pred, L_evidence, h_evidence, α | belief_post. |
| 12 | **PoseUpdateFrobeniusRecompose** | belief_post | belief_recomposed (pose applied to anchor). |
| 13 | **PoseCovInflationPushforward** | belief_recomposed, scan_bins, R_hat, t_hat | map_increments (delta_S_dir, etc.). |
| 14 | **AnchorDriftUpdate** | belief_recomposed | belief_final. |

**Where each topic feeds in:**

- **LiDAR:** Steps 1–2 (raw points); 3 (deskew uses points + IMU); 4–8 (bins, Wahba, WLS); 9 (lidar_quadratic_evidence); 14 (map update uses scan_bins).
- **IMU:** Step 3 (deskew twist); step 9 (gyro evidence, accel vMF evidence, preintegration factor); also IW updates for Σg, Σa after the pipeline.
- **Odom:** Step 9 (odom_quadratic_evidence); first-odom reference applied in backend before pipeline.

---

## 3. Sensor Breakdown: What Each Provides

### 3.1 IMU (Inertial Measurement Unit)

The **IMU** is a single package containing two sensors:

- **Gyroscope**: measures angular velocity (rad/s).
- **Accelerometer**: measures specific force (reaction to gravity + linear acceleration; units g, converted to m/s²).

In this project the IMU is the Livox Mid-360 onboard IMU (ICM-40609), outputting 6-axis data at ~200 Hz.

### 3.2 Gyroscope (inside IMU)

| Aspect | Detail |
|--------|--------|
| **Measures** | Angular velocity ω (rad/s), 3-axis. |
| **Provides** | Relative rotation change; integrate to get Δθ over an interval. |
| **Does not provide** | Absolute orientation (no world reference). |
| **In code** | `imu_gyro_rotation_evidence()`: Gaussian evidence on rotation block from gyro preintegration vs predicted pose. |

### 3.3 Accelerometer (inside IMU)

| Aspect | Detail |
|--------|--------|
| **Measures** | Specific force (m/s² after scale): gravity + linear acceleration. |
| **Provides** | Gravity direction when stationary (→ pitch/roll); linear acceleration when moving. |
| **Does not provide** | Position, velocity, or **yaw** (current implementation uses only mean direction → gravity alignment). |
| **In code** | `imu_vmf_gravity_evidence()`: vMF-style factor on **direction only**; converted to Gaussian information (rotation block [3:6]) via Laplace at δθ=0. |

**Critical limitation:** The accelerometer evidence uses a **single vMF on the mean direction** of all accel samples. That measures “how much the mean direction deviates from vertical,” which constrains **pitch/roll only**. It does **not** constrain yaw; centrifugal force is not exploited for yaw in the current design.

### 3.4 Odometry (wheel encoder)

| Aspect | Detail |
|--------|--------|
| **Measures** | Pose estimate from wheel encoders (position + orientation). |
| **Provides** | Full 6D pose (x, y, z, roll, pitch, yaw) with covariance; typically 2D-strong (x, y, yaw). |
| **In code** | `odom_quadratic_evidence()`: Gaussian evidence on full 6D pose; covariance from message (ROS order [x,y,z,roll,pitch,yaw] = GC [trans, rot]). |

### 3.5 LiDAR

| Aspect | Detail |
|--------|--------|
| **Provides** | Scan-to-map alignment (Wahba + translation WLS) → rotation R_hat, translation t_hat, and covariance. |
| **In code** | `lidar_quadratic_evidence()`: Quadratic (Gaussian) evidence on full 22D state at belief_pred; blocks for pose [0:6], plus coupling for dt (index 15) and extrinsic (16:22). |

### 3.6 Summary: What Constrains What

| Sensor | Constrains | Does not constrain |
|--------|------------|--------------------|
| **Gyro** | Relative rotation (all 3 axes) | Absolute orientation |
| **Accel** | Pitch/roll (gravity direction) | Yaw, position, velocity |
| **Odom** | Position + orientation (all 6) | Global consistency (drift) |
| **LiDAR** | Position + orientation (scan-to-map) | Fast motion without other sensors |

Yaw is under-constrained (gyro gives only relative change; accel does not use horizontal/centrifugal component), which contributes to large yaw errors (e.g. 100–130°) when gyro/odom are misaligned or weak.

---

## 4. Frontend and Backend: Topic-by-Topic Handling

### 4.1 LiDAR

- **Frontend (livox_converter):** CustomMsg → PointCloud2; fields x, y, z, intensity, ring, tag, time_offset (if present), timebase. No frame or unit change.
- **Backend (on_lidar):** Parse PointCloud2 → points, timestamps, weights, ring, tag. Apply **T_base_lidar** (parameter [x,y,z,rx,ry,rz]): **p_base = R_base_lidar @ p_lidar + t_base_lidar**. Scan time = `header.stamp`; scan bounds for deskew from timebase and header. Then call pipeline with points, last odom, and IMU buffer slice.

### 4.2 IMU

- **Frontend (imu_normalizer):** Pass-through; finite check; optional output_frame. No g→m/s² or extrinsics.
- **Backend (on_imu):** Per message: gyro = R_base_imu @ [angular_velocity]; accel_raw → accel = accel_raw * GC_IMU_ACCEL_SCALE (9.81) → accel = R_base_imu @ accel. Append (stamp_sec, gyro, accel) to imu_buffer. At scan time: take last max_imu_buffer samples into fixed-size arrays; integration window (t_last_scan, t_scan); dt_int = sum of IMU sample intervals in that window. Pipeline uses these arrays (base frame, accel in m/s²) for deskew, preintegration, gyro evidence, and accel (vMF) evidence.

### 4.3 Odometry

- **Frontend (odom_normalizer):** Pass-through; optional output_frame/child_frame; validate covariance present.
- **Backend (on_odom):** Convert pose to SE(3) [trans, rotvec]. Store first pose as reference. **last_odom_pose = first_odom^{-1} ∘ odom_absolute**. Store 6×6 pose covariance (ROS order). Pipeline receives last_odom_pose and last_odom_cov_se3 at each scan.

---

## 5. Evidence Types: Gaussians vs vMF

All evidence is converted to **Gaussian information form** (L, h) on the 22D tangent chart so it can be added and fused in one place.

### 5.1 Gaussian Evidence (Gyro, Odom, LiDAR, IMU Preintegration)

- **Gyro** (`imu_gyro_evidence.py`): Residual on SO(3), **r = Log(R_end_pred^T @ R_end_imu)**. Covariance **Σ_rot ≈ Σ_g * dt_int** (Σ_g from IW/datasheet). Information **L_rot = Σ_rot^{-1}**; placed in **[3:6, 3:6]** (rotation block). **Gaussian on rotation**, no vMF.
- **Odom** (`odom_evidence.py`): Pose error **T_err = belief_pred^{-1} ∘ T_odom**, **ξ = Log(T_err)**. **L = Σ^{-1}** from ROS 6×6 pose covariance; placed in **[0:6, 0:6]**. **Gaussian on full 6D pose.**
- **LiDAR** (`lidar_evidence.py`): Wahba + WLS give R_hat, t_hat, t_cov. Quadratic at belief_pred on full 22D with pose block [0:6], plus dt (15) and extrinsic (16:22) coupling. **Gaussian (quadratic) on 22D.**
- **IMU preintegration** (`imu_preintegration_factor.py`): Velocity/position from accel preintegration vs predicted state → quadratic term; **Gaussian** on the relevant tangent block.

### 5.2 vMF Evidence (Accelerometer Only)

**File:** `fl_slam_poc/backend/operators/imu_evidence.py`

- **Model:** Likelihood is **ℓ(δθ) = -κ · μ(δθ)ᵀ · x̄** (vMF-style direction-only factor).
  - **μ(δθ)** = predicted gravity direction in body frame = R(δθ)ᵀ·(-g_hat).
  - **x̄** = mean resultant direction of accel samples (normalize each sample, then average).
- **κ:** From mean resultant length **R̄** via `kappa_from_resultant_v2()` (continuous blend of low-R and high-R approximations; **not** per-axis Gaussian).
- **Conversion to Gaussian:** Laplace at **δθ = 0**:
  - **Gradient:** **g_rot = -κ · (μ₀ × x̄)** (exact).
  - **Hessian (approximation):** **H_rot ≈ κ · [ (x̄·μ₀)·I - 0.5·(x̄μ₀ᵀ + μ₀x̄ᵀ) ]**, then symmetrized and PSD-projected.
- **Placement:** **L_imu** and **h_imu** go into **[3:6, 3:6]** and **[3:6]** only (rotation block). So it is **one** directional (vMF-derived) factor on the 3D rotation perturbation; it effectively measures “tilt away from vertical,” i.e. **pitch/roll**, not yaw.

**Why this doesn’t constrain yaw:** The cross product **μ₀ × x̄** is large for pitch/roll error and near zero for pure yaw (rotation about vertical). So the accel evidence does not pull on yaw; centrifugal force would require a separate non-gravity acceleration term.

---

## 6. Fusion: How Evidence Is Combined

### 6.1 Additive Evidence (Information Form)

All evidence is in **information form** (L, h) on the 22D chart:

- **L_evidence = L_lidar + L_odom + L_imu + L_gyro + L_imu_preint**
- **h_evidence = h_lidar + h_odom + h_imu + h_gyro + h_imu_preint**

So Gaussians and the vMF-derived term are combined by **adding** their L and h. This is equivalent to a product of Gaussians in information form (Bregman/additive fusion). No separate “vMF fusion” step; vMF has already been turned into (L_imu, h_imu).

### 6.2 Excitation Scaling (Prior Scaling)

**File:** `fl_slam_poc/backend/operators/excitation.py`

Before applying evidence, the **prior** (belief_pred) is scaled so that strong evidence can dominate:

- **s_dt = e_dt / (e_dt + pi_dt + ε)** with **e_dt = L_evidence[15,15]**, **pi_dt = L_prior[15,15]** (dt index).
- **s_ex = e_ex / (e_ex + pi_ex + ε)** with **e_ex = trace(L_evidence[16:22, 16:22])**, **pi_ex = trace(L_prior[16:22, 16:22])** (extrinsic block).

Then prior rows/cols for dt (15) and extrinsic (16:22) are scaled by **(1 - s_dt)** and **(1 - s_ex)** (and same for h). So when **L_evidence** (including IMU) is strong, the prior is weakened on those blocks. This affects **all** evidence indirectly by changing the effective prior before fusion.

### 6.3 Fusion Scale α (FusionScaleFromCertificates)

**File:** `fl_ws/.../operators/fusion.py`

- **Conditioning:** Taken from the **6×6 pose block** of **L_evidence** (not full 22×22), so physically relevant pose strength drives α. **cond_pose6 = eig_max_pose / eig_min_pose** (eigenvalues of L_evidence[0:6, 0:6]).
- **Quality:** **cond_quality = c0_cond / (cond_evidence + c0_cond)**; **support_quality = ess_evidence / (ess_evidence + 1)**; **quality = sqrt(cond_quality * support_quality)**.
- **Alpha:** **α = alpha_min + (alpha_max - alpha_min) * quality**, then clamped to [alpha_min, alpha_max].

So IMU (and all other evidence) contributes to the **pose block** of L_evidence; that block’s conditioning and support determine **α**.

### 6.4 InfoFusionAdditive

**L_post = L_pred + α * L_evidence**  
**h_post = h_pred + α * h_evidence**

Then PSD projection is applied to L_post. The result is the **belief_post** used for recompose (Frobenius correction) and then anchor drift to get the final belief.

So: **one** additive fusion step; Gaussians and the vMF-derived term are already summed in L_evidence/h_evidence, then scaled by α and added to the (excitation-scaled) prior.

---

## 7. Belief, Bins, and Map: How IMU Fits In

### 7.1 Belief Flow (Per Scan)

1. **belief_prev** → **predict_diffusion** → **belief_pred**.
2. All evidence (odom, IMU accel, IMU gyro, IMU preint, LiDAR) is computed **at** **belief_pred** (same linearization point).
3. **L_evidence**, **h_evidence** = sum of all L and h (additive evidence).
4. **Excitation scaling:** s_dt, s_ex from L_evidence vs belief_pred.L; prior L, h scaled for dt and extrinsic; **belief_pred** replaced by this scaled version.
5. **Fusion:** α from certificates (pose-6 conditioning + support); **belief_post** = belief_pred + α·(L_evidence, h_evidence); PSD on L_post.
6. **Recompose:** belief_post → Frobenius correction → **belief_recomposed** → anchor drift → **belief_final**.

IMU contributes **L_imu**, **L_gyro**, **L_imu_preint** and their h terms to L_evidence/h_evidence, so it directly shifts the belief in the same way as odom and LiDAR, and it also affects excitation scaling and α (via the pose block).

### 7.2 Bins (Directional Buckets)

- **Bins** are defined by the **bin atlas** (fixed directions). **Scan bins** are computed from the **current scan** only.
- **Deskew** uses IMU: weights **w_imu_scan** over (scan_start_time, scan_end_time), preintegration → **delta_pose_scan** → twist **xi_body** → **deskew_constant_twist** → **deskewed_points**.
- **BinSoftAssign** + **ScanBinMomentMatch** run on deskewed points → **scan_bins** (s_dir, N, kappa_scan, p_bar, Sigma_p, etc.).

So IMU does **not** define the bin grid, but it **does** change the points that fall into bins (via deskew), hence the scan statistics in each bin.

### 7.3 Map (map_stats)

- **map_stats** are per-bin accumulations: S_dir, N_dir, N_pos, sum_p, sum_ppT.
- Updated in the **backend** after the pipeline using **map_increments** from the pipeline.
- **R_for_map**, **t_for_map**:
  - **First scan (map empty):** **belief_recomposed.mean_world_pose()** → pose already includes IMU + odom (no LiDAR alignment yet).
  - **Later scans:** **R_hat**, **t_hat** from Wahba + translation WLS (scan-to-map).
- Scan-side inputs to map update: scan_bins (N, s_dir, p_bar, Sigma_p), which come from **deskewed** points (IMU-dependent).

So IMU influences the map by: (1) first-scan placement using a pose that has already been updated by IMU (and odom); (2) every scan, the statistics pushed into the map come from deskewed points (IMU-dependent).

### 7.4 How IMU Influences Other Evidence

- **Same linearization point:** Odom, LiDAR, and all IMU terms use **belief_pred**. IMU doesn’t change where others are linearized; it adds more (L, h) at that point.
- **Excitation scaling:** L_evidence (including L_imu, L_gyro, L_imu_preint) is used to compute s_dt, s_ex. Strong IMU → stronger total evidence → prior scaled down more → all evidence counts more relative to prior.
- **Fusion α:** Conditioning for α is computed on the **6×6 pose block** of combined evidence (including IMU). So IMU’s contribution to the pose information matrix affects α.
- **LiDAR evidence:** Uses **map_bins** (from map_stats, which depend on previous poses and deskewed scan_bins) and **scan_bins** (deskewed). So IMU influences LiDAR evidence **indirectly** via deskew and map placement/update.
- **Gyro/preint and biases:** Use **pose0** (previous pose) and **gyro_bias**, **accel_bias** from **belief_pred**. So current belief’s bias estimates affect what rotation/velocity the IMU implies → feeds into L_gyro, L_imu_preint.

---

## 8. Summary Table

| Topic | Detail |
|-------|--------|
| **Raw → canonical** | livox_converter (lidar), odom_normalizer (odom), imu_normalizer (imu); backend subscribes only to /gc/sensors/*. |
| **Pipeline trigger** | LiDAR message on /gc/sensors/lidar_points; IMU and odom consumed from buffer/last when each scan runs. |
| **Pipeline steps** | 1–14 run every scan; deskew (3) uses IMU; evidence (9) uses odom + IMU + LiDAR; fusion (10–12) then recompose (13), map update (14), anchor drift (15). |
| **Belief** | IMU adds L_imu, L_gyro, L_imu_preint (and h) to the same information sum as odom and LiDAR; same fusion and recompose. |
| **Bins** | Scan bins come from **deskewed** scan; deskew uses IMU → IMU shapes scan_bins. |
| **Map** | Map updated from scan_bins + pose; first-scan pose = IMU+odom belief; later poses = Wahba/WLS on deskewed scan; scan_bins IMU-dependent. |
| **Gaussians** | Gyro, odom, LiDAR, IMU preint: all Gaussian (or quadratic) in information form on 22D (or pose block). |
| **vMF** | Accel only: single vMF on mean direction → Laplace at δθ=0 → (L_imu, h_imu) on rotation block [3:6]; constrains pitch/roll, not yaw. |
| **Fusion** | L_evidence = sum of all L; h_evidence = sum of all h; excitation scaling scales prior; α from pose-6 conditioning + support; L_post = L_pred + α·L_evidence, h_post = h_pred + α·h_evidence; PSD on L_post. |
| **Outputs** | /gc/state, /gc/trajectory, /gc/runtime_manifest, /gc/certificate, /gc/status; trajectory file (TUM); diagnostics NPZ. |

---

## 9. Document Consistency

The following are **canonical** and should match across docs and code:

- **GC state ordering:** `[trans 0:3, rot 3:6, vel 6:9, gyro_bias 9:12, accel_bias 12:15, dt 15, extrinsic 16:22]`. LiDAR/odom/IMU evidence block indices are defined by this ordering (e.g. rotation = `[3:6]`, not `[0:3]`).
- **Evidence sum:** `L_evidence = L_lidar + L_odom + L_imu + L_gyro + L_imu_preint` (five terms; same for h).
- **Gyro covariance:** `Sigma_rot = Sigma_g * dt_int` (dt_int = sum of IMU sample intervals in window), not dt_scan.
- **Pipeline steps:** 14 numbered steps (1–14); excitation scaling is applied after evidence combine, before FusionScale.
- **Fusion alpha:** Conditioning for α is taken from the **6×6 pose block** of L_evidence (overwritten in pipeline before `fusion_scale_from_certificates`).

See also: `docs/SIGMA_G_AND_FUSION_EXPLAINED.md`, `docs/SYSTEM_DATAFLOW_DIAGRAM.md`.

---

## 10. References

- **Launch and topic flow:** `fl_ws/.../launch/gc_rosbag.launch.py`
- **Bag topics:** `docs/BAG_TOPICS_AND_USAGE.md`
- **Spec:** `docs/GOLDEN_CHILD_INTERFACE_SPEC.md`
- **Frames and gravity:** `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`
- **Sigma_g and fusion setup:** `docs/SIGMA_G_AND_FUSION_EXPLAINED.md`
- **Frontend:** `fl_slam_poc/frontend/sensors/` (livox_converter.py, odom_normalizer.py, imu_normalizer.py); `fl_slam_poc/frontend/hub/gc_sensor_hub.py`
- **Backend:** `fl_slam_poc/backend/backend_node.py` (on_lidar, on_imu, on_odom)
- **Operators:** `fl_slam_poc/backend/operators/` (imu_evidence.py, imu_gyro_evidence.py, odom_evidence.py, lidar_evidence.py, fusion.py, excitation.py)
- **Pipeline:** `fl_slam_poc/backend/pipeline.py` (`process_scan_single_hypothesis`)

---

*Section 9 (Document Consistency) was added to keep this doc aligned with SIGMA_G_AND_FUSION_EXPLAINED.md and SYSTEM_DATAFLOW_DIAGRAM.md; discrepancies were corrected in those docs to match code and this pipeline reference.*
