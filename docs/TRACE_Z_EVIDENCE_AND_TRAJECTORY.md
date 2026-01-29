# Trace: Where Does Z in the Pose and Trajectory Come From? (Legacy analysis + current fixes)

We observed **large z movement** in the estimated trajectory in a **legacy pipeline** run (pre‑planarization). The robot is planar (x, y, yaw); odom explicitly does not trust z (covariance 1e6 m²). This doc traces the **legacy mechanism** and notes the **current fixes** now in code.

**Current status (in code):**
- **Planar translation evidence** with self‑adaptive z precision (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/matrix_fisher_evidence.py:502`)
- **Always‑on planar priors** on z and v_z (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:859`)
- **Map update planar z** (forces `t_hat[2]=0`) (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/map_update.py:104`)

**References:** `docs/PIPELINE_MESSAGE_TRACE_MESSAGE_5.md`, `docs/RAW_MEASUREMENTS_VS_PIPELINE.md`, `tools/inspect_odom_covariance.py`, odom/IMU/LiDAR operators.

---

## 1. Raw data: what actually constrains z?

### 1.1 Odom (nav_msgs/Odometry)

- **Pose:** We use position (x, y, z) and quat. From the bag, **pose covariance** for the first 354 messages is **diagonal** with:
  - **cov[0,0] = 0.001** m² (x)
  - **cov[1,1] = 0.001** m² (y)
  - **cov[2,2] = 1e6** m² (**z**)
  - cov[3:6, 3:6] = 1e6, 1e6, 1000 rad² (roll, pitch, yaw)

So in **odom_evidence** we use:

- **L_pose = inv(cov)** → **L_pose[2,2] = 1 / 1e6 = 1e-6** (1/m²).
- Residual: **delta_pose_z[2] = odom_z - pred_z** (m).
- Contribution to information vector: **h[2] += L_pose[2,2] * delta_pose_z[2] = 1e-6 * (odom_z - pred_z)**.

So **odom gives very weak z evidence**: we pull z toward odom_z with strength **1e-6**. For example, if odom_z = -0.028 m (relative) and pred_z = 0, we add a tiny pull toward -0.028 m. Odom alone cannot explain “a ton of z movement” unless we integrate over an enormous number of scans (and even then the information stays 1e-6 per scan).

**Conclusion:** Odom z is used but with **very low weight** (1e-6). It is not the main source of large z.

---

## 2. Legacy LiDAR translation evidence: full 3D, no z-downweighting

### 2.1 Legacy TranslationWLS (translation estimate)

- **Model:** `c_map_b = R_hat @ p_bar_scan_b + t + noise` (per bin). We solve for **t_hat** (3,) and **t_cov** (3×3) in **world frame**.
- **t_hat** is the **full 3D** translation (x, y, **z**). In the legacy pipeline there was no separate treatment of z; the same WLS fit gave t_hat[0], t_hat[1], **t_hat[2]**.
- **t_cov** is 3×3. It comes from **A = Σ_b w_b * inv(Sigma_b)** where **Sigma_b = Sigma_c_map + Sigma_scan_rotated + Sigma_meas**. **Sigma_meas** is **isotropic**: `GC_LIDAR_SIGMA_MEAS = 0.01` m² (same for x, y, z). So we do **not** inflate z uncertainty in the WLS.
- Geometry: If bins are distributed in 3D (e.g. forward, left, right, some height spread), **A** is well-conditioned in all three dimensions and **t_cov** has similar magnitude in (2,2) as in (0,0) and (1,1). So **t_hat[2]** is estimated with similar precision as t_hat[0] and t_hat[1].

### 2.2 LiDAR quadratic evidence (lidar_evidence.py)

- **Translation block:** We build **H_trans = inv(t_cov)** (after PSD and rotation to body). So **H_trans[2,2]** is the information for **z** (1/m²).
- **L_lidar[0:3, 0:3] = info_scale * H_trans.** So we add **full 3D** translation evidence, including **z**, with strength set by **t_cov[2,2]**.
- Legacy pipeline: **no** z‑downweighting, no planar constraint, and no “z unobserved” prior. So **LiDAR was the only source of strong z evidence**.

### 2.3 Where does t_hat[2] (LiDAR z) get its value?

- **t_hat** is the translation that aligns **R_hat @ p_bar_scan** to **c_map** (map centroids). So **t_hat = (c_map - R_hat @ p_bar_scan)** in the WLS sense.
- **c_map** and **p_bar_scan** are **3D** centroids (x, y, z). They come from:
  - **Scan:** Points are in **base frame** after **T_base_lidar** (translation **[−0.011, 0, 0.778]** m). So scan centroids have **z ≈ 0.778 m** (lidar height) for typical forward-looking bins.
  - **Map:** Built by transforming previous scan(s) into **world frame** using **belief pose**. So **map centroids have z = belief_z + (R @ p_scan)_z**. If **belief_z** is non-zero (from previous evidence), the map has a z offset. Next scan aligns to that map → **t_hat[2]** will reflect that offset (body z in world frame).

So:

1. **First scan:** Map is built with belief at origin (0,0,0). Map centroids have z ≈ 0.78 m (lidar height). **t_hat** is the body position in world; if we’re at origin, **t_hat ≈ (0, 0, 0)** and z stays ~0.
2. **Later scans (legacy):** If at any point **belief_z** becomes non-zero (e.g. from odom’s weak pull, or from numerical/geometry effects in t_hat), the **map** is updated with that z. Then **c_map** has that z offset. The next TranslationWLS fit gives **t_hat[2]** consistent with that map. We then feed **t_hat** (including **t_hat[2]**) into LiDAR evidence with **full weight** (inv(t_cov)). So we **reinforce** z every scan: **belief_z → map z → t_hat[2] → LiDAR evidence → belief_z**.  
**Current pipeline:** map update forces `t_hat[2]=0` and translation z precision is downweighted, so this loop is broken.
3. **Centroid mismatch:** Any systematic z offset between scan and map (e.g. different lidar height, calibration, or drift) produces a non-zero **t_hat[2]** and thus a z pull with **strong** weight (inverse of t_cov[2,2], which is not large).

**Legacy conclusion:** **LiDAR translation evidence was the dominant source of z.** We treated z like x and y (full 3D, isotropic Sigma_meas, no planar prior). Once z appeared in the map (from belief), it was reinforced by the next scan’s t_hat[2] and strong L_lidar[2,2].

---

## 3. Process model and velocity: z is not suppressed

### 3.1 Process noise Q

- **State blocks:** [trans(3), rot(3), vel(3), bg(3), ba(3), dt(1), ex(6)].
- **Trans block:** One scalar **GC_PROCESS_TRANS_DIFFUSION = 1e-4** m²/s is used for **all three** diagonal entries of the trans block. So **Q[0,0] = Q[1,1] = Q[2,2]** (in the trans block). We do **not** use a smaller diffusion for z.
- **Prediction:** OU propagation adds **Q * (1 - exp(-2λ dt)) / (2λ)** to the covariance. So **z uncertainty** grows at the **same rate** as x and y. We never “pin” z or give it a stronger prior.

### 3.2 Velocity and preintegration

- **Velocity** is 3D in the state (indices 6:9). **IMU preintegration** gives **delta_p_int** and **delta_v_int** (3D). The preintegration factor compares these to predicted position/velocity change. There is **no** constraint that velocity z or delta_v z be zero. So any small z from IMU (bias, tilt, numerical) can enter the state and then be propagated by the process model (position += velocity * dt).

**Conclusion:** **Process and velocity treat z like x and y.** So once z (or velocity_z) is in the state from LiDAR (or odom/IMU), it can **grow** over time and is not damped by a planar prior.

---

## 4. End-to-end trace: initial values → final z

| Step | What happens to z | Units / values |
|------|-------------------|----------------|
| **Odom** | We use odom_pose[2] (tz) and cov[2,2] = 1e6 m². L_pose[2,2] = 1e-6 1/m². Residual = odom_z − pred_z (m). | Very weak pull toward odom_z (e.g. −0.028 m relative). |
| **LiDAR (legacy)** | TranslationWLS returns t_hat (3D), t_cov (3×3). Sigma_meas isotropic 0.01 m². H_trans = inv(t_cov) → full 3D. L_lidar[0:3,0:3] = info_scale * H_trans. | **Strong** z evidence from t_hat[2] and t_cov[2,2]. No z-downweighting. |
| **Map update** | Map centroids = f(belief pose, scan). If belief_z ≠ 0, map has z offset. | belief_z → map z → next t_hat[2] → reinforces z. |
| **Process** | Q trans block = diag(1e-4, 1e-4, 1e-4) m²/s. Prediction adds same diffusion to x, y, z. | z uncertainty grows like x, y; no planar damping. |
| **Velocity** | State has vel (3D). IMU preintegration gives delta_v (3D). No “vel_z = 0” constraint. | Any vel_z or delta_v_z gets fused and propagated. |
| **Fusion** | L_evidence = L_odom + L_imu + L_gyro + L_preint + L_lidar. Pose block [0:6,0:6] includes [2,2] from odom (1e-6) and from LiDAR (large). | **L_pose[2,2] is dominated by LiDAR** (inv(t_cov[2,2])), not by odom (1e-6). |

So:

1. **Initial:** Belief at origin, map built with z ≈ 0.78 m (lidar height). t_hat ≈ (0,0,0) if alignment is good.
2. **First z error:** Can come from (a) a small t_hat[2] from centroid/geometry/calibration, or (b) odom’s weak pull (1e-6 * odom_z). (a) is weighted by **strong** LiDAR info; (b) is tiny.
3. **Feedback:** belief_z ≠ 0 → map gets z → next t_hat[2] matches that → LiDAR evidence pulls z again with **full** weight.
4. **Growth:** Process noise and velocity propagation allow z to drift like x and y; there is no planar constraint.

**Legacy summary:** **Z in the pose matrix and trajectory was driven primarily by LiDAR translation evidence (full 3D t_hat/t_cov with no z‑downweighting) and then reinforced by map–scan feedback and process/velocity. Odom contributed only a very small z term (1e-6).**  
**Current status:** Planar translation + planar priors + planar map update now prevent the z feedback loop; z should stay bounded near z_ref.
