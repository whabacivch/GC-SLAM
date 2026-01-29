# Pipeline Trace: One Document — Value as Object, Mechanism and Causality

One document. We treat each **value as an object** and follow it through the pipeline. Like putting a radioactive signature on a physical thing: we see where the raw value goes at every step and how it **contaminates** (contributes to) downstream outputs. The process is **deterministic** — just math in sequence. No branching on data; one path per operator.

**Trigger:** One LiDAR scan runs `process_scan_single_hypothesis`. It uses the **latest** odom and the **last M** buffered IMU samples. We trace: (1) the pipeline’s fixed step order (the spine), (2) each object (IMU message 5, Odom message 5, one LiDAR point) through that spine to final outputs, (3) **preintegration** (all steps P1–P8 plus body-frame outputs), (4) **belief and 6D pose Hessian** (22D information form, fusion, recompose), and (5) **adaptive noise** (Inverse-Wishart: where Q and Sigma_* come from and when they are updated).

All units are stated. Concrete numbers use **raw message 5** from the IMU and odom CSVs; LiDAR uses one representative point and the same config.

---

# Part 1: The pipeline spine (deterministic step order)

When `on_lidar` runs, it executes this sequence. Every scan runs the same steps in the same order.

| Step | What happens | Code / location |
|------|--------------|------------------|
| **L1** | Read `msg.header.stamp` → `t_scan`. | `on_lidar`: stamp_sec |
| **L2** | Parse PointCloud2: x, y, z, timebase, time_offset, ring, tag; per-point timestamps; range-based weights. Non-finite x,y,z → sentinel. | `parse_pointcloud2_vectorized()` |
| **L3** | Transform points to base: `p_base = R_base_lidar @ p_lidar + t_base_lidar`. | `on_lidar`: pts_base, points |
| **L4** | Scan bounds and `dt_sec = t_scan - t_prev_scan`. | `on_lidar`: scan_start_time, scan_end_time, dt_sec |
| **L5** | Copy last M IMU samples from buffer → `imu_stamps`, `imu_gyro`, `imu_accel`. | `on_lidar`: tail → imu_stamps_j, imu_gyro_j, imu_accel_j |
| **L6** | Call `process_scan_single_hypothesis(...)` with raw_points, raw_timestamps, raw_weights, odom_pose, odom_cov_se3, imu_*, scan times, dt_sec, Q, bin_atlas, map_stats, config. | `on_lidar` |
| **L7** | **Step 1 – PointBudgetResample:** Cap points; resample (ring, tag preserved). | pipeline: point_budget_resample() |
| **L8** | **Step 2 – PredictDiffusion:** belief_pred from previous belief, Q, dt_sec. | pipeline: predict_diffusion() |
| **L9** | **Step 3 – DeskewConstantTwist:** IMU preintegration over scan window → delta_pose_scan, xi_body; deskew points. deskewed_covs = zeros. | pipeline: preintegrate_imu_relative_pose_jax(), deskew_constant_twist() |
| **L10** | **Step 4 – BinSoftAssign:** Point directions → soft assignments to bin atlas. | pipeline: bin_soft_assign() |
| **L11** | **Step 5 – ScanBinMomentMatch:** Weighted moment match per bin (responsibilities, weights, point_covariances=zeros). | pipeline: scan_bin_moment_match() |
| **L12** | **Step 6 – KappaFromResultant:** Inside ScanBinMomentMatch. | (same) |
| **L13** | **Step 7 – MatrixFisherRotation:** Scan-vs-map rotation → R_hat. | pipeline: matrix_fisher_rotation_evidence() |
| **L14** | **Step 8 – PlanarTranslationEvidence:** Scan-vs-map translation → t_hat, L_trans (planarized). | pipeline: planar_translation_evidence() |
| **L15** | **Step 9 – Evidence:** Odom pose + odom twist; IMU time‑resolved vMF + gyro + preintegration; LiDAR (Matrix Fisher + planar translation); planar priors. Combined additively (L_evidence, h_evidence). | pipeline: odom_*_evidence(), imu_*_evidence(), build_combined_lidar_evidence_22d(), planar_prior() |
| **L16** | **Step 10 – FusionScaleFromCertificates:** Scale evidence by alpha. | pipeline: fusion_scale_from_certificates_jax() |
| **L17** | **Step 11 – InfoFusionAdditive:** Posterior info = prior_info + scaled_evidence. | pipeline: info_fusion_additive() |
| **L18** | **Step 12 – PoseUpdateFrobeniusRecompose:** Tangent update → SE(3) belief; anchor update. | pipeline: pose_update_frobenius_recompose() |
| **L19** | **Step 13 – PoseCovInflationPushforward:** Map update from scan stats with pose covariance inflation. | pipeline: pos_cov_inflation_pushforward() |
| **L20** | **Step 14 – AnchorDriftUpdate:** Reanchor on belief_recomposed. | pipeline: anchor_drift_update() |
| **L21** | IW sufficient-statistics accumulation; hypothesis and map_stats updated. | on_lidar: accum_*, hypothesis, map_update |

So: **Raw PointCloud2** → parse → T_base_lidar → budget → predict → deskew (IMU) → bin → moment match → Matrix Fisher + planar translation → evidence (odom pose + odom twist + IMU + LiDAR + planar priors) → fusion → recompose → map update → anchor drift. One sequence; every value we trace below flows through this spine.

---

# Part 2: Object 1 — IMU message 5 (stamp, gyro, accel)

**Source:** Row 6 of `docs/raw_sensor_dump/imu_raw_first_300.csv` (5th message).

## 2.1 Raw object (bag / CSV)

| Field | Value | Unit |
|-------|--------|------|
| stamp_sec | 1732437229.419865 | s |
| gyro_x, gyro_y, gyro_z | 0.00348, -0.01219, -0.00835 | rad/s (IMU frame) |
| accel_x, accel_y, accel_z | -0.4655, -0.02375, 0.8787 | **g** (IMU frame) |

Message covariances (orientation, angular_velocity, linear_acceleration) are **not** used.

## 2.2 Where the object goes before the spine (on_imu)

**Step 1 – Scale accel g → m/s²**

- `accel_imu_mps2 = accel_raw * GC_IMU_ACCEL_SCALE` (9.81 m/s² per g).
- Object: **[-4.566, -0.233, 8.620] m/s²** (IMU frame). Still **specific force** (reaction to gravity).

**Step 2 – Rotate into base frame**

- `gyro_base = R_base_imu @ gyro_imu`, `accel_base = R_base_imu @ accel_imu_mps2`.  
  R_base_imu from T_base_imu rotvec [-0.015586, 0.489293, 0] rad.
- Object: gyro_base = **[-0.000804, -0.01232, -0.00882] rad/s** (base); accel_base = **[0.0217, -0.0869, 9.757] m/s²** (base). No gravity subtraction.

**Step 3 – Buffer**

- Append `(stamp_sec, gyro_base, accel_base)` to `imu_buffer`.  
So message 5’s object lives in the buffer until a LiDAR scan runs and pulls the last M samples (**L5**).

## 2.3 Object in the spine (when a LiDAR scan runs)

- **L5:** Last M samples (including message 5 if still in window) copied to `imu_stamps`, `imu_gyro`, `imu_accel` and passed into `process_scan_single_hypothesis`. No change to values.
- **L9 (Deskew):** Preintegration over (scan_start, scan_end) uses these samples. See **2.4** for the exact preintegration steps. Output: delta_pose_scan, xi_body → deskewed_points. So **message 5’s gyro/accel** contribute to where LiDAR points end up.
- **L15 (Evidence):** Same IMU arrays used for (a) preintegration over (t_last_scan, t_scan) → delta_pose_int, delta_p_int, delta_v_int; (b) gyro evidence; (c) vMF gravity evidence (accel_base vs predicted gravity); (d) preintegration factor (delta_p_int, delta_v_int vs predicted position/velocity). So **message 5** contaminates L_odom+L_imu+L_gyro+L_imu_preint+L_lidar → **L_evidence, h_evidence**.
- **L17–L20:** Evidence → fusion → posterior belief → recompose → map update → anchor drift. So **message 5** contaminates **belief_final** and thus **trajectory** (and indirectly map via belief pose).

## 2.4 Preintegration: exact steps (object = one sample i, e.g. message 5)

Preintegration receives: `imu_stamps`, `imu_gyro`, `imu_accel` (base frame), weights, `rotvec_start_WB`, `gyro_bias`, `accel_bias`, `gravity_W = (0, 0, -9.81)` m/s². For **one sample i**:

| Step | Formula | What we get |
|------|---------|-------------|
| **P1** | dt_i = t_{i+1} - t_i, dt_eff = w_i * dt_i (s) | Effective time step |
| **P2** | omega = gyro_i - gyro_bias; dR = Exp(omega * dt_eff); R_next = R_k @ dR | Rotation update (no gravity). Object (gyro) → R_next. |
| **P3** | a_body = accel_i - accel_bias (m/s²) | Bias-corrected **specific force** (body). For msg 5: ≈ (0.02, -0.09, 9.76). |
| **P4** | a_world_nog = R_k @ a_body (m/s²) | Specific force in world (still not linear accel). |
| **P5** | **a_world = a_world_nog + gravity_W** | **Only place gravity is subtracted.** a_world = linear accel (world). For msg 5 (level): ≈ (0.02, -0.09, -0.05) ≈ 0. |
| **P6** | v_next = v_k + a_world * dt_eff (m/s) | Velocity (world). Uses a_world (gravity already out). |
| **P7** | p_next = p_k + v_k*dt_eff + 0.5*a_world*dt_eff² (m) | Position (world). Uses a_world. |
| **P8** | Accumulate sum_a_body, sum_a_world_nog, sum_a_world | For diagnostics / IW. |

**After the full window:** Relative pose and body-frame outputs: **delta_R** = R_start^T @ R_end (relative rotation); **p_body_frame** = R_start^T @ p_end (world displacement in start body frame, m); **v_body_frame** = R_start^T @ v_end (velocity in start body frame, m/s); **delta_pose** (6,) = [p_body_frame, rotvec(delta_R)] for SE(3) relative pose. So **message 5’s accel** (after P5) contributes to **delta_pose_scan** (deskew), **delta_pose_int**, **delta_p_int**, **delta_v_int** → deskew twist, gyro evidence, preintegration factor → evidence → fusion → **trajectory**. Full formulas and units: `PREINTEGRATION_STEP_BY_STEP.md`.

## 2.5 Contamination summary (IMU message 5)

- **Deskew:** Via delta_pose_scan / xi_body → deskewed_points.
- **Gyro evidence:** delta_pose_int[3:6] → L_gyro, h_gyro.
- **vMF gravity:** accel_base (specific force direction) → L_imu, h_imu.
- **Preintegration factor:** delta_p_int, delta_v_int → L_imu_preint, h_imu_preint.
- **Fusion:** All of the above in L_evidence, h_evidence → posterior → **trajectory**.

---

# Part 3: Object 2 — Odom message 5 (pose, covariance)

**Source:** Row 6 of `docs/raw_sensor_dump/odom_raw_first_300.csv` (5th message). **Twist (vx,vy,vz,wx,wy,wz) is never read.**

## 3.1 Raw object (bag / CSV)

| Field | Value | Unit |
|-------|--------|------|
| stamp_sec | 1732437229.607023716 | s |
| x, y, z | 3.07225, 3.96221, 29.96797 | m (parent frame) |
| qx, qy, qz, qw | -0, 0, 0.66237, -0.74918 | — |
| pose covariance (diagonal) | 0.001, 0.001, 1e6, 1e6, 1e6, 1000 | m², m², m², rad², rad², rad² [x,y,z,roll,pitch,yaw] |

First odom (message 1): trans **[3.07019, 3.97681, 29.99595]** m, rotvec **[0, 0, -1.41998]** rad (stored as reference).

## 3.2 Where the object goes before the spine (on_odom)

**Step 1 – Absolute pose from message 5**

- Position **[3.07225, 3.96221, 29.96797]** m; quat → rotvec **[0, 0, -1.44796]** rad.  
  odom_pose_absolute = SE3(trans, rotvec).

**Step 2 – Relative to first odom**

- `last_odom_pose = first_odom_inv ∘ odom_pose_absolute`.  
  Object: **last_odom_pose** trans **[0.01474, -0.00015, -0.02798]** m, rotvec **[0, 0, -0.02798]** rad.

**Step 3 – Covariance**

- `last_odom_cov_se3 = reshape(msg.pose.covariance, (6,6))`. Unchanged. So z information = 1/1e6 = **1e-6** (very weak).

## 3.3 Object in the spine

- **L6:** `last_odom_pose` and `last_odom_cov_se3` passed into `process_scan_single_hypothesis` as odom_pose, odom_cov_se3.
- **L15 (Evidence):** **odom_quadratic_evidence:** residual = log(pred⁻¹ ∘ odom_pose) in tangent space (6D); L_odom = inv(cov_psd); h_odom = L_odom @ delta_z_star. So **message 5’s pose and cov** set the odom measurement and its information (z very weak: 1e-6).
- **L17–L20:** L_odom, h_odom in L_evidence, h_evidence → fusion → posterior → **trajectory**.

## 3.4 Contamination summary (Odom message 5)

- **Evidence:** last_odom_pose and last_odom_cov_se3 → L_odom, h_odom.
- **Fusion:** → posterior belief → **trajectory**. Twist never used.

---

# Part 4: Object 3 — LiDAR (one representative point)

**Source:** No dump of “message 5” point cloud; we trace one point through the formulas. Config: T_base_lidar = [-0.011, 0, 0.778] m, R_base_lidar = I; range weight 0.25, 0.5, 50 m.

## 4.1 Raw point (example)

- p_lidar = **(1.0, 0.2, 0.5)** m (LiDAR frame); time_offset, ring, tag from message.

## 4.2 Parse and transform (before spine)

- t = timebase_sec + time_offset×1e-9 (s). dist = sqrt(x²+y²+z²) = **1.135** m; weight w from sigmoid range formula → **w ≈ 1**.
- **p_base = p_lidar + t_base_lidar** = (1, 0.2, 0.5) + (-0.011, 0, 0.778) = **(0.989, 0.2, 1.278)** m.

## 4.3 Object in the spine

- **L2–L3:** Parse and transform (above).
- **L7:** Budget → object may be kept or dropped; ring, tag preserved.
- **L9:** Deskew: constant twist (from IMU) applied using per-point t → deskewed point.
- **L10–L11:** Soft assign to bins; moment match → scan_bins (centroids, directions, Sigma_p, etc.). Object contributes to bin statistics.
- **L13–L14:** Matrix Fisher → R_hat; PlanarTranslation → **t_hat (3D)** with **planarized z precision** (self‑adaptive from map scatter). Object’s contribution is part of **t_hat**, but z precision is downweighted.
- **L15:** LiDAR evidence from R_hat + planarized translation (pose block only). Translation z information is **scaled down**, and map update later forces `t_hat[2]=0` when writing to the map.
- **L17–L20:** Fusion → posterior → **trajectory** and **map** (map update uses R_hat, t_hat / belief pose).

## 4.4 Contamination summary (LiDAR point)

- **Scan bins** → R_hat, t_hat (planarized) → LiDAR evidence (planarized translation) → fusion → **trajectory** and **map** (map z fixed).

---

# Part 5: Combined flow — raw objects → evidence → final outputs

1. **IMU message 5:** raw (g, rad/s) → scale → rotate → buffer (specific force). When scan runs: **L5** → preintegration **P1–P8**; at **P5** gravity subtracted (a_world = a_world_nog + (0,0,-9.81)) → linear accel → delta_pose, delta_p, delta_v. These feed **L9** deskew, **L15** gyro evidence, vMF gravity, preintegration factor → L_evidence, h_evidence → **L17** fusion → **belief_updated** → **trajectory**.
2. **Odom message 5:** raw pose + cov **and twist + twist cov** → relative pose, cov unchanged → last_odom_pose, last_odom_cov_se3, last_odom_twist → **L6** → **L15** odom pose + twist evidence (pose, velocity, yaw‑rate, pose–twist consistency) → fusion → **trajectory**.
3. **LiDAR point:** raw → parse → base → **L7** budget → **L9** deskew (using IMU) → **L10–L11** bin → **L13–L14** Matrix Fisher + PlanarTranslation → R_hat, **t_hat (planarized)** → **L15** LiDAR evidence → fusion → **trajectory** and **map**.

**Trajectory pose (including z)** now gets contributions from: (1) odom pose (z weak, per covariance); (2) LiDAR translation **with planarized z precision**; (3) IMU via delta_p, delta_v and gyro/vMF; (4) always‑on planar priors. Map update forces `t_hat[2]=0`, so map–scan feedback no longer reinforces z. Full preintegration step detail: `PREINTEGRATION_STEP_BY_STEP.md`.

---

# Part 6: Belief and 22D information (6D pose Hessian)

The pipeline keeps state and belief in **information form** on a **22D tangent space**. Every evidence term and the fusion/recompose steps operate on this same representation.

## 6.1 State dimension and blocks (D_Z = 22)

| Slice | Block | Dimension | Meaning |
|-------|--------|-----------|---------|
| 0:3 | trans | 3 | Translation (m) |
| 3:6 | rot | 3 | Rotation (rotvec, rad) |
| 6:9 | vel | 3 | Velocity (m/s) |
| 9:12 | bg | 3 | Gyro bias (rad/s) |
| 12:15 | ba | 3 | Accel bias (m/s²) |
| 15:16 | dt | 1 | Time offset (s) |
| 16:22 | ex | 6 | LiDAR–IMU extrinsic (6D pose) |

**Pose block** = 0:6 = [trans, rot]. This is the 6D block that becomes SE(3) pose; its **information matrix** is **L[0:6, 0:6]** (the 6×6 Hessian of the NLL in tangent space).

## 6.2 Belief representation

- **Belief** = (L, h, z_lin, X_anchor, …). **L** = (22, 22) information matrix (= Hessian of negative log-likelihood at z_lin). **h** = (22,) information vector; mean in tangent space is **μ = L⁻¹ h** (when L is invertible). **z_lin** = (22,) linearization point (chart origin).
- **Covariance** (when needed): Σ = L⁻¹ (e.g. for map inflation, diagnostics). Pose covariance = Σ[0:6, 0:6].

## 6.3 Where the 22D / 6D Hessian appears in the spine

| Step | What happens to L / belief |
|------|----------------------------|
| **L8 (PredictDiffusion)** | belief_prev (L_prev, h_prev) → convert to cov_prev = L_prev⁻¹ → OU diffusion: cov_pred = f(cov_prev, Q, dt_sec) → invert to L_pred, h_pred; output belief_pred. **Q** (22×22) from process-noise IW state. |
| **L15 (Evidence)** | Each evidence term produces (L_*, h_*) (22×22, 22). Odom: L_odom[0:6,0:6] = inv(odom_cov_se3), h_odom[0:6] = L_odom @ delta_z_star. LiDAR: L_lidar from R_hat, t_hat, t_cov (pose block + optional extrinsic block). IMU gyro: rotation block. IMU vMF: rotation block. IMU preint: position and velocity blocks (0:3, 6:9). All combined: **L_evidence** = L_lidar + L_odom + L_imu + L_gyro + L_imu_preint, **h_evidence** = sum of h_*. |
| **L16 (FusionScale)** | Scale evidence by alpha (from certificates): L_ev_scaled, h_ev_scaled. Excitation scaling can scale prior L/h on extrinsic block 16:22. |
| **L17 (InfoFusionAdditive)** | **L_post = L_prior + α L_evidence**, **h_post = h_prior + α h_evidence**. Posterior belief in information form; pose block L_post[0:6,0:6] is the updated 6D pose Hessian. |
| **L18 (PoseUpdateFrobeniusRecompose)** | **delta_z** = L_post⁻¹ @ h_post (MAP increment, 22D). **delta_pose_z** = delta_z[0:6]. Frobenius BCH correction on pose → delta_pose_corrected. New world pose: X_new = X_anchor ∘ Exp(delta_pose_corrected). Then shift chart: z_lin_new = z_lin - shift (pose part), h_new = h - L @ shift so non-pose state is preserved. Output belief_recomposed. |
| **L20 (AnchorDriftUpdate)** | Optional reanchor (rho); updates z_lin and h, same 22D form. |

So: **6D pose Hessian** = L[0:6, 0:6] at every step; it is updated by predict (L8), then by additive evidence (L15–L17), then the pose part is used in recompose (L18) to get the SE(3) trajectory pose.

---

# Part 7: Adaptive noise (Inverse-Wishart)

Noise is **not** fixed: **Q** (process), **Sigma_g** (gyro), **Sigma_a** (accel), **Sigma_meas** (LiDAR) come from **IW state** and are updated every scan from sufficient statistics. So raw values also “contaminate” the **noise** used on the next scan.

## 7.1 Where IW state is read (start of scan, before L6)

- **backend_node** (once per scan, before looping over hypotheses):  
  - **config.Sigma_meas** = measurement_noise_mean_jax(measurement_noise_state, idx=2) (LiDAR).  
  - **config.Sigma_g** = measurement_noise_mean_jax(measurement_noise_state, idx=0) (gyro).  
  - **config.Sigma_a** = measurement_noise_mean_jax(measurement_noise_state, idx=1) (accel).  
  - **Q_scan** = process_noise_state_to_Q_jax(process_noise_state) (22×22).  
- These are passed into `process_scan_single_hypothesis` as **config** and **Q**. So **this scan** uses the IW state **from the end of the previous scan**.

## 7.2 Where adaptive noise is used in the spine

| Step | Uses |
|------|------|
| **L8 (PredictDiffusion)** | **Q_scan** (22×22) in OU diffusion. |
| **L15 (Evidence)** | **Sigma_g** in gyro evidence and (via dt) in gyro rotation covariance; **Sigma_a** in preintegration factor and vMF/accel IW; **Sigma_meas** (or per-bucket IW) in LiDAR translation WLS and LiDAR evidence. Odom uses **message** covariance (not IW). |
| **L21 (IW accumulation)** | During pipeline, each hypothesis returns **iw_process_dPsi, iw_process_dnu**, **iw_meas_dPsi, iw_meas_dnu**, **iw_lidar_bucket_dPsi, iw_lidar_bucket_dnu**. These are accumulated (weighted by hypothesis weights). |

## 7.3 Where IW state is updated (after hypotheses combined)

- **After** all hypotheses run and are combined, **backend_node** applies IW updates **once per scan** (no gating; readiness is a weight on sufficient stats):  
  - **process_noise_state** ← process_noise_iw_apply_suffstats_jax(accum_dPsi, accum_dnu, dt_sec).  
  - **measurement_noise_state** ← measurement_noise_apply_suffstats_jax(accum_meas_dPsi, accum_meas_dnu).  
  - **lidar_bucket_noise_state** ← lidar_bucket_iw_apply_suffstats_jax(accum_lidar_bucket_dPsi, accum_lidar_bucket_dnu).  
- Then **Q** and **config.Sigma_g, Sigma_a, Sigma_meas** are updated from the new state for the **next** scan.

So: **adaptive noise** = read IW state at scan start → use Q and Sigma_* in L8 and L15 → accumulate sufficient stats in L21 → apply IW update after scan → next scan uses updated noise. Raw residuals (odom, IMU, LiDAR) from this scan therefore influence the **noise** used for the next scan.

---

# Units summary

| Quantity | Unit |
|----------|------|
| Time, stamp_sec | s |
| Gyro | rad/s |
| Accel (raw Livox) | g |
| Accel (after scale) | m/s² |
| gravity_W | m/s² |
| Position, translation | m |
| Rotation (rotvec) | rad |
| Pose covariance (x,y,z) | m² |
| Pose covariance (roll,pitch,yaw) | rad² |
| Sigma_meas (LiDAR) | m² |
| Weights | — |

---

This is the single trace document: one mechanism, deterministic math, values as objects followed from raw inputs to final outputs.

---

# Part 8: Does the pipeline trace explain bad performance? (Legacy run vs current pipeline)

**Note:** The metrics below were from `results/gc_20260128_105746`, **before** planar translation, planar priors, and odom‑twist evidence were added. They are kept for historical context; re‑evaluate with the current pipeline for updated performance.

## 8.1 Z drift (legacy failure mode; now mitigated)

**Legacy mechanism (pre‑fix):** Full‑3D TranslationWLS + strong LiDAR z evidence + map feedback caused z to drift (belief_z → map z → t_hat[2] → LiDAR z → belief_z).

**Current pipeline fixes (in code):**
- **Planar translation evidence** with self‑adaptive z precision (`pipeline.py:633`, `matrix_fisher_evidence.py:502`)
- **Always‑on planar priors** on z and v_z (`pipeline.py:859`)
- **Map update planar z** (`map_update.py:104`)

## 8.2 X,Y and rotation (legacy under‑constrained; now improved)

**Legacy issue:** Odom twist was unused; pose–twist coupling was missing.

**Current pipeline:** Odom twist now contributes **velocity**, **yaw‑rate**, and **pose–twist consistency** factors (`pipeline.py:884`). Remaining gap: explicit **cross‑sensor Δyaw consistency likelihoods** (gyro ↔ odom ↔ LiDAR) are still diagnostics only.

## 8.3 Map–scan feedback (still present, now planarized)

Map–scan feedback remains inherent (belief → map → alignment → evidence → belief), but alignment is now **Matrix Fisher + planar translation**, and map z is fixed to the plane. Errors can still propagate if alignment is wrong, so sign/frames diagnostics remain important.

## 8.4 Summary (current status)

- Legacy z‑drift mechanism is **fixed** by planar translation + planar priors + planar map update.
- Odom twist is **now used** (velocity/yaw‑rate/pose–twist consistency).
- Remaining gaps are **cross‑sensor consistency likelihoods** and **nonlinear approximation stress** (vMF/MF → Gaussian).
