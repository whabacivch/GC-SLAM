# Map and Belief Audit (2026-01-30)

**Trigger:** Run GC eval (Kimera, 60s); analyze why estimates may be low or misaligned and audit map/belief system.

**Run:** `results/gc_20260130_125224` (bag: 10_14_acl_jackal-005, first 60s, 160 scans).

---

## 1. Run Summary

| Metric | Value |
|--------|--------|
| ATE translation RMSE | 0.38 m (P95: 0.86 m) |
| ATE rotation RMSE | 0.65 deg |
| Per-axis trans RMSE | X: 0.36 m, Y: 0.12 m, Z: 0.01 m |
| Per-axis rot RMSE | Roll: 0.48°, Pitch: 0.32°, Yaw: 0.29° |

**Displacement (ptp) after initial-pose alignment:**

| Axis | GT (m) | EST (m) |
|------|--------|---------|
| X | 0.19 | **0.97** |
| Y | **0.32** | 0.20 |
| Z | 0.05 | 0.05 |

**Finding:** We **over-move in X** (0.97 vs GT 0.19) and **under-move in Y** (0.20 vs GT 0.32). So the issue is not “estimates too low” globally — it is **axis-wise mismatch**: we drift/scale strongly in X and weakly in Y relative to ground truth.

---

## 2. Pipeline Trace: Odom → Evidence → Fusion

**Verified in code (actual behavior):**

1. **Odom ingestion** (`backend_node.py`):
   - `on_odom`: Reads pose (position + quat → rotvec), uses `GC_PLANAR_Z_REF` for z. Builds `odom_pose_absolute` = T_{odom←base} (no invert). Then `last_odom_pose = first_odom_inv ∘ odom_pose_absolute` (anchor-relative). `last_odom_cov_se3` = 6×6 from `msg.pose.covariance` [x,y,z,roll,pitch,yaw]. Twist: `last_odom_twist` = [vx,vy,vz,wx,wy,wz] in body; `last_odom_twist_cov` from `msg.twist.covariance`.

2. **Odom pose evidence** (`pipeline.py` ~708, `odom_evidence.py`):
   - `odom_quadratic_evidence(belief_pred_pose, odom_pose, odom_cov_se3)`.
   - `T_err = se3_relative(odom_pose, belief_pred_pose)` = `belief_pred^{-1} ∘ odom_pose` (pull pred toward odom). L = Σ^{-1} from cov_ros (PSD projected), h = L @ delta_z_star. L_odom, h_odom added to L_evidence, h_evidence.

3. **Odom twist evidence** (`pipeline.py` ~912–951):
   - `odom_velocity_evidence(v_pred_world, R_world_body, v_odom_body, Sigma_v)`: Sigma_v = odom_twist_cov[0:3,0:3]. L_vel, h_vel added.
   - `odom_yawrate_evidence(omega_z_pred, omega_z_odom, sigma_wz)`: sigma_wz = sqrt(odom_twist_cov[5,5]). L_wz, h_wz added.
   - `pose_twist_kinematic_consistency(pose_prev, pose_pred, v_body, omega_body, dt, Sigma_v, Sigma_omega)`: L_kin, h_kin added.

4. **Fusion** (`pipeline.py` ~885–976):
   - `L_evidence = L_lidar + L_odom + L_imu + L_gyro + L_preint + L_planar + L_vz + L_vel + L_wz + L_kinematic`.
   - Fusion scale: alpha = 1.0 (alpha_min = alpha_max = 1.0 in constants). No scaling down.
   - Info fusion: L_post = L_pred + alpha * L_evidence, h_post = h_pred + alpha * h_evidence; then recompose.

**Conclusion:** Odom pose, velocity, yaw rate, and pose–twist kinematic consistency are all in the evidence sum at full strength. The pipeline is using odom fully; the issue is not “odom not used.”

---

## 3. Likely Cause: Frame / Axis Convention (GT vs Odom)

- **Observation:** In the **estimate frame** (after initial-pose alignment), GT moves 0.19 m in X and 0.32 m in Y; we move 0.97 m in X and 0.20 m in Y. So in our frame we emphasize X; GT in our frame emphasizes Y.
- **Hypothesis:** Odometry (and thus our state) may use a different **forward** direction than the frame in which Kimera GT is expressed. For example:
  - If **odom** uses ROS REP-103 (X forward, Y left, Z up) and the Jackal actually drove mostly “forward” in the world, our state would move mainly in our X.
  - If **GT** was exported or defined with **Y forward** (or a 90° rotation relative to odom), then after we align the first pose, GT’s “forward” motion would appear as motion along our Y. That would match: we move a lot in X (our odom-forward), GT moves a lot in Y (GT’s forward mapped into our frame).
- **Action:** Check Kimera dataset docs and GT export: which axis is “forward” in the GT TUM file, and what is the odom frame’s “forward” (child_frame_id, convention)? If they differ by 90°, either (a) transform GT (or odom) so both use the same forward axis before comparison, or (b) document the convention and compare in a single, agreed frame.

---

## 4. Map and Belief Audit

**Map state** (`map_stats`):
- **Structure:** Directional bin statistics: `S_dir`, `N_dir`, scatter, etc. (see `map_update.py`, `bin_atlas.py`). Used by LiDAR evidence (Matrix Fisher rotation + planar translation).
- **Update:** `pos_cov_inflation_pushforward` in `map_update.py`: scan statistics pushed into bins with covariance inflation. No explicit “splat map” or BEV grid; map is bin-based.
- **Influence on belief:** LiDAR evidence (step 7–8) uses `map_stats` (e.g. directional resultants) to build rotation and translation evidence. If bin directions or frame conventions were wrong, LiDAR could pull pose in the wrong direction; that could reinforce an X/Y bias if the scan–map residual were systematically aligned with one axis.

**Belief state:**
- 22D information form: [trans(0:3), rot(3:6), vel(6:9), bg(9:12), ba(12:15), dt(15:16), ex(16:22)]. Anchor-relative; first odom defines origin; anchor_correction can apply smoothed initial frame.
- **Prediction:** OU-style diffusion (predict step); then evidence step adds L_evidence, h_evidence; recompose gives updated belief. No hidden gates; alpha = 1.0.

**Checks to do (code + config):**
- Confirm **bin atlas** and **directional bins** use the same world/base convention as odom (Z-up, X/Y consistent with child_frame_id). See `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`, `binning.py`, `map_update.py`.
- Confirm **LiDAR points** are transformed into the same base/world frame as the belief (T_base_lidar, pointcloud_frame_id). Already documented for Kimera in `docs/KIMERA_FRAME_MAPPING.md`.

---

## 5. Odom Covariance and Strength

- If odom **pose** covariance is very small on x,y and large on z,roll,pitch, we trust odom x,y strongly — so we follow odom in the plane. That is expected.
- If the **bag’s** odom frame has a different orientation than GT (e.g. 90°), we would still “follow” odom correctly in odom’s frame, but when we compare to GT (in GT’s frame, then aligned to our frame), we see the axis swap (our X vs GT’s Y).
- **Recommendation:** Log or inspect first few odom poses (x, y, yaw) and first few belief poses (x, y, yaw) to confirm they are aligned. Optionally log odom_twist (vx, vy, omega_z) and compare to finite-difference pose change to confirm twist↔pose consistency.

---

## 6. Odom vs belief diagnostic (added)

- **Backend:** When `odom_belief_diagnostic_file` is set (e.g. by the eval script to `results/gc_XXX/odom_belief_diagnostic.csv`), the backend writes one CSV row per scan with: `scan`, `t_sec`, raw odom (`odom_x`, `odom_y`, `odom_yaw_deg`, `odom_vx`, `odom_vy`, `odom_wz`), belief at scan start (`bel_start_*`), and belief at scan end (`bel_end_*`). So you can compare **raw odom** vs **belief before** vs **belief after** (estimate) over time.
- **Eval script:** `run_and_evaluate_gc.sh` passes the diagnostic file path and `odom_belief_diagnostic_max_scans:=200` so the CSV is written under the run's results directory (first 200 scans).

---

## 6.1 Odom/belief CSV analysis (run gc_20260130_130459)

**Source:** `results/gc_20260130_130459/odom_belief_diagnostic.csv` (149 scans, first 200).

**Findings:**

1. **Early run (scans 1–~80):** Raw odom in the CSV is effectively **stuck at origin**: `odom_x ≈ 0.000006`, `odom_y = 0`, `odom_yaw_deg ≈ 0.001`. Belief is driven by LiDAR/IMU and drifts (e.g. `bel_end_y` grows to ~0.015 m by scan 80).
2. **Later run (scans 129–149):** Raw odom is **large and moving**: e.g. scan 149 `odom_x ≈ 9.76` m, `odom_y ≈ -13.0` m, `odom_yaw_deg ≈ -31°`. So the bag does publish meaningful odometry; it was just near zero for the first portion of the run.
3. **Belief under-tracks odom:** At scan 149, **belief end** is `bel_end_x ≈ 1.23` m, `bel_end_y ≈ -0.24` m — i.e. the estimate is only ~13% of odom in X and ~2% in Y. So the fused state is **much closer to the LiDAR-driven motion** than to odom. That implies LiDAR (and/or IMU) evidence has much higher information (tighter covariance) than odom in this run, or odom is down-weighted by large reported covariance.
4. **Axis-wise:** The ratio odom/belief is much larger in Y than in X (≈54 vs ≈8), so belief under-tracks odom more strongly in Y. Together with "over-move in X, under-move in Y" vs GT, this is consistent with: (a) GT vs our frame having an axis convention difference (e.g. 90°), and/or (b) LiDAR evidence biasing the estimate along one axis in our frame.

**Conclusion:** The diagnostic confirms that **odom is in the pipeline** (same anchor-relative frame as belief; values logged from `last_odom_pose`). The estimate does not follow odom closely because **belief is dominated by LiDAR (and IMU)** — odom is either sparse early, or its covariance in the bag makes it weak relative to scan matching. Next: (1) Confirm GT vs odom frame (which axis is forward); (2) Inspect odom pose/twist covariances in the bag to see effective weight vs LiDAR.

---

## 7. Checklist (Next Steps)

- [ ] **GT vs odom frame:** Confirm Kimera GT TUM axis convention (which axis is “forward”) and odom frame (child_frame_id, REP-103 or dataset-specific). If they differ, apply or document a fixed transform (e.g. 90° about Z) for evaluation.
- [ ] **Inspect odom covariances:** From the bag, print or plot `msg.pose.covariance` and `msg.twist.covariance` (e.g. diagonal) to see relative strength on x, y, yaw vs z, roll, pitch.
- [x] **Use odom/belief CSV:** Analyzed `results/gc_20260130_130459/odom_belief_diagnostic.csv` — belief under-tracks odom; LiDAR dominates; see §6.1.
- [ ] **Map/bin frame:** Verify bin directions and map_update use the same world/base convention as state and odom (no spurious 90° or axis swap in LiDAR evidence).
- [ ] **RPE:** RPE @ 1 m was nan (trajectory length &lt; 1 m in some dimension or strict delta). Consider lower RPE delta (e.g. 0.5 m) for short 60 s runs, or run longer bag for RPE.

---

## 8. References

- **Code:** `backend_node.py` (on_odom, last_odom_pose, last_odom_twist), `pipeline.py` (step 9, odom + twist + kinematic), `odom_evidence.py`, `odom_twist_evidence.py`.
- **Frames:** `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`, `docs/KIMERA_FRAME_MAPPING.md`.
- **Eval:** `tools/evaluate_slam.py` (align_trajectories_initial, per-axis errors), `tools/run_and_evaluate_gc.sh`.
