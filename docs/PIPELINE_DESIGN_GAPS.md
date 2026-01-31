# Pipeline Design Gaps (Known Limitations)

This doc records **known design gaps** in the Golden Child SLAM v2 pipeline, based on the raw-measurements audit, message trace, and covariance inspection.

## Audit status (source of truth)

**Code is the source of truth.** This document was audited against the currently running backend pipeline code on **2026-01-29**.

Key code anchors (actual behavior):
- Backend sensor handling: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py` (IMU buffering at `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:466`, odom twist read at `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:547`, LiDAR parsing at `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:129`).
- Per-scan evidence assembly: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py` (time-resolved accel evidence at `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:713`, planar translation at `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:633`, planar priors at `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:859`, odom twist evidence at `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:884`).
- Map update planar z fix: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/map_update.py` (sets `t_hat[2]=0` at `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/map_update.py:104`).

**References (design context, may be stale vs code):** `docs/PIPELINE_TRACE_SINGLE_DOC.md`, `tools/inspect_odom_covariance.py`. M3DGR-era audits (raw measurements, z evidence, trajectory/GT frame) are in `archive/docs/`.

---

## 1. We are not using a lot of available information

| Source | What we use | What we leave on the table |
|--------|-------------|----------------------------|
| **Odom** | Pose (x,y,z, quat) + pose covariance (6×6). **Twist (vx, vy, vz, wx, wy, wz) + twist covariance (6×6)** — **now used** (velocity factor, yaw-rate factor, pose–twist kinematic consistency; see Phase 2 odom twist evidence). | — (twist is now read and fused.) |
| **IMU** | Gyro, accel (scaled, rotated to base); preintegration; vMF gravity; gyro evidence; preint factor. | **Message covariances** (orientation, angular_velocity, linear_acceleration) — carried through the frontend but not consumed by the backend (backend `on_imu()` reads angular_velocity and linear_acceleration only; see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:466`). **Orientation** (if present) — not used in backend. No explicit forward/lateral decomposition. |
| **LiDAR** | x, y, z, timebase, time_offset, ring, tag; range-based weights; bucket reliability (ring/tag). | **Intensity (reflectivity)** — present in PointCloud2 schema but not returned/used by the backend parser (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:129`). **Per-point covariance** — message has none; backend passes zeros (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:555`). |

So we still underuse IMU (no message cov, no orientation) and LiDAR (no intensity). Odom twist is now used.

---

## 2. We treat measurements as more independent than they are

Physically, **pose and twist are related** by kinematics:

- **dp/dt = R @ v** (world velocity = R × body velocity)
- **dR/dt = R @ ω̂** (orientation rate from body angular rate)

So if we are moving forward (vx > 0) and yawing (wz ≠ 0), we **should** see motion in x and y over time. The odom message gives us both **pose** and **twist** at the same time; they are **not** independent.

**What the code does now (partial fix):**
- Adds odom twist evidence (velocity factor + yaw-rate factor) and a pose–twist kinematic consistency factor (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:884`).

**Remaining gap:** we still do not treat odom pose and odom twist as a single *joint* observation with a declared pose–twist cross-covariance. We also do not include an explicit likelihood term for **gyro-integrated Δyaw vs odom Δyaw**; yaw increment agreement is logged as diagnostics (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:756`) but not fused as evidence.
Similarly, **odom pose** and **IMU preintegration** are related: integrated gyro should match odom yaw change; integrated accel + gravity should be consistent with odom position change. We log dyaw_gyro / dyaw_odom / dyaw_wahba for diagnostics but **do not** feed that consistency (or inconsistency) into the fusion model. So we treat odom pose, IMU gyro, and IMU accel as separate evidence streams without modeling their dependence.

---

## 3. The 6D pose evidence (odom) is poorly designed for what we have

Current design:

- **Input:** One 6D pose (trans + rotvec) and one 6×6 pose covariance from the message.
- **Math:** Residual = log(pred⁻¹ ∘ odom_pose) in tangent space (6D). Information = inverse(covariance). Evidence = Gaussian on that 6D residual with that information matrix.
- **Covariance:** Whatever the bag publishes (in our bag: diagonal 0.001, 0.001, 1e6, 1e6, 1e6, 1000 in [x,y,z,roll,pitch,yaw] with units m² and rad²; later messages can switch to different diagonals, e.g. yaw 1e-9). We do **not**:
  - Use twist to shape or validate the pose residual (e.g. “pose change should align with integrated twist”).
  - Model pose–twist coupling (e.g. joint observation on [pose; twist] or pose given twist).
  - Enforce planar structure inside the **odom pose factor** beyond what the message covariance says (the pipeline does add separate planar priors on z and v_z; see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:859`).
  - Use forward/lateral decomposition (vx, vy, wz) so that motion in x/y is tied to “moving forward while turning.”

So the “pose 6 matrix of evidence” is just **inverse(message 6×6 pose covariance)** applied to a 6D pose error. It is not designed around:

- Kinematic coupling (pose ↔ twist).
- Joint pose↔twist modeling (including pose–twist cross-covariance and conditioning).
- Consistency with IMU (gyro ∫ vs Δyaw; velocity vs integrated accel).

That is what we mean by “poorly designed”: it uses only part of the odom message and does not account for how pose and twist (and IMU) depend on each other.

---

## 4. What we are not doing that we need for a better design

1. **Odom twist (DONE)** — We now read vx, vy, vz, wx, wy, wz and twist covariance and feed velocity/yaw-rate/pose–twist consistency factors (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:884`).

2. **Planar z / map-z feedback break (DONE)** — The pipeline adds planar priors on z and v_z (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:859`), and the map update forces `t_hat[2]=0` before writing scan stats into the map (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/map_update.py:104`).

3. **Time-resolved accel tilt evidence (DONE)** — Accel evidence is time-resolved with transport-consistency weighting (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:713`).

4. **Model pose–twist dependence (still missing as a declared joint observation)**
   - Either: joint observation on [pose; twist] with a covariance that reflects kinematics, or
   - Pose observation conditioned on twist (e.g. “pose change given twist” residual), so we don’t treat pose as independent of how the robot is moving.

5. **Reflect kinematics in the evidence (still incomplete)**
   - “Moving forward + yaw” should imply motion in x and y; the observation model should enforce that structure beyond separate scalar-ish factors.

6. **Use forward/lateral structure (still missing)**
   - Decompose motion into forward (vx), lateral (vy), and yaw (wz) from odom; optionally from IMU (ax, ay in body). Fuse in a way that respects that structure instead of one opaque 6D pose Gaussian.

7. **Consistency between sensors (still mostly diagnostics)**
   - Use dyaw_gyro vs dyaw_odom vs dyaw_mf agreement inside the model (currently logged as diagnostics; see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:756`).

8. **Use more of the raw message (still missing)**
   - IMU: message covariances and orientation are not consumed by the backend (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:466`).
   - LiDAR: intensity is present in the schema but not returned/used by the backend parser (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:129`).

---

## 5. Summary

- **Odom:** Twist and twist covariance **are now used** (velocity/yaw-rate/pose–twist consistency factors). **Underuse of info:** We do not use IMU message covariances or orientation; we do not use LiDAR intensity for weighting or features.
- **Independence assumption:** We now couple pose and twist via kinematic consistency; we still do not feed dyaw_gyro / dyaw_odom / dyaw_mf agreement into fusion (diagnostics only).
- **6D pose evidence:** Odom pose evidence remains “pose residual + inverse(message 6×6 covariance)” and is supplemented by separate twist factors and planar priors. Forward/lateral structure and declared joint pose–twist covariance remain open design gaps.

These remaining gaps (IMU cov, LiDAR intensity, consistency-based weighting) can be addressed in future observation-model work.

---

### 5.4.1 Initial alignment: smoothed reference from first odom/IMU (design note)

**Idea:** Right now we anchor the whole trajectory to the **first single odom message** (`first_odom_pose = odom_pose_absolute` on first callback). If that sample is noisy or an outlier (bag start, robot not yet stable), the reference frame is biased and the entire trajectory is shifted/rotated relative to ground truth. Evaluation then shows a mix of that **initial anchor error** and real drift.

**Data availability:** In typical bags (e.g. M3DGR Dynamic01), IMU and odom start before or with the first LiDAR. At 200 Hz IMU and ~10 Hz LiDAR, even a short delay to first scan (e.g. 0.5 s) gives **on the order of 100 IMU samples** before the first LiDAR callback; we also get several odom messages in that window. So we already have plenty of pre–first-scan data to form a smoothed initial reference (mean position + quaternion mean over first K odom, optionally gravity/bias from first T_init seconds of IMU) without delaying the pipeline.

**Proposal:** Form the initial reference from a **fixed window** of the first few odom (and optionally IMU) measurements, instead of a single sample.

- **Odom:** Buffer the first K odom poses (or first T_init seconds). Then set `first_odom_pose` to an **aggregate** of that buffer, e.g.:
  - Translation: mean of positions.
  - Orientation: mean quaternion (e.g. Markley quaternion mean) of the buffer poses.
  Then all subsequent odom is still `odom_relative = first_odom^{-1} ∘ odom_absolute`; we only change how `first_odom_pose` is chosen.
- **IMU (optional):** Use the first T_init seconds of IMU (when the robot is often stationary) to compute mean accel (gravity direction) and mean gyro (bias). That could refine initial orientation or `T_base_imu` before we start integrating, instead of relying on a single sample or the preconfigured extrinsic only.

**By construction:** No heuristic gating. Define a **fixed initialization window** (e.g. `init_window_sec` or `init_window_odom_count`) and a **fixed procedure** (e.g. mean position + quaternion mean over that window). Pipeline does not run (or odom is not used for reference) until the window is full; then we set the reference once and proceed. No "if first N then smooth else don't" — we always use the same init procedure. The window size is a **prior/budget** (e.g. 0.5 s or 10 odom messages), not data-dependent.

**Where it would go:** `backend_node.py`: in the odom callback, if `first_odom_pose` is not yet set, either (a) buffer this pose and, once the buffer has K poses (or T_init has elapsed), compute the aggregate and set `first_odom_pose`, or (b) delay starting the pipeline until the init window is full, then set reference and start. LiDAR scans that arrive before the reference is set would need a defined policy (e.g. drop, or use identity relative to first odom and then re-anchor when reference is set — the latter is trickier). Prefer (a) with a small buffer (e.g. 10 odom messages) so the first few scans still get odom (e.g. relative to first sample) and we only commit the **reference** once the buffer is full; that requires defining "relative to first sample" vs "relative to smoothed reference" for the interval before the reference is set (e.g. use first sample as temporary reference, then switch to smoothed reference and apply a constant correction to existing state — or simply delay pipeline start until smoothed reference is ready).

**Why it might help:** A more stable initial frame could reduce the constant rotation/translation offset we see vs GT (part of which may be "first sample was bad") and improve time alignment in evaluation. It does not fix drift over time but can improve the anchor.

**Implemented (2026-01-28):** Explicit anchor A; provisional A0 on first odom (no scan drops); after K odom, A_smoothed = weighted t̄ + polar(M) with IMU stability weights; export uses `anchor_correction ∘ pose_belief`. See `backend_node.py` (A0, odom_init_buffer, anchor_correction, _polar_so3, _imu_stability_weights), `constants.py` (GC_INIT_ANCHOR_*), `gc_unified.yaml` (init_window_odom_count).

---

### 5.4.2 “Huge trajectory changes at first, then tiny later” and constant frame shift (investigation)

**Observation (dashboard + 3D plot):** Trajectory shows large corrections in the first few scans (e.g. ||Log(R_pred^T R_MF)|| up to ~80°, condition numbers high then dropping, ΔI_rot/ΔI_xy huge at scan 0 then negligible). Later, the estimated path roughly follows ground-truth shape but with a persistent spatial/rotational offset.

**Why “huge first, then tiny later” (code and data):**

1. **Empty map on first scan** — When `map_is_empty` (`pipeline.py:1102`), we do **not** use Matrix Fisher scan-to-map alignment for map placement; we use the **posterior** belief (odom + IMU + LiDAR evidence fused) to place the map (`pipeline.py:1108–1122`). So scan 0 adds a large amount of LiDAR evidence to an **ill-conditioned** state (high cond(MF), high cond(pose6)); the information gain ΔI_rot and ΔI_xy is huge because we’re constraining an under-determined system. By scans 1–3 the map exists, MF alignment kicks in, and the state is better conditioned, so subsequent ΔI drops.

2. **Conditioning** — Dashboard shows `log10(MF cond)` and `log10(cond pose6)` starting high (~3–4) and dropping to ~1 within a few scans. Ill-conditioning implies large updates for the same residual; once well-conditioned, updates become smaller. So “huge then tiny” is consistent with the estimator going from ill- to well-conditioned.

3. **Anchor timing** — Export is `pose_export = anchor_correction ∘ pose_6d` (`backend_node.py:1001`). `anchor_correction` is identity until we have K odom messages, then it is set to `inv(A_smoothed) ∘ A0`. We do **not** retroactively rewrite already-exported TUM poses. So if the first LiDAR scan occurs **before** we have K odom, the first few poses in the TUM file are in A0 frame and later poses are in A_smoothed frame — i.e. a **one-time frame jump** in the trajectory file. If the first scan occurs **after** K odom (typical when odom/IMU start before LiDAR), every published pose already has `anchor_correction` applied; the “huge first” is then purely from (1) and (2). To confirm which case applies, check whether `Anchor smoothed: K=...` is logged before or after the first `on_lidar` callback in a run.

4. **Scan-0 feature quality** — Dashboard “Top-12 bins (scan 0)” shows variable `k_scan` and near-zero `k_map` / planarity / aniso (map not yet formed). Weak or inconsistent geometric constraints at scan 0 can contribute to large initial corrections.

**Does the data indicate a constant frame shift?**

- **No pure constant shift** — The evaluator uses evo’s SE(3) Umeyama alignment (`evaluate_slam.py`: `est_sync.align(gt_sync, correct_scale=False)`). A **constant** SE(3) error would be absorbed by that alignment; the reported ATE is the **residual** after best-fit SE(3). So the ~2 m translation and ~134° rotation ATE we see are **non-constant** residuals (drift, or initial-phase vs later-phase structure).

- **What the 3D plot suggests** — After the chaotic start, the estimated trajectory (orange) follows the ground-truth (cyan) shape but with a **consistent offset**. That can be (a) residual after Umeyama (so the error is not exactly constant), (b) **initial anchor bias**: first poses in A0, then switch to A_smoothed, leaving an unresolved offset, or (c) **residual frame/calibration**: e.g. `body_T_wheel` inexact or GT in a slightly different frame (see `archive/docs/TRACE_TRAJECTORY_AND_GROUND_TRUTH.md`).

- **Odom drift vs frame** — Dashboard “deg” panel shows Δyaw odom with a **sawtooth** (drift then sharp correction) and Δyaw MF tracking it. So odom yaw **drifts** and the SLAM system **corrects** it; that is not an uncorrected constant frame shift. A **constant** frame error would appear as a persistent bias in rotation/translation that Umeyama would largely remove; the fact that we still have large ATE after alignment means the error has time-varying or structurally varying components.

**Summary:** “Huge first, tiny later” is explained by empty map + ill-conditioning on scan 0, then MF alignment and better conditioning; anchor timing can add a one-time frame jump in the export if the first scan is before K odom. The data does **not** show a single constant frame shift that we could remove with one fixed transform; it shows corrected odom drift and a residual error (ATE) that is partly initial/anchor and partly residual frame or drift. Next steps: (1) Confirm in logs whether anchor smoothing completes before or after first LiDAR. (2) If before: consider strengthening first-scan observability (e.g. planar prior, or slightly larger initial pose uncertainty) to reduce overshoot. (3) Revisit `body_T_wheel` and GT frame definition if residual rotation/translation bias remains after alignment.

---

### 5.5 Z leakage and rotation (roll) — brainstorm

**Observation:** We still see z drift/leakage in some runs and a **dominant rotation error (roll ~125°)** in ATE. Below: root causes and improvement ideas.

#### Why z might still leak

1. **Process noise Q treats z like x,y** — `create_datasheet_process_noise_state()` in `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/structures/inverse_wishart_jax.py` uses a **single** scalar per block: trans block gets `GC_PROCESS_TRANS_DIFFUSION` (1e-4 m²/s) for **all three** diagonal entries. So **Q[2,2] = Q[0,0] = 1e-4**. We have `GC_PROCESS_Z_DIFFUSION = 1e-8` in `constants.py` but it is **never used**. So every prediction step adds the same diffusion to z as to x,y; the planar prior has to fight that. **Fix:** Build the trans block of Q with x,y = `GC_PROCESS_TRANS_DIFFUSION` and z = `GC_PROCESS_Z_DIFFUSION` (e.g. diagonal [1e-4, 1e-4, 1e-8]). That requires changing the IW process-noise structure from one scalar per block to either (a) a 3-vector for trans (x, y, z) or (b) a separate 1D “z” block. Option (a) is minimal: initialize trans Psi so that the IW mean has diag(trans) = [TRANS, TRANS, Z_DIFFUSION].

2. **Planar translation still injects z when map has vertical structure** — `z_precision_scale = lambda3 / lambda1` is self-adaptive; when the map has vertical structure (walls), lambda3 > 0 and we add LiDAR z information. If that conflicts with the planar prior (e.g. map built with wrong z or drift), z can move. **Mitigation:** Cap `z_precision_scale` from above (e.g. min(z_precision_scale, 0.1)) so LiDAR never dominates z; or add an explicit “planar robot” mode that forces z_precision_scale = 0 for the trans evidence and relies only on planar prior + odom (with odom z variance already loose).

3. **Fusion scale α** — If conditioning-based or excitation-based α scales down **all** evidence (including planar prior), the prior’s pull on z weakens and prediction (with isotropic Q) can dominate. **Check:** Log L[2,2] from planar prior vs L[2,2] from planar translation and from odom; ensure planar prior remains the dominant z constraint after α.

#### Why rotation (especially roll) is wrong

1. **Gravity / T_base_imu** — Roll and pitch are driven by **IMU vMF gravity evidence** (tilt: “R^T @ (-g_world) ≈ accel_body”) and by **Matrix Fisher** (LiDAR directions). If `GC_GRAVITY_W` or the IMU extrinsic `T_base_imu` is wrong (sign, axis, or angle), we get a **systematic tilt error** (roll/pitch). The config notes “28° rotation to align IMU gravity with base +Z” and “Previous 154.5° was inverting both gravity and yaw.” So we’ve had convention/sign issues before. **Actions:** (a) **Audit:** Log at startup the predicted gravity direction in body frame (R_base_imu @ [0,0,-1] or equivalent) and the mean accel direction over the first N samples; they should align. (b) **Validate:** Run `tools/estimate_imu_base_extrinsic_rotation.py` (or equivalent) and compare to current `T_base_imu`; if they differ, fix the extrinsic or the gravity constant. (c) **Fail-fast:** If xbar @ mu0 (gravity alignment) is strongly negative on the first few seconds, emit a hard error or warning that gravity/IMU frame is likely wrong.

2. **Odom 6D pose includes roll/pitch** — We use the full 6×6 odom covariance. If the bag publishes **tight** roll/pitch variance (e.g. 0.001), we’d trust odom tilt; if **loose** (1e6), we don’t. For planar wheel odom, roll/pitch are usually **unobservable** and should be very loose. **Check:** Inspect `last_odom_cov_se3[3:5, 3:5]` (roll, pitch) in the bag; if they’re tight, consider **inflating** roll/pitch variance (e.g. max(cov, 1e2)) so we don’t trust odom tilt and rely on IMU + LiDAR instead.

3. **First-scan map at wrong tilt** — On the first scan we place the map using the **posterior** pose (odom + IMU + LiDAR). If that pose has wrong roll/pitch (e.g. from one bad odom or wrong IMU tilt), the map is built tilted and subsequent Matrix Fisher alignment locks onto that tilt. **Mitigation:** (a) Use a **stronger tilt prior** on the first scan (e.g. roll ≈ 0, pitch ≈ 0 with small sigma) when we know the robot is planar. (b) Or delay “committing” the map orientation until we have K scans and then use a smoothed tilt (similar to anchor smoothing).

4. **Planar robot = roll ≈ 0, pitch ≈ 0** — We don’t have an explicit **tilt prior**. For a planar base, we could add a soft prior: roll ≈ 0, pitch ≈ 0 (rad), with a small sigma (e.g. 0.01 rad ≈ 0.6°), and only yaw free. That would pull tilt toward level and reduce the chance that wrong IMU/odom tilt dominates. **Implementation:** New evidence block (or extend planar prior): inject L[3,3] and L[4,4] (roll, pitch) with precision 1/sigma_tilt^2 and h[3], h[4] toward 0; leave yaw (index 5) unconstrained.

5. **body_T_wheel residual** — We already apply `body_T_wheel` at evaluation time. If the residual rotation error is **constant** in the wheel frame, it could be a residual error in that transform or GT being in a slightly different frame. Re-check M3DGR calibration.md and the exact frame that GT uses; if needed, re-estimate or refine `body_T_wheel`.

#### Suggested order of work

1. **Z:** Use `GC_PROCESS_Z_DIFFUSION` in Q (trans block z entry) so prediction doesn’t add full trans diffusion to z.  
2. **Rotation:** Audit gravity/IMU alignment at startup (log xbar @ mu0, predicted vs mean accel direction); validate T_base_imu and GC_GRAVITY_W.  
3. **Rotation:** Inflate odom roll/pitch variance if the bag publishes tight tilt (so we don’t trust odom tilt).  
4. **Rotation (optional):** Add soft tilt prior (roll ≈ 0, pitch ≈ 0) for planar robot.  
5. **Rotation (optional):** First-scan tilt: stronger tilt prior or delayed map-orientation commit.

---

### 5.6 Frame or axis convention mismatch (investigation)

**Report first (code vs docs):**

- **BAG_TOPICS_AND_USAGE.md** (line 79): "`livox_frame` uses **Z-down** convention; requires 180° rotation about X-axis to convert to Z-up `base_footprint` frame."
- **FRAME_AND_QUATERNION_CONVENTIONS.md** (lines 72, 165–177): "`livox_frame` is **Z-up** for this dataset (ground normal points +Z)" and "`T_base_lidar` rotation is identity (rotvec=[0,0,0])."
- **Code:** `backend_node.py:585` applies `pts_base = R_base_lidar @ pts_np.T + t_base_lidar`; `gc_rosbag.launch.py:149` and `gc_unified.yaml:78` set `T_base_lidar = [-0.011, 0.0, 0.778, 0.0, 0.0, 0.0]` (rotation identity). So the code assumes livox_frame is Z-up.

**Observed (evaluation):** For M3DGR frame correction (wheel vs body, anchor smoothing) see `archive/docs/TRACE_TRAJECTORY_AND_GROUND_TRUTH.md`.

**Interpretation:** If livox_frame is actually Z-down and we use identity, all geometry (points, gravity alignment) is Z-flipped; the estimated pose would be in a frame that is 180° about X from base_footprint.

**Recommendation:** Run `tools/diagnose_coordinate_frames.py` on the bag; align `FRAME_AND_QUATERNION_CONVENTIONS.md` and `T_base_lidar` with the result. For M3DGR export/GT frame see `archive/docs/TRACE_TRAJECTORY_AND_GROUND_TRUTH.md`.

---

### 5.7 Compute bottleneck on scan throughput

**What limits how many scans we can process per second:**

1. **IMU preintegration `jax.lax.scan` over a fixed 4000-step buffer** (dominant)
   - **Code:** `backend_node.py` passes `imu_stamps_j`, `imu_gyro_j`, `imu_accel_j` of length `M = max_imu_buffer = 4000` every scan (`backend_node.py:755–767`). `preintegrate_imu_relative_pose_jax` in `imu_preintegration.py` runs `jax.lax.scan(step, ...)` over that full length.
   - **Per scan:** The pipeline calls preintegration **twice per hypothesis** (deskew within-scan + scan-to-scan interval). With `K_HYP = 4` that is **4 × 2 × 4000 = 32,000** scan iterations per LiDAR scan. Only a small fraction of slots (e.g. ~20–200) have non-zero weight; the rest are zero-padded. So the bottleneck is **fixed-cost 4000-length scan** instead of a variable-length scan over only the valid time window.
   - **Mitigation (by construction):** Slice IMU arrays to the actual integration window (e.g. indices where `t_last_scan ≤ stamp ≤ t_scan` for scan-to-scan, and similarly for within-scan deskew) and pass only that slice so `jax.lax.scan` runs over ~20–200 steps instead of 4000. Requires fixed upper bound on window size for JIT (e.g. max 512 or 1024 steps) and padding to that size if smaller.

2. **Top-level pipeline not JITted**
   - `process_scan_single_hypothesis` (`pipeline.py:275`) is a **plain Python function**. Each scan triggers many small JIT kernel dispatches (preintegrate, Wahba SVD, Matrix Fisher, Cholesky solve, etc.). Python overhead and multiple kernel launches per scan add up. A single JIT-compiled “scan pipeline” (with fixed sizes) would reduce dispatch and could improve throughput.

3. **First-scan JIT compilation**
   - The eval script notes “First-scan JIT compilation may take ~30s”. The first time each JIT kernel is hit (preintegrate, Wahba, evidence, etc.) it is traced and compiled. One-time cost; subsequent scans use cached compilations.

4. **Point cloud size**
   - Binning and soft assignment are vectorized over N points (e.g. 10k–50k). Cost is O(N × B_BINS). Typically secondary to the 4000-step IMU scan unless N is very large.

**Summary:** The main compute bottleneck was the **fixed 4000-step IMU preintegration scan**. **Fixed:** IMU is now sliced to the integration window `[min(t_last_scan, scan_start), max(t_scan, scan_end)]`, capped at `GC_MAX_IMU_PREINT_LEN = 512`, and padded to 512; preintegration runs over 512 steps instead of 4000 (`backend_node.py`, `constants.GC_MAX_IMU_PREINT_LEN`). **Wahba removed:** Pipeline uses Matrix Fisher only; `wahba.py` moved to `archive/legacy_operators/`, removed from operators `__init__` and tests.

---

### 5.8 Future: Replace per-point deskew with bin-level motion marginalization (natural-parameter “deskew”)

**Report first (actual behavior today):**

- The backend currently performs **per-point deskew** via a constant-twist model:
  - Operator: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/deskew_constant_twist.py:1` (per-point warp `p0 = Exp(alpha*xi)^{-1} ⊙ p` with `alpha=(t - scan_start)/(scan_end-scan_start)`).
  - Pipeline callsite: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:561`, producing `deskewed_points` used for binning and moment match.

**Design goal:** Keep fixed-cost, single-path behavior but remove the expensive per-point SE(3) warp by marginalizing within-scan motion into a **bin-local covariance inflation** term:

- Do not transform points (no per-point `Exp` / inverse action).
- Inflate the LiDAR likelihood covariance in information form: `Σ_eff,b = Σ_sensor + Σ_motion,b`.
- Treat this as an explicit approximation operator (series truncation + independence assumptions) and apply Frobenius correction when triggered.

#### 5.8.1 Closed-form 2nd-order rotational smear (constant ω)

Assume constant angular velocity `ω` over the scan interval. Let `A = [ω]×` and approximate:

- `R(Δt) p ≈ p + (A p) Δt + (1/2)(A² p) Δt²`
- Rotation-induced displacement: `δp_rot(Δt) ≈ (A p) Δt + (1/2)(A² p) Δt²`

For bin `b`, define:

- Point second moment: `M_b = E[p pᵀ]`
- Time moments: `μ_k = E[Δt^k]` for `k = 1..4`
- Central scalars:
  - `Var(Δt) = μ_2 − μ_1²`
  - `Var(Δt²) = μ_4 − μ_2²`
  - `Cov(Δt, Δt²) = μ_3 − μ_1 μ_2`

Under the bin-local approximation that `(p ⟂ Δt)` (optional refinement below), a symmetric 2nd-order rotational smear covariance is:

```
Σ_motion,rot,b
  ≈ Var(Δt)       · A  M_b  Aᵀ
   + 1/4 Var(Δt²) · A² M_b (A²)ᵀ
   + 1/2 Cov(Δt,Δt²) · ( A M_b (A²)ᵀ + A² M_b Aᵀ )
```

This stays bin-level: a handful of 3×3 matrix products per bin (e.g. B=48).

#### 5.8.2 Required per-bin sufficient statistics

Let per-point weights be `w_i` and responsibilities be `r_{i,b}` from `BinSoftAssign`.

Define bin mass `N_b = Σ_i w_i r_{i,b}` (no hard thresholds; use `eps_mass` for stability).

To compute `M_b` and `μ_1..μ_4`, you need (per bin):

- `Σ w r`  (mass)
- `Σ w r p pᵀ`  (point second moment)
- `Σ w r Δt`, `Σ w r Δt²`, `Σ w r Δt³`, `Σ w r Δt⁴`  (time moments)

**Optional refinement (reduce the (p ⟂ Δt) error):**

- `Σ w r (Δt p)` and `Σ w r (Δt² p)` (fixed-cost; helps with time-skewed sampling patterns).

#### 5.8.3 Δt definition (must match current deskew conventions)

To align with the current deskew operator’s `alpha=(t−t0)/(t1−t0)` (`deskew_constant_twist.py:46`), define:

- `t0 = scan_start_time`, `t1 = scan_end_time`
- `Δt_i = (timestamp_i − t0)` in seconds (and optionally also store `dt_scan = t1 − t0` for scale checks)

#### 5.8.4 Pipeline integration (explicit operator, no fallbacks)

**Where it fits:**

- Keep Step 4 (`BinSoftAssign`) unchanged.
- Add a new operator between Steps 4 and 5:
  - **Step 4.5 (BinMotionSmearCovariance)** consumes `{points, timestamps, weights, responsibilities, ω}` and emits `Σ_motion,b` for each bin.
- Step 5 (`ScanBinMomentMatch`) adds `Σ_motion,b` into each bin covariance before evidence extraction:
  - `Σ_bin,eff = Σ_bin,geom + Σ_sensor + Σ_motion,b` (then SPD-project as a DomainProjection if needed).

**Single-path selection (non-negotiable):**

- Introduce an explicit config selection, e.g. `deskew_model: per_point | bin_smear_2nd_order` (no implicit fallbacks).
- Fail-fast at startup if the selected model cannot be executed (missing timestamps, missing ω estimate, etc.).

#### 5.8.5 ω source (deterministic plug-in, logged)

Rotation smear needs a scan-local `ω` (rad/s). Candidate sources already computed:

- IMU: `omega_avg` from debiased, windowed gyro mean (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:531`).
- Odom: yaw-rate `wz` from odom twist (if used as an explicit competing hypothesis; otherwise keep as diagnostics only).

No gating: if multiple candidates are used, handle them as explicit hypotheses with soft responsibilities (declared model), or pick one declared plug-in and log IMU↔odom disagreement.

#### 5.8.6 Certification + Frobenius correction

This operator introduces approximation triggers by construction:

- `series_truncation` (2nd-order expansion of `Exp([ω]× Δt)`),
- `p_dt_independence_assumption` (if using the simplified factorization),
- optional `moment_correction` / `psd_projection` (if enforcing SPD).

Per policy: `approximation_triggers != ∅` ⇒ `frobenius_applied == True` (no accept/reject branching).

#### 5.8.7 Interaction with self-adaptive LiDAR IW noise

If `Σ_eff,b = Σ_sensor + Σ_motion,b` is used in the likelihood, decide explicitly how LiDAR IW noise is updated:

- Conservative (simple): update IW on residuals under `Σ_eff,b` (can over-inflate sensor noise).
- Better (explicit approximation + certified): treat `Σ_motion,b` as known additive covariance and update IW for `Σ_sensor` using a corrected sufficient statistic (with a DomainProjection to SPD).

This choice changes the model class and must be documented in `docs/GOLDEN_CHILD_INTERFACE_SPEC.md` terms (family in/out, approximation triggers, Frobenius applied).

---

## 6. Operator-by-operator improvement plan

Below is a concrete improvement plan that (1) fixes the known structural failures in the current fixed-cost, single-path pipeline (run per hypothesis), (2) extends it cleanly to **MHT**, and (3) shows where **2nd/3rd-order tensors, higher derivatives, and information-geometry / Hessian / Monge–Ampère ideas** can be used—separating **production-safe** from **high-risk research**.

### 6.0 What the pipeline currently is (and why it fails)

**Key structural facts (from current code):**

- The pipeline is **fixed-cost and branch-free**: every LiDAR scan runs the same steps in order.
- State is a **22D tangent chart** with blocks [t, θ, v, b_g, b_a, dt, ex].
- Runtime always maintains **K_HYP=4 hypotheses** (fixed budget) and combines them via a continuous barycenter projection; current code keeps **uniform hypothesis weights** (no scoring/prune/merge yet).
- **Odom twist is used**: velocity factor + yaw-rate factor + pose–twist kinematic consistency factor (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:884`).
- Accel evidence is **time-resolved** with transport-consistency weighting; it is still vMF-derived, Laplace-approximated, and **PSD-projected**, and it lives in the rotation block only (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:713`).
- LiDAR evidence uses **Matrix Fisher rotation** + **planarized translation evidence** with self-adaptive z precision scaling (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:613` and `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/matrix_fisher_evidence.py:502`).
- The z feedback loop (“map z = belief_z”) is explicitly broken: map update uses `t_hat[2]=0` (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/map_update.py:104`), while belief z is constrained by always-on planar priors (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:859`).
- Fusion is **additive info-form** with an **α** computed from pose-block conditioning/support + excitation scaling on dt/extrinsic, then PSD projection.

**Failure modes implied (model-class errors, not “bugs”):**

1. **Cross-sensor dependence is still under-modeled:** several “agreement checks” (e.g., gyro Δyaw vs odom Δyaw vs LiDAR Δyaw) are diagnostics rather than likelihood terms (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:756`).
2. **Nonlinear likelihoods are still quadraticized** (vMF → Laplace + PSD projection; Matrix Fisher → Gaussian info embedding). When the local approximation is invalid, this can produce overconfident wrong pulls.
3. **Available uncertainty metadata is still ignored:** IMU message covariances/orientation and LiDAR intensity are not consumed by the backend (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:466`, `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:129`).

---

### 6.1 Production-safe pipeline upgrades (do these first)

#### 6.1.1 Make the estimator match the platform: SE(2.5) not SE(3)

**Goal:** Keep full SO(3) (tilt is needed), but treat translation as planar unless vertical observability is truly available.

**Changes:**

- **DONE in code:** planar translation evidence with downweighted z precision (self-adaptive) replaces full-3D isotropic translation evidence (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/matrix_fisher_evidence.py:502`).
- **DONE in code:** always-on planar priors `z ≈ z_ref` and `v_z ≈ 0` (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:859`).
- **DONE in code:** map update prevents z feedback by forcing the map translation to live in z=0 (`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/map_update.py:104`).

**Acceptance test:** z stays bounded near the true robot height instead of running to -50 to -80 m (as in the trace).

#### 6.1.2 Odom twist factors (implemented)

**Implemented 3 evidence operators:**

1. **Body velocity factor** (child frame): \( r_v = v_b - v^{\text{odom}}_b \). Inject into the **vel block [6:9]**, with cov from wheel model (or learned IW).
2. **Yaw-rate factor:** \( r_{\omega_z} = (\omega_{b,z} - b_{g,z}) - \omega^{\text{odom}}_z \). Inject into **gyro-bias / rotation dynamics** (helps stabilize yaw under turning).
3. **Pose–twist kinematic consistency factor** across scan dt: \( \text{Log}(X_k^{-1} X_{k+1}) \approx [R_k v_b \Delta t; \omega_b \Delta t] \). This directly repairs “pose snapshots with no dynamic linkage.”

**Net effect:** Stop asking LiDAR to do everything; reduce sensitivity to wrong scan-matching.

**Status:** Implemented in code (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:884` and `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/odom_twist_evidence.py`).

#### 6.1.3 Replace “single accel mean vMF” with time-resolved, consistency-weighted tilt evidence

**Current model (code):** Time-resolved vMF-style accel direction evidence with transport-consistency weighting → Laplace at δθ = 0 → PSD projection; only constrains pitch/roll (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:713`).

**Status:** Implemented (time-resolved, transport-consistency-weighted tilt evidence; see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:713`).

**Remaining:** PSD projection is mandated by the branch-free numeric contract. The practical goal is to reduce the *magnitude* of PSD projection deltas by improving the local approximation and/or using robust likelihoods upstream (e.g., Student‑t directional), rather than treating PSD projection as a “repair step.”

#### 6.1.4 Make evidence strength depend on measurement quality

Fusion is additive info-form and depends heavily on L magnitudes. If any operator produces “constant strength regardless of reliability,” it will dominate incorrectly.

**Rule:** Every evidence operator must output L = Σ⁻¹ where Σ is actually tied to: dt window length, IMU consistency residuals, scan alignment residuals / ESS, IW-adapted noise state. The trace describes IW feedback (residuals update noise used next scan).

---

### 6.2 Turning this into MHT (multiple hypothesis tracking)

The pipeline math path is branch-free. Runtime has a fixed K_HYP hypothesis container (budgeted), but current code does not implement MHT semantics (no per-scan scoring/prune/merge/branching; hypothesis weights are currently uniform). MHT means introducing branching and weight management correctly.

#### 6.2.1 What a “hypothesis” is

Each hypothesis must contain **everything that affects future evidence:** belief (L, h, z_lin, X_anchor, …), **map_stats** (map drives next scan’s LiDAR alignment: Matrix Fisher rotation + planar translation), **noise state (IW)** (affects Q and Σ in future steps), any latent calibration that varies (dt, extrinsic). So a hypothesis is **pose + map + noise + calibration**, not just pose.

#### 6.2.2 When to branch (practical triggers)

Branch when there is **structural ambiguity:** scan matching has multiple comparable solutions (symmetry/corridor), LiDAR rotation evidence has competing maxima, translation evidence is ill-conditioned, evidence conditioning is bad (α already uses conditioning).

#### 6.2.3 How to score and prune hypotheses (log-evidence in information form)

For each hypothesis j, maintain weight w_j ∝ p(data | H_j). A practical scoring rule: compute the incremental negative log-likelihood at the fused solution; include the Gaussian normalization term (½ log det Σ) or (-½ log det L) in information form. This prevents “overconfident wrong” hypotheses from surviving.

#### 6.2.4 Keep hypothesis count bounded (MHT discipline)

Use a strict cap B (e.g. 4–16). **Prune** by posterior weight; **merge** near-duplicate hypotheses on SE(3) using a proper distance (e.g. geodesic on SO(3) + Mahalanobis on translation in tangent space); **delay commitment:** keep multiple for K scans, then collapse.

#### 6.2.5 How to merge hypotheses without destroying geometry

Moment match in tangent space at a chosen reference (SE(3) log map), but only when modes are close. Use OT barycenters (see Monge–Ampère section) to avoid mode collapse artifacts.

---

### 6.3 Where 2nd- and 3rd-order tensors actually improve estimation

Use higher-order objects only where they fix a known failure (linearization, curvature, multimodality).

#### 6.3.1 Second-order: metric (Hessian/Fisher) should drive the step (not just be inverted)

L is already “the Hessian of NLL” in the tangent chart. **Upgrade:** Use **Riemannian trust-region** steps on SE(3): solve for δ using L but accept/reject with a trust-region ratio; retract via SE(3) exp map. This prevents “flat subspace flinging” when conditioning is bad (which α/conditioning logic is trying to detect).

#### 6.3.2 Third-order: correct the Laplace approximation pathologies

vMF conversion uses an approximate Hessian and then PSD projection. When PSD projection is needed, it is often because we are outside the regime where the 2nd-order approximation is reliable.

**Upgrade options:**

- **Cubic-regularized Newton** (uses 3rd-order Lipschitz bound rather than explicit tensor): stabilizes steps when curvature changes fast; reduces reliance on PSD projection.
- **Explicit 3rd-order correction** (research): compute the 3rd derivative tensor of the directional log-likelihood w.r.t. δθ; apply a cubic Taylor correction to the local model. Use only for accel vMF (and maybe Matrix Fisher residual) where curvature is high and approximation stress is already seen.

---

### 6.4 High-risk research (not production-safe)

#### 6.4.1 Monge–Ampère / Optimal-Transport map updates (⚠ high risk)

**Idea:** Treat map update as transport of a distribution of scan-bin Gaussians into map-bin Gaussians using OT (transport map T = ∇φ, convex potential φ, Jacobian satisfying Monge–Ampère). Practically, approximate using Gaussian OT barycenters per bin. **Why it might help:** Reduces map contamination feedback; principled merge for MHT. **Why it’s not safe:** Expensive, brittle under outliers, hard to tune online, risk of artificial “mass conservation” that fights real motion.

#### 6.4.2 Full Hessian geometry (dually flat) fusion via convex potentials (⚠ medium–high risk)

Choose a convex potential φ and perform mirror descent in natural coordinates; treat sensor fusion as movement along e-/m-geodesics. **Why it might help:** Fusion less sensitive to chart choice and α hacks. **Why it’s risky:** Must pick the right potential and ensure numerical stability; mismatch between true likelihoods (vMF + scan matching) and assumed exponential family can backfire.

#### 6.4.3 Third-order Amari–Chentsov tensor for bias dynamics (⚠ high risk)

Use the 3rd-order information tensor to model skewness/non-Gaussianity in bias evolution or scan residuals. **Why it might help:** Can outperform Student-t in some non-Gaussian regimes. **Why it’s not safe:** Estimation extremely sensitive to modeling assumptions; easy to overfit curvature to noise.

#### 6.4.4 “Jerk-informed” slip classifiers driving hypothesis branching (⚠ medium risk)

Use 2nd/3rd derivatives of accel (ḟ, f̈, f⃛) to detect wheel slip, impacts, nonrigid mounting; then branch hypotheses with different motion models/noise. **Why it might help:** Principled trigger for MHT branching. **Why it’s risky:** Numerical differentiation is noisy; false positives create hypothesis explosion.

---

### 6.5 Implementation checklist (mapped to pipeline steps)

| Step | Change |
|------|--------|
| **Step 2 (PredictDiffusion)** | Make Q anisotropic for planar vehicle: constrain z, optionally v_z. If keeping SE(3), enforce a gauge (z prior or v_z = 0 soft factor). |
| **Step 3 (DeskewConstantTwist)** | Use odom twist as alternative or complementary deskew twist when IMU is inconsistent. In MHT: branch via **soft responsibilities** (no hard thresholds) based on IMU↔odom twist disagreement. |
| **Step 4.5 (BinMotionSmearCovariance)** | **Future:** Replace per-point deskew with bin-level motion marginalization: compute `Σ_motion,b` from per-bin time moments (`Σ Δt^k`) and point moments (`Σ p pᵀ`) using 2nd-order rotational smear; add into bin covariance in info-form evidence (see §5.8). |
| **Step 8 (PlanarTranslationEvidence)** | **DONE:** planarized z precision (self-adaptive) in LiDAR translation evidence (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/matrix_fisher_evidence.py:502`). |
| **Step 9 (Evidence)** | **DONE:** odom twist factors, time-resolved tilt evidence, and planar priors are present. **Remaining:** add explicit cross-sensor consistency likelihoods (beyond diagnostics) and use IMU message covariances / LiDAR intensity. |
| **Steps 10–11 (α + additive fusion)** | Keep additive fusion; compute α per-subspace (translation vs rotation) so one weak axis does not nuke everything; replace PSD projection “repair” with robust weighting upstream (Student-t weights per factor/per bin). |
| **Step 13 (PoseCovInflationPushforward + map update)** | **DONE:** map update forces `t_hat[2]=0` (planar map z) to avoid z feedback (see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/map_update.py:104`). |
| **Hypothesis layer (new, wraps Steps 2–13)** | Per scan: run Steps 2–13 per hypothesis; compute log-evidence increment; normalize weights; prune/merge to keep B bounded; optionally delay map updates until a hypothesis survives K steps (prevents map poisoning). |

---

### 6.6 Recommended ROI sequence

1. **Planarize z** — **DONE** (planar priors + planar translation + planar map z).
2. **Use odom twist** — **DONE** (velocity/yaw-rate/pose–twist consistency factors).
3. **Fix accel evidence** — **DONE** (time-resolved, transport-consistency-weighted tilt evidence).
4. Add **MHT with strict caps and delayed map commit** (future; requires scoring/prune/merge/branching semantics beyond the current fixed K_HYP container).
5. Next ROI after MHT: consume IMU message covariances/orientation, use LiDAR intensity, and add explicit cross-sensor consistency likelihoods (beyond diagnostics).

Everything “Monge–Ampère / higher-order IG” becomes meaningful only after 1–3 stop the current runaway feedback loops.

**Patch plan:** A literal patch plan keyed to code entrypoints (e.g. `planar_translation_evidence`, `imu_vmf_gravity_evidence_time_resolved`, `odom_quadratic_evidence`, fusion operators)—including which blocks in the 22D state each new factor writes into, and how to compute log-evidence per hypothesis from (L, h)—can be derived from this section and the pipeline trace.
