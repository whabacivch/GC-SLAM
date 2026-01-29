# Trace: Trajectory Export and Ground Truth (Frame Verification)

This doc traces **where the estimated trajectory is written and in which frame** (a), and **how the ground-truth file is produced and used** (b), to support debugging the ~180° roll offset when LiDAR Z convention is already confirmed as Z-up.

**References:** `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`, `docs/PIPELINE_DESIGN_GAPS.md` §5.5, `tools/run_and_evaluate_gc.sh`, `tools/align_ground_truth.py`, `tools/evaluate_slam.py`. **M3DGR official:** [M3DGR README](https://github.com/sjtuyinjie/M3DGR?tab=readme-ov-file) (IROS 2025 dataset).

---

## Summary: What we did, why we did it, and what stats led us to do it

### What stats led us to believe something was wrong

After running the GC pipeline and evaluating against M3DGR ground truth (Dynamic01.txt), we saw:

- **ATE rotation RMSE ~171°** with very low variance (std ~6°).
- **Per-axis rotation:** roll RMSE **~170°**, pitch ~12°, yaw ~98°. So the error was almost entirely **roll** (rotation about X).
- The evaluator flagged **`constant_rotation_offset_suspected`**: mean relative rotation ~172°, std ~6°, with diagnostic note *"Likely frame mismatch (e.g., base vs IMU/LiDAR) or axis convention flip."*
- **ATE translation RMSE ~51 m** — large, but the dominant, *consistent* signal was the rotation: it looked like a **fixed** frame offset, not drift.

**Note:** These stats were from a **legacy pipeline run** (pre‑planar translation/priors and pre‑odom‑twist evidence). This doc focuses on **frame semantics** and GT alignment; rerun evaluation for current metrics.

That pattern (near-constant ~180° roll, low variance) suggested we were comparing two trajectories in **different frames** (e.g. one frame rotated ~180° about X relative to the other), not that our SLAM was diverging.

### Why we did it (hypothesis)

We ruled out:

1. **LiDAR Z-down vs Z-up** — We ran `tools/diagnose_coordinate_frames.py` on the bag; it reported **LiDAR Z-up** and recommended identity `T_base_lidar` rotation. So the ~180° was not from a missing livox Z-flip.
2. **Wrong export direction** — We traced the backend: it exports **T_world←base_footprint** (pose of body in world) in standard TUM form, with no inversion. So the bug was not “we’re writing T_body←world.”

The remaining hypothesis: **our “body” and the ground truth “body” are not the same frame.** Our pipeline uses **base_footprint** from the odom message (`child_frame_id`) — i.e. the **wheel** frame (where wheel odometry is reported). M3DGR provides a calibration file that includes **body_T_wheel** (T_body←wheel): the transform from **wheel** to **body** (camera_imu). If the dataset’s ground truth is in **body** (camera_imu) frame — e.g. from mocap tracking that frame — then we were comparing **T_world←wheel** (our estimate) to **T_world←body** (GT). Those differ by a constant transform; the relative rotation between them is **inv(R_body←wheel)**, which can have a large roll component. So we decided to **transform our estimate into body frame** before evaluation and see if the rotation error dropped.

### What we did

1. **Stored the M3DGR wheel→body transform**  
   We took **body_T_wheel** (T_body←wheel) from the official M3DGR **calibration.md** (section `# wheel 2 camera_imu`) and:
   - Saved a copy in **`config/m3dgr_body_T_wheel.yaml`** (plain YAML, 16 numbers row-major 4×4).
   - Allowed the transform script to load either that file or **calibration.md** directly (parsing the `!!opencv-matrix` data block without OpenCV).

2. **Transformed our estimated trajectory into body frame**  
   For each pose **T_world_wheel** in our TUM file we computed:
   - **T_world_body = T_world_wheel @ inv(body_T_wheel)**  
   so the output TUM is **T_world←body** (pose of body in world), same semantic as the GT file.

3. **Re-ran evaluation**  
   We aligned GT timestamps to the body-frame estimate (time-only, as before) and ran the same ATE/RPE evaluation (evo, SE(3) Umeyama, scale-fixed). So the only change was: **estimate in body frame** vs **estimate in wheel frame**.

### Result

- **Before (wheel-frame estimate vs GT):** ATE rotation RMSE ~171°, roll RMSE ~170°.
- **After (body-frame estimate vs GT):** ATE rotation RMSE ~117°, roll RMSE ~66°.

So **transforming to body frame reduced rotation error** (especially roll), which supports the hypothesis that GT is in body (camera_imu) frame and we were previously comparing wheel to body. Rotation and translation errors are still large (117° and ~51 m), so either the exact **body_T_wheel** or world/time alignment is not perfect, or there is additional drift — but the **frame correction** (wheel → body) was justified by the stats and improved the metrics.

**Tools added:** `tools/transform_estimate_to_body_frame.py` (and `config/m3dgr_body_T_wheel.yaml`). Use when comparing to M3DGR GT that is in body frame.

### Why we're still off so much (117° rotation, ~51 m translation)

After the body-frame correction, rotation improved (171° → 117°, roll 170° → 66°) but errors remain large. Plausible reasons:

1. **Residual frame / calibration** — The **body_T_wheel** we use is from M3DGR calibration.md; it might be for a slightly different robot config or the GT might be in a frame that is only approximately body (e.g. a different link in the chain). A small error in the transform or in which frame GT actually is would leave a residual constant rotation.
2. **Real SLAM drift** — Translation RMSE ~51 m is very large. Part of the error is likely **actual estimation drift** (scale, integration error, map–scan feedback, etc.), not just frame. So even with perfect frame alignment we would see non-zero ATE. See `docs/PIPELINE_DESIGN_GAPS.md` for **current** gaps; the z‑drift mechanism documented in `docs/TRACE_Z_EVIDENCE_AND_TRAJECTORY.md` is legacy and now mitigated by planar translation/priors/map‑z fix.
3. **Time alignment** — We align GT to estimate by matching the **first** timestamp. If there is a constant time offset or different sampling (GT ~297 Hz, estimate ~0.5 Hz), the sync is only approximate; comparing sparse estimate to dense GT can inflate errors.
4. **Umeyama absorbs only one constant SE(3)** — The evaluator finds the best rigid transform (rotation + translation, no scale) to align estimate to GT. If the error is **not** constant (e.g. drift over time), that residual will still show up as large ATE/RPE.

So: the wheel→body step fixed a **frame mismatch** (and reduced rotation error). The remaining error is a mix of possible residual frame/calibration issues, real drift, and alignment/sync limitations.

### Extrinsics: are IMU, LiDAR, and gyro correct?

**Short answer:** Yes, **for the choice of base frame we use**. Our pipeline uses **base_footprint** (from odom `child_frame_id`) as "base" — that is the **wheel** frame, not M3DGR's **body** (camera_imu). Our extrinsics are defined and used **relative to wheel**.

- **LiDAR (T_base_lidar):** We use **rotation identity** (Z-up confirmed by `diagnose_coordinate_frames.py`) and **translation [-0.011, 0, 0.778]** m. So we assume livox_frame is Z-up and orientation-aligned with base_footprint; only the origin offset is applied. That is **correct for wheel as base** given the diagnostic. We do **not** use M3DGR's **T_camera_imu←mid360** (which has a large rotation) in the pipeline, because our base is wheel, not body. If we ever switched to "body" as base, we would need to use the calibration chain (e.g. wheel←body from body_T_wheel, then body←mid360 from calibration).
- **IMU (T_base_imu):** We use **translation [0, 0, 0]** (co-located with LiDAR in the Livox unit) and **rotation ~[-0.015586, 0.489293, 0]** rad (~28°) so that gravity aligns with +Z in base. That value was set from gravity alignment (diagnostic reported ~25° misalignment; we use ~28°). So **accel_base = R_base_imu @ accel_imu** is correct for **wheel** as base. We do **not** use M3DGR's body←mid360_imu in the pipeline; we use wheel as base and this empirical rotation.
- **Gyro:** The same **T_base_imu** is applied to gyro before preintegration (IMU data are rotated to base frame). So gyro is **consistent with IMU** and correct for wheel as base.

**Summary:** LiDAR, IMU, and gyro extrinsics are **internally consistent and correct for base_footprint (wheel) as base**. They are **not** taken from the M3DGR calibration file's body←sensor transforms; we use wheel as base with identity LiDAR orientation and ~28° IMU rotation (gravity-aligned). The only place we use M3DGR calibration (body_T_wheel) is **at evaluation time**: we transform our **wheel-frame** estimate to **body frame** so we can compare fairly to GT that is in body frame. No change was made to pipeline extrinsics (LiDAR/IMU/gyro); they remain as above.

---

## (a) Where the trajectory is written and in which frame

### 1. Source of the pose

- **Location:** `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
- **Flow:**
  - After each successful pipeline run, the backend calls `combined_belief.mean_world_pose()` (line 899).
  - `mean_world_pose()` is an alias for `world_pose()` in `fl_slam_poc/common/belief.py:432–439`.
  - **Semantics:** `world_pose(eps_lift) = X_anchor @ Exp(δξ_pose)` (belief.py:417–430). The state is defined as **pose of the body in the world**: `X = T_world←body`. So the returned 6D pose is **T_world←body** (translation = body origin in world, rotation = body-to-world).
  - **Body frame:** The pipeline and odom ingest use `base_footprint` as the body frame (launch param `base_frame`, odom `child_frame_id`). So the exported pose is **T_world←base_footprint**.

### 2. Conversion to ROS / TUM

- **Location:** `backend_node.py:914–966` (`_publish_state_from_pose`).
  - `pose_6d` is split into `rotvec` and `trans` via `se3_to_rotvec_trans(pose_6d)`.
  - Quaternion is computed with `Rotation.from_rotvec(rotvec).as_quat()` → **xyzw** (scipy convention).
  - **TUM write** (lines 962–966):
    ```text
    stamp_sec  trans[0]  trans[1]  trans[2]  quat[0]  quat[1]  quat[2]  quat[3]
    ```
    i.e. `timestamp x y z qx qy qz qw` (TUM standard).
  - **Interpretation:** Position = origin of body (base_footprint) in world. Orientation = quaternion that rotates from body to world (same as ROS Odometry: `p_parent = R_parent_child @ p_child + t_parent_child` with parent=world, child=body). So the file is **T_world←base_footprint** in standard TUM form.

### 3. World frame for the estimate

- Odom is ingested as `T_{header.frame_id←child_frame_id}` (odom_combined←base_footprint) and **not** inverted (backend_node.py:508–521).
- The backend then makes poses **relative to the first odom**: `odom_relative = first_odom^{-1} ∘ odom_absolute` (backend_node.py:535–539). So the effective “world” for the SLAM state is **first-odom-at-origin** (same orientation and scale as odom_combined, origin at first odom pose).
- The exported TUM trajectory is therefore: **pose of base_footprint in this odom-derived world**, with first pose near identity.

### 4. Summary (a)

| Item | Value |
|------|--------|
| **Pose semantics** | T_world←base_footprint (body pose in world) |
| **World** | odom_combined with first odom at origin |
| **TUM format** | timestamp x y z qx qy qz qw (xyzw) |
| **File** | `trajectory_export_path` (e.g. `/tmp/gc_slam_trajectory.tum`) |

**Conclusion:** The code exports the intended frame (T_world←base_footprint). There is no inversion or extra transform at export. If a ~180° roll offset appears vs ground truth, it is not caused by exporting the wrong direction (e.g. T_body←world).

---

## (b) How the ground-truth file is produced and used

### 1. Source file and path

- **Path:** `rosbags/m3dgr/Dynamic01.txt` (set in `tools/run_and_evaluate_gc.sh` as `GT_FILE`).
- **Origin:** Supplied with the M3DGR dataset (see `docs/datasets/M3DGR_STATUS.md`: “Dynamic01.txt # Ground truth trajectory (4.2 MB)”). It is **not** generated by this repo; it is either provided by the dataset authors or exported elsewhere from the bag.
- **Bag GT topic (documented):** `docs/BAG_TOPICS_AND_USAGE.md` states that ground truth for evaluation comes from **`/vrpn_client_node/UGV/pose`** (OptiTrack motion capture, 360 Hz). So `Dynamic01.txt` is **likely** a TUM export of that topic (or an equivalent mocap trajectory). We do not have in-repo proof of how exactly `Dynamic01.txt` was produced (script or tool).
- **M3DGR official README** ([github.com/sjtuyinjie/M3DGR](https://github.com/sjtuyinjie/M3DGR?tab=readme-ov-file)): For **Dynamic01** (Indoor, Visual Challenge, "Dynamic Person"), the dataset provides separate Rosbag and GT download links. The README states: *"If the GT is obtained by RTK/Mocap, you can directly use evo to evaluate: `evo_ape tum GTDynamic01.txt Dynamic01.txt -ap`"* and *"You can quickly get the trajectory in TUM format through the **TF tree method**."* So GT is in TUM format and comes from RTK/Mocap; trajectory can be obtained via the TF tree (e.g. from `/vrpn_client_node/UGV/pose` or equivalent). The README does **not** specify the exact frame (world vs body), axis convention (Z-up vs Z-down), or quaternion order for the GT file; if a ~180° roll persists after alignment, check M3DGR calibration/export docs or their baseline code for frame and convention.

### 2. Format and frame (documented vs assumed)

- **Format:** TUM: `timestamp x y z qx qy qz qw` (same as the estimated trajectory). `align_ground_truth.py` and `evaluate_slam.py` assume 8 columns per line.
- **Frame:** Not specified in this repo. Typical VRPN/OptiTrack setups provide **pose of the tracked body in the motion-capture world frame**. If the UGV is tracked at base_footprint (or similar), then the file would be **T_mocap←body**. For evaluation we assume both GT and estimate are “pose of body in world”; the **world** differs (mocap vs odom-first-origin). The evaluation script uses **SE(3) Umeyama alignment** (scale-fixed) to align estimate to GT, so a constant translation and rotation between the two worlds is absorbed by the alignment—**except** if the two use different **axis or orientation conventions** (e.g. one Z-up and one Z-down, or one T_world←body and the other T_body←world). A consistent ~180° roll then remains after alignment.

### 3. Alignment and evaluation flow

- **Script:** `tools/run_and_evaluate_gc.sh`.
  1. **Align ground truth** (lines 337–341):  
     `align_ground_truth.py "$GT_FILE" "$EST_FILE" "$GT_ALIGNED"`  
     This script **only shifts GT timestamps** so that the first GT timestamp matches the first estimate timestamp (GT is absolute UNIX time, estimate is simulation time near 0). It does **not** change poses or frames.
  2. **Evaluate** (lines 355–361):  
     `evaluate_slam.py "$GT_ALIGNED" "$EST_FILE" "$RESULTS_DIR" ...`  
     Loads both as TUM trajectories, syncs by timestamp, runs **SE(3) Umeyama** (scale-fixed), then computes ATE/RPE and per-axis rotation (Euler XYZ from relative rotation est_inv*gt).

### 4. Summary (b)

| Item | Value |
|------|--------|
| **GT file** | `rosbags/m3dgr/Dynamic01.txt` (from M3DGR dataset) |
| **Likely source** | `/vrpn_client_node/UGV/pose` (OptiTrack) or equivalent |
| **Frame (assumed)** | T_mocap←body (body = UGV/base); **not** verified in-repo |
| **Alignment** | Time-only via `align_ground_truth.py`; no frame or pose transform |
| **Evaluation** | SE(3) Umeyama + ATE/RPE; per-axis rotation = Euler XYZ(est_inv*gt) |

**Conclusion:** The ~180° roll could be explained if (1) M3DGR GT uses a different world convention (e.g. Y-up or Z-down) than our odom-derived Z-up world, or (2) the GT was exported with a different pose direction or quaternion convention. **Recommended:** Check M3DGR dataset documentation or the tool used to generate `Dynamic01.txt` for the exact frame and convention; if possible, re-export GT from `/vrpn_client_node/UGV/pose` with a known T_world←body and xyzw quaternion.

---

## Cross-check with §5.5

- **LiDAR Z:** Diagnostic reported Z-up; `T_base_lidar` rotation = identity is correct. So the ~180° roll is **not** from livox Z-down.
- **Export frame:** Backend exports T_world←base_footprint; no inversion at write.
- **GT frame:** Unknown in-repo; GT may be in a different world or convention. Next step: confirm M3DGR GT frame and, if needed, add a documented transform or re-export step so GT and estimate share the same convention.

---

## M3DGR calibration: wheel vs body (camera_imu)

M3DGR provides a calibration file that includes:

- **Frames:** camera, camera_imu, avia (Livox Avia), mid360 (Livox MID-360), mid360_imu, **wheel**.
- **body_T_wheel** (4×4): transform **T_body←wheel**. In M3DGR baselines, "body" is typically the main body/IMU frame (e.g. camera_imu). The **wheel** frame is the odometry frame (wheel encoder).
- **mid360 2 camera_imu** / **camera_imu 2 mid360**: T_camera_imu←mid360 has a **large rotation** (R is not identity); mid360 is rotated relative to camera_imu.
- **mid360 2 mid360_imu**: small translation, R = I (LiDAR and IMU in the MID-360 unit are aligned).

**Implication for our pipeline:**

- We use **base_footprint** as "base", which matches `/odom.child_frame_id` in the bag — i.e. the **wheel** frame (where wheel odom is reported).
- Our estimate is therefore **T_world←wheel** (pose of wheel in world).
- If M3DGR ground truth is exported from mocap (or TF tree) in **body** (camera_imu) frame, then GT is **T_world←body** (pose of body in world).
- Wheel and body differ by **T_body←wheel** from calibration. So:
  - **T_world←body = T_world←wheel @ T_wheel←body = T_world←wheel @ inv(T_body←wheel)**.
  - The **constant** relative rotation between our trajectory and GT is **R_wheel←body = inv(R_body←wheel)**. If that rotation has a large roll (e.g. ~180° about X), it would explain the observed ~170° roll ATE.

**Fix (if GT is in body frame):** Before evaluation, transform our trajectory into body frame so both are "pose of body in world":

- For each estimated pose T_world_wheel (4×4 or R,t), compute **T_world_body = T_world_wheel @ inv(T_body_wheel)** (using the calibration `body_T_wheel`).
- Export that as the TUM trajectory and run evo against GT as usual. If the rotation error drops to a few degrees, the frame mismatch was wheel vs body.

**Calibration and script:** The transform **body_T_wheel** (T_body←wheel) comes from M3DGR **calibration.md** (section `# wheel 2 camera_imu`, `body_T_wheel: !!opencv-matrix`). We store a copy in **`config/m3dgr_body_T_wheel.yaml`** (plain YAML, no OpenCV). The script can load either:

- **Default:** `config/m3dgr_body_T_wheel.yaml`
- **M3DGR calibration.md:** `--calib /path/to/calibration.md` (parses the !!opencv-matrix data block; no OpenCV dependency)

To convert our estimate to body frame before evaluation:

```bash
# Using bundled config
python tools/transform_estimate_to_body_frame.py /tmp/gc_slam_trajectory.tum /tmp/gc_slam_trajectory_body.tum

# Using M3DGR calibration.md directly
python tools/transform_estimate_to_body_frame.py /tmp/gc_slam_trajectory.tum /tmp/gc_slam_trajectory_body.tum --calib /home/will/Downloads/calibration.md
```

Then evaluate using the body-frame trajectory and the same GT file (e.g. `Dynamic01.txt`). If rotation error drops to a few degrees, the mismatch was wheel vs body. The script uses `T_world_body = T_world_wheel @ inv(body_T_wheel)` per pose.
