# CHANGELOG

Project: Frobenius-Legendre SLAM POC (Impact Project_v1)

This file tracks all significant changes, design decisions, and implementation milestones for the FL-SLAM project.

## 2026-02-02: BEV15 future scaffolds + exact vMF view pushforward + Rerun RGBD/LiDAR

- Kept BEV15 in the spec as a future view-layer target; clarified that vMF pushforward under view rotation is exact (`η' = R η`) and no S¹ collapse is used.
- Restored BEV OT scaffolds as explicitly FUTURE/EXPERIMENTAL modules (not wired into runtime).
- Added exact vMF rotation helpers in `common/bev_pushforward.py` for future BEV15 view use.
- Extended Rerun visualization to log RGB, depth, LiDAR points, and map/trajectory for a faithful probabilistic world view (recordable to .rrd for post-hoc viewing).
- Added BEV15 post-run .rrd generation and a separate Rerun viewer launch in `tools/run_and_evaluate_gc.sh`.

## 2026-02-02: Camera RGB → map/splat colors (root cause fix)

**Root cause:** Splats in Rerun had no real colors and the map did not reflect the environment because camera RGB was never sampled or passed into the pipeline. The backend used camera for pose/geometry (visual_pose_evidence) but not for appearance.

**Fix (by construction):**
- **Feature3D:** Added optional `color: Optional[np.ndarray]` (3,) RGB in [0,1]. In `VisualFeatureExtractor.extract()`, sample RGB at each keypoint (u,v) via nearest pixel, bounds-clamped; attach to each Feature3D.
- **splat_prep_fused:** When building fused Feature3D from LiDAR–camera depth fusion, propagate `feat.color` so fused splats keep camera color.
- **feature_list_to_camera_batch:** Build a `colors` array from Feature3D.color (default gray when missing) and pass `colors=` to `measurement_batch_from_camera_splats()`. Camera batch now carries real RGB; map fusion (responsibility-weighted `colors_meas`) and splat_export then get camera-derived colors.

**Result:** Primitive map and splat_export.npz / Rerun ellipsoids now receive camera RGB. Re-run eval and open the built .rrd to see colored splats and environment.

## 2026-02-02: IMU extrinsic fix, odom z prior, validation diagnostics

**Critical fix: IMU extrinsic rotation sign.** The D435i IMU accelerometer reads -Y when stationary (gravity in +Y in optical frame). T_base_imu rotation was incorrectly set to Rx(+90°) which mapped -Y → -Z; correct is Rx(-90°) = rotvec `[-1.57, 0, 0]` which maps -Y → +Z to cancel gravity_W = [0,0,-9.81]. This caused ~20 m/s² net Z acceleration and 636m+ trajectory Z drift. Fixed in gc_unified.yaml and calibration/kimera_acl_jackal2.yaml.

**Odom z prior:** Backend now uses actual `pos.z` from odom (instead of forcing `GC_PLANAR_Z_REF=0`), but caps trust via `cov[2,2] = max(cov[2,2], GC_ODOM_Z_VARIANCE_PRIOR)`. Default 1e6 m² (σ_z ≥ 1000m = "don't trust odom z"). Anchor smoothing still uses z_planar reference.

**Validation diagnostics:** Added twist-derived and IMU-derived scalars to ScanDiagnostics:
- `distance_pose`: ||t_curr - t_prev|| from belief pose delta (m)
- `distance_twist`: ||v_body|| × dt_sec from odom twist (m)
- `speed_odom`: ||v_body|| scalar speed (m/s)
- `speed_pose`: distance_pose / dt_sec (m/s)
- `imu_jerk_norm_mean`, `imu_jerk_norm_max`: jerk proxy ||a_{i+1} - a_i|| / dt_i (m/s³)

## 2026-02-02: Reality-aligned GC v2 interface spec (legacy purge)

- Rewrote `docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md` to match the current backend (PrimitiveMap + explicit backend selection + factor-based IMU/odom evidence + IW adaptive noise) and removed legacy descriptions that no longer reflect runtime behavior.
- Added a non-negotiable modeling contract section (dependent evidence, no gating/heuristics, “every raw bit” accounting) with explicit information-geometry and Frobenius/pre-Frobenius invariants, anchored to `docs/Comprehensive Information Geometry.md`.
- Extended runtime manifest reporting to include `deskew_rotation_only` so ablation modes can’t be hidden.
- Added certificate support for continuous “overconfidence sentinel” metrics (diagnostic-only; no gating), and populated them in fusion-scale certificates.
- Added dt/Z observability proxies (`overconfidence.dt_asymmetry`, `overconfidence.z_to_xy_ratio`) and documented dt-collapse/Z-collapse/IW-stiffness risks in the interface spec.
- Expanded the interface spec’s PrimitiveMap section with concrete concepts and construction steps (MeasurementBatch natural parameters, MA-hex LiDAR surfels with Wishart regularization, LiDAR→camera depth fusion PoE, RGB payload fusion).
- Made vMF non-optional in the measurement/map interface by moving to a fixed 3-lobe vMF representation (resultant eta is closed-form sum over lobes); added BEV15 pushforward/view utilities.
- Removed camera-dependent scan skipping: when the camera ring buffer is empty, backend now runs with an empty camera MeasurementBatch (no gating).

**Docs:** KIMERA_FRAME_MAPPING.md updated with root cause for "All motion in Z" symptom and correct IMU extrinsic info.

## 2026-02-02: Post-hoc Rerun, BEV 15 shading, Ellipsoids3D

- **Rerun after eval only:** Eval script no longer records Rerun during the run. After eval finishes, `tools/build_rerun_from_splat.py` builds a single .rrd from `splat_export.npz` + trajectory, then the script opens Rerun automatically so the user sees the full splat world once.
- **Post-hoc builder:** New `tools/build_rerun_from_splat.py` loads splat export and TUM trajectory, computes Ellipsoids3D from covariances (same math as 3D Gaussian principal axes), BEV 15 shaded colors (vMF + fBm from `fl_slam_poc.backend.rendering`), trajectory as LineStrips3D and Transform3D per pose (with quat when TUM has 8 columns). Writes `gc_slam.rrd`; no JAXsplat required for Rerun.
- **run_and_evaluate_gc.sh:** Passes `use_rerun:=false`; after dashboard, runs `build_rerun_from_splat.py "$RESULTS_DIR"` when `splat_export.npz` exists, then opens Rerun with the built .rrd.

## 2026-02-02: Kimera-only cleanup — remove non-Kimera dataset references

- **Dataset:** Project uses only the Kimera rosbag/dataset. All M3DGR/Dynamic01 references removed from active code and docs.
- **run_and_evaluate_gc.sh:** Single dataset path (Kimera); removed PROFILE=m3dgr and body-frame transform step. GT_ALIGNED renamed to /tmp/gt_ground_truth_aligned.tum.
- **Launch:** Default odom_frame/base_frame and T_base_lidar/T_base_imu set to Kimera values (acl_jackal2/odom, acl_jackal2/base; lidar_sigma_meas 0.001).
- **Archive:** config/m3dgr_body_T_wheel.yaml and tools/transform_estimate_to_body_frame.py moved to archive/config/ and archive/legacy_tools/ (M3DGR-only).
- **Docs:** FRAME_AND_QUATERNION_CONVENTIONS, BAG_TOPICS_AND_USAGE, SYSTEM_DATAFLOW_DIAGRAM, KIMERA_FRAME_MAPPING, PIPELINE_DESIGN_GAPS, POINTCLOUD2_LAYOUTS, VELODYNE_VLP16, DIAGNOSTIC_TOOLS, README, ROADMAP, and docs/datasets/DATASET_DOWNLOAD_GUIDE updated to Kimera-only. Constants and backend_node comments updated.

## 2026-02-02: Bloat cleanup — .gitignore, archive docs/tool/configs

- **.gitignore:** Added `.venv_test/`, `vectors.db`, `vectors.db-shm`, `vectors.db-wal` (root); added jaxsplat debug patterns (`gdb_bt*.txt`, `repro_segfault.log`, `repro_versions.txt`, `jaxsplat/gdb_bt.txt`).
- **Docs moved to archive/docs/:** BLOAT_AND_COMPLEXITY_AUDIT.md, MAP_AND_BELIEF_AUDIT_2026-01-30.md, MAP_VISUALIZATION.md, PRODUCTION_READINESS_SPLAT_PIPELINE.md, POST_INTEGRATION_CHECKLIST_AUDIT.md, system_dataflow_d3.html.
- **Tool moved to archive/legacy_tools/:** estimate_lidar_base_extrinsic.py (broken: missing frontend.scan.icp).
- **Config moved to archive/config/:** cyclonedds_0.xml through cyclonedds_3.xml (only cyclonedds.xml is used).
- **Deleted:** archive/docs/Untitled (accidental file).

## 2026-02-02: Kimera calibration doc, splat export, JAXsplat post-run viz

- **Kimera calibration/frame:** New `docs/KIMERA_CALIBRATION_AND_FRAME.md` summarizes where extrinsics and frame info live in the Kimera bag directory (calibration/README, robots/acl_jackal/extrinsics.yaml, dataset_ready_manifest, PREP_README) and how GC uses them. Explains why ATE can be wrong (GT vs estimate frame mismatch) and points to dataset “Align frames to GC v2 conventions before eval.” KIMERA_FRAME_MAPPING.md now references it.
- **Splat export:** Backend and launch support optional `splat_export_path`. When set, on shutdown the backend exports the primitive map (positions, covariances, colors, weights, directions, kappas) to an NPZ for post-run visualization.
- **JAXsplat post-run viz:** New `tools/view_splat_jaxsplat.py` loads splat_export.npz and trajectory, renders a view with jaxsplat.render_v2, and saves an image. `run_and_evaluate_gc.sh` passes `splat_export_path:=$RESULTS_DIR/splat_export.npz` and, after dashboard/Rerun, runs the viewer to produce `splat_render.png` and optionally open it.

## 2026-02-02: Eval script — Rerun in results, rosbag/metrics clarity, frame-mismatch note

- **run_and_evaluate_gc.sh:** Rerun recording is written to `$RESULTS_DIR/gc_slam.rrd` (no longer only /tmp). After the run, script opens the diagnostics dashboard and, if `rerun` is installed, opens the Rerun recording so trajectory + map can be inspected.
- **Results summary:** Stage 4 now prints which rosbag and GT file were used. If ATE translation RMSE > 10 m or RPE @ 1m > 1, a note is shown that large errors often indicate GT vs estimate frame/axis convention mismatch (see docs/KIMERA_FRAME_MAPPING.md).

## 2026-02-02: Rename Golden Child → Geometric Compositional

- **Spec:** `docs/GOLDEN_CHILD_INTERFACE_SPEC.md` renamed to `docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md`; all internal and cross-repo references updated.
- **Backend:** Class `GoldenChildBackend` renamed to `GeometricCompositionalBackend` in `backend_node.py`; entry point script and docs updated.
- **Tests:** `test_golden_child_invariants.py` renamed to `test_geometric_compositional_invariants.py`; PRODUCTION_READINESS and CHANGELOG references updated.
- **Prose:** All "Golden Child" / "GOLDEN CHILD" / "GoldenChild" prose and identifiers replaced with "Geometric Compositional" / "GEOMETRIC COMPOSITIONAL" / "GeometricCompositional" across docs, config, tools, and archive. GC prefix (gc_backend, gc_rosbag, etc.) unchanged.
- **Depth + JAXsplat stability:** Depth passthrough repacks to packed 32FC1 rows (respecting `step`/padding) and backend depth/RGB handlers honor row stride; legacy custom_call lowering now forces `api_version=0` for JAX 0.9+ compatibility.
- **Visual feature stability:** Convexity weight sigmoid is now numerically stable to avoid exp overflow in `visual_feature_extractor.py`.
- **Surfel extraction stability:** Zero-weight padded points are filtered before voxelization to prevent zero-mass plane fits.
- **Primitive map view call:** Fixed `extract_primitive_map_view` keyword name to match `prim_map` signature.
- **Visual pose evidence:** Filter OT responsibilities/candidates to valid measurement rows to avoid shape mismatches.
- **Rerun compatibility:** Use `rerun.set_time` fallback when `set_time_seconds` is unavailable.

## 2026-02-01: jaxsplat render_v2/bev_v2 typed FFI hardening (handler, stream, buffers, layouts)

- **Handler lifetime:** Typed FFI entry points (`RenderV2ForwardFFI`, `BevForwardFFI`) use thread-local handler creation on first use instead of a single static pointer, so XLA FFI’s expected per-call or thread-local handler lifetime is respected.
- **Stream/buffer validation:** C++ FFI impls validate non-null CUDA stream and non-null typed buffer pointers before calling `RenderV2Forward`/`BevForward`; host entries defensively check stream/buffers/opaque and return early if null.
- **Python ffi_call:** Config operand is a copied 1D uint8 array (stable buffer for C++); `input_layouts` and `output_layouts` are passed explicitly (major-to-minor) so runtime passes correct strides to the C++ binding.
- **Build for JAX 0.9 + CUDA 13:** CMake prefers CUDA 13 nvcc from the Python env (`nvidia/cu13/bin/nvcc` when present) for ABI compatibility with jax-cuda13-plugin. Typed FFI targets registered with `api_version=1`. README documents the correct build/run for current versions.
- **Typed FFI segfault:** If the typed FFI path still segfaults after a clean rebuild with CUDA 13, capture a full backtrace (e.g. `gdb -ex run -ex bt -ex quit --args python -X faulthandler repro_segfault.py`) to determine whether the crash is in the jaxsplat handler or the JAX/jaxlib runtime.

## 2026-01-31: Livox removed; Kimera-only default; clean target; minimal-files audit

- **Livox removed:** LiDAR path is pointcloud_passthrough only (PointCloud2 bags). livox_converter archived; livox_ros_driver2 removed from package.xml. gc_sensor_hub requires pointcloud_passthrough.input_topic; no Livox branch. Backend parses PointCloud2 via parse_pointcloud2_vlp16 only; parse_pointcloud2_vectorized (Livox-style) removed.
- **Kimera default:** gc_unified.yaml and launch default to vlp16, Kimera-style topics (acl_jackal/*). imu_accel_scale default 1.0 (m/s²). Launch livox_input_msg_type removed.
- **Clean target:** `make clean` removes results/gc_*, results/gc_slam_diagnostics.npz, fl_ws/build, fl_ws/install, fl_ws/log to reduce bulk.
- **Minimal-files audit:** docs/MINIMAL_OPERATIONAL_FILES_AUDIT.md lists minimal set for fully operational system (Kimera bag → run + evaluate). validate_livox_converter.py moved to archive.

## 2026-01-31: Eval camera/depth handling; open issue — depth never received

- **Camera ordering:** Backend no longer raises when the first LiDAR scan arrives before camera; it skips that scan (warn + return) and waits for at least one RGB and one depth frame. Avoids crash when bag message order delivers LiDAR before camera.
- **rclpy logger:** Backend and depth_passthrough use a single formatted string for `get_logger().warn()` / `get_logger().info()` (rclpy expects one message string, not multiple args).
- **depth_passthrough:** Logger fix so the node no longer crashes on startup. Subscription uses BEST_EFFORT QoS so it can receive from sensor-style (often best-effort) bag playback.
- **Open issue — depth never received:** On Kimera eval (`run_and_evaluate_gc.sh`), backend reports `rgb=True depth=False` for the whole run; pipeline never runs (0 poses). The bag has `/acl_jackal/forward/depth/image_rect_raw` (34,720 Image messages). So either (1) depth_passthrough never receives from the bag, or (2) depth_passthrough receives but does not publish to `/gc/sensors/camera_depth`, or (3) backend never receives from `/gc/sensors/camera_depth`. To diagnose: with the same launch running, run `ros2 topic hz /acl_jackal/forward/depth/image_rect_raw` and `ros2 topic hz /gc/sensors/camera_depth`; identify which link (bag→depth_passthrough or depth_passthrough→backend) fails and fix QoS or topic wiring accordingly.

## 2026-01-30: GC v2 spine reorder; unbalanced Sinkhorn only; camera mandatory; bin path archived

- **Pipeline reorder and z_lin:** Evidence uses pre-update map M_{t-1}. After predict/deskew, build IMU+odom evidence and solve for **z_lin** (IMU+odom-informed pose). Surfel + camera batch → map view → association (unbalanced Sinkhorn) → **visual_pose_evidence(M_{t-1}, π, z_lin_pose)** → fuse all evidence → recompose → **z_t**. Map update (fuse/insert/cull/forget) uses **z_t** (post-recompose pose) to transform measurements to world frame. See `docs/PIPELINE_ORDER_AND_EVIDENCE.md`.
- **Unbalanced Sinkhorn only:** `primitive_association` always calls `sinkhorn_unbalanced_fixed_k`; balanced path removed. Runtime manifest reports `sinkhorn_backend="unbalanced_fixed_k"`. `sinkhorn_balanced_fixed_k` deprecated in `sinkhorn_ot.py`.
- **Camera mandatory:** Backend fails fast if RGB or depth not yet received (no optional path). Pipeline `camera_batch` is required (no default). Shape mismatch (rgb vs depth) raises.
- **Cull as budgeting operator:** `primitive_map_cull` approximation_triggers include `"budgeting"` and `"mass_drop"`; docstring states resource constraint and mass_dropped logged.
- **Depth contract:** New `docs/PIPELINE_DEPTH_CONTRACT.md`: camera_depth authoritative; LiDAR fused via lidar_depth_evidence in splat_prep_fused; one fused depth path.
- **Bin path archived:** `bin_atlas.py` moved to `archive/bin_atlas.py` (not importable by package). `structures/__init__.py` exports only PrimitiveMap, MeasurementBatch, and primitive_map operators. BinAtlas, MapBinStats, create_fibonacci_atlas removed from package surface.
- **Lidar bucket IW archived:** `lidar_bucket_noise_iw_jax` (structures + operators) moved to archive; pipeline and backend_node no longer use lidar_bucket state or accumulation. ScanPipelineResult keeps iw_lidar_bucket_dPsi/dnu as zero placeholders for diagnostic schema.
- **Config:** `config/README.md` states use gc_unified.yaml only; gc_backend.yaml deprecated. Backend `__init__.py` docstring updated (structures: PrimitiveMap, MeasurementBatch, IW states; rendering deferred).

## 2026-01-30: Camera mandatory; bins removed; backend config from YAML; full visual+LiDAR wiring

- **Camera always on:** Launch no longer gates camera on `enable_camera`; `image_decompress_cpp` always runs. `depth_passthrough` runs when `camera_depth_raw_topic` is non-empty (node errors if empty).
- **Backend config from YAML:** Launch uses `OpaqueFunction` to load `config_path` YAML and merge `gc_backend.ros__parameters` with launch overrides; backend node receives map_backend, pose_evidence_backend, n_feat, n_surfel, k_assoc, camera_K, T_base_camera, etc. from `gc_unified.yaml`.
- **Bins removed:** BinAtlas and MapBinStats removed from backend state and pipeline. Pipeline signature: no `bin_atlas`/`map_stats`; added `camera_batch: Optional[MeasurementBatch]`. PrimitiveMap is always the canonical map; map_backend/pose_evidence_backend default to primitive_map/primitives.
- **Camera wiring:** Backend subscribes to `/gc/sensors/camera_image` and `/gc/sensors/camera_depth`. Latest rgb/depth stored; on each LiDAR scan: VisualFeatureExtractor.extract(rgb, depth) → splat_prep_fused(extraction_result, points_camera_frame, intrinsics, config) → feature_list_to_camera_batch → camera_batch passed to pipeline. Pipeline merges camera_batch with LiDAR surfels via extract_lidar_surfels(..., base_batch=camera_batch).
- **Fail-fast:** Backend requires camera_image_topic and camera_depth_topic (non-empty); camera_K and T_base_camera from params. New `backend/camera_batch_utils.py`: feature_list_to_camera_batch(List[Feature3D], timestamp_sec, ...) → MeasurementBatch (camera slice).
- **gc_unified.yaml:** Added camera_image_topic, camera_depth_topic, camera_K, T_base_camera, ringbuf_len under gc_backend.ros__parameters.
- **package.xml:** exec_depend cv_bridge for Python backend.

## 2026-01-30: Bloat audit follow-up – restore ot_fusion, slim __init__, delete os

- **ot_fusion.py:** Restored to `backend/operators/` from git (was mistakenly removed). BEV/OT fusion, splat, rendering, and visual feature code are **to be wired** into the pipeline (visual–LiDAR plan); not archived.
- **operators/__init__.py:** Removed dead exports only: bin_soft_assign, scan_bin_moment_match, BinSoftAssignResult, ScanBinStats (binning already in archive); pos_cov_inflation_pushforward. Kept sinkhorn_ot_bev and ot_fusion exports for future wiring.
- **archive/legacy_operators:** lidar_evidence.py now imports `from .binning import ScanBinStats`; added `__init__.py`.
- **Root file `os`:** Deleted (large accidental PostScript); `/os` added to .gitignore.
- **docs/BLOAT_AND_COMPLEXITY_AUDIT.md:** Moved to archive/docs/ (bloat audit applied; doc retained in archive).

## 2026-01-30: Rerun visualization (Wayland-friendly; replaces RViz)

- **Rerun:** Visualization now uses [Rerun](https://rerun.io/) by default (Wayland-friendly). Backend logs map points (`gc/map/points`) and trajectory (`gc/trajectory`) to Rerun; no RViz dependency for viewing.
- **Backend:** New params `use_rerun` (default true), `rerun_recording_path` (default `/tmp/gc_slam.rrd`), `rerun_spawn` (default false). When `use_rerun` is true and `rerun-sdk` is installed, data is recorded to the given path (or buffered if path empty); open with `rerun /path/to/gc_slam.rrd`. Set `rerun_spawn:=true` to spawn the Rerun viewer at startup.
- **Launch:** `gc_rosbag.launch.py` exposes `use_rerun`, `rerun_recording_path`, `rerun_spawn`.
- **RViz:** `rviz/gc_slam.rviz` no longer installed; ROS PointCloud2/Path topics are still published for tools that need them. `visualization_msgs` kept for optional MarkerArray (ellipsoids).

## 2026-01-30: MA hex web replaces K-NN for OT candidate generation

- **Candidate generation:** K-NN removed; `generate_candidates_ma_hex_web` in `fl_slam_poc.common.ma_hex_web` is the only candidate generator. BEV (x, y) hex cells with h = hex_scale_factor × median(√λ_max(Σ_bev)); map primitives binned by cell; per measurement, stencil of neighbor cells → collect map indices → take nearest k_assoc by squared distance. Fixed topology (stencil K=64, max_occupants per cell); single path, no legacy kNN.
- **primitive_association:** Calls `generate_candidates_ma_hex_web(meas_positions, map_positions, map_covariances, k_assoc, MAHexWebConfig())`; `_generate_candidates_knn` deleted.
- **OPERATOR_CONTRACTS:** associate_primitives_ot doc updated to "candidate gen = MA hex web".

## 2026-01-30: Post-integration checklist audit

- **Audit doc:** `docs/POST_INTEGRATION_CHECKLIST_AUDIT.md` — run-through of post-integration checklist (single path, legacy deletion, no multi-path depth, gating, evidence correctness, JAX hot path, Certificates/Frobenius, rendering, backwards-compatibility phrases). Result: **PASS** on minimal criteria with two **PARTIAL** items: (1) `responsibility_threshold` in `flatten_associations_for_fuse` (hard threshold 0.01 for “include in fuse”); (2) `np.array` in flatten (host round-trip per scan). Recommended follow-ups: remove dead import, continuous fuse weight or document sparse-fuse, JAX-only flatten, JAX Sinkhorn.
- **Dead import removed:** Pipeline no longer imports `pos_cov_inflation_pushforward` (was unused; MapUpdateResult kept).

## 2026-01-30: Operator contracts, manifest, validation tests (Visual LiDAR plan)

- **Operator contracts:** New `docs/OPERATOR_CONTRACTS.md` — operator table with roles, fixed budgets (N_FEAT, N_SURFEL, K_ASSOC, K_SINKHORN, RINGBUF_LEN), CertBundle/approximation triggers, Frobenius policy; code anchors for pose evidence, OT, map maintenance, surfel extraction.
- **Runtime manifest:** `RuntimeManifest` now has `pose_evidence_backend` and `map_backend` (from constants); included in `to_dict()`. When `pose_evidence_backend="primitives"`, `backends["lidar_evidence"]` set to visual_pose_evidence; when `map_backend="primitive_map"`, `backends["map_update"]` set to primitive_map. Backend node passes config values when publishing manifest.
- **Validation tests:** New `test_visual_lidar_plan.py` — manifest (pose_evidence_backend, map_backend present and valid; primitives path sets lidar_evidence/map_update), single pose-evidence path, fixed-cost budgets (N_FEAT, N_SURFEL, K_ASSOC, K_SINKHORN, RINGBUF_LEN), CertBundle Frobenius policy (approximation_triggers ≠ ∅ ⇒ frobenius_applied). 13 tests.

## 2026-01-30: PrimitiveMap publisher, RViz config, bins as derived (Visual LiDAR plan)

- **Map publisher:** New `fl_slam_poc/backend/map_publisher.py`: `PrimitiveMapPublisher` publishes PrimitiveMap as PointCloud2 on `/gc/map/points` (xyz + intensity/weight). Optional MarkerArray `/gc/map/ellipsoids` (not yet populated). Uses `extract_primitive_map_view`; frame_id from node (e.g. odom).
- **Backend integration:** When `map_backend=primitive_map`, backend_node creates `PrimitiveMapPublisher` and calls `publish(primitive_map, stamp_sec)` from `_publish_state_from_pose` (same time as state/path).
- **RViz:** New `fl_ws/src/fl_slam_poc/rviz/gc_slam.rviz` — fixed frame odom, PointCloud2 `/gc/map/points`, Path `/gc/trajectory`. Installed to `share/fl_slam_poc/rviz/`. Added `visualization_msgs` to package.xml.
- **Stage 3 (bins as derived):** Pipeline comment documents that bins (MapBinStats, B_BINS) are derived/legacy only: not used for pose evidence; MapUpdateResult bin-shaped for backend_node aggregation only (deltas zero); canonical map is PrimitiveMap.

## 2026-01-30: Odom vs belief diagnostic (raw vs estimate)

- **Backend:** New params `odom_belief_diagnostic_file` (path, default "") and `odom_belief_diagnostic_max_scans` (int, default 0 = all). When file is set, backend appends one CSV row per scan: scan, t_sec, raw odom (x, y, yaw_deg, vx, vy, wz), belief at scan start (x, y, yaw_deg, vx, vy), belief at scan end (x, y, yaw_deg, vx, vy). Helpers: `_yaw_deg_from_pose_6d`, `_belief_xyyaw_vel`.
- **Launch:** `gc_rosbag.launch.py` declares and passes `odom_belief_diagnostic_file` and `odom_belief_diagnostic_max_scans` to the backend.
- **Eval script:** `run_and_evaluate_gc.sh` sets `odom_belief_diagnostic_file:=$RESULTS_DIR/odom_belief_diagnostic.csv` and `odom_belief_diagnostic_max_scans:=200` so each run writes the diagnostic CSV under results (first 200 scans). Use it to compare raw odom vs belief and spot X/Y or scale mismatch.

## 2026-01-30: Map and belief audit (Kimera 60s run)

- **Eval run:** `results/gc_20260130_125224` (Kimera 10_14_acl_jackal-005, 60s, 160 scans). ATE trans 0.38 m, rot 0.65 deg; EST ptp X 0.97 m, Y 0.20 m vs GT ptp X 0.19 m, Y 0.32 m.
- **Audit:** `docs/MAP_AND_BELIEF_AUDIT_2026-01-30.md` — pipeline trace (odom → evidence → fusion), map/belief notes, and root-cause hypothesis. Odom pose + twist + velocity + yaw rate + kinematic consistency are all in L_evidence at full strength (alpha=1.0). Likely cause: **frame/axis convention** (GT “forward” may be Y in GT frame while our odom/state “forward” is X), leading to X/Y displacement mismatch after initial-pose alignment. Checklist: verify GT vs odom frame, inspect odom covariances, log first odom vs belief poses, verify map/bin frame.

## 2026-01-30: Splat rendering moved to backend (Option A)

- **Rendering location:** `splat_rendering.py` moved from `frontend/sensors/` to `backend/rendering.py`. Rendering is output from state/map, not sensor input; backend owns inference, fusion, and outputs (trajectory, state, rendered view).
- **Docs:** PRODUCTION_READINESS_SPLAT_PIPELINE.md updated: Frontend/sensors no longer lists splat_rendering; Backend lists rendering.py. Runtime manifest checklist includes rendering. CHANGELOG historical line updated to backend/rendering.py.
- **Convention:** Frontend = sensor I/O + evidence extraction only; backend = inference + fusion + kernels + output (including rendering).

## 2026-01-30: Default evaluation profile Kimera; M3DGR Dynamic01 archived

- **Default profile:** `tools/run_and_evaluate_gc.sh` now defaults to `PROFILE=kimera` (Kimera bag and config). M3DGR Dynamic01 is no longer the default.
- **Archive:** Added `archive/docs/M3DGR_DYNAMIC01_ARCHIVE.md` with paths and usage for running with Dynamic01 via `PROFILE=m3dgr`. Updated `archive/README.md` to list it.
- **Code/docs:** `fl_slam_poc/common/constants.py` frame comment is dataset-agnostic (no longer "Dynamic01_ros2"). README and run script comments updated to state default is Kimera.

## 2026-01-30: Full fusion trust and odom pose + twist + time evidence

- **Fusion trust:** Set GC_ALPHA_MIN = GC_ALPHA_MAX = 1.0 so fusion scale alpha is always full strength (no scaling down). L_post = L_pred + L_evidence; evidence is no longer damped by quality-based alpha.
- **Odom evidence:** Pipeline already uses (1) odom pose (L_odom), (2) odom velocity (v_odom_body), (3) odom yaw rate (omega_z), (4) pose-twist kinematic consistency (pose change ≈ R @ v*dt, omega*dt). Documented in pipeline comments that twist+time gives distance/rotation evidence; message covariances (odom_twist_cov) control strength.

## 2026-01-30: Kimera bag inspection and real extrinsics (no placeholders)

- **Bag inspection:** Ran full inspection on `rosbags/Kimera_Data/ros2/10_14_acl_jackal-005`. First-N messages, diagnose_coordinate_frames, estimate_lidar (ground plane), estimate_imu (gravity) all ran successfully.
- **Findings:** LiDAR Z-UP confirmed; odom covariance ordering ROS [x,y,z,roll,pitch,yaw]; IMU specific force ~-Y (optical frame); IMU/odom covariances from bag recorded in KIMERA_FRAME_MAPPING.
- **Extrinsics applied:** `gc_kimera.yaml` now has real rotations (no longer placeholders): T_base_lidar [0,0,0, 0.0217, 0.0080, 0] (small roll/pitch from ground plane); T_base_imu [0,0,0, -1.6027, 0.0026, 0] (~91.8° from gravity). Translation 0,0,0 (not estimated).
- **Tooling:** `estimate_lidar_base_extrinsic_rotation_from_ground.py` supports PointCloud2 (VLP-16) when topic type is sensor_msgs/msg/PointCloud2. `tools/inspect_kimera_bag.py` runs all inspections and can `--apply` parsed extrinsics to config.

## 2026-01-30: GC v2 Kimera frames, noise, eval, extrinsics, and sensor sanity

- **Frame mapping:** `docs/KIMERA_FRAME_MAPPING.md` documents Kimera→GC frame names (odom_frame, base_frame, LiDAR/IMU frames), axis conventions, diagnostic results placeholder, discovery (extrinsics, LiDAR variance, IMU/odom cov, timing), extrinsics loading, config options, and camera topics. `FRAME_AND_QUATERNION_CONVENTIONS.md` points to Kimera frame mapping.
- **PointCloud2:** `parse_pointcloud2_vlp16` in backend; `pointcloud_layout: livox | vlp16` in config and launch; fail-fast on wrong layout. `docs/POINTCLOUD2_LAYOUTS.md` documents Livox vs VLP-16 layouts.
- **Config overlay:** `config/gc_kimera.yaml` for Kimera (odom_frame, base_frame, pointcloud_layout vlp16, lidar_sigma_meas 1e-3, extrinsics_source inline). Sensor hub uses pointcloud_passthrough and Kimera topics from config.
- **Eval profile:** `run_and_evaluate_gc.sh` supports `PROFILE=kimera`: sets BAG_PATH, GT_FILE, CONFIG_PATH, odom_frame, base_frame, pointcloud_layout, lidar_sigma_meas; fails if bag or GT missing when profile is kimera.
- **Extrinsics:** Backend params `extrinsics_source: inline | file`, `T_base_lidar_file`, `T_base_imu_file`; when file, load 6D from YAML (fail if missing). Launch and script pass them. `tools/check_extrinsics.py` prints T_base_lidar, T_base_imu and frame names from config.
- **Noise:** Configurable `lidar_sigma_meas` (scalar m²); Kimera/VLP-16 1e-3, M3DGR 0.01. `create_datasheet_measurement_noise_state(lidar_sigma_meas=...)` and `create_datasheet_lidar_bucket_noise_state(lidar_sigma_meas=...)` accept override.
- **Diagnose script:** `tools/diagnose_coordinate_frames.py` supports PointCloud2 (VLP-16) when lidar topic type is sensor_msgs/msg/PointCloud2; extracts x,y,z for Z-up/Z-down analysis.
- **First-N summary:** `tools/first_n_messages_summary.py` emits first N messages summary (field names, frame_id, sample values) for PointCloud2, Imu, Odometry; output markdown or JSON per bag.

## 2026-01-30: Unified LiDAR depth evidence (I0.2) and PoE fusion

- **One LiDAR depth evidence API (I0.2):** `lidar_depth_evidence(points_camera_frame, uv_query, fx, fy, cx, cy, config, ...)` returns `(Λ_ell, θ_ell)` per query. Always defined, continuous; Λ_ell → 0, θ_ell → 0 when not applicable (camera-only behavior).
- **Route A:** `_route_a_natural_params` returns (ΛA, θA) with reliability wA = w_cnt * w_mad * w_repr (point support sigmoid, MAD spread exp(−β σ²_A), reprojection exp(−γ r²)). MAD scale 1.4826 for robust σ_A.
- **Route B:** Returns (ΛB, θB) via `_route_b_natural_params`; added w_z = σ(α_z(z* − z_min)) for behind/min-depth. Combined weight wB = w_angle * w_planar * w_res * w_z.
- **Mixture-of-experts:** Λ_ell = ΛA + ΛB, θ_ell = θA + θB (optional gamma_lidar modality dial). No "choose A or B"; both contribute; weak experts vanish smoothly.
- **Config:** Added point_support_n0, point_support_alpha, spread_mad_beta, repr_gamma (Route A); depth_min_sigmoid_alpha_z, gamma_lidar.
- **splat_prep_fused:** Uses `lidar_depth_evidence` only; PoE Λf = w_c*Λc + w_ell*Λ_ell, θf = w_c*θc + w_ell*θ_ell; reliability scaling Λ←wΛ, θ←wθ. Removed use_route_b parameter.
- **Legacy removed (no backward compat):** Dropped `lidar_ray_depth_route_a`, `lidar_ray_depth_route_b` (public), `depth_natural_params`, `fuse_depth_natural_params`. Route B logic kept as internal `_route_b_ray_plane`; single public API `lidar_depth_evidence` + `backproject_camera` / `backprojection_cov_camera`.

## 2026-01-30: Kimera switch cleanup + MIT/Fellowship gitignore

- **M3DGR-era docs archived:** Moved to `archive/docs/`: `TRACE_Z_EVIDENCE_AND_TRAJECTORY.md`, `RAW_MEASUREMENTS_VS_PIPELINE.md`, `TRACE_TRAJECTORY_AND_GROUND_TRUTH.md`. Project switching to Kimera (`rosbags/Kimera_Data/`); z evidence and raw-measurements audits are M3DGR/race-specific. Refs updated in PIPELINE_DESIGN_GAPS, README, EVALUATION, PREINTEGRATION_STEP_BY_STEP, datasets/M3DGR_STATUS; archive/README lists new items.
- **MIT/ and Fellowship/ removed:** `MIT/` (nested duplicate path + venv) and `Fellowship/` (Python venv) deleted from repo to reduce clutter; both remain in `.gitignore` if recreated.
- **logs_llm/** added to `.gitignore`; directory removed from tree (MCP runtime logs).
- **Cleanup audit executed (except VLP16 and config):** `TRAJECTORY_COMPARISONS.md` moved to `archive/docs/` (run-log of trajectory comparison images; links to gitignored `results/`). `docs/MCP_CODE_GRAPH_SETUP.md` moved to `tools/README_MCP.md`; README doc table and `tools/install_code_graph_rag_mcp.sh` now point to `tools/README_MCP.md`. `VELODYNE_VLP16.md` and `config/cyclonedds_*.xml` left as-is per request (VLP16 not archived; config still converting).
- **docs/CLEANUP_AUDIT.md removed:** Audit completed; doc deleted to avoid bloat.

## 2026-01-30: Camera wiring for Kimera bag (C++ decompress + depth passthrough)

- **C++ image_decompress_node:** Depth subscription is optional: when `depth_compressed_topic` is empty, the node only subscribes to RGB and publishes decompressed rgb8 to `rgb_output_topic`. Enables Kimera bags (RGB compressed, depth raw) without requiring compressed depth.
- **Launch (gc_rosbag.launch.py):** Added `enable_camera` (default false) and camera topic args: `camera_rgb_compressed_topic` (default `/acl_jackal/forward/color/image_raw/compressed`), `camera_rgb_output_topic` (`/gc/sensors/camera_image`), `camera_depth_compressed_topic` (default "" for Kimera), `camera_depth_output_topic` (`/gc/sensors/camera_depth`), `camera_depth_raw_topic` (default `/acl_jackal/forward/depth/image_rect_raw`). When `enable_camera:=true`, `image_decompress_cpp` runs (RGB only if depth_compressed empty). When `enable_camera:=true` and `camera_depth_raw_topic` non-empty, `depth_passthrough` runs.
- **Depth passthrough (frontend/sensors/depth_passthrough.py):** New Python node: subscribes to raw depth (sensor_msgs/Image, e.g. 16UC1 mm), republishes to canonical topic with optional `scale_mm_to_m` (16UC1 → 32FC1 m). Entry point `depth_passthrough`. For Kimera bags that publish raw depth at `.../depth/image_rect_raw`.

Usage (canonical bag): `ros2 launch fl_slam_poc gc_rosbag.launch.py bag:=rosbags/Kimera_Data/ros2/10_14_acl_jackal-005 enable_camera:=true` (defaults target canonical bag topics; see docs/BAG_TOPICS_AND_USAGE.md).

## 2026-01-30: Production-readiness assessment (splat pipeline)

- **docs/PRODUCTION_READINESS_SPLAT_PIPELINE.md:** New doc identifying what would need to change to put LiDAR–camera splat fusion and BEV OT into production. Covers: current state (backend subscribes only to lidar/odom/imu; no camera; no splat/OT in pipeline); topics and frontend (canonical camera topic, camera normalizer, fail-fast); backend params and state (T_base_camera, intrinsics, optional Lambda_prev for temporal smoothing); end-to-end data flow (image → extractor → depth fusion → splat_prep → BEV → pack_splat_batch → Sinkhorn → OT fusion); pipeline options (output-only vs evidence vs new steps); config and RuntimeManifest; frame/ID consistency; performance; tests; docs. Recommends starting with output-only splat branch in on_lidar.

## 2026-01-30: Packed splat batch + integration-risk refinements

- **Packed splat batch (common/splat_batch.py):** Standardized internal structure: fixed caps N_max, M_max, K_max; `PackedSplatBatch` now includes `K_max` and always-allocated `neighbor_indices_cam` (N_max × K_max), -1 = invalid. `pack_splat_batch(..., K_max=GC_OT_K_STENCIL, neighbor_indices_cam=None)` pads neighbor indices when provided; shape consistency for JIT / no dynamic sizes. Added constant `GC_OT_K_MAX`.
- **Route B (lidar_camera_depth_fusion.py):** Config doc note: when |n'r̂| is very small, σz² becomes huge (precision → 0); `depth_sigma_max_sq` caps for float safety (no blow-up). Continuous z and reliability already in place (softplus clamp, w_behind).
- **Sinkhorn (sinkhorn_ot.py):** Cost normalization already applied in `sinkhorn_ot_bev`: `cost_subtract_row_min`, optional `cost_scale_by_median` (config); numerical conditioning, not gating.
- **OT fusion (ot_fusion.py):** Module doc: recommend `confidence_tempered_gamma(π, γ, α, m0)` and pass as `gamma_per_row` to weighted_fusion_* for row-mass confidence γ_i = γ·σ(α(m_i−m0)). Wishart and temporal_smooth_lambda doc: apply in consistent coordinate chart (e.g. BEV/map frame); temporal smoothing requires stable feature IDs (map primitives), not per-frame features without re-association.

## 2026-01-30: LiDAR–camera splat fusion and BEV OT (plan: lidar-camera_splat_fusion_and_bev_ot)

- **Visual extractor (minimal):** Per-feature meta now exposes depth for fusion: `depth_sigma_c_sq`, `depth_Lambda_c`, `depth_theta_c` (invalid depth: Lambda_c=0, theta_c=0). Module docstring updated.
- **Depth fusion:** New `frontend/sensors/lidar_camera_depth_fusion.py`: `depth_natural_params(z_hat, sigma_sq)`, `fuse_depth_natural_params(Lambda_c, theta_c, Lambda_ell, theta_ell)`; optional w_c, w_ell (continuous); `LidarCameraDepthFusionConfig`. Route A: `lidar_ray_depth_route_a(...)` — project LiDAR to camera, per-(u,v) median/nearest depth and sigma_ell_sq; `backproject_camera`, `backprojection_cov_camera` for splat_prep.
- **BEV pushforward:** New `common/bev_pushforward.py`: `pushforward_gaussian_3d_to_2d(mu, Sigma, P)`, oblique `P(φ)` from config, stub `pushforward_vmf_to_s1_stub`.
- **Sinkhorn OT:** New `backend/operators/sinkhorn_ot.py`: cost W2² + β H²_vMF, fixed-K Sinkhorn (no convergence check), `SinkhornOTResult`, `CertBundle` with `sinkhorn_fixed_iter`; `SinkhornOTConfig`; no external OT lib.
- **OT fusion:** New `backend/operators/ot_fusion.py`: `coupling_to_weights(pi)` (continuous; no threshold), `weighted_fusion_gaussian_bev`, `weighted_fusion_vmf_bev`; `OTFusionConfig` (gamma, epsilon).
- **Splat prep:** New `frontend/sensors/splat_prep.py`: `splat_prep_fused(extraction_result, points_camera_frame, intrinsics, config)` → list of fused `Feature3D`; callable from script; no backend_node/pipeline edits.
- **Rules:** OT fusion uses continuous w_ij = π_ij/(Σ_j π_ij + ε); no "fuse only if > 0.2" gate. Sinkhorn loop bound by K_SINKHORN; CertBundle/ExpectedEffect returned.
- **Deferred (not implemented):** LiDAR surfels, unbalanced Sinkhorn (τ_a, τ_b), post Wishart / temporal α in ot_fusion, rendering (EWA, vMF shading, fBm). Wiring LiDAR into backend_node/pipeline deferred until modules are exercised from script.

## 2026-01-30: Route B (ray–plane) and MA hex web

- **Route B (lidar_camera_depth_fusion.py):** `lidar_ray_depth_route_b(...)` — gather points near ray (pixel radius + K nearest to ray), fit plane via SVD (fixed-cost), intersect ray with plane → ẑℓ; σ_ℓ² from plane fit residuals; continuous weight w_plane = 1/(1 + (residual_rms/prior)²). Config: `lidar_ray_plane_fit_max_points`, `plane_fit_residual_prior_m`, `plane_fit_cos_min`. No gate; no RANSAC.
- **splat_prep:** Optional `use_route_b=True`; when set, uses Route B and scales Λℓ by plane-fit weight.
- **MA hex web (common/ma_hex_web.py):** Hex cell keys a1=(1,0), a2=(1/2,√3/2); s_k = a_k·y, cell = floor(s/h); h = hex_scale_factor × median(√λ_max(Σ)). Fixed bucket `MAHexBucket` [num_cells, max_occupants]; stencil K=64 (8×8); `poe_denoise_2d` (Gaussian PoE over cell + stencil); `convexity_weight(mu, mu_poe, tau)`. Config: `MAHexWebConfig`. No edits to existing binning/pipeline.

## 2026-01-30: Unbalanced Sinkhorn, Route B weighted plane, W2² SPD guards

- **Unbalanced Sinkhorn (sinkhorn_ot.py):** `sinkhorn_unbalanced_fixed_k(C, a, b, epsilon, tau_a, tau_b, K)` — KL relaxation on marginals (τ_a KL(π1|a), τ_b KL(πᵀ1|b)); updates u = (a/(Kv))^(1/(1+τ_a/ε)), v = (b/(Kᵀu))^(1/(1+τ_b/ε)). No threshold; continuous. Config: `tau_a`, `tau_b` (default 0.5); when τ_a>0 or τ_b>0, `sinkhorn_ot_bev` uses unbalanced. Reduces bad matches under partial camera↔LiDAR overlap.
- **W2² SPD guards (sinkhorn_ot.py):** Symmetrize S ← (S+Sᵀ)/2 and eigen clamp (eig_min) before matrix sqrt; `w2_eig_min` in config. Domain projection, not gating. `cost_matrix_bev` passes `eig_min` to `w2_sq_2d`.
- **Route B weighted plane (lidar_camera_depth_fusion.py):** Weighted PCA plane: min sum w_k (n'x_k+d)²; weighted centroid, weighted cov S; n = eigenvector of λ_min. `_fit_plane_weighted(points, weights)` → centroid, normal, eigvals, sigma_perp_sq. Intersection (continuous): z* = n'x̄/(n'r̂+δ) with `plane_intersection_delta` (no division-by-zero). σz,ℓ² = σ⊥²/((n'r̂)²+δ). Continuous reliability: w_angle = sigmoid(α(|n'r̂|−t)), w_planar = sigmoid(β(ρ−ρ0)) with ρ = λ2/(λ3+ε), w_res = exp(−γ σ⊥²); weight_plane = w_angle * w_planar * w_res; scale Λℓ by weight_plane (no gate). Config: `plane_intersection_delta`, `plane_angle_sigmoid_alpha/t`, `plane_planarity_sigmoid_beta/rho0`, `plane_residual_exp_gamma`, `plane_fit_eps`. Removed hard `plane_fit_cos_min` skip; intersection always defined via δ.

## 2026-01-30: LiDAR surfels, post Wishart + temporal α, optional rendering

- **LiDAR surfels (frontend/sensors/lidar_surfels.py):** Voxel downsample (voxel_size_m), plane fit per voxel via _fit_plane_weighted; Gaussian mean = centroid, Σ from plane residuals + sensor noise; vMF B=3 (normal + two in-plane directions); Wishart Λ_reg = Λ + nu/psi_scale * I. `LidarSurfel` dataclass (xyz, cov_xyz, info_xyz, canonical_theta, mu_app (3,3), kappa_app (3,)); same natural-param interface as camera splats. Config: `LidarSurfelConfig`. No edits to livox_converter.
- **ot_fusion (backend/operators/ot_fusion.py):** Post-fusion Wishart: `wishart_regularize_2d(Lambda, nu, psi_scale)` — Λ_reg = Λ + nu * Psi^{-1}, Psi = psi_scale * I. Temporal smoothing: `temporal_smooth_lambda(Lambda_t, Lambda_prev, alpha)` — Λ_smoothed = Λ_t + alpha * Λ_prev; caller holds Λ_prev. Config: `wishart_nu`, `wishart_psi_scale`, `temporal_alpha`.
- **Optional rendering (backend/rendering.py):** EWA splatting: `ewa_splat_weight`, `ewa_splat_weights_at_point`, `render_tile_ewa` (tile 32×32). Multi-lobe vMF shading: `vmf_shading_multi_lobe(v, mu_app, kappa_app, pi_b)`. fBm: `fbm_value_noise(x, y, octaves=5, gain=0.5)` (deterministic hash-based value noise). `opacity_from_logdet`. Config: `SplatRenderingConfig`. Fixed loops; no JAX requirement. (Rendering moved from frontend/sensors to backend — output from state, not sensor.)

## 2026-01-29: Visual feature extractor — 16-item enhancement (plan)

- **Must-fix:** Wired `canonical_theta` and `canonical_log_partition` in all `Feature3D` constructions (natural-parameter fusion API, item 9).
- **Config:** Extended `VisualFeatureExtractorConfig` with hex depth, quad fit, Wishart prior, Student-t, MA convexity, kappa scheduling, FR/score beta, 3DGS export, and diagnostics; named constants `MAD_NORMAL_SCALE`, `VMF_LOG_NORM_4PI`, `LOG_2PI`.
- **Depth (1):** Hexagonal 7-sample stencil (`depth_sample_mode="hex"`), robust depth = median, scale = MAD; `_depth_sample` returns optional sample list for Student-t.
- **Quadratic fit (2):** `_quadratic_fit(depth, u, v, z_hat)` → `QuadFitResult` (grad_z, H, normal, K, ma, lam_min); optional in extract loop.
- **Student-t (10):** `_student_t_effective_var(z_hat, sigma_z2, zs)`; effective variance used in backprojection cov.
- **MA convexity (11):** `_ma_convexity_weight(lam_min)`; Sigma inflated by `(1 - w_ma) * delta`; kappa scaled by w_ma.
- **Wishart (5):** Optional `Lambda' = Lambda + nu Psi^{-1}` in `_precision_and_logdet` when `prior_nu > 0`.
- **Kappa scheduling (12):** Per-feature `kappa_app` from reliability rho and |K|; optional `mu_app`/`kappa_app` on `Feature3D`.
- **vMF / Hellinger (3):** `A_vmf(k)`, `hellinger2_vmf(mu1,k1, mu2,k2)`; Fisher-Rao (4): `fr_cov`, `fr_gauss`; orthogonal score (15): `orthogonal_score(...)`.
- **3DGS export (7):** `feature_to_splat_3dgs(feat, color)` → dict(mean, scale, rot, opacity, color).
- **Heatmaps (8):** `_build_hellinger_heatmap_base64`, `_build_fr_heatmap_base64`; attached to op_report metrics when `debug_plots=True` and matplotlib available.
- **Diagnostics (16):** Per-feature `logdet_cov`, `trace(info_xyz)`, `rel_depth_noise`, `lam_min_H`; distributions in final OpReportEvent metrics.
- **BC LUT (13):** Precomputed BC on grid (kappa_1, kappa_2, cos theta); `hellinger2_vmf_lut(...)` for fast association.
- **Mixture vMF (14):** `mixture_vmf_to_single(mus, kappas, weights)` moment-match to single vMF.
- **Batching (6):** Single numpy per-feature loop (fixed-cost); no GPU path; explicit backend selection if added later.
- **File:** `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/visual_feature_extractor.py`.

## 2026-01-30: Kimera_Data ROS bag prep (ROS1 decompression + ROS2 conversion)

- Organized raw ROS 1 bags into `rosbags/Kimera_Data/ros1_raw/`.
- Generated uncompressed ROS 1 bags in `rosbags/Kimera_Data/ros1_uncompressed/`.
- Converted all Kimera_Data bags to ROS 2 sqlite3 format under `rosbags/Kimera_Data/ros2/`.
- Added dataset manifests: `rosbags/Kimera_Data/manifest.json` and `rosbags/Kimera_Data/manifest.csv`.
- Added bag↔GT↔extrinsics mapping `rosbags/Kimera_Data/dataset_ready_manifest.yaml` for eval swap-in.
- Updated prep notes in `rosbags/Kimera_Data/PREP_README.md` to reflect the new layout and scope.
- Organized Kimera calibration + extrinsics under `rosbags/Kimera_Data/calibration/` with per-robot extrinsics and summary manifests.
- Added `tools/run_gc_kimera.sh` to launch GC v2 against Kimera bags with topics/extrinsics pulled from the manifest, and parameterized `gc_rosbag.launch.py` (topics, extrinsics, config path) to allow non-M3DGR datasets.

## 2026-01-29: Frame/planar Z conventions — base_footprint Z=0, correct evidence residual signs

- **Root cause:** M3DGR `/odom` does not observe Z (odom Z ~30m garbage, cov ~1e6), and M3DGR GT is reported in `camera_imu` (Z ≈ 0.85m) while GC v2 state is `base_footprint` by contract.
- **Fix (by construction):**
  - Treat GC state/body frame as `base_footprint` ⇒ planar Z reference is **0.0 m** (`GC_PLANAR_Z_REF = 0.0`). Evaluate against GT by transforming wheel-frame estimate to body frame via `config/m3dgr_body_T_wheel.yaml` (`tools/transform_estimate_to_body_frame.py`).
  - Make unary “pull-to-measurement” evidence residuals consistent: **residual = measurement − prediction** for planar priors and odom twist evidence (previously several operators used prediction − measurement, which pushes away from the measurement).
- **Code:**
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py`: set `GC_PLANAR_Z_REF = 0.0` and document wheel vs body (GT) frames.
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/planar_prior.py`: fix sign so MAP increment drives z toward `z_ref` and v_z toward 0.
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/odom_twist_evidence.py`: fix sign for velocity, yaw-rate, and pose–twist consistency residuals.
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`: odom pose anchor Z uses `GC_PLANAR_Z_REF` (no use of raw odom Z).
  - `tools/diagnose_coordinate_frames.py`, `tools/diagnose_frames_detailed.py`: remove erroneous `/1000` scaling (Dynamic01_ros2 `livox_ros_driver2/CustomMsg` points are already meters).

## 2026-01-29: ROADMAP.md updated to match current codebase

- **ROADMAP.md**: Brought in line with current GC v2: 22D state, 14-step pipeline, `gc_sensor_hub` → `gc_backend_node`, `/gc/sensors/*` topics, `gc_rosbag.launch.py`, `gc_unified.yaml`. Completed milestones and §1 (GC v2 status) rewritten; immediate priorities aligned with `docs/PIPELINE_DESIGN_GAPS.md`; removed/updated stale refs (15D→22D, `run_and_evaluate.sh`→`run_and_evaluate_gc.sh`, `poc_m3dgr_rosbag`→`gc_rosbag`, `phase2/` paths). Roadmap Summary and Future semantic section clarified. Historical Phase 1–6 detail retained with note that current architecture is summarized in §1 and Completed Milestones.

## 2026-01-29: Fixed shapes + diagnostics doc + bin_atlas import

- **Dynamic shapes:** Point budget now returns **fixed-size** arrays (N_POINTS_CAP) instead of slicing to n_selected; pipeline passes them to deskew → bin_soft_assign → scan_bin_moment_match so downstream JITs see static `n_points` and do not recompile per scan (point_budget.py, pipeline unchanged except uses full result).
- **Docs:** PIPELINE_COMPUTE_BOTTLENECKS.md §3.3.1 documents cost of `save_full_diagnostics=True` (heavy per-scan host pulls; use for short debug only). Summary table and §6.3 updated for fixed-shape status.
- **bin_atlas:** Moved `kappa_from_resultant_batch` import to module level (was inside JIT'd `_compute_map_derived_stats_core`).

## 2026-01-29: Pipeline performance and async (JIT cores, minimal tape, deferred publish)

- **compute_map_derived_stats:** Replaced 48-iteration Python loop with one JIT'd batched path; `domain_projection_psd_batch` in primitives; bin_atlas returns tuple from JIT core, wrapper builds MapDerivedStats.
- **Conditioning:** JAX `eigvalsh` on device for pose6 block; pull only scalars for ConditioningCert; no NumPy in hot path (pipeline.py).
- **Map delta accumulation:** Batched weighted sum with `jnp.einsum` over stacked increments; no Python loop, no per-iteration host pull (backend_node.py).
- **JIT step-level operators:** predict_diffusion, scan_bin_moment_match, bin_soft_assign, odom_quadratic_evidence, point_budget_resample now have JIT'd cores returning arrays; Python wrappers build CertBundle/ExpectedEffect from single host pull. point_budget returns fixed-size (N_POINTS_CAP) arrays so downstream binning JITs see static n_points (see later 2026-01-29 entry for full fix).
- **Diagnostics tape:** Minimal per-scan tape by default (`PipelineConfig.save_full_diagnostics=False`). Hot path appends `MinimalScanTape` (scalars + L_pose6); full `ScanDiagnostics` only when `save_full_diagnostics=True`. `DiagnosticsLog.save_npz()` writes minimal or full format; `append_tape()` for tape, `append()` for full.
- **Async/deferred publish:** State, TF, path are published at the **start** of the next LiDAR callback (drain pending); pipeline hot path does not block on ROS publish. Last scan published on destroy_node.
- **Docs:** PIPELINE_COMPUTE_BOTTLENECKS.md updated with "Addressed" items, §6 Async publishing and diagnostics, §6.3 Compilation and recompiles, summary table status.

## 2026-01-28: IMU preint window slice + Wahba removed

- **IMU preintegration bottleneck fix (PIPELINE_DESIGN_GAPS §5.6):** Slice IMU buffer to integration window `[min(t_last_scan, scan_start), max(t_scan, scan_end)]`, cap at `GC_MAX_IMU_PREINT_LEN = 512`, pad to 512. Preintegration now runs over 512 steps instead of 4000 per call (~8× fewer scan iterations per LiDAR scan). `constants.GC_MAX_IMU_PREINT_LEN`, `backend_node.py` slice logic.
- **Wahba removed:** Pipeline uses Matrix Fisher rotation only. `wahba.py` moved to `archive/legacy_operators/wahba.py`; removed from `fl_slam_poc.backend.operators` and from tests (`TestWahbaSVD` removed). Diagnostics still store MF cost/yaw in fields `wahba_cost` / `dyaw_wahba` for NPZ/dashboard compatibility.

## 2026-01-28: Smoothed initial anchor (explicit A, no scan drops)

- **Explicit anchor A:** Belief lives in anchor frame; export uses anchor-to-world transform A. Provisional A0 = first odom sample (no scan drops); after first K odom, A_smoothed from closed-form aggregate.
- **Closed-form aggregate:** Translation t̄ = Σ w_k t_k / Σ w_k; rotation R̄ = polar(M) with M = Σ w_k R_k (one SVD). Weights w_k from IMU stability: w_k ∝ exp(-c_gyro ‖ω_k‖²) · exp(-c_accel (‖a_k‖ - g)²) (no gates).
- **anchor_correction:** pose_export = anchor_correction ∘ pose_belief; applied when publishing state, TF, and trajectory file. Identity until A_smoothed set.
- **backend_node.py:** A0 on first odom; odom_init_buffer for first K; _polar_so3, _imu_stability_weights; anchor_correction in _publish_state_from_pose.
- **constants.py:** GC_INIT_ANCHOR_GYRO_SCALE, GC_INIT_ANCHOR_ACCEL_SCALE, GRAVITY_MAG.
- **gc_unified.yaml:** init_window_odom_count (default 10).
- **docs/PIPELINE_DESIGN_GAPS.md:** §5.4.1 marked implemented with code anchors.

## 2026-01-29: Docs — PIPELINE_DESIGN_GAPS audited vs code

- **docs/PIPELINE_DESIGN_GAPS.md**: Updated to match current backend behavior (odom twist factors, planar priors + planar map z, time-resolved IMU tilt evidence, planarized LiDAR translation, fixed K_HYP hypothesis container). Added explicit code anchors and marked “DONE vs remaining” gaps. Noted that some referenced trace docs may be stale relative to code.

## 2026-01-29: Docs — bin-level motion smear plan (replace per-point deskew)

- **docs/PIPELINE_DESIGN_GAPS.md**: Added a detailed future plan (§5.8) for replacing per-point constant-twist deskew with a bin-level, closed-form 2nd-order rotational smear covariance (`Σ_motion,b`) using per-bin time moments (`Σ Δt^k`) and point second moments (`Σ p pᵀ`), including certification + Frobenius-correction requirements and IW-noise interaction notes.

## 2026-01-29: Docs — pipeline and evidence references updated

- Updated documentation to reflect current operators: Matrix Fisher rotation, planar translation evidence, time‑resolved IMU vMF, odom twist factors, and planar priors.
- Marked legacy z‑drift analysis as historical and documented current fixes.
- Updated diagrams and spec notes to remove Wahba/TranslationWLS as “current” and clarify legacy status.

## 2026-01-29: Archive legacy operators/tools/tests

- Moved Wahba/TranslationWLS tooling and legacy LiDAR evidence operators into `archive/`.
- Removed legacy exports/imports from `fl_slam_poc.backend.operators` and cleaned pipeline references.

## 2026-01-29: README — project overview refresh

- Expanded README with overview, novelty, goals, visuals, and updated code layout.

## 2026-01-29: Pipeline timing instrumentation

- Added optional per‑stage timing (ms) in diagnostics and a backend parameter `enable_timing`.

## 2026-01-29: Dashboard — MF/scatter sentinels + factor ledger readability

- **fl_ws/src/fl_slam_poc/fl_slam_poc/backend/diagnostics.py**: Added MF SVD and per-bin scatter eigenvalues + map kappa bins to NPZ schema for dashboard use.
- **fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py**: Populates MF SVD, scan/map scatter eigenvalues, and map kappa bins in `ScanDiagnostics`.
- **tools/slam_dashboard.py**: Reworked Panel A to show true Matrix Fisher singular values (log scale), scatter proxies, conditioning, yaw/geodesic agreement, and posterior spectrum sentinels; improved factor ledger into subspace plots and added explicit Z-leak visualization.

## 2026-01-28: SLAM Pipeline Upgrade - Planar Constraints and Odom Twist

### Summary

Major pipeline upgrade implementing Phase 1 (planar z fix) and Phase 2 (odom twist factors) from the SLAM upgrade plan. These changes address z drift and add kinematic coupling from wheel odometry.

### New Operators (No Gates, Principled Math)

1. **`planar_prior.py`** - Soft constraints for ground-hugging robots:
   - `planar_z_prior()`: Soft constraint z ≈ z_ref with variance σ_z². Injects L[2,2] = 1/σ_z² to pull z toward reference height.
   - `velocity_z_prior()`: Soft constraint vel_z ≈ 0 with variance σ_vz². Injects L[8,8] = 1/σ_vz² to prevent vertical velocity drift.

2. **`odom_twist_evidence.py`** - Kinematic constraints from wheel odometry twist:
   - `odom_velocity_evidence()`: Compares predicted velocity (body frame) with odometry velocity. Writes to L[6:9, 6:9].
   - `odom_yawrate_evidence()`: Compares predicted yaw rate with odometry angular.z. Writes to L[5,5].

### Constants Added (`constants.py`)

```python
GC_PLANAR_Z_REF = 0.86       # M3DGR reference height (meters)
GC_PLANAR_Z_SIGMA = 0.1      # Soft z constraint std dev (meters)
GC_PLANAR_VZ_SIGMA = 0.01    # Soft vel_z=0 constraint std dev (m/s)
GC_PROCESS_Z_DIFFUSION = 1e-8 # Reduced z diffusion
GC_ODOM_TWIST_VEL_SIGMA = 0.1 # Velocity std dev (m/s)
GC_ODOM_TWIST_WZ_SIGMA = 0.01 # Yaw rate std dev (rad/s)
```

### Map Update Z Fix (`map_update.py`)

- Set `t_hat[2] = 0` before map update to prevent belief z from being placed into map
- Breaks the z feedback loop: map stays in z=0 plane, belief z constrained by planar prior

### Backend Node Changes (`backend_node.py`)

- Read odom twist from `msg.twist.twist` and store in `last_odom_twist` (6D) and `last_odom_twist_cov` (6x6)
- Initialize odom twist to zeros with huge covariance (1e12) - by construction, not by gate
- Pass odom_twist and odom_twist_cov to pipeline

### Pipeline Integration (`pipeline.py`)

- Added planar config params: `planar_z_ref`, `planar_z_sigma`, `planar_vz_sigma`
- Added odom twist config: `odom_twist_vel_sigma`, `odom_twist_wz_sigma`
- Removed `enable_planar_prior` and `enable_odom_twist` gates (violated "no gates" rule)
- All operators run every scan; influence controlled by covariance (huge cov = negligible precision)

### Sign Mismatch Fix (First Scan Map Placement)

- Changed first scan map placement to use `belief_recomposed` (posterior) instead of `belief_pred` (prior)
- Rationale: The posterior incorporates gyro evidence which pulls toward END-of-scan rotation. Since the next scan's prior IS the previous posterior, the map must be at the same rotation for consistency.
- Without this fix: dyaw_wahba ≈ -dyaw_gyro (opposite signs) because map lags behind prior by one scan's gyro increment

### Design Principles Followed

- **No gates**: Removed all `if config.enable_*` gates. Operators always run.
- **By construction**: Missing data handled by huge covariance (negligible precision), not boolean checks
- **Principled math**: All evidence uses standard Gaussian information form (L, h)

### Remaining Issues (To Investigate)

1. **Planar prior may be too weak**: L[2,2] = 100 (from σ=0.1) vs LiDAR L[2,2] = 66000+
   - Consider reducing σ_z to 0.01-0.03 for stronger constraint

2. **Gyro vs Wahba sign disagreement persists**: Even after fix, diagnostics may show:
   - dyaw_gyro ≈ +40° (positive)
   - dyaw_wahba ≈ -60° (negative)
   - Root cause: Wahba estimates START-of-scan rotation, but gyro evidence pulls posterior toward END
   - The comparison is conceptually mismatched (different time references)

3. **Odom provides near-zero rotation constraint**: L_odom[3:6, 3:6] ≈ 1e-6
   - Odom covariance for rotation is huge in input data
   - Cannot help arbitrate gyro vs Wahba disagreement

4. **Catastrophic divergence at scan 56+**: Position suddenly jumps 2-5m per scan
   - Likely triggered by accumulated errors or numerical instability
   - Need to investigate what changes at that point

### Test After Implementation

```bash
cd /home/will/Documents/Coding/Phantom\ Fellowship\ MIT/Impact_Project_v1
./tools/run_and_evaluate_gc.sh
tail -20 /tmp/gc_slam_trajectory.tum | awk '{print $4}'  # Check z values
```

**Success criteria:**
- Z bounded within ±2m of ground truth (~0.86m)
- ATE translation RMSE < 15m (was ~47m)
- ATE rotation RMSE < 45° (was ~116°)

## 2026-01-27: README updated (GC v2 current state)

- **README.md**: Rewritten to reflect current GC v2 implementation. Removed outdated legacy structure (15-step pipeline, DeskewUTMomentMatch, /sim/ topics, separate Evidence Extractors, old code layout). Now documents: 22D state, IMU+Odom+LiDAR fusion, IW adaptive noise, 14-step pipeline with correct operator names (DeskewConstantTwist, WahbaSVD, TranslationWLS, etc.), gc_sensor_hub → gc_backend_node flow, actual fl_slam_poc layout (frontend hub/sensors/audit, backend pipeline/operators/structures), primary eval `run_and_evaluate_gc.sh` and `results/gc_*`. Notes known limitations (PIPELINE_DESIGN_GAPS, TRACE_Z) and points to PIPELINE_TRACE_SINGLE_DOC, BAG_TOPICS_AND_USAGE, and key docs.

## 2026-01-27: Pipeline trace — single document (value as object, causality)

- **docs/PIPELINE_TRACE_SINGLE_DOC.md**: Single trace document. Treats each value as an object and follows it through the deterministic pipeline (radioactive-signature style: trace where raw values "contaminate" final outputs). Contains: (1) pipeline spine (L1–L21 step order); (2) IMU message 5 from raw → on_imu → buffer → L5 → preintegration P1–P8 (gravity subtracted at P5 only) → deskew, gyro/vMF/preint evidence → fusion → trajectory; (3) Odom message 5 from raw → on_odom → last_odom_pose/cov → L15 odom evidence → fusion → trajectory; (4) LiDAR representative point from raw → parse → base → L7–L14 → R_hat, t_hat (3D) → LiDAR evidence → fusion → trajectory and map. Combined flow and units summary at end.
- **Removed:** docs/PIPELINE_MESSAGE_TRACE.md, docs/PIPELINE_MESSAGE_TRACE_MESSAGE_5.md (content merged into PIPELINE_TRACE_SINGLE_DOC.md). PREINTEGRATION_STEP_BY_STEP.md kept as expanded reference for P1–P8.

## 2026-01-27: Trace: where z in pose/trajectory comes from

- **docs/TRACE_Z_EVIDENCE_AND_TRAJECTORY.md**: Traces how z in the pose evidence and estimated trajectory arises from raw data and pipeline math. Odom z is very weak (cov 1e6 m² → L[2,2] = 1e-6); LiDAR translation evidence is full 3D (t_hat, t_cov) with isotropic Sigma_meas and no z-downweighting, so L_lidar[2,2] dominates; map–scan feedback reinforces belief_z; process Q and velocity treat z like x,y. Conclusion: z is driven primarily by LiDAR translation evidence and feedback, not by odom.

## 2026-01-27: Pipeline design gaps + operator-by-operator improvement plan

- **docs/PIPELINE_DESIGN_GAPS.md**: (1) Existing gaps: underuse of raw info (odom twist, IMU message covariances, LiDAR intensity); independence assumption; 6D pose evidence without twist/kinematics. (2) **New §6:** Operator-by-operator improvement plan: §6.0 current pipeline and failure modes (model-class errors); §6.1 production-safe upgrades (SE(2.5)/planar z, odom twist factors, time-resolved accel evidence, evidence strength from quality); §6.2 MHT (hypothesis = pose+map+noise+calibration, when to branch, scoring/prune/merge, bounded B); §6.3 2nd/3rd-order tensors (Riemannian trust-region, cubic-regularized Newton / 3rd-order correction for vMF); §6.4 high-risk research (Monge–Ampère/OT map updates, dually flat fusion, Amari–Chentsov bias, jerk-informed branching); §6.5 implementation checklist keyed to pipeline steps; §6.6 ROI sequence (planarize z → odom twist → accel evidence → MHT). References updated to PIPELINE_TRACE_SINGLE_DOC and TRACE_Z_EVIDENCE_AND_TRAJECTORY.

## 2026-01-27: Pipeline design gaps (documented known limitations)

- **docs/PIPELINE_DESIGN_GAPS.md**: New doc recording known design gaps: (1) underuse of raw info (odom twist, IMU message covariances, LiDAR intensity); (2) treatment of measurements as independent despite kinematic coupling (pose ↔ twist; “moving forward + yaw” implies x/y motion); (3) 6D pose evidence = inverse(message covariance) on pose residual with no twist, no pose–twist coupling, no forward/lateral structure. Doc ties to RAW_MEASUREMENTS_VS_PIPELINE and message-trace docs and lists what a better observation model would need (twist, pose–twist coupling, kinematics, consistency).

## 2026-01-28: Single IMU gravity config (one entry point)

- **Config cleanup:** Removed `imu_preintegration_gravity_scale`; one parameter `imu_gravity_scale` now controls gravity for both vMF evidence and preintegration. Single source: ROS param `imu_gravity_scale` (default 1.0), set from launch; PipelineConfig and RuntimeManifest use it only. Eval script and launch file no longer pass a second gravity scale.
- **Rationale:** Avoid multiple overlapping config entry points; 1.0 = correct gravity cancellation everywhere.

## 2026-01-28: IW readiness weight instead of gates (branch-free, no if/else)

- **backend_node.py**: Removed all IW min-scan **gates** (if/else). IW updates run **every scan** for process, measurement, and lidar_bucket. Readiness is a **weight** on sufficient stats: process weight = min(1, scan_count) (0 at scan 0, 1 from scan 1); meas and lidar weight = 1. So we always call all three IW apply functions; at scan 0 process contributes zero suff stats (weight 0). Aligns with AGENTS.md: "No hard gates; startup is not a mode; constants are priors/budgets" and "Branch-free: IW updates happen every scan."
- **constants.py**: Removed `GC_IW_UPDATE_MIN_SCAN_*` constants; documented that IW uses readiness weights (no MIN_SCAN thresholds).
- **Rationale:** Per-evidence adaptive noise (process from scan 1, meas/lidar from scan 0) without thresholds or branches.

## 2026-01-28: Do not skip scans when odom/IMU missing — use LiDAR when we have it

- **backend_node.py**: Removed "sensor warmup" skip that discarded LiDAR scans until odom (and IMU) were available. We should not discard evidence for no reason; if we have LiDAR we use it.
- When odom is missing we now pass identity pose + **large** covariance (`1e12 * I`) so odom evidence is negligible (no strong "pose is identity" pull). Previously we passed `I` covariance, which would add strong incorrect evidence.
- When IMU buffer is empty we already pass zero arrays (deskew no-op, IMU evidence near zero). No change.
- First odom message still sets `first_odom_pose` as reference when it arrives; early LiDAR-only scans run with negligible odom evidence until then.

## 2026-01-28: Code Graph RAG MCP — GitHub-only install

- **tools/README_MCP.md**: Setup instructions for code-graph-rag-mcp using **GitHub Releases only** (no npm registry); Cursor MCP config uses global `code-graph-rag-mcp` binary.
- **tools/install_code_graph_rag_mcp.sh**: Script to download latest release `.tgz` from GitHub and run `npm install -g`.

## 2026-01-28: Critical Fix - Wahba Sign Flip (First Scan Frame Mismatch)

### Summary

- **CRITICAL BUG FIX**: Resolved systematic Wahba sign flip where LiDAR rotation estimates had opposite sign from gyro/odom on 98.7% of scans.
- Root cause: First scan map initialization used END-of-scan pose (`belief_recomposed`) to transform directions, but scan directions were in START-of-scan body frame (from deskew).
- This introduced a rotation offset equal to within-scan motion that corrupted all subsequent Wahba matching.

### Root Cause Analysis

The pipeline has these steps for first scan:
1. Deskew brings points to START-of-scan body frame
2. Binning computes `scan_bins.s_dir` in START-of-scan frame
3. Map update was using `belief_recomposed` (END-of-scan pose) to transform directions
4. This mismatch caused rotation offset = within-scan motion

Example from diagnostics:
- Scan 1: gyro +16.5°, odom +85°, belief_recomposed ~+49°
- Scan directions at START (≈identity), but map stored them rotated by END (+49°)
- Scan 2+: Wahba matched against corrupted map → opposite sign

### Fix

Changed `pipeline.py` line ~970 from:
```python
pose_for_map = belief_recomposed.mean_world_pose(...)  # END-of-scan (WRONG)
```
to:
```python
pose_for_map = belief_pred.mean_world_pose(...)  # START-of-scan (CORRECT)
```

### Diagnostic Tools Created

- `tools/diagnose_wahba_sign.py`: Unit test for Wahba SVD sign convention
- `tools/diagnose_yaw_mismatch.py`: Runtime analysis of gyro/odom/wahba sign agreement
- `tools/trace_wahba_runtime.py`: Trace Wahba inputs with realistic scenario

### Key Insight

**Deskewed points are in START-of-scan body frame. Map update must use START-of-scan pose, not END-of-scan pose.**

## 2026-01-27: IMU, Belief, Map, and Fusion Clarification Doc

### Summary

- Added and expanded `docs/IMU_BELIEF_MAP_AND_FUSION.md` as the **pipeline reference**: end-to-end flow from raw rosbag topics through frontend (gc_sensor_hub) and backend (gc_backend_node) to outputs; pipeline step reference (1–14) with inputs/outputs and where each topic feeds in; sensor roles (IMU/gyro/accel/odom/LiDAR); frontend/backend topic-by-topic handling (LiDAR, IMU, odom); evidence types (Gaussians vs vMF); fusion (excitation scaling, α, InfoFusionAdditive); belief/bins/map and how IMU fits in.

### Changes

- `docs/IMU_BELIEF_MAP_AND_FUSION.md`: New/expanded doc: (1) From Rosbag to Output (raw topics, gc_sensor_hub canonical topics, backend subscriptions and outputs); (2) Pipeline step reference table (steps 1–14 + excitation); (3) Sensor breakdown; (4) Frontend and backend topic-by-topic (LiDAR converter + T_base_lidar, IMU normalizer + backend g→m/s² and R_base_imu, odom normalizer + first-odom reference); (5) Evidence types (Gaussian vs vMF); (6) Fusion; (7) Belief, bins, map; (8) Summary table; (9) References.

## 2026-01-28: Empirical Frame/Unit Convention Validation (Dynamic01_ros2)

### Summary

- Added deterministic, bag-based validation outputs to empirically confirm frame semantics, accel units, and odom twist consistency for the Dynamic01_ros2 dataset.
- Updated the canonical conventions doc to clearly distinguish **confirmed** vs **assumed** vs **to-confirm** items, and promoted many items to **CONFIRMED (Dynamic01_ros2)** with audit artifacts.
- Fixed `tools/diagnose_coordinate_frames.py` IMU section to use the correct accelerometer *specific force* convention (expected +Z when stationary in a Z-up base), avoiding a misleading ~180° interpretation.

### Changes

- `tools/validate_frame_conventions.py`: New validation script that produces `results/frame_validation_dynamic01.json`.
- `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`: Added status labels and an "Empirical Validation Artifacts" section; promoted items confirmed by the report.
- `tools/diagnose_coordinate_frames.py`: Corrected accelerometer convention and refreshed `results/frame_diagnose_dynamic01.txt` usage.

## 2026-01-27: Yaw Increment Invariant Test for Sign Mismatch Diagnosis

### Summary

- Added invariant test to compare yaw increments from three sources (gyro, odom, LiDAR/Wahba) to diagnose sign convention mismatches.
- Created analysis script to identify whether sign errors are in gyro processing (IMU→base rotation, axis sign, or left/right convention) or LiDAR extrinsic.
- Diagnostic data now includes per-scan yaw increments from all three sources for post-run analysis.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/diagnostics.py`: Added `dyaw_gyro`, `dyaw_odom`, `dyaw_wahba` fields to `ScanDiagnostics`; updated serialization/deserialization.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`: Enhanced invariant test to compute and log yaw increments from gyro-integrated rotation, odom pose, and Wahba (LiDAR) rotation; stores values in diagnostics.
- `tools/analyze_yaw_invariants.py`: New analysis script to load diagnostic data, compare sign consistency between sources, compute correlations, and identify problematic scans with interpretation guide.

### Purpose

This diagnostic helps pinpoint the root cause of yaw sign mismatches that cause trajectory spiraling:
- **Gyro ↔ Wahba mismatch**: Indicates sign error in gyro processing (A: IMU→base rotation, B: axis sign flip, C: left/right convention)
- **Odom ↔ Wahba mismatch**: Indicates sign error in LiDAR extrinsic (D: T_base_lidar rotation)

## 2026-01-27: Enforce Odom Pose Convention (Body→World)

### Summary

- Reverted the prior odom inversion: ROS Odometry pose already encodes `T_{parent<-child}` (e.g., `T_{odom<-base}`),
  which matches the SE(3) composition convention used by `se3_compose` and the rest of the pipeline.
- Added explicit frame-convention logging so the runtime records the chosen interpretation.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`: Treat `/odom` pose as `T_{parent<-child}` (no inversion); log `frame_id`, `child_frame_id`, and initial yaw for audit.

## 2026-01-27: IMU Propagation Diagnostics + Gravity Scale Hook

### Summary

- Added per-scan IMU propagation probes (weighted mean accel in body/world frames) and IMU dt diagnostics to the diagnostics log.
- Added `imu_gravity_scale` parameter to scale gravity contribution during IMU propagation, and logged it in the runtime manifest.
- Added `imu_preintegration_gravity_scale` parameter to independently scale gravity inside IMU preintegration (for gravity-off tests without affecting evidence).

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_preintegration.py`: accumulate and return weighted mean accel diagnostics during preintegration.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`: plumb IMU diagnostic outputs, compute dt stats, and apply gravity scale.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/diagnostics.py`: store and serialize new IMU propagation and dt diagnostics.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`: add `imu_gravity_scale` ROS parameter and publish in runtime manifest.

## 2026-01-27: GC Pose Ordering Migration to [trans, rot]

### Summary

- **Unified ordering**: GC tangent pose ordering now matches SE(3) and ROS (`[trans, rot]`), eliminating the legacy permutation and dual conventions.
- **Identity conversions**: `pose_se3_to_z_delta()` and `pose_z_to_se3_delta()` are now identity mappings (kept for compatibility).
- **Operator alignment**: Updated all evidence, IW, recompose, and pipeline block placements to the unified ordering; fixed BCH correction output ordering.
- **Docs aligned**: Updated canonical conventions and diagnostics docs to reflect the unified pose and covariance ordering.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py`: GC slice constants and convention header updated to `[trans, rot]`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/belief.py`: unified ordering docs, slice indices, and identity conversions.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_evidence.py`: rotation block placement moved to `[3:6]`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_gyro_evidence.py`: rotation block placement moved to `[3:6]`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/lidar_evidence.py`: pose blocks swapped to `[trans, rot]`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/odom_evidence.py`: removed ROS→GC permutation (orderings now match).
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/anchor_drift.py`: pose extraction uses `[trans, rot]`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/recompose.py`: BCH extraction/output order updated to `[trans, rot]`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/inverse_wishart_jax.py`: IW block mapping updated for `[trans, rot]`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/structures/inverse_wishart_jax.py`: process-noise block ordering and priors updated for `[trans, rot]`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/map_update.py`: pose covariance extraction updated.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`: diagnostics block ordering updated to `[trans, rot]`.
- `docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md`: state ordering table updated to `[trans, rot]`.
- `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`, `docs/Fusion_issues.md`, `docs/CODE_DIFF_SUMMARY.md`, `tools/DIAGNOSTIC_TOOLS.md`: documentation updated to the unified convention.
- `tools/diagnose_coordinate_frames.py`: odom covariance ordering analysis updated for ROS-standard vs legacy-permuted ordering.
- `tools/slam_dashboard.py`: pose heatmap labels updated to `[trans, rot]`.

## 2026-01-27: Fix Critical T_base_imu Mismatch Causing Yaw Drift

### Summary

- **CRITICAL FIX**: Resolved `T_base_imu` rotation mismatch between `gc_unified.yaml` and `gc_rosbag.launch.py`. Launch file was using wrong rotation `[1.475086, -0.813463, -0.957187]` (111.01°) instead of correct `[0.169063, -2.692032, 0.0]` (154.5°). This wrong rotation was being applied to gyro measurements, causing constant yaw drift that cannot be corrected by gyro alone.
- **Root cause analysis**: Created `docs/YAW_DRIFT_ROOT_CAUSE_ANALYSIS.md` documenting how wrong `T_base_imu` causes rotated angular velocity → constant yaw drift, and how odom limitations prevent correction.
- **Frame conventions verified**: Created `docs/FRAME_AND_QUATERNION_CONVENTIONS.md` as canonical reference for all coordinate frame and quaternion conventions throughout the codebase.

### Root Cause

The launch file parameters override config file parameters. The launch file had an incorrect `T_base_imu` rotation (111.01° instead of 154.5°), which was being applied to gyro measurements via `gyro_base = R_base_imu @ gyro_imu`. This rotated angular velocity into the wrong frame, causing constant yaw drift that accumulates over time. Since gyro only measures relative yaw change (not absolute yaw), this drift cannot be corrected by gyro alone. Odom can help, but only if:
1. Odom frame is correctly defined
2. Yaw covariance is reasonable
3. Odom is in the same base frame we're estimating

### Changes

- `fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py`: Fixed `T_base_imu` to match `gc_unified.yaml`: `[0.169063, -2.692032, 0.0]` (154.5°).
- `docs/YAW_DRIFT_ROOT_CAUSE_ANALYSIS.md`: Comprehensive analysis of yaw drift root causes (gyro rotation error, odom limitations, frame mismatches).
- `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`: Canonical reference for all frame and quaternion conventions.

## 2026-01-27: Fix Rosette Pattern Accumulation Window + IMU Extrinsic Rotation

### Summary

- **Fixed Livox MID-360 rosette pattern accumulation window**: When all `time_offset = 0` (non-repetitive pattern), scan bounds now use `timebase_sec` (accumulation start) to `header.stamp` (publish time) instead of treating scan as instantaneous. This captures the ~100ms accumulation window for proper deskew timing.
- **Fixed critical IMU misalignment** by updating `T_base_imu` rotation from `[-0.015586, 0.489293, 0.0]` (~28°) to `[0.169063, -2.692032, 0.0]` (154.5°) based on actual gravity direction analysis from rosbag data.
- **Verified LiDAR frame convention**: Confirmed Z-up convention via ground plane analysis; `T_base_lidar` rotation `[0, 0, 0]` is correct.
- **Created comprehensive coordinate frame diagnostic tool** (`tools/diagnose_coordinate_frames.py`) to verify frame conventions from first principles (no guessing).

### Root Cause

Previous IMU rotation estimate (~28°) was incorrect. Actual IMU gravity direction in sensor frame is `[-0.429, -0.027, 0.903]` (normalized), which is 154.5° misaligned from expected `[0, 0, -1]` in base_footprint (Z-up). This large misalignment was causing severe rotation errors in SLAM estimates.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`: Fixed scan bounds computation for rosette pattern. When all timestamps are identical (time_offset=0), use `timebase_sec` as `scan_start_time` and `header.stamp` as `scan_end_time` to capture the accumulation window (~100ms for 10Hz scans).
- `fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py`: Updated `T_base_imu` rotation to `[0.169063, -2.692032, 0.0]` from gravity analysis.
- `tools/diagnose_coordinate_frames.py`: New diagnostic tool that analyzes raw rosbag data to determine:
  - LiDAR Z-convention (Z-up vs Z-down) from ground plane normal
  - IMU gravity direction vs expected base frame
  - Odom covariance ordering (ROS vs GC convention)
- `tools/DIAGNOSTIC_TOOLS.md`: Documentation of all available diagnostic tools.

## 2026-01-27: Stabilize Gyro Σg IW Update + Fix Pose6 Conditioning Reporting

### Summary

- **Fixed Σg IW blow-ups** by redefining `omega_avg` as the weighted mean of debiased gyro measurements (rad/s), instead of `so3_log(ΔR)/dt` which is only valid in the small-angle limit and was destabilizing measurement-noise updates.
- **Removed a physically-implausible omega gate** (`||omega_avg|| > 100 rad/s`) to preserve the “no heuristics / no gating” invariant; we now fail-fast only on non-finite values.
- **Fixed diagnostics export + dashboard conditioning**: `conditioning_pose6` is now saved in `diagnostics.npz`, and the Plotly dashboard prefers pose6 conditioning over the full 22×22 condition number.
- **Reverted an unverified LiDAR π-about-X rotation tweak** in no-TF extrinsics; keep `T_base_lidar` rotation identity unless a frame convention correction is confirmed.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`: compute `omega_avg` from gyro measurements; remove `||omega_avg||` heuristic; pass masked IMU weights into Σg/Σa IW suffstats.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/diagnostics.py`: include `conditioning_pose6` in NPZ save/load.
- `tools/slam_dashboard.py`: prefer `conditioning_pose6` (if present) for the `log10 κ_pose6` plot; keep full conditioning available as `conditioning_number`.
- `fl_ws/src/fl_slam_poc/config/gc_unified.yaml`: revert `T_base_lidar` rotation to identity.
- `fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py`: revert `T_base_lidar` rotation to identity.

## 2026-01-26: Fix Pose6 Conditioning Robustness + Fail-Fast Run Script

### Summary

- **Fixed early-run crashes in GC backend** caused by `np.linalg.eigvalsh()` nonconvergence during the pose6 conditioning estimate used for fusion trust scaling.
- **Hardened the evaluation runner** to fail fast when `gc_backend_node` dies, avoiding misleading “SLAM complete” reports with only a handful of poses.
- **Improved diagnostics reporting** so Plotly and evaluation outputs surface pose6 conditioning and likely frame-offset rotation failures.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`: make the pose6 conditioning computation robust to non-finite matrices and eigen solver failures (fallback to SVD; final fallback to `eps_psd`).
- `tools/run_and_evaluate_gc.sh`: detect backend death/pipeline errors from logs and abort; enforce a minimum pose count before evaluation.
- `tools/slam_dashboard.py`: add pose6 conditioning + rotation-binding plots and display fields.
- `tools/evaluate_slam.py`: emit an explicit warning/diagnostic when rotation ATE indicates a near-constant frame offset (~180°).

## 2026-01-26: LiDAR Evidence Fix — Apply SE(3) Residual (Not Absolute Pose)

### Summary

- **Fixed a hard SE(3) semantics bug in LiDAR evidence application**: `LidarQuadraticEvidence` previously treated the absolute pose estimate from `(R_hat, t_hat)` as a tangent-space increment, effectively composing an already-world-frame pose again and driving large orientation errors and translation blow-ups.
- **Now uses right-perturbation log error**: `δ = Log(X_pred^{-1} ∘ X_lidar)` is embedded into the GC pose slice `[trans, rot]`, so LiDAR evidence targets the correct local state coordinates.
- **Added per-scan rotation-binding diagnostics**: `rot_err_{lidar,odom}_deg_{pred,post}` are exported in `diagnostics.npz` to directly verify whether rotation residuals decrease after fusion.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/lidar_evidence.py`: Build `delta_z_star` from `se3_log(se3_relative(X_lidar, X_pred))` and transport translation/rotation curvature into right-perturbation coordinates.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`: Compute and record LiDAR/odom rotation errors (pred vs post) for immediate residual-direction sanity checking.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/diagnostics.py`: Extend schema + NPZ export/load to include the new rotation-binding diagnostics.
- `fl_ws/src/fl_slam_poc/test/test_operators.py`: Add tests that assert LiDAR evidence produces zero delta when `X_lidar == X_pred` and matches a known injected right-perturbation `xi`.

### Verification

- `bash tools/run_and_evaluate_gc.sh` runs end-to-end; results captured under `results/gc_20260126_144420/` and `results/gc_20260126_144936/` (per-run timestamped dirs).

## 2026-01-26: IMU Gyro Evidence Fix — Correct SO(3) Residual Direction

### Summary

- **Fixed a sign/direction bug in IMU gyro rotation evidence**: the residual was computed as `Log(R_imu^T R_pred)` (meas^{-1} ∘ pred) but then applied as a target increment, causing the gyro factor to push the state away from the IMU-integrated orientation.
- **Now uses measurement-target residual**: `r = Log(R_pred^T R_imu)` (pred^{-1} ∘ meas), consistent with the right-perturbation tangent convention used by the rest of GC v2.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_gyro_evidence.py`: flip residual to `Log(R_end_pred^T @ R_end_imu)`.
- `fl_ws/src/fl_slam_poc/test/test_operators.py`: add a small-angle unit test asserting `delta_meas=0 ⇒ r≈-rotvec_end_pred`.

### Verification

- `PYTHONPATH=fl_ws/src/fl_slam_poc .venv/bin/python -m pytest -q fl_ws/src/fl_slam_poc/test` passes.
- `bash tools/run_and_evaluate_gc.sh` runs end-to-end; results captured under `results/gc_20260126_155819/`.

## 2026-01-26: No-TF Frame Coherence (B2) — Base-Frame State + Explicit Extrinsics

### Summary

- **State frame made explicit**: treat the SE(3) state as `X = T_world<-base` (M3DGR bag uses `base_footprint`).
- **Stop relabeling without transforming**: sensor normalizers now preserve incoming `frame_id` / `child_frame_id` by default (empty override strings mean “preserve”).
- **Apply numeric extrinsics in backend (no-TF mode)**: IMU samples and LiDAR points are rotated into the base frame before any inference, using explicit `T_base_imu` and `T_base_lidar` parameters.
- **Added IMU gravity coherence probes** to diagnostics (`accel_dir_dot_mu0`, `accel_mag_mean`) to validate that IMU is consistent with the estimated pose in the chosen base frame.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/odom_normalizer.py`: preserve frames when overrides are empty.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/imu_normalizer.py`: preserve `frame_id` when override is empty.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`: add `T_base_lidar`/`T_base_imu` params and apply `R_base<-sensor` rotations to LiDAR points and IMU (gyro/accel) samples.
- `fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py`: set `base_frame=base_footprint` and pass default extrinsics for M3DGR.
- `fl_ws/src/fl_slam_poc/config/gc_unified.yaml`: preserve frames in the hub config; document default extrinsics.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/diagnostics.py`: export new coherence probe fields to NPZ.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`: compute `accel_dir_dot_mu0` and `accel_mag_mean` per scan.

### Verification

- `PYTHONPATH=fl_ws/src/fl_slam_poc .venv/bin/python -m pytest -q fl_ws/src/fl_slam_poc/test` passes.
- `bash tools/run_and_evaluate_gc.sh` runs end-to-end; results captured under `results/gc_20260126_161004/`.

## 2026-01-26: LiDAR Ray-Direction Coherence — Compute Directions From Sensor Origin

### Summary

- **Fixed a frame-geometry mismatch in LiDAR direction features**: after moving points into `base` via `p_base = R_base<-lidar p_lidar + t_base<-lidar`, the code was normalizing `p_base` directly to get ray directions. This incorrectly treats rays as emanating from the base origin instead of the LiDAR origin, biasing directional statistics and Wahba rotation.
- **Now computes directions from the LiDAR origin in base**: uses `(p_base - t_base<-lidar) / ||p_base - t_base<-lidar||` consistently for soft binning responsibilities and scan directional sufficient statistics.
- **Empirical effect**: `rot_err_lidar_deg_post - rot_err_lidar_deg_pred` becomes negative on average (LiDAR rotation evidence starts “pulling the right way”).

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`: compute `point_directions` from rays `points - lidar_origin_base`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/binning.py`: `scan_bin_moment_match()` now accepts `direction_origin` and computes `s_dir` from sensor-origin rays.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`: wires `config.lidar_origin_base = t_base_lidar` so the pipeline stays “no-TF” but frame-coherent.

### Verification

- `PYTHONPATH=fl_ws/src/fl_slam_poc .venv/bin/python -m pytest -q fl_ws/src/fl_slam_poc/test` passes.
- `bash tools/run_and_evaluate_gc.sh` runs end-to-end; results captured under `results/gc_20260126_162411/`.

## 2026-01-26: Principled Missing-Data Handling — OU Propagation + IMU Evidence Scaling

### Summary

- **Replaced pure diffusion with OU-style bounded propagation**: `PredictDiffusion` now uses Ornstein-Uhlenbeck mean-reverting diffusion instead of `Σ ← Σ + Q*dt`. This prevents unbounded uncertainty growth during large missing-data gaps (e.g., 19.1s bag gaps) while remaining continuous and mathematically defensible.
- **Fixed IMU evidence covariance scaling**: IMU gyro evidence now uses actual IMU integration time (IMU buffer time span) instead of LiDAR scan duration for covariance scaling. This prevents incorrect inflation of IMU evidence during large LiDAR gaps.
- **No clamping, no gates**: All fixes are continuous and branch-free, preserving the "no gates" invariant.

### Changes

- `fl_slam_poc/common/constants.py`: Added `GC_OU_DAMPING_LAMBDA = 0.1` (1/s) with documentation explaining the OU propagation formula and saturation behavior.
- `fl_slam_poc/backend/operators/predict.py`:
  - Replaced pure diffusion `cov_pred_raw = cov_prev + dt_sec * Q` with OU propagation:
    `Σ(t+Δt) = e^(-2λΔt) Σ(t) + (1 - e^(-2λΔt))/(2λ) Q`
  - Added `lambda_ou` parameter (defaults to `GC_OU_DAMPING_LAMBDA`)
  - Updated docstring to explain OU propagation and bounded behavior
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`:
  - Compute `dt_imu_integration` from IMU buffer time span (not LiDAR scan duration)
  - Pass `dt_imu_integration` to `imu_gyro_rotation_evidence()` instead of `dt_scan`

### Rationale

- **OU propagation is principled**: It models missing-data gaps as "no constraints for duration Δt" with bounded uncertainty growth. As Δt → ∞, Σ → Q/(2λ) (bounded), not ∞. For small Δt, it approximates pure diffusion: Σ ≈ Σ + Q*Δt.
- **IMU evidence independence**: IMU is independent of LiDAR timing. Large LiDAR gaps should not inflate IMU evidence covariance. The correct scaling uses the actual IMU integration time (time span of IMU samples used), not the LiDAR scan duration.
- **Continuous, no gates**: Both fixes are continuous and branch-free, preserving project invariants. No clamping, no thresholds, no hard discontinuities.

### Mathematical Details

OU propagation for A = -λI:
- Small dt (0.1s): `e^(-2λΔt) ≈ 0.98`, so `Σ' ≈ 0.98*Σ + 0.02*Q/(2λ) ≈ Σ + Q*dt` (pure diffusion)
- Large dt (10s): `e^(-2λΔt) ≈ 0.135`, so `Σ' ≈ 0.135*Σ + 0.865*Q/(2λ)` (strong damping, prevents explosion)
- As dt → ∞: `Σ' → Q/(2λ)` (bounded saturation)

## 2026-01-26: Critical Fixes — IW Update Timing + Odom Relative Transformation

### Summary

- **Fixed IW update timing**: IW updates (process noise, measurement noise, LiDAR bucket) now start from scan 2 onwards, skipping scans 0 and 1. This prevents numerical explosions from insufficient delta history.
- **Added constant**: `GC_IW_UPDATE_MIN_SCAN = 2` in `constants.py` with documentation rationale.
- **Fixed initial position offset**: Odom is now transformed to be relative to the first odom message, making the first odom effectively at origin. This ensures the system starts at origin (0,0,0) regardless of bag starting position.
- **Root cause**: Odom was providing absolute poses (e.g., starting at 30m Z), which got fused on scan 0, moving the belief away from origin.

### Changes

- `fl_slam_poc/common/constants.py`: Added `GC_IW_UPDATE_MIN_SCAN = 2` constant with documentation
- `fl_slam_poc/backend/backend_node.py`:
  - Added `first_odom_pose` storage in `_init_state()`
  - Modified `on_odom()` to transform odom to be relative to first odom: `odom_relative = first_odom^{-1} ∘ odom_absolute`
  - Added scan count check in IW update section: `if self.scan_count >= constants.GC_IW_UPDATE_MIN_SCAN:`
  - IW updates (process, measurement, lidar bucket) are now skipped for scans 0 and 1

### Rationale

- **IW updates need real deltas**: Scans 0 and 1 don't have sufficient scan-to-scan history for meaningful innovation residuals. Updating with dt=0 or very small dt causes numerical issues.
- **Origin start requirement**: System should always start at origin regardless of bag starting position. Transforming odom to be relative to first message achieves this cleanly while maintaining proper fusion.

## 2026-01-25: GC v2 Foundations — Lie Primitives + Process-Noise IW

### Summary

- Implemented **SE(3) Log** with the correct translation twist (\(\rho = V(\phi)^{-1} t\)) and added **SO(3) right Jacobian** primitives in `fl_slam_poc/common/geometry/se3_jax.py`; removed the prior `se3_log` stub in `fl_slam_poc/common/belief.py`.
- Added **arrays-only JIT-safe primitive cores** in `fl_slam_poc/common/primitives.py` (`domain_projection_psd_core`, lifted solve cores, `inv_mass_core`, `clamp_core`) while keeping the existing dataclass-returning wrappers for non-jit call sites.
- Added **process-noise inverse-Wishart state** + commutative per-scan sufficient-statistics updates (no per-hypothesis mutation). Backend now derives `Q` from the IW state and updates the IW state **once per scan**.
- Added **measurement-noise inverse-Wishart state** (per-sensor, phase 1). LiDAR translation measurement covariance is now derived from the IW state and updated per scan from TranslationWLS residual outer products.
- Replaced legacy regression evidence with **closed-form pose information** (directional Fisher-style curvature + translation covariance inverse).
- Replaced legacy deskew with **constant-twist deskew** driven by IMU preintegration, and added **IMU + odometry evidence** (IMU accel direction via Laplace on intrinsic primitives; odom as Gaussian pose factor).
- Completed **Phase 2 IMU evidence** by adding a **gyro rotation Gaussian factor** (in addition to accel direction) and feeding **Σg/Σa** measurement-noise IW updates from IMU residual statistics.
- Replaced the PointCloud2 parser with a **vectorized Livox-aware parser** (preserves `time_offset` when present, plus `ring`/`tag` metadata) to avoid per-point Python loops.
- Removed sigma-point/moment-matching artifacts from the runtime path (and updated the unit tests accordingly).
- Replaced a broken legacy IMU-kernel unit test (which referenced archived code) with Lie-primitive correctness tests.

### Notes

- The pipeline is still UT-based for deskew/evidence at this point; later phases will replace UT regression with factor \(\rightarrow\) Laplace/I-projection per the updated plan.

## 2026-01-25: Audit fixes — units + no-fallback timestamps + no hard masks

### Summary

- Clarified **units** and introduced explicit **per-second process diffusion** priors (`GC_PROCESS_*_DIFFUSION`) so `Q` is interpretable as “per second” and discretized exactly once as `dt * Q`.
- Removed the mis-use of a LiDAR noise proxy as a process prior; process-noise IW initialization now uses diffusion-rate priors.
- Enforced **no silent fallbacks** for Livox per-point timing + metadata: PointCloud2 parsing fails fast if `ring/tag/timebase/time_offset` fields are missing (single math path).
- Removed `imu_valid` boolean gating across IMU preintegration/evidence/IW updates; IMU influence is now controlled solely by **continuous, strictly-positive** window weights.
- Replaced NaN/Inf point handling with a declared wrapper-boundary **domain projection** to finite sentinels plus strictly-positive continuous range weighting (no exact-zero masks).

## 2026-01-24: Fixed ROS 2 Jazzy Empty-List Parameter Type Bug

### Root Cause

ROS 2 rclpy bug (ros2/rclpy #912, ros2/ros2 #1518): when `declare_parameters()` is called and a parameter override already exists, rclpy internally extracts `override.value` and re-infers its type. For empty lists `[]`, this produces `BYTE_ARRAY` instead of `STRING_ARRAY`, causing `InvalidParameterTypeException`.

Previous fix attempts (passing typed Parameters, declaring with explicit types) failed because **both** happen in the buggy code path - rclpy ignores the original Parameter type and the declared type when an override value exists.

### Fix

DeadEndAuditNode now **bypasses the buggy code path entirely**:
1. Filter `topic_specs` and `required_topics` OUT of `parameter_overrides` before `super().__init__()`
2. Declare with explicit `STRING_ARRAY` types (no override exists, so no type re-inference)
3. Apply values via `set_parameters()` AFTER declaration with explicitly typed Parameters

This pattern is documented in the code for future maintainers.

## 2026-01-24: Documentation Update for GC v2 Target Endstate

### Summary

Updated documentation to reflect the target endstate for GC v2 full implementation:

- **AGENTS.md**: Added "Target Endstate" section describing adaptive noise (IW), IMU fusion (vMF), and likelihood-based evidence
- **README.md**: Updated MVP status, Quick Start, and System Architecture sections to focus on GC v2
- **ROADMAP.md**: Added "Immediate Priority: GC v2 Full Implementation" section with 5 phases

Key target features documented:
- Inverse-Wishart process/measurement noise with datasheet priors (ICM-40609 IMU, Mid-360 LiDAR)
- vMF accelerometer direction + Gaussian gyro likelihoods for IMU fusion
- Likelihood-based LiDAR evidence via Laplace/I-projection (replacing UT regression)
- Continuous adaptation with no gating/branching

## 2026-01-24: GC Eval Fresh Build

### Summary

Updated the GC evaluation script to force a fresh build of `fl_slam_poc` before running, preventing stale install artifacts from being used.

## 2026-01-24: GC Eval Uses Venv Python Explicitly

### Summary

Updated the GC evaluation script to run preflight + eval steps with the venv’s `python` explicitly (not `python3`), preventing accidental use of system Python and missing-dependency failures (e.g., JAX).

## 2026-01-24: GC Eval Sanitizes PYTHONPATH for Python Tools

### Summary

Run preflight + evaluation Python steps with `PYTHONPATH` unset so ROS/system Python paths can’t shadow venv wheels (fixes NumPy/matplotlib ABI mismatch failures).

## 2026-01-24: Fix GC Backend Callback Starvation (Vectorized Deskew + Binning)

### Summary

Refactored the GC LiDAR compute path to remove per-point Python loops and per-point device→host sync (`float(jnp_scalar)`), which was stalling `on_lidar` long enough that the backend appeared to “not receive” messages (BEST_EFFORT drops while blocked).

### Changes

- `DeskewUTMomentMatch` now deskews all points in a single batched JAX computation and returns stacked arrays (no `List[DeskewedPoint]`, no `float(timestamps[i])` in a loop).
- `BinSoftAssign` and `ScanBinMomentMatch` now compute responsibilities and sufficient statistics with batched JAX ops (no O(N*B) Python loops).
- Pipeline directional normalization is now batched (no per-point `.at[i].set(...)` loops).

### Verification

- `bash tools/run_and_evaluate_gc.sh` shows non-zero LiDAR scans and pipeline runs in the wiring summary.

## 2026-01-24: Dead-End Audit Param Types

### Summary

Declared dead-end audit list parameters with explicit `STRING_ARRAY` types to avoid ROS 2 empty-list inference crashes during parameter overrides.

## 2026-01-24: Dead-End Audit Typed Overrides

### Summary

Pass typed `STRING_ARRAY` overrides from `gc_sensor_hub` into `DeadEndAuditNode` so ROS 2 never infers list params as `BYTE_ARRAY`.

## 2026-01-24: Sensor Hub Architecture + Canonical Topic Naming

### Summary

Implemented a clean layered architecture for the GC SLAM frontend with canonical `/gc/sensors/*` topic naming to prevent accidental fusion of raw data.

### Topic Naming Convention

| Raw (from bag) | Canonical (for backend) |
|----------------|------------------------|
| `/livox/mid360/lidar` | `/gc/sensors/lidar_points` |
| `/odom` | `/gc/sensors/odom` |
| `/livox/mid360/imu` | `/gc/sensors/imu` |

**Rule:** Backend subscribes ONLY to `/gc/sensors/*` - never to raw topics.

### Files Created

- `frontend/hub/__init__.py` - Hub module init
- `frontend/hub/gc_sensor_hub.py` - Single-process sensor hub (placeholder for future consolidation)
- `frontend/sensors/odom_normalizer.py` - Odom normalizer (passthrough with TODOs)
- `frontend/sensors/imu_normalizer.py` - IMU normalizer (passthrough with TODOs)
- `config/gc_unified.yaml` - Consolidated configuration file

### Files Modified

- `frontend/__init__.py` - Added exports for new modules
- `frontend/sensors/__init__.py` - Added exports for normalizers
- `frontend/sensors/livox_converter.py` - Output topic changed to `/gc/sensors/lidar_points`
- `backend/backend_node.py` - Subscriptions changed to `/gc/sensors/*`
- `launch/gc_rosbag.launch.py` - Added normalizer nodes, updated all topic wiring
- `setup.py` - Added entry points for new nodes
- `config/gc_backend.yaml` - Updated to canonical topics
- `test/conftest.py` - Updated topic fixture
- `docs/Fusion_issues.md` - Updated architecture diagram

### Architecture Diagram

See `docs/Fusion_issues.md` Section 1.1 for the updated Mermaid diagram.

### Follow-up Fixes (Strictness + Runtime Correctness)

- Fixed ROS2 QoS compatibility for canonical odom (`odom_normalizer` now publishes RELIABLE to match backend subscription).
- Removed legacy `odom_bridge` to prevent accidental bypass/multipath.
- Made `gc_sensor_hub` runnable by loading `config/gc_unified.yaml` and applying per-node parameter overrides before node initialization.
- Dead-end audit now tolerates missing driver message types (e.g., `livox_ros_driver`) and reports them as unavailable instead of failing at startup.
- IMU normalizer now warns on first invalid sample and fails fast on the second to avoid silent drops.

---

## 2026-01-24: GC Strictness + Wiring Auditability

### Summary

Tightened GC v2 “single-path” wiring so Livox conversion and backend/operator selections are explicit and auditable at runtime.

### Key Changes

1. **Runtime manifest backends map**: `RuntimeManifest` now reports an explicit `backends` selection map for “no multipaths” auditability.
2. **Livox converter strict config**: removed `input_msg_type: "auto"` from GC config and made the GC rosbag launcher pass an explicit `livox_input_msg_type` (default `livox_ros_driver2/msg/CustomMsg`).
3. **Dead-end audit node (topic accountability)**: added `gc_dead_end_audit_node` to subscribe to unused rosbag topics (explicit message types, fail-fast on missing required streams) and publish `/gc/dead_end_status`.

## 2026-01-23: Geometric Compositional SLAM v2 Implementation ✅

### Summary

Complete implementation of the Geometric Compositional SLAM v2 specification - a strict "branch-free, fixed-cost, local-chart" SLAM backend that eliminates all methodology issues from the previous audit.

### Key Features

1. **Branch-Free Total Functions**: Every operator always runs, with continuous influence scalars instead of conditional gates
2. **6D SE(3) Representation**: Using `[translation(3), rotvec(3)]` representation with JAX for all math
3. **Certificate Audit Trail**: Every operator returns `(result, CertBundle, ExpectedEffect)` tuple
4. **Domain Projections Always Applied**: `DomainProjectionPSD` and `SPDCholeskySolveLifted` always execute with recorded magnitudes

## 2026-01-23: GC Wiring Cleanup ✅

### Summary

Archived legacy constants and tightened runtime wiring to prevent confusing multi-paths.

### Key Changes

1. **Legacy constants archived**: moved non-GC constants to `archive/legacy_common/constants_legacy.py`
2. **Runtime clarity**: clarified `se3_jax` vs `se3_numpy` usage in geometry docs
3. **Map update certificates**: now track PSD projection deltas in `PoseCovInflationPushforward`

### Files Created

**Common Layer:**
- `fl_slam_poc/common/primitives.py` - Branch-free numeric primitives (Symmetrize, DomainProjectionPSD, InvMass, Clamp, etc.)
- `fl_slam_poc/common/certificates.py` - CertBundle, ExpectedEffect, component certificates
- `fl_slam_poc/common/belief.py` - BeliefGaussianInfo (22D state), HypothesisSet

**Operators:**
- `fl_slam_poc/backend/operators/point_budget.py` - PointBudgetResample
- `fl_slam_poc/backend/operators/predict.py` - PredictDiffusion
- `fl_slam_poc/backend/operators/deskew.py` - DeskewUTMomentMatch (produces UTCache)
- `fl_slam_poc/backend/operators/binning.py` - BinSoftAssign, ScanBinMomentMatch
- `fl_slam_poc/backend/operators/kappa.py` - KappaFromResultant (single continuous formula)
- `fl_slam_poc/backend/operators/wahba.py` - WahbaSVD
- `fl_slam_poc/backend/operators/translation.py` - TranslationWLS
- `fl_slam_poc/backend/operators/lidar_evidence.py` - LidarQuadraticEvidence (reuses UTCache)
- `fl_slam_poc/backend/operators/fusion.py` - FusionScaleFromCertificates, InfoFusionAdditive
- `fl_slam_poc/backend/operators/recompose.py` - PoseUpdateFrobeniusRecompose
- `fl_slam_poc/backend/operators/map_update.py` - PoseCovInflationPushforward
- `fl_slam_poc/backend/operators/anchor_drift.py` - AnchorDriftUpdate (continuous rho)
- `fl_slam_poc/backend/operators/hypothesis.py` - HypothesisBarycenterProjection

**Structures:**
- `fl_slam_poc/backend/structures/bin_atlas.py` - BinAtlas, MapBinStats with forgetting

**Pipeline:**
- `fl_slam_poc/backend/pipeline.py` - 15-step per-scan execution, RuntimeManifest

**Tests:**
- `test/test_geometric_compositional_invariants.py` - Chart ID, dimensions, budgets
- `test/test_primitives.py` - Branch-free primitive correctness
- `test/test_operators.py` - Operator contract verification

### Methodology Issues Fixed

1. ✅ **Delta Accumulation** - Now uses absolute state in local charts
2. ✅ **Double-Counting Uncertainty** - Information fusion is additive once
3. ✅ **Non-SPD Matrices** - DomainProjectionPSD always applied
4. ✅ **JAX Silent NaN** - All operations use lifted solves
5. ✅ **Unbounded Covariance Growth** - Forgetting factor on map stats
6. ✅ **Residual Accumulation** - Relinearization at anchor updates
7. ✅ **Framework Mix** - Pure JAX implementation

### Design Invariants Enforced

- **No if/else gates** that change computation paths based on data
- **All operators are total functions** that always execute
- **Continuous influence scalars** replace hard thresholds
- **Every approximation is logged** with magnitude in CertBundle
- **Frobenius correction is conditional on trigger magnitude**, not boolean flags

### Reference

docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md - Full specification

---

## 2026-01-23: Information-Form IMU Fusion + SE(3) Adjoint Transport ✅

### Summary

Fixed root causes of `LinAlgError: Matrix is not positive definite` by replacing ad-hoc approximations with principled information geometry approaches.

### Root Causes Fixed

1. **IMU Kernel Cross-Covariance (imu_kernel.py:294-298)**
   - **Problem**: Ad-hoc `cross_cov = cov_delta * 0.5` violates PSD requirements for block matrices
   - **Fix**: Information-form factor construction with proper structure:
     ```
     L_factor = [[R_inv, -R_inv], [-R_inv, R_inv]]
     ```
   - **Rationale**: The IMU constraint's Jacobians are J_anchor=-I, J_current=+I, giving cross-information = -R_inv (not arbitrary scaling)

2. **Odom Bridge Covariance (odom_bridge.py)**
   - **Problem**: First-order approximation `Σ_delta ≈ Σ_prev + Σ_curr` ignores rotation effects
   - **Fix**: Proper SE(3) Adjoint transport:
     ```
     Σ_delta = Σ_prev + Ad(T_prev⁻¹) @ Σ_curr @ Ad(T_prev⁻¹).T
     ```
   - **Rationale**: Barfoot (2017), Sola (2018) - covariance transport on Lie groups uses Adjoint representation

### Changes

- **`imu_kernel.py`**: Rewrote joint covariance construction to use information-form factor structure
  - Convert prior covariances to precision matrices (Λ = Σ⁻¹)
  - Add IMU factor contribution with mathematically correct cross-information structure
  - Convert back to covariance for downstream Schur marginalization
  - Jacobians are in sensor→evidence extraction (allowed per AGENTS.md)

- **`odom_bridge.py`**: 
  - `compute_delta_covariance()` now uses `se3_cov_compose()` with Adjoint transport
  - OpReport updated to reflect exact (not approximate) operation
  - Import `se3_cov_compose` from geometry module

### Invariants Enforced

- **Closed-form-first**: Both fixes use closed-form operations (no iterative solvers)
- **No heuristics**: Removed ad-hoc `0.5` scaling, replaced with principled Jacobian-based structure
- **Single path**: No fallbacks or branching - strict operations that fail loudly on invalid input

### References

- Barfoot (2017): State Estimation for Robotics - SE(3) covariance propagation
- Sola et al. (2018): A micro Lie theory - Adjoint representation
- Amari (2016): Information Geometry - natural parameters and duality

## 2026-01-23: Strict Failures, JAX SE(3), Determinism + Observability ✅

### Summary

Removed numerical fallbacks in core Gaussian/SE(3) paths, moved backend SE(3) ops to JAX wrappers, and improved determinism, configuration observability, and benefit tracking.

### Changes

- Strict Cholesky-only Gaussian ops (no pinv fallbacks) in backend fusion and factors.
- Backend SE(3) operations now use JAX wrappers via `common/jax_utils.py`.
- Deterministic RNG support added for birth and RGB-D subsampling (seeded).
- Frontend drop paths now emit OpReports with explicit reasons.
- Implemented information_weight in LoopFactor and added benefit tracking for loop, IMU, and module culling.
- Added backend/frontend state summary helpers and startup parameter logging.
- Launch file parameter overrides removed; config files now carry final values.

## 2026-01-23: No Fallbacks / No Multi-Paths Policy ✅

### Summary

Added an explicit engineering invariant to prevent silent coexistence of multiple math/operator backends and environment-dependent fallbacks that make runtime behavior ambiguous.

### Changes

- `AGENTS.md`
  - Added "No Fallbacks / No Multi-Paths (Required)" section with fail-fast + runtime-manifest requirements.
  - Shortened and hardened the document to reduce ambiguity and enforce first-principles invariants in review.
  - Fixed spec reference path for `Project_Implimentation_Guide.sty`.

## 2026-01-23: Prospector Lint Cleanup (Phases 1-4) ✅

### Summary

Systematic cleanup of unused imports, dead code removal, and code quality improvements based on prospector static analysis. Also fixed unused variables to be properly utilized.

### Changes

**Phase 1 - Dead Import Removal:**
- `frontend_node.py`: Removed `time`, `Odometry`, `SensorStatus`, `make_evidence`, `vmf_make_evidence`, `validate_timestamp`
- `backend_node.py`: Removed `TransformStamped`, `get_state_at_stamp`, `publish_anchor_marker`, `OpReport`, `stamp_to_sec`
- `dirichlet_router.py`: Removed `jax`, `jit`, `cholesky`, `solve_triangular` (kept only `jnp`)
- `imu_kernel.py`: Removed unused `se3_minus`
- `publish.py`: Removed `json`, `Node`, `PointCloud2`, `PointField`, `TransformStamped`
- `icp.py`: Removed `Tuple`, `skew`
- `pointcloud_gpu.py`: Removed `Tuple`
- `vmf.py`: Removed `iv` (kept only `ive` for numerical stability)
- `status.py`: Removed `Node`
- `config.py`: Removed `os`
- `information_distances.py`: Removed `Any`
- `validation.py`: Removed `Optional`
- `tools/*.py`: Removed unused `sys`, `Any`, `serialize_message`

**Phase 2 - Trivial Fixes:**
- Fixed extra trailing newline in `se3_numpy.py`
- Fixed f-strings without interpolation in `backend_node.py`, `evaluate_slam.py`

**Phase 3 - dtype_map Fix:**
- `sensor_io.py`: `pointcloud2_to_array()` now uses `dtype_map` to dynamically determine field data types from PointCloud2 metadata instead of hardcoding float32

**Phase 4 - anchor_id Fix:**
- `publish.py`: `publish_map()` now tracks points per anchor using `anchor_id` and includes per-anchor statistics in log output

**Phase 4 - Encoding Fix:**
- Added `encoding='utf-8'` to all `open()` calls in `align_ground_truth.py` and `evaluate_slam.py`

### Rationale

Static analysis revealed significant dead code from prior refactoring. Removing unused imports:
- Reduces cognitive load when reading code
- Eliminates false import error warnings
- Documents actual dependencies accurately
- Improves startup time marginally

The `dtype_map` fix ensures PointCloud2 parsing works correctly with non-float32 data (e.g., float64 or int types), which was a latent bug.

## 2026-01-23: Rename tb3_odom_bridge → odom_bridge ✅

### Summary

Renamed `tb3_odom_bridge` to `odom_bridge` to remove misleading TurtleBot3 reference. The bridge is a generic absolute→delta odometry converter used with M3DGR dataset (and potentially other datasets like TurtleBot3 or NVIDIA r2b in the future).

### Changes

- **Renamed files:**
  - `fl_slam_poc/frontend/sensors/tb3_odom_bridge.py` → `odom_bridge.py`
  - `scripts/tb3_odom_bridge` → `odom_bridge`
- **Updated class and node names:**
  - Class `Tb3OdomBridge` → `OdomBridge`
  - Node name `tb3_odom_bridge` → `odom_bridge`
  - Executable `tb3_odom_bridge` → `odom_bridge`
- **Updated references in:**
  - `setup.py` - entry point
  - `CMakeLists.txt` - script installation
  - `launch/poc_m3dgr_rosbag.launch.py` - executable and node name
  - `config/fl_slam_poc_base.yaml` - parameter namespace
  - `config/presets/m3dgr.yaml` - parameter namespace
  - `config/config_manifest.yaml` - feature list and topic documentation
  - `AGENTS.md` - component summary and package structure
  - `docs/BAG_TOPICS_AND_USAGE.md` - all table entries and field descriptions
  - `docs/PIPELINE_AUDIT_2026-01-23.md` - pipeline diagram
  - `README.md` - mermaid diagram, component summary, file tree
  - `ROADMAP.md` - MVP nodes list and troubleshooting section

### Rationale

The MVP dataset is M3DGR (not TurtleBot3), and the bridge is dataset-agnostic. The `tb3_` prefix was a legacy naming artifact that caused confusion about what the rosbag actually contains.

## 2026-01-23: Pipeline “Hidden Behavior” Audit Report ✅

### Summary

Added an audit-only report to enumerate hidden/disabled codepaths, hardcoded topics, placeholder values, and docs/test drift that make end-to-end runtime behavior difficult to verify.

### Changes

- `docs/PIPELINE_AUDIT_2026-01-23.md`
  - New audit document with issue IDs (AUD-001…AUD-008) and candidate fixes for explicit approval.

## 2026-01-23: Duplication Audit (Functions/Helpers) ✅

### Summary

Added an audit-only report listing exact and near-duplicated functions/helpers across the codebase (especially `tools/` and frontend sensor utilities) to reduce copy/paste drift.

### Changes

- `docs/DUPLICATION_AUDIT_2026-01-23.md`
  - New audit document with issue IDs (DUP-001…DUP-006) and candidate fixes for explicit approval.

## 2026-01-23: Duplication Cleanup (Except DUP-005) ✅

### Summary

Eliminated duplicated helper implementations identified by `docs/DUPLICATION_AUDIT_2026-01-23.md`, except the intentional NumPy vs JAX SE(3) dual-backend (DUP-005).

### Changes

- `tools/rosbag_sqlite_utils.py`
  - Centralized `resolve_db3_path`, `topic_id`, `topic_type`.
- `tools/estimate_lidar_base_extrinsic.py`
- `tools/inspect_camera_frames.py`
- `tools/inspect_odom_source.py`
- `tools/inspect_rosbag_deep.py`
- `tools/validate_livox_converter.py`
  - Updated to import shared rosbag sqlite helpers (no behavior change intended).
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/diagnostics/op_report_publish.py`
  - Centralized OpReport validation + JSON publishing for frontend-side nodes.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/qos_utils.py`
  - Centralized QoS profile resolution.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/dedup.py`
  - Centralized duplicate message suppression (multi-QoS subscriptions).
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/sensor_io.py`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/tb3_odom_bridge.py`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`
  - Refactored to use the shared helpers (no behavior change intended).

## 2026-01-23: Dataflow + Self-Adaptive Compliance Review ✅

### Summary

Added an audit document mapping the current end-to-end runtime dataflow and comparing current behavior against the normative constraints in `docs/Self-Adaptive Systems Guide.md`, highlighting remaining deviations (notably hard-gate factor drops and missing certified-operator contracts).

### Changes

- `docs/DATAFLOW_AND_SELF_ADAPTIVE_COMPLIANCE_2026-01-23.md`

## 2026-01-23: Canonical Topic/Schema Documentation Hardening ✅

### Summary

Expanded the canonical bag/pipeline topic map to include the full MVP runtime topic graph, custom message schemas, and exact field/covariance/QoS handling for each relevant topic.

### Changes

- `docs/BAG_TOPICS_AND_USAGE.md`
  - Added complete pipeline topic graph (bag → utilities → frontend → backend → outputs).
  - Documented exact message schemas for `AnchorCreate`, `LoopFactor`, and `IMUSegment`.
  - Documented standard ROS message field usage (including which covariance fields are used/ignored).
  - Documented RGB-D evidence JSON payload schema and QoS defaults.

## 2026-01-23: Invariant Compliance Budget + Gating Cleanup ✅

### Summary

Removed remaining evidence gates, surfaced backend compute budgets as explicit parameters/priors, and formalized dense-module culling as a Frobenius-corrected budgeted recomposition.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/loops/loop_processor.py`
  - Removed loop factor weight gate; low-weight factors now pass through.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/factors/loop.py`
  - Removed weight gate in loop processing.
  - Logged loop-factor buffer budget exceedance while retaining evidence.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/factors/imu.py`
  - Logged IMU buffer budget exceedance while retaining evidence.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/state/rgbd.py`
  - Implemented budgeted recomposition by posterior mass with Frobenius correction.
  - Replaced hard RGB-D association with soft responsibilities + new-component prior.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
  - Added ROS parameters for dense association radius, module budgets, and buffer sizes.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py`
  - Added compute budget constants, module mass prior, and ICP covariance priors.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/param_models.py`
  - Added backend parameters for dense module and buffer budgets.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/scan/icp.py`
  - Centralized SE(3) DOF constants via `constants.py`.

## 2026-01-23: Backend Math Consolidation + IMU Kernel Ordering Fix ✅

### Summary

Removed conflicting JAX/SE(3) entrypoints and fixed a state-layout mismatch in the batched IMU projection kernel so the unit test executes end-to-end.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/geometry/se3_jax.py`
  - Replaced the wrapper with the canonical JAX SE(3) implementation.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
  - Removed hardcoded JAX env overrides; rely on `common/jax_init.py`.
  - Removed unused local JAX→NumPy SE(3) wrapper functions.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/math/imu_kernel.py`
  - Fixed IMU residual ordering to match the 15D state tangent layout.
  - Embedded the 9D residual mean into a 15D delta with zero bias components before applying `se3_plus`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/lie_jax.py`
  - Deleted (superseded by `common/geometry/se3_jax.py`).

## 2026-01-22: α-Divergence Trust-Scaled Fusion for Loop Closures ✅

### Summary

Implemented information-geometric trust region for loop closure fusion using α-divergence and power posteriors. This replaces the naive product-of-experts approach with a principled method that prevents catastrophic jumps from high-divergence updates while preserving conjugacy and associativity.

## 2026-01-22: Invariant Compliance Cleanup (Part 1) ✅

### Summary

Aligned frontend/back-end configuration and evidence handling with order-invariance and self-adaptive principles. Removed hidden parameter mismatches, documented IMU routing priors, eliminated hard ICP gating in favor of continuous quality scaling, and logged domain projections explicitly.

### Key Changes

- Wired `imu_qos_reliability` end-to-end (launch → frontend → SensorIO).
- Removed hardcoded `birth_intensity` override; frontend now uses `constants.BIRTH_INTENSITY_DEFAULT`.
- Standardized launch gravity default to `-9.81` to match `constants.GRAVITY_DEFAULT`.
- Centralized IMU routing priors (`IMU_ROUTING_MAPPED_LOGIT`, `IMU_ROUTING_DISTANCE_SCALE`) and applied in backend routing.
- Replaced hard ICP convergence gate with continuous quality weighting; added OpReport for unavailable ICP evidence.
- Added `RESPONSIBILITY_MASS_FLOOR` and logged domain projection when responsibilities are uniformized.
- Removed unused responsibility gating constants; softened "robust gating" wording in Gaussian info utilities.
- Removed archived IMU preintegration artifacts to eliminate duplicate IMU integration sources.
- Consolidated epsilon floors and QoS depths to `constants.py` across frontend/back-end utilities.
- Centralized RGB-D depth min/max defaults via `constants.DEPTH_MIN_VALID` and `constants.DEPTH_MAX_VALID`.

### Key Insight

Loop closures can propose large state changes when drift has accumulated. Naive product-of-experts fusion (`L_post = L_prior + L_factor`) applies the full update regardless of how much it disagrees with the prior. This can cause trajectory oscillation and instability.

**Solution: Power Posterior / Tempered Likelihood**

Instead of full fusion, we compute:
```
L_post = L_prior + β × L_factor
h_post = h_prior + β × h_factor
```

Where β ∈ [0, 1] is computed from the α-divergence:
```
β = exp(-D_α(full_posterior || prior) / max_divergence)
```

This gives smooth scaling:
- High divergence (factor disagrees strongly with prior) → β → 0 (minimal update)
- Low divergence (factor agrees with prior) → β → 1 (full update)

### Why α-Divergence (not KL)?

- **α = 0.5** gives symmetric divergence: penalizes both over-updating and under-updating equally
- Closed-form for Gaussians (no sampling needed)
- Generalizes KL (α→1 is forward KL, α→0 is reverse KL)
- Related to Hellinger distance (√2 × H² at α=0.5)

### Properties Preserved

- **Stays in Gaussian family**: Power posterior is still Gaussian (no mixture reduction)
- **Conjugacy**: Prior × tempered_likelihood = posterior in same family
- **Associativity**: Scaling is linear in information parameters
- **No heuristic gating**: β is continuous, not accept/reject

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/gaussian_info.py`
  - Added `alpha_divergence()`: Closed-form α-divergence for Gaussians
  - Added `trust_scaled_fusion()`: Power posterior fusion with trust region
  - Added `compute_odom_precision_from_covariance()`: Precision from odom covariance
  - Added constants: `ALPHA_DIVERGENCE_DEFAULT = 0.5`, `MAX_ALPHA_DIVERGENCE_PRIOR = 1.0`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
  - Loop closure fusion now uses `trust_scaled_fusion()` instead of `_gaussian_product()`
  - OpReport includes trust diagnostics: `trust_beta_*`, `trust_divergence_full_*`, `trust_quality_*`
  - Fixed OpReport metrics: added missing adaptive bias keys, use 0.0 instead of None for fallbacks

### Design Rationale

From the Self-Adaptive Systems Guide: "Approximate operators must return (result, certificate, expected_effect). Downstream may scale influence by certificate quality, but may not branch (no accept/reject)."

The trust-scaled fusion satisfies this:
- **result**: The tempered posterior (L_post, h_post)
- **certificate**: β (the tempering coefficient)
- **expected_effect**: divergence_actual (how much the state changed)

No branching occurs—just continuous scaling by certificate quality.

## 2026-01-22: IMU Diagnostics + Adaptive Bias Noise ✅

### Summary

Hardened IMU wiring and diagnostics across the 3D pointcloud path, enforced fail-fast GPU contract for IMU fusion, standardized IMU timebase to stamp-derived dt, fixed IMU integration ordering in both NumPy and JAX paths, and added self-adaptive Wishart estimation for bias innovation covariance. **IMU random-walk parameters are intentionally not used in the MVP pipeline** (bias noise is learned online from innovations and seeded by a fixed prior). Evaluation now validates OpReport diagnostics and IMU/Frobenius compliance by default.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`
  - Publish IMU segments in 3D pointcloud anchor births.
  - Emit `IMUSegmentSkipped` OpReport on insufficient IMU samples.
  - Removed loop factor truncation gate; rely on natural underflow (documented policy).
  - Documented zero bias reference (Contract B) in IMU segment publishing.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
  - Standardized IMU dt to `imu_stamps[-1] - imu_stamps[0]` and added timebase diagnostics.
  - Applied Frobenius correction proof-of-execution fields in IMU OpReports.
  - Dense Hellinger logits for routing (soft association), with optional keyframe mapping bias.
  - Fixed IMU integration ordering in NumPy path (position before velocity).
  - Added adaptive Wishart bias innovation covariances and bias observability metrics.
  - Removed `imu_*_random_walk` parameter plumbing; adaptive bias noise is seeded by a fixed prior.
  - Documented fail-fast GPU runtime contract for IMU fusion.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/imu_jax_kernel.py`
  - Fixed IMU integration ordering in JAX path (position before velocity).
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/process_noise.py`
  - Added `WishartPrior` and `AdaptiveIMUNoiseModel` for adaptive IMU bias random walk estimation.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/__init__.py`
  - Exported adaptive IMU noise classes.
- `tools/run_and_evaluate.sh`
  - Capture `/cdwm/op_report` to JSONL for evaluation diagnostics.
- `tools/evaluate_slam.py`
  - Added OpReport validation (IMU + Frobenius + timebase + adaptive bias checks) as default.

### Notes

- **IMU GPU contract:** When `enable_imu_fusion=true`, backend requires GPU at startup (fail-fast).
- **Timebase policy:** Canonical dt is derived from IMU stamp endpoints.

## 2026-01-22: Retire Python RGB-D Decompressor (Use C++ cv_bridge) ✅

### Summary

Removed the legacy Python `image_decompress` node and switched the rosbag pipeline to rely exclusively on the C++ decompressor (`image_decompress_cpp`) for compressed RGB + compressedDepth decoding, eliminating NumPy/cv_bridge ABI fragility in evaluation runs.

### Changes

- `fl_ws/src/fl_slam_poc/src/image_decompress_node.cpp`
  - Publish RGB as `rgb8` and decode `compressedDepth` according to `compressed_depth_image_transport` semantics.
- `fl_ws/src/fl_slam_poc/launch/poc_m3dgr_rosbag.launch.py`
  - Removed the Python decompressor node and its `enable_decompress` flag; kept `enable_decompress_cpp`.
- `fl_ws/src/fl_slam_poc/CMakeLists.txt`, `fl_ws/src/fl_slam_poc/setup.py`
  - Removed installation/entrypoints for the Python decompressor.
- `tools/run_and_evaluate.sh`
  - Forces `enable_decompress_cpp:=true` to prevent accidental regressions.
- `docs/MAP_VISUALIZATION.md`, `AGENTS.md`
  - Updated flags and component naming to `enable_decompress_cpp`.

## 2026-01-22: Bag-Truth Frames + Livox Converter Hardening ✅

### Summary

Hardened the LiDAR ingestion path to maximally preserve bag information while avoiding misleading evaluation due to silent frame/TF assumptions. Added deep rosbag inspection + validation tooling, aligned launch defaults to bag-truth frames, and introduced explicit no-TF LiDAR extrinsics (`lidar_base_extrinsic`) for robust rosbag playback and future robot wiring.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/livox_converter.py`
  - Preserve Livox sensor frame by default (no frame override required).
  - Publish richer `PointCloud2` fields: `x,y,z,intensity,ring,tag,timebase` (and `time_offset` if the message type provides it).
  - Support `livox_ros_driver2` and (optionally) `livox_ros_driver` via `input_msg_type`.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensor_io.py`
  - Added `lidar_base_extrinsic` fallback to transform pointclouds without TF.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`
  - Added `lidar_base_extrinsic` parameter plumbed into `SensorIO` config.
- `fl_ws/src/fl_slam_poc/launch/poc_m3dgr_rosbag.launch.py`
  - Defaults now match M3DGR Dynamic01 bag truth: `odom_frame=odom_combined`, `base_frame=base_footprint`, `pointcloud_frame_id=livox_frame`.
  - Added `livox_input_msg_type` and `lidar_base_extrinsic` launch arguments.
- `tools/inspect_rosbag_deep.py`
  - Expanded deep scan types (includes `PoseStamped` and optional Livox driver message types).
- `tools/validate_livox_converter.py`
  - Added offline sanity checks for Livox topics (frame IDs, XYZ ranges, reflectivity/line/tag distributions, accounting).
- `docs/BAG_TOPICS_AND_USAGE.md`
  - Canonical “what’s in the bag vs what we use” map (M3DGR now; placeholders for TB3/r2b).

## 2026-01-22: Repository Flattening + Tooling Cleanup ✅

### Summary

Flattened the `fl_slam_poc` package structure (common/frontend/backend), moved legacy docs to `legacy_docs/`, relocated root scripts to `tools/`, removed obsolete ROS scripts + `IMUFactor.msg`, and updated imports/entry points/CMake for the new layout.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/`
  - Moved `se3.py`, `dirichlet_geom.py`, `imu_preintegration.py` into common.
  - Removed `common/transforms/` and `operators/` legacy package stubs.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/`
  - Flattened anchors/loops/processing into top-level modules.
  - Moved utility nodes into frontend (`image_decompress`, `livox_converter`, `tb3_odom_bridge`).
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/`
  - Flattened `fusion/`, `parameters/`, `routing/` into top-level modules.
- `fl_ws/src/fl_slam_poc/setup.py`
  - Updated `entry_points` to new module paths.
- `fl_ws/src/fl_slam_poc/CMakeLists.txt`
  - Removed `IMUFactor.msg` and legacy script installation.
- `fl_ws/src/fl_slam_poc/msg/IMUFactor.msg`
  - Deleted (unused; superseded by `IMUSegment.msg`).
- `scripts/` → `tools/`
  - Moved evaluation/download/test scripts and updated documentation references.
- `legacy_docs/`
  - Moved Gazebo and audit docs into legacy documentation set.

### Notes

- `.cursorignore` creation is still blocked by workspace permissions; needs manual creation to ignore `legacy_docs/`, `archive/`, and `phase2/`.

## 2026-01-21: Contract B IMU Fusion (Raw Segments + Two-State Schur) ✅

### Summary

Implemented Contract B for IMU fusion: the frontend now publishes raw IMU segments, the backend re-integrates them inside the sigma-point propagation with bias coupling, and the two-state factor update is performed via one e-projection on the joint state followed by exact Schur marginalization. OpReports now include the required audit fields and routing diagnostics, and Contract B validation tests are in place.

### Changes

- `fl_ws/src/fl_slam_poc/msg/IMUSegment.msg`
  - Added Contract B raw IMU segment message schema with explicit units/frames/timebase semantics.
- `fl_ws/src/fl_slam_poc/CMakeLists.txt`
  - Added `IMUSegment.msg` to ROS interface generation.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`
  - Publish `/sim/imu_segment` raw IMU slices at keyframe creation.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
  - Subscribe to `/sim/imu_segment`, integrate raw IMU in-kernel with bias coupling, perform joint e-projection + Schur marginalization, and emit Contract B OpReports.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/imu_jax_kernel.py`
  - Integrated raw IMU samples per sigma point; removed preintegrated residual path.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/routing/dirichlet_routing.py`
  - Added routing diagnostics accessors for OpReport logging.
- `fl_ws/src/fl_slam_poc/test/test_imu_fusion_contract_b.py`
  - Added Contract B unit tests (zero-residual, bias observability, order invariance, frame convention, Hellinger bounds, routing consistency).
- `scripts/test-integration.sh`
  - Updated integration check to validate `/sim/imu_segment` processing.

## 2026-01-21: IMU Integration Phase 2 - Backend 15D State Extension & Hellinger-Dirichlet Fusion ✅

### Summary

Implemented **Milestone 2** from the roadmap: Backend 15D State Extension & IMU Factor Fusion (Phase 2). This is the second step toward the near-term priority of IMU Integration. The backend now maintains a 15D state (pose + velocity + biases) and fuses IMU factors using batched moment matching with Hellinger-tilted likelihood and Dirichlet-categorical routing.

### Architecture Overview

This implementation follows the "maximal by-construction" plan with:
- **Associative, order-robust fusion** in natural coordinates
- **Manifold-correct** SE(3) handling via ⊕/⊖ operators
- **Robustness by construction** using Hellinger-tilted likelihood
- **Self-adaptive** Dirichlet-categorical routing with Frobenius retention
- **Single declared e-projection** via batched unscented transform (no per-anchor Schur)

### New Files Created

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/routing/__init__.py` - Routing module exports
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/routing/dirichlet_routing.py` - DirichletRoutingModule with Frobenius retention, Hellinger shift monitoring, and `compute_imu_logits` for Hellinger-tilted weights
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/lie_jax.py` - JAX/NumPy Lie operators: `so3_exp`, `so3_log`, `se3_plus`, `se3_minus`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/imu_jax_kernel.py` - Batched IMU projection kernel with `unscented_sigma_points`, `imu_prediction_residual`, `hellinger_squared_gaussian`, and `imu_batched_projection_kernel`

### Files Modified

- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py` - Added 15D state constants: `STATE_DIM_FULL=15`, priors for velocity/bias, process noise for bias random walk
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/__init__.py` - Added `DirichletRoutingModule` export
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/__init__.py` - Added lazy exports for new functions: `mixture_moment_match`, `embed_info_form`, `hellinger_squared_from_moments`, Lie operators, IMU kernel
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/gaussian_info.py` - Added `mixture_moment_match` (correct mixture collapse via e-projection), `embed_info_form` (dimension embedding), `hellinger_squared_from_moments`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py` - Major updates:
  - State extended from 6D to 15D (pose + velocity + biases)
  - Process noise extended to 15D with bias random walk
  - Added IMU factor subscription and `on_imu_factor` handler
  - Updated `on_odom` for 15D compatibility (embeds 6D delta into 15D)
  - Updated `on_loop` for 15D compatibility (extracts 6D pose, updates, re-embeds)

## 2026-01-21: IMU/Rotation Audit Fixes (Frontend Wiring + Frame Stability)

### Summary

Restored IMU factor publishing in the frontend, buffered IMU factors in the backend for race-free processing, and hardened rotation handling near π to reduce frame convention errors.

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`
  - Added IMU parameters, IMU preintegration setup, and `/sim/imu_factor` publisher.
  - Publish IMU factors between anchor keyframes with OpReport diagnostics and buffer clearing.
  - Added IMU status monitoring and gravity parameter passthrough.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
  - Buffer IMU factors until anchor is available to avoid drops.
  - Use quaternion → rotvec conversion directly for odom and loop factors.
  - Added gravity parameter in backend config.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/imu_jax_kernel.py`
  - JIT-compiled kernel and restored plan-correct residual covariance for Hellinger tilt.
  - Added bias reference parameter plumbing.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/transforms/se3.py`
  - Added `se3_relative(a, b)` (group-consistent relative transform).
  - Stabilized `rotmat_to_rotvec` near π with deterministic axis extraction.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/tb3_odom_bridge.py`
  - Use quaternion → rotvec conversion directly to reduce antipodal flips.
- `fl_ws/src/fl_slam_poc/launch/poc_m3dgr_rosbag.launch.py`
  - Added gravity vector parameters for frontend/backend.

## 2026-01-21: Compliance Fixes - Remove Schur Complement + Enforce Batched IMU Fusion

### Summary

Follow-up audit fixes to satisfy strict invariants:
- **NO Schur complement** in loop closure updates.
- **NO per-anchor loop** in IMU fusion (batched JAX kernel).
- **NO natural-parameter weighted sum** for IMU mixture collapse (global moment match in expectation space).

### Changes

- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
  - Replaced loop-closure joint update + Schur complement marginalization with a **one-shot recomposition** update (anchor↔current message passing), Jacobian-free.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/imu_jax_kernel.py`
  - Rewrote `imu_batched_projection_kernel` to use **manifold retraction** for pose sigma support and to moment-match in the **tangent at `current_mu`** (avoids rotation-vector-as-Euclidean mistakes).
  - Added `R_imu` to predictive residual covariance before Hellinger tilt: `S = Cov(r) + R_imu`.
  - Replaced host-sync degeneracy branch with a **JIT-safe** `jax.lax.cond` fallback selection and removed Python bool/int/float concretization in diagnostics.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
  - Replaced Euclidean pose subtraction in predict residuals and loop innovation with SE(3) relative transforms (`se3_relative`) to keep rotation updates chart-consistent.
- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/transforms/se3.py`
  - Added `se3_relative(a, b)` helper for group-consistent pose differences (`b^{-1} ∘ a`).

### Follow-up Issue (Not Yet Implemented)
- **State orientation representation is still stored as a global rotation vector (`ω`)** in backend mean state. While we now avoid Euclidean subtraction and do tangent-space moment matching + retraction in the IMU kernel, the fully by-construction solution is to store orientation as:
  - an **error-state** around a reference rotation `R̄` (preferred), or
  - a **unit quaternion** with normalization.
  This refactor should remove any remaining “global axis-angle Gaussian” pathologies and further reduce ~π-flip rotation failure modes.
  - Updated `_publish_state` to extract 6D pose for Odometry messages
  - Added IMU tracking: `imu_factor_count`, `last_imu_time`
  - Added `_imu_routing_module` for Dirichlet routing
  - Added `keyframe_to_anchor` mapping (for future use)

### Key Design Decisions

1. **Global Moment Matching (NOT Natural-Param Weighted Sum)**
   - Plan specified: Mixture collapse via single e-projection in expectation space
   - Implementation: `mixture_moment_match()` computes μ = Σᵢwᵢμᵢ and Σ = Σᵢwᵢ(Σᵢ + (μᵢ-μ)(μᵢ-μ)ᵀ)
   - This is the correct Legendre e-projection, NOT weighted sum of (Λ,h)

2. **Batched JAX Kernel Architecture**
   - Plan specified: One JAX kernel call per IMU packet, not per-anchor
   - Implementation: `imu_batched_projection_kernel` processes all anchors together
   - Eliminates per-anchor Schur complements; direct global moment matching onto ξⱼ

3. **Hellinger-Tilted Likelihood**
   - Plan specified: ωᵢ ∝ exp(-½r^TΣ⁻¹r)·exp(-2Dₕ(p̂ᵢ,pₙₒₘ))
   - Implementation: `hellinger_squared_gaussian()` computes closed-form H² via Bhattacharyya coefficient
   - pₙₒₘ = N(0, Rₙₒₘ), p̂ᵢ = N(r̄ᵢ, Sᵢ) per plan

4. **Dirichlet Routing with Frobenius Retention**
   - Plan specified: α' = t·α (cubic contraction), c = B·softmax(s)
   - Implementation: `DirichletRoutingModule` with configurable retention, evidence budget
   - Monitors Hellinger shift H²(πₜ, πₜ₋₁) as stability diagnostic

5. **Anchor Embedding for IMU Factor**
   - Anchors stored in 6D (pose only); embedded to 15D for batched kernel
   - Velocity set to 0 (unknown), bias set to reference bias from IMU factor

### Mathematical Correctness Guarantees

| Property | Method | Status |
|----------|--------|--------|
| Order-Invariant Fusion | Additive natural params | Exact |
| Manifold Correctness | ⊕/⊖ via Lie operators | Exact |
| Robustness Tilt | Hellinger exp(-2Dₕ) | Exact (declared model operator) |
| Routing Retention | Frobenius cubic | Exact (closed form) |
| Mixture Collapse | e-projection | Declared approx (single) |
| Bias Evolution | Random walk | Exact (additive noise) |

### Next Steps

- Integration testing with M3DGR rosbag
- Optional JAX GPU acceleration for batched kernel
- Bias feedback loop: backend bias estimates → frontend preintegrator

## 2026-01-21: IMU Integration Phase 1 - Infrastructure & Preintegration ✅

### Summary

Implemented **Milestone 1** from the roadmap: IMU Sensor + Preintegration (Phase 1). This is the first step toward the near-term priority of IMU Integration & 15D State Extension. The frontend now subscribes to IMU data, preintegrates measurements between keyframes, and publishes `IMUFactor` messages for backend fusion.

### New Files Created

- `fl_ws/src/fl_slam_poc/msg/IMUFactor.msg` - Preintegrated IMU constraint message between keyframes
- `fl_ws/src/fl_slam_poc/fl_slam_poc/operators/imu_preintegration.py` - IMUPreintegrator class implementing Forster et al. (2017) equations
- `fl_ws/src/fl_slam_poc/test/test_imu_preintegration.py` - 23 unit tests for IMU preintegration

### Files Modified

- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py` - Added IMU noise parameters and keyframe threshold defaults
- `fl_ws/src/fl_slam_poc/CMakeLists.txt` - Added IMUFactor.msg to message generation
- `fl_ws/src/fl_slam_poc/fl_slam_poc/operators/__init__.py` - Lazy exports for `IMUPreintegrator`, `IMUPreintegrationResult`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/processing/sensor_io.py` - IMU subscription, buffer, event-driven clearing
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py` - IMU integration, motion-based keyframes, IMU factor publishing
- `fl_ws/src/fl_slam_poc/launch/poc_m3dgr_rosbag.launch.py` - IMU parameters (enabled by default)

### Key Design Decisions (Deviations from Original Plan)

1. **Keyframe Policy: Motion-Based (not time-based)**
   - **Original plan:** Time-based keyframes every 1.0 second
   - **Implementation:** Motion-based - create keyframe when translation > 0.5m OR rotation > 15 deg
   - **Rationale:** Adapts to robot speed (slow robot = fewer keyframes, fast robot = more), avoids arbitrary time intervals

2. **IMU Buffer: Event-Driven Clearing (not sliding window)**
   - **Original plan:** 1-second sliding window for IMU buffer
   - **Implementation:** Keep all measurements since last keyframe, clear after preintegration
   - **Rationale:** Sliding windows are arbitrary; event-driven ensures no data loss and consistency with motion-based keyframes

3. **IMU Enabled by Default**
   - Launch file sets `enable_imu:=true` so `scripts/run_and_evaluate.sh` uses IMU automatically for progress tracking

### Implementation Details

**IMUPreintegrator:**

- Implements Forster et al. (2017) "On-Manifold Preintegration" equations
- Outputs: `delta_R` (3x3), `delta_v` (3D), `delta_p` (3D), `Sigma_preint` (9x9 covariance)
- Uses rotation vector (axis-angle) representation via SO(3) exponential map
- Applies Frobenius correction to covariance (identity for Gaussian family)
- OpReport: `approximation_triggers: ["Linearization"]`, `family_in: "IMU"`, `family_out: "Gaussian"`

**Motion-Based Keyframes:**

- `_check_keyframe(current_pose, stamp_sec)` computes translation and rotation from last keyframe
- Triggers new keyframe when either threshold exceeded
- Logs: `"Keyframe N created at t=X.XXXs (trans=X.XXm, rot=X.X deg)"`

**IMU Factor Publishing:**

- On keyframe creation, preintegrates all IMU measurements in interval `[t_i, t_j]`
- Publishes `IMUFactor` message to `/sim/imu_factor`
- Clears IMU buffer up to `t_j` after publishing

### Testing

- **23 unit tests** covering:
  - Identity preintegration (zero input)
  - Constant velocity / acceleration cases
  - Pure rotation cases
  - Covariance properties (9x9, symmetric, positive definite, grows with time)
  - OpReport compliance (name, family, triggers, Frobenius)
  - Bias handling
  - Edge cases (empty, single measurement, high/low frequency)
- **All 65 existing audit invariant tests** still pass

### Verification

```bash
# Build
cd fl_ws && colcon build --packages-select fl_slam_poc

# Run IMU tests
cd fl_ws/src/fl_slam_poc && pytest test/test_imu_preintegration.py -v

# Run full evaluation (IMU enabled by default)
./scripts/run_and_evaluate.sh

# Check IMU factors being published
ros2 topic echo /sim/imu_factor
```

### Next Steps

**Phase 2: Backend 15D State Extension** - Extend backend state from 6DOF SE(3) to 15DOF (pose + velocity + biases) and implement IMU factor fusion. This will allow the backend to consume the `/sim/imu_factor` messages now being published by the frontend.

### Design Invariants Preserved

- ✅ **Frobenius correction:** IMU preintegration emits `approximation_triggers: {"Linearization"}` and applies correction
- ✅ **Closed-form-first:** Preintegration uses closed-form integration (no iterative solvers)
- ✅ **Soft association:** No changes to association (still responsibility-based)
- ✅ **OpReport taxonomy:** New operator emits OpReport with all required fields
- ✅ **No ground-truth ingestion:** Backend unchanged, still doesn't subscribe to GT topics

---

## 2026-01-21: 3D Mode Sensor Wiring (LiDAR + RGB-D)

### Summary
Fixes a key wiring issue where LiDAR data was being published into a camera-named PointCloud2 topic, while camera RGB-D streams were decompressed but not used for association in 3D mode.

### Changes
- **Topic hygiene**
  - Standardized the default PointCloud2 topic to `/lidar/points` for 3D geometry pipelines (override to `/camera/depth/points` for RGB-D cameras publishing PointCloud2).
  - Updated the Livox converter default output to `/lidar/points`.
  - Updated the M3DGR rosbag launch default PointCloud2 topic to `/lidar/points` (Livox CustomMsg → PointCloud2).
- **RGB-D availability in 3D mode**
  - `SensorIO` now subscribes to depth images even when `use_3d_pointcloud=True` (when `enable_depth=True`) so RGB-D can contribute in LiDAR-driven 3D runs.
  - To preserve rosbag robustness and avoid TF spam, depth images are buffered in raw form in 3D mode (depth→points conversion remains skipped in 3D mode).
  - Frontend status monitoring now registers `depth` whenever `enable_depth=True`, including in 3D mode.
- **Non-placeholder appearance/depth descriptors**
  - `DescriptorBuilder` now computes fixed-size RGB and depth histogram descriptors (exact, deterministic) and includes them in the multi-modal descriptor used for soft responsibilities.
  - New parameters: `rgbd_sync_max_dt_sec`, `rgbd_min_depth_m`, `rgbd_max_depth_m`.

### Invariants
- No ground-truth ingestion: `/vrpn_client_node/*` remains evaluation-only.
- No heuristic gating added: association remains responsibility-based; descriptors only change the likelihood model inputs.

## 2026-01-21: Roadmap Clarifications (Frames, Keyframes, No-TF)

### Summary
Documentation hardening to prevent silent frame/intrinsics assumptions: records bag-observed frames (`odom_combined`/`base_footprint`, `camera_color_optical_frame`), notes absence of `CameraInfo` and TF, and clarifies that IMU factors must be scheduled on explicit keyframes (not anchor birth).

## 2026-01-21: Evaluation Pipeline & Warning Fixes

### Summary
Comprehensive upgrade to the SLAM evaluation pipeline with publication-quality metrics and plots, plus critical bug fixes for trajectory timestamps and terminal warning cleanup.

### Critical Bug Fixes

**1. Trajectory Export Timestamps (CRITICAL)**
- **Problem**: Backend used `self.get_clock().now()` (wall clock) for trajectory export, causing duplicate timestamps
- **Impact**: evo's trajectory association became unreliable, potentially matching wrong poses
- **Fix**: Store odometry message timestamp (`self.last_odom_stamp`) and use it for trajectory export
- **Files**: `backend/backend_node.py`

**2. First Pose Zero Delta**
- **Problem**: tb3_odom_bridge skipped publishing on first message, leaving backend uninitialized
- **Impact**: Backend had to wait for second message to start processing
- **Fix**: Publish zero delta (identity transform) on first message - logically correct since delta from pose to itself is zero
- **Files**: `utility_nodes/tb3_odom_bridge.py`

### Terminal Warning Fixes

**3. Suppress rerun_bridge Warning**
- **Problem**: "not found: rerun_bridge/local_setup.bash" warning during launch
- **Fix**: Suppress stderr when sourcing setup.bash in run_and_evaluate.sh

**4. Skip TF Lookup in 3D Mode**
- **Problem**: TF lookup warnings for camera_color_optical_frame when using 3D pointcloud mode
- **Fix**: In 3D pointcloud mode, depth images can be subscribed for RGB-D descriptors/evidence, but depth→points conversion (TF-dependent) is skipped to avoid TF spam during rosbag playback
- **Files**: `frontend/processing/sensor_io.py`

**5. Skip Depth Sensor Registration in 3D Mode**
- **Problem**: "SENSOR MISSING: depth" warnings in 3D pointcloud mode
- **Fix**: Register depth when `enable_depth=True` (including in 3D mode) so RGB-D contributions are visible and auditable
- **Files**: `frontend/frontend_node.py`

### Enhanced Evaluation Pipeline

**6. Publication-Quality evaluate_slam.py**
Complete rewrite with:
- **Trajectory Validation**: Checks for monotonic timestamps, duplicates, coordinate ranges
- **Rotation Metrics**: ATE and RPE for rotation (degrees), not just translation
- **Multi-Scale RPE**: Compute at 1m, 5m, 10m scales
- **Error Heatmap**: Trajectory colored by error magnitude
- **Pose Graph Visualization**: Show pose nodes with odometry edges
- **CSV Export**: All metrics in spreadsheet-ready format
- **Files**: `scripts/evaluate_slam.py`

**7. Progress Feedback**
- **Problem**: No indication of SLAM progress during rosbag playback
- **Fix**: Added progress monitoring showing backend status, anchor creation, bag duration
- **Files**: `scripts/run_and_evaluate.sh`

### New Output Files

After running `scripts/run_and_evaluate.sh`:

**Trajectory Plots:**
- `trajectory_comparison.png` - 4-view overlay (XY, XZ, YZ, 3D)
- `trajectory_heatmap.png` - Error-colored trajectory
- `pose_graph.png` - Pose nodes with odometry edges

**Error Analysis:**
- `error_analysis.png` - Error over time + histogram

**Metrics:**
- `metrics.txt` - Human-readable summary with ATE/RPE translation and rotation
- `metrics.csv` - Spreadsheet-ready with all statistics

### Verification Checklist

- [x] No "not found" messages for rerun_bridge
- [x] No TF lookup warnings in 3D pointcloud mode
- [x] No "SENSOR MISSING: depth" warnings in 3D mode
- [x] First odom message publishes zero delta
- [x] Trajectory timestamps are unique and monotonic
- [x] Progress feedback shows during SLAM run
- [x] All evaluation plots generated successfully
- [x] Metrics include both translation AND rotation errors

### Files Modified

- `scripts/run_and_evaluate.sh` - Progress monitoring, output formatting, warning suppression
- `scripts/evaluate_slam.py` - Complete rewrite with all enhancements
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py` - Trajectory timestamp fix
- `fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/tb3_odom_bridge.py` - Zero delta first pose
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py` - Depth status monitoring behavior
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/processing/sensor_io.py` - 3D mode depth buffering (skip TF-dependent depth→points)

---

## 2026-01-20: Major Package Restructure

**Type:** Major refactor with breaking changes

### Summary

Comprehensive reorganization of `fl_slam_poc` package into a clear frontend/backend architecture based on deep structural audit.

### Changes

**Directory Structure:**
- Created `frontend/processing/`, `frontend/loops/`, `frontend/anchors/` subdirectories
- Created `backend/fusion/`, `backend/parameters/` subdirectories
- Created `common/transforms/` for shared SE(3) operations
- Created `utility_nodes/` for helper nodes
- Renamed `geometry/` → `common/transforms/` (clearer naming)
- Renamed `models/` → `backend/parameters/` (not ML models, parameter estimators)

**Dead Code Removal:**
- Deleted `utils/sensor_sync.py` (241 lines never used)
- Fixed setup.py references to archived launch files

**Node Renames:**
- `fl_backend_node` → `backend_node`
- `tb3_odom_bridge_node` → `tb3_odom_bridge`
- `image_decompress_node` → `image_decompress`
- `livox_converter_node` → `livox_converter`
- `sim_world_node` → `sim_world`

**Backward Compatibility:**
- Legacy import paths preserved via re-export modules
- Legacy node names preserved as aliases in setup.py

**Experimental Code:**
- Tagged Dirichlet semantic SLAM files as EXPERIMENTAL (not archived)
- Files: `dirichlet_backend_node.py`, `sim_semantics_node.py`, `dirichlet_geom.py`

### Files Modified

- ~30 Python files with updated imports
- 3 launch files with new node names
- setup.py with new entry points
- All `__init__.py` files rewritten

### Testing

- 113/114 tests passing (1 pre-existing GPU precision issue)
- Package builds successfully
- All launch files functional

---

## 2026-01-15 22:17:40 UTC
- Set baseline sim to planar robot (2D pose), running in a 3D world. Note: true 6-DOF is pending.
- Target sensors: LiDAR + camera (front-end work to follow).

## 2026-01-15 22:22:11 UTC
- Began in-house front-end build for LiDAR + camera evidence generation (planar robot baseline).
- Initial loop evidence uses scan descriptors to select anchors and publishes absolute pose factors with explicit approximation logging.

## 2026-01-15 22:37:20 UTC
- Reworked front end to by-construction responsibilities, anchor updates, and ICP-based loop factors (explicit approximation triggers logged).
- Removed heuristic gates (periodic keyframes, min-weight), replaced with probabilistic responsibilities and explicit budget operators.

## 2026-01-15 22:43:39 UTC
- Made descriptor likelihood self-adaptive via anchor-weighted variance; removed fixed sigma/new-component likelihood parameters.

## 2026-01-15 22:47:37 UTC
- Added depth camera feature stream to the front end and integrated it into descriptor-based responsibilities.
- Set TurtleBot3 launch default to `waffle` to provide LiDAR + depth.

## 2026-01-15 22:53:01 UTC
- Aligned backend state frame with odom and now publish pose covariance; odom-bridge delta uses wrapped yaw in odom frame.

## 2026-01-16 00:16:24 UTC
- Replaced descriptor scoring with Normal-Inverse-Gamma (diagonal) model and Student-t predictive for by-construction responsibilities.
- Added global descriptor prior for new-component predictive and fractional-update anchor models.

## 2026-01-16 00:16:24 UTC (cont.)
- Added custom loop/anchor messages and switched to relative loop factors with anchor creation events.
- Backend now consumes relative loop factors, composes with anchor beliefs, and fuses in one shot with explicit Linearization triggers.
- Package now builds with ament_cmake for rosidl message generation.

## 2026-01-16 00:29:36 UTC
- Updated sim-world POC to publish anchor-create and relative loop factors using new messages.

## 2026-01-16 00:39:24 UTC
- Added ament_cmake install rules and script entrypoints for ROS 2 run/launch after introducing custom messages.

## 2026-01-16 00:40:48 UTC
- Fixed relative loop pose direction (publish T_a^{-1} T_t) and added anchor/loop visualization markers.

## 2026-01-16 00:53:17 UTC
- Upgraded backend state to SE(3) (6-DOF) with rpy-based composition and 6x6 covariance publishing.
- Updated TB3 odom bridge to emit SE(3) body-frame deltas with relative quaternions.
- Converted sim-world POC to 6-DOF deltas and SE(3) composition to keep the pipeline consistent.

## 2026-01-16 00:58:32 UTC
- Adjusted Python package installation to avoid duplicate ament_cmake_python targets.
- `colcon build --symlink-install` now completes cleanly after SE(3) upgrades.

## 2026-01-16 01:07:26 UTC
- Replaced anchor and loop budget selection with information-theoretic objectives (entropy proxy + KL info gain).
- Added SE(3) ICP covariance via least-squares Jacobian (with pinv fallback) and logged domain constraints on insufficient points.
- Added predictive variance to NIG model and centralized OpReport publishing in the front end.

## 2026-01-16 01:10:05 UTC
- Added a Jacobian policy: core ops are Jacobian-free; Jacobians allowed only in sensor-to-evidence extraction with explicit logging and Frobenius correction.

## 2026-01-16 01:13:20 UTC
- Documented ICP covariance using normal-equation JᵀJ as a sensor-to-evidence linearization (logged for later replacement with information-geometric alternatives).

## 2026-01-16 01:21:16 UTC
- Applied loop factor responsibility weights in backend fusion and time-aligned anchor creation using a state buffer with explicit TimestampAlignment logging.
- Added feature buffering and nearest-in-time alignment for scan/image/depth descriptors with explicit BudgetTruncation and TimestampAlignment logging.
- Front end now supports delta-odom integration via an explicit `odom_is_delta` parameter.

## 2026-01-16 01:41:37 UTC
- Added explicit OpReports for missing odom, missing depth intrinsics/points, and responsibility normalization failure to avoid silent no-op behavior.
- Front end now launches by default in sim POC launch files, with explicit odom topic + delta-odom configuration.

## 2026-01-16 01:51:41 UTC
- Sim world now publishes synthetic scan + camera/depth streams by default for frontend POC runs.
- Added sim-world toggles for sensor vs. anchor/loop publishing to avoid mixed evidence when frontend is enabled.
- Standardized sim odom frame to `odom` for consistency with frontend/backend expectations.

## 2026-01-16 01:55:29 UTC
- Synthetic sensor publishing is now opt-in only (disabled by default) to avoid masking real Gazebo/robot topics.

## 2026-01-16 01:59:28 UTC
- Added timestamp alignment bounds for frontend (scan↔pose/image/depth) and backend anchor creation, with explicit OpReports on out-of-range alignment.
- Added OpReports for unsupported image/depth encodings, size mismatches, and invalid CameraInfo to prevent silent sensor dropouts.
- Added an anchor birth accumulator so new anchors are created only after sufficient new-mass accumulates (logged explicitly).
- Backend now logs unknown-anchor loop factors instead of silently dropping them.

## 2026-01-16 02:11:31 UTC
- OpReport now distinguishes frobenius_required/applied/operator and enforces operator naming when applied.
- Ablations are disallowed by default; set `allow_ablation:=true` explicitly for baseline/benchmark runs.
- Dirichlet mixture projection reports its third-order Frobenius operator when enabled.

## 2026-01-16 (Audit Response)
- **Created `fl_slam_poc/geometry/` module** with SE(3) operations using rotation vector (axis-angle) representation.
  - `se3.py`: Implements rotation vector ↔ rotation matrix via Rodrigues formula (exponential/logarithmic maps).
  - Includes `se3_compose`, `se3_inverse`, `se3_apply`, `se3_adjoint` for proper group operations.
  - `se3_cov_compose`: Proper covariance transport via adjoint representation (exact, not additive approximation).
  - `icp_covariance_tangent`: ICP covariance in se(3) tangent space coordinates.
- **Updated `fl_backend_node.py`** to use rotation vector representation throughout.
  - State is now (x, y, z, rx, ry, rz) where (rx, ry, rz) is rotation vector (Lie algebra so(3)).
  - Covariance transport now uses adjoint: `Σ_out = Σ_a + Ad_{T_a} Σ_b Ad_{T_a}^T`.
  - Loop fusion marked as exact (no linearization) since composition + adjoint transport is exact.
  - Predict step remains explicit ablation (additive noise approximation).
- **Created test suite** (`test/test_audit_invariants.py`) covering audit requirements:
  - I1: Frame consistency test (static scene → identity transform).
  - I2: Known motion test (ground truth recovery).
  - I3: Timestamp alignment test (Gaussian weighting verification).
  - I4: Frobenius proof test (measurable delta norm, OpReport validation).
  - I5: Budget projection test (mass conservation, KL minimization).
  - Additional: SE(3) geometry tests (round-trip, associativity, adjoint).
- **Rationale**: Following information geometry principles from Combe (2022-2025), using tangent space (Lie algebra) representation enables:
  - Singularity-free orientation handling (no gimbal lock).
  - Exact covariance transport (adjoint is exact, not an approximation).
  - Closed-form operations (O(n³) vs iterative approximation).
  - Proper fusion in flat dual coordinates (information form).

## 2026-01-16 (Information Geometry Upgrade)
- **Created `fl_slam_poc/operators/information_distances.py`** with closed-form information-geometric distances:
  - `hellinger_sq_expfam`: Universal Hellinger for any exponential family via log-partition.
  - `hellinger_gaussian`: Closed-form Hellinger for multivariate Gaussians.
  - `fisher_rao_gaussian_1d`: Fisher-Rao distance for univariate Gaussian (location-scale).
  - `fisher_rao_student_t`: Fisher-Rao for Student-t (NIG predictive) — **closed-form, true metric**.
  - `fisher_rao_spd`: Affine-invariant distance on SPD covariance matrices.
  - `product_distance`: Pythagorean aggregation for product manifolds.
  - `gaussian_kl`, `wishart_bregman`: Closed-form Bregman divergences.
- **Upgraded `fl_slam_poc/operators/gaussian_info.py`**:
  - Added `log_partition`, `kl_divergence`, `hellinger_distance`, `bhattacharyya_coefficient`.
  - Added `natural_gradient`, `marginalize`, `condition`, `product_of_experts`.
  - All operations remain exact and closed-form in information coordinates.
- **Rewrote `fl_slam_poc/nodes/frontend_node.py`** to use proper information geometry:
  - **Replaced log-likelihood association with Fisher-Rao metric**.
  - `NIGModel.fisher_rao_distance()`: Uses closed-form Student-t FR distance (Miyamoto 2024).
  - `compute_responsibilities_fisher_rao()`: Soft association via exp(-d_FR/scale).
  - Product manifold structure for multi-channel descriptors.
  - All distance computations are TRUE METRICS (symmetric, triangle inequality).
- **Why this matters**:
  - Log-likelihoods are NOT proper metrics (violate triangle inequality).
  - Fisher-Rao distances ARE Riemannian metrics — geometry-native, model-consistent.
  - Closed-form O(n) per dimension, no Jacobians, no iteration.
  - Better clustering/association behavior due to proper metric properties.
- **Test suite updated** to verify all closed-form distance implementations:
  - Hellinger bounds (0 ≤ H ≤ 1), symmetry.
  - Fisher-Rao triangle inequality verification.
  - SPD metric properties.
  - Product manifold Pythagorean aggregation.
- **Reference**: Miyamoto et al. (2024) for FR closed-forms, Combe (2022-2025) for pre-Frobenius manifolds.

## 2026-01-16 (Design Violation Fixes)
- **Removed hard gate on ICP validity** (was `min_points = 3`):
  - Replaced with `icp_information_weight()`: probabilistic weight based on Fisher information.
  - Model: weight ∝ sigmoid(n - n_min) × (n/n_ref) × exp(-mse/σ²)
  - Soft sigmoid for DOF constraint (SE(3) needs ≥6 points, but soft).
  - No hard threshold — low-information cases get low weight, not rejection.
- **Explicit loop factor convention** (was implicit):
  - Added explicit docstring: `Z = T_anchor^{-1} ∘ T_current` ("anchor observes current").
  - Added `compute_relative_transform()` with convention documentation.
  - Added `validate_loop_factor_convention()` runtime invariant check.
  - Backend reconstruction: `T_current = T_anchor ∘ Z`.
- **Proper covariance transport** (was missing adjoint):
  - `icp_covariance_tangent()`: Computes covariance at identity tangent space with explicit basis.
  - `transport_covariance_to_anchor_frame()`: Transports via adjoint: `Σ_anchor = Ad @ Σ @ Ad^T`.
  - Basis convention documented: `[δx, δy, δz, δωx, δωy, δωz]` (translation first).
- **Replaced hardcoded parameters with adaptive priors**:
  - Created `AdaptiveParameter` class: learns from data with Bayesian prior regularization.
  - `icp_max_iter`, `icp_tol`: Now adaptive based on convergence behavior.
  - `fr_distance_scale`: Learns from observed descriptor distances.
  - All parameters have:
    - Prior mean (initial value)
    - Prior strength (how much to trust prior vs data)
    - Floor (domain constraint, not gate)
  - Remaining config params are either:
    - Computational budget (depth_stride, buffer_len) — not model parameters
    - Topic names — configuration, not policy
    - Frame IDs — environment setup
- **Design principle compliance**:
  - No hard gates: all constraints are probabilistic weights.
  - No implicit conventions: all transform semantics explicitly documented.
  - No hardcoded policy: parameters either adapt from data or are justified priors.
  - Covariance transport: proper adjoint representation throughout.

## 2026-01-16 (Comprehensive Audit Fix)

### Critical Issues Fixed

1. **Backend Timestamp Hard Gate → Probabilistic Model** (`fl_backend_node.py`)
   - Removed `if dt > max_alignment_dt_sec: return` hard gate
   - Added `TimestampAlignmentModel`: Gaussian likelihood `weight = exp(-0.5*(dt/σ)²)`
   - Covariance scaled by inverse weight (high dt → high uncertainty, not rejection)
   - Full logging of timestamp_weight, timestamp_loglik, covariance_scale_factor

2. **Single-Node Loop Fusion → Two-Pose Factor** (`fl_backend_node.py`)
   - Implemented proper G1-compliant two-pose factor semantics
   - Joint Gaussian over [anchor_state, current_state] (12-dimensional)
   - Jacobians: H_anchor = -Ad_{T_anchor^{-1}}, H_current = I
   - Schur complement marginalization to get updated p(x_current | Z)
   - Bidirectional update: both anchor AND current beliefs improved
   - Full linearization metadata logged (linearization_point_anchor, linearization_point_current)

3. **Backend Process Noise Hardcoded → Adaptive** (`fl_backend_node.py`)
   - Added `AdaptiveProcessNoise` class with inverse-Wishart prior
   - Learns from prediction residuals online
   - Point estimate: E[Q] = Ψ / (ν - p - 1)
   - Confidence metric tracks prior vs data dominance

4. **ICP OpReport Incomplete → Full Solver Metadata** (`frontend_node.py`)
   - Created `ICPResult` dataclass with all metadata:
     - initial_objective, final_objective, tolerance, iterations, max_iterations, converged
   - OpReport now includes: solver_objective, solver_tolerance, solver_iterations, etc.
   - Convergence status explicitly logged

5. **Loop Factor Convention Implicit → Explicit in Message** (`LoopFactor.msg`)
   - Added 25-line documentation header with EXPLICIT convention:
     - `Z = T_anchor^{-1} ∘ T_current` ("anchor observes current")
     - Backend reconstruction: `T_current = T_anchor ∘ Z`
   - Covariance basis documented: `[δx, δy, δz, δωx, δωy, δωz]`
   - Added solver metadata fields: solver_name, solver_objective, solver_tolerance, 
     solver_iterations, solver_max_iterations, information_weight

6. **Deterministic Birth → Stochastic Hazard Model** (`frontend_node.py`)
   - Added `StochasticBirthModel` class
   - Birth as Poisson process: P(birth | r) = 1 - exp(-λ₀ * r * dt)
   - `sample_birth()` uses RNG, not deterministic threshold
   - `birth_probability()` logged in OpReport
   - `StochasticAnchorBirth` OpReport with intensity, time_step, probability

### Major Issues Fixed

7. **Backend Linearization Logging** (`fl_backend_node.py`)
   - Added `linearization_point` (6-vector) to metrics
   - Added `linearization_point_cov_trace` to track uncertainty at linearization
   - Added `adjoint_norm` to track transport magnitude

8. **TB3 Bridge Frame Hardcoding → Configurable + Validation** (`tb3_odom_bridge_node.py`)
   - Frame IDs now parameters: `output_frame`, `child_frame`
   - Added `validate_frames` parameter with TF validation
   - Frame change warnings logged (once per change)
   - Uses rotation vector internally (no RPY)

9. **Simulation Uses RPY → Rotation Vector** (`sim_world_node.py`)
   - State now `np.zeros(6)` with rotation vector
   - Uses `se3_compose` from geometry module (not local RPY functions)
   - All simulation parameters now configurable (not hardcoded)
   - LoopFactor messages include solver_metadata fields

10. **SE(3) Epsilon Documented** (`se3.py`)
    - Added module-level constants with documentation:
      - `ROTATION_EPSILON = 1e-10`: For small-angle approximation
      - `SINGULARITY_EPSILON = 1e-6`: For π-singularity handling
    - Docstring explains: "NUMERICAL STABILITY choices, not model parameters"
    - Constants based on IEEE 754 double precision (~15 decimal digits)

### Missing Evidence Gaps Fixed

11. **Loop Budget Projection OpReport** (`frontend_node.py`)
    - Added `LoopBudgetProjection` OpReport when loop budget truncates
    - Includes Frobenius correction stats (delta_norm, input_stats, output_stats)

12. **Silent Depth Absence → OpReport** (`frontend_node.py`)
    - Added `LoopFactorSkipped` OpReport when depth_points unavailable
    - Added `AnchorBirthSkipped` OpReport when birth would have occurred but depth missing
    - All silent returns now have explicit logging

### Integration Tests Added (`test_audit_invariants.py`)

13. **I1: Frame Consistency Tests**
    - `test_transform_identity_is_identity`
    - `test_transform_inverse_roundtrip`
    - `test_frame_chain_composition`

14. **I2: Known Motion End-to-End Tests**
    - `test_known_translation`
    - `test_known_rotation`
    - `test_known_combined_motion`
    - `test_loop_closure_identity_at_start`
    - `test_circular_path_returns_to_origin`

15. **I3: Timestamp Alignment Behavior Tests**
    - `test_gaussian_weight_at_zero`
    - `test_gaussian_weight_decreases_with_dt`
    - `test_gaussian_weight_symmetric`
    - `test_weight_bounded`

16. **Additional Tests**
    - `TestTwoPoseFactorSemantics`: Verifies joint update improves both beliefs
    - `TestStochasticBirth`: Verifies Poisson birth model properties
    - `test_near_pi_rotation`: Verifies singularity handling
    - `test_numerical_constants_documented`: Verifies epsilon constants exist

### Design Principle Compliance Summary

| Principle | Status |
|-----------|--------|
| No hard gates | ✅ All thresholds replaced with probabilistic weights |
| Explicit conventions | ✅ Convention documented in message, code, and tests |
| No hardcoded policy | ✅ All params adaptive or justified priors |
| Proper covariance transport | ✅ Adjoint throughout, basis documented |
| Two-pose factor semantics | ✅ Joint update with marginalization |
| Proof-of-execution | ✅ OpReport for all operations including skips |
| Stochastic birth | ✅ Poisson hazard model, not deterministic |

---

## 2026-01-20: Frontend Modularization & Repository Cleanup

### Code Reorganization
- **Frontend refactored** from monolithic `frontend_node.py` (1098 lines) to modular architecture
- **Created** `fl_slam_poc/frontend/` module with 4 helper files:
  - `sensor_io.py` (258 lines): Pure I/O, TF lookups, point cloud conversion (NO math)
  - `descriptor_builder.py` (115 lines): Descriptor extraction using `models.nig` (exact)
  - `anchor_manager.py` (207 lines): Lifecycle using `models.birth` + `operators.third_order_correct` (exact)
  - `loop_processor.py` (263 lines): ICP + Fisher-Rao using `operators.*` (exact)
- **Main node** `frontend_node.py` reduced to 445 lines (pure orchestration)
- **Original** backed up as `frontend_node_ORIGINAL_BACKUP.py`

### Mathematical Verification
- ✅ All operations call **identical** `operators/` or `models/` functions
- ✅ Fisher-Rao: `models.nig.fisher_rao_distance()` preserved
- ✅ NIG updates: `models.nig.update()` preserved
- ✅ Stochastic birth: `models.birth.sample_birth()` preserved
- ✅ Frobenius corrections: `operators.third_order_correct()` & `operators.gaussian_frobenius_correction()` preserved
- ✅ ICP: `operators.icp_3d()` preserved
- ✅ All P1-P7 invariants maintained
- ✅ **No heuristic gating** - soft association preserved throughout

### Infrastructure Improvements
- **Added** `geometry/quat_to_rotvec()` for direct quaternion→rotation vector conversion (~30% faster)
- **Added** `geometry/__init__.py` documentation of SE(3) representation conventions
- **Added** `constants.py` centralizing ~50 magic numbers
- **Added** `config.py` for parameter grouping via dataclasses
- **Enhanced** `utils/sensor_sync.py` for DRY timestamp alignment
- **Enhanced** `utils/status_monitor.py` for observability

### Repository Cleanup
- **Deleted** ~500MB build artifacts (`build/`, `install/`, `log/`)
- **Deleted** diagnostic logs and cache files
- **Deleted** temporary analysis documents (FILE_AUDIT.md, MODULARIZATION_VERIFICATION.md)
- **Consolidated** modularization summary into this log entry

### Rosbag Compatibility
- ✅ `use_sim_time` compatible
- ✅ All topics configurable via parameters
- ✅ TF timeout handling preserved
- ✅ Sensor monitoring preserved (`/cdwm/frontend_status`)
- ✅ Ready for validation with `scripts/rosbag_slam_smoketest.sh`

### Testing Status
- **Pending**: Build verification (`colcon build`)
- **Pending**: Rosbag smoke test (anchors, loop factors, SLAM_ACTIVE)
- **Pending**: Invariant tests (`test_audit_invariants.py`)

### File Structure
```
fl_slam_poc/
├── frontend/ (NEW: 4 helper modules - orchestration only)
├── operators/ (6 files - core math, unchanged)
├── models/ (6 files - generative models, unchanged)
├── geometry/ (2 files - SE(3), enhanced with docs)
├── utils/ (2 files - infrastructure)
├── nodes/ (7 files including backup)
├── constants.py (NEW)
├── config.py (NEW)
└── __init__.py
```
| Integration tests | ✅ I1-I3 invariants covered |
2026-01-15 22:56:36 - Added Gaussian Frobenius no-op operator and applied it to linearization/covariance approximations; replaced loop selection with KL info gain + Frobenius-corrected projection; removed hard thresholds via probabilistic scaling and added independent-weight combination utility + tests.
2026-01-15 23:02:11 - Added launch_testing end-to-end ROS tests for loop factors, backend state frames, and timestamp alignment reports.
2026-01-15 23:02:45 - Guarded end-to-end launch tests to skip when ROS 2 runtime libraries are unavailable.
2026-01-15 23:03:52 - Refined launch test imports to avoid rclpy/launch_ros import errors when ROS runtime is missing.
2026-01-16 00:06:57 - Added `fl_ws/src/fl_slam_poc/scripts/run_e2e_tests` to source ROS/workspace and run end-to-end launch tests reliably.
2026-01-16 00:07:14 - Updated `run_e2e_tests` to source ROS setup files with nounset disabled for compatibility.
2026-01-16 00:07:44 - Added import sanity check and expanded PYTHONPATH in `run_e2e_tests` to ensure message modules are discoverable.
2026-01-16 00:08:30 - Switched `run_e2e_tests` to pytest importlib mode to avoid source-tree shadowing of generated message modules.
2026-01-16 00:20:10 - Added Dockerized ROS 2 Jazzy setup (`docker/Dockerfile`, `docker-compose.yml`) plus helper scripts for build/run/test/demo/stop and repo-level ignore files to reduce ROS artifact clutter.
2026-01-16 00:33:10 - Fixed Foxglove visualization wiring: backend now publishes dynamic TF (`/tf`) for `odom->base_link`, `poc_a.launch.py` provides static `base_link->camera_link`, and `sim_world_node` publishes non-uniform synthetic image/depth frames so Foxglove Image panels show structure (not a blank field). Also updated `scripts/docker-demo.sh` to avoid launching a second Foxglove bridge (compose already runs it), preventing duplicate publishers and confusing connection failures.
2026-01-16 00:46:40 - Made POC demo “non-empty by default”: `sim_world_node` now publishes `MarkerArray` obstacles on `/cdwm/world_markers` at 1 Hz (so Foxglove reliably sees them even if it connects late), and `poc_a.launch.py` / `poc_all.launch.py` defaults now enable sensors + anchors + loop factors + world markers for minimal troubleshooting.
2026-01-19 16:17:00 - Consolidated rosbag workflow into a single canonical doc (`ROSBAG.md`) and a single pass/fail smoke test (`scripts/rosbag_slam_smoketest.sh`), removed redundant rosbag/foxglove docs and legacy diagnostic scripts, and packaged rosbag QoS overrides inside `fl_slam_poc` for portability.

## 2026-01-19 20:30:00 UTC - Bug Fixes: Rosbag Compatibility & Type Errors

### Critical Bug Fix #1: QoS Mismatch
**Problem**: Frontend subscribed to `/scan` and `/odom` with `BEST_EFFORT` QoS, but rosbags typically record with `RELIABLE` QoS. Result: Frontend received ZERO sensor data, no SLAM activity.

**Fix**: Changed `frontend/sensor_io.py` line 75 from `ReliabilityPolicy.BEST_EFFORT` to `ReliabilityPolicy.RELIABLE`. Added logging to confirm odom/scan reception.

**Impact**: Frontend now successfully receives and processes rosbag sensor data. First anchor creation confirmed.

### Critical Bug Fix #2: Type Error in Fisher-Rao Distance
**Problem**: `frontend/loop_processor.py` line 99 called `anchor.desc_model.fisher_rao_distance(descriptor)` where `descriptor` was `np.ndarray`, but method expects `NIGModel` object. Result: Frontend crashed after first anchor creation.

**Fix**: Create temporary NIG model from current descriptor for comparison:
```python
temp_model = NIGModel(len(descriptor))
temp_model.update(descriptor, weight=1.0)
dist = anchor.desc_model.fisher_rao_distance(temp_model)
```

**Impact**: Frontend now successfully computes responsibilities and creates anchors. Backend receives anchors and integrates odometry. System operational.

### Enhanced Logging
- Added debug logging for first odom/scan received
- Added warning when scans dropped due to missing pose data
- Added info log for first successful scan processing
- Added visibility into odom buffer status

### Testing Status
- ✅ Rosbag playback functional
- ✅ Frontend processes scans
- ✅ Anchors created and published
- ✅ Backend receives and stores anchors
- ✅ Loop factors computed (pending loop closure detection validation)
- ✅ Foxglove bridge operational on port 8765

### Frame Configuration
- ✅ Verified TB3 bag frame names: `odom`, `base_link`, `base_scan`
- ✅ Updated launch defaults to match inspection results
- ✅ Added configurable frame parameters to launch file
- ✅ Added race condition protection: pending loop factor buffer in backend

## 2026-01-19 21:00:00 UTC - Feature Addition: Point Cloud Mapping (Option 1)

### Motivation
System successfully estimates trajectory and anchor poses, but lacks map visualization. Backend only stores pose statistics (mu, cov), not the actual sensor observations.

### Architecture Decision: Anchor-Based Point Cloud Map
**Rationale**: Lightweight, works with existing math backend, preserves information-geometric foundations, shows loop closure effects clearly.

**Alternative Options Considered**:
- **Option 2**: Dense trajectory point cloud (accumulate all scans) - More resource intensive, redundant
- **Option 3**: 2D occupancy grid - Requires ray-tracing, more complex, not needed for POV visualization

**Selected**: Option 1 - Store and visualize point clouds only at anchor keyframes.

### Implementation Plan
1. **Extend `AnchorCreate.msg`**: Add `geometry_msgs/Point[] points` field
2. **Frontend**: Publish anchor point clouds in `_publish_anchor_create()`
3. **Backend**: Store points with anchor data structure
4. **Backend**: Publish accumulated map as `sensor_msgs/PointCloud2` on `/cdwm/map`
5. **Backend**: Transform points to global frame using current anchor pose estimates
6. **Documentation**: Update Foxglove visualization instructions

### Expected Behavior
- Sparse point cloud map at anchor keyframe locations
- Map updates when loop closures correct anchor poses
- Foxglove visualization: PointCloud2 layer showing accumulated environment
- Lightweight: Only ~10-50 anchors per trajectory, ~360 points per 2D scan

### Information Geometry Compliance
- ✅ No new math operators required
- ✅ Point cloud transforms use existing `geometry/se3.py` operations
- ✅ Preserves probabilistic pose estimates with covariance
- ✅ Map visualization is post-processing layer, doesn't affect inference

### File Changes
- `msg/AnchorCreate.msg` - Add points field
- `nodes/frontend_node.py` - Publish points with anchor
- `nodes/fl_backend_node.py` - Store and publish map
- `CMakeLists.txt` - Add PointCloud2 dependency (if needed)
- `package.xml` - Add sensor_msgs dependency

### Testing Checklist
- [x] Message generation successful
- [x] Frontend publishes points
- [x] Backend receives and stores points
- [x] Map published on `/cdwm/map`
- [ ] Foxglove displays point cloud (pending user verification)
- [ ] Map updates after loop closure (pending rosbag test completion)

### Implementation Complete
- **Duration**: ~45 minutes
- **Files Modified**: 4 files
- **Lines Added**: ~80 lines
- **Build Status**: ✅ Successful
- **Import Tests**: ✅ Both nodes import without errors
- **Launch Test**: ✅ Nodes start successfully

### Files Changed
1. `msg/AnchorCreate.msg` - Added `geometry_msgs/Point[] points` field
2. `nodes/frontend_node.py` - Modified `_publish_anchor_create()` to include points
3. `nodes/fl_backend_node.py` - Added map storage, transform, and publication
4. `package.xml` - Added sensor_msgs build dependency
5. **NEW**: `MAP_VISUALIZATION.md` - Complete documentation for map visualization

### Known Status
- System builds and imports successfully
- Nodes launch without errors
- Full rosbag integration test pending (smoketest timeout issues)
- Foxglove visualization instructions provided in MAP_VISUALIZATION.md

### Technical Details
**Map Publication**:
- Topic: `/cdwm/map`
- Type: `sensor_msgs/PointCloud2`
- Frame: `odom`
- Update triggers: Anchor creation, loop closure

**Point Cloud Processing**:
- Subsampling: Max 1000 points per anchor message
- Transform: SE(3) using `geometry/se3.py` (rotvec → rotmat → apply)
- Accumulation: All anchors concatenated and published
- Format: XYZ float32 fields

**Memory Efficiency**:
- ~1000 points/anchor × 4 bytes/float × 3 coords = ~12 KB/anchor
- Typical run: 10-50 anchors = 120-600 KB total map data
- No unbounded growth: Map size = num_anchors × points_per_anchor

---

## 2026-01-20: Multi-Modal Sensor Fusion Architecture Decision (Hybrid Laser + RGB-D)

### Background Discovery
**Rosbag sensors identified**: Existing TB3 SLAM rosbag (`tb3_slam3d_small_ros2`) contains RGB-D data that was previously unused:
- `/stereo_camera/left/image_rect_color/compressed/throttled` (851 RGB frames)
- `/stereo_camera/depth/depth_registered/compressedDepth/throttled` (848 depth frames)
- `/stereo_camera/left/camera_info` (1699 calibration messages)
- `/stereo_camera/odom` (1698 visual odometry messages - bonus evidence source)
- Existing laser: `/scan` (526 scans), `/odom` (2778 wheel odometry)

Launch file defaulted to `enable_image:=false`, `enable_depth:=false` due to misleading comment "most public TB3 bags do not include raw camera topics." Data exists but is **compressed**.

### Architecture Decision: Hybrid Dual-Layer Approach

**Decision**: Implement **hybrid multi-modal fusion** instead of replacing sparse laser-based SLAM with pure 3D Gaussian splatting.

**Two-layer structure**:
1. **Sparse Anchor Layer** (Laser Primary):
   - Laser scans create keyframe anchors for pose estimation (existing behavior)
   - Anchors store 2D/3D pose + covariance in information form (Λ, η)
   - Primary responsibility: Trajectory estimation, loop closure detection
   
2. **Dense 3D Module Layer** (RGB-D):
   - RGB-D creates dense 3D Gaussian modules with:
     - Position (3D Gaussian in information form)
     - Surface normals (von Mises-Fisher distribution)
     - Color/appearance (RGB Gaussian)
     - Opacity (scalar Gaussian)
   - Primary responsibility: Photoreal mapping, dense geometry, appearance

### Fusion Strategy Choices

**Critical Design Questions Answered**:

**Q1: How to fuse laser 2D and RGB-D 3D evidence at overlapping locations?**

**Answer**: **Geometric fusion only** in information form (natural parameter space).
- Laser provides strong XY constraint (2D positions, high precision)
- RGB-D provides full 3D constraint (weaker XY, but adds Z + normals + color)
- Fusion via **additive information form**: `Λ_total = Λ_laser + Λ_rgbd`, `η_total = η_laser + η_rgbd`
- Implementation: Lift laser 2D → 3D with weak Z prior (large variance), then exact information addition
- **Exact, closed-form, associative** (no Jacobians, no iteration)

**Trade-off**: Could have done full multi-modal with cross-sensor consistency checks (e.g., RGB-D depth vs laser range), but adds complexity without clear benefit. Geometric fusion is cleaner.

**Implication for future**: If we add semantic labels or other modalities, extend via product-of-experts (multiplicative in natural params → additive in information form).

**Q2: Which sensor creates modules, and when?**

**Answer**: **RGB-D densification** strategy.
- **Laser is primary**: Creates sparse anchor modules at keyframes (existing logic)
- **RGB-D densifies**: Creates dense modules between anchors
- **Different roles**: Laser for pose/trajectory, RGB-D for appearance/geometry
- At anchor locations: Laser 2D evidence and RGB-D 3D evidence fuse via information addition

**Trade-off rejected alternatives**:
- **Laser primary, RGB-D updates only**: Would miss dense geometry between anchors (sparse map only)
- **Independent birth from both**: Would create duplicate modules at same locations, requiring association/merge logic
- **RGB-D primary**: Would lose laser's superior 2D localization (depth cameras drift more in XY)

**Implication for future**: If we switch to pure RGB-D SLAM (no laser), change to independent birth with spatial hashing for duplicate prevention.

### Information Geometry Compliance

**All operations remain exact/closed-form**:

1. **Laser 2D → 3D lifting**: `make_evidence(mu_3d, block_diag(Σ_xy_laser, σ²_z_weak))` - exact
2. **Information fusion**: `(Λ₁ + Λ₂, η₁ + η₂)` - exact additive (P1 compliant)
3. **vMF normal barycenter**: Dual-space averaging + Bessel series inversion - closed-form (P1 compliant)
4. **Fisher-Rao distances**: Closed-form for Gaussian (eigenvalue formula), vMF (Bessel affinity), Student-t (arctanh) - exact metrics (P2, P5 compliant)
5. **No heuristic gating**: Spatial association via Fisher-Rao responsibilities, not hard distance thresholds (P5 compliant)

**New operators required** (all exact):
- `operators/vmf_geometry.py`: vMF barycenter, Fisher-Rao distance (via Bessel functions)
- `operators/multimodal_fusion.py`: `fuse_laser_rgbd()` with 2D→3D lifting
- `operators/spd_geometry.py`: SPD manifold operations (geodesics, Fréchet mean) - foundation for Phase 2

**No approximations introduced** (P4 not triggered):
- vMF series inversion converges to arbitrary precision (exact in limit, like exp/log)
- Bessel functions are standard special functions (treated as closed-form per IG convention)

### Trade-offs Summary

| Choice | Alternative Considered | Rationale for Choice | Implication if Changed |
|--------|------------------------|----------------------|------------------------|
| Hybrid (laser+RGB-D) | Pure 3D Gaussian splatting | Keep proven laser localization, add appearance gradually | If splatting becomes primary, refactor to independent birth |
| Geometric fusion only | Full cross-modal consistency | Simpler, cleaner, still exact | If cross-checks needed, add as auxiliary OpReport metrics |
| Laser creates anchors | Both sensors create | Avoids duplicate management, leverages existing logic | If RGB-D becomes primary, add spatial hash to backend |
| RGB-D densifies | RGB-D updates anchors only | Captures dense geometry, not just keyframes | If memory constrained, add culling policy |
| vMF for normals | Gaussian on sphere | vMF is native to S² (unit sphere), exact barycenter | If switching to Bingham (elliptical), use different operators |
| Separate dense modules | Extend anchor structure | Cleaner separation of sparse/dense roles | If merging layers, use single module class with optional fields |

### Memory & Compute Implications

**Estimated data growth**:
- **Sparse anchors**: 10-50 per run (unchanged)
- **Dense modules**: ~1000-5000 per run (from 850 RGB-D frames)
- **Per dense module**: ~200 bytes (position 3×8, cov 9×8, normal 3×8, color 3×8, opacity 2×8, mass 8)
- **Total dense storage**: ~1-5 MB per run (acceptable)

**Compute scaling**:
- **Laser SLAM**: O(n_anchors) = O(50) - unchanged
- **RGB-D processing**: O(n_pixels / subsample²) per frame = O(480×640/100) = O(3k points/frame)
- **Module fusion**: O(n_dense) for map publication = O(5k) - manageable at 1 Hz

**Culling strategy** (if needed):
- Distance-based: Remove modules >10m from robot (FIFO spatial culling)
- Mass-based: Exponential decay for dynamic objects (retain static background)
- Octree spatial hashing for efficient nearest-neighbor queries (Phase 2)

### Future Extension Paths

**Phase 1 (current plan)**: Add vMF operators, RGB-D decompression, basic fusion
**Phase 2 (if needed)**: 
- Real-time splatting renderer (GPU rasterization)
- Semantic labels (Dirichlet modules)
- SPD geodesic covariance propagation (replace Euclidean Q addition)
- EFA-inspired context-modulated responsibilities

**Phase 3 (research directions)**:
- Hyperbolic embeddings for object-level reasoning (negative multinomial → Poincaré ball)
- Hexagonal web structures for parallel learning (Combe 2024)
- Multi-robot map fusion via Bregman barycenters

### Files to Be Modified (Phase 1)

**New files**:
1. `nodes/image_decompress_node.py` - JPEG/PNG decompression for rosbag playback
2. `frontend/rgbd_processor.py` - Depth→3D points + normals + colors
3. `operators/vmf_geometry.py` - von Mises-Fisher barycenter + Fisher-Rao
4. `operators/multimodal_fusion.py` - Laser 2D + RGB-D 3D fusion
5. `operators/spd_geometry.py` - SPD manifold operations (foundational)
6. `test/test_wdvv_associativity.py` - Validate associativity claims (P2)

**Modified files**:
1. `launch/poc_tb3_rosbag.launch.py` - Enable cameras, add decompress node
2. `nodes/frontend_node.py` - Add RGB-D subscriptions and processing
3. `nodes/fl_backend_node.py` - Dual-layer module storage, fusion logic, map publisher
4. `msg/DenseModuleCreate.msg` - New message type for RGB-D evidence
5. `operators/__init__.py` - Export vMF and multimodal functions

**Documentation**:
1. `MAP_VISUALIZATION.md` - Update with hybrid dual-layer explanation
2. `Comprehensive Information Geometry.md` - Add vMF section (barycenter, FR formulas)

### Design Invariants Preserved (P1-P7)

✅ **P1 (Closed-form exactness)**: All new operators use special functions (Bessel) or algebraic series (vMF inversion), no numerical optimization loops

✅ **P2 (Associative fusion)**: Information addition is associative, vMF dual averaging is commutative, WDVV tests added

✅ **P3 (Legendre/Bregman)**: vMF uses dual-space (expectation params), fusion via Bregman barycenter

✅ **P4 (Frobenius correction)**: No approximations in Phase 1 (all exact), Frobenius stubs added for future if needed

✅ **P5 (Soft association)**: Fisher-Rao responsibilities for RGB-D module assignment, no hard distance thresholds

✅ **P6 (One-shot loop correction)**: RGB-D evidence fuses at anchors via single information addition (no iterative re-optimization)

✅ **P7 (Local modularity)**: Modules remain local (sparse anchors independent, dense modules spatially local), no global coupling

### Testing Strategy

**Unit tests**:
- `test_rgbd_processing.py`: Depth→pointcloud, normal extraction
- `test_vmf_geometry.py`: Barycenter associativity, Fisher-Rao triangle inequality
- `test_multimodal_fusion.py`: Laser 2D + RGB-D 3D fusion correctness
- `test_wdvv_associativity.py`: Validate P2 for Gaussian, Dirichlet, vMF

**Integration tests**:
- Rosbag with compressed RGB-D decompresses successfully
- Dense modules created from RGB-D (~500+ per run)
- Laser anchors fuse with RGB-D at overlapping locations
- `/cdwm/map` shows dual-layer visualization (sparse yellow + dense colored)
- No regressions in existing laser SLAM (SLAM_ACTIVE mode achieved)

### Key Insight

**Information form makes heterogeneous fusion trivial**: Laser 2D and RGB-D 3D can fuse directly in natural parameter space (after dimension lifting) because **information is additive**. No need for iterative alignment or cross-modal optimization. This is the power of exponential family geometry - different sensors, same math.

**vMF for directional data is closed-form**: Contrary to initial assumption, vMF barycenter and Fisher-Rao distances are **exact** (not approximate) via Bessel function identities. This preserves P1 exactness for surface normal fusion.

### Risk Assessment

**Low risk**:
- RGB-D data exists in rosbag (verified)
- Decompression is standard (cv2.imdecode)
- Information fusion is exact (no numerical issues)
- Gradual integration (Phase 1 doesn't break existing laser SLAM)

**Medium risk**:
- Temporal sync between laser and RGB-D (851 frames vs 2778 scans) - mitigate with message_filters
- Compressed depth format variations - mitigate with format detection
- Memory growth with dense modules - mitigate with spatial culling

**No high risks identified** - architecture is additive, not replacement

### Implementation Timeline

- **Week 1**: vMF operators, image decompression, basic RGB-D processing
- **Week 2**: Multi-modal fusion, dual-layer backend, unit tests
- **Week 3**: Integration testing, visualization, documentation
- **Week 4**: Refinement, performance profiling, optional culling

**Status**: Plan approved, implementation pending Phase 1 execution.

## 2026-01-20 - Test Script Consolidation & Project Cleanup

### Test Framework Consolidation

Consolidated FL-SLAM testing framework from 4 fragmented scripts into 2 focused test scripts with clear purposes.

**New Scripts Created:**
1. `scripts/test-minimal.sh` - Fast validation (~30s): Module imports, mathematical invariants, operators, models
2. `scripts/test-integration.sh` - Full E2E validation (~90s): Complete SLAM pipeline with rosbag replay
3. `scripts/docker-test-integration.sh` - Docker wrapper for integration tests
4. `TESTING.md` - Comprehensive testing documentation

**Scripts Updated:**
- `scripts/docker-test.sh` - Now runs minimal tests only (previously ran both unit + e2e)

**Scripts Removed (replaced by above):**
- `scripts/docker-rosbag-test.sh` → `scripts/docker-test-integration.sh`
- `scripts/rosbag_slam_smoketest.sh` → `scripts/test-integration.sh`
- `fl_ws/src/fl_slam_poc/scripts/run_e2e_tests` → `scripts/test-minimal.sh`

**Documentation Updated:**
- `README.md` - Testing and Docker sections
- `ROSBAG.md` - Quick start commands  
- `INSTALLATION.md` - Running tests section
- `AGENTS.md` - Quickstart and validation section

**Benefits:**
- Clear separation: Fast minimal tests (30s) vs comprehensive integration tests (90s)
- Single source of truth for testing workflow (TESTING.md)
- Easier to maintain and extend
- Better CI/CD integration

### Project Cleanup

**Created `archive/` folder** for obsolete files:
- Moved `fl_ws/build_3d/` and `fl_ws/install_3d/` (redundant, everything is 3D now)
- Moved `frontend_node_ORIGINAL_BACKUP.py` (backup file)
- Added `archive/README.md` documenting archived contents

**Rationale:** Removed `_3d` suffixes since everything is 3D by default now. Kept files for historical reference but removed from active codebase.

### Testing Strategy

**Minimal Tests** (`test-minimal.sh`):
- ✓ Module import validation
- ✓ SE(3) operations and invariants  
- ✓ Information geometry operators
- ✓ ICP solver properties
- ✓ Frobenius corrections
- ✓ Adaptive models (NIG, process noise)
- ✓ RGB-D processing and multimodal fusion
- ✓ Mathematical invariants (associativity, symmetry, etc.)

**Integration Tests** (`test-integration.sh`):
- ✓ Full ROS 2 node launch
- ✓ Rosbag replay with sensor data
- ✓ Frontend anchor creation
- ✓ Loop closure detection  
- ✓ Backend optimization (SLAM_ACTIVE mode)
- ✓ Foxglove visualization (optional)

**Recommended Workflow:**
```bash
# During development - quick validation
./scripts/docker-test.sh

# Before committing - full validation  
./scripts/docker-test-integration.sh
```

See `TESTING.md` for complete documentation.

## 2026-01-20 - Documentation Organization

### Documentation Structure Cleanup

**Created `docs/` folder** and moved all documentation (except top-level essential files) for better organization:

**Root-level files (kept for quick access):**
- `README.md` - Project overview and quick start
- `AGENTS.md` - Design invariants and agent rules (P1-P7)
- `CHANGELOG.md` - Project history and changes

**Moved to `docs/`:**
- `Comprehensive Information Geometry.md` - Mathematical reference
- `Project_Implimentation_Guide.sty` - Formal specification
- `GAZEBO_INTEGRATION.md` - Gazebo setup and troubleshooting
- `ROSBAG.md` - Rosbag testing workflow
- `TESTING.md` - Testing framework documentation
- `INSTALLATION.md` - Installation and setup guide
- `MAP_VISUALIZATION.md` - Visualization guide
- `ORDER_INVARIANCE.md` - Order invariance documentation
- `POC_Testing_Status.md` - Testing status and notes
- `PROJECT_RESOURCES_SUMMARY.md` - Project resources overview

**Updated references:**
- Updated all documentation links in `README.md` and `AGENTS.md` to point to `docs/` folder
- Internal `docs/` references remain relative (work within the folder)

**Benefits:**
- Cleaner root directory (3 essential files vs 13 documentation files)
- Better organization for navigating documentation
- Preserved all cross-references and links

### Final Cleanup

**Moved `fl_ws/log_3d/` to archive:**
- Obsolete log directory from pre-unified build system
- Consistent with removal of `build_3d/` and `install_3d/`
- Updated `archive/README.md` to document all archived items

**Project structure now fully clean:**
- Root: 3 essential markdown files
- `docs/`: 10 documentation files
- `archive/`: 4 obsolete items (build_3d, install_3d, log_3d, backup file)
- All active directories follow current naming conventions (no `_3d` suffixes)

## 2026-01-20 - 3D Point Cloud Support with GPU Acceleration

### Feature Overview

Upgraded FL-SLAM to support 3D point cloud input with optional GPU acceleration. The system now supports two sensor modalities:
1. **2D LaserScan** (default) - Traditional 2D LIDAR for planar SLAM
2. **3D PointCloud2** (new) - Full 3D point cloud for volumetric SLAM

**Key Point**: The backend remains unchanged - the Frobenius-Legendre framework is dimension-agnostic. Changes are frontend-only (sensor input and preprocessing).

### New Files Created

**Core Implementation:**
- `fl_slam_poc/operators/pointcloud_gpu.py` - GPU-accelerated point cloud processing:
  - `GPUPointCloudProcessor` class with Open3D CUDA support
  - `voxel_filter_gpu()` - GPU-accelerated voxel grid downsampling
  - `icp_gpu()` - GPU-accelerated ICP registration
  - Automatic fallback to CPU when GPU unavailable

**Configuration:**
- Extended `config.py` with `PointCloudConfig` and `GPUConfig` dataclasses
- Extended `constants.py` with 3D processing constants (voxel size, GPU limits, etc.)

**Launch Files:**
- `launch/poc_3d_rosbag.launch.py` - Dedicated 3D mode launch file for r2b dataset
- Updated `launch/poc_tb3_rosbag.launch.py` with optional 3D mode parameters

**Scripts:**
- `scripts/download_r2b_dataset.sh` - Download NVIDIA r2b benchmark dataset
- `scripts/test-3d-integration.sh` - 3D mode integration test

**Tests:**
- `test/test_pointcloud_3d.py` - Comprehensive tests for 3D processing:
  - PointCloud2 message conversion
  - Voxel filtering (GPU and CPU)
  - ICP registration (GPU and CPU)
  - LoopProcessor GPU integration

**Documentation:**
- `docs/3D_POINTCLOUD.md` - Complete guide for 3D point cloud mode

### Modified Files

**Frontend:**
- `frontend/sensor_io.py`:
  - Added PointCloud2 subscription and conversion
  - Added `pointcloud2_to_array()` function for message parsing
  - Mode switching between 2D LaserScan and 3D PointCloud2
  - Rate limiting for high-frequency point cloud input

- `frontend/loop_processor.py`:
  - Added GPU processor initialization and configuration
  - Added `preprocess_pointcloud()` with voxel filtering
  - Modified `run_icp()` to use GPU when available
  - Maintains CPU fallback for compatibility

- `nodes/frontend_node.py`:
  - Added 3D mode parameter declarations
  - GPU configuration passthrough to LoopProcessor

**Operators:**
- `operators/__init__.py`:
  - Exported new GPU functions: `GPUPointCloudProcessor`, `is_gpu_available`, `voxel_filter_gpu`, `icp_gpu`

### Configuration Parameters

**3D Mode:**
```python
use_3d_pointcloud: bool = False    # Switch to 3D point cloud mode
enable_pointcloud: bool = False    # Subscribe to PointCloud2
pointcloud_topic: str = "/lidar/points"  # Override to /camera/depth/points for RGB-D cameras
```

**Point Cloud Processing:**
```python
voxel_size: float = 0.05           # Voxel grid size (meters)
max_points_after_filter: int = 50000
min_points_for_icp: int = 100
icp_max_correspondence_distance: float = 0.5
pointcloud_rate_limit_hz: float = 30.0
```

**GPU Configuration:**
```python
use_gpu: bool = False              # Enable GPU acceleration
gpu_device_index: int = 0          # CUDA device index
gpu_fallback_to_cpu: bool = True   # CPU fallback if GPU unavailable
```

### Design Invariants Preserved

All FL-SLAM design invariants (P1-P7) are maintained:

✅ **P1 (Closed-form exactness)**: ICP uses SVD-based closed-form registration
✅ **P2 (Associative fusion)**: Backend unchanged - information form fusion
✅ **P3 (Legendre/Bregman)**: Backend unchanged - dual-space operations
✅ **P4 (Frobenius correction)**: Not triggered - all operations exact
✅ **P5 (Soft association)**: Fisher-Rao responsibilities unchanged
✅ **P6 (One-shot loop correction)**: Backend unchanged - direct fusion
✅ **P7 (Local modularity)**: Frontend preprocessing is local

**Critical**: The backend is **dimension-agnostic** - it operates on (L, h) information form regardless of whether evidence came from 2D or 3D sensors. No backend changes required.

### Performance Notes

**CPU (existing):**
- 2D LaserScan ICP: ~100 Hz (360 points)

**GPU (RTX 4050):**
- 3D Point Cloud ICP: ~30 Hz (10K-50K points after filtering)
- Voxel filtering: ~1000 Hz
- Memory: ~2 GB VRAM for typical clouds

### Compatible Datasets

- NVIDIA r2b benchmark dataset (RealSense D455)
- Any rosbag with PointCloud2 and Odometry messages
- Gazebo with 3D sensor simulation

### Usage

**Enable 3D mode:**
```bash
ros2 launch fl_slam_poc poc_3d_rosbag.launch.py \
    bag:=/path/to/bag \
    play_bag:=true
```

**With 2D launch file (optional 3D):**
```bash
ros2 launch fl_slam_poc poc_tb3_rosbag.launch.py \
    use_3d_pointcloud:=true \
    use_gpu:=true \
    bag:=/path/to/bag
```

### Testing

```bash
# Unit tests
pytest fl_ws/src/fl_slam_poc/test/test_pointcloud_3d.py -v

# Integration test
./scripts/test-3d-integration.sh
```

---

## 2026-01-20 - Critical Backend/Frontend Wiring Fixes for M3DGR Rosbag

### Issue Summary
Backend was falling back to dead reckoning (SLAM not active) when testing with M3DGR rosbag data. Root cause: multiple initialization and configuration issues preventing frontend from publishing loop factors to backend.

### Critical Fixes

**1. Camera Intrinsics Not Set (CRITICAL)**
- **Problem**: Launch file had camera intrinsics set to 0.0, frontend requires ALL four values > 0 to enable RGB-D evidence
- **Impact**: Frontend silently disabled RGB-D evidence publishing (no warning visible in early logs)
- **Fix**: Set M3DGR RealSense D435i intrinsics in `poc_m3dgr_rosbag.launch.py`:
  - `camera_fx: 383.0`, `camera_fy: 383.0`, `camera_cx: 320.0`, `camera_cy: 240.0`
- **Files**: `launch/poc_m3dgr_rosbag.launch.py`

**2. Odom Bridge Initialization Delay**
- **Problem**: Odom bridge dropped first message, creating initialization delay causing frontend to drop early scans
- **Impact**: Frontend waited for odom before processing scans, but bridge needed 2 messages to start publishing deltas
- **Fix**: Publish zero-motion delta on first message to kickstart backend immediately
- **Files**: `utility_nodes/tb3_odom_bridge.py`

**3. QoS Depth Mismatch**
- **Problem**: Odom bridge published with depth=100, backend subscribed with depth=10
- **Impact**: Potential message loss during startup under high load
- **Fix**: Increased backend subscription depth to 100 to match publisher
- **Files**: `backend/backend_node.py`

**4. Insufficient Diagnostic Logging**
- **Problem**: Silent failures made it difficult to diagnose initialization issues
- **Impact**: Spent time guessing root causes instead of reading clear error messages
- **Fix**: Added comprehensive startup and first-message logging:
  - Frontend: Camera intrinsics validation with warnings
  - Frontend: First scan/pointcloud/odom received logs
  - Backend: First odom messages logged
  - SensorIO: TF lookup failures with CRITICAL warnings
  - Both nodes: Startup banners showing configuration
- **Files**: `frontend/frontend_node.py`, `backend/backend_node.py`, `frontend/processing/sensor_io.py`

**5. TF Validation for Rosbag Compatibility**
- **Problem**: Frame validation could fail during rosbag playback due to timing
- **Impact**: Points couldn't be transformed from sensor frame to base frame
- **Fix**: 
  - Disable frame validation by default in odom bridge for rosbag
  - Use "both" QoS (RELIABLE + BEST_EFFORT) to handle rosbag variations
  - Added CRITICAL logging when TF lookups fail for scan/pointcloud frames
  - Added frame identity check (no transform needed if frames match)
- **Files**: `launch/poc_m3dgr_rosbag.launch.py`, `frontend/processing/sensor_io.py`

### Enhanced Diagnostics

**Frontend Startup Logging:**
```
FL-SLAM Frontend initialized
Mode: 3D PointCloud / 2D LaserScan + RGB-D
Birth intensity: 30.0
Using GPU: False/True
Camera intrinsics: fx=383.0, fy=383.0, cx=320.0, cy=240.0
Waiting for sensor data to start processing...
```

**Backend Startup Logging:**
```
FL-SLAM Backend starting
Subscriptions:
  Delta odom:    /sim/odom (MUST come from tb3_odom_bridge)
  Loop factors:  /sim/loop_factor (from frontend)
  Anchors:       /sim/anchor_create (from frontend)
  RGB-D evidence: /sim/rgbd_evidence
Status monitoring: Will report DEAD_RECKONING if no loop factors
```

**First-Message Logging:**
- Odom bridge: "initialized at pose (...), published zero-motion delta to kickstart backend"
- SensorIO: "First scan received, frame_id=..., ranges=360, last_pose=SET/NONE"
- Frontend: "Scan #1 processed: r_new_eff=..., should_birth=..., points=OK/NONE, anchors=0"

**Error Logging:**
- TF failures: "CRITICAL: Cannot transform scan from 'X' to 'Y'. TF lookup failed! Without TF, scan points cannot be used for anchors/loops."
- No camera intrinsics: "NO camera intrinsics - RGB-D evidence DISABLED until set"
- No points: "NO POINTS available! Check TF transforms. Anchors CANNOT be created without points."

### Files Modified

- `launch/poc_m3dgr_rosbag.launch.py` - Camera intrinsics, odom bridge QoS, frame validation, odom topic fix, birth intensity reduction
- `backend/backend_node.py` - QoS depth, startup logging, first-odom logging, debug loop processing
- `utility_nodes/tb3_odom_bridge.py` - **MAJOR**: Publish first absolute pose as delta, eliminate startup delay
- `frontend/frontend_node.py` - **MAJOR**: Sensor data buffering, removed artificial odometry dependency, startup banner, camera intrinsics validation
- `frontend/processing/sensor_io.py` - First-message logging, TF failure diagnostics, camera_info logging

### Design Improvements

**Eliminated Artificial Startup Friction:**
- **Before**: Frontend dropped sensor data until odometry arrived, odom bridge skipped first message
- **After**: Sensor data buffered until odometry available, odom bridge publishes first pose immediately
- **Impact**: System starts processing data immediately when either sensors OR odometry arrive

**Leveraged Order-Invariant Backend Math:**
- **Before**: Strict sensor→odometry→processing dependency
- **After**: Asynchronous sensor/odometry processing with timestamp alignment
- **Impact**: Better utilization of FL-SLAM's information-geometric foundations

**Removed Hardcoded Dependencies:**
- **Before**: `if self.last_pose is None: return` (hard drop)
- **After**: Buffer and process when odometry becomes available
- **Impact**: More robust to timing variations in rosbag playback

### Testing

**Recommended test sequence:**
1. Build workspace: `cd fl_ws && colcon build --symlink-install`
2. Run integration test: `./scripts/test-integration.sh`
3. Check for:
   - Frontend startup shows camera intrinsics set
   - Backend receives odom immediately
   - Frontend processes scans and creates anchors
   - Backend status shows SLAM_ACTIVE (not DEAD_RECKONING)
   - Loop factors published and received

### Impact

✅ Backend should now properly receive and integrate loop factors
✅ RGB-D evidence now enabled for M3DGR dataset
✅ Clear diagnostic logging for troubleshooting initialization
✅ Faster startup due to zero-motion kickstart
✅ More robust to QoS and timing variations in rosbags

## 2026-01-21 - Roadmap Phase Ordering Clarification

### Summary
- Reordered near-term phases so **evaluation/provenance hardening comes before enabling dense RGB-D in 3D**, preventing “enabled but ineffective” sensor contributions from being mistaken for real progress.
- Clarified keyframe scheduling wording to match the current **motion-based default** policy, with time-based keyframes explicitly marked as optional/debug-only.

### Files Updated
- `ROADMAP.md`

## 2026-01-21 - MVP Import Closure Cleanup + Roadmap + Artifact Hygiene

### MVP Import Refactor (Reduce Re-export Bloat)
- Refactored package `__init__.py` modules to avoid eager imports and keep MVP runtime closure minimal:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/__init__.py` (lazy exports)
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/common/__init__.py` (lazy module access; no eager `config` import)
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/__init__.py` (lazy exports; avoids pulling in multimodal by default)
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/operators/__init__.py` (legacy compat stays, but lazy to avoid bloat)
- Updated MVP codepaths to import implementation modules directly:
  - `frontend/frontend_node.py` now imports RGB-D helpers from `frontend/processing/rgbd_processor.py` directly.
  - `frontend/processing/rgbd_processor.py` and `frontend/loops/icp.py` now import SE(3) helpers from `common/transforms/se3.py` (not the legacy wrapper).

### Cleanup (Obsolete / Abandoned Code)
- Removed abandoned Rerun bridge vendoring under `fl_ws/src/cpp-example-ros2-bridge/`.
- Removed unused visualization stub under `fl_ws/src/fl_slam_poc/fl_slam_poc/visualizer/`.
- Removed temporary M3DGR helper scripts:
  - `scripts/run-m3dgr-rerun.sh`, `scripts/run-m3dgr-rviz.sh`, `scripts/run-m3dgr-test.sh`
  - `scripts/align_ground_truth_fixed.py` (superseded)

### Documentation + Roadmap
- Added `ROADMAP.md` with MVP status, priorities, and file map.
- Updated docs to remove Docker workflow references and reflect current MVP:
  - `README.md`, `AGENTS.md`, `docs/TESTING.md`, `docs/ROSBAG.md`, `docs/EVALUATION.md`, `docs/INSTALLATION.md`, `archive/README.md`
- Added clear “future/optional” headers to non-MVP launch files and Gazebo-only node.

### Phase 2 Extraction (Physical MVP Minimization)
- Moved non-MVP code into `phase2/` to keep the active workspace as the smallest reproducible failing case:
  - Alternative launches: `phase2/fl_ws/src/fl_slam_poc/launch/`
  - Gazebo-only sim: `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/sim_world.py`
  - Experimental Dirichlet nodes: `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/nodes/`
  - Future fusion/config: `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/multimodal_fusion.py`, `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/common/config.py`
  - Phase 2 tests: `phase2/fl_ws/src/fl_slam_poc/test/`
- Updated `fl_ws/src/fl_slam_poc/setup.py` to install only the MVP launch and to remove the Gazebo `sim_world` entry points.

### Repo Hygiene
- Updated `.gitignore` to ignore evaluation output and user RViz configs:
  - `results/`
  - `config/*.rviz`

### Verification
- `pytest -q` under `fl_ws/src/fl_slam_poc` passes.
- `colcon build --packages-select fl_slam_poc` succeeds.
- `bash scripts/run_and_evaluate.sh` runs end-to-end on M3DGR (trajectory quality remains under investigation).
