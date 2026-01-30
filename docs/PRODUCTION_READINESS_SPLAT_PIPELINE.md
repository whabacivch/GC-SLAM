# Production Readiness: LiDAR–Camera Splat Fusion and BEV OT

This document identifies **what would need to be changed or adjusted** to put the LiDAR–camera splat fusion and BEV OT stack into production. It is based on tracing the **actual** pipeline and backend code; docs are design, code is source of truth.

**References:** `docs/IMU_BELIEF_MAP_AND_FUSION.md`, `fl_ws/.../backend/backend_node.py`, `fl_ws/.../backend/pipeline.py`, `AGENTS.md`.

---

## 1. Current State (Actual Behavior)

### 1.1 Backend subscriptions and pipeline trigger

- **Code:** `backend_node.py` subscribes only to:
  - `/gc/sensors/lidar_points` → `on_lidar()` (triggers pipeline)
  - `/gc/sensors/odom` → `on_odom()`
  - `/gc/sensors/imu` → `on_imu()`
- **No camera or image topic** is subscribed to anywhere in the **backend** or pipeline.
- **Camera decompression already exists:** C++ node `image_decompress_cpp` (`src/image_decompress_node.cpp`) subscribes to `/camera/color/image_raw/compressed` and `/camera/aligned_depth_to_color/image_raw/compressedDepth`, publishes `/camera/image_raw` (rgb8) and `/camera/depth/image_raw` (32FC1). The backend does not subscribe to these outputs.
- Pipeline is **LiDAR-driven**: each LiDAR message runs the full 14-step scan pipeline (point budget → … → anchor drift). No visual/splat/OT steps run.

### 1.2 Pipeline steps (actual)

- Steps 1–14 in `pipeline.py` use: raw points, timestamps, weights, ring, tag, IMU (deskew + evidence), odom (pose + twist), bin atlas, map_stats (directional bins: `S_dir`, `N_dir`, etc.), belief.
- **No step** uses: image, camera intrinsics, visual features, splats, BEV pushforward, Sinkhorn OT, or OT fusion.
- Map state is **directional bin statistics** (`map_stats`), not BEV splats. `pos_cov_inflation_pushforward` writes scan increments into those bins; there is no “splat map” or BEV grid state.

### 1.3 Implemented but unwired modules

- **Frontend/sensors:** `visual_feature_extractor`, `lidar_camera_depth_fusion`, `splat_prep`, `lidar_surfels`, `splat_rendering`.
- **Common:** `bev_pushforward`, `splat_batch` (packed batch with N_max, M_max, K_max, masks, neighbor indices), `ma_hex_web`.
- **Backend operators:** `sinkhorn_ot_bev`, `coupling_to_weights`, `weighted_fusion_gaussian_bev`, `weighted_fusion_vmf_bev`, `confidence_tempered_gamma`, `wishart_regularize_2d`, `temporal_smooth_lambda`.
- These are **callable from scripts** but **not invoked** from `backend_node` or `pipeline.py`.

### 1.4 Runtime manifest

- **Code:** `RuntimeManifest` in `pipeline.py` (and published at startup from `backend_node._publish_runtime_manifest`) includes `backends` (core_array, se3, deskew, imu_evidence, odom_evidence, lidar_evidence, map_update, lidar_converter, pointcloud_parser).
- **No** entries for: visual_extractor, splat_prep, bev_pushforward, sinkhorn_ot, ot_fusion.
- **No** “enabled_sensors” or “resolved camera topic” in the manifest today.

---

## 2. What Would Need to Change for Production

### 2.1 Topics and frontend (camera input)

**Existing C++ node:** `fl_ws/src/fl_slam_poc/src/image_decompress_node.cpp` (executable `image_decompress_cpp`) already subscribes to camera topics and publishes decompressed images:

- **Subscribes to:** `/camera/color/image_raw/compressed` (CompressedImage), `/camera/aligned_depth_to_color/image_raw/compressedDepth` (CompressedImage).
- **Publishes:** `/camera/image_raw` (sensor_msgs/Image, rgb8), `/camera/depth/image_raw` (sensor_msgs/Image, 32FC1, m). Params: `rgb_compressed_topic`, `depth_compressed_topic`, `rgb_output_topic`, `depth_output_topic`, `depth_scale_mm_to_m`, `qos_reliability`.

So the **camera input path exists**: bags with compressed camera topics can be decompressed by this node. The gap is that the **backend** does not subscribe to those outputs (or to canonical topics fed by them).

| Item | Current | Change needed |
|------|--------|----------------|
| Camera image source | C++ **image_decompress_cpp** publishes `/camera/image_raw` and `/camera/depth/image_raw`. Built via CMakeLists.txt; **not** currently included in `gc_rosbag.launch.py`. | When enabling camera: add `image_decompress_cpp` to the launch file (or run it separately) so decompressed images are available for the backend. |
| Canonical topic | Backend subscribes only to `/gc/sensors/*`. | Either (a) Backend subscribes directly to `/camera/image_raw` and `/camera/depth/image_raw`, or (b) Add a republish to canonical topic(s) (e.g. `/gc/sensors/camera_image`, `/gc/sensors/camera_depth`) and backend subscribes to those. Option (b) matches the lidar/odom/imu pattern (canonical under `/gc/sensors/`). |
| Backend subscription | None | Backend subscribes to image (and depth) topic(s). **Fail-fast:** if `use_camera=True` and topic unavailable at startup, node fails (per AGENTS.md no fallbacks). |
| Time sync | — | Decide: trigger on LiDAR (current) and use “latest image” or nearest stamp to scan time; or document that image is sampled at scan time. No TF at runtime; sync by stamp or fixed policy. |

### 2.2 Backend node: parameters and state

| Item | Current | Change needed |
|------|--------|----------------|
| Extrinsics | `T_base_lidar`, `T_base_imu` | Add **T_base_camera** (or T_camera_lidar) for transforming LiDAR points to camera frame for depth fusion. |
| Intrinsics | — | Camera intrinsics (fx, fy, cx, cy) from `camera_info` topic or ROS params. |
| Config flags | — | e.g. `use_camera: bool`, `camera_topic`, `camera_info_topic` (or intrinsics from params). Fail-fast if enabled and topic/params missing. |
| State | belief, map_stats, IMU/odom buffers | If temporal smoothing for OT fusion is used: **persistent BEV/splat state** (e.g. Lambda_prev per cell or per map primitive) with **stable IDs**. Otherwise temporal_smooth_lambda has no prior; doc says “do not use on per-frame features without re-association.” |

### 2.3 Data flow: from image and LiDAR to fused BEV

End-to-end flow that **does not exist** in code today:

1. **Image** → visual feature extractor → list of `Feature3D` (camera frame, with depth meta).
2. **LiDAR points** in base frame → transform to **camera frame** (using T_base_camera) → depth fusion (Route A or B) + backproject → **splat_prep_fused** → fused `Feature3D` list.
3. **Fused camera splats** (3D) → **BEV pushforward** (oblique P(φ)) → camera BEV: `mu_cam`, `Sigma_cam`, `mu_n_cam`, `kappa_cam`.
4. **LiDAR** (same scan) → **lidar_surfels** (voxel + plane fit) → **BEV pushforward** → LiDAR BEV: `mu_lidar`, `Sigma_lidar`, etc.
5. **pack_splat_batch**(cam BEV, lidar BEV, N_max, M_max, K_max, neighbor_indices?) → **PackedSplatBatch** (fixed shapes, masks).
6. **sinkhorn_ot_bev**(packed batch, config) → π, CertBundle, ExpectedEffect.
7. **coupling_to_weights**(π) → w; **confidence_tempered_gamma**(π, γ, α, m0) → γ_i; **weighted_fusion_gaussian_bev** / **weighted_fusion_vmf_bev**(…, gamma_per_row=γ_i) → Λ_f, θ_f (and vMF).
8. **wishart_regularize_2d** / **temporal_smooth_lambda** (in BEV/map frame; temporal only if stable IDs).

**Where this runs:** Either (A) **new pipeline steps** (e.g. after 9 or 14) that run when camera is enabled, or (B) a **parallel “splat branch”** in `on_lidar`: after `process_scan_single_hypothesis` / `process_hypotheses`, run the chain above and publish or store the result. Option (A) requires adding steps and possibly evidence (e.g. L_visual); option (B) is output-only (no pose feedback) unless we add a separate evidence path.

### 2.4 Pipeline integration options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **Output-only** | Run splat chain in `on_lidar` after existing pipeline; publish fused BEV or 3D splats (e.g. `/gc/splat_map` or export). | Minimal change to steps 1–14; no pose coupling. | Camera–LiDAR fusion does not affect pose belief. |
| **Evidence path** | Fused BEV (or alignment residual) produces an **evidence term** (e.g. L_visual) added in step 9 with other evidence. | Camera–LiDAR can correct pose. | Requires design: what residual, what information matrix, how to avoid double-counting with LiDAR bins. |
| **New pipeline steps** | Add formal steps (e.g. “Visual splat prep”, “BEV pushforward”, “Sinkhorn OT”, “OT fusion”) in `pipeline.py`. | Explicit, auditable. | Larger refactor; step order and certificate aggregation must be defined. |

Recommendation: **Start with output-only** (splat branch in `on_lidar`, publish or write fused splats). Add evidence or new steps once the data path is validated and the observation model is specified.

### 2.5 Configuration and constants

- **PipelineConfig** (and ROS params): add camera/splat/OT parameters (e.g. enable_camera, camera_topic, T_base_camera, intrinsics or camera_info_topic, LidarCameraDepthFusionConfig, BEVPushforwardConfig, SinkhornOTConfig, OTFusionConfig). N_max, M_max, K_max can stay in `splat_batch` constants or be overridable via config.
- **RuntimeManifest:** extend `backends` with: `visual_extractor`, `splat_prep`, `bev_pushforward`, `sinkhorn_ot`, `ot_fusion`. Add **enabled_sensors** (e.g. `["lidar", "imu", "odom"]` and `"camera"` when enabled) and **resolved** camera topic and intrinsics/extrinsics so tests can assert manifest content (per AGENTS.md).

### 2.6 Frame and ID consistency (already documented in code)

- **Wishart / temporal:** Applied on Λ in a **consistent frame** (e.g. BEV/map), not camera frame. **Temporal smoothing** requires **stable feature IDs** (e.g. map primitives or BEV cell keys); not per-frame feature indices. So: either (i) persist a BEV grid or splat map with stable IDs and run temporal_smooth_lambda across scans, or (ii) skip temporal smoothing until that state exists.

### 2.7 Performance and cost

- **Cost matrix:** O(N×M) per scan (e.g. 2048×2048 ≈ 4M W2² + vMF cost evaluations). Current implementation is Python/NumPy; for real-time, consider JAX JIT for `cost_matrix_bev` and Sinkhorn, or lower N_max/M_max.
- **Visual extractor:** Per-feature loop in NumPy; batching or GPU would help for large feature counts.
- **Packed batch:** Fixed caps and masks are in place to avoid dynamic shapes and recompiles when moving to JIT.

### 2.8 Tests and validation

- **Runtime manifest:** `test_golden_child_invariants` (or equivalent) should assert manifest contains expected backends and, when camera is enabled, camera topic and enabled_sensors. Today manifest is logged; tests can parse published JSON.
- **Splat path:** No test currently runs splat_prep → BEV → Sinkhorn → OT fusion end-to-end. Add a unit or integration test (script or test node) that mocks image + LiDAR and runs the full chain to catch shape/contract errors.
- **Frame conventions:** When T_base_camera is added, validate in tests (or doc) that LiDAR points are transformed correctly (e.g. p_cam = R_base_cam^T @ (p_base - t_base_cam) or equivalent per `FRAME_AND_QUATERNION_CONVENTIONS.md`).

### 2.9 Documentation

- **IMU_BELIEF_MAP_AND_FUSION.md:** Add a section (or link to a dedicated doc) for camera–LiDAR splat fusion: topics, data flow, where it plugs in (output-only vs evidence), and frame conventions.
- **PIPELINE_DESIGN_GAPS.md:** Add a gap “Camera/visual and BEV OT not in live pipeline” and mark it closed when the above wiring is done and documented.

---

## 3. Checklist Summary

| Category | Action |
|----------|--------|
| **Topics** | Define canonical camera topic(s); add camera normalizer node; backend subscribe + fail-fast if use_camera and missing. |
| **Params / state** | T_base_camera, intrinsics (camera_info or params), use_camera, camera_topic; optional Lambda_prev / BEV map state for temporal smoothing. |
| **Data flow** | In on_lidar (or new steps): image → extractor; LiDAR→camera frame → depth fusion → splat_prep_fused; cam BEV + lidar BEV → pack_splat_batch → sinkhorn_ot_bev → coupling_to_weights + confidence_tempered_gamma → weighted_fusion_* → wishart + temporal (if state exists). |
| **Pipeline** | Choose output-only vs evidence vs new steps; implement one path first (recommend output-only). |
| **Config** | PipelineConfig + ROS params for camera/splat/OT; RuntimeManifest extended (backends, enabled_sensors, camera topic/intrinsics). |
| **Frames / IDs** | Apply Wishart/temporal in BEV/map frame; temporal only with stable IDs (persistent splat map or BEV grid). |
| **Performance** | Consider JAX JIT for cost + Sinkhorn; or reduce N_max/M_max; batch or accelerate visual extractor if needed. |
| **Tests** | Assert manifest when camera enabled; add end-to-end splat path test; validate T_base_camera usage. |
| **Docs** | Update pipeline doc with camera/splat flow; record and close “camera not in pipeline” gap. |

---

## 4. Principle (from AGENTS.md)

- **Actual behavior comes from the code.** This doc was derived by tracing `backend_node.py` and `pipeline.py`. When implementing, keep code and docs aligned; report mismatches first, then fix.
- **No fallbacks:** If camera is enabled, required topic and params must be present or the node fails at startup.
- **Single path:** One canonical data path for splat fusion (e.g. always use packed batch → Sinkhorn → OT fusion with confidence tempering); no alternate backends or environment-based selection for this stack.
