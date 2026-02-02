# Bloat and Excess Complexity Audit

**Date:** 2026-01-30  
**Purpose:** Identify files to move to `archive/` or delete to slim the tree. Code is source of truth; pipeline and entry points define reachability.

---

## 1. Runtime reachability (actual)

**Entry points:** `gc_backend_node`, `gc_sensor_hub`, pipeline (LiDAR → process_scan_single_hypothesis).

**Pipeline imports** (from `pipeline.py`): point_budget, predict, deskew, measurement_noise_iw, odom_evidence, odom_twist_evidence, imu_evidence, planar_prior, imu_gyro_evidence, imu_preintegration_factor, fusion, excitation, recompose, MapUpdateResult (map_update), anchor_drift, hypothesis, bin_atlas (structures), primitive_map, measurement_batch, lidar_surfel_extraction, primitive_association, visual_pose_evidence, flatten_associations_for_fuse, primitive_map_fuse.

**Pipeline does NOT import or call:** bin_soft_assign, scan_bin_moment_match, matrix_fisher_evidence, planar_translation_evidence, pos_cov_inflation_pushforward, sinkhorn_ot_bev, ot_fusion, bev_pushforward, splat_batch, rendering, splat_prep, visual_feature_extractor.

**Frontend (gc_sensor_hub):** livox_converter, pointcloud_passthrough, odom_normalizer, imu_normalizer, dead_end_audit. No splat_prep, visual_feature_extractor.

**Used by pipeline/frontend but not entry:** lidar_camera_depth_fusion._fit_plane_weighted (used by lidar_surfel_extraction, lidar_surfels). So lidar_camera_depth_fusion stays; only _fit_plane_weighted is on path.

---

## 2. Archive candidates (code not on runtime path)

### 2.1 Backend operators – replaced legacy only (archived)

| Item | Location | Action |
|------|----------|--------|
| binning.py | fl_slam_poc/backend/operators/ | **Done:** in archive/legacy_operators/ |
| matrix_fisher_evidence.py | fl_slam_poc/backend/operators/ | **Done:** in archive/legacy_operators/ |

**Do NOT archive:** ot_fusion.py, sinkhorn_ot_bev, bev_pushforward, splat_batch, rendering, splat_prep, visual_feature_extractor. These are implementation to **wire into the pipeline** (visual–LiDAR integration; see `.cursor/plans/visual_lidar_rendering_integration_209f2c28.plan.md` Section 3). ot_fusion.py was mistakenly removed; restored from git.

**After move:** Removed from `fl_slam_poc/backend/operators/__init__.py` only: bin_soft_assign, scan_bin_moment_match, BinSoftAssignResult, ScanBinStats; pos_cov_inflation_pushforward. Kept: sinkhorn_ot_bev, ot_fusion exports (for future wiring).

**Note:** archive/legacy_operators/lidar_evidence.py uses `from .binning import ScanBinStats`; archive/legacy_operators/__init__.py added.

### 2.2 Common / backend – to wire (do NOT archive)

| Item | Location | Action |
|------|----------|--------|
| bev_pushforward.py | fl_slam_poc/common/ | **Keep** – wire into pipeline (visual–LiDAR plan). |
| splat_batch.py | fl_slam_poc/common/ | **Keep** – wire into pipeline. |
| rendering.py | fl_slam_poc/backend/ | **Keep** – wire into pipeline. |

### 2.3 Frontend sensors – to wire (do NOT archive)

| Item | Location | Action |
|------|----------|--------|
| splat_prep.py | fl_slam_poc/frontend/sensors/ | **Keep** – wire into pipeline. |
| visual_feature_extractor.py | fl_slam_poc/frontend/sensors/ | **Keep** – wire into pipeline. |

Keep lidar_camera_depth_fusion (used for _fit_plane_weighted by lidar_surfel_extraction, lidar_surfels).

---

## 3. Dead exports only (keep file, slim __init__)

| Export | File | Action |
|--------|------|--------|
| pos_cov_inflation_pushforward | map_update.py | Remove from backend/operators/__init__.py (never called; MapUpdateResult stays). |

map_update.py stays; only the pushforward function is dead. MapUpdateResult is used by pipeline.

---

## 4. Delete (bloat / accidental)

| Item | Location | Reason |
|------|----------|--------|
| os | project root | **Done:** Deleted; `/os` added to .gitignore. |

---

## 5. Config and docs (keep vs archive)

**Keep:** config/cyclonedds*.xml, config/m3dgr_body_T_wheel.yaml (used by run_and_evaluate_gc.sh and tools).

**Docs to consider archiving** (run-logs, superseded, or low value):

| Doc | Suggestion |
|-----|------------|
| docs/MAP_AND_BELIEF_AUDIT_2026-01-30.md | Run-log; move to archive/docs/ if desired. |
| docs/PREINTEGRATION_STEP_BY_STEP.md | If superseded by OPERATOR_CONTRACTS / PIPELINE_TRACE_SINGLE_DOC, archive. |
| docs/MAP_VISUALIZATION.md | Describes hybrid laser + RGB-D architecture; if not current GC v2, archive. |
| docs/PRODUCTION_READINESS_SPLAT_PIPELINE.md | Describes unwired splat/BEV stack; keep as reference or move to archive. |
| docs/Comprehensive Information Geometry.md | Long theory doc; keep for reference. |
| docs/system_dataflow_d3.html | If obsolete, archive. |

**Keep:** FRAME_AND_QUATERNION_CONVENTIONS, GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC, IMU_BELIEF_MAP_AND_FUSION, OPERATOR_CONTRACTS, PIPELINE_*, POST_INTEGRATION_CHECKLIST_AUDIT, SIGMA_G_AND_FUSION_EXPLAINED, BAG_TOPICS_AND_USAGE, TESTING, EVALUATION, POINTCLOUD2_LAYOUTS, etc.

---

## 6. Tools (one-off vs keep; broken)

| Tool | Status |
|------|--------|
| run_and_evaluate_gc.sh | **Keep** – primary eval. |
| evaluate_slam.py, evaluate_trajectory_2d.py | **Keep** – used by eval. |
| transform_estimate_to_body_frame.py, kimera_gt_to_tum.py, prep_kimera_data.sh, run_gc_kimera.sh | **Keep** – used by eval/Kimera. |
| estimate_lidar_base_extrinsic.py | **Broken** – imports `fl_slam_poc.frontend.scan.icp.icp_3d`; no `frontend/scan/icp` in tree. Either archive tool or add minimal icp stub (e.g. in tools/ or common). |
| diagnose_*, compare_*, check_*, validate_*, inspect_*, dump_*, plot_*, confirm_*, capture_*, dead_reckon_*, first_n_*, generate_trajectory_*, align_ground_truth, apply_imu_extrinsic_*, estimate_* (others) | One-off diagnostics; **keep** in tools/ or move rarely-used to archive/tools/ to slim. |
| install_code_graph_rag_mcp.sh, README_MCP.md | **Keep** if MCP is used; else archive. |

---

## 7. Summary of recommended actions

**Done:**

1. **Archived (replaced legacy only):** binning.py, matrix_fisher_evidence.py → archive/legacy_operators/. Removed their exports and pos_cov_inflation_pushforward from backend/operators/__init__.py. Updated archive/legacy_operators/lidar_evidence.py to `from .binning import ScanBinStats`; added archive/legacy_operators/__init__.py.
2. **Restored:** ot_fusion.py (was mistakenly removed) – kept in backend/operators; exports remain for future wiring.
3. **Do NOT archive:** bev_pushforward, splat_batch, rendering, splat_prep, visual_feature_extractor – these are to be **wired into the pipeline** (visual–LiDAR integration plan).
4. Deleted root file `os`; added `/os` to .gitignore.

**Remaining:**

5. Fix or archive: tools/estimate_lidar_base_extrinsic.py (broken import frontend.scan.icp).

**Optional (docs):**

6. Archive run-log or superseded docs (MAP_AND_BELIEF_AUDIT, PREINTEGRATION_STEP_BY_STEP, MAP_VISUALIZATION, PRODUCTION_READINESS_SPLAT_PIPELINE) if desired.
7. Move rarely-used diagnostic tools to archive/tools/ to slim tools/ (optional).

---

## 8. What stays (no change)

- bin_atlas (structures) – used for map_stats state.
- map_update.py – MapUpdateResult used; only pushforward export removed from __init__.
- sinkhorn_ot.py – used by primitive_association (fixed_k); sinkhorn_ot_bev and ot_fusion exports kept in __init__ for future BEV/visual wiring.
- lidar_camera_depth_fusion.py – _fit_plane_weighted used by lidar_surfel_extraction, lidar_surfels.
- All other backend operators and structures on the pipeline path.
- config/ (cyclonedds, m3dgr_body_T_wheel).
- Core docs and eval scripts.
