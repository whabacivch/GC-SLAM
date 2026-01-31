# Usage and Redundancy Audit (Visual + LiDAR Pipeline)

**Date:** 2026-01-30  
**Scope:** Files built for visual/LiDAR/splat/OT/rendering; whether each is used or redundant.

---

## Used and Wired

| File / symbol | Where used | Role |
|---------------|------------|------|
| **primitive_association.py** | `pipeline.py` | `associate_primitives_ot()` — OT association measurement ↔ map primitives |
| **visual_pose_evidence.py** | `pipeline.py` | `visual_pose_evidence()`, `build_visual_pose_evidence_22d()` — pose evidence from primitive alignment |
| **sinkhorn_ot.py** (internal) | `primitive_association.py` | `sinkhorn_balanced_fixed_k`, `sinkhorn_unbalanced_fixed_k`, `w2_sq_2d`, `_hellinger2_vmf`, `SinkhornOTConfig` — 3D cost and fixed-K Sinkhorn |
| **camera_batch_utils.py** | `backend_node.py` | `feature_list_to_camera_batch()` — camera splats → MeasurementBatch |
| **map_publisher.py** | `backend_node.py` | `PrimitiveMapPublisher` — map → PointCloud2 + optional Rerun |
| **rerun_visualizer.py** | `backend_node.py`, `map_publisher.py` | `RerunVisualizer` — map/trajectory → Rerun |
| **lidar_surfel_extraction.py** | `pipeline.py` | `extract_lidar_surfels()` — LiDAR → MeasurementBatch, merge with camera batch |
| **ma_hex_web.py** | `primitive_association.py` | `MAHexWebConfig`, `generate_candidates_ma_hex_web()` — candidate generation for OT |
| **primitives.py** | Many | SPD/solve/lift, domain_projection_psd; used by primitive_map, IW, predict, odom_evidence |
| **lidar_camera_depth_fusion.py** | `splat_prep.py`, `lidar_surfel_extraction.py`, `lidar_surfels.py` | Depth fusion, backproject, `_fit_plane_weighted` |
| **pointcloud_passthrough.py** | `gc_sensor_hub.py` | LiDAR path when config uses pointcloud_passthrough (e.g. Kimera) |
| **splat_prep.py** | `backend_node.py` | `splat_prep_fused()` — visual + LiDAR depth → Feature3D list |
| **visual_feature_extractor.py** | `backend_node.py`, `splat_prep.py`, `camera_batch_utils.py` | Features, ExtractionResult, Feature3D, PinholeIntrinsics |
| **lidar_surfels.py** | `lidar_surfel_extraction.py` | Voxel surfels, plane fit, Gaussian+vMF; `_fit_plane_weighted` from fusion |

---

## Unused / Redundant

### 1. **backend/rendering.py**

- **What:** EWA splatting, vMF shading, fBm, `render_tile_ewa()`, `SplatRenderingConfig`.
- **Used by:** Nothing. Only referenced in `backend/__init__.py` as a comment.
- **Why redundant:** Map output is PointCloud2 + Rerun points (`map_publisher` + `rerun_visualizer`). No code path produces a rendered image from splats.
- **Action:** Archive or remove if image-from-splats is not planned.

### 2. **common/bev_pushforward.py**

- **What:** `pushforward_gaussian_3d_to_2d()`, `oblique_P()`, `BEVPushforwardConfig`, vMF→S¹ stub.
- **Used by:** Nothing.
- **Why redundant:** Designed for a BEV OT path (3D → 2D then Sinkhorn on BEV). The implemented path is 3D primitives + MA hex + Sinkhorn on 3D cost in `primitive_association.py`; no BEV projection is used.
- **Action:** Archive or remove unless BEV OT is reintroduced.

### 3. **common/splat_batch.py**

- **What:** `PackedSplatBatch`, `pack_splat_batch()`, caps `GC_OT_N_MAX`, `GC_OT_M_MAX`, `GC_OT_K_STENCIL`.
- **Used by:** Nothing.
- **Why redundant:** Pipeline uses `MeasurementBatch` (3D positions/covs/directions) and `associate_primitives_ot()` with MA hex + sparse cost; it does not pack camera/LiDAR BEV splats into `PackedSplatBatch` or call `pack_splat_batch`.
- **Action:** Archive or remove unless a BEV packed-batch OT path is added.

### 4. **sinkhorn_ot_bev** (in backend/operators/sinkhorn_ot.py)

- **What:** Public function `sinkhorn_ot_bev(mu_cam, Sigma_cam, ..., kappa_cam, mu_lidar, ...)` and helper `cost_matrix_bev()`.
- **Used by:** Only internally (cost_matrix_bev is called by sinkhorn_ot_bev). No other module imports `sinkhorn_ot_bev`.
- **Why redundant:** Association uses `associate_primitives_ot()` → `_compute_sparse_cost_matrix()` (3D W2² + H²_vMF) → `sinkhorn_balanced_fixed_k` / `sinkhorn_unbalanced_fixed_k`. The BEV entry point that takes pre-projected 2D splats is never called.
- **Action:** Stop exporting `sinkhorn_ot_bev` from `operators/__init__.py`; optionally move `sinkhorn_ot_bev` and `cost_matrix_bev` to archive or leave in file but document as unused BEV path.

---

## Summary

- **Used:** primitive_association, visual_pose_evidence, sinkhorn_ot (3D helpers only), camera_batch_utils, map_publisher, rerun_visualizer, lidar_surfel_extraction, ma_hex_web, primitives, lidar_camera_depth_fusion, pointcloud_passthrough, splat_prep, visual_feature_extractor, lidar_surfels.
- **Unused:** rendering.py (full file), bev_pushforward.py (full file), splat_batch.py (full file), sinkhorn_ot_bev + cost_matrix_bev (dead public BEV path in sinkhorn_ot.py).

No duplicate implementations of the same behavior were found; the redundancy is “alternate design not wired” (BEV OT + packed batch + image rendering) vs the wired 3D primitive OT + PointCloud2/Rerun output.
