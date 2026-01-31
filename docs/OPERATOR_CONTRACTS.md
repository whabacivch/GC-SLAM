# Operator List with Contracts (GC v2)

Canonical operator list for the Visual–LiDAR integration pipeline. Each operator has fixed budgets/iterations, returns `(result, CertBundle, ExpectedEffect)`, and obeys the Frobenius rule: `approximation_triggers ≠ ∅ ⇒ frobenius_applied`.

Reference: `.cursor/plans/visual_lidar_rendering_integration_*.plan.md` Section 11.

## Budget names (required config; no silent defaults)

- **N_FEAT** — Camera/splat feature budget
- **N_SURFEL** — LiDAR surfel budget
- **K_ASSOC** — Candidate neighborhood size per measurement
- **K_SINKHORN** — Sinkhorn iteration count (fixed; no convergence check)
- **RINGBUF_LEN** — Camera frame ring buffer length

## Operator table

| Operator | Role | Fixed budgets / iters | CertBundle / approximation triggers | Frobenius |
|----------|------|------------------------|-------------------------------------|-----------|
| soft_time_association | Camera frame selection at scan time | RINGBUF_LEN; no convergence | Optional triggers (time association) | If triggers ⇒ applied |
| visual_feature_extract_fixed_budget | Features from image | N_FEAT | — | — |
| lidar_depth_evidence_unified | (Λ_ℓ, θ_ℓ) per pixel; Route A+B combined | Per (u,v); no use_route_b param | — | — |
| splat_prep_fused | Fused 3D splats from camera + LiDAR depth | N_FEAT (camera splats) | — | — |
| lidar_surfels_fixed_budget | LiDAR → surfels (voxel + plane fit) | N_SURFEL | Surfel extraction | If triggers ⇒ applied |
| associate_primitives_ot | MeasurementBatch × PrimitiveMapView → π; candidate gen = MA hex web (hex cells + stencil) | K_ASSOC; K_SINKHORN iters; π shape [N_meas, K_ASSOC] | OT (fixed iters) | If triggers ⇒ applied |
| primitive_alignment_pose_evidence | Expected NLL → Laplace at z_lin → (L, h) | — | linearization, mixture_reduction | **frobenius_applied = True** when triggers nonempty |
| PrimitiveMapFuse | Fuse associated measurements into map primitives | Every scan; fixed-cost | Optional (PoE/Wishart) | If triggers ⇒ applied |
| PrimitiveMapInsert | Insert new primitives | Every scan; fixed-cost | — | — |
| PrimitiveMapCull | Cull low-weight primitives | Every scan; fixed-cost | Mass drop logged as approximation | If triggers ⇒ applied |
| PrimitiveMapMergeReduce | Mixture reduction | Every scan; fixed-cost | mixture_reduction | **frobenius_applied = True** when out-of-family |
| PrimitiveMapForget | Continuous forgetting factor | Every scan; fixed-cost | — | — |
| render_from_primitive_map | PrimitiveMapView + pose → image | Tile binning cap; input = PrimitiveMapView only | Lobe reduction, clipping (if any) logged | — |

## Invariants

1. **Single pose-evidence path:** When `pose_evidence_backend="primitives"`, LiDAR pose evidence comes only from `primitive_alignment_pose_evidence` (no bin MF/planar path).
2. **CertBundle rule:** For every operator, `approximation_triggers ≠ ∅ ⇒ frobenius_applied`.
3. **Fixed-cost:** All loops use fixed iteration counts (e.g. K_SINKHORN); no data-dependent convergence gates.
4. **Map canonical:** Canonical map is PrimitiveMap; bins are derived/legacy only (see pipeline Step 13 comment).

## Code anchors

- Pose evidence (primitives): `fl_slam_poc.backend.operators.visual_pose_evidence` → `visual_pose_evidence`, `build_visual_pose_evidence_22d`
- OT association: `fl_slam_poc.backend.operators.primitive_association` → `associate_primitives_ot`
- Map maintenance: `fl_slam_poc.backend.structures.primitive_map` → `primitive_map_fuse`, `primitive_map_insert`, `primitive_map_cull`, `primitive_map_merge_reduce`, `primitive_map_forget`
- Surfel extraction: `fl_slam_poc.backend.operators.lidar_surfel_extraction` → `extract_lidar_surfels`
