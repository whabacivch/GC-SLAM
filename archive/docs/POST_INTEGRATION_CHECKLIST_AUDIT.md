# Post-Integration Checklist Audit

Blunt audit against the post-integration checklist. Code is the source of truth; this doc records pass/fail and evidence.

**Audit date:** 2026-01-30 (after Visual LiDAR plan execution)

---

## 0) Single math path at runtime

### ✅ Verify

| Check | Status | Evidence |
|-------|--------|----------|
| Single explicit config: `pose_evidence_backend`, `map_backend` | **PASS** | `PipelineConfig` and `RuntimeManifest` have `pose_evidence_backend`, `map_backend` (constants). No `association_backend` param; OT is the only association and is implicit when `map_backend=primitive_map`. |
| Runtime manifest prints/publishes resolved values | **PASS** | `RuntimeManifest.to_dict()` includes `pose_evidence_backend`, `map_backend`; `backend_node._publish_runtime_manifest()` passes config values. |
| Tests assert the manifest | **PASS** | `test_visual_lidar_plan.py`: manifest has `pose_evidence_backend` and `map_backend`; primitives path sets `backends["lidar_evidence"]` to visual_pose_evidence. |

### 🚩 Common mistake: both paths run

**Finding:** Pipeline does **not** import or call bin evidence. `pipeline.py` imports: point_budget, predict, deskew, measurement_noise_iw, odom_evidence, odom_twist, imu_evidence, **planar_prior** (planar_z_prior, velocity_z_prior — robot z prior, not bin planar translation), imu_gyro, imu_preintegration_factor, fusion, excitation, recompose, map_update (MapUpdateResult only; pos_cov_inflation_pushforward is imported but **never called**), anchor_drift, hypothesis, primitive_map, measurement_batch, lidar_surfel_extraction, primitive_association, visual_pose_evidence. It does **not** import: bin_soft_assign, scan_bin_moment_match, matrix_fisher_evidence, planar_translation_evidence. So when the pipeline runs, only the primitive path (surfel → OT → visual_pose_evidence) can contribute LiDAR pose evidence. **PASS** (single path).

---

## 1) Legacy deletion audit

### 1.1 Bin-based LiDAR evidence removed / not reachable

| Check | Status | Evidence |
|-------|--------|----------|
| Steps 4–8 (bin assign → moments → MF → planar) deleted or not importable from runtime | **PASS** | Pipeline never imports or calls `bin_soft_assign`, `scan_bin_moment_match`, `matrix_fisher_*`, `planar_translation_*`. Those live in `operators/binning.py`, `operators/matrix_fisher_evidence.py` but are not on the pipeline’s import chain. |
| No bin map objects constructed when primitives selected | **PASS** | When `map_backend=primitive_map`, backend_node creates PrimitiveMap only. Bin atlas/map_stats still exist for dummy MapUpdateResult aggregation (documented as derived/legacy). |
| No MF/planar evidence added to L/h | **PASS** | L_lidar comes only from `build_visual_pose_evidence_22d(visual_result)` or zero. |

**Caveat:** `pos_cov_inflation_pushforward` is still **imported** in `pipeline.py` (line 73) but never called; MapUpdateResult is built with zeros. Recommend removing the dead import.

### 1.2 No duplicate implementations per operator

| Check | Status | Evidence |
|-------|--------|----------|
| Single `associate_primitives_*` reachable | **PASS** | Only `associate_primitives_ot` in `primitive_association.py`; pipeline imports it. |
| Single `lidar_depth_evidence` in runtime namespace | **PASS** | One function `lidar_depth_evidence` in `lidar_camera_depth_fusion.py`; used by splat_prep. |
| Single Sinkhorn; balanced/unbalanced explicit | **PASS** | `sinkhorn_ot.sinkhorn_balanced_fixed_k` and `sinkhorn_unbalanced_fixed_k`; primitive_association uses one based on config (balanced vs unbalanced). |

---

## 2) No multi-path in visual depth fusion

### 2.1 Unified depth evidence (Route A+B)

| Check | Status | Evidence |
|-------|--------|----------|
| No `use_route_b` (or equivalent) in runtime signature | **PASS** | Grep: no `use_route_b` in lidar depth code. `lidar_depth_evidence(...)` has no route selector. |
| Single output `(Λ_ell, θ_ell)`; Route A and B only via continuous weights | **PASS** | `lidar_camera_depth_fusion.py`: Route A and B combined; docstring "Λℓ = ΛA + ΛB, θℓ = θA + θB". No branch that switches model. |

---

## 3) No gating audit

### 3.1 OT fusion weights continuous

| Check | Status | Evidence |
|-------|--------|----------|
| No thresholds like `if mass > 0.2`, `if w_ij > ...`, `if reproj_error < ... then accept` | **PARTIAL** | **Exception:** `flatten_associations_for_fuse` uses `responsibility_threshold=0.01` and `mask = resp_np > responsibility_threshold` to select which (i,k) pairs are passed to `primitive_map_fuse`. So we **do** have a hard threshold for “include in fuse” (not for evidence weight). Checklist says such effects should be precision scaling / smooth sigmoids; alternatively document as fixed-cost support selection and log as approximation. |

**Recommendation:** Either (a) replace with continuous weighting (e.g. pass all pairs with weights = responsibilities), or (b) keep as declared “sparse fuse” operator with `responsibility_threshold` in config and log in CertBundle/ExpectedEffect.

---

## 4) Evidence correctness

### 4.1 LiDAR not double-used for pose evidence

| Check | Status | Evidence |
|-------|--------|----------|
| When `pose_evidence_backend="primitives"`, LiDAR only via surfels + lidar_depth_evidence + primitive alignment NLL | **PASS** | Pipeline: L_lidar only from `build_visual_pose_evidence_22d(visual_result)` when assoc_result/map_view/measurement_batch exist; else zero. No separate bin L_lidar term. |
| Bin-based term absent from evidence sum | **PASS** | Evidence sum is L_evidence = L_lidar + odom + imu + gyro + preint + planar_prior + twist; L_lidar is only visual primitive. |

### 4.2 Collapse to camera-only (Λ_ell, θ_ell → 0 when no LiDAR support)

| Check | Status | Evidence |
|-------|--------|----------|
| If LiDAR has no support at pixel, (Λ_ell, θ_ell) → 0; fused = camera-only | **BY CONSTRUCTION** | Docstring in `lidar_camera_depth_fusion.py`: "Λℓ, θℓ → 0 when not applicable." Route A/B use continuous weights; no clamp that injects precision. (No dedicated unit test found; add if required.) |

---

## 5) JAX performance audit

### 5.1 Few jitted kernels vs many small jits

| Check | Status | Evidence |
|-------|--------|----------|
| Per-scan compute in a small number of jitted kernels | **PARTIAL** | `visual_pose_evidence` uses `@jax.jit` on `_compute_translation_evidence_wls` and `_compute_rotation_evidence_vmf`. Sinkhorn in `sinkhorn_ot` is **NumPy** (for i in range(N), for j in range(M) for cost; for _ in range(K) for iterations) — not JAX-jitted. So OT hot path is CPU NumPy, not one big JIT. |

### 5.2 Static shapes

| Check | Status | Evidence |
|-------|--------|----------|
| MeasurementBatch fixed sizes; OT cost [N, K]; no Python list in hot path | **PASS** | MeasurementBatch has fixed N_surfel; primitive_association uses fixed K_ASSOC, K_SINKHORN. Shapes are fixed. |

### 5.3 Fixed iteration counts only

| Check | Status | Evidence |
|-------|--------|----------|
| Sinkhorn fixed `for _ in range(K)`; no while/break/converge | **PASS** | `sinkhorn_balanced_fixed_k` and `sinkhorn_unbalanced_fixed_k`: "for _ in range(K)". No convergence check. |

### 5.4 No host–device thrashing in hot path

| Check | Status | Evidence |
|-------|--------|----------|
| No `.numpy()` / unnecessary `np.array(jax_array)` in pipeline hot path | **PARTIAL** | **Pipeline:** `np.array` used in diagnostics block (when save_full_diagnostics) and in one invariant `print` (first 10 scans) — diagnostics/print are off hot path by intent. **flatten_associations_for_fuse:** does `resp_np = np.array(result.responsibilities)` and `cand_np = np.array(result.candidate_indices)` and uses them for mask/flat indices — this is in the per-scan fuse path, so **host round-trip each scan**. |

**Recommendation:** Move flatten to JAX (e.g. vectorized gather with valid mask) to avoid sync in hot path, or accept as known cost and document.

---

## 6) Certificates + Frobenius compliance

### 6.1 Every operator returns (result, CertBundle, ExpectedEffect)

| Check | Status | Evidence |
|-------|--------|----------|
| VisualPoseEvidence returns cert; triggers include "linearization" when used; frobenius_applied when triggers nonempty | **PASS** | `visual_pose_evidence` returns CertBundle with `approximation_triggers` and `frobenius_applied=True`. Tests in `test_visual_lidar_plan.py`: cert policy (triggers ≠ ∅ ⇒ frobenius_applied). |

### 6.2 SPD projection logged as DomainProjection

| Check | Status | Evidence |
|-------|--------|----------|
| Eigenvalue clamp / symmetrize logged as domain projection, not "exact" | **ASSUMED** | Common primitives use `domain_projection_psd_core`; operators that project log conditioning/approximation. (Not re-audited file-by-file here.) |

---

## 7) Integration wiring

### 7.1 Camera extrinsics/intrinsics

| Check | Status | Evidence |
|-------|--------|----------|
| Extrinsics parameters; frame conventions per FRAME_AND_QUATERNION_CONVENTIONS | **DOCUMENTED** | Config/params for T_base_camera, intrinsics; fail-fast when primitives selected and missing. (Not re-verified here.) |

### 7.2 Time association soft and fixed-cost

| Check | Status | Evidence |
|-------|--------|----------|
| No "nearest frame" gate; ring buffer fixed size; selection soft | **BY DESIGN** | Plan and config: RINGBUF_LEN; soft time association. (Implementation not re-audited here.) |

---

## 8) Rendering

### 8.1 Renderer consumes PrimitiveMapView only

| Check | Status | Evidence |
|-------|--------|----------|
| Renderer takes primitives from canonical map | **PASS** | Plan: rendering from PrimitiveMapView; map_publisher uses `extract_primitive_map_view`. No bin atlas used for rendering. |

### 8.2 Rendering not in inference hot path

| Check | Status | Evidence |
|-------|--------|---------- |
| Rendering does not block/sync JAX in pipeline | **PASS** | Map publisher runs after state publish; rendering is separate from process_scan_single_hypothesis. |

---

## 9) Backwards-compatibility creep phrases

| Phrase | Status |
|--------|--------|
| "Kept old path for safety" | Not found |
| "Fallback if primitives fail" | Not found |
| "Debug mode uses bins" | Not found |
| "We can switch between them later" | Not found |
| "Auto-select GPU/CPU" | Not found |
| "Try JAX else NumPy" | Not found |

**PASS** (grep found no matches).

---

# Minimal Pass/Fail (checklist criteria)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Bin-based pose evidence is **gone** | **PASS** — Not imported or called in pipeline. |
| 2 | `lidar_depth_evidence` is **unified** (no route flags) | **PASS** — Single function; Route A+B internal; no use_route_b. |
| 3 | OT is the only association (no NN fallback) | **PASS** — Only `associate_primitives_ot`. |
| 4 | MeasurementBatch shapes fixed and JAX-jittable; no lists in hot path | **PASS** — Fixed budgets; flatten uses arrays (but has numpy in path). |
| 5 | No device sync / numpy conversion in hot path | **PARTIAL** — flatten_associations_for_fuse uses np.array(responsibilities/candidate_indices) per scan. |
| 6 | All operators return certificates; Frobenius on triggers | **PASS** — Visual pose evidence and tests. |
| 7 | Runtime manifest proves selected backends and single path | **PASS** — pose_evidence_backend, map_backend in manifest; tests. |

**Overall:** **PASS** with two **PARTIAL** items: (3) responsibility threshold in flatten_associations_for_fuse, (5) numpy in flatten_associations_for_fuse hot path. Recommend addressing both in a follow-up (continuous weighting or documented sparse-fuse + JAX-only flatten).

---

# Recommended follow-ups

1. **Dead import:** Remove `pos_cov_inflation_pushforward` from pipeline imports (keep `MapUpdateResult`).
2. **responsibility_threshold:** Either make fuse input continuous (all pairs, weights = responsibilities) or document as fixed-cost “sparse fuse” with approximation logged.
3. **flatten_associations_for_fuse:** Implement flatten in JAX (or document host round-trip) to satisfy “no numpy in hot path” fully.
4. **Sinkhorn:** Consider JAX implementation (e.g. fori_loop) for cost and iterations to align with “one big JIT” and avoid Python loops over N,M.
