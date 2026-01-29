# Pipeline Compute Bottlenecks and JAX Usage

This document summarizes **compute bottlenecks**, **JAX usage** (JIT, vmap, host round-trips), **caching**, and **Python loops / calls** in the Golden Child SLAM pipeline. It is based on tracing the code; the code is the source of truth.

---

## 1. Executive Summary

| Category | Finding |
|----------|--------|
| **JIT coverage** | Many low-level ops are JIT'd (se3_jax, imu_preintegration, imu_evidence, kappa, map_update, recompose, IW/measurement noise). **Pipeline-step-level** operators that run every scan are mostly **not** JIT'd: `point_budget_resample`, `predict_diffusion`, `bin_soft_assign`, `scan_bin_moment_match`, `matrix_fisher_rotation_evidence`, `planar_translation_evidence`, `odom_quadratic_evidence`, `fusion_scale_from_certificates`, `info_fusion_additive`. |
| **Host round-trips** | **Heavy**: When `config.enable_timing` is True, `_record_timing` calls `block_until_ready()` after **every** pipeline step → ~14 device→host syncs per scan. **Heavy**: End-of-pipeline diagnostics pull large arrays to host (`np.array(L_evidence)`, `np.array(h_evidence)`, scan_bins, map_stats, etc.) every scan. **Moderate**: Conditioning for fusion scale pulls `L_evidence[0:6,0:6]` to host and runs NumPy `eigvalsh`/`svd` every scan. **Moderate**: Many `float(jnp.sum(...))` and `np.array(...)` in point_budget, bin_atlas, matrix_fisher, etc. |
| **Python loops** | **Critical**: `backend_node.parse_pointcloud2_vectorized` uses a listcomp `for i in range(n_points)` (up to **8192**) to read `time_offset` from the PointCloud2 buffer. **Critical**: `bin_atlas.compute_map_derived_stats` uses `for b in range(n_bins)` (**48** iterations) with host pulls and 48 separate JIT calls to `kappa_from_resultant_v2`. **Moderate**: `backend_node.on_lidar` builds IMU arrays with `for i, (t, g, a) in enumerate(window)` (up to **512**). **Moderate**: `backend_node` accumulates map deltas with `for i, result in enumerate(results)` (**K_HYP=4**) and `float(self.hyp_weights[i])` each time. |
| **Caching** | JAX JIT caches compiled functions by (approximate) input shapes. Inner JIT'd functions (e.g. `kappa_from_resultant_v2`, `_matrix_fisher_core`) are cached. There is **no** single JIT'd “pipeline” or “scan step”; each scan runs through Python, dispatching to many small JIT'd or non-JIT'd functions, so **no end-to-end compilation**. |
| **NumPy in hot path** | `compute_imu_integration_time` (pipeline.py) is **pure NumPy**: converts `imu_stamps` to numpy, filters, sorts, sums. Called every scan. |

---

## 2. What Is JIT'd vs Not

### 2.1 JIT'd (used in pipeline or by pipeline operators)

| Module / function | Notes |
|-------------------|--------|
| `common/geometry/se3_jax` | `so3_exp`, `so3_log`, `se3_*`, etc. — heavily JIT'd. |
| `common/belief` | `to_moments`, `mean_world_pose`, `mean_increment`, etc. — JIT'd. |
| `operators/imu_preintegration` | `smooth_window_weights`, `preintegrate_imu_relative_pose_jax` — JIT'd. |
| `operators/imu_evidence` | Multiple JIT'd helpers for gyro/accel evidence. |
| `operators/deskew_constant_twist` | Inner `one_point` JIT'd; `deskew_constant_twist` uses `jax.vmap(one_point)` (not JIT'd at top level). |
| `operators/matrix_fisher_evidence` | `_matrix_fisher_core`, `compute_scatter_metrics` JIT'd; `matrix_fisher_rotation_evidence` **not** JIT'd (calls core + does `float(jnp.sum(...))` etc.). |
| `operators/map_update` | One JIT'd function; uses `jax.vmap` for PSD project. |
| `operators/binning` | No top-level JIT on `bin_soft_assign` or `scan_bin_moment_match`; internal `jax.vmap(proj_one)` in scan_bin_moment_match. |
| `operators/kappa` | `kappa_from_resultant_v2` JIT'd (and called **48 times** from Python in `compute_map_derived_stats`). |
| `operators/recompose` | `pose_update_frobenius_recompose` inner JIT'd. |
| `operators/inverse_wishart_jax`, `measurement_noise_iw_jax`, `lidar_bucket_noise_iw_jax` | Multiple JIT'd helpers. |
| `operators/hypothesis` | JIT'd solve; `combine_hypotheses_*` builds stacks from Python list of beliefs. |
| `operators/excitation` | JIT'd. |

### 2.2 Not JIT'd (called every scan)

| Function | File | Impact |
|-----------|------|--------|
| `point_budget_resample` | point_budget.py | Uses `float(jnp.sum(weights))`, `int(jnp.maximum(...))`; core indexing is pure JAX but whole function is Python. |
| `predict_diffusion` | predict.py | Pure JAX math but no @jax.jit. |
| `bin_soft_assign` | binning.py | Dot products and softmax; no JIT. |
| `scan_bin_moment_match` | binning.py | Batched einsum/vmap; no top-level JIT. |
| `deskew_constant_twist` | deskew_constant_twist.py | Wrapper; inner one_point JIT'd, vmap(one_point) not JIT'd. |
| `matrix_fisher_rotation_evidence` | matrix_fisher_evidence.py | Calls JIT'd _matrix_fisher_core; wrapper does float()/cert building. |
| `planar_translation_evidence` | matrix_fisher_evidence.py | No JIT. |
| `odom_quadratic_evidence` | odom_evidence.py | No JIT (uses se3_jax JIT'd primitives). |
| `fusion_scale_from_certificates` | fusion.py | Uses CertBundle (Python objects); conditioning block uses NumPy. |
| `info_fusion_additive` | fusion.py | No JIT. |
| `compute_imu_integration_time` | pipeline.py | **Pure NumPy** (np.asarray, mask, sort, sum). |
| `process_scan_single_hypothesis` | pipeline.py | **Entire pipeline** is Python; no JIT. |

So we are **not** leveraging JAX end-to-end: the hot path is Python dispatching to a mix of JIT'd and non-JIT'd functions, with multiple host round-trips.

### 2.3 Addressed (2026-01-29): JIT step-level cores

The following now have **JIT'd cores** with Python wrappers that build certs from returned arrays (single host pull after JIT return):

| Operator | File | Status |
|----------|------|--------|
| `compute_map_derived_stats` | bin_atlas.py | JIT batched; no 48-iteration Python loop. |
| Conditioning (pose6) | pipeline.py | JAX `eigvalsh` on device; pull only scalars for cert. |
| Map delta accumulation | backend_node.py | Batched `einsum` weighted sum; no Python loop. |
| `predict_diffusion` | predict.py | `_predict_diffusion_core` JIT'd; wrapper builds BeliefGaussianInfo/cert. |
| `scan_bin_moment_match` | binning.py | `_scan_bin_moment_match_core` JIT'd (static_argnames n_points, n_bins). |
| `bin_soft_assign` | binning.py | `_bin_soft_assign_core` JIT'd. |
| `odom_quadratic_evidence` | odom_evidence.py | `_odom_quadratic_evidence_core` JIT'd. |
| `point_budget_resample` | point_budget.py | `_point_budget_resample_core` JIT'd; returns **fixed-size** (N_POINTS_CAP) arrays; pipeline passes them to binning so downstream JITs see static n_points (no recompile per scan). |

---

## 3. Host Round-Trips and Syncs

### 3.1 Timing: `block_until_ready` every step

**Location:** `pipeline.py` L338–349, `_record_timing`.

When `config.enable_timing` is True (e.g. for profiling or diagnostics), after **each** of these steps we call:

```python
jax.tree_util.tree_map(
    lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
    out,
)
```

So we **synchronize device → host** after: point_budget, imu_preint_scan, imu_preint_int, deskew, bin_assign, bin_moment, matrix_fisher, planar_translation, lidar_bucket_iw, map_update. That is **10+ syncs per scan**, which serializes the pipeline and can dominate runtime when timing is on.

**Recommendation:** When timing is not required for production, run with `enable_timing=False`. If per-step timing is needed, consider a single sync at the end of the pipeline and use JAX’s async dispatch; or record timestamps on device and read once.

### 3.2 Conditioning for fusion scale — **ADDRESSED**

**Was:** `pipeline.py`: pull pose block to host, NumPy `eigvalsh`/svd every scan.

**Now:** Conditioning is computed in JAX on device (`jnp.linalg.eigvalsh(L_pose)`); only scalars (eig_min, eig_max, cond, near_null_count) are pulled once for the cert. No NumPy in hot path.

### 3.3 Diagnostics — **ADDRESSED (minimal tape)**

**Was:** Full `ScanDiagnostics` built every scan (L_total, h_total, R_hat, rotation errors, IMU stats, etc.) → many host pulls per scan.

**Now:** Default is **minimal tape** only: `PipelineConfig.save_full_diagnostics=False`. Hot path appends a `MinimalScanTape` (scan_number, timestamp, dt_sec, n_points_raw/budget, fusion_alpha, cond_pose6, conditioning_number, eigmin_pose6, L_pose6 6×6, total_trigger_magnitude, timing scalars). Full `ScanDiagnostics` is built only when `save_full_diagnostics=True`. At shutdown, `save_npz` writes either minimal tape format or full format. See **§6 Async publishing and diagnostics** below.

### 3.3.1 Cost of `save_full_diagnostics=True`

When `PipelineConfig.save_full_diagnostics=True`, the **per-scan** hot path (not at shutdown) builds full `ScanDiagnostics`. That path:

- Pulls many arrays to host: `np.array(pose_final)`, `np.array(L_evidence)`, `np.array(h_evidence)`, `np.array(R_hat)`, IMU arrays, rotation matrices, etc.
- Runs NumPy linear algebra (eigvalsh, SVD, rotation error, logdet, trace) and Python loops for block checks.

So **full diagnostics are expensive every scan** and are not suitable for real-time or high-rate runs. Use `save_full_diagnostics=True` only for short debugging or offline analysis. For crash-tolerant, low-overhead logging, keep the default minimal tape; full NPZ can be materialized at shutdown from tape if needed (not currently implemented).

### 3.4 Other host pulls in operators

- **point_budget.py:** `float(jnp.sum(weights))`, `int(jnp.maximum(...))`, `float(jnp.minimum(...))` — several scalars per call.
- **bin_atlas.compute_map_derived_stats:** Inside the loop: `float(N_d)`, `float(jnp.linalg.norm(S_b))`, then `kappa_from_resultant_v2(Rbar)` (Rbar is Python float) — **48 iterations** of host pulls + JIT call.
- **matrix_fisher_evidence:** `float(jnp.sum(scan_N))`, `float(jnp.sum(map_N_dir))`, `float(delta_rot @ L_rot_psd @ delta_rot)` — scalars for certs/diagnostics.
- **scan_bin_moment_match:** `float(jnp.sum(N))` for ESS — one scalar.

---

## 4. Python Loops (Detailed)

### 4.1 Critical: Parse PointCloud2 `time_offset` — O(N) Python loop

**Location:** `backend_node.py` L687–689.

```python
offsets_raw = np.array([
    struct.unpack_from('<I', msg.data, i * msg.point_step + off)[0]
    for i in range(n_points)
], dtype=np.uint32)
```

- **N** = number of points, up to **8192** (N_POINTS_CAP).
- **Effect:** 8192 Python iterations and 8192 `struct.unpack_from` calls per scan when `time_offset` is present.
- **Fix:** Use a single numpy view or structured array (e.g. `np.frombuffer` with the right dtype and offset for the `time_offset` field) so that all values are read in one vectorized operation. The same pattern as for `x,y,z` etc. in the rest of the parser can be applied.

### 4.2 Critical: `compute_map_derived_stats` — O(B_BINS) Python loop

**Location:** `fl_ws/.../backend/structures/bin_atlas.py` L181–211.

```python
for b in range(n_bins):  # n_bins = 48
    S_b = map_stats.S_dir[b]
    N_d = map_stats.N_dir[b]
    # ...
    S_norm = float(jnp.linalg.norm(S_b))
    Rbar = S_norm * inv_N_d
    kappa_result, _, _ = kappa_from_resultant_v2(Rbar)  # JIT'd but called 48× from Python
    kappa = kappa.at[b].set(kappa_result.kappa)
    # ... centroid, Sigma_c per bin
```

- **Effect:** 48 iterations, each: slice from JAX array, pull norm to host, call `kappa_from_resultant_v2` (JIT'd but dispatched 48 times), then `at[b].set`. No single batched JIT over all bins.
- **Fix:** Express the whole computation as a single function of `(map_stats.S_dir, map_stats.N_dir, ...)` and use `jax.vmap(kappa_from_resultant_v2)` (or a batched kappa) over bins, then one JIT'd function returning `(mu_dir, kappa, centroid, Sigma_c)` for all bins. This matches the pattern already used in `scan_bin_moment_match` (batched kappa via `kappa_from_resultant_batch`).

### 4.3 Moderate: IMU window fill — O(M) Python loop

**Location:** `backend_node.py` L765–779.

```python
window = [(t, g, a) for (t, g, a) in self.imu_buffer if t_min - eps_t <= t <= t_max + eps_t]
# ...
for i, (t, g, a) in enumerate(window):
    imu_stamps[i] = float(t)
    imu_gyro[i, :] = np.array(g)
    imu_accel[i, :] = np.array(a)
```

- **M** = length of `window`, capped at **GC_MAX_IMU_PREINT_LEN = 512**.
- **Effect:** Up to 512 Python iterations and list comprehensions per scan. Then we copy into numpy and convert to JAX. Acceptable for 512, but still pure Python in the hot path.
- **Optional:** Pre-store IMU data in a ring buffer (numpy or JAX) and slice in one go; or keep as-is if this is negligible relative to the rest.

### 4.4 Moderate: Map delta accumulation — K_HYP Python loop + host weight

**Location:** `backend_node.py` L922–928.

```python
for i, result in enumerate(results):  # K_HYP = 4
    w_h = float(self.hyp_weights[i])  # host pull every iteration
    delta_S_dir = delta_S_dir + w_h * result.map_increments.delta_S_dir
    # ...
```

- **Effect:** 4 iterations; each pulls `hyp_weights[i]` to host and then does JAX ops. Could be one batched op: stack `map_increments`, multiply by `hyp_weights` (vector), sum.
- **Fix:** `weights = self.hyp_weights` (JAX), `delta_S_dir = jnp.einsum('i,i...->...', weights, jnp.stack([r.map_increments.delta_S_dir for r in results]))` (or equivalent) so there is no Python loop over hypotheses and no per-iteration host pull.

### 4.5 Other loops

- **backend_node L374:** `for i in range(K_HYP)` at **init** only — not per-scan.
- **pipeline.py L813–815:** `while angle > 180: angle -= 360` (normalize angle for diagnostics) — bounded, small cost.
- **diagnostics.py:** Loops over `self.scans` for NPZ export — not in the scan hot path.
- **inverse_wishart_jax structures L74:** `for i in range(7)` — fixed 7; likely init or small helper.

---

## 5. Caching and Compilation

- **JAX JIT:** Compilations are cached by (approximate) input shapes and dtypes. So:
  - First call to e.g. `kappa_from_resultant_v2` with a given shape compiles; later calls with same shape hit the cache.
  - Calling from Python in a loop (e.g. 48× in `compute_map_derived_stats`) still does 48 dispatches; the **compilation** is cached but the **dispatch and return** are repeated.
- **No pipeline-level JIT:** `process_scan_single_hypothesis` is a long Python function that calls many operators. There is no `@jax.jit` on it and no single “pipeline” trace. So every scan:
  - Executes Python control flow.
  - Dispatches to each operator (some JIT'd, some not).
  - For non-JIT'd operators, every call goes through Python and may trigger tracing if shapes change.
- **Dataclasses / CertBundle:** Many operators return `(result, CertBundle, ExpectedEffect)`. CertBundle is a Python dataclass with floats and lists, so it cannot be part of a JIT'd function signature. That limits how much of the pipeline can be fused into one JIT without refactoring certs to arrays.

---

## 6. Recommendations (Prioritized)

1. **Turn off timing in production**  
   Use `enable_timing=False` so `block_until_ready()` is not called after every step. If you need timings, consider one sync at the end or device-side timing.

2. **Remove O(N) Python loop in PointCloud2 parsing**  
   Replace the listcomp over `n_points` for `time_offset` with a single vectorized read (e.g. structured buffer or `np.frombuffer` with correct stride/offset). This is a direct bottleneck for large scans.

3. **Vmap/batch `compute_map_derived_stats`**  
   Replace the 48-iteration Python loop with a single JIT'd function that uses `jax.vmap` (or the existing `kappa_from_resultant_batch`) over bins, so one dispatch per scan instead of 48.

4. **Keep conditioning on device**  
   Replace NumPy `eigvalsh`/`svd` on `L_evidence[0:6,0:6]` with `jnp.linalg.eigvalsh` and only pull the final scalar(s) for logging if needed.

5. **JIT pipeline-step operators where possible**  
   Add `@jax.jit` to pure-array operators that currently have none: e.g. `point_budget_resample`, `predict_diffusion`, `bin_soft_assign`, `scan_bin_moment_match`, `odom_quadratic_evidence`, and the “core” part of `matrix_fisher_rotation_evidence` (already has JIT'd core; ensure no Python/float in the traced path). For those that return CertBundle, you can JIT the **numeric** part and build certs in Python from the returned arrays.

6. **Batch map delta accumulation**  
   Replace the K_HYP loop in `backend_node` with a single weighted sum over stacked map increments using `hyp_weights` as a JAX vector (no `float(self.hyp_weights[i])` in a loop).

7. **Diagnostics**  
   If diagnostics are not needed every scan, sample every N scans or move array gathering to a separate thread/async path so the hot path does not pay for large host copies every time.

8. **Optional: Fused pipeline JIT**  
   For maximum throughput, consider a “core pipeline” that takes only arrays and returns arrays (no CertBundle in the traced path), JIT that, and build certs/diagnostics in Python from the outputs. That would require refactoring the current step-by-step Python orchestration.

---

## 6. Async publishing and diagnostics (addressed 2026-01-29)

### 6.1 Deferred publish

**Design:** Publish (state, TF, path) is **deferred** to the start of the **next** LiDAR callback. The hot path (pipeline + map update) does not call `publish()` or `sendTransform()`; it only sets `_pending_publish = (pose_6d, stamp_sec)`. When the next scan arrives, we drain: call `_publish_state_from_pose(pending_pose_6d, pending_stamp_sec)`, then run the pipeline for the current scan. On shutdown, we drain any pending publish so the last scan state is written.

**Pros:** Pipeline callback returns sooner; no blocking on ROS publish in the same scan. **Cons:** State/TF/path are one scan behind (published when next scan starts). Ordering is preserved (main thread only).

### 6.2 Diagnostics tape (minimal per-scan)

**Design:** Default is **minimal tape** only (`PipelineConfig.save_full_diagnostics=False`). Hot path appends a `MinimalScanTape` (scalars + L_pose6 6×6 + timing). Full `ScanDiagnostics` is built only when `save_full_diagnostics=True`. At shutdown, `DiagnosticsLog.save_npz()` writes either minimal-tape format or full format. **Never drop** state/TF; diagnostics can be minimal or full by config.

### 6.3 Compilation and recompiles

JIT'd cores use `static_argnames` where shape varies (e.g. `n_points`, `n_bins` in binning). Changing those across scans triggers recompile. **Addressed:** Point budget now returns fixed-size arrays (N_POINTS_CAP); the pipeline passes them to deskew → bin_soft_assign → scan_bin_moment_match, so `n_points` is always N_POINTS_CAP and binning JITs do not recompile per scan. For stable throughput, keep bin count fixed (B_BINS). A future dashboard could expose `jit_compile_count` or log "compiled!" once per JIT to make recompiles visible (ms_compile vs ms_execute).

---

## 7. Summary Table

| Issue | Location | Severity | Status |
|-------|----------|----------|--------|
| `block_until_ready` every step when timing on | pipeline.py _record_timing | High | Disable timing in prod; or single sync / device timing |
| O(N) Python loop parsing time_offset | backend_node L687–689 | Critical | Vectorized read (numpy view / frombuffer) |
| O(48) Python loop + 48× JIT in compute_map_derived_stats | bin_atlas.py | Critical | **Addressed:** JIT batched |
| NumPy conditioning (L_pose 6×6) every scan | pipeline.py | Moderate | **Addressed:** JAX eigvalsh on device |
| Diagnostics: many np.array per scan | pipeline.py | Moderate | **Addressed:** minimal tape default |
| O(512) Python loop filling IMU arrays | backend_node L776–779 | Moderate | Optional: pre-filled buffer / slice |
| K_HYP loop + float(hyp_weights[i]) for map deltas | backend_node | Moderate | **Addressed:** batched einsum |
| Many pipeline steps not JIT'd | point_budget, predict, binning, odom_evidence | Moderate | **Addressed:** JIT cores + wrapper |
| Dynamic n_points into binning JITs | point_budget slice → pipeline → binning | Moderate | **Addressed:** point_budget returns fixed size; downstream static n_points |
| save_full_diagnostics cost | pipeline.py when save_full_diagnostics=True | Moderate | **Documented:** §3.3.1; heavy per-scan host pulls; use for short debug only |
| compute_imu_integration_time pure NumPy | pipeline.py L244–270 | Low | Optional: JAX version if keeping all on device |
| Publish blocking hot path | backend_node | Moderate | **Addressed:** deferred publish to next callback |

This document should be updated when the pipeline or operators change; the code remains the source of truth.
