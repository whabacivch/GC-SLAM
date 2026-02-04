# Codebase Simplification Opportunities

**Purpose:** Reduce code volume to improve auditability and maintainability while preserving intent per `docs/GC_SLAM.md`.

**Scope:** Targeted search across `fl_ws/src/fl_slam_poc` and related common/certificate code. Each item states: **what** to simplify, **where**, **how**, and **spec alignment**.

---

## 1. Pipeline: certificate collection (low–medium impact)

**Where:** `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`

**What:** 16+ `all_certs.append(...)` calls; each step does `result, cert, effect = op(...); all_certs.append(cert)`.

**How:** Option A: keep current pattern (minimal change). Option B: refactor steps that are independent into a list of “step results” `(result, cert, effect)` and then `all_certs = [c for _, c, _ in step_results]`. Option B is a larger refactor and may not pay off unless combined with a more structured pipeline (e.g. list of step functions). Recommendation: defer unless doing a broader pipeline restructure.

**Spec:** §8 order unchanged; only how certs are collected is internal.

---

## 2. Unused or redundant code (targeted cleanup)

**Where:** Previously identified (e.g. Vulture): unused parameters in `fusion.py`, `primitive_map.py`; unused import in `tiling.py`.

**What:** Remove dead code.

**How:** Remove unused args/imports; keep one canonical path per operator. Verify with Vulture or prospector; after cleanup, cite resolved file:line or state "verified clean."

**Spec:** Keeps “one canonical path” and “no superlinear materialization.”

---

## Summary table

| # | Area | Impact | Effort | Spec-safe |
|---|------|--------|--------|-----------|
| 1 | Pipeline cert collection | Low–Med | Med (if refactor) | Yes |
| 2 | Dead code cleanup | Correctness + cleanup | Low | Yes |

**Suggested order:** 2 (dead code) first for safety; then 1 if doing a broader pipeline restructure.

---

## 3. Syncs and host/device crossings (performance, §12.5)

**Where:** `pipeline.py`, `backend_node.py`, `primitive_map.py`. (Note: `sinkhorn_ot.py` is **not on the runtime path** — see item 5 and §7; the live association path is `primitive_association.py` with JAX cost + JAX Sinkhorn.)

**What:** Many `float(...)` and `np.array(jnp_thing)` calls force host sync (JAX device → host). Each sync blocks until the device is done and can bottleneck the hot path.

**Findings:**

- **Pipeline (process_scan_single_hypothesis):** Dozens of `float(...)` on JAX scalars/arrays (e.g. `float(dt_asym_raw)`, `float(z_to_xy_raw)`, `float(s_dt)`, `float(cond_pose6)`, `float(jnp.sum(...))`, etc.). Some are needed for Python control (e.g. tempering beta, cert construction); others could be deferred or batched.
- **Pipeline diagnostics block:** Full ScanDiagnostics has been removed in favor of minimal tape + cert summaries. Current code syncs only **L_pose6** (6×6 = 36 floats) for `MinimalScanTape`, plus many `float(...)` for cert fields — not full 22×22 L/h. Keep future diagnostics in JAX; compute scalars (logdet, trace, eigvals, etc.) in JAX and sync one small struct to avoid reintroducing device→host bottlenecks.
- **Backend_node _process_lidar_msg:** After `parse_pointcloud2_vlp16` we have JAX arrays. We then call `record_device_to_host(points, syncs=1)`, `pts_np = np.array(points)`, do NumPy transform `pts_base = (self.R_base_lidar @ pts_np.T).T + ...`, then `points = jnp.array(pts_base, ...)`. So we do JAX→host→NumPy op→host→JAX. The transform could be done in JAX (keep points on device, use `config.lidar_origin_base` / R as JAX) to remove one round-trip.
- **Backend_node:** `time_bounds = jnp.array([jnp.min(timestamps), jnp.max(timestamps)], ...)` then `record_device_to_host(time_bounds, syncs=1)` and `timestamps_min, timestamps_max = float(time_bounds[0]), float(time_bounds[1])` — one small sync per scan (2 floats); acceptable.
- **Primitive_map merge_reduce:** Inside the Python loop `for idx in np.array(order)`, each iteration does `int(i_idx[idx])`, `int(j_idx[idx])`, `float(dist[idx])` — each indexes into JAX arrays and forces sync. With `order` of size M*(M-1)/2 this is a massive number of syncs when M is large (and already forbidden by merge cap). After pair selection, `for i, j in selected_pairs` does `float(weights_new[i])` etc. and JAX `.at[i].set(...)` in a loop — each float() syncs. See item 5.

**How:** (1) Diagnostics: compute logdet, trace, L_dt, trace_L_ex, eigvals in JAX; sync only the resulting scalars (one small array or a struct). (2) LiDAR transform: do base transform in JAX so parsed points stay on device until they enter the pipeline. (3) Merge_reduce: eliminate the Python loop over `order` (see item 5); for the apply loop over `selected_pairs`, batch the updates in JAX (e.g. vmap over pairs, single scatter) so no per-pair float() or sync.

**Spec:** §12.5 host↔device discipline; certificate requirement for host_sync_count_est / device_to_host_bytes_est.

---

## 4. Callbacks and queues (bottlenecks)

**Where:** `backend_node.py`: `on_imu`, `on_odom`, `on_lidar`, `_enqueue_lidar_msg`, `_lidar_worker_loop`, `_process_lidar_msg`, `_drain_publish_queue`, buffer lock.

**What:** Callback design can block the executor or serialize work unnecessarily.

**Findings:**

- **LiDAR:** `on_lidar` only enqueues; a worker thread runs `_process_lidar_msg`. So the ROS callback is short (enqueue + notify). Good. Worker holds no lock during `process_scan_single_hypothesis`; buffers are snapshotted once under `_buffer_lock` at the start of `_process_lidar_msg`. So the long pipeline run does not block IMU/odom callbacks. No change needed.
- **Publish queue:** Drained by a **timer** (`create_timer(_publish_timer_period_sec, _drain_publish_queue)`), so poses are published at a fixed rate (e.g. 10 Hz) independent of scan rate. Good. One pose per timer tick; if the pipeline produces poses faster than the timer, the queue fills (and drops when full). So the only bottleneck is pipeline throughput, not publish drain.
- **Buffer lock:** Snapshot of `imu_buffer`, `visual_feature_ringbuf`, `camera_ringbuf`, and last odom/IMU covariances is taken under one lock. Copying `list(self.imu_buffer)` and ringbufs can be non-trivial if buffers are large; lock is held only for the copy, then released. Acceptable; could be optimized later with copy-on-write or double-buffering if needed.
- **IMU callback:** No JAX ops per message; only NumPy and list append. Good per comment “Keep callback CPU-only (no JAX ops per message).”

**Inefficiency:** Per-point time_offset parsing in `_process_lidar_msg`: when `time_offset` is present, the code does `offsets_raw = np.array([struct.unpack_from('<I', msg.data, i * msg.point_step + off)[0] for i in range(n_points)], dtype=np.uint32)` — a **Python loop over every point** (e.g. 30k–100k iterations). This runs on the worker thread and can add tens of ms per scan.

**How:** Replace with a single vectorized read: e.g. create a view of `msg.data` as `uint8`, then use `np.frombuffer` with dtype `np.uint32` and the correct offset/stride so the entire `time_offset` column is read in one go (or one chunked read). Removes O(n_points) Python loop.

**Spec:** §12.11 bounded buffers and deterministic selection; no spec change.

---

## 5. Massive loops (CPU-bound and JAX sync)

**Where:** `primitive_map.py` (`primitive_map_merge_reduce`), `sinkhorn_ot.py` (cost matrix build — **not on runtime path**, see below), `backend_node.py` (time_offset), `pipeline.py` (insert event_log).

**What:** Large Python loops or loops that force repeated JAX→host sync.

**Findings:**

- **primitive_map_merge_reduce (critical):**  
  - `order = jnp.argsort(dist)` has shape `(M*(M-1)/2,)`. Then `for idx in np.array(order):` iterates in **Python** over the full sorted index array. So we sync M*(M-1)/2 ints to host and run up to that many loop iterations. With M=2000 that’s ~2e6 iterations; with merge cap (e.g. M≤2000) we still shouldn’t run this at all when M is large (return no-op). When M is within cap, the loop does `int(i_idx[idx])`, `int(j_idx[idx])`, `float(dist[idx])` — each is a JAX array index and **forces a sync**. So we have O(M²) syncs in one operator.  
  - Then `for i, j in selected_pairs:` (at most `max_pairs` iterations, e.g. 4) does `float(weights_new[i])`, etc., and JAX `Lambdas_new.at[i].set(...)` in a loop. So a few syncs per merge; the dominant cost is the pair-selection loop above.  
  **How:** (1) Cap is already enforced (primitive_merge_max_tile_size); keep it to avoid O(M²) work on large tiles. (2) Move pair selection into JAX: e.g. take the first K indices from argsort (K = min(max_pairs*2, safe_size)), compute (i_idx, j_idx, dist) for those in JAX, then run a **fixed-size** JAX loop (lax.while_loop or fori_loop) with a “used” mask to pick disjoint pairs, so that only the final selected pairs (or their indices) are transferred to host once. (3) Apply merges in JAX: given a small list of (i,j) pairs, compute all new (Lambda, theta, eta, …) in JAX and do one batched scatter instead of a Python loop with .at[i].set.

- **sinkhorn_ot.py — cost matrix (not on runtime path):** The module `sinkhorn_ot.py` is **not imported anywhere** in the codebase. The runtime association path uses `primitive_association.py`, which already has JAX cost (`_compute_sparse_cost_matrix_jax`) and JAX Sinkhorn (`_sinkhorn_unbalanced_fixed_k_jax`). So the double loop in `sinkhorn_ot.py` does not affect current performance. That file is experimental/future (BEV scaffold). If we keep it: vectorize `cost_matrix_bev` (e.g. (N,M) broadcasting) for code hygiene. Simplifying it is **optional**.

- **backend_node time_offset:** See item 4 (Python loop over n_points with struct.unpack_from).

- **pipeline.py — insert event_log_entries:** For each active tile we do `mu_np = np.array(mu_world_ins, ...)`, `w_np = np.array(weights_ins, ...)`, `c_np = np.array(colors_ins, ...)` then `for i_ins in range(mu_np.shape[0]): event_log_entries.append({...})`. So we sync at most N_ACTIVE_TILES * k_insert_tile primitives (e.g. 7*64 = 448) to host and then a small Python loop. Cost is modest; could be reduced by building the log entries in a single batch and syncing only the minimal dict list, or by making event log optional/sampled.

- **pipeline.py — IMU jerk:** `MinimalScanTape` defines `imu_jerk_norm_mean` and `imu_jerk_norm_max`, but the pipeline **does not currently compute or set them** (they remain 0.0). Either: (1) remove these unused tape fields, or (2) add a vectorized jerk computation in JAX (diff along time, norm, jnp.mean/jnp.max) and fold into the diagnostics kernel; then sync once with other diagnostics.

**Spec:** §1.5 no hidden iteration (fixed-size loops only); §1.7 no superlinear materialization; §12.4 loop discipline (JAX control flow in JIT). Moving merge pair selection into JAX and making it fixed-size satisfies the spec.

---

## 6. JAX → Python → JAX elimination

**Where:** Same as items 11 and 13; summary of round-trips that can be removed or reduced.

**What:** Any path that does: compute in JAX → transfer to host (sync) → compute in Python/NumPy → transfer back to JAX adds latency and can prevent fusion/JIT.

**Findings:**

- **LiDAR points (backend_node):** Parse produces NumPy; we then `jnp.array(pts, ...)` (host→device), then in _process_lidar_msg we `record_device_to_host(points)`, `pts_np = np.array(points)`, transform in NumPy, then `points = jnp.array(pts_base, ...)` (host→device). So we have host→device→host→device. **Eliminate:** Keep points as NumPy after parse; do base transform in NumPy; then a single `jnp.array(pts_base, ...)` for the pipeline. Or, move transform into the pipeline as JAX (R_base_lidar, t_base_lidar as JAX); then parse can return JAX once and we have no extra round-trip for transform.

- **Diagnostics (pipeline):** Current code syncs only L_pose6 (6×6) for the minimal tape, plus many float() for cert fields. **Eliminate:** Compute all diagnostic scalars in JAX (jnp.linalg.eigvalsh, jnp.trace, etc.); then sync only the resulting scalar bundle (e.g. one struct or ~30 floats). Keeps work on device and avoids per-field float() syncs.

- **Merge_reduce pair selection:** Currently we build full (i_idx, j_idx, dist) in JAX, then sync `order` and iterate in Python, doing more syncs per index. **Eliminate:** Implement greedy pair selection inside JAX (fixed-size loop or bounded scan) and transfer only the few selected (i,j) pairs or the updated tile; see item 5.

- **Tempering / excitation (pipeline):** We compute dt_asym_raw, z_to_xy_raw, s_dt, s_ex, beta in JAX, then `float(...)` to build temper_cert and apply scaling. The float() calls are necessary to pass scalars to Python dataclasses (CertBundle, etc.). We could keep a single “evidence scaling” result as a JAX struct and sync it once (beta, s_dt, s_ex, …) instead of many separate float() calls, then build cert from that struct. Reduces number of sync points.

**Spec:** §12.5 host↔device transfer discipline; §12.8 boundary rule (ROS/viz/diagnostics may stay NumPy/Python; inference core should minimize sync).

---

## 7. JAX migration and kernel consolidation (design goal)

**Goal:** Move as much inference and numeric work into JAX as possible; consolidate small kernels into fewer, larger JIT-compiled units to reduce launch overhead, sync points, and code paths. Per §12: single-device, single-path; no Python control flow in jitted paths; minimal host↔device.

### 7.1 What to move into JAX (priority order)

| Priority | Item | Where | Current | Target |
|----------|------|--------|---------|--------|
| P0 | **Merge_reduce pair selection + apply** | `primitive_map.py` | Python loop over `order` (O(M²) syncs); Python loop over `selected_pairs` with `.at[i].set` | One JIT: Bhattacharyya distances → argsort → fixed-size lax loop to pick disjoint pairs → batched merge (vmap) → single scatter into tile. One sync for (updated tile, n_merged). |
| P0 | **Diagnostics scalar bundle** | `pipeline.py` | Sync L_pose6 + many float() for certs; no full L/h sync | One JIT: inputs (L_evidence, h_evidence, pose, imu_accel, …) → outputs (logdet_L, trace_L, L_dt, trace_L_ex, eig_min/max, jerk_mean/max, distance_pose, …) as a single struct or flat array. One sync of ~30 floats. |
| P1 | **LiDAR base transform** | `backend_node.py` | NumPy R_base_lidar @ pts.T + t; then jnp.array(pts_base) | Either keep in NumPy (one host→device after transform) or move to pipeline: accept points in sensor frame + (R, t) in JAX and do transform inside pipeline JAX so no extra round-trip. |
| P1 | **Sinkhorn cost matrix** | `sinkhorn_ot.py` (optional) | Python double loop in **unused** module; runtime uses `primitive_association.py` (JAX) | If keeping sinkhorn_ot: vectorize cost_matrix_bev for hygiene. Runtime hot path is already JAX in primitive_association. |
| P1 | **Tempering + excitation scalars** | `pipeline.py` | Many float(dt_asym_raw), float(z_to_xy_raw), float(s_dt), float(beta), … | One small JIT: (L_evidence_raw, L_prior, config scalars) → (beta, s_dt, s_ex, dt_asym_raw, z_to_xy_raw). Return one struct; sync once; build CertBundle in Python from that. |
| P2 | **IMU jerk diagnostics** | `pipeline.py` / `diagnostics.py` | Tape fields exist but are never set (0.0) | Either remove unused fields or add vectorized JAX jerk (diff, norm, jnp.mean/jnp.max) and fold into diagnostics kernel. |
| P2 | **Insert event_log payloads** | `pipeline.py` | Sync mu_world_ins, weights_ins, colors_ins; Python loop to build list of dicts | Optional: JIT that returns (mu_log, w_log, c_log) for the K insert slots; sync once; build list of dicts in Python from arrays. Or keep as-is if K is small. |
| P2 | **Time_offset parsing** | `backend_node.py` | Python loop struct.unpack_from per point | Vectorized NumPy (no JAX): view msg.data as (n_points, point_step), take column at time_offset offset as uint32. Removes O(n_points) Python loop; stays on host (I/O boundary). |
| P2 | **ma_hex_web → JAX** | `fl_slam_poc/common/ma_hex_web.py` | NumPy loops (e.g. `compute_hex_scale_h`: `for i in range(N)` over Sigma_bev); some JAX helpers exist (`compute_hex_scale_h_jax`) | **Migrate to JAX:** Prefer JAX paths everywhere tiling/hex scale are used in the hot path. Either (1) migrate callers to use existing JAX helpers (e.g. `compute_hex_scale_h_jax`) and deprecate NumPy loops, or (2) add JAX versions for all hot-path routines and keep NumPy only at I/O boundary. Goal: no Python loops over N in ma_hex_web on the inference path. |

### 7.2 Kernel consolidation opportunities

| Kernel group | Current | Consolidated design |
|--------------|---------|----------------------|
| **Evidence + tempering + excitation** | Pipeline calls: compute_excitation_scales_jax, apply_excitation_prior_scaling_jax, then tempering (beta * L_evidence_raw), then fusion_scale_from_certificates, then info_fusion_additive. Multiple JIT entry points and syncs for beta, s_dt, s_ex, alpha. | One JIT (e.g. `evidence_to_fused_belief`): inputs (L_evidence_raw, h_evidence_raw, L_prior, h_prior, config scalars for tempering and excitation) → outputs (L_fused, h_fused, alpha, beta, s_dt, s_ex, conditioning scalars). Pipeline syncs once for (L_fused, h_fused, alpha, beta, s_dt, s_ex, cond); builds certs from those. Reduces several small kernels and multiple float() syncs. |
| **Merge_reduce** | One large Python function with JAX pieces (Bhattacharyya, argsort) then Python loops. | Single JIT: (tile arrays, merge_threshold, max_pairs) → (updated tile arrays, n_merged). All pair selection and merge application inside JIT; fixed-size lax loop for greedy pair selection (max_pairs iterations). |
| **Association: cost + Sinkhorn** | **Runtime path:** `primitive_association.py` already uses JAX cost (`_compute_sparse_cost_matrix_jax`) and JAX Sinkhorn (`_sinkhorn_unbalanced_fixed_k_jax`). The separate module `sinkhorn_ot.py` is not imported; it is experimental. | No change needed for runtime. If sinkhorn_ot is kept for future use: vectorize its cost matrix (optional). |
| **Per-scan map update (per-tile)** | Pipeline: for tid in active_tile_ids: fuse, insert, cull, forget, merge_reduce. Each op is separate; some have JIT internals, some Python. | Harder: tiles are independent but share primitive_map state. Could have one JIT that takes (atlas_map, active_tile_ids, measurement_batch, …) and returns (updated atlas_map, cert_aggregates). That’s a big refactor. Lighter touch: ensure fuse/insert/cull/forget/merge each have a single JIT core and Python only does loop over tiles + cert assembly. |
| **Diagnostics** | Many small NumPy ops after syncing L, h, pose, IMU. | One “diagnostics kernel” JIT: (L_evidence, h_evidence, pose_pred, pose0, odom_twist, imu_accel, imu_stamps, …) → (logdet_L, trace_L, L_dt, trace_L_ex, eigvals_pose6, jerk_mean, jerk_max, distance_pose, speed_odom, …). Single sync of output struct. |

### 7.3 Boundary rule (what stays Python/NumPy)

- **ROS:** Message parse, serialization, subscribe/publish. No JAX in callbacks except trivial jnp.array from already-host data if needed.
- **Certificates / ExpectedEffect:** Built in Python from scalars (so sync is required for cert construction). Minimize by syncing one struct of scalars per “phase” (e.g. evidence phase, map phase).
- **Config and control flow:** Pipeline order, tile list iteration, “which operator to call” — stay in Python. Only the numeric core of each operator moves into JAX.
- **Rerun / viz:** Export of poses, primitives, events for Rerun can stay NumPy; convert to numpy only at publish boundary and do not require sync or holds for this.

### 7.3b ma_hex_web migration to JAX

**Where:** `fl_slam_poc/common/ma_hex_web.py`.

**Goal:** Migrate tiling/hex scale logic to JAX so the inference hot path does not rely on NumPy Python loops. The module currently has both NumPy loops (e.g. `compute_hex_scale_h`: `for i in range(N)` over Sigma_bev) and some JAX helpers (e.g. `compute_hex_scale_h_jax`). Callers in the pipeline and primitive_association should use JAX paths where possible.

**How:** (1) Prefer existing JAX helpers (`compute_hex_scale_h_jax`, `tile_ids_from_cells_jax`, etc.) on the hot path; migrate callers away from NumPy versions. (2) Add JAX versions for any remaining hot-path routines that still use Python loops over N. (3) Keep NumPy only at I/O or config boundaries. This aligns with §12 (single-device, minimal host↔device) and avoids O(N) syncs in tiling/hex scale.

### 7.4 Phased implementation order

1. **Phase 1 (correctness + one big win):** Merge_reduce full JIT (pair selection + batched apply in JAX). Cap is already enforced; keep it. Removes O(M²) syncs and Python loops from the hottest map path.
2. **Phase 2 (diagnostics):** Diagnostics kernel JIT: all L/h/pose/IMU-derived scalars in one JIT, one sync. Drop current “sync L,h then NumPy eigvalsh/trace” path.
3. **Phase 3 (evidence fusion):** Consolidate evidence + tempering + excitation into one JIT; single sync for (L_fused, h_fused, alpha, beta, s_dt, s_ex); build certs from that.
4. **Phase 4 (association):** Runtime already uses JAX in primitive_association. Optional: vectorize cost in `sinkhorn_ot.py` if that module is kept for future use.
5. **Phase 5 (cleanup):** LiDAR transform in JAX; tempering scalar JIT if not already in Phase 3; IMU jerk in diagnostics kernel (or remove unused tape fields); **ma_hex_web migration to JAX** (prefer JAX helpers on hot path, remove/deprecate NumPy loops). Ensure we remove all legacy code and simplify!

### 7.5 Existing JIT coverage (no regression)

Current codebase already uses `@jax.jit` in: predict, deskew, point_budget, odom_evidence, imu_evidence, imu_gyro_evidence, imu_preintegration, visual_pose_evidence, recompose, hypothesis, fusion (internal), primitives (PSD, solve), belief (mean_pose, etc.), primitive_map (top-k, etc.), lidar_surfel_extraction, inverse_wishart, measurement_noise_iw, kappa. Keep these; consolidation should **merge** or **call** them, not reimplement in Python.

---

## Summary table (updated)

| # | Area | Impact | Effort | Spec-safe |
|---|------|--------|--------|-----------|
| 1 | Pipeline cert collection | Low–Med | Med (if refactor) | Yes |
| 2 | Dead code cleanup | Correctness + cleanup | Low | Yes |
| **3** | **Syncs / host–device crossings** | **High (perf)** | **Med** | **Yes (§12.5)** |
| **4** | **Callbacks / queues (time_offset loop)** | **Medium** | **Low** | **Yes** |
| **5** | **Massive loops (merge, sinkhorn C, jerk)** | **High (merge), Med (sinkhorn)** | **Med–High** | **Yes (§1.5, §1.7, §12.4)** |
| **6** | **JAX→Python→JAX elimination** | **High (diagnostics, merge)** | **Med** | **Yes (§12.5, §12.8)** |
| **7** | **JAX migration + kernel consolidation** | **High (perf, maintainability)** | **Phased (see §7)** | **Yes (§12)** |
| **8** | **ma_hex_web → JAX** | **Medium (hot-path tiling)** | **Med** | **Yes (§12)** |

**Suggested order (additions):** Follow **§7.4 phased order**: (1) Merge_reduce full JIT; (2) Diagnostics kernel JIT; (3) Evidence + tempering + excitation consolidation; (4) Optional sinkhorn_ot vectorization if kept; (5) LiDAR transform / tempering / jerk cleanup / **ma_hex_web JAX migration**. Do time_offset vectorization as a quick win early.

---

*Doc generated from GC_SLAM.md-aligned codebase review. Update this file when simplification work is done or priorities change.*
