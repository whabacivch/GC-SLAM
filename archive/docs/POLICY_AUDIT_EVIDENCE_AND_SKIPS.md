# Policy Audit: Evidence and Skips

This document lists policy-style behaviors in the GC v2 pipeline and frontend that could affect evidence use. Goal: ensure we don't consistently discard or block evidence without a clear, justified reason.

**Reference:** The "sensor warmup" skip (discard LiDAR until odom/IMU) was removed 2026-01-28 because it discarded evidence for no mathematical reason; see CHANGELOG and backend_node.py.

---

## 1. Policies that could affect evidence (reviewed)

### 1.1 IW updates: readiness weight, no gates (backend_node.py ~790–820, constants.py)

- **What:** We run the **full pipeline** (belief, map, LiDAR/odom/IMU evidence) for every scan. Inverse-Wishart **noise adaptation** is applied **every scan** for all three (process, measurement, lidar_bucket). There are **no if/else gates**; readiness is a **weight** on sufficient stats (prior/budget):
  - **Process:** weight = min(1, scan_count) — at scan 0 there is no previous belief so prediction residuals are undefined; we add zero suff stats. From scan 1 we add full suff stats.
  - **Measurement / LiDAR bucket:** weight = 1 — high-rate IMU/odom and first-scan residuals available, so we always add full suff stats.
- **Evidence between scans:** Scans are much less frequent than IMU and odom. Evidence between scans is buffered and fused at the next scan. Per-evidence weights let process adapt from scan 1 and meas/lidar from scan 0 without any branch.
- **Evidence impact:** We **do use** all evidence on every scan. IW updates run every scan; the **effect** of process IW at scan 0 is zero (weight 0). No evidence discarded; no gates.
- **Why weight not gate:** The old gate was "we don't have the info we need yet" — which is equivalent to not having the sufficient statistics. So we always apply IW with the suff stats we actually have: at scan 0 process has no prediction, hence no innovation residuals, hence zero suff stats; we add 0. No need to branch on "do we have enough"; we just contribute the suff stats we have.
- **Verdict:** **No** gates; branch-free. Readiness is a continuous weight, not a threshold.

---

### 1.2 PointCloud2 required fields — fail-fast (backend_node.py ~153–158)

- **What:** If a PointCloud2 message is missing required fields (`x`, `y`, `z`, `ring`, `tag`, `timebase_low`, `timebase_high`, `time_offset`), we **raise** and do not run the pipeline for that message.
- **Rationale:** Single math path; no silent fallbacks. We cannot parse the message without those fields.
- **Evidence impact:** We don't "skip" arbitrarily — we fail fast on invalid input. The alternative would be to define a fallback (e.g. publish empty scan), which would be a different policy. Current choice: invalid input → error, not guess.
- **Verdict:** **Data contract**, not evidence-discarding policy. Consistent with "no fallbacks" in AGENTS.md.

---

### 1.3 Empty or invalid PointCloud2 (backend_node.py ~143–144, 581–582)

- **What:** If `n_points <= 0` or message has no x/y/z, `parse_pointcloud2` returns **empty arrays**. In `on_lidar`, when `n_points == 0` we supply a **single zero-weight dummy point** and still run the pipeline.
- **Evidence impact:** We do **not** skip the scan. We run the pipeline with a dummy point so step count and state stay consistent; the scan contributes negligible geometric evidence.
- **Verdict:** **No** discard; we run with empty/dummy so we don't drop the scan from the pipeline.

---

### 1.4 Livox converter: empty or invalid CustomMsg (livox_converter.py ~166, 182)

- **What:** If `points_list` is empty or no finite points, we **return** without publishing a PointCloud2.
- **Rationale:** Nothing to publish; backend would get empty scan anyway.
- **Evidence impact:** No LiDAR message is sent for that CustomMsg, so the backend never sees that "scan." So we **do** discard that message — but it's **invalid** (no points or no finite points). The alternative would be to publish an empty PointCloud2; backend would then run with dummy point (see 1.3).
- **Verdict:** **Invalid-data policy.** We could change to "publish empty PointCloud2" so the backend still gets a scan (with dummy point); then we'd have one more pipeline run and no "silent drop" of a message. Optional consistency improvement.

---

### 1.5 Livox converter: per-point time offset (livox_converter.py ~192–209)

- **What:** If the driver doesn't provide per-point timestamps (e.g. livox_ros_driver2 CustomPoint has no `offset_time`), we set all `time_offset` to 0. Comment: "Better to skip deskewing than apply wrong corrections."
- **Evidence impact:** We **do not** discard the scan. We use the scan; we just don't apply within-scan deskew (wrong timestamps would corrupt evidence). So we're **not** discarding evidence; we're avoiding **wrong** evidence.
- **Verdict:** **Correct policy:** no fake timestamps; deskew effectively disabled when not available.

---

### 1.6 IMU normalizer: non-finite values (imu_normalizer.py ~104–114)

- **What:** First time we see non-finite gyro/accel we **warn and return** (don't publish that message). Second time we **fail-fast** (shutdown).
- **Evidence impact:** We **discard** that one IMU message (first occurrence) or shut down (second). So invalid IMU is not passed downstream.
- **Verdict:** **Invalid-data policy.** We could pass nothing (no evidence) or fail on first invalid; current "warn once, then fail" is a policy. Not "discard good evidence"; we're rejecting **invalid** data.

---

### 1.7 Dead-end audit: required_topics timeout (dead_end_audit_node.py)

- **What:** If configured "required" topics don't receive messages within a timeout, the audit reports missing and can affect wiring validation. This does **not** stop the backend or discard pipeline evidence; it's an audit/wiring check.
- **Verdict:** **Not** an evidence policy; it's a validation/wiring policy.

---

## 2. Summary table

| Location | Behavior | Discards evidence? | Verdict |
|----------|----------|--------------------|---------|
| backend_node (sensor warmup) | ~~Skip scan until odom/IMU~~ | ~~Yes~~ | **Removed 2026-01-28** |
| backend_node (IW readiness weight) | IW every scan; process weight min(1, scan_count), meas/lidar weight 1 | No (no gates; weight on suff stats) | Branch-free; prior/budget |
| backend_node (PointCloud2 required fields) | Fail-fast on missing fields | N/A (invalid input) | Data contract |
| backend_node (n_points == 0) | Run pipeline with dummy point | No | OK |
| livox_converter (empty / no finite points) | Return without publishing | That message not sent (invalid) | Invalid-data policy; could publish empty |
| livox_converter (no per-point time) | Set time_offset=0 | No | Correct (no wrong deskew) |
| imu_normalizer (non-finite) | Don't publish / fail-fast | That message (invalid) | Invalid-data policy |

---

## 3. Recommendations

1. **No other "wait for sensor X" skips** — The only scan-level skip that discarded good evidence was the sensor warmup; it's removed.
2. **IW readiness** — IW runs every scan with readiness weight: process min(1, scan_count), meas/lidar 1. No gates or constants; no change needed unless you want a different weight formula.
3. **Livox empty/invalid** — Optional: publish empty PointCloud2 instead of returning so the backend always sees one message per CustomMsg (pipeline runs with dummy point); keeps message count consistent.
4. **Invalid data (IMU, PointCloud2 fields)** — Current fail-fast / warn-once is consistent with "no silent fallbacks"; no change needed unless you want different invalid-data handling.

---

*Last updated: 2026-01-28 (after sensor warmup skip removal).*
