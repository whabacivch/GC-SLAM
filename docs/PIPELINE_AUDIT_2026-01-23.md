# Pipeline Audit (No Fixes Yet) — 2026-01-23

Scope: identify “hidden” configuration, dead/disabled codepaths, hardcoded topic/value plumbing, and ad‑hoc heuristics that make it hard to know what is *actually* running end‑to‑end. This document is an **audit only** (no behavior changes); each “Candidate Fix” is for explicit approval.

## TL;DR (High-Risk Findings)

- **AUD-001 (Critical): RGB-D evidence publisher is unreachable** in the refactored frontend. **Status (current working tree): FIXED** — RGB-D publisher init moved into `_init_publishers()`. See `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:410`.
- **AUD-002 (High): Hardcoded topics prevent end-to-end rewiring**. **Status (current working tree): FIXED** — frontend/backend now parameterize `/sim/*` topics and frontend logs a wiring banner. See `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:386` and `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:89`.
- **AUD-003 (High): Docs/scripts reference non-existent launch files** (`poc_tb3_rosbag.launch.py`, `poc_3d_rosbag.launch.py`) and “phase2” paths. **Status (current working tree): appears FIXED** — remaining mentions are in this audit document only.
- **AUD-004 (Medium): LoopFactor contains a placeholder field** (`information_weight`) which is currently just copied from the solver objective, not a defined information-geometric weight. **Status: still present**. See `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:1109`.

## Ground Truth: MVP (M3DGR) Runtime Wiring

### Launch + Config Precedence

The M3DGR runner uses:
- Launch: `fl_ws/src/fl_slam_poc/launch/poc_m3dgr_rosbag.launch.py`
- Base YAML: `fl_ws/src/fl_slam_poc/config/fl_slam_poc_base.yaml`
- Preset YAML: `fl_ws/src/fl_slam_poc/config/presets/m3dgr.yaml`

Parameter precedence in `Node(parameters=[config_base, config_preset, {...inline...}])` is:
1. `config_base` (lowest)
2. `config_preset`
3. inline dict (highest)

This means inline launch dict values can silently override YAML presets even if the preset is “correct”.

### Topic Graph (M3DGR)

Rosbag topics (source) → Utility nodes → Frontend → Backend → Outputs:

1. `/livox/mid360/lidar` → `livox_converter` → `/lidar/points`
2. `/odom` → `odom_bridge` → `/sim/odom`
3. `frontend_node` subscribes (via `SensorIO`) to `/sim/odom` and `/lidar/points` (3D mode), optionally IMU
4. `frontend_node` publishes:
   - `/sim/anchor_create`
   - `/sim/loop_factor`
   - `/sim/imu_segment` (if enabled)
   - `/sim/rgbd_evidence` (if `publish_rgbd_evidence:=true`; AUD-001 addressed in current working tree)
5. `backend_node` subscribes:
   - `/sim/odom`
   - `/sim/anchor_create`
   - `/sim/loop_factor`
   - `/sim/imu_segment` (if IMU fusion enabled)
   - `/sim/rgbd_evidence` (topic configurable; see AUD-001 impact)
6. `backend_node` publishes:
   - `/cdwm/state`, `/cdwm/trajectory`, `/cdwm/map`, `/cdwm/op_report`, `/cdwm/backend_status`, `/cdwm/markers`, `/cdwm/loop_markers`

## Findings (with Candidate Fixes)

### AUD-001 — Unreachable RGB-D publisher (Critical)

**Where**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:423` (and subsequent)

**What**
- `self.pub_rgbd = self.create_publisher(...)` and `self._scan_count = 0` are located *after* a `return` inside `_update_descriptor_models`, making them unreachable.
- Any attempt to enable `publish_rgbd_evidence:=true` will eventually hit `self.pub_rgbd.publish(...)` without guaranteed initialization, or will simply never publish.

**Why it’s “hidden”**
- `fl_ws/src/fl_slam_poc/config/presets/m3dgr.yaml` sets `publish_rgbd_evidence: false`, so the broken codepath is not exercised in the MVP run.

**Candidate Fix (needs approval)**
- Move RGB‑D publisher initialization into `_init_publishers()` (and keep it conditional on `publish_rgbd_evidence`), and initialize `_scan_count` in `__init__`.
- Add an OpReport when RGB‑D publishing is disabled by configuration to make it observable.

**Status (current working tree)**
- Implemented as described: RGB‑D publisher init is in `_init_publishers()` and is conditional on `publish_rgbd_evidence`. See `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:410`.

---

### AUD-002 — Hardcoded topic names bypass parameters (High)

**Where**
- Frontend publishes hardcoded topics:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:381` (`/sim/loop_factor`, `/sim/anchor_create`, `/sim/imu_segment`)
- Backend subscribes hardcoded topics:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:151` (`/sim/odom`, `/sim/loop_factor`, `/sim/anchor_create`, `/sim/imu_segment`)

**Impact**
- Launch/YAML parameters can give the appearance of flexibility, but the runtime wiring is effectively fixed.
- Tests can pass while silently exercising a different wiring than intended (especially if other launch files exist externally).

**Candidate Fix (needs approval)**
- Promote `/sim/*` topic strings to ROS params for both frontend and backend, then thread launch args → YAML → Node params end-to-end.
- Add a startup “wiring banner” that prints resolved topic names and whether each subscription callback is active.

**Status (current working tree)**
- Frontend publishers use `loop_factor_topic`, `anchor_create_topic`, `imu_segment_topic` parameters and log a wiring banner. See `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:386`.
- Backend subscriptions use `odom_topic`, `loop_factor_topic`, `anchor_create_topic`, `imu_segment_topic` parameters. See `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:89`.

---

### AUD-003 — Status topics are duplicated / ambiguous (Medium)

**Where**
- `StatusMonitor` default publishes to `/cdwm/status`: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/diagnostics/status_monitor.py:79`
- Frontend also publishes `/cdwm/frontend_status`: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:386`
- Backend publishes `/cdwm/backend_status`: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py:263`

**Impact**
- Observability is split across 2–3 topics, which encourages “watch the wrong topic” failure modes.

**Candidate Fix (needs approval)**
- Decide on one canonical status topic schema (e.g., `/cdwm/frontend_status` and `/cdwm/backend_status` only) and remove/rename `/cdwm/status`, or make the `StatusMonitor` topic explicit in frontend initialization.

---

### AUD-004 — Placeholder LoopFactor field (Medium)

**Where**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:1109`

**What**
- `loop.information_weight = icp_result.final_objective  # Placeholder`

**Impact**
- Downstream debugging and evaluation can be misled: “information_weight” looks meaningful but is not defined.

**Candidate Fix (needs approval)**
- Either:
  - remove the field usage entirely, or
  - compute it from a declared operator (e.g., `icp_information_weight(...)`) and document the semantics in the message schema docs.

---

### AUD-005 — Parameters declared but not used (Medium)

**Observed examples**
- Frontend declares/validates IMU noise densities but does not use them in the frontend (they are used by backend):
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/common/param_models.py:18` (BaseSLAMParams includes IMU noise densities)
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:150` (declares `imu_gyro_noise_density`, `imu_accel_noise_density`)
- “Keyframe thresholds” exist as params but are currently only emitted in OpReports, not used to decide keyframes:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:153`

**Impact**
- Creates “phantom knobs”: values are set in launch/scripts but don’t affect behavior.

**Candidate Fix (needs approval)**
- Either wire these parameters into the actual operators that should consume them, or explicitly deprecate them (log at startup: “param X is accepted but currently unused”).

---

### AUD-006 — Silent compute truncations / approximations without explicit OpReport (Medium)

**Examples**
- AnchorCreate point subsampling to 1000 points (message size / compute) is silent:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py:958`
- 3D voxel filtering in `LoopProcessor.preprocess_pointcloud()` drops points without emitting an OpReport:
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/loops/loop_processor.py:257`

**Impact**
- These materially affect loop factor quality, but may not be visible in `/cdwm/op_report`.

**Candidate Fix (needs approval)**
- Emit explicit OpReports for point truncation and voxel filtering (as “BudgetTruncation” / “SensorPreprocessApprox”), with Frobenius correction if it changes an in-family distribution approximation.

---

### AUD-007 — Docs/scripts drift: missing launch files (High)

**Where**
- Docs/scripts reference launch files that do not exist in this repo tree:
  - `poc_tb3_rosbag.launch.py`
  - `poc_3d_rosbag.launch.py`
  - and “phase2/…” paths
- Examples:
  - `docs/MAP_VISUALIZATION.md:139`
  - `docs/3D_POINTCLOUD.md:93`
  - `tools/download_r2b_dataset.sh:51`

**Impact**
- Easy to run outdated instructions and then debug behavior that doesn’t match the current MVP launcher.

**Candidate Fix (needs approval)**
- Update docs/scripts to either:
  - point at `poc_m3dgr_rosbag.launch.py`, or
  - explicitly label phase2 items as archived and gated behind a feature flag / separate workspace.

---

### AUD-008 — Stale file paths in “mental model” (Medium)

**Observed**
- References to files that do not exist in the current tree (likely renamed/moved):
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/imu_jax_kernel.py` (not present; kernel is `backend/math/imu_kernel.py`)
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/visual_feature_extractor.py` (not present; exists under `frontend/scan/visual_feature_extractor.py`)

**Impact**
- Engineers chase the wrong files while debugging.

**Candidate Fix (needs approval)**
- Add a small “moved files map” to docs or add import-path compatibility shims (temporary) with deprecation warnings.

## Suggested Approval Workflow (Fix-by-Fix)

If you want, I can open PR-sized changes in this order, each gated behind your explicit “approve fix AUD-XXX”:
1. **AUD-001** (RGB‑D publisher reachability) + a unit/integration check that toggling `publish_rgbd_evidence:=true` produces at least one message.
2. **AUD-002** (topic parametrization) + “startup wiring banner” so we can see resolved topics and enabled callbacks.
3. **AUD-007** (docs/scripts drift) so onboarding/testing stops tripping on non-existent launch files.
4. **AUD-004/AUD-006** (placeholder + silent truncations) with explicit OpReport emissions and semantics.
