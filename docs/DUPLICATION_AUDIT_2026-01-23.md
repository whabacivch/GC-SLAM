# Duplication Audit (Functions/Helpers) — 2026-01-23

Scope: identify duplicated or near-duplicated functions/helpers across the repo to improve engineering discipline and reduce “mystery wiring” from copy/paste drift. This is an **audit only** (no fixes applied); each “Candidate Fix” is for explicit approval.

## Snapshot

- Python files scanned (package + tools): ~71
- Function/method definitions scanned: ~442
- Exact (AST-identical) duplicate groups (non-trivial): 4
- Notable same-name near-duplicates (high similarity): QoS resolver helper

## Status Check (Current Working Tree)

Re-scan results after your recent edits:
- **DUP-001:** FIXED — consolidated into `tools/rosbag_sqlite_utils.py`.
- **DUP-002:** FIXED — consolidated into `tools/rosbag_sqlite_utils.py`.
- **DUP-003:** FIXED — frontend-side publishing consolidated via `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/diagnostics/op_report_publish.py`.
- **DUP-004:** FIXED — QoS resolution consolidated via `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/qos_utils.py`.
- **DUP-005:** still present (intentional dual backend; parity discipline still needed).
- **DUP-006:** FIXED — duplicate suppression consolidated via `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/dedup.py`.

## Exact Duplicates (AST-identical)

### DUP-001 — `resolve_db3_path` / `_resolve_db3_path` repeated across tools

**Where**
- Previously duplicated across multiple scripts; now consolidated into `tools/rosbag_sqlite_utils.py`.

**Risk**
- When a bugfix or behavior change is needed (e.g., rosbag2 layout edge cases), it will likely be applied to only one copy.

**Candidate Fix (needs approval)**
- Create a shared helper module (e.g., `tools/rosbag_sqlite_utils.py`) and import it from all scripts; keep CLI behavior identical.

---

### DUP-002 — Tiny rosbag topic helpers duplicated in tools

**Where**
- Previously duplicated across scripts; now consolidated into `tools/rosbag_sqlite_utils.py`.

**Candidate Fix (needs approval)**
- Fold into the same shared helper module as DUP-001.

---

### DUP-003 — “publish OpReport as JSON” helper duplicated

**Where**
- Frontend uses `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/diagnostics/op_report_publish.py`.

**Risk**
- Small divergence (topic, validation behavior, throttling, exception handling) can make op-reporting inconsistent across nodes.

**Candidate Fix (needs approval)**
- Use one canonical publisher helper (likely `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/diagnostics/publish.py:122`), or move a minimal `publish_op_report(node, pub, report)` into `fl_slam_poc/common/`.

## Near Duplicates / Same-Name Collisions Worth Reviewing

These are not identical, but represent likely copy/paste patterns or API parity requirements.

### DUP-004 — QoS “reliability string → QoSProfile list” duplicated

**Where**
- Consolidated into `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/qos_utils.py`.

**Risk**
- If “both/system_default/reliable/best_effort” behavior changes (or QoS depth defaults change), we get mismatch between sensor subscriptions and bridge subscriptions.

**Candidate Fix (needs approval)**
- Centralize in a shared module (likely `fl_slam_poc/common/qos.py`) and have both call the same function.

---

### DUP-005 — SE(3) API duplicated across NumPy and JAX backends (intentional, but needs parity discipline)

**Where (examples)**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/geometry/se3_numpy.py` vs `fl_ws/src/fl_slam_poc/fl_slam_poc/common/geometry/se3_jax.py`
- Same-name functions include: `se3_compose`, `se3_inverse`, `se3_exp`, `se3_relative`, `se3_adjoint`, `se3_cov_compose`

**Risk**
- API drift: different semantics/edge-case handling between NumPy vs JAX implementations can silently change behavior depending on which backend is used.

**Candidate Fix (needs approval)**
- Add a small “parity test” suite that asserts both implementations agree on randomly sampled inputs (within tolerances), and document any intentional differences.

---

### DUP-006 — `_is_duplicate` helper duplicated

**Where**
- Consolidated into `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/dedup.py`.

**Note**
- Kept semantics identical; only the stamp-keying / last-seen storage was centralized.

**Candidate Fix (needs approval)**
- Centralize into a shared “stamp key” helper (or unify the logic and naming).

## IDE/Open-Tab Note

Your open tab path `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/visual_feature_extractor.py` does not exist in this repo; the file is `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/scan/visual_feature_extractor.py`. This kind of path drift is a common contributor to duplicated logic (people re-create code in a “guessed” location).

## Proposed Approval Workflow (Fix-by-Fix)

If you want, I can propose small PR-sized patches, each gated behind “approve DUP-00X”:
1. DUP-001/DUP-002: consolidate rosbag sqlite helpers under `tools/`
2. DUP-004/DUP-006: consolidate QoS + duplicate-stamp helpers under `fl_slam_poc/common/`
3. DUP-003: unify OpReport publishing helper to remove divergence risk
4. DUP-005: add parity tests for `se3_numpy` vs `se3_jax` APIs
