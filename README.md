# Geometric Compositional SLAM 

**Compositional inference for dynamic SLAM using information‑geometric methods.**

Embracing uncertainty is a cornerstone of robust robotics and SLAM: we use principled tools (Inverse‑Wishart adaptive noise, Frobenius corrections for approximations, information‑form fusion) so the system can adapt amid conflicting data rather than assuming perfect models. GC SLAM  is a geometric compositional system in which global structure, topology, and semantic understanding emerge from the algebraic composition of local information-geometric primitives, without discrete control flow or heuristic gating.

---

## Overview

**Geometric Compositional SLAM v2** is a **strict, branch‑free, fixed‑cost** backend designed for auditability and robustness. The pipeline builds **explicit likelihood‑based evidence** (vMF/Matrix Fisher + Gaussians), fuses in information form, and applies **self‑adaptive noise** at every scan — no gates, no hidden iteration.

### Novelty / Why This Is Different

- **Information‑geometric operators** with **certificates** and **Frobenius corrections** when approximations are triggered.
- **Adaptive noise everywhere** via Inverse‑Wishart conjugacy — no fixed noise constants in the loop.
- **Branch‑free** fixed‑cost pipeline per scan: same math path every time.
- **Matrix Fisher rotation evidence** (scan‑to‑map) + **planarized translation** with self‑adaptive z precision.
- **Auditable runtime manifest** enumerating all enabled operators and backends (no hidden fallbacks).

### Goals

- **Robust SLAM in dynamic environments** without fragile heuristics or gating.
- **Traceable, reproducible behavior** (single‑path execution, fixed cost, explicit evidence).
- **Self‑adaptation** to changing noise and sensor reliability.

---

## Visuals

#fill in later with renders

---

## Status

The **primary implementation** is **Geometric Compositional SLAM v2** — a strict, branch-free, fixed-cost SLAM backend. Evaluation uses the **Kimera** dataset (see `docs/KIMERA_FRAME_MAPPING.md`).

- **22D augmented state:** pose (6D) + velocity (3D) + gyro bias (3D) + accel bias (3D) + time offset (1D) + LiDAR–IMU extrinsic (6D)
- **Sensors fused:** LiDAR (Matrix Fisher rotation + planar translation evidence), IMU (time-resolved vMF tilt + gyro rotation + preintegration factor), odometry (pose + twist with kinematic consistency)
- **Adaptive noise:** Inverse-Wishart for process Q and measurement Σ (gyro, accel, LiDAR); updates every scan (no gates)
- **14-step pipeline per scan:** Predict → Deskew → IMU+odom → z_lin → surfel+camera → association → visual_pose_evidence(M_{t-1}, z_lin) → fuse → recompose → map update(z_t) → cull/forget → anchor drift. See `docs/PIPELINE_ORDER_AND_EVIDENCE.md`.
- **Canonical topic/bag reference:** `docs/BAG_TOPICS_AND_USAGE.md`

**Known limitations** (see `docs/PIPELINE_DESIGN_GAPS.md`): cross-sensor consistency likelihoods are still diagnostics (gyro↔odom↔LiDAR yaw); IMU message covariances and LiDAR intensity are not consumed; nonlinear evidence still uses local quadraticization (vMF/MF → Gaussian info). Pipeline trace and causality: `docs/PIPELINE_TRACE_SINGLE_DOC.md`.

---

## Quick Start

### Build

```bash
cd fl_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select fl_slam_poc
source install/setup.bash
```

### Run GC v2 (primary evaluation)

```bash
# Full pipeline: SLAM + alignment + metrics + plots + audit tests (default: Kimera)
bash tools/run_and_evaluate_gc.sh
```

Uses the single canonical Kimera bag (`rosbags/Kimera_Data/ros2/10_14_acl_jackal-005`); all testing goes through this script and this bag. Artifacts go to `results/gc_YYYYMMDD_HHMMSS/` (trajectory, metrics, diagnostics, wiring summary, dashboard).

### Viewing (Rerun, Wayland-friendly)

Visualization uses [Rerun](https://rerun.io/) by default (replaces RViz; works on Wayland). The backend records map and trajectory to a file (default `/tmp/gc_slam.rrd`). After a run, open it with:

```bash
rerun /tmp/gc_slam.rrd
```

Or set `rerun_spawn:=true` when launching to spawn the Rerun viewer at startup. Disable with `use_rerun:=false`.

### Legacy pipeline (optional)

```bash
bash tools/run_and_evaluate.sh
```

Results under `results/` (legacy script).

---

## System Architecture

**GC v2** uses a single-process sensor hub and the backend node; the backend subscribes only to canonical `/gc/sensors/*` topics.

```
Rosbag (Kimera)  →  gc_sensor_hub  →  gc_backend_node  →  /gc/state, /gc/trajectory, TF
                 pointcloud_passthrough   (14-step pipeline
                 odom_normalizer          + IW updates
                 imu_normalizer           + hypothesis combine)
                 dead_end_audit
```

- **Raw topics (Kimera bag):** PointCloud2 LiDAR, odom, IMU, RGB-D (see `gc_kimera.yaml`)
- **Canonical (hub → backend):** `/gc/sensors/lidar_points`, `/gc/sensors/odom`, `/gc/sensors/imu`, `/gc/sensors/camera_image`, `/gc/sensors/camera_depth`
- **Outputs:** `/gc/state`, `/gc/trajectory`, `/gc/status`, `/gc/runtime_manifest`, TF

**14-step pipeline (per scan, per hypothesis):**

1. PointBudgetResample  
2. PredictDiffusion (OU-style, uses adaptive Q)  
3. DeskewConstantTwist (IMU preintegration over scan window)  
4. BinSoftAssign  
5. ScanBinMomentMatch  
6. KappaFromResultant (inside ScanBinMomentMatch)  
7. MatrixFisherRotation (rotation)  
8. PlanarTranslationEvidence (translation with self-adaptive z precision)  
9. Evidence: odom pose + odom twist + planar priors + IMU (time-resolved vMF + gyro + preint) + LiDAR (Matrix Fisher + planar translation)  
10. FusionScaleFromCertificates  
11. InfoFusionAdditive  
12. PoseUpdateFrobeniusRecompose  
13. PoseCovInflationPushforward (map update)  
14. AnchorDriftUpdate  

IW sufficient-statistics accumulation and hypothesis combine run after the pipeline in the backend node.

---

## Code Layout

```
fl_ws/src/fl_slam_poc/
├── fl_slam_poc/
│   ├── frontend/
│   │   ├── hub/gc_sensor_hub.py      # Single process: converter + normalizers
│   │   ├── sensors/
│   │   │   ├── pointcloud_passthrough.py
│   │   │   ├── imu_normalizer.py
│   │   │   └── odom_normalizer.py
│   │   └── audit/
│   │       ├── dead_end_audit_node.py
│   │       └── wiring_auditor.py
│   ├── backend/
│   │   ├── backend_node.py            # Orchestration, IW state, hypothesis combine
│   │   ├── pipeline.py                # process_scan_single_hypothesis (14 steps)
│   │   ├── rendering.py               # Splat rendering (output from state/map)
│   │   ├── operators/
│   │   │   ├── point_budget.py
│   │   │   ├── predict.py
│   │   │   ├── deskew_constant_twist.py
│   │   │   ├── imu_preintegration.py
│   │   │   ├── binning.py
│   │   │   ├── odom_evidence.py
│   │   │   ├── odom_twist_evidence.py
│   │   │   ├── imu_evidence.py, imu_gyro_evidence.py, imu_preintegration_factor.py
│   │   │   ├── matrix_fisher_evidence.py
│   │   │   ├── planar_prior.py
│   │   │   ├── fusion.py
│   │   │   ├── recompose.py
│   │   │   ├── map_update.py
│   │   │   ├── anchor_drift.py
│   │   │   ├── inverse_wishart_jax.py, measurement_noise_iw_jax.py, lidar_bucket_noise_iw_jax.py
│   │   │   └── ...
│   │   └── structures/
│   │       ├── bin_atlas.py
│   │       ├── inverse_wishart_jax.py
│   │       ├── measurement_noise_iw_jax.py
│   │       └── lidar_bucket_noise_iw_jax.py
│   └── common/
│       ├── belief.py                  # BeliefGaussianInfo (22D information form)
│       ├── certificates.py
│       ├── constants.py
│       ├── geometry/se3_jax.py
│       ├── jax_init.py
│       └── primitives.py
├── launch/gc_rosbag.launch.py
├── config/gc_unified.yaml
└── test/
```

---

## Validation & Evaluation

**Primary:** `bash tools/run_and_evaluate_gc.sh`

- Runs SLAM on the Kimera rosbag, aligns estimated trajectory to ground truth, computes ATE/RPE and per-axis errors, runs audit-invariant tests, and (if diagnostics are exported) builds a diagnostics dashboard.
- Outputs: `results/gc_YYYYMMDD_HHMMSS/` — `metrics.txt`, `metrics.csv`, trajectory plots, `estimated_trajectory.tum`, `ground_truth_aligned.tum`, `diagnostics.npz`, `wiring_summary.json`, `audit_invariants.log`.

Performance is under active iteration; see `docs/PIPELINE_DESIGN_GAPS.md` for current gaps and `docs/PIPELINE_TRACE_SINGLE_DOC.md` for a full trace.

---

## Docs to Start With

- **Pipeline & dataflow:** `docs/IMU_BELIEF_MAP_AND_FUSION.md`
- **Design gaps / roadmap:** `docs/PIPELINE_DESIGN_GAPS.md`
- **Trace (single scan):** `docs/PIPELINE_TRACE_SINGLE_DOC.md`
- **Frame conventions:** `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`

---

## Documentation

| Doc | Description |
|-----|-------------|
| **[AGENTS.md](AGENTS.md)** | Project invariants, quickstart, canonical references |
| **[CHANGELOG.md](CHANGELOG.md)** | History and design decisions |
| **docs/BAG_TOPICS_AND_USAGE.md** | Bag topics and pipeline usage (canonical) |
| **docs/PIPELINE_TRACE_SINGLE_DOC.md** | Single pipeline trace: value-as-object, spine, IMU/odom/LiDAR, belief/IW, z/performance |
| **docs/GC_SLAM.md** | GC v2 interface and operator contracts |
| **docs/IMU_BELIEF_MAP_AND_FUSION.md** | Pipeline reference: topics, steps, evidence, fusion |
| **docs/FRAME_AND_QUATERNION_CONVENTIONS.md** | Frames, quaternions, SE(3) |
| **docs/PIPELINE_DESIGN_GAPS.md** | Known limitations (cross-sensor consistency, unused covariances, nonlinear approximations) |
| **archive/docs/** | Archived dataset/docs (e.g. M3DGR_DYNAMIC01_ARCHIVE) |
| **docs/PREINTEGRATION_STEP_BY_STEP.md** | IMU preintegration steps (including gravity) |
| **docs/EVALUATION.md** | Evaluation metrics and workflow |
| **docs/TESTING.md** | Testing framework |
| **tools/DIAGNOSTIC_TOOLS.md** | Diagnostic and inspection tools |
| **tools/README_MCP.md** | Code Graph RAG MCP install (GitHub releases only) |

---

## Dependencies

- ROS 2 Jazzy  
- Python 3.10+  
- NumPy, SciPy  
- JAX (GPU recommended for IMU/Lie ops)  
- evo, matplotlib (for evaluation)

```bash
pip install -r requirements.txt
```

---

## References


---

## Outputs & Artifacts

Run outputs go to `results/gc_YYYYMMDD_HHMMSS/` and include:
- `trajectory.tum` (estimate)
- `metrics.json` (ATE/RPE)
- `diagnostics.npz` (per‑scan evidence and sentinels)
- `dashboard.html` (Plotly summary)

---

## Contact

**William Habacivch**  
Email: whab13@mit.edu
