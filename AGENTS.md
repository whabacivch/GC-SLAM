# Impact Project_v1: Agent Instructions

These rules apply only to this project. Other projects have their own rules.

## Project Intent
- Build a Frobenius–Legendre compositional inference backend for dynamic SLAM.
- Preserve the design invariants in `Project_Implimentation_Guide.sty`.

## References and Repositories
- Design invariants and formal notation: `docs/Project_Implimentation_Guide.sty`
- Mathematical reference compendium: `docs/Comprehensive Information Geometry.md`
- System architecture diagram: `.codeviz/Impact Project 1/Impact-Project-1.md`
- Gazebo integration guide (experimental/future): `archive/legacy_docs/GAZEBO_INTEGRATION.md` (see ROADMAP.md - Priority 3/4)
- Development log (required updates): `CHANGELOG.md`
- POC testing status and notes: `docs/POC_Testing_Status.md`
- ROS 2 workspace (code + packages): `fl_ws/`
- Primary package repository: `fl_ws/src/fl_slam_poc/`
- Scripts and evaluation entrypoints: `tools/`

## Quickstart and Validation
- Prereqs: ROS 2 + Python (see `requirements.txt`); keep any system deps documented in `CHANGELOG.md`.
- Build: `cd fl_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fl_slam_poc && source install/setup.bash`.
- MVP run + evaluation (M3DGR): `bash tools/run_and_evaluate.sh` (produces metrics/plots under `results/`).
- Key nodes (launched by `poc_m3dgr_rosbag.launch.py`): `frontend_node`, `backend_node`, plus utility nodes for decompression/conversion/odom bridging.
- Run integration test (alternative datasets): `tools/test-integration.sh`. See `docs/TESTING.md` for complete documentation.
- Package structure: `fl_slam_poc/common/` (pure Python utilities), `fl_slam_poc/frontend/` (sensor processing + utility nodes), `fl_slam_poc/backend/` (state estimation + fusion).
- Logs/artifacts (ignore in reviews): `fl_ws/log/`, `fl_ws/build*/`, `fl_ws/install*/`.

## Current Priorities (See ROADMAP.md)
- **Priority 1 (Immediate):** IMU integration + 15D state extension, wheel odom separation, dense RGB-D in 3D mode, evaluation hardening.
- **Priority 2 (Near-term):** Camera-frame Gaussian splat map with vMF shading.
- **Priority 3 (Medium-term):** Alternative datasets (TurtleBot3, NVIDIA r2b), GPU acceleration, Gazebo live testing.
- **Priority 4 (Long-term):** Visual loop factors, GNSS integration, semantic observations, backend optimizations.

## Experimental/Future Workflows

### Gazebo Simulation (Not Part of MVP - Priority 3/4)
- **Status:** Experimental/future work. See `ROADMAP.md` - Gazebo is listed under "Medium-Term (Priority 3)" and "Long-Term (Priority 4)".
- **Current MVP:** Focus is on M3DGR rosbag evaluation pipeline. Gazebo integration is deferred.
- **If re-enabled:** Principle: Gazebo is a sensor/world *source* only; do not move estimation invariants into the simulator layer.
- **Files:** Gazebo launch files and nodes are stored under `phase2/` and are not installed by the MVP package by default.
- **Reference:** See `archive/legacy_docs/GAZEBO_INTEGRATION.md` for historical documentation (may be outdated).

## System Snapshot (MVP - M3DGR Rosbag Pipeline)
**Component Summary**
- `tb3_odom_bridge`: absolute → delta odom conversion (`/odom` → `/sim/odom`). Generic odom bridge (legacy name, not TB3-specific).
- `image_decompress`: rosbag image decompression for RGB-D processing (optional, when `enable_decompress:=true`).
- `livox_converter`: Livox LiDAR message conversion (`/livox/mid360/lidar` → `/lidar/points`).
- `frontend_node`: sensor association, ICP loop detection, anchor management (`/scan`, `/lidar/points`, `/camera/*`, `/odom` → `/sim/loop_factor`, `/sim/anchor_create`, `/sim/imu_segment`).
- `backend_node`: information-geometric fusion, trajectory estimation (`/sim/odom`, `/sim/loop_factor`, `/sim/anchor_create`, `/sim/imu_segment` → `/cdwm/state`, `/cdwm/trajectory`, `/cdwm/map`).

**Key Data Flow (MVP)**
Rosbag Topics → Utility Nodes (decompress/convert/bridge) → Frontend (association + ICP) → LoopFactor/AnchorCreate → Backend (fusion) → State/Trajectory/Map

## Non-Negotiable Design Invariants
- Closed-form-first: prefer analytic operators; only use solvers when no closed-form exists.
- Associative, order-robust fusion: when evidence is in-family and product-of-experts applies, fusion must be commutative/associative.
- Soft association only: no heuristic gating; use responsibilities from a declared generative model.
- One-shot loop correction: late loop evidence is inserted via barycenter recomposition, not global iterative re-optimization.
- Local modularity: state is an atlas of local modules; updates stay local by construction.

## Frobenius Correction Policy (Mandatory When Applicable)
- If any approximation is introduced, Frobenius third-order correction MUST be applied.
  - Approximation triggers: linearization, mixture reduction, or out-of-family factor approximation.
  - Implementation rule: `approximation_triggered => apply_frobenius_retraction`.
  - Log each trigger and correction with the affected module id and operator name.
- If an operation is exact and in-family (e.g., Gaussian info fusion), correction is not applied.

## Evidence Fusion Rules
- Use Bregman barycenters for fusion and projection.
- For Gaussian and Dirichlet/Categorical families, use closed-form in expectation coordinates.
- If inversion is not closed-form, use geometry-defined solvers only (mirror descent or natural-gradient Newton).

## Jacobian Policy (Core vs Front End)
- Core fusion/propagation/recomposition must remain Jacobian-free; use natural/dual coordinate operators only.
- Jacobians are permitted only in sensor-to-evidence extraction (e.g., registration) and must be logged as `Linearization` with Frobenius correction applied.
- Prefer information-geometric alternatives to Jacobians in the front end when available; document the model family and operator used.

## Loop Closure Rules
- Treat loop closures as late-arriving evidence factors.
- Exact recomposition scope is model-defined: the minimal set of modules whose likelihood terms include the loop factor, plus any explicitly shared-parameter modules.
- No neighborhood expansion unless implied by declared coupling.
- If the exact scope is too large, use a declared approximation operator (see BudgetedRecomposition).
- If loop evidence required linearization, apply Frobenius correction.

## Loop Closure Rules (By-Construction)
- L1. Loop closures are explicit evidence factors in a declared family, or an explicit approximation trigger (Linearization or OutOfFamilyApprox).
- L2. Exact scope is the model-defined affected set (factor argument list + shared-parameter modules). No heuristic expansion.
- L3. If scope must be reduced, it must be an explicit approximation operator:
  - ApproxOp: BudgetedRecomposition(e_loop, budget B)
  - Objective must be defined (no rule-of-thumb): maximize a model-intrinsic proxy for posterior change.
  - Recommended objectives: Bregman divergence reduction or mutual information (closed-form for Gaussian/Dirichlet).
  - Fisher information gain proxy is allowed if documented.
- L4. Mandatory correction + logging:
  - Linearization or BudgetedRecomposition triggers Frobenius correction.
  - Log: loop factor id, selected scope, objective value, diagnostics (predicted vs realized posterior change).
- L5. No iterative global optimization: recomposition only (barycenter + Frobenius correction if triggered).
- L6. If model-defined scope is large, use hierarchical/recursive recomposition via associativity (exact, not an approximation).
- L7. Pre-emptive anchor selection is allowed only if anchors are declared as part of the generative model (not heuristic).

## Implementation Conventions (Project-Specific)
- ROS 2 workspace lives in `Impact Project_v1/fl_ws`.
- Primary package: `Impact Project_v1/fl_ws/src/fl_slam_poc`.
- Package structure (flattened):
  - `fl_slam_poc/common/` - Pure Python utilities (no ROS imports): SE(3) operations, Dirichlet geometry, IMU preintegration, constants, op reports.
  - `fl_slam_poc/frontend/` - Sensor processing + utility nodes: frontend orchestration, sensor I/O, anchor management, loop processing, ICP, point cloud GPU, utility nodes (image_decompress, livox_converter, tb3_odom_bridge).
  - `fl_slam_poc/backend/` - State estimation + fusion: backend orchestration, Gaussian fusion, IMU kernels, information distances, parameter models (NIG, birth, adaptive), routing.
- Add new operators/utilities to `fl_slam_poc/common/`.
- Add new sensor processing or utility nodes to `fl_slam_poc/frontend/`.
- Add new fusion/estimation code to `fl_slam_poc/backend/`.
- Add launch files under `fl_slam_poc/launch/`.

## Operator Taxonomy (Required Reporting)
- Every operator must emit an OpReport:
  - exact: bool
  - approximation_triggers: set[Trigger] (empty if exact)
  - family_in: Family, family_out: Family
  - closed_form: bool
  - solver_used: Optional[SolverType]
  - frobenius_applied: bool
- Enforcement rule: if approximation_triggers is non-empty, frobenius_applied must be True.

## Guardrails vs Heuristics
- Heuristic gating is forbidden: thresholds that change model structure or association.
- Domain constraints are allowed and must be logged:
  - Positivity (e.g., alpha > 0)
  - SPD constraints (e.g., Sigma >> 0)
  - Stable inversion safeguards
- Log these as DomainProjection, not as gating.

## Soft Association (Strict)
- Responsibilities are mandatory for association.
- Forbidden: top-k truncation, silent mass drops, or thresholded births (e.g., r_new > tau).
- Compute budgeting is allowed only via explicit approximation operators:
  - BudgetTruncation must preserve total mass via renormalization or log mass drop.
  - BudgetTruncation is an approximation trigger and requires Frobenius correction.

## Review Checklist (Use Before Merging Changes)
- Does the change preserve the non-negotiable invariants?
- Did the change introduce any approximation? If yes, is Frobenius correction applied and logged?
- Is evidence fusion performed by barycenters (closed-form when available)?
- Are loop closures handled by recomposition (not iterative global optimization)?
- Are responsibilities used for association (no gating)?

## Development Log (Required)
- Add a brief, timestamped entry to `CHANGELOG.md` for any material change in scope, assumptions, sensors, or model fidelity.
