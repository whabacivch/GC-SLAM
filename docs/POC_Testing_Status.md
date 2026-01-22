### FL‑SLAM POC: Current State, What’s Tested, and What’s Missing

This document is a **truthful status** of the current SLAM “proof of concept” (POC) implementation in `fl_ws/src/fl_slam_poc`, what we have actually validated with tests, and what we still need in order to claim we can test a SLAM system (and later navigation) in a meaningful way.

---

### What we have (today)

#### Codebase structure (high level)

- **Core pose math (`fl_slam_poc/geometry/`)**
  - **`se3.py`**: SE(3) group operations using rotation vectors (no RPY), plus adjoint covariance transport and exp/log utilities.

- **Information‑geometric operators (`fl_slam_poc/operators/`)**
  - **`gaussian_info.py`**: Gaussian belief representation and fusion in **information form** (natural parameters).
  - **`information_distances.py`**: closed‑form distances (Hellinger / Fisher‑Rao for Gaussian(1D)/Student‑t/SPD; product manifold aggregation).
  - **`dirichlet_geom.py`**: Dirichlet mixture projection + optional third‑order Frobenius correction (Newton + OpReport).
  - **`icp.py`**: ICP with solver metadata + covariance estimate + “information weight”.
  - **`op_report.py`**: `OpReport` schema + validation enforcing audit policy (e.g., approximation triggers).
  - **`gaussian_geom.py`**: explicit Gaussian Frobenius “proof‑of‑execution” (no‑op with stats).

- **Probabilistic/adaptive models (`fl_slam_poc/models/`)**
  - **`nig.py`**: Normal‑Inverse‑Gamma descriptor model; predictive Student‑t; Fisher‑Rao distance for association.
  - **`timestamp.py`**: probabilistic timestamp alignment weight \(w(\Delta t)\).
  - **`birth.py`**: stochastic anchor birth (Poisson process), replacing hard gates.
  - **`adaptive.py`**: online adaptive parameters (Welford stats + prior regularization).
  - **`process_noise.py`**: adaptive process noise \(Q\) via Inverse‑Wishart posterior mean.
  - **`weights.py`**: stable product of independent weights in log space.

- **ROS nodes (`fl_slam_poc/nodes/`)**
  - **Phase 2**: Gazebo/simulation nodes live under `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/` (e.g. `sim_world.py`).
  - **`frontend_node.py`**: descriptor formation + Fisher‑Rao association + stochastic births + ICP loop‑factor publication.
  - **`fl_backend_node.py`**: Gaussian state inference + two‑pose loop update + markers + TF publication.
  - **`tb3_odom_bridge_node.py`**: converts TB3 absolute odometry to delta odometry for backend interface.
  - **Phase 2**: `sim_semantics_node.py` / `dirichlet_backend_node.py` live under `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/nodes/`.

- **Launch files (`fl_ws/src/fl_slam_poc/launch/`)**
  - **`poc_a.launch.py`**: synthetic sim wiring demo (sim_world + frontend + backend + basic TF).
  - **`poc_all.launch.py`**: synthetic sim + Dirichlet demo nodes (+ optional foxglove bridge).
  - **`poc_b.launch.py`**: Dirichlet demo only.
  - **Phase 2**: Gazebo and alternative launches are stored under `phase2/fl_ws/src/fl_slam_poc/launch/`.

- **Messages (`fl_ws/src/fl_slam_poc/msg/`)**
  - **`LoopFactor.msg`**: loop closure factor with explicit convention + covariance + solver metadata.
  - **`AnchorCreate.msg`**: anchor creation events.

- **MVP evaluation harness (`tools/`)**
  - `tools/run_and_evaluate.sh` runs the current M3DGR rosbag MVP end-to-end (SLAM + plots/metrics).

---

### What we have tested (and what that actually means)

#### Unit tests (`fl_ws/src/fl_slam_poc/test/test_audit_invariants.py`)

What is validated:
- **SE(3) correctness** at the math‑library level: compose/inverse/round‑trip + numerical stability edges.
- **Adjoint covariance transport** behaves as expected for independent uncertainties.
- **Gaussian information form** conversions and fusion are associative/commutative.
- **Closed‑form IG distances** satisfy metric properties (symmetry, triangle inequality) where expected.
- **ICP is bounded** (max iterations) and produces plausible transforms and covariance.
- **Model utilities** (timestamp weight, stochastic birth probability bounds, adaptive parameter behavior).
- **Audit enforcement** via `OpReport.validate()` for “Frobenius required” rules.

What it does **not** validate:
- That any ROS node is wired correctly end‑to‑end.
- That the synthetic sensor stream is physically consistent with the simulated trajectory/world.
- That loop closures improve map quality, reduce drift, or close trajectories.
- Any navigation/control behavior.

#### Launch/integration tests (`fl_ws/src/fl_slam_poc/test/test_end_to_end_launch.py`)

What is validated:
- Nodes can be launched together and communicate over ROS topics.
- Backend publishes state with correct frame metadata.
- Frontend publishes loop factors with required metadata.
- `OpReport` messages are emitted and JSON‑parseable.

What it does **not** validate:
- Correctness of the estimated trajectory vs ground truth.
- Quality/robustness under realistic sensor conditions.
- Any meaningful “SLAM success” criteria (consistency, drift, loop closure benefit).

#### Foxglove visualization sanity

What is validated:
- Bridge connectivity and topic visibility.
- Basic TF tree availability (so frames are placeable in 3D).
- Sensors are non‑empty (images show structure; world markers appear).

What it does **not** validate:
- SLAM correctness (visualization is not evaluation).
- Real sensor realism or closed‑loop robot behavior.

---

### What we still need to **actually** test a SLAM system (and why)

Below is a prioritized list of missing components that are required for tests to mean “we can test SLAM” rather than “we can launch nodes”.

#### 1) A consistent world + sensor simulator (minimum viable realism)

- **What’s missing**: sensors (scan/depth/image) generated from a **single coherent world model** and the robot pose.
- **Why it matters**: without consistency, the frontend’s ICP/association success is not evidence of SLAM behavior; it’s just exercising code paths.
- **Minimum bar**:
  - Define a world geometry (even simple boxes/walls) in one place.
  - Ray‑cast (2D lidar) and/or project (depth) from pose into that world.
  - Add noise models that match the declared observation likelihoods.

#### 2) Ground truth + quantitative SLAM metrics

- **What’s missing**: stored ground truth trajectory + standard evaluation metrics.
- **Why it matters**: SLAM is an estimation problem; we need objective measures.
- **Minimum bar**:
  - Publish/record a ground truth pose (in the same frame semantics).
  - Compute **ATE** (absolute trajectory error) and **RPE** (relative pose error).
  - Add tests with thresholds and regression tracking (e.g., “ATE must not worsen by >X%”).

#### 3) A control loop / closed‑loop motion

- **What’s missing**: an action/command interface and a policy/controller that drives motion based on state or goals.
- **Why it matters**: open‑loop straight‑line motion does not stress SLAM (no turns, revisits, occlusions, loop closure opportunities).
- **Minimum bar**:
  - Commanded velocity input (`cmd_vel`) and a simple controller (waypoints / circle / figure‑8).
  - Deterministic motion scripts for repeatability.

#### 4) Loop closure realism and coverage

- **What’s missing**: repeated revisits + ambiguous scenes + false positives/negatives.
- **Why it matters**: loop closure is where SLAM systems fail in the real world.
- **Minimum bar**:
  - Trajectories that revisit places.
  - Controlled “perceptual aliasing” cases.
  - Tests that assert loop closures improve ATE/RPE (not just “a message published”).

#### 5) Map representation (even minimal)

- **What’s missing**: any persistent map product to inspect or evaluate.
- **Why it matters**: SLAM ≠ just pose; the map is the other half (and drift is measured against it).
- **Minimum bar** (choose one):
  - 2D occupancy grid (nav_msgs/OccupancyGrid), or
  - sparse landmark map (markers/points) with update rules.

#### 6) Navigation integration (Nav2) — later but important

- **What’s missing**: planner, controller, costmaps, and goal execution.
- **Why it matters**: to test “useful things” like navigation, we must integrate the state estimate into a navigation stack or equivalent.
- **Minimum bar**:
  - Bring up Nav2 in Gazebo or in our simulator.
  - Ensure TF chain: `map -> odom -> base_link`.
  - Validate that following goals doesn’t break state estimation (and vice versa).

#### 7) Reproducible regression harness (CI‑like)

- **What’s missing**: deterministic runs, rosbag replays, and automated acceptance criteria.
- **Why it matters**: without reproducibility, we can’t confidently iterate or detect regressions.
- **Minimum bar**:
  - fixed seeds + recorded logs/bags.
  - scripted “run scenario → compute metrics → pass/fail”.

---

### What “the POC currently proves”

Right now, the POC proves:
- The **math and geometry primitives** behave sensibly (unit tests).
- The **ROS graph wiring** for the SLAM pipeline launches and communicates (launch tests).
- We have a foundation for **audit compliance** via `OpReport` and explicit conventions.

It does **not** yet prove:
- The system performs SLAM in a way we can measure and improve.
- The system works under realistic motion/sensor constraints.
- The system supports navigation or closed‑loop autonomy.

---

### Recommended next milestone (smallest “real SLAM test” we can build)

If we want the smallest step that becomes a real SLAM evaluation:

- **Build one coherent 2D world** (walls/boxes), **ray‑cast a lidar** from the true pose, and publish:
  - ground truth pose
  - noisy odometry
  - lidar scan from ray‑casting
- Run the existing frontend/backend and compute:
  - ATE/RPE vs ground truth
  - “loop closures improve error” on a looped trajectory (figure‑8)

That gets us from “wiring works” to “we can measure SLAM behavior”.

---

### Leverageable assets (with invariant-first review before adoption)

These are candidates we can reuse, but **must be audited** against the by-construction rules in `Project_Implimentation_Guide.sty` before integration.

#### Candidate sources
- **Coherent sensors + world** (Gazebo, real sensor topics): `Phantom Fellowship MIT/AIF2/frobenius_nav_v2/README.md`
- **Ground truth publisher** (Phase 2): `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/sim_world.py` (publishes `/sim/ground_truth`)
- **Deterministic test harness pattern** (seeded configs + JSON results): `Info_Geo-FORK1/Other_test/run_opencda_validation.py`
- **Closed-loop command interface** (`/cmd_vel`, `/goal_pose`): `Phantom Fellowship MIT/AIF2/frobenius_nav_v2/frobenius_nav/navigator_node.py`
- **Deterministic waypoint planning** (geodesic paths): `Info_Geo-FORK1/cognitive_agent/planning.py`
- **Sparse map/memory structure** (k‑d tree of splats): `Info_Geo-FORK1/cognitive_agent/pipeline.py`

#### Invariant-first adoption checklist (required)
- **Family + operator fit**: Identify family in/out and confirm closed-form or declared solver; document in `OpReport`.
- **Approximation triggers**: If any linearization/heuristic reduction occurs, apply Frobenius correction and log.
- **Association policy**: Ensure responsibilities are used; no gating or top‑k truncation without explicit BudgetTruncation.
- **Loop evidence handling**: Late evidence only via recomposition (no iterative global optimization).
- **Jacobian boundary**: Jacobians only allowed in sensor extraction; core operators remain Jacobian‑free.
- **Domain projections**: If constraints are enforced (SPD, positivity), log as DomainProjection.

If any candidate fails this checklist, we treat it as a reference only and do not integrate.

### Next step: Gazebo as the primary sensor source

To avoid non-coherent synthetic sensors, prefer Gazebo (or real hardware) as the source of `/scan`, `/odom`, and optionally `/camera/*`. See `GAZEBO_INTEGRATION.md` for the local, project-scoped run instructions and topic crosswalk.

---

### Validation harness pattern (from external projects, adapted)

We should keep a **seeded config + JSON report** structure for any SLAM evaluation runs. This is a pattern used in external validation code and maps cleanly to our ATE/RPE needs.

Minimum config fields (example):
- `seed`: fixed RNG seed for deterministic runs
- `scenario`: world/trajectory name
- `duration_sec`: run length
- `sensors`: which topics were active (scan/depth/camera)
- `noise_model`: declared observation noise settings (if simulated)

Minimum report fields (example):
- `overall_status`: `VALIDATED` or `REQUIRES_IMPROVEMENT`
- `metrics`: ATE, RPE, loop-closure delta
- `diagnostics`: sensor availability, number of loop factors, anchor count
- `run_metadata`: git hash or run id, start time, config snapshot
