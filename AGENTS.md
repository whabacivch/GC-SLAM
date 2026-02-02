# Impact Project_v1: Agent Instructions

These rules apply only to this project. Other projects have their own rules.

## Project Intent
- Build a Frobenius–Legendre compositional inference backend for dynamic SLAM.
- Implement and verify **Geometric Compositional SLAM v2** as a strict, branch-free, fixed-cost SLAM backend.

## Target Endstate (GC v2 Full Implementation)

The current pipeline is LiDAR-only with fixed noise parameters. The target endstate implements the full spec from `docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md` including:

### Adaptive Noise (Inverse-Wishart)
- **Process noise Q**: Per-block Inverse-Wishart states (rot, trans, vel, biases, dt, extrinsic) with conjugate updates from innovation residuals.
- **Measurement noise Σ**: IW states for gyro, accel, odom, and LiDAR measurement noise.
- **Initialization**: Datasheet priors with low pseudocounts (ν = p + 0.5) for fast adaptation.
  - IMU (ICM-40609): gyro σ² ≈ 8.7e-7 rad²/s², accel σ² ≈ 9.5e-5 m²/s⁴
  - LiDAR (Mid-360): range σ ≈ 2cm, angular σ ≈ 0.15°, combined ~1e-3 m²/axis
  - Odometry: use real 2D-aware covariances from `/odom` message (0.001 for x,y; 1e6 for z/roll/pitch)

### Sensor Evidence (IMU + Odom)
- **Accelerometer direction**: von Mises-Fisher (vMF) likelihood on S² with random concentration κ.
- **Gyro integration**: Gaussian likelihood with IW-adaptive Σg.
- **Odom partial observation**: Constrain `[x, y, yaw]` strongly, `[z, roll, pitch]` weakly; use message covariance or IW-adaptive.
- **Time offset warp**: Soft membership kernel based on Δt uncertainty (no hard window boundaries).
- **Per-scan evidence**: IMU + Odom + LiDAR evidence fused additively before fusion scale.

### Likelihood-Based Evidence (Laplace/I-Projection)
- **Replace UT regression**: Build explicit likelihood terms (vMF directional + Gaussian translational).
- **Laplace approximation**: Compute (g, H) at z_lin via JAX autodiff or closed-form exponential family Hessians.
- **Exponential family closed forms**: Gaussian H = Σ⁻¹, vMF H = κ(I - μμᵀ).

### Invariants Preserved
- No gating: κ adaptation is continuous via resultant statistics, not threshold-based.
- No fixed constants: all noise parameters are IW random variables with weak priors.
- Branch-free: IW updates happen every scan regardless of "convergence".

## Canonical References (Do Not Drift)
- Geometric Compositional strict interface/spec anchor: `docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md`
- Self-adaptive constraints: `docs/Self-Adaptive Systems Guide.md`
- Math reference: `docs/Comprehensive Information Geometry.md`
- **Build-by-construction anchors** (conventions and pipeline written first; implementation and tracing must align):
  - Frame and quaternion conventions: `docs/FRAME_AND_QUATERNION_CONVENTIONS.md` (single source of truth for frames, quat order, SE(3) semantics).
  - Pipeline and data flow: `docs/IMU_BELIEF_MAP_AND_FUSION.md` (rosbag → frontend → backend, 14-step pipeline, evidence and fusion).
- Development log (required): `CHANGELOG.md`

## Build by construction
Conventions and pipeline docs are written **before** or alongside code so behavior is knowable by construction. When implementing or tracing: align with the canonical docs above; do not improvise frame semantics, pipeline order, or evidence flow. Use `FRAME_AND_QUATERNION_CONVENTIONS.md` for frames/quat/SE(3); use `IMU_BELIEF_MAP_AND_FUSION.md` (and related dataflow docs) for topic flow and pipeline steps. When in doubt, cite the doc and the code location that satisfies it.

## Quickstart and Validation
- Workspace: `fl_ws/` (ROS 2), package: `fl_ws/src/fl_slam_poc/`, tools: `tools/`
- Build: `cd fl_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fl_slam_poc && source install/setup.bash`
- GC eval (primary): `bash tools/run_and_evaluate_gc.sh` (artifacts under `results/`)
- Legacy eval (if needed): `bash tools/run_and_evaluate.sh` (artifacts under `results/`)

## Non-Negotiable Invariants (GC v2)
- Closed-form-first: prefer analytic operators; only use solvers when no closed-form exists.
- Associative, order-robust fusion: when evidence is in-family and product-of-experts applies, fusion must be commutative/associative.
- Soft association only: no heuristic gating; use responsibilities from a declared generative model.
- Loop closure is late evidence: recomposition only (no iterative global optimization); any scope reduction must be an explicit approximation operator with an internal objective + predicted vs realized effect.
- Local modularity: state is an atlas of local modules; updates stay local by construction.
- Core must be Jacobian-free; Jacobians allowed only in sensor→evidence extraction and must be logged as `Linearization` (approx trigger) with Frobenius correction.
- Self-adaptive rules are hard constraints: no hard gates; startup is not a mode; constants are priors/budgets; approximate operators return (result, certificate, expected_effect) with no accept/reject branching.
- No hidden iteration: disallow data-dependent solver loops inside a single operator call (fixed-size loops only).
- Fail-fast on contract violations: chart id mismatches, dimension mismatches, and missing configured backends/sensors are hard errors.

## No Fallbacks / No Multi-Paths (Required)

The root failure mode to prevent is: *multiple math paths silently coexist*, making it impossible to know what behavior is actually running.

**Hard rules (enforced in review):**
- One runtime implementation per operator: delete duplicates or move them under `archive/` (not importable by installed entrypoints).
- No fallbacks: no environment-based selection, no `try/except ImportError` backends, no “GPU if available else CPU”.
- If variants are unavoidable, selection is explicit (`*_backend` param) and the node fails-fast at startup if unavailable.
- Nodes must emit a runtime manifest (log + status topic) listing resolved topics, enabled sensors, and selected backends/operators; tests must assert it.

## Frobenius Correction Policy (Mandatory When Applicable)
- If any approximation is introduced, Frobenius third-order correction MUST be applied.
  - Approximation triggers: linearization, mixture reduction, or out-of-family likelihood approximation.
  - Implementation rule: `approximation_triggered => apply_frobenius_retraction`.
  - Log each trigger and correction with the affected module id and operator name.
- If an operation is exact and in-family (e.g., Gaussian info fusion), correction is not applied.

## Evidence Fusion Rules
- Fusion/projection use Bregman barycenters (closed-form when available; otherwise geometry-defined solvers only).
- All sensor evidence is constructed via Laplace/I-projection at z_lin: compute (g, H) of joint NLL, project H to SPD.
- Noise covariances (Σg, Σa, Σlidar) are IW random variables, not fixed constants; use IW posterior mean as plug-in estimate.

## Implementation Conventions (Project-Specific)
- `fl_slam_poc/common/`: pure Python utilities (no ROS imports).
- `fl_slam_poc/frontend/`: sensor I/O + evidence extraction + utility nodes (no rendering; rendering is output from state).
- `fl_slam_poc/backend/`: inference + fusion + kernels; outputs (trajectory, state, rendering) live in backend (e.g. `backend/rendering.py`).

## Operator Reporting (Required)
- Every operator returns `(result, CertBundle, ExpectedEffect)` per `docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md`.
- `CertBundle` must report: `exact`, `approximation_triggers`, `family_in/out` (where applicable), `closed_form`, `solver_used`, `frobenius_applied`.
- Enforcement rule: `approximation_triggers != ∅` ⇒ `frobenius_applied == True` (no exceptions).

## No Heuristics (Hard)
- No gating: no threshold/branch that changes model structure, association, or evidence inclusion.
- Domain constraints are allowed only as explicit DomainProjection logs (positivity/SPD/stable inversion safeguards).
- Compute budgeting is allowed only via explicit approximation operators; any mass drop must be preserved by renormalization or explicitly logged.
- Expected vs realized benefit is logged only in internal objectives (divergence/ELBO/etc.), never external metrics (ATE/RPE).

## Rigor and validation (mandatory)

A single convention slip, frame mix-up, or ordering error can invalidate the whole stack. Rigor is non-negotiable.

**Critical: docs can be wrong.** The code is what actually runs. We must always identify **what the code is ACTUALLY doing** vs **what we thought or designed** (the docs). The docs are intended design; the code is observed behavior. When they disagree, **report first, then fix**: state the mismatch clearly (code does X at file:line; doc says Y) and only then propose or apply a fix (fix code or update doc). Do not fix without reporting the mismatch. Never assume the doc is correct without verifying against the code.

**When implementing or reviewing any change:**

1. **Establish actual behavior from the code** — Read the code to determine what it does (frame direction, quat order, block indices, pipeline order, evidence terms). Cite file:line. Then compare to the docs. If they match, say so. If they disagree: **report first** — "Code does X at file:line; doc says Y" — then propose or apply a fix (fix code or update doc). Do not fix without reporting.
2. **Explicit convention checks** — For anything touching frames, transforms, or state layout: verify in the code: frame direction (who is parent/child), quat order (xyzw vs wxyz), block indices (GC state order), which topic/callback feeds which pipeline step. One wrong index or transposed rotation is a silent bug; the code is the source of truth for what is currently running.
3. **Flag mismatches and ambiguities** — If code and doc disagree, or something is underspecified, **report first**: "Code does X at file:line; doc says Y." Do not paper over with "probably" or "usually". Only after reporting, propose or apply a fix (fix code or update doc). Do not fix without reporting.
4. **TF and extrinsics** — We do not rely on `/tf` at runtime; extrinsics are parameters. When checking or adding transform usage: read the code to see what it actually does; compare to `FRAME_AND_QUATERNION_CONVENTIONS.md`; if they differ, flag and fix code or update doc.
5. **Pipeline and evidence flow** — Trace the data path in the code (pipeline, evidence sum, block layout). Compare to the pipeline doc. Report actual vs documented; reconcile by fixing code or updating doc.

**Principle:** We maintain extremely high rigor so that a small detail or convention never throws the whole thing off. **Actual behavior comes from the code; docs are design.** Always identify actual vs intended; when they diverge, **report the mismatch first**, then fix code or update doc. Prefer reporting prior to fixing.

## Review Checklist (Use Before Merging Changes)
- Does the change preserve the non-negotiable invariants?
- **Rigor:** Have we established what the code actually does (frames, quat order, state block order, pipeline step order, evidence terms) and compared to the docs? Any code–doc mismatch flagged and resolved (fix code or update doc)?
- Did the change introduce any new backend/operator variant or fallback path? If yes, remove it or move it under `archive/` and enforce explicit selection + fail-fast.
- Did the change introduce any approximation? If yes, is Frobenius correction applied and logged?
- Is evidence fusion performed by barycenters (closed-form when available)?
- Are loop closures handled by recomposition (not iterative global optimization)?
- Are responsibilities used for association (no gating)?

## Development Log (Required)
- Add a brief, timestamped entry to `CHANGELOG.md` for any material change in scope, assumptions, sensors, or model fidelity.
