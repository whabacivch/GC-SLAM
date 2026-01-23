# Impact Project_v1: Agent Instructions

These rules apply only to this project. Other projects have their own rules.

## Project Intent
- Build a Frobenius–Legendre compositional inference backend for dynamic SLAM.
- Preserve the design invariants in `archive/legacy_docs/Project_Implimentation_Guide.sty`.

## Canonical References (Do Not Drift)
- Design invariants/spec anchor: `archive/legacy_docs/Project_Implimentation_Guide.sty`
- Self-adaptive constraints: `docs/Self-Adaptive Systems Guide.md`
- Math reference: `docs/Comprehensive Information Geometry.md`
- Development log (required): `CHANGELOG.md`

## Quickstart and Validation
- Workspace: `fl_ws/` (ROS 2), package: `fl_ws/src/fl_slam_poc/`, tools: `tools/`
- Build: `cd fl_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fl_slam_poc && source install/setup.bash`
- MVP eval: `bash tools/run_and_evaluate.sh` (artifacts under `results/`)

## Non-Negotiable Design Invariants
- Closed-form-first: prefer analytic operators; only use solvers when no closed-form exists.
- Associative, order-robust fusion: when evidence is in-family and product-of-experts applies, fusion must be commutative/associative.
- Soft association only: no heuristic gating; use responsibilities from a declared generative model.
- Loop closure is late evidence: recomposition only (no iterative global optimization); any scope reduction must be an explicit approximation operator with an internal objective + predicted vs realized effect.
- Local modularity: state is an atlas of local modules; updates stay local by construction.
- Core must be Jacobian-free; Jacobians allowed only in sensor→evidence extraction and must be logged as `Linearization` (approx trigger) with Frobenius correction.
- Self-adaptive rules are hard constraints: no hard gates; startup is not a mode; constants are priors/budgets; approximate operators return (result, certificate, expected_effect) with no accept/reject branching.

## No Fallbacks / No Multi-Paths (Required)

The root failure mode to prevent is: *multiple math paths silently coexist*, making it impossible to know what behavior is actually running.

**Hard rules (enforced in review):**
- One runtime implementation per operator: delete duplicates or move them under `archive/` (not importable by installed entrypoints).
- No fallbacks: no environment-based selection, no `try/except ImportError` backends, no “GPU if available else CPU”.
- If variants are unavoidable, selection is explicit (`*_backend` param) and the node fails-fast at startup if unavailable.
- Nodes must emit a runtime manifest (log + status topic) listing resolved topics, enabled sensors, and selected backends/operators; tests must assert it.

## Frobenius Correction Policy (Mandatory When Applicable)
- If any approximation is introduced, Frobenius third-order correction MUST be applied.
  - Approximation triggers: linearization, mixture reduction, or out-of-family factor approximation.
  - Implementation rule: `approximation_triggered => apply_frobenius_retraction`.
  - Log each trigger and correction with the affected module id and operator name.
- If an operation is exact and in-family (e.g., Gaussian info fusion), correction is not applied.

## Evidence Fusion Rules
- Fusion/projection use Bregman barycenters (closed-form when available; otherwise geometry-defined solvers only).

## Implementation Conventions (Project-Specific)
- `fl_slam_poc/common/`: pure Python utilities (no ROS imports).
- `fl_slam_poc/frontend/`: sensor I/O + evidence extraction + utility nodes.
- `fl_slam_poc/backend/`: inference + fusion + kernels.

## Operator Taxonomy (Required Reporting)
- Every operator emits an OpReport with: `exact`, `approximation_triggers`, `family_in/out`, `closed_form`, `solver_used`, `frobenius_applied`.
- Enforcement rule: `approximation_triggers != ∅` ⇒ `frobenius_applied == True` (no exceptions).

## No Heuristics (Hard)
- No gating: no threshold/branch that changes model structure, association, or evidence inclusion.
- Domain constraints are allowed only as explicit DomainProjection logs (positivity/SPD/stable inversion safeguards).
- Compute budgeting is allowed only via explicit approximation operators; any mass drop must be preserved by renormalization or explicitly logged.
- Expected vs realized benefit is logged only in internal objectives (divergence/ELBO/etc.), never external metrics (ATE/RPE).

## Review Checklist (Use Before Merging Changes)
- Does the change preserve the non-negotiable invariants?
- Did the change introduce any new backend/operator variant or fallback path? If yes, remove it or move it under `archive/` and enforce explicit selection + fail-fast.
- Did the change introduce any approximation? If yes, is Frobenius correction applied and logged?
- Is evidence fusion performed by barycenters (closed-form when available)?
- Are loop closures handled by recomposition (not iterative global optimization)?
- Are responsibilities used for association (no gating)?

## Development Log (Required)
- Add a brief, timestamped entry to `CHANGELOG.md` for any material change in scope, assumptions, sensors, or model fidelity.
