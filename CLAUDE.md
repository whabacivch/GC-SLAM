# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 1. Rigor (always on)

This project requires **extremely high rigor**. A single convention slip, frame mix-up, or ordering error can invalidate the whole system.

**Critical: docs can be wrong.** The code is what actually runs. Always identify **what the code is ACTUALLY doing** vs **what we thought/designed** (the docs). When they disagree, **report first, then fix**: state the mismatch ("Code does X at file:line; doc says Y") before proposing or applying a fix. Do not fix without reporting. Prefer reporting prior to fixing.

**You must:**

1. **Establish actual behavior from the code** — Read the code to see what it does (frame direction, quat order, block indices, pipeline order, evidence terms). Cite file:line. Compare to the docs. If they match, say so. If they disagree: **report first** — "Code does X at file:line; doc says Y" — then propose or apply a fix. Do not fix without reporting.
2. **Check conventions in code first** — For frames, transforms, state: verify in the code; then compare to `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`, `docs/IMU_BELIEF_MAP_AND_FUSION.md`, `docs/SIGMA_G_AND_FUSION_EXPLAINED.md`. Flag any mismatch; do not assume the doc is right.
3. **Flag mismatches** — If code and doc disagree, **report first**: "Code does X at file:line; doc says Y." Only after reporting, propose or apply a fix. Do not fix without reporting.
4. **TF and extrinsics** — No runtime `/tf`; extrinsics are parameters. Read what the code actually does; compare to the frame convention doc; reconcile by fixing code or updating doc.
5. **Pipeline and evidence** — Trace the data path in the code; compare to the pipeline doc. Report actual vs documented; fix code or update doc so they align.

**Principle:** Actual behavior comes from the code; docs are design. Always identify actual vs intended; when they diverge, **report the mismatch first**, then fix code or update doc.

---

## 2. By construction

This project is built **by construction**: design and conventions are explicit; implementation aligns with them. Avoid classic shortcuts. When something is wrong, **find the root cause** and fix it; do not patch symptoms with thresholds, workarounds, or extra branches.

### Do not use

- **Magic numbers** — No unexplained literals that change behavior. Use config, priors, or IW/params from the spec (constants are priors/budgets).
- **Heuristics** — No gating, no threshold/branch that changes model structure, association, or evidence inclusion.
- **Gates** — No hard gates; no "if first N scans then X else Y". Startup is not a mode.
- **Fallbacks** — No environment-based selection, no try/except backends, no "GPU if available else CPU". One runtime implementation per operator; duplicates go to archive.
- **Hidden iteration** — No data-dependent solver loops inside a single operator call. Fixed-size loops only.
- **Accept/reject branching** — Approximate operators return (result, certificate, expected_effect); no accept/reject branching.
- **Iterative global optimization for loop closure** — Loop closure is recomposition only; no iterative global optimization.
- **Implicit domain constraints** — Domain constraints (positivity, SPD, etc.) only as explicit DomainProjection logs, not hidden clamps.
- **Implicit compute budgeting** — Compute budgeting only via explicit approximation operators; any mass drop must be renormalized or explicitly logged.
- **External metrics for expected vs realized** — Expected vs realized benefit only in internal objectives (ELBO, divergence, etc.); never in external metrics (ATE, RPE).

### Prefer

- **Information geometry and closed-form solutions** — Prefer analytic operators and information-geometric constructions (Bregman barycenters, exponential family, etc.) over typical numerical methods or ad-hoc approximations. Use solvers only when no closed-form exists. Core is Jacobian-free; Jacobians only in sensor→evidence extraction and must be logged as Linearization (approx trigger) with Frobenius correction.
- **Explicit operators and contracts** — Every operator returns (result, CertBundle, ExpectedEffect) per spec. If approximation is introduced, Frobenius correction must be applied and logged. approximation_triggers ≠ ∅ ⇒ frobenius_applied.

**Principle:** Find the root cause. Do not paper over with heuristics, magic numbers, or extra branches.

---

## 3. Canonical references

- **Geometric Compositional spec:** `docs/GC_SLAM.md`
- **Frame and quaternion conventions:** `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`
- **Pipeline and data flow:** `docs/IMU_BELIEF_MAP_AND_FUSION.md`
- **Project instructions and checklist:** `AGENTS.md`

When in doubt, cite the doc and the code location that satisfies it. When code and doc disagree, report the mismatch first, then fix code or update doc.
