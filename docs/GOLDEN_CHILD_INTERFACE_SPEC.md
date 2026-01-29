# Golden Child SLAM v2 — Strict Interface Spec (Branch-Free, Fixed-Cost, Local-Chart) (2026-01-23)

This document is the **strict interface + budget spec** for the “Golden Child SLAM Method (v2)”, rewritten to eliminate **all `if/else` gates and regime switches**. Every operator is a **total function** (always runs), with **continuous influence scalars** whose effect may smoothly go to ~0. Any numerical/domain stabilization is **declared**, **always applied in the same way**, and **accounted for** in certificates.

Scope:

* Concrete field names, dimensions, invariants, and fixed-cost budgets.
* Every approximation is an **ApproxOp** returning `(result, certificate, expected_effect)` and must be logged.
* No silent fallbacks, no multipaths, no hidden iteration (data-dependent solver loops) inside a single operator call.
* Local charts (anchors) are mandatory and **continuous** (no threshold gating).

Non-goals:

* This does not prescribe ROS message schemas; it specifies the internal library contracts that ROS nodes must call.

---

## 1) Global Invariants (Hard)

### 1.1 Chart convention + state ordering (fail-fast)

* `chart_id = "GC-RIGHT-01"` is the global chart convention for all beliefs and evidence.
* SE(3) perturbation is **right**: `X(δξ) = X · Exp(δξ)`.
* Tangent ordering is fixed:

|               Slice | Symbol       | Dim | Indices (0-based) |
| ------------------: | ------------ | --: | ----------------- |
|                   t | δt           |   3 | 0..2              |
|               SO(3) | δθ           |   3 | 3..5              |
|                   v | δv           |   3 | 6..8              |
|           gyro bias | δbg          |   3 | 9..11             |
|          accel bias | δba          |   3 | 12..14            |
|         time offset | δΔt          |   1 | 15                |
| LiDAR–IMU extrinsic | δξLI (se(3)) |   6 | 16..21            |

* Augmented tangent dimension: `D_Z = 22`.
* Deskew tangent dimension: `D_DESKEW = 22`.
* Any `chart_id` mismatch is a hard error.
* Any dimensional mismatch is a hard error.

### 1.2 Local charts (anchors) are mandatory (hard)

* `chart_id` specifies the convention; **local charts are specified by anchors**.
* Every belief and evidence carries:

  * `anchor_id: str` (local chart instance id; stable within a hypothesis stream)
  * `X_anchor: SE3` (anchor pose in module-local/world frame)
* No operator is allowed to branch on anchor logic. Anchor evolution is continuous and always applied by `AnchorDriftUpdate` (ApproxOp).

### 1.3 Fixed-cost budgets (compile-time constants)

All are hard constants:

* `K_HYP = 4`
* `HYP_WEIGHT_FLOOR = 0.01 / K_HYP`
* `B_BINS = 48`
* `T_SLICES = 5`
* `SIGMA_POINTS = 2 * D_DESKEW + 1 = 45`
* `N_POINTS_CAP = 8192`

Any reduction of these budgets must be done by an explicit ApproxOp and logged.

### 1.4 No multipaths / no fallbacks (hard)

* No “GPU if available else CPU”.
* No alternate runtime implementations in codepaths.
* Backend choices are explicit in configuration and in the runtime manifest.

### 1.5 No hidden iteration (hard, precisely defined)

Disallowed inside any single operator call:

* data-dependent solver loops (Newton/CG/LS until tolerance, line search, adaptive iterations).

Allowed:

* repeating the fixed pipeline over time (time advances).
* fixed-size loops with compile-time constant bounds (`B_BINS`, `SIGMA_POINTS`, `T_SLICES`) with no early exit.

---

## 2) Core Data Structures (float64)

### 2.1 Gaussian belief on augmented tangent (information form)

```text
BeliefGaussianInfo:
  chart_id: str            # must be "GC-RIGHT-01"
  anchor_id: str           # local chart instance id (stable)
  X_anchor: SE3            # anchor pose
  stamp_sec: float

  z_lin: (D_Z,)            # linearization point in chart coordinates
  L: (D_Z, D_Z)            # symmetric PSD/PD information matrix
  h: (D_Z,)                # information vector

  cert: CertBundle
```

Interpretation:

* MAP increment is defined by a **fixed lifted SPD solve** (no conditional SPD checks):
  [
  \delta z^* = (L + \varepsilon_{lift} I)^{-1} h
  ]
  where `eps_lift` is a manifest constant. This is the *declared solve*.

### 2.2 Certificates + expected effect (required everywhere)

```text
CertBundle:
  chart_id: str
  anchor_id: str

  exact: bool
  approximation_triggers: list[str]     # always present for ApproxOps (may be empty for ExactOps)
  frobenius_applied: bool               # True iff any trigger magnitude is nonzero

  conditioning:
    eig_min: float
    eig_max: float
    cond: float
    near_null_count: int

  support:
    ess_total: float
    support_frac: float                 # retained_mass / total_mass (continuous)

  mismatch:
    nll_per_ess: float                  # continuous proxy; may be 0 if undefined
    directional_score: float            # continuous proxy; may be 0 if undefined

  excitation:
    dt_effect: float                    # continuous
    extrinsic_effect: float             # continuous

  influence:
    lift_strength: float                # eps_lift * D_Z (always)
    psd_projection_delta: float         # ||M_proj - sym(M)||_F (always computed)
    mass_epsilon_ratio: float           # eps_mass / (mass + eps_mass) (continuous)
    anchor_drift_rho: float             # continuous reanchor strength in [0,1]
    dt_scale: float                     # in [0,1]
    extrinsic_scale: float              # in [0,1]
    trust_alpha: float                  # fusion scale alpha in [alpha_min, alpha_max]

ExpectedEffect:
  objective_name: str
  predicted: float
  realized: float | null
```

Downstream rule:

* Consumers may only **scale** influence continuously using certificates (e.g., `trust_alpha`, `dt_scale`), never branch/skip.

---

## 3) Branch-Free Numeric Primitives (Library Contracts)

All of these are **total functions** (always run) and return a certificate magnitude that can be exactly zero if no change occurred.

### 3.1 ApproxOp: `Symmetrize`

Always compute:

* `M_sym = 0.5*(M + M^T)`
* magnitude `sym_delta = ||M_sym - M||_F`

Trigger is present always; `frobenius_applied = (sym_delta > 0)`.

### 3.2 ApproxOp: `DomainProjectionPSD`

Always compute:

* `M_sym = Symmetrize(M)`
* eigendecomp of `M_sym`
* `vals_clamped = max(vals, eps_psd)` with fixed `eps_psd=1e-12`
* `M_psd = V diag(vals_clamped) V^T`
* projection magnitude:
  [
  \Delta_{psd} = |M_{psd} - M_{sym}|_F
  ]
* conditioning from `vals_clamped`

No conditional “only if needed”; clamp always executed.

### 3.3 ExactOp: `SPDCholeskySolveLifted`

Always solve:

* ((L + eps_lift I)x = b), with fixed `eps_lift` from manifest.
* `lift_strength = eps_lift * D_Z` recorded always.

No alternate solvers.

### 3.4 Total function: `InvMass`

Always compute:

* `inv_mass(m) = 1 / (m + eps_mass)` with fixed `eps_mass` in manifest.
* `mass_epsilon_ratio = eps_mass / (m + eps_mass)` recorded always.

This removes all division-by-zero gating.

### 3.5 Total function: `Clamp`

Always compute:

* `Clamp(x, lo, hi) = min(max(x, lo), hi)`
* `clamp_delta = |Clamp(x)-x|` recorded always.

No conditional execution.

---

## 4) Core Structures for Mapping and Binning

### 4.1 Bin atlas

```text
BinAtlas:
  dirs: (B_BINS, 3)        # fixed unit vectors
```

### 4.2 Map bin stats (additive sufficient stats)

```text
MapBinStats:
  S_dir: (3,)
  N_dir: float
  N_pos: float
  sum_p: (3,)
  sum_ppT: (3, 3)
```

Derived values are computed using **InvMass** (no if):

* `inv_N_dir = 1/(N_dir + eps_mass)`
* `Rbar = ||S_dir|| * inv_N_dir` (well-defined)
* `mu_dir = S_dir / (||S_dir|| + eps_mass)` (well-defined)
* `inv_N_pos = 1/(N_pos + eps_mass)`
* `c = sum_p * inv_N_pos`
* `Sigma_c_raw = (sum_ppT * inv_N_pos) - c c^T`
* `Sigma_c = DomainProjectionPSD(Sigma_c_raw).M_psd` (always computed)

### 4.3 Deskewed point

```text
DeskewedPoint:
  p_mean: (3,)
  p_cov: (3, 3)
  time_sec: float
  weight: float
```

### 4.4 Scan bin stats

```text
ScanBinStats:
  N: (B_BINS,)
  s_dir: (B_BINS, 3)
  p_bar: (B_BINS, 3)
  Sigma_p: (B_BINS, 3, 3)
  kappa_scan: (B_BINS,)
```

No `support` boolean is used for control. Zero-mass bins contribute weight 0 automatically through `N[b]`.

---

## 5) ApproxOps and ExactOps (All Fixed-Cost, Branch-Free)

All operators return `(result, cert, expected_effect)`.

### 5.1 ApproxOp: `PointBudgetResample`

Always produce an output of size `<= N_POINTS_CAP` by deterministic resampling.

* If `N_raw <= N_POINTS_CAP`, resampling returns the same set (identity outcome) with `support_frac=1`.
* If `N_raw > N_POINTS_CAP`, resampling drops points and renormalizes weights so retained mass is preserved.

Certificate:

* `support_frac` computed always
* `approximation_triggers` always includes `"PointBudgetResample"`
* `frobenius_applied` determined by `1 - support_frac`

ExpectedEffect:

* `objective_name="predicted_mass_retention"`
* `predicted=support_frac`

### 5.2 ExactOp: `PredictDiffusion`

Always predicts between timestamps at fixed cost.

Inputs:

* `belief_prev`
* PSD diffusion `Q: (22,22)` from process noise module
* `dt_sec`

Construction (always):

1. Convert info to moments:

   * Solve `mu = (L + eps_lift I)^{-1} h` using `SPDCholeskySolveLifted`
   * Compute `Sigma` as ((L + eps_lift I)^{-1}) via multiple solves (fixed cost `D_Z` solves)
2. Predict:

   * `mu_pred = mu`
   * `Sigma_pred_raw = Sigma + Q * dt_sec`
3. `Sigma_pred = DomainProjectionPSD(Sigma_pred_raw).M_psd` (always)
4. Convert back:

   * `L_pred_raw = inverse(Sigma_pred)` computed by Cholesky solves (fixed cost)
   * `L_pred = DomainProjectionPSD(L_pred_raw).M_psd` (always)
   * `h_pred = L_pred @ mu_pred`

Certificate includes:

* lift_strength
* psd_projection_delta for `Sigma_pred` and `L_pred`
* conditioning

ExpectedEffect:

* `objective_name="predicted_trace_increase"`
* `predicted=trace(Q)*dt_sec`

### 5.3 ApproxOp: `DeskewUTMomentMatch`

Always uses exactly `T_SLICES*SIGMA_POINTS` evaluations.

Inputs:

* `belief_pred`
* points (budgeted)
* timing model

Outputs:

* `deskewed_points` with `(p_mean, p_cov)` for each point
* excitation scalars computed continuously from UT contributions:

  * `dt_effect >= 0`
  * `extrinsic_effect >= 0`
* also output `ut_cache` (required) containing per-slice, per-sigma transforms and deltas for reuse downstream (fixed size)

Certificate:

* triggers include `"DeskewUTMomentMatch"`
* excitation fields filled always

ExpectedEffect:

* `objective_name="predicted_deskew_cov_trace"`
* `predicted=weighted_mean_i trace(p_cov_i)`

### 5.4 ApproxOp: `BinSoftAssign`

Always compute responsibilities:
[
r_{i,b} = \frac{\exp(\langle u_i, d_b\rangle/\tau)}{\sum_{b'}\exp(\langle u_i, d_{b'}\rangle/\tau)}
]
where `tau = tau_soft_assign` is constant.

Certificate:

* directional_score = mean responsibility entropy (continuous)

ExpectedEffect:

* `objective_name="predicted_assignment_entropy"`
* `predicted=directional_score`

### 5.5 ApproxOp: `ScanBinMomentMatch`

Always compute:

* `N[b] = Σ_i w_i r_{i,b}`
* `p_bar[b] = (Σ_i w_i r_{i,b} p_mean_i) * InvMass(N[b])`
* `Sigma_p[b] = within-bin scatter + Σ_i w_i r_{i,b} p_cov_i`, then `DomainProjectionPSD` always applied to each `Sigma_p[b]`

Certificate includes:

* `support.ess_total` always
* `support.support_frac` based on effective mass distribution (continuous)
* `psd_projection_delta` aggregated

ExpectedEffect:

* `objective_name="predicted_scatter_trace"`
* `predicted=Σ_b N[b] trace(Sigma_p[b]) / (Σ_b N[b] + eps_mass)`

### 5.6 ApproxOp: `KappaFromResultant:v2_single_formula`

No piecewise, no iteration.

Input:

* `Rbar_raw = ||S_dir|| * InvMass(N_dir)`

Always compute:

* `Rbar = Clamp(Rbar_raw, 0, 1 - eps_r)` with fixed `eps_r`
* `den = 1 - Rbar^2 + eps_den` with fixed `eps_den`
* `kappa = Rbar * (3 - Rbar^2) / den`

Certificate:

* clamp_delta recorded in influence fields
* `frobenius_applied` based on clamp_delta + denom regularization magnitude

ExpectedEffect:

* `objective_name="predicted_kappa_magnitude"`
* `predicted=kappa`

### 5.7 ExactOp: `WahbaSVD`

**Status (code):** Legacy. Current implementation uses **Matrix Fisher rotation evidence** (`matrix_fisher_rotation_evidence`, see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:608`). This section is retained for historical reference.

Always compute Wahba matrix:
[
M = \sum_b w_b, \mu_{map}[b]\mu_{scan}[b]^\top,\quad
w_b = N[b]\kappa_{map}[b]\kappa_{scan}[b]
]
No bin skipping; if `N[b]=0`, contribution is exactly zero.

Output `R_hat` from SVD.

Certificate:

* conditioning based on singular values

ExpectedEffect:

* `objective_name="predicted_wahba_rank_proxy"`
* `predicted=σ_2/σ_1` (continuous)

### 5.8 ExactOp: `TranslationWLS`

**Status (code):** Legacy. Current implementation uses **Planar Translation Evidence** with self‑adaptive z precision (`planar_translation_evidence`, see `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py:633`). This section is retained for historical reference.

Always compute per-bin:

* `Sigma_b_raw = Sigma_c_map[b] + R Sigma_p[b] R^T + Sigma_meas`
* `Sigma_b = DomainProjectionPSD(Sigma_b_raw).M_psd` always
* WLS normal equations accumulated over all bins with weight `w_b` (zero weight bins contribute nothing)
* solve for `t_hat` by SPDCholeskySolveLifted on `L_tt` (always lifted)

Outputs:

* `t_hat`, `L_tt`

Certificate:

* aggregated PSD projection deltas and conditioning

ExpectedEffect:

* `objective_name="predicted_translation_info_trace"`
* `predicted=trace(L_tt)`

### 5.9 ApproxOp: `LidarQuadraticEvidence`

**Status (code):** Legacy. Current implementation builds LiDAR evidence from **Matrix Fisher rotation + planar translation** (`build_combined_lidar_evidence_22d`) and does **not** use `ut_cache`.

Produces `BeliefGaussianInfo evidence` on full 22D tangent at fixed cost, reusing `ut_cache`.

Inputs:

* `belief_pred`
* `scan_bins`, `map_bins`
* `R_hat`, `t_hat`
* `ut_cache` from DeskewUTMomentMatch
* constants `c_dt`, `c_ex` from manifest

Outputs:

* `evidence` with same chart/anchor/z_lin
* `L_lidar` PSD (DomainProjectionPSD always)
* `h_lidar`

Branch-free coupling rule:

* define continuous excitation scales:
  [
  s_{dt} = \frac{dt_effect}{dt_effect + c_{dt}},\quad
  s_{ex} = \frac{extrinsic_effect}{extrinsic_effect + c_{ex}}
  ]
* `dt_scale = s_dt`, `extrinsic_scale = s_ex` recorded always
* apply scales by multiplication (always):

  * blocks involving index 15 multiplied by `s_dt`
  * blocks involving indices 16..21 multiplied by `s_ex`

Quadratic construction (fixed cost, deterministic):

1. Compute target pose increment `δξ_pose*` from `(R_hat, t_hat)` in right perturbation at `z_lin`.
2. Build `δz*` by:

   * pose slice = `δξ_pose*`
   * all other slices start at 0
   * compute continuous least-squares coupling for `[δΔt, δξLI]` using fixed design derived from `ut_cache`:

     * form `J_u` (pose residual sensitivity to those slices) by closed-form regression over all UT samples (one normal equation solve with lift)
     * solve `δu* = (J_u^T J_u + eps_lift I)^{-1} J_u^T δξ_pose*`
     * insert δu* into indices 15 and 16..21
3. Build `L_lidar_raw` from UT regression with fixed feature map:

   * Use all UT deltas `δz^(s)` to form a fixed quadratic feature vector φ(δz) containing:

     * all 22 linear terms
     * all 253 symmetric quadratic terms (upper triangular of δzδz^T)
   * Solve normal equations once (lifted) to get quadratic coefficients, assemble symmetric `L_lidar_raw`
4. `L_lidar = DomainProjectionPSD(L_lidar_raw).M_psd` always
5. Apply excitation scaling to relevant blocks (always)
6. Set `h_lidar = L_lidar @ δz*`

Certificate:

* mismatch proxies computed continuously from residual surrogates in UT regression
* excitation, dt_scale, extrinsic_scale always filled
* psd_projection_delta always filled

ExpectedEffect:

* `objective_name="predicted_quadratic_nll_decrease"`
* `predicted=0.5 * δz*^T L_lidar δz*`

### 5.10 ApproxOp: `FusionScaleFromCertificates`

Always computes `alpha` (trust) as a continuous function.

Inputs:

* `cert_evidence`, `cert_belief`
* constants `alpha_min`, `alpha_max`, `c0`, `kappa_scale`

Always compute:
[
s = \exp(-nll_per_ess)\cdot \frac{1}{1 + cond/c0}
]
[
\alpha = Clamp(kappa_scale \cdot s,\ alpha_min,\ alpha_max)
]
Record `trust_alpha = alpha` always.

ExpectedEffect:

* `objective_name="predicted_alpha"`
* `predicted=alpha`

### 5.11 ExactOp: `InfoFusionAdditive`

Always compute:

* `L_post_raw = L_pred + alpha * L_evidence`
* `h_post = h_pred + alpha * h_evidence`
* `L_post = DomainProjectionPSD(L_post_raw).M_psd` always (projection magnitude recorded)
* return belief_post

ExpectedEffect:

* `objective_name="predicted_info_trace_increase"`
* `predicted=trace(alpha * L_evidence)`

### 5.12 ApproxOp: `PoseUpdateFrobeniusRecompose`

Always recomposes with a continuous Frobenius strength.

Inputs:

* `belief_post`
* `frobenius_strength = min(1, total_trigger_magnitude / (total_trigger_magnitude + c_frob))`
  where `total_trigger_magnitude` is the sum of `psd_projection_delta`, clamp deltas, mass eps ratios, etc., and `c_frob` is constant.

Always:

1. Solve `δz_map = (L + eps_lift I)^{-1} h`
2. Apply right perturbation update to pose using BCH3 correction blended by `frobenius_strength`:

   * compute BCH3 corrected increment `δξ_BCH3`
   * apply `δξ_apply = (1 - frobenius_strength) * δξ_linear + frobenius_strength * δξ_BCH3`
3. Euclidean slices add: `u += δu` (always)

Certificate:

* `anchor_drift_rho` unaffected here
* record `frobenius_strength` as continuous effect magnitude

ExpectedEffect:

* `objective_name="predicted_step_norm"`
* `predicted=||δz_map||`

### 5.13 ApproxOp: `PoseCovInflationPushforward`

Always updates map increments with continuous inflation computed from pose covariance.

Inputs:

* belief_post (convert to covariance once with lifted solves)
* scan bins

Always:

* inflate scan covariances by first-order pose covariance contribution
* update `ΔMapBinStats` additively
* apply DomainProjectionPSD to any derived covariance used internally (always)

ExpectedEffect:

* `objective_name="predicted_map_update_norm"`
* `predicted=norm(ΔMapBinStats)` (fixed norm)

### 5.14 ApproxOp: `AnchorDriftUpdate` (Continuous local chart maintenance)

Replaces threshold “AnchorPromote”. Always runs, no discrete switching.

Inputs:

* `belief_post`
* constants: `m0`, `a`, `c0`, `b`, `s0`, `d` (manifest)
* uses current mean increment norm and cert fields

Always compute mean pose increment norm:

* `mu = (L + eps_lift I)^{-1} h`
* `pose_norm = ||mu[0:6]||`

Compute continuous reanchor strength:
[
\rho = \sigma(a(pose_norm - m0)) \cdot \sigma(b(\log(cond) - \log(c0))) \cdot \sigma(d(s0 - support_frac))
]
with `σ(x)=1/(1+exp(-x))`.

Always update anchor on SE(3):

* `X_mean = X_anchor · Exp(mu_pose)`
* `Δ = Log(X_anchor^{-1} X_mean)`
* `X_anchor_next = X_anchor · Exp( ρ * Δ )`

Always transport belief to the new anchor using first-order adjoint pushforward (no gating):

* update `X_anchor` to `X_anchor_next`
* set `z_lin` to zero vector
* transport `(L,h)` accordingly with a fixed first-order `ChartTransportRightSE3Continuous` step (internal to this op)

Certificate:

* `anchor_drift_rho = ρ`
* expected effect:

  * `objective_name="predicted_linearization_error_reduction_proxy"`
  * `predicted=ρ * pose_norm`

### 5.15 ApproxOp: `HypothesisBarycenterProjection`

Always produces a single belief for publishing.

Inputs:

* hypotheses `{(w_j, belief_j)}`

Always enforce weight floor continuously:

* `w'_j = max(w_j, HYP_WEIGHT_FLOOR)` then renormalize
* record floor adjustment magnitude in certificate

Always barycenter in info form:

* `L_out_raw = Σ w'_j L_j`
* `h_out = Σ w'_j h_j`
* `L_out = DomainProjectionPSD(L_out_raw).M_psd` always

ExpectedEffect:

* `objective_name="predicted_projection_spread_proxy"`
* `predicted` computed as weighted variance of mean increments (fixed formula with InvMass)

---

## 6) Runtime Manifest (Required)

```text
RuntimeManifest:
  chart_id: "GC-RIGHT-01"

  D_Z: 22
  D_DESKEW: 22
  K_HYP: 4
  HYP_WEIGHT_FLOOR: 0.0025
  B_BINS: 48
  T_SLICES: 5
  SIGMA_POINTS: 45
  N_POINTS_CAP: 8192

  tau_soft_assign: float

  eps_psd: 1e-12
  eps_lift: float
  eps_mass: float
  eps_r: float
  eps_den: float

  alpha_min: float
  alpha_max: float
  kappa_scale: float
  c0_cond: float

  c_dt: float
  c_ex: float
  c_frob: float

  anchor_drift_params:
    m0: float
    a: float
    c0: float
    b: float
    s0: float
    d: float

  backends: dict[str, str]      # explicit and singular (no fallback)
```

---

## 7) Deterministic Per-Scan Execution Order (Per Hypothesis)

1. `PointBudgetResample`
2. `PredictDiffusion`
3. `DeskewConstantTwist` (IMU preintegration over scan window)
4. `BinSoftAssign`
5. `ScanBinMomentMatch`
6. `KappaFromResultant:v2_single_formula` (map and scan)
7. `MatrixFisherRotationEvidence`
8. `PlanarTranslationEvidence`
9. `LidarEvidence` (Matrix Fisher + planar translation)
10. `FusionScaleFromCertificates` (alpha)
11. `InfoFusionAdditive`
12. `PoseUpdateFrobeniusRecompose` (continuous Frobenius strength)
13. `PoseCovInflationPushforward`
14. `AnchorDriftUpdate` (continuous reanchoring; always)

After all hypotheses:
15. `HypothesisBarycenterProjection`

All steps run every time; influence may go to ~0 smoothly. No gates.

---

## 8) How This Solves the Original Backend Audit Issues (by construction)

1. Delta accumulation drift: no frame-to-frame delta integration updates; effects enter via fixed operators with continuous scaling.
2. Double counting: evidence enters exactly once through additive info fusion; no separate prior inflation with the same covariance.
3. Non-SPD joint: no block joint construction; PSD projection is a declared ApproxOp always applied; solves always lifted.
4. `R_imu` unused: IMU uncertainty enters deskew UT and induced point covariances, and therefore the quadratic evidence.
5. Round-trip instability: conversions are explicit, fixed, and accounted for with projection deltas and conditioning.
6. JAX NaN propagation: DomainProjectionPSD + lifted solve defines the domain; NaNs are prevented by construction and logged via projection deltas.
7. Unbounded growth: prediction is dt-scaled diffusion; no unscaled accumulation.
8. Residual accumulation: process noise adaptation is required to be forgetful; all “mass eps” and trust effects are continuous and logged.
9. Framework mix: moment matching is confined to deskew/binning operators; fusion is pure info addition; no conflation of posterior spread with measurement noise.

---

 Golden Child SLAM v2 — Adaptive Process Noise via Inverse-Wishart Conjugacy

**Addendum to**: Golden Child SLAM v2 Strict Interface Spec (Branch-Free, Fixed-Cost, Local-Chart)  
**Version**: 2026-01-24  
**Status**: Proposed Extension

---

## 1. Motivation and Theoretical Foundation

### 1.1 The Problem with Static Q

The current spec (§5.2 PredictDiffusion) takes Q as a fixed matrix from the manifest:

```
Input:
  - Q: (22, 22) from process noise module  ← unspecified
```

This is problematic because:
1. **Mismatch accumulates**: If true process noise differs from assumed Q, the filter becomes inconsistent
2. **Environment-dependent**: Noise characteristics change (smooth floor vs rough terrain, slow vs fast motion)
3. **Sensor-dependent**: Different IMUs have different noise profiles
4. **Violates information-geometric principles**: We're treating a *distribution over covariances* as a point estimate

### 1.2 Why Inverse-Wishart?

The Inverse-Wishart distribution is the **conjugate prior** for the covariance matrix of a multivariate Gaussian. This means:

- **Prior**: Q ~ InvWishart(Ψ, ν)
- **Likelihood**: Innovation residuals r ~ N(0, Q)
- **Posterior**: Q | {r_t} ~ InvWishart(Ψ', ν')

The posterior update is **closed-form** and **additive in natural parameters** — exactly matching your Frobenius-Legendre framework where fusion is barycenter computation via sufficient statistic accumulation.

### 1.3 Connection to Your Theoretical Framework

From your project documents:

> "The space of all covariance matrices forms a convex cone of positive semidefinite matrices. This space is naturally endowed with a Wishart structure..." (Quantum Geometry Insights)

> "Closed-form coverage. In Gaussian information and Dirichlet pseudo-count parameterizations, [barycenter] yields closed-form fusion" (Technical Implementation Details)

The Inverse-Wishart parameterization slots directly into this:
- Natural parameters: η₁ = -(ν + p + 1)/2, η₂ = -½Ψ
- Sufficient statistics: T(Q) = [log|Q|, Q⁻¹]
- Conjugate update is additive in (Ψ, ν)

---

## 2. State Representation

### 2.1 Block-Diagonal Structure

The 22×22 process noise Q is **not** a dense matrix — it has block-diagonal structure reflecting the independence of different state components:

```
Q = diag(Q_rot, Q_trans, Q_vel, Q_bg, Q_ba, Q_dt, Q_ex)
```

| Block | Symbol | Dimension | Physical Meaning |
|-------|--------|-----------|------------------|
| SO(3) rotation | Q_rot | 3×3 | Orientation random walk |
| Translation | Q_trans | 3×3 | Position random walk |
| Velocity | Q_vel | 3×3 | Velocity diffusion |
| Gyro bias | Q_bg | 3×3 | Gyro bias drift |
| Accel bias | Q_ba | 3×3 | Accel bias drift |
| Time offset | Q_dt | 1×1 | Time sync drift |
| Extrinsic | Q_ex | 6×6 | LiDAR-IMU calibration drift |

**Key insight**: We maintain **separate Inverse-Wishart distributions** for each block, exploiting the monoidal structure from your Wishart geometry paper:

> "For Wp₁(n₁, Σ₁) and Wp₂(n₂, Σ₂), the tensor product is Wp₁+p₂(n₁+n₂, Σ₁⊕Σ₂) where Σ₁⊕Σ₂ denotes the block-diagonal matrix"

### 2.2 Per-Block Inverse-Wishart State

```python
@dataclass
class InvWishartState:
    """Inverse-Wishart distribution state for one covariance block"""
    p: int                      # dimension of this block
    nu: float                   # degrees of freedom (ν > p + 1 for mean to exist)
    Psi: jnp.ndarray           # scale matrix (p × p), PSD
    
    # Derived (computed on demand, not stored)
    # Mean: E[Q] = Ψ / (ν - p - 1) for ν > p + 1
    # Mode: Ψ / (ν + p + 1)
```

### 2.3 Full Process Noise State

```python
@dataclass
class ProcessNoiseState:
    """Collection of Inverse-Wishart states for all blocks"""
    chart_id: str               # must be "GC-RIGHT-01"
    
    # Per-block states (7 blocks total)
    iw_rot: InvWishartState     # 3×3
    iw_trans: InvWishartState   # 3×3
    iw_vel: InvWishartState     # 3×3
    iw_bg: InvWishartState      # 3×3
    iw_ba: InvWishartState      # 3×3
    iw_dt: InvWishartState      # 1×1 (scalar, degenerates to Inverse-Gamma)
    iw_ex: InvWishartState      # 6×6
    
    # Bookkeeping
    n_updates: int              # total innovation updates received
    stamp_sec: float            # timestamp of last update
```

---

## 3. Operator Specifications

All operators follow the branch-free, certificate-returning pattern.

### 3.1 ExactOp: InvWishartMean

Computes the posterior mean (point estimate) of Q from the Inverse-Wishart state.

**Inputs**:
```
iw: InvWishartState
eps_nu: float              # regularization for degrees of freedom (manifest constant)
```

**Outputs**:
```
Q_mean: (p, p)             # E[Q] = Ψ / (ν - p - 1)
cert: CertBundle
expected_effect: ExpectedEffect
```

**Computation** (always, no branching):
```python
def inv_wishart_mean(iw: InvWishartState, eps_nu: float):
    p = iw.p
    # Regularize to ensure ν > p + 1 (mean exists)
    nu_effective = jnp.maximum(iw.nu, p + 1 + eps_nu)
    denom = nu_effective - p - 1
    
    Q_mean = iw.Psi / denom
    
    # Ensure PSD (DomainProjectionPSD always applied)
    Q_mean, psd_cert = domain_projection_psd(Q_mean, eps_psd)
    
    cert = CertBundle(
        exact=True,
        nu_effective=nu_effective,
        nu_regularization_applied=(iw.nu < p + 1 + eps_nu),
        conditioning=psd_cert,
        ...
    )
    
    effect = ExpectedEffect(
        objective_name="trace_Q_mean",
        predicted=jnp.trace(Q_mean)
    )
    
    return Q_mean, cert, effect
```

**Certificate fields**:
- `nu_effective`: actual degrees of freedom used
- `nu_regularization_applied`: True if ν was boosted
- `conditioning`: eigenvalue stats of Q_mean

---

### 3.2 ExactOp: InvWishartMode

Computes the posterior mode (MAP estimate) — often preferred for robustness.

**Computation**:
```python
Q_mode = iw.Psi / (nu_effective + p + 1)
```

The mode is always smaller than the mean, providing a more conservative estimate.

---

### 3.3 ApproxOp: InnovationResidualCompute

Computes innovation residuals from the filter update, which drive the Q adaptation.

**Context**: After InfoFusionAdditive (Step 11), we have:
- `belief_pred`: prior belief before incorporating LiDAR evidence
- `belief_post`: posterior belief after fusion

The innovation in information form is the difference in information vectors, but we need the *residual in state space* for Q estimation.

**Inputs**:
```
belief_pred: BeliefGaussianInfo
belief_post: BeliefGaussianInfo
manifest: RuntimeManifest
```

**Outputs**:
```
residuals: InnovationResiduals
cert: CertBundle
expected_effect: ExpectedEffect
```

**Computation** (always, branch-free):
```python
@dataclass
class InnovationResiduals:
    """Innovation residuals partitioned by state block"""
    r_rot: jnp.ndarray      # (3,)
    r_trans: jnp.ndarray    # (3,)
    r_vel: jnp.ndarray      # (3,)
    r_bg: jnp.ndarray       # (3,)
    r_ba: jnp.ndarray       # (3,)
    r_dt: jnp.ndarray       # (1,)
    r_ex: jnp.ndarray       # (6,)
    
    weights: jnp.ndarray    # (7,) per-block reliability weights
    dt_sec: float           # time interval for scaling

def compute_innovation_residuals(belief_pred, belief_post, manifest):
    # Solve for mean increments
    mu_pred, _ = spd_cholesky_solve_lifted(belief_pred.L, belief_pred.h, manifest.eps_lift)
    mu_post, _ = spd_cholesky_solve_lifted(belief_post.L, belief_post.h, manifest.eps_lift)
    
    # Innovation = posterior - prior mean
    delta_mu = mu_post - mu_pred  # (22,)
    
    # Partition into blocks (fixed indices from spec §1.1)
    r_rot = delta_mu[0:3]
    r_trans = delta_mu[3:6]
    r_vel = delta_mu[6:9]
    r_bg = delta_mu[9:12]
    r_ba = delta_mu[12:15]
    r_dt = delta_mu[15:16]
    r_ex = delta_mu[16:22]
    
    # Compute reliability weights from posterior information gain
    # Higher info gain → more reliable residual → higher weight
    info_gain = jnp.diag(belief_post.L) - jnp.diag(belief_pred.L)  # (22,)
    
    # Aggregate per block (sum of diagonal info gains)
    w_rot = jnp.sum(jnp.maximum(info_gain[0:3], 0))
    w_trans = jnp.sum(jnp.maximum(info_gain[3:6], 0))
    w_vel = jnp.sum(jnp.maximum(info_gain[6:9], 0))
    w_bg = jnp.sum(jnp.maximum(info_gain[9:12], 0))
    w_ba = jnp.sum(jnp.maximum(info_gain[12:15], 0))
    w_dt = jnp.maximum(info_gain[15], 0)
    w_ex = jnp.sum(jnp.maximum(info_gain[16:22], 0))
    
    weights = jnp.array([w_rot, w_trans, w_vel, w_bg, w_ba, w_dt, w_ex])
    # Normalize with InvMass pattern
    weights = weights / (jnp.sum(weights) + manifest.eps_mass)
    
    return InnovationResiduals(
        r_rot=r_rot, r_trans=r_trans, r_vel=r_vel,
        r_bg=r_bg, r_ba=r_ba, r_dt=r_dt, r_ex=r_ex,
        weights=weights,
        dt_sec=belief_post.stamp_sec - belief_pred.stamp_sec
    ), cert, effect
```

---

### 3.4 ApproxOp: InvWishartConjugateUpdate

The core Bayesian update. This is where the Inverse-Wishart conjugacy shines.

**Mathematical Background**:

For a single observation r ~ N(0, Q) with Q ~ InvWishart(Ψ, ν):

```
Posterior: Q | r ~ InvWishart(Ψ + r·rᵀ, ν + 1)
```

This is exact conjugate update — no approximation needed for in-family.

**Inputs**:
```
iw_prev: InvWishartState          # prior
residual: jnp.ndarray             # (p,) innovation residual for this block
weight: float                     # reliability weight in [0, 1]
rho: float                        # retention/forgetting factor in (0, 1]
dt_sec: float                     # time interval
manifest: RuntimeManifest
```

**Outputs**:
```
iw_post: InvWishartState
cert: CertBundle
expected_effect: ExpectedEffect
```

**Computation** (always, branch-free):
```python
def inv_wishart_conjugate_update(iw_prev, residual, weight, rho, dt_sec, manifest):
    p = iw_prev.p
    
    # === FORGETFUL RETENTION (continuous, no branching) ===
    # Scale down prior contribution to prevent unbounded accumulation
    # rho < 1 makes old observations decay exponentially
    # rho = 1 would accumulate forever (not recommended)
    
    # Effective prior after forgetting
    nu_retained = rho * iw_prev.nu
    Psi_retained = rho * iw_prev.Psi
    
    # === CONJUGATE UPDATE ===
    # Outer product of residual (the sufficient statistic for covariance)
    rrt = jnp.outer(residual, residual)  # (p, p)
    
    # Scale by:
    # - weight: reliability of this residual
    # - dt_sec: longer intervals → more diffusion expected
    # We observe r² ~ Q·dt, so effective observation is r·rᵀ / dt
    dt_safe = jnp.maximum(dt_sec, manifest.eps_mass)
    scaled_rrt = weight * rrt / dt_safe
    
    # Posterior parameters
    Psi_post_raw = Psi_retained + scaled_rrt
    nu_post = nu_retained + weight  # fractional count update
    
    # === DOMAIN PROJECTION (always applied) ===
    Psi_post, psd_cert = domain_projection_psd(Psi_post_raw, manifest.eps_psd)
    
    # === DEGREE OF FREEDOM BOUNDS ===
    # Ensure nu stays in valid range [nu_min, nu_max]
    # nu_min > p + 1 ensures mean exists
    # nu_max prevents over-concentration
    nu_post = jnp.clip(nu_post, manifest.nu_min, manifest.nu_max)
    
    iw_post = InvWishartState(p=p, nu=nu_post, Psi=Psi_post)
    
    # === CERTIFICATE ===
    cert = CertBundle(
        exact=False,  # ApproxOp due to forgetting and scaling
        approximation_triggers=["forgetful_retention", "weighted_update", "dt_scaling"],
        
        # Effective sample size (how many "observations" worth of data)
        support=SupportCert(
            ess_total=nu_post - p - 1,  # effective observations
            support_frac=weight,
        ),
        
        # How much the prior was forgotten
        influence=InfluenceCert(
            retention_rho=rho,
            prior_mass_retained=rho * iw_prev.nu / (nu_post + manifest.eps_mass),
            observation_contribution=weight / (nu_post + manifest.eps_mass),
        ),
        
        # PSD projection stats
        psd_projection_delta=psd_cert['psd_projection_delta'],
        conditioning=psd_cert,
    )
    
    # === EXPECTED EFFECT ===
    # Predict how Q_mean will change
    Q_mean_prev = iw_prev.Psi / jnp.maximum(iw_prev.nu - p - 1, manifest.eps_nu)
    Q_mean_post = Psi_post / jnp.maximum(nu_post - p - 1, manifest.eps_nu)
    
    effect = ExpectedEffect(
        objective_name="trace_Q_change",
        predicted=jnp.trace(Q_mean_post) - jnp.trace(Q_mean_prev)
    )
    
    return iw_post, cert, effect
```

---

### 3.5 ApproxOp: ProcessNoiseFullUpdate

Orchestrates the update across all 7 blocks.

**Inputs**:
```
pn_state: ProcessNoiseState
residuals: InnovationResiduals
rho_config: RetentionConfig       # per-block retention factors
manifest: RuntimeManifest
```

**Outputs**:
```
pn_state_new: ProcessNoiseState
cert: CertBundle
expected_effect: ExpectedEffect
```

**Computation**:
```python
@dataclass
class RetentionConfig:
    """Per-block forgetting factors — slower drift states forget slower"""
    rho_rot: float = 0.995        # rotation drifts slowly
    rho_trans: float = 0.99       # translation moderate
    rho_vel: float = 0.95         # velocity changes faster
    rho_bg: float = 0.999         # bias drifts very slowly
    rho_ba: float = 0.999         # bias drifts very slowly
    rho_dt: float = 0.9999        # time offset very stable
    rho_ex: float = 0.9999        # extrinsic very stable

def process_noise_full_update(pn_state, residuals, rho_config, manifest):
    dt = residuals.dt_sec
    w = residuals.weights
    
    # Update each block (all always execute, no branching)
    iw_rot_new, cert_rot, _ = inv_wishart_conjugate_update(
        pn_state.iw_rot, residuals.r_rot, w[0], rho_config.rho_rot, dt, manifest)
    
    iw_trans_new, cert_trans, _ = inv_wishart_conjugate_update(
        pn_state.iw_trans, residuals.r_trans, w[1], rho_config.rho_trans, dt, manifest)
    
    iw_vel_new, cert_vel, _ = inv_wishart_conjugate_update(
        pn_state.iw_vel, residuals.r_vel, w[2], rho_config.rho_vel, dt, manifest)
    
    iw_bg_new, cert_bg, _ = inv_wishart_conjugate_update(
        pn_state.iw_bg, residuals.r_bg, w[3], rho_config.rho_bg, dt, manifest)
    
    iw_ba_new, cert_ba, _ = inv_wishart_conjugate_update(
        pn_state.iw_ba, residuals.r_ba, w[4], rho_config.rho_ba, dt, manifest)
    
    iw_dt_new, cert_dt, _ = inv_wishart_conjugate_update(
        pn_state.iw_dt, residuals.r_dt, w[5], rho_config.rho_dt, dt, manifest)
    
    iw_ex_new, cert_ex, _ = inv_wishart_conjugate_update(
        pn_state.iw_ex, residuals.r_ex, w[6], rho_config.rho_ex, dt, manifest)
    
    pn_state_new = ProcessNoiseState(
        chart_id=pn_state.chart_id,
        iw_rot=iw_rot_new, iw_trans=iw_trans_new, iw_vel=iw_vel_new,
        iw_bg=iw_bg_new, iw_ba=iw_ba_new, iw_dt=iw_dt_new, iw_ex=iw_ex_new,
        n_updates=pn_state.n_updates + 1,
        stamp_sec=residuals.stamp_sec
    )
    
    # Aggregate certificates
    cert = aggregate_certs([cert_rot, cert_trans, cert_vel, 
                           cert_bg, cert_ba, cert_dt, cert_ex])
    
    return pn_state_new, cert, effect
```

---

### 3.6 ExactOp: AssembleQMatrix

Assembles the full 22×22 Q matrix from block means.

**Inputs**:
```
pn_state: ProcessNoiseState
manifest: RuntimeManifest
```

**Outputs**:
```
Q: (22, 22)                       # block-diagonal process noise
cert: CertBundle
expected_effect: ExpectedEffect
```

**Computation**:
```python
def assemble_q_matrix(pn_state, manifest):
    # Get mean of each block
    Q_rot, _, _ = inv_wishart_mean(pn_state.iw_rot, manifest.eps_nu)
    Q_trans, _, _ = inv_wishart_mean(pn_state.iw_trans, manifest.eps_nu)
    Q_vel, _, _ = inv_wishart_mean(pn_state.iw_vel, manifest.eps_nu)
    Q_bg, _, _ = inv_wishart_mean(pn_state.iw_bg, manifest.eps_nu)
    Q_ba, _, _ = inv_wishart_mean(pn_state.iw_ba, manifest.eps_nu)
    Q_dt, _, _ = inv_wishart_mean(pn_state.iw_dt, manifest.eps_nu)
    Q_ex, _, _ = inv_wishart_mean(pn_state.iw_ex, manifest.eps_nu)
    
    # Assemble block-diagonal (using jax.scipy.linalg.block_diag)
    Q = jnp.zeros((22, 22))
    Q = Q.at[0:3, 0:3].set(Q_rot)
    Q = Q.at[3:6, 3:6].set(Q_trans)
    Q = Q.at[6:9, 6:9].set(Q_vel)
    Q = Q.at[9:12, 9:12].set(Q_bg)
    Q = Q.at[12:15, 12:15].set(Q_ba)
    Q = Q.at[15:16, 15:16].set(Q_dt)
    Q = Q.at[16:22, 16:22].set(Q_ex)
    
    # Final PSD projection (always)
    Q, psd_cert = domain_projection_psd(Q, manifest.eps_psd)
    
    return Q, cert, effect
```

---

## 4. Integration with Main Pipeline

### 4.1 Modified Step 2: PredictDiffusion

The existing Step 2 now takes Q from the ProcessNoiseState:

```python
def predict_diffusion_adaptive(belief_prev, pn_state, dt_sec, manifest):
    # Get current Q estimate
    Q, q_cert, _ = assemble_q_matrix(pn_state, manifest)
    
    # Rest of prediction unchanged
    # ... (existing §5.2 logic)
    
    return belief_pred, cert, effect
```

### 4.2 New Step 11.5: ProcessNoiseUpdate

Insert after InfoFusionAdditive (Step 11), before PoseUpdateFrobeniusRecompose (Step 12):

```
Pipeline Order (Updated):
  ...
  10. FusionScaleFromCertificates
  11. InfoFusionAdditive
  11.5 ProcessNoiseUpdate  ← NEW
  12. PoseUpdateFrobeniusRecompose
  ...
```

**ApproxOp: ProcessNoiseUpdate**

```python
def process_noise_update(belief_pred, belief_post, pn_state, rho_config, manifest):
    # Compute innovation residuals
    residuals, res_cert, _ = compute_innovation_residuals(
        belief_pred, belief_post, manifest)
    
    # Update Inverse-Wishart states
    pn_state_new, pn_cert, effect = process_noise_full_update(
        pn_state, residuals, rho_config, manifest)
    
    # Aggregate certificate
    cert = merge_certs(res_cert, pn_cert)
    
    return pn_state_new, cert, effect
```

---

## 5. Manifest Extensions

Add to RuntimeManifest (§6):

```yaml
# Process noise adaptation
adaptive_process_noise:
  enabled: true
  
  # Degrees of freedom bounds
  nu_min: 5.0           # minimum DoF (must be > p + 1 for largest block)
  nu_max: 1000.0        # maximum DoF (prevents over-concentration)
  eps_nu: 1e-6          # regularization for mean computation
  
  # Per-block retention factors (forgetting)
  retention:
    rho_rot: 0.995
    rho_trans: 0.99
    rho_vel: 0.95
    rho_bg: 0.999
    rho_ba: 0.999
    rho_dt: 0.9999
    rho_ex: 0.9999
  
  # Initial prior (weak prior, high uncertainty)
  initial_prior:
    nu_init: 10.0       # low DoF = high uncertainty
    # Psi_init = nu_init * Q_init where Q_init is from manifest
    Q_init_rot: [0.001, 0.001, 0.001]      # diagonal
    Q_init_trans: [0.01, 0.01, 0.01]
    Q_init_vel: [0.1, 0.1, 0.1]
    Q_init_bg: [1e-5, 1e-5, 1e-5]
    Q_init_ba: [1e-4, 1e-4, 1e-4]
    Q_init_dt: [1e-6]
    Q_init_ex: [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
```

---

## 6. Certificate Extensions

### 6.1 ProcessNoiseCertBundle

```python
@dataclass
class ProcessNoiseCertBundle:
    """Certificate for process noise module"""
    
    # Per-block effective sample sizes
    ess_rot: float
    ess_trans: float
    ess_vel: float
    ess_bg: float
    ess_ba: float
    ess_dt: float
    ess_ex: float
    
    # Per-block Q trace (magnitude of noise)
    trace_Q_rot: float
    trace_Q_trans: float
    trace_Q_vel: float
    trace_Q_bg: float
    trace_Q_ba: float
    trace_Q_dt: float
    trace_Q_ex: float
    
    # Aggregate stats
    total_ess: float                 # sum of block ESS
    total_trace_Q: float             # trace of full Q
    max_condition_number: float      # worst conditioning across blocks
    
    # Convergence indicators
    q_change_norm: float             # ||Q_new - Q_old||_F
    relative_change: float           # q_change_norm / ||Q_old||_F
```

### 6.2 Logging Requirements

Log per scan:
- `pn.ess_total`: total effective sample size
- `pn.trace_Q`: trace of assembled Q
- `pn.max_cond`: worst block conditioning  
- `pn.relative_change`: how much Q changed this update

Anomaly thresholds:
| Metric | Warning | Error |
|--------|---------|-------|
| `ess_total` | < 10 | < 5 |
| `trace_Q` (any block) | > 10× initial | > 100× initial |
| `max_cond` | > 1e6 | > 1e10 |
| `relative_change` | > 0.1 (10%) | > 0.5 (50%) |

---

## 7. Theoretical Properties

### 7.1 Conjugacy Proof Sketch

For Q ~ InvWishart(Ψ, ν) and r ~ N(0, Q):

```
p(Q | r) ∝ p(r | Q) · p(Q)
        ∝ |Q|^(-1/2) exp(-½ rᵀQ⁻¹r) · |Q|^(-(ν+p+1)/2) exp(-½ tr(ΨQ⁻¹))
        ∝ |Q|^(-(ν+p+2)/2) exp(-½ tr((Ψ + rrᵀ)Q⁻¹))
        = InvWishart(Ψ + rrᵀ, ν + 1)
```

This is **exact** — no approximation within the Inverse-Wishart family.

### 7.2 Forgetful Retention Interpretation

The retention factor ρ < 1 implements exponential forgetting:

- Effective window: ~1/(1-ρ) observations
- ρ = 0.99 → ~100 scan window
- ρ = 0.999 → ~1000 scan window

This ensures:
1. **Bounded memory**: Old observations decay, preventing unbounded accumulation
2. **Adaptivity**: System can track changing noise characteristics
3. **Robustness**: Outlier residuals eventually wash out

### 7.3 Connection to Barycenter Framework

The forgetting + update can be viewed as a **weighted barycenter** in Inverse-Wishart space:

```
θ_post = Barycenter({(θ_prior, ρ), (θ_obs, w)})
```

where θ = (ν, Ψ) are the natural parameters. This fits directly into your Frobenius-Legendre framework.

---

## 8. JAX Implementation Notes

### 8.1 Multivariate Gamma Function

JAX provides `jax.scipy.special.multigammaln` for the log multivariate gamma:

```python
def log_mv_gamma(a, p):
    """Log of multivariate gamma function Γ_p(a)"""
    return jax.scipy.special.multigammaln(a, p)
```

Needed for normalizing constants if computing likelihoods (not needed for conjugate updates).

### 8.2 Numerical Stability

Key considerations:
1. Always use `domain_projection_psd` on Ψ after updates
2. Clip ν to [nu_min, nu_max] to prevent degenerate distributions
3. Use `InvMass` pattern for division by dt_sec
4. Block-diagonal structure means 7 small eigendecomps, not one large

### 8.3 JIT Compatibility

All operations are:
- Fixed shape (block sizes are compile-time constants)
- No data-dependent branching
- Pure functions

This makes the entire ProcessNoise module JIT-friendly.

---

## 9. Verification Checklist

### 9.1 Unit Tests

```
☐ InvWishart mean matches scipy.stats.invwishart.mean()
☐ Conjugate update produces valid InvWishart (ν > p+1, Ψ PSD)
☐ Forgetful retention decays ESS as expected
☐ Block assembly produces correct 22×22 structure
☐ Zero residual → Q unchanged (up to retention decay)
☐ Large residual → Q increases appropriately
```

### 9.2 Integration Tests

```
☐ Static scene: Q converges to stable values
☐ Sudden motion change: Q adapts within ~1/(1-ρ) scans
☐ Sensor noise increase: corresponding Q block increases
☐ Certificates log correctly
☐ Pipeline runs without NaN/Inf
```

### 9.3 Sanity Checks

```
☐ trace(Q) stays bounded (not exploding)
☐ ESS doesn't collapse to near-zero
☐ Block conditioning stays reasonable (< 1e8)
☐ Relative change stabilizes after initial transient
```

---

## 10. Summary

This specification adds **adaptive process noise estimation** to Golden Child SLAM v2 using:

1. **Inverse-Wishart conjugate priors** for each block of the process noise covariance
2. **Forgetful retention** to bound memory and enable adaptivity
3. **Block-diagonal structure** exploiting independence assumptions
4. **Branch-free, certificate-returning operators** matching the existing spec pattern
5. **Closed-form updates** — no iterative optimization required

The approach is theoretically grounded in your information-geometric framework:
- Wishart/Inverse-Wishart forms a dually-flat manifold
- Conjugate updates are barycenter computations in natural parameter space
- The block structure matches the monoidal (tensor product) decomposition

This completes the previously unspecified "process noise module" referenced in §5.2 of the main spec.

---

I) Golden Child IMU v2 — Strict First-Principles Spec (Rosbag-Constrained)
0) Non-negotiable system invariants
0.1 No gates, no heuristics, branch-free

No accept/reject logic, no thresholding, no “if accel near g then…”.

All operators are total functions; “robustness” occurs only through posterior expectations of latent variables and conjugate sufficient-statistic updates.

Any numeric stabilization is applied unconditionally (e.g., PSD projection always).

0.2 Manifold-intrinsic state geometry

Pose/orientation live on 
SE(3)
SE(3)/
SO(3)
SO(3).

Updates use 
\Exp/\Log
\Exp/\Log, group actions, and Adjoint transport.

Avoid coordinate Jacobians; only use intrinsic differentials of group actions (closed-form cross-product maps) and Riemannian Gauss-Newton.

0.3 Canonical evidence interface (project-to-Gaussian by construction)

All sensor factors (IMU, LiDAR, etc.) are represented as true likelihood factors and then projected to the canonical 22D Gaussian information object via a fixed-cost Laplace / I-projection (second-order local projection). This is the unifying principle that replaces ad hoc “quadratic regression evidence.”

This is compatible with your current 22D evidence machinery but is defined by first principles: “project a factor product to the Gaussian family at 
zlin
z
lin
	​

”.

1) Rosbag input contract (what must exist)
1.1 IMU topic(s)

The rosbag provides at least one sensor_msgs/Imu stream including:

header.stamp

header.frame_id

angular_velocity (rad/s)

linear_acceleration (in g for your current dataset; converted upstream to m/s²)

angular_velocity_covariance, linear_acceleration_covariance, orientation_covariance

Your proof-of-concept uses the calibrated IMU stream for which you have extrinsics (per your response and bag usage constraints). 

BAG_TOPICS_AND_USAGE

1.2 No dependence on /tf

Because bag reality may omit TF, the IMU spec does not require /tf or /tf_static for correctness. TF may be ingested if present, but the canonical extrinsics source is calibration (§2). 

BAG_TOPICS_AND_USAGE

2) Calibration and frames (required, not optional)
2.1 Extrinsics are calibration priors and are estimated

Let 
XLI∈SE(3)
X
LI
	​

∈SE(3) be the LiDAR→IMU extrinsic transform (pose of IMU in LiDAR frame). This is:

initialized from the provided calibration (because that’s what exists in the bag ecosystem for the chosen IMU),

treated as a state variable with a prior factor (not a fixed constant),

refined online by the global estimator.

This is “always needed” calibration, as you noted, and is the mathematically correct way to incorporate it: as a prior on a group element.

2.2 Time offset is a state variable

Maintain 
Δt
Δt (IMU↔LiDAR clock offset) as a state variable with a prior factor. Time alignment is handled continuously (no discrete matching or gating).

3) Units and covariance ingestion (strict)
3.1 Acceleration in g → m/s² conversion (upstream, deterministic)

Given bag reality, accelerometer linear acceleration is treated as “g units” and converted:

a(m/s2)=a(g)⋅g0,g0=9.80665
a
(m/s
2
)
=a
(g)
⋅g
0
	​

,g
0
	​

=9.80665

This is a deterministic preprocessing step; it is not part of inference and introduces no heuristics. 

BAG_TOPICS_AND_USAGE

3.2 Covariance fields are consumed as likelihood hyper-evidence

The sensor_msgs/Imu covariance matrices are not ignored. They are treated as observations of sensor noise (hyper-evidence) and used to initialize/update the measurement-noise distributions (§6). 

BAG_TOPICS_AND_USAGE

Branch-free sanitization rule:
All covariance matrices are passed through an unconditional SPD projection:

Σ←ΠSPD(12(Σ+Σ⊤))
Σ←Π
SPD
	​

(
2
1
	​

(Σ+Σ
⊤
))

(no “if invalid then…”). Any missing/negative entries simply become part of this projection and are recorded in certificates as projection magnitude. This is branch-free and deterministic.

4) First-principles IMU generative model (with vMF direction)

We model IMU measurements as observing the specific force arrow and angular velocity, with robustness and adaptivity arising from latent scales and inverse-Wishart noise.

4.1 State (minimal)
x(t)=(XWL(t), vW(t), bg(t), ba(t), Δt, XLI)
x(t)=(X
WL
	​

(t), v
W
	​

(t), b
g
	​

(t), b
a
	​

(t), Δt, X
LI
	​

)

XWL∈SE(3)
X
WL
	​

∈SE(3): LiDAR body pose in world

vW∈R3
v
W
	​

∈R
3

biases 
bg,ba∈R3
b
g
	​

,b
a
	​

∈R
3

Δt∈R
Δt∈R

XLI∈SE(3)
X
LI
	​

∈SE(3)

IMU pose:

XWI(t)=XWL(t) XLI
X
WI
	​

(t)=X
WL
	​

(t)X
LI
	​


and 
RWI
R
WI
	​

 is its rotation.

4.2 Gyro measurement likelihood (Euclidean, but robust/adaptive)
ωm(t)=ω(t)+bg(t)+εg(t)
ω
m
	​

(t)=ω(t)+b
g
	​

(t)+ε
g
	​

(t)
εg(t)∣Σg(t)∼N(0,Σg(t))
ε
g
	​

(t)∣Σ
g
	​

(t)∼N(0,Σ
g
	​

(t))
4.3 Accelerometer likelihood (spherical “force-arrow” model)

Define measured specific force vector (bias corrected):

fm(t)=am(t)−ba(t)
f
m
	​

(t)=a
m
	​

(t)−b
a
	​

(t)

Decompose into direction and magnitude:

x(t)=fm(t)∥fm(t)∥,r(t)=∥fm(t)∥
x(t)=
∥f
m
	​

(t)∥
f
m
	​

(t)
	​

,r(t)=∥f
m
	​

(t)∥

Define the predicted specific force direction in IMU frame as gravity direction (strictly, in the absence of modeling linear acceleration explicitly as a state):

μ(t)=RWI(t)⊤(−gW)∥gW∥∈S2
μ(t)=
∥g
W
	​

∥
R
WI
	​

(t)
⊤
(−g
W
	​

)
	​

∈S
2

where 
gW=[0,0,−g0]
g
W
	​

=[0,0,−g
0
	​

].

Directional likelihood (vMF):

p(x(t)∣x(t)∈S2,μ(t),κ(t))∝exp⁡(κ(t) μ(t)⊤x(t))
p(x(t)∣x(t)∈S
2
,μ(t),κ(t))∝exp(κ(t)μ(t)
⊤
x(t))

Magnitude likelihood (Gaussian):

p(r(t)∣ρ(t),σr2(t))=N(r(t);ρ(t),σr2(t))
p(r(t)∣ρ(t),σ
r
2
	​

(t))=N(r(t);ρ(t),σ
r
2
	​

(t))

with 
ρ(t)
ρ(t) treated as a nuisance “expected norm of specific force” parameter, learned online as a latent (see §6.3). This avoids “assume 
ρ=g
ρ=g” heuristics while remaining branch-free.

5) Robustness and adaptivity (no gates; all continuous)
5.1 Random concentration for vMF (continuous reliability, no gating)

Concentration is not fixed. Introduce a latent precision 
τκ(t)>0
τ
κ
	​

(t)>0 and set

κ(t)=κ0 E[τκ(t)]
κ(t)=κ
0
	​

E[τ
κ
	​

(t)]

with 
τκ(t)
τ
κ
	​

(t) updated from residual statistics in a conjugate scale-mixture manner (continuous, deterministic). The result: when accelerometer direction is inconsistent (e.g., heavy dynamics), the posterior drives 
E[τκ]↓
E[τ
κ
	​

]↓, smoothly making the vMF factor uninformative—without any gates.

5.2 Measurement noise as inverse-Wishart random variables (consume covariance fields)

Maintain measurement-noise distributions:

Σg∼InvWishart(Ψg,νg),Σa∼InvWishart(Ψa,νa)
Σ
g
	​

∼InvWishart(Ψ
g
	​

,ν
g
	​

),Σ
a
	​

∼InvWishart(Ψ
a
	​

,ν
a
	​

)

The IMU message covariance fields are ingested as observations that increment these sufficient statistics (branch-free SPD-projected).

Innovation residuals (post-update) further update these IW states conjugately.

This is exactly the “noise adapts to regime” mechanism you want (underwater, vibration, etc.) and it is by construction.

5.3 Process noise 
Q
Q as inverse-Wishart (adaptive dynamics; no tuning)

Maintain blockwise inverse-Wishart priors for process noise blocks (rotation, velocity, biases, time offset, extrinsics). Update them conjugately from realized process innovations. This formalizes the adaptive-
Q
Q plan in first principles (no fixed hand-tuned 
Q
Q matrix). 

gc_slam_adaptive_process_noise_…

6) Inference architecture (factor → Laplace/I-projection → Gaussian info)
6.1 Windowing (design choice; choose the principled one)

The canonical integration boundary is per LiDAR scan interval:

[tk,tk+1]
[t
k
	​

,t
k+1
	​

]

You may keep an intermediate IMUSegment topic for engineering, but the mathematical object is: “all IMU samples that support that scan interval,” with continuous time-offset warping via 
Δt
Δt. The spec does not require the segment topic because the bag already contains raw IMU; segmenting is a pipeline decision. 

BAG_TOPICS_AND_USAGE

6.2 Time-offset warp (continuous)

Given 
Δt
Δt state, the effective IMU time is 
timu=tlidar+Δt
t
imu
=t
lidar
+Δt. Membership of IMU samples in the scan interval is implemented as a smooth kernel weight derived from 
Δt
Δt’s current uncertainty (Normal CDF membership), not a hard include/exclude decision.

6.3 vMF sufficient statistics (exact, additive; no Jacobians)

For a scan window, compute weighted resultant:

S=∑iwi xi,N=∑iwi,xˉ=S∥S∥+ε
S=
i
∑
	​

w
i
	​

x
i
	​

,N=
i
∑
	​

w
i
	​

,
x
ˉ
=
∥S∥+ε
S
	​


Compute resultant length 
R=∥S∥/(N+ε)
R=∥S∥/(N+ε), then compute 
κ^
κ
^
 via your closed-form kappa-from-resultant map (the same primitive you already have). This produces a single window vMF factor with:

ηwin=κ^ xˉ
η
win
	​

=
κ
^
x
ˉ

All of this is additive natural-parameter calculus (dually flat), requiring no linearization.

6.4 Laplace / I-projection to 22D Gaussian information (fixed-cost, canonical)

For each scan interval, construct the joint negative log-likelihood over the state 
z
z in the current right-invariant chart:

ℓ(z)=ℓpreint(z)+ℓgyro(z)+ℓvMF(z)+ℓmag(z)+ℓpriors(z)
ℓ(z)=ℓ
preint
	​

(z)+ℓ
gyro
	​

(z)+ℓ
vMF
	​

(z)+ℓ
mag
	​

(z)+ℓ
priors
	​

(z)

where:

ℓpreint
ℓ
preint
	​

 is the IMU kinematics constraint (preintegration factor)

ℓgyro
ℓ
gyro
	​

 uses IW-mode 
Σg
Σ
g
	​


ℓvMF(z)=−κ^ μ(z)⊤xˉ+A(κ^)
ℓ
vMF
	​

(z)=−
κ
^
μ(z)
⊤
x
ˉ
+A(
κ
^
)

ℓmag
ℓ
mag
	​

 is the magnitude factor with learned 
ρ,σr2
ρ,σ
r
2
	​


ℓpriors
ℓ
priors
	​

 includes priors on biases, 
Δt
Δt, and extrinsics.

Then compute:

gradient 
g=∇ℓ(zlin)
g=∇ℓ(z
lin
	​

)

Hessian 
H=∇2ℓ(zlin)
H=∇
2
ℓ(z
lin
	​

)

Return Gaussian info evidence:

L=ΠSPD(12(H+H⊤))(always)
L=Π
SPD
	​

(
2
1
	​

(H+H
⊤
))(always)
h=−g
h=−g

This is the canonical projection of a factor product onto the Gaussian family at 
zlin
z
lin
	​

 (Laplace approximation / I-projection surrogate), replacing ad hoc quadratic evidence construction.

6.5 “Minimize Jacobians” implementation rule

No coordinate Jacobians of parameterizations.

Differentials are computed only as:

closed-form intrinsic action derivatives (e.g., for 
μ(R)=R⊤g^
μ(R)=R
⊤
g
^
	​

, the variation under right perturbation is 
δμ=−[μ]× δθ
δμ=−[μ]
×
	​

δθ);

group right/left Jacobians of 
\Exp/\Log
\Exp/\Log (intrinsic, closed-form);

optional automatic differentiation over these intrinsic primitives (still not “Jacobians” in the extrinsic sense; it is exact differentiation of intrinsic maps).

This is branch-free and mathematically canonical.

7) Full operator schedule per scan (deterministic; always runs)

For each LiDAR scan interval 
[tk,tk+1]
[t
k
	​

,t
k+1
	​

]:

Ingest IMU into ring buffer (raw rosbag stream).

Apply unit conversion (g→m/s²).

SPD-project and ingest IMU covariances into IW hyperstates 
(Ψg,νg)
(Ψ
g
	​

,ν
g
	​

), 
(Ψa,νa)
(Ψ
a
	​

,ν
a
	​

).

Continuous time-offset warp (use 
Δt
Δt) and compute smooth membership weights 
wi
w
i
	​

.

IMU preintegration factor on 
SE(3)
SE(3) using intrinsic 
\Exp/\Log
\Exp/\Log and IW-mode process/measurement covariances.

Compute vMF window factor via sufficient statistics 
(S,N,κ^,xˉ)
(S,N,
κ
^
,
x
ˉ
).

Compute magnitude latent updates for 
ρ,σr2
ρ,σ
r
2
	​

 by conjugate/second-order update (fixed-cost).

Assemble full IMU factor set (preint + vMF + mag + priors).

Laplace/I-projection to 22D Gaussian info: compute 
g,H
g,H, SPD-project 
H→L
H→L, form 
h=−g
h=−g.

Fuse evidence additively in information space with other sensors.

Update IW hyperstates for process noise 
Q
Q and measurement noise 
Σg,Σa
Σ
g
	​

,Σ
a
	​

 from posterior innovations (conjugate sufficient-statistic increments). 

gc_slam_adaptive_process_noise_…

No step is conditional; “influence” only changes continuously via inferred covariances and latent precisions.

8) What this spec changes relative to current Golden Child decisions

IMU is not handled as “diffusion prediction + ad hoc evidence.” It becomes true factors + canonical Gaussian projection.

vMF is not “optional.” It is the primary direction channel by construction, with reliability coming from latent concentration and resultant length (continuous).

IMU covariance fields are consumed, not ignored, and directly support the adaptive noise plan. 

BAG_TOPICS_AND_USAGE

The IMU segment boundary is treated as a pipeline implementation choice, not a mathematical constraint. 

BAG_TOPICS_AND_USAGE

II) LiDAR edits that raise the whole project to the same standard

These are strict, first-principles upgrades that align LiDAR with the IMU spec above (factorized exp-family evidence + Laplace/I-projection), and remove remaining hand-tuned knobs.

1) **Implemented (current code):** LiDAR evidence is now built from **Matrix Fisher rotation + planar translation** (no UT cache). Full exp‑family factorization is still an open research item.

Instead of regressing a quadratic surrogate, define LiDAR as explicit likelihood factors (categorical soft assignments; vMF directional terms; Gaussian translational terms) and compute 
g,H
g,H at 
zlin
z
lin
	​

, then SPD-project to BeliefGaussianInfo. This makes LiDAR evidence construction canonical and mathematically aligned with IMU.

2) Make tau_soft_assign an inferred latent (no fixed temperature)

Treat the inverse temperature as a latent variable (or slowly varying state) inferred by maximizing local marginal likelihood / matching predicted vs realized responsibility entropy under a conjugate prior. This eliminates a major tuning parameter without introducing gating.

3) Put LiDAR measurement covariances under inverse-Wishart adaptation

Any fixed 
Σmeas
Σ
meas
	​

 terms become IW random variables updated from post-fit residual sufficient statistics (same mechanism as adaptive 
Q
Q). This yields continuous, regime-adaptive trust without gates.

4) Unify all directional evidence as explicit vMF natural parameters

Make LiDAR directional constraints explicitly emit 
η=κμ
η=κμ objects (natural parameters) and feed them into the same Laplace evidence machinery. Then “IMU gravity direction” and “LiDAR normals/directions” are literally the same statistical type, only sourced differently.


A) Global System Invariants
A1. Branch-free totality (no gates, no heuristics)

No gating of any kind: no hard/soft accept-reject, no thresholding, no regime switches.

All operators are total: they must return valid outputs for all inputs in-domain (including malformed covariances, missing TF, irregular sampling).

No conditional execution paths that alter semantics; any stabilization must be applied unconditionally and exposed via certificates (e.g., PSD projection magnitude).

Robustness must be Bayesian/model-derived: all “downweighting” occurs only via posterior expectations of latent variables (e.g., random concentration, inverse-Wishart noise), never via hand-designed confidence rules.

A2. Manifold-intrinsic geometry (eliminate coordinate Jacobians)

Pose/orientation live on 
SE(3)
SE(3)/
SO(3)
SO(3); perturbations are applied via group retractions:

X←X\Exp(δξ)
X←X\Exp(δξ) (right perturbation by default unless explicitly changed).

Residuals are defined via 
\Log
\Log on the manifold:

rotation residuals: 
\Log(R1⊤R2)
\Log(R
1
⊤
	​

R
2
	​

)

pose residuals: 
\Log(X1−1X2)
\Log(X
1
−1
	​

X
2
	​

)

No extrinsic coordinate Jacobians. Allowed differentials are only:

intrinsic differentials of group actions (cross-product / adjoint maps),

SO(3)
SO(3)/
SE(3)
SE(3) right/left Jacobians of 
\Exp/\Log
\Exp/\Log (closed-form),

automatic differentiation over the intrinsic primitives above (still exact, not “coordinate Jacobians”).

A3. Canonical “Factor → Gaussian” embedding (overwrites UT regression evidence)

All sensors emit true likelihood factors (exponential-family where appropriate).

The only mechanism to produce a Gaussian information evidence block is:

a fixed-cost Laplace / I-projection at 
zlin
z
lin
	​

: compute 
(g,H)
(g,H) of the joint negative log-likelihood and project to SPD.

Any previous evidence construction based on quadratic regression/UT feature fitting is deprecated by contract and replaced by the Laplace/I-projection rule.

A4. Information-form fusion is the only fusion primitive

Evidence fusion is additive in natural parameters:

Gaussian: 
(L,h)
(L,h) add

exponential-family factors: natural parameters add internally before projection

No blending in Euclidean parameter space, no hand-weighted averaging outside the probabilistic model.

A5. Mandatory SPD domain enforcement (always-on)

Every covariance/precision-like matrix is passed through:

symmetrization: 
Σ←12(Σ+Σ⊤)
Σ←
2
1
	​

(Σ+Σ
⊤
)

SPD projection: 
ΠSPD(⋅)
Π
SPD
	​

(⋅) unconditionally

Any “lift” (e.g., 
L+εI
L+εI) is applied unconditionally and logged.

A6. Self-adaptive noise by construction (no hand-tuned 
Q
Q, no hand-tuned 
R
R)

Process noise 
Q
Q and measurement noises (
Σg,Σa,Σlidar,…
Σ
g
	​

,Σ
a
	​

,Σ
lidar
	​

,…) are random variables with conjugate priors:

matrix blocks: inverse-Wishart states updated via sufficient statistics

scalar reliabilities: Gamma/scale-mixtures (random concentration, Student-t-style)

Fixed covariances are not permitted except as priors on these noise states.

B) Rosbag-Driven Input Contract Invariants
B1. Data availability constraints

The pipeline may not assume /tf exists. If TF exists, it can be ingested, but correctness cannot depend on it.

IMU topic selection is a configuration choice, but the contract assumes at least one valid sensor_msgs/Imu stream exists.

B2. Units and covariance consumption (overwrites prior “ignore covariances” behavior)

If the dataset provides acceleration in g, the pipeline applies a deterministic upstream conversion to m/s² (no conditional logic).

sensor_msgs/Imu covariance fields are always consumed as hyper-evidence:

angular_velocity_covariance, linear_acceleration_covariance, orientation_covariance

Each is SPD-projected and incorporated into the measurement-noise inference state.

C) State and Calibration Invariants (IMU-specific but system-level)
C1. Extrinsics are priors + estimable state (not constants)

The LiDAR↔IMU extrinsic 
XLI∈SE(3)
X
LI
	​

∈SE(3) is:

initialized from calibration,

included as a state variable,

constrained by a prior factor (not hard-fixed).

C2. Time offset is priors + estimable state (continuous)

The IMU↔LiDAR time offset 
Δt
Δt is:

included as a state variable,

constrained by a prior factor,

used as a continuous time warp (no discrete alignment).

D) IMU Measurement Modeling Invariants (vMF is mandatory)
D1. Accelerometer direction is modeled as vMF (not optional)

The primary accelerometer “orientation” constraint is a directional vMF likelihood on 
S2
S
2
:

measured direction 
x=am−ba∥am−ba∥
x=
∥a
m
	​

−b
a
	​

∥
a
m
	​

−b
a
	​

	​


predicted direction 
μ=RWI⊤(−gW)∥gW∥
μ=
∥g
W
	​

∥
R
WI
⊤
	​

(−g
W
	​

)
	​

 (or the agreed state-augmented version)

likelihood 
∝exp⁡(κ μ⊤x)
∝exp(κμ
⊤
x)

D2. vMF reliability is random concentration (no manual κ tuning)

κ
κ is not fixed. It is driven by:

window sufficient statistics (resultant length), and/or

latent random concentration (Gamma/scale-mixture) with posterior expectation used deterministically.

No “trust accel” logic is permitted; all trust emerges from the inferred 
κ
κ (continuous).

D3. Sufficient-statistics fusion is the only directional fusion rule

Directional evidence is aggregated via vMF natural parameters/sufficient statistics:

accumulate resultant 
S=∑wixi
S=∑w
i
	​

x
i
	​

, mass 
N=∑wi
N=∑w
i
	​


compute 
κ^
κ
^
 from resultant via a deterministic map

The system never averages directions by Euclidean heuristics; all direction aggregation is vMF-consistent.

E) Temporal Integration Boundary Invariants (pipeline-flexible but mathematically fixed)
E1. Mathematical boundary: per scan interval factors

IMU constraints are defined over the LiDAR scan interval 
[tk,tk+1]
[t
k
	​

,t
k+1
	​

] (or the system’s canonical interval) as a factor bundle.

Any IMUSegment-style message is an implementation detail and must preserve:

exact timestamps,

raw measurements and covariances,

continuous membership weighting under 
Δt
Δt,

fixed compute budget (resampling allowed only if deterministic and weight-preserving).

F) Evidence Construction Invariants (overwrites prior LiDAR evidence design)
F1. Single canonical evidence constructor

There is exactly one way to produce BeliefGaussianInfo evidence from sensor factors:

build joint negative log-likelihood 
ℓ(z)
ℓ(z)

compute 
(g,H)
(g,H) at 
zlin
z
lin
	​


set 
L=ΠSPD(sym(H))
L=Π
SPD
	​

(sym(H)), 
h=−g
h=−g

Any previously separate “IMU evidence,” “LiDAR quadratic evidence,” or “UT regression evidence” must be expressed as factors feeding this constructor.

G) Certification and Telemetry Invariants (mandatory outputs)

Every operator returns a certificate bundle containing at minimum:

PSD projection magnitude (matrix delta norm, eigenvalue floor applied)

lift strength applied

effective sample size / support summaries for any windowed aggregation

inferred noise state summaries (IW 
(Ψ,ν)
(Ψ,ν) or equivalent) used in the step

Certificates are diagnostic only; they may not trigger gating.

H) Overwrites / Deprecations (explicit)

Deprecate UT-regression / feature-fit quadratic evidence as a primary evidence mechanism; replace with factor-based Laplace/I-projection.

Deprecate fixed measurement covariance usage except as priors on inverse-Wishart noise states.

Deprecate ignoring IMU covariance fields; they are mandatory inputs to noise inference.

Deprecate any accel “gravity gating” or “motion detection” logic; replaced by random concentration and inferred noise.
