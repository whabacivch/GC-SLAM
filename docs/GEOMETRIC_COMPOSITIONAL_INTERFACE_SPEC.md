# Geometric Compositional SLAM v2 — Interface + Runtime Contract (Reality-Aligned) (2026-02-02)

This document is the **current** interface + runtime contract for the GC v2 backend **as implemented in code**.

Why this rewrite:
- The prior version of this document contained large “legacy” sections (UT deskew, bin-map-only pose evidence, UT-regression
  LiDAR quadratic evidence, Wahba/Translation WLS, and a long “proposed” IW addendum) that are **not** the canonical runtime
  behavior anymore.
- The project’s hard rule is “no multipaths / no fallbacks”: we must not keep multiple incompatible math descriptions around
  in a “master reference” document.

Primary implementation anchors:
- Pipeline: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`
- Manifest constants: `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py`
- Certificates: `fl_ws/src/fl_slam_poc/fl_slam_poc/common/certificates.py`
- Numeric primitives: `fl_ws/src/fl_slam_poc/fl_slam_poc/common/primitives.py`
- Frame conventions (canonical): `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`

Non-goals:
- ROS message schemas.

---

## 0) Modeling Guide (Design Contract, Non-Negotiable)

This section governs how we *model* and what we *forbid*. It is not an implementation tutorial.

### 0.1 One world model, not many sensors

There is one latent physical world and one robot embedded in it.
Sensors are different projections of the same latent dynamics; they are not independent “voters”.

Contract:
- Independence is the exception, not the default.
- Any independence assumption must be explicit and justified (and treated as an approximation).
- Disagreement is information (drives state *and* noise learning), not a reason to gate.

### 0.2 Sensors constrain a shared latent state

Sensors do not “vote”; they constrain the same latent state via different physical mechanisms.
Consensus is emergent from consistent modeling and uncertainty learning, not enforced by heuristics.

### 0.3 Every raw bit must become exactly one of: evidence, uncertainty, or certificate

Every piece of raw information the system ingests must flow into exactly one of:
1) Evidence (a likelihood / energy term that affects `(L, h)`),
2) Uncertainty (noise learning and modulation, e.g., IW updates; continuous readiness/influence scaling),
3) Certificate (conditioning, approximation triggers, contract checks, expected vs realized effect).

If a raw bit contributes to none of these, it is wasted and must be removed or wired in.

### 0.4 Information geometry is the governing math

GC v2 is an information-geometric backend:
- Prefer exponential-family, dually-flat coordinates and closed-form composition where available.
- Use Bregman/KL structure for projection/fusion; avoid Jacobian chains and hidden iteration.
- Maintain order-robustness via associative/commutative composition in the correct coordinates.

Canonical math reference:
- `docs/Comprehensive Information Geometry.md`

---

## 1) Global Invariants (Hard)

### 1.1 Chart + state ordering (fail-fast)

- `chart_id = "GC-RIGHT-01"` is the global chart convention for all beliefs and evidence.
- Tangent ordering is fixed (22D):

| Slice | Symbol | Dim | Indices (0-based) |
|---:|---|---:|---|
| translation | δt | 3 | 0..2 |
| rotation (SO(3) rotvec) | δθ | 3 | 3..5 |
| velocity | δv | 3 | 6..8 |
| gyro bias | δbg | 3 | 9..11 |
| accel bias | δba | 3 | 12..14 |
| time offset | δΔt | 1 | 15 |
| extrinsic (se(3)) | δξ_ex | 6 | 16..21 |

- `D_Z = 22`, `D_DESKEW = 22`.
- Any `chart_id` mismatch is a hard error.
- Any dimensional mismatch is a hard error.

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py`.

### 1.2 Local charts (anchors) are mandatory

Every belief/evidence carries:
- `anchor_id: str` (local chart instance id)
- `X_anchor: SE3` (anchor pose)

Anchors are maintained by a continuous `AnchorDriftUpdate` operator (no discrete anchor promotion).

### 1.3 Fixed-cost budgets (compile-time constants)

Hard budgets:
- `K_HYP = 4`
- `HYP_WEIGHT_FLOOR = 0.01 / K_HYP` (current constant is `0.0025`)
- `N_POINTS_CAP = 8192`
- `MAX_IMU_PREINT_LEN = 512` (fixed padding for JIT)

Any reduction must be an explicit ApproxOp and logged.

### 1.4 No multipaths / no fallbacks

- No “GPU if available else CPU”.
- Backend/operator selection is explicit (`pose_evidence_backend`, `map_backend`) and must be reported in the runtime manifest.

### 1.5 No hidden iteration

Disallowed inside any single operator call:
- data-dependent solver loops (Newton/CG/LS “until tolerance”, line search, adaptive iterations).

Allowed:
- fixed-size loops with compile-time constant bounds (e.g., fixed OT iterations, fixed IMU pad length).

### 1.6 No gates, no multipaths, no hidden heuristics (definition)

GC v2 prohibits:
- **Gates**: threshold/branch decisions that disable a factor, skip an update, or change model structure.
- **Multipaths**: “try this backend else that backend” logic; implicit fallbacks.
- **Hidden heuristics**: any unlogged rule that changes association, evidence inclusion, or uncertainty without a declared model.

Allowed (but must be declared and certifiable):
- Domain projections (PSD floor, lifted solves, clamp) as explicit numeric primitives with recorded magnitudes.
- Fixed-budget masking (e.g., padded arrays) where “validity” enters as continuous weights/masks, not as mode switches.

---

## 2) Core Data Structures

### 2.1 Belief Gaussian (information form)

```text
BeliefGaussianInfo:
  chart_id: str
  anchor_id: str
  X_anchor: SE3
  stamp_sec: float

  z_lin: (22,)
  L: (22,22)   # symmetric PSD/PD
  h: (22,)

  cert: CertBundle
```

MAP increment is defined by the declared lifted solve:
`δz* = (L + eps_lift I)^{-1} h`.

### 2.2 Certificates + expected effect (required)

Certificates are the audit trail for approximations and stabilizations.
Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/common/certificates.py`.

```text
CertBundle:
  chart_id, anchor_id
  exact: bool
  approximation_triggers: list[str]
  frobenius_applied: bool

  conditioning: (eig_min, eig_max, cond, near_null_count)
  support: (ess_total, support_frac)
  mismatch: (nll_per_ess, directional_score)
  excitation: (dt_effect, extrinsic_effect)
  influence:
    lift_strength
    psd_projection_delta
    nu_projection_delta
    mass_epsilon_ratio
    anchor_drift_rho
    dt_scale
    extrinsic_scale
    trust_alpha
    power_beta

  overconfidence:
    excitation_total
    ess_to_excitation
    cond_to_support
    dt_asymmetry
    z_to_xy_ratio

ExpectedEffect:
  objective_name: str
  predicted: float
  realized: float | null
```

Downstream rule:
- Consumers may only scale influence continuously using certificates; they must not branch/skip factors.

Overconfidence sentinel (hard):
- Operators must record continuous overconfidence sentinels when relevant (especially at fusion / evidence aggregation):
  - if ESS grows faster than excitation (motion/timing/extrinsic excitation proxies), and/or
  - if conditioning improves without corresponding residual support.
- These sentinels are diagnostic-only. The only allowed response is continuous conservatism (never gating).

Closed-form conservatism mechanism (recommended, Wishart/IW):
- When dependence cannot be modeled jointly or via explicit cross-covariance blocks, conservatism should be injected by
  learning an inflation covariance on the relevant residual(s) using an Inverse-Wishart state:
  - Prior: `Sigma_dep ~ IW(Psi0, nu0)`
  - Update from residual(s): `Psi <- Psi + r r^T`, `nu <- nu + 1` (or weighted / batched equivalent)
  - Plug-in mean: `E[Sigma_dep] = Psi / (nu - p - 1)` (with declared ν regularization when needed)
  - Use `E[Sigma_dep]` to inflate the effective covariance, which downscales information continuously (closed form).

This turns “dependence uncertainty” into a learned random variable on the SPD cone, instead of a hand-tuned heuristic.

Tempered posteriors / power EP (hard, closed-form):
- For known-dependence sensor families (or when sentinels indicate overconfidence), apply a tempered likelihood:
  - `p(z|x)^β` with `β ∈ (0,1]`
- In Gaussian information form this is closed-form:
  - `L ← β L`, `h ← β h`
- β must be computed as a deterministic, continuous, bounded control law from certificates (no hidden heuristics).
- β must be recorded as `influence.power_beta` and contributes to Frobenius trigger magnitude via `|1-β|`.

dt observability collapse (must be monitored):
- dt can remain weakly observable when excitation is low and multiple residuals absorb timing error in similar ways.
- The system must publish a continuous proxy `overconfidence.dt_asymmetry ∈ [0,1]` indicating whether dt coupling is
  asymmetric across residual families (e.g., velocity-consistency vs pose-alignment). Low values are a warning sign.

Z/vertical observability (hardest subspace; invariant):
- Under planar motion, LiDAR geometry often under-excites Z; planar priors can be fragile.
- The system must publish a continuous proxy `overconfidence.z_to_xy_ratio` (relative Z vs XY information strength).
- If Z is under-constrained, the only allowed response is continuous conservatism and/or adding a non-degenerate physical
  constraint factor (never gating).

IW hyperprior scaling discipline:
- IW is principled but can silently over-stiffen:
  - ν too large ⇒ noise stops learning,
  - Ψ too strong ⇒ early mistakes fossilize.
- Priors must be weak by default; if overconfidence sentinels rise, the system must react via continuous scaling (trust alpha,
  IW update strength, learned inflation), not by gating.

---

## 3) Numeric Primitives (Library Contracts)

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/common/primitives.py`.

All primitives are total functions (always run) and return a certificate magnitude that can be exactly zero if no change occurred.

- `Symmetrize(M)` → `0.5*(M+M^T)`; records `sym_delta`.
- `DomainProjectionPSD(M, eps_psd)` → symmetric eigendecomp with eigenvalue floor; records projection delta and conditioning.
- `SPDCholeskySolveLifted(L, b, eps_lift)` → solves `(L + eps_lift I)x=b`; records lift strength.
- `InvMass(m, eps_mass)` → `1/(m+eps_mass)`; records `mass_epsilon_ratio`.
- `Clamp(x, lo, hi)` → records `clamp_delta`.

---

## 4) Dependence Policy (How we encode “deeply dependent evidence”)

### 4.1 The 22D state is the coupling medium (not “extra DOFs”)

The augmented 22D state exists to express dependence:
- `pose` couples shared geometry
- `velocity` couples pose ↔ inertial ↔ odom
- `gyro bias` couples IMU ↔ odom ↔ LiDAR/camera yaw dynamics
- `accel bias` couples IMU ↔ gravity ↔ planar constraints
- `time offset (dt)` couples IMU ↔ LiDAR ↔ camera ↔ odom in time
- `extrinsics` couples all sensors geometrically

Hard rule:
- If a state dimension never appears in a residual (likelihood term), it is not a state; it is dead weight.

### 4.2 Independence is a modeling decision (and dangerous by default)

Allowed independence assumptions only when:
- sensors do not share latent variables at the modeled scale, or
- the dependence is provably negligible compared to noise.

Forbidden independence assumptions (must not be modeled as independent factors without coupling):
- odom pose ↔ odom twist
- IMU ↔ LiDAR timing/deskew
- IMU ↔ odom yaw rate / velocity
- LiDAR ↔ camera extrinsic/depth geometry
- measurements sharing timestamps/integration windows (unless modeled via shared latents/masks)

If dependence is not modeled explicitly, conservatism must be injected explicitly (e.g., cross-covariance, bounds, or
declared approximation with Frobenius correction downstream).

#### 4.2.1 Cross-covariances are sometimes mandatory (not “optional”)

Cross-covariance is allowed but not always optional in spirit: some dependencies are so central that if they are not
modeled as a joint factor, they must be represented via explicit cross-covariance (or a certified conservative bound),
otherwise the system will become overconfident during smooth motion.

Minimum set (must be handled jointly or via cross-cov/bound):
- odom pose ↔ odom twist
- IMU gyro ↔ IMU accel (bias-coupled)
- IMU ↔ LiDAR deskew / association residuals (via `dt`)

### 4.3 How dependence enters the math (three mechanisms)

Mechanism A — Joint factors:
- Prefer one joint factor that constrains multiple state blocks simultaneously, instead of multiple “independent” factors
  that double-count shared latent structure.

#### 4.3.1 Mandatory joint-factor rule (hard)

If two or more measurements constrain the same physical law (e.g., kinematics continuity, timing consistency, rotation-rate
consensus), they must be represented either:
1) as a single joint factor, or
2) with explicit cross-covariance / conservative bounding, certified and logged.

Independent factors without such coupling are forbidden (they cause double counting and false confidence).

Mechanism B — Cross-covariances (or certified conservative bounds):
- When two measurements are related but not merged into one factor, include off-diagonal covariance blocks or a declared
  conservative bound; otherwise fusion must be explicitly conservative.

Mechanism C — Shared latent parameters:
- Biases, extrinsics, and `dt` are shared latents. If two sensors both depend on the same latent, they are not independent
  even if their raw noise is.

---

## 5) Canonical Map Backend: `PrimitiveMap`

The canonical map is `PrimitiveMap`: a fixed-budget probabilistic atlas of local 3D Gaussian primitives with optional
directional (vMF) and appearance payloads. It is designed so that:
- map fusion is closed-form in natural parameters (product-of-experts / additive information),
- association is soft (responsibilities),
- map maintenance (cull/forget/merge-reduce) is explicit and certified.

Implementation anchors:
- Map structure + update ops: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/structures/primitive_map.py`
- MeasurementBatch (unified measurement primitives): `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/structures/measurement_batch.py`
- LiDAR surfel extraction (MA-hex 3D + Wishart-regularized precision): `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/lidar_surfel_extraction.py`
- Camera feature extraction + LiDAR depth fusion (closed-form PoE on depth):  
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/visual_feature_extractor.py`  
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/lidar_camera_depth_fusion.py`  
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/splat_prep.py`  
  - `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/camera_batch_utils.py`
- Association (OT, fixed iterations): `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/primitive_association.py`

### 5.1 Measurement primitives (the “one world model” interface)

All geometry-bearing sensor outputs are converted into a fixed-budget `MeasurementBatch` of “measurement primitives”.
Each primitive is (at minimum):
- a 3D Gaussian in **information form** `(Lambda, theta)` where `theta = Lambda @ mu`,
- a **mandatory multi-lobe vMF** natural-parameter bundle with `B=3` lobes:
  - `etas[b] = kappa_b * mu_b` for `b ∈ {0,1,2}` (each `mu_b ∈ S²`)
  - the “resultant” natural parameter is `eta_sum = Σ_b etas[b]` and is the canonical reduction used when a single-lobe
    consumer is required (closed form; no heuristics),
- a continuous reliability `weight`,
- source metadata (camera vs LiDAR) and timestamp,
- optional appearance payload (RGB).

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/structures/measurement_batch.py`.

Design contract:
- Natural-parameter discipline is mandatory: fusion is closed-form addition in natural coordinates whenever in-family.
- Fixed budgets are mandatory: `N_FEAT` camera primitives and `N_SURFEL` LiDAR primitives; padding uses `valid_mask`.
- vMF is not optional: every primitive carries `B=3` lobes (even if only the resultant lobe is populated initially).

### 5.2 LiDAR surfel extraction (geometry-rich primitives)

LiDAR contributes geometry-dominant primitives (“surfels”) constructed from deskewed point clouds:
1) fixed-size MA-hex 3D bucketing (bounded compute; collisions are a declared approximation),
2) batched weighted plane fit per voxel (closed-form eigendecomposition on 3×3 covariance),
3) Gaussian covariance from in-plane spread + perpendicular residual,
4) **Wishart regularization in precision space** to prevent pathological overconfidence in low-support cells,
5) multi-lobe vMF direction bundle (default: populate lobe 0 with the plane normal resultant).

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/lidar_surfel_extraction.py`.

Why Wishart here:
- It is closed-form, lives on the SPD cone (Monge–Ampère/Wishart geometry), and gives a principled “pseudocount” prior
  on precision instead of ad hoc epsilons.

### 5.3 Camera primitives, enriched by LiDAR (LiDAR → RGBD depth)

Camera produces keypoints+descriptors and a per-feature depth likelihood. Depth is then **enriched by LiDAR** via a single
closed-form LiDAR depth evidence API that returns natural parameters.

Pipeline contract (per feature at pixel `(u,v)`):
1) Camera depth evidence (from the depth image) produces a scalar depth natural parameter pair `(Lambda_c, theta_c)` in the
   feature metadata (`depth_Lambda_c`, `depth_theta_c`).
2) LiDAR depth evidence produces `(Lambda_ell, theta_ell)` at the same `(u,v)` using a continuous, always-defined mixture
   of two routes:
   - Route A: project LiDAR into the image and robustly sample local depth support
   - Route B: ray–plane intersection from a local LiDAR plane fit
3) Fuse depth in natural coordinates (product-of-experts):
   - `Lambda_f = w_c * Lambda_c + w_ell * Lambda_ell`
   - `theta_f = w_c * theta_c + w_ell * theta_ell`
   - `z_f = theta_f / Lambda_f`, `sigma_z^2 = 1/Lambda_f`
4) Backproject `(u,v,z_f)` into 3D and compute an analytic 3×3 covariance; convert to `(Lambda, theta)` for the batch.

Implementation anchors:
- Visual features and camera depth natural params: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/visual_feature_extractor.py`
- LiDAR depth evidence (natural params; Route A/B): `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/lidar_camera_depth_fusion.py`
- Fused splat prep (PoE in depth natural params → 3D Gaussian): `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/splat_prep.py`
- Packing to `MeasurementBatch`: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/camera_batch_utils.py`

Design contract:
- LiDAR depth fusion must be continuous (no hard windows); when LiDAR is uninformative the LiDAR contribution must go to 0
  by weight, not by branching.

### 5.4 RGB enriches LiDAR and the map (appearance payload, not “just visualization”)

RGB is not just for rendering: it is a payload attached to primitives and fused into the map as responsibility-weighted
color evidence. This is critical because down the line we can use appearce of things to layer in categorical distrobution evidence for semantic identification of objects and places. The infrastructure must exist first to enable that kind of enhancement. 

Current behavior:
- Camera features carry per-feature RGB sampled at the keypoint and propagate into the camera slice of `MeasurementBatch`.
- When fusing measurement primitives into the map, colors are blended responsibility-weighted.

Implementation anchors:
- Camera color sampling + `Feature3D.color`: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/sensors/visual_feature_extractor.py`
- Camera batch colors: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/camera_batch_utils.py`
- Responsibility-weighted map color update: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/structures/primitive_map.py`

### 5.6 BEV15 views (pushforward + geodesic view path; output-side)

GC uses information-geometric pushforwards to derive BEV (bird’s-eye) views from 3D primitives.

Core pushforward (exact, closed form):
- 3D Gaussian → 2D BEV Gaussian via linear pushforward `μ_bev = P μ`, `Σ_bev = P Σ Pᵀ`.
- Directional vMF payloads are rotated into the view frame by SO(3) pushforward:
  - vMF has natural parameter `η = κ μ`, density `f(u) ∝ exp(ηᵀ u)` on S².
  - Under view rotation `R ∈ SO(3)`, the pushforward is exact and in-family: `η′ = R η` (κ unchanged).
- No S¹ reduction is used; any S¹ collapse would be an approximation and is not part of GC v2.

BEV15 (future view-layer; not active in the runtime pipeline):
- BEV15 is a fixed set of `N=15` oblique projection matrices `P_k` generated by sweeping the oblique angle along a 1D
  geodesic in angle space (linear interpolation in φ), producing 15 consistent BEV “slices”.
- This is a view-side construct for rendering/association diagnostics; it must not introduce gates or change inference
  operator structure. It is preserved in the spec as a future expansion target.

Implementation anchors:
- BEV pushforward helpers (Gaussian pushforward + BEV15 `P_k` generation):  
  `fl_ws/src/fl_slam_poc/fl_slam_poc/common/bev_pushforward.py`
- Splat rendering (EWA) with multi-lobe vMF shading and view-stable fBm texture:  
  `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/rendering.py`

### 5.5 Soft association and map update (no gating)

Association is soft (responsibilities), with fixed compute budgets. No binarized matches, no threshold gating.

Map update is executed every scan:
- Fuse: add measurement natural parameters to the targeted map primitives (PoE / natural addition),
- Insert: add new primitives when map support is insufficient,
- Cull/forget/merge-reduce: explicit budgeting operators with certificates (mass drop logged; Frobenius when family changes).

Implementation anchors:
- Association: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/primitive_association.py`
- Fuse/insert/cull/forget/merge-reduce: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/structures/primitive_map.py`

#### α-geometry knob (Amari α-divergence; declared)

GC v2 supports a single declared geometric knob for “mode-seeking vs mass-covering” behavior:
- α controls the projection geometry used in (a) soft association and (b) any reduction/projection operators.

Contract:
- α must be explicit in config/manifest (no implicit defaults hidden in code).
- α must be driven continuously by certificates when adaptive (e.g., mismatch/support), never by thresholds.
- Compute must remain fixed-budget (e.g., fixed-K Sinkhorn; fixed-budget merge-reduce).

#### Projection typing (I-projection vs M-projection; declared)

Any operator that performs a projection / reduction must declare its projection type in the certificate triggers:
- I-projection: mass-covering (stable under ambiguity)
- M-projection: mode-seeking (sharp but brittle)
- α-projection: intermediate

This is mandatory for:
- hypothesis publishing projection,
- mixture reduction / merge-reduce,
- any out-of-family approximation.

---

## 6) Operators (Canonical Runtime)

All operators return `(result, CertBundle, ExpectedEffect)`.

### 6.1 `PointBudgetResample` (ApproxOp)

Deterministic, weight-preserving resampling to enforce `N_POINTS_CAP`.

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/point_budget.py`.

### 6.2 `PredictDiffusion` (ExactOp)

Predicts belief with declared diffusion `Q` (discretized as `dt_sec * Q`) and PSD projects.

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/predict.py`.

### 6.3 `DeskewConstantTwist` (ApproxOp)

Deskew uses IMU preintegration over the scan window to produce a constant-twist model used to deskew points.

Implementation anchors:
- Operator: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/deskew_constant_twist.py`
- Pipeline usage: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`

### 6.4 IMU + Odom evidence (factor-based; closed-form/Laplace)

Per-scan evidence includes IMU and odometry factors (no heuristic gating).

Implementation anchors (non-exhaustive):
- vMF gravity evidence: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_evidence.py`
- gyro rotation evidence: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_gyro_evidence.py`
- IMU preintegration factor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_preintegration_factor.py`
- odom SE(3) pose evidence: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/odom_evidence.py`
- planar z and v_z priors: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/planar_prior.py`
- odom twist / kinematic consistency: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/odom_twist_evidence.py`

The pipeline computes an IMU+odom-informed linearization point `z_lin` before pose evidence construction.

### 6.5 Pose evidence backend (explicit, single-path selection)

Pose evidence backend is selected explicitly (no fallback) and must be declared in the runtime manifest:
- `pose_evidence_backend="primitives"`: primitive alignment / visual+LiDAR pose evidence (Laplace at `z_lin`)
  - Implementation: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/visual_pose_evidence.py`
- `pose_evidence_backend="bins"`: bin-based pose evidence path (present for compatibility; not the canonical long-term map backend)

### 6.6 Adaptive noise (Inverse-Wishart; implemented)

Noise is modeled as IW random variables with per-scan commutative sufficient-stat updates.

Implementation anchors:
- Process noise IW: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/inverse_wishart_jax.py`
- Measurement noise IW: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/measurement_noise_iw_jax.py`
- Priors/hyperpriors: `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py`

### 6.6.1 Noise is a random variable, not a constant

Contract:
- “Constants” are priors/hyperpriors and budgets, not truths.
- All meaningful noise terms must be represented as random variables (IW where applicable), updated every scan, and
  influenced by context *continuously*.
- When sensors disagree, do not choose a winner; treat disagreement as information about both the state and the noise.

### 6.6.2 Context is not a switch; it is a modulator

Context signals (jerk, vibration, roughness, slip proxies, etc.) may:
- scale information matrices continuously,
- modulate IW update strength,
- adjust readiness/influence weights.

They may never:
- disable a factor,
- branch the pipeline,
- skip an update.

All context mappings must be smooth, bounded, and declared.

#### Context routing contract (hard)

Context may influence the system only via:
- information scaling (continuous scaling of `(L, h)` as an operator output),
- IW sufficient-stat updates (noise learning strength),
- fusion trust `alpha` (continuous).

Context must never:
- modify residual definitions,
- alter association topology,
- change operator order.

### 6.6.3 Time is a latent variable

Contract:
- `dt` must influence likelihood construction (differentiably / continuously).
- Time warp must be soft (kernel-based), never hard-windowed.
- If `dt` does not affect residuals, it will not converge.

#### dt coupling requirement (hard)

Any operator whose residual depends on temporal alignment must expose a differentiable dependence on `dt` and must include
the corresponding coupling in the projected quadratic form (i.e., the `(15,·)` blocks of `H/L`), unless the operator emits a
certificate that explicitly declares `dt`-insensitivity for that measurement stream.

This prevents “dt exists in the state but does nothing” failures.

### 6.7 `FusionScaleFromCertificates` (ExactOp)

Computes continuous trust `alpha` from certificates and manifest constants.

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/fusion.py`.

Control-law contract (hard):
- Treat this mapping as a control law, not glue code.
- Must be deterministic, continuous, and bounded; must never gate factors.
- Must be monotone in “risk” proxies (worse conditioning/support/excitation/overconfidence ⇒ no larger alpha).
- Must be testable in isolation: add unit tests for extreme certificates (ill-conditioned, low support, low excitation, Z under-constraint, dt symmetry).

### 6.8 `InfoFusionAdditive` (ApproxOp)

Performs additive information fusion and PSD projects the result.

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/fusion.py`.

### 6.9 `PoseUpdateFrobeniusRecompose` (ApproxOp)

Applies Frobenius/BCH3 correction with continuous strength blended by total trigger magnitude.

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/recompose.py`.

#### Frobenius / pre-Frobenius correction policy (information-geometry invariant)

Contract:
- Approximation triggers include (at minimum): linearization, PSD/domain projection, mixture/association reduction, and any
  out-of-family likelihood approximation.
- If approximations are introduced, a Frobenius/pre-Frobenius correction must be applied and logged (order-robustness /
  third-order lift). In the current backend this is enforced downstream via a continuous correction step driven by total
  certificate trigger magnitude.

Canonical reference:
- `docs/Comprehensive Information Geometry.md` (associativity/WDVV, pre-Frobenius third-order lifts, Wishart/MA domains)

### 6.10 `AnchorDriftUpdate` (ApproxOp)

Continuous local chart maintenance (no discrete anchor promotion).

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/anchor_drift.py`.

### 6.11 `HypothesisBarycenterProjection` (ApproxOp)

Produces a single belief for publishing by weight-floored barycenter in information form.

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/hypothesis.py`.

---

## 7) Runtime Manifest (Required)

The runtime must emit a manifest listing:
- chart id, dimensions, budgets, epsilons
- explicit backend selections (`pose_evidence_backend`, `map_backend`)
- a short list of “resolved operators/backends” used at runtime
- any configuration toggles that materially change evidence construction (no hidden modes)

Implementation anchor: `RuntimeManifest` in `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`.

---

## 8) Deterministic Per-Scan Order (Canonical)

The canonical order is defined by the pipeline implementation
`fl_ws/src/fl_slam_poc/fl_slam_poc/backend/pipeline.py`.

At a high level (per hypothesis):
1. `PointBudgetResample`
2. `PredictDiffusion`
3. IMU membership weights (soft window)
4. IMU preintegration for deskew and scan-to-scan evidence
5. `DeskewConstantTwist`
6. IMU + odom evidence; compute `z_lin`
7. Pose evidence backend (explicit selection)
8. Excitation prior scaling (continuous)
9. `FusionScaleFromCertificates`
10. `InfoFusionAdditive`
11. `PoseUpdateFrobeniusRecompose`
12. IW sufficient statistics updates (process + measurement)
13. Map update (PrimitiveMap fuse/insert/cull/forget/merge-reduce)
14. `AnchorDriftUpdate`

After all hypotheses:
15. `HypothesisBarycenterProjection`

---

## 9) Explicit Deprecations (Not Canonical Here)

This spec intentionally excludes and deprecates the following older descriptions as “master reference” content:
- UT-based deskew moment matching and UT-cache-driven LiDAR quadratic regression evidence
- Wahba SVD / Translation WLS pose extraction descriptions as the primary path
- Treating fixed measurement covariances as anything other than priors/hyperpriors (adaptive IW is canonical)
