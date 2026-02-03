<!-- Do not edit this document unless explicitly instructed to. -->

# Geometric Compositional SLAM v2 — Interface + Runtime Contract (Reality-Aligned) (2026-02-02)

This document is the **current** interface + runtime contract for the GC v2 backend **as implemented in code** and where the current implimentation falls short, this document illuminates the endstate goal.

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

**Current vs end state:** What the code does now and what the spec requires are made explicit in §10 (Declared Non-Idealities), with code anchors and planned resolution. The target map architecture (atlas, additive insertion, tile-local fuse) is in §5.7. Keep §10 and §5.7 updated as gaps are closed so the spec stays the single source of truth.

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

### 0.5 ExactOp vs ApproxOp (definition)

- **ExactOp**: composition is an in-family morphism in declared coordinates (e.g., natural-parameter addition, rigid rotation of vMF natural params, linear Gaussian pushforward).
- **ApproxOp**: any out-of-family projection, truncation, budget enforcement, or linearization.

This enforces "associativity when intact" and makes compute optimizations spec-legitimate.

### 0.6 Belief geometry and success (what understanding means in GC)

**Core idea:** The belief should asymptotically align with the Euclidean/geometric structure of the world, not because Euclidean geometry is imposed, but because it is the fixed point of consistent statistical evidence. The belief lives on a statistical manifold; the world is Euclidean 3D with rigid-body kinematics. Inference does not assume Euclidean structure — it discovers and stabilizes it as the configuration that best explains all observations. When things go well, the statistical manifold becomes locally isometric to the Euclidean one in the directions that matter for action.

**Euclidean structure is earned, not privileged.** Surfaces, occupancy, and topology are not hard-coded; likelihood structure, association transport (π), additive information fusion, and uncertainty learning pull the belief toward a configuration that behaves like a Euclidean world because that is the only way to remain consistent across sensors and time.

**Convergence means three alignments:** (1) **Statistical** — residuals shrink, support grows, IW noise stabilizes, overconfidence sentinels quiet down. (2) **Geometric** — means and covariances of primitives flatten into surfaces, normals cohere, vMF concentrations sharpen. (3) **Topological** — the atlas nerve stabilizes; overlaps become persistent; connectivity reflects actual spatial adjacency. When these agree, the belief is structurally consonant with the world and supports prediction, navigation, and action — not merely localization.

**Operational formulation:** GC aims for beliefs whose induced geometry converges toward the Euclidean structure of the environment wherever evidence permits, yielding an internal model that supports prediction, navigation, and action rather than merely localization.

**Limits by design:** The belief will never exactly equal Euclidean reality — sensors are finite and noisy, some DOFs are weakly observable (Z, scale, dt, extrinsics), and the world is dynamic and partially observed. GC does not pretend otherwise: where evidence is strong, belief becomes sharply Euclidean; where evidence is weak, belief remains appropriately diffuse; and the system knows the difference via certificates. That self-knowledge is part of understanding. Topology and rendering (§11) are not decorations; they are mirrors of belief quality — coherent rendered geometry is a diagnostic that belief geometry is aligning with physical geometry; sparsity, flicker, and broken connectivity are visible symptoms of epistemic uncertainty.

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

- No "GPU if available else CPU".
- Backend/operator selection is explicit (`pose_evidence_backend`, `map_backend`) and must be reported in the runtime manifest.
- Deprecated and legacy code paths must not be retained. There is one canonical implementation per operator. No compatibility shims; no bin-map, no bins pose-evidence backend. If such code exists, it is a spec violation and must be removed.

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

### 1.7 No superlinear materialization (hard)

Contract:
- No operator may materialize an intermediate array whose size is O(N_meas · K_assoc) unless it is the operator output (and declared in the manifest).
- All reduce-by-key computations (map fuse, IW stats, association accumulation) must be implemented as streaming/chunked reductions with fixed chunk sizes and must not allocate “flat K” buffers.
- PSD projections and matrix factorizations must occur at most once per map primitive per scan (unless explicitly declared as ApproxOp with certificate magnitudes).

This turns the exact OOM culprit class into a spec violation, not a surprise.

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

  compute:   # runtime correctness signals (OOM / K blowups)
    alloc_bytes_est
    largest_tensor_shape   # string or (ndim, size) proxy
    segment_sum_k         # effective K if segment operations used
    psd_projection_count
    chol_solve_count
    scan_io:              # per-scan I/O window + buffer accounting
      scan_seq
      scan_stamp_sec
      scan_window_start_sec
      scan_window_end_sec
      streams             # per-stream window/capacity/drop stats
    device_runtime:       # host/device transfer + sync estimates
      host_sync_count_est
      device_to_host_bytes_est
      host_to_device_bytes_est
      jit_recompile_count

  overconfidence:
    excitation_total
    ess_to_excitation
    cond_to_support
    dt_asymmetry
    z_to_xy_ratio
    ess_growth_rate       # smoothed
    excitation_growth_rate
    nullspace_energy_ratio # captures Z/dt collapse better than raw cond

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

This turns “dependence uncertainty” into a learned random variable on the SPD cone, instead of a hand-tuned heuristic. Mandatory trigger conditions and scope are specified in §6.6.4 and §6.6.5.

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
color evidence. Down the line we can use appearance of things to layer in categorical distribution evidence for semantic identification of objects and places. The infrastructure must exist first to enable that enhancement. 

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
- Any 2D directional quantity derived from vMF for BEV rendering (e.g., Pη) is a **derived visualization feature**, not evidence and not a probability model.

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

#### 5.5.1 MA-Hex candidate generation (hard)

- Candidate generation must be local and bounded using MA-Hex (3D for surfels, 2D “web” for BEV candidate sets) with fixed stencils and fixed per-cell occupancy caps.
- Global kNN or global sorts over map size are forbidden in the association backend.
- The association backend must expose in its certificate bundle: `hex_h` (cell scale), `stencil_radius`, `max_occupants`, `candidate_saturation_frac` (hit-rate of occupancy cap), and `candidate_count_mean`.

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
- α may affect only: association objective / normalization, and mixture reduction projection geometry.
- α must not affect: which candidates are considered (candidate topology must come from MA-Hex constraints), the set of operators run, or budgets.

#### Projection typing (I-projection vs M-projection; declared)

Any operator that performs a projection / reduction must declare its projection type in the certificate triggers:
- I-projection: mass-covering (stable under ambiguity)
- M-projection: mode-seeking (sharp but brittle)
- α-projection: intermediate

This is mandatory for:
- hypothesis publishing projection,
- mixture reduction / merge-reduce,
- any out-of-family approximation.

### 5.7 Target map architecture (atlas end state)

The canonical end state for the map backend is an **atlas of fixed-budget tiles**: one global structure (AtlasMap) owning many local tiles (PrimitiveMapTile), with fixed-cost per-scan updates and additive growth. This section specifies that target; implementation must converge to it.

#### 5.7.1 Atlas and tile structure

- **PrimitiveMapTile**: fixed capacity `M_TILE` primitives per tile. Same primitive content as today (Λ, θ, η, color, mass, timestamp, ids, valid_mask). Maintenance ops (fuse, insert, cull, forget, merge-reduce) are tile-local and certified.
- **AtlasMap**: host-side store mapping `tile_id → PrimitiveMapTile`. No single global flat array of size proportional to world map; candidate and fuse sizes are bounded by the active set and per-tile caps.
- **Frame**: Primitives are stored in **world frame**. The tile is an indexing and budgeting structure, not a coordinate frame. Tile ID is determined only by 3D position (MA-Hex 3D cell) for deterministic addressing.

#### 5.7.2 Tile addressing and active set

- **Tile ID**: From 3D position (x, y, z) using **MA-Hex 3D** key with scale `H_TILE` (voxel/tile size in m). Deterministic.
- **Ordering**: Total order on tile IDs (e.g. lexicographic on 3D cell coords) for padding, cache eviction, and tie-breaking. Replay and eviction must be deterministic.
- **Active set**: Each scan, a **fixed-size** set of tiles is active: e.g. fixed-radius 3D hex neighborhood around the robot tile, so `|active_tiles| = N_ACTIVE_TILES` (constant). Pad with empty tile slots if fewer tiles exist. Tiles outside the active set are **not updated** this scan (explicit budgeting approximation; must be logged in certificates). No gating: the active set is a fixed compute window, not “if observed then run.”

#### 5.7.3 Association: responsibilities = transported mass

- **Coupling**: π[i,k] ≥ 0 is the transported mass from measurement i to candidate k. Output of unbalanced Sinkhorn (fixed iterations) over cost C and marginals a, b.
- **Responsibilities for fusion**: **Mass responsibility** is π itself. Fusion uses `r[i,k] = π[i,k]` as the PoE weight (no row-normalization of π for the fusion path). Do not replace π by a row-stochastic proxy unless explicitly declared as an approximation with certificate.
- **Effort (diagnostics only)**: `r_effort[i,k] = π[i,k] * C[i,k]` may be used for diagnostics or adaptive noise (e.g. IW updates). It must not change fusion weights or model structure unless specified.

#### 5.7.3.1 Association inputs contract (auditable)

To make OT/association implementable and auditable, the following must be declared and logged.

- **Measurement mass budget** `a[i]`: Must be computed by a declared rule (e.g. per-point weight, per-surfel mass, per-feature confidence). Sum and p95 must be logged. Novelty uses `n_i = clamp(a[i] - m_i, 0, a[i])` as already specified.
- **Candidate mass prior** `b[k]`: Must be declared (e.g. tile primitive mass, uniform within view, or mass-tempered) and logged.
- **Cost** `C[i,k]`: Must be decomposed into named terms with units (e.g. point-to-plane, normal mismatch, color residual). Each term’s scale must be a surfaced prior/budget (no hidden constants); matches the “constants must be surfaced” discipline.
- **Unbalanced OT parameters** (ε, τ_a, τ_b or equivalent): Must be compile-time constants (fixed-cost), reported in the manifest, with a certificate field for **marginal defect / dual gap proxy** (already referenced in §5.7.7).

#### 5.7.4 Candidate pool (bounded)

- **Per-tile view**: Each tile exposes a **view** of size at most `M_TILE_VIEW` (e.g. top by mass, tie-break by primitive_id) for association. This caps the candidate pool per tile.
- **Per-measurement candidates**: For each measurement, candidate tiles = fixed stencil around the measurement’s tile (radius `R_STENCIL_TILES`). Candidate primitives = primitives in those tiles’ views. Total per measurement is bounded by `N_STENCIL_TILES * M_TILE_VIEW`; downselect to `K_ASSOC` by fixed-cost top-k (e.g. by cost or distance proxy), deterministic.
- **Indexing**: Store `(cand_tile_id[i,k], cand_slot[i,k])` (or equivalent); **no** flattening to a single global primitive index for the full map. Fusion uses tile-local scatter-add.

#### 5.7.5 Novelty and insertion (additive every scan)

- **Novelty mass**: For each measurement i, transported mass into the map is `m_i = sum_k π[i,k]`. Measurement mass budget is `a[i]` (from fixed rule, e.g. valid weight). **Novelty** `n_i = clamp(a[i] - m_i, 0, a[i])`. Continuous; no gate.
- **Insertion tile**: Each measurement is assigned to an insertion tile deterministically (e.g. `tile_id` of measurement position). No “if empty then insert”; insertion runs every scan for the active set.
- **Per-tile insert**: For each active tile, select a fixed number `K_INSERT_TILE` of measurements with highest `s_i = n_i * w_i` (tie-break deterministic). Evict `K_INSERT_TILE` lowest-mass slots in that tile (tie-break by `primitive_id`). Insert new primitives into those slots (world-frame Λ, θ, η, color; initial mass = novelty or n_i*w_i; timestamp = scan time). **Insert count per scan is fixed**: `N_ACTIVE_TILES * K_INSERT_TILE` (constant). No “insert only when map empty”; that condition is forbidden.

#### 5.7.6 Fusion: tile-local streaming reduce-by-key (engineering contract)

- **Block size**: `B_PAIR` is a compile-time constant. Process `(i,k)` pairs in blocks of size `B_PAIR`.
- **Per block**: Compute deltas; scatter-add into **per-tile** accumulators of size `M_TILE` (not global). Do **not** materialize a flat array of size N_meas × K_assoc for fusion; stream contributions into per-tile buffers.
- **End of scan**: Apply deltas once per tile; PSD/domain projection at most once per primitive per scan (or declared ApproxOp with certificate). Color update: responsibility-weighted blend as today.
- **Certificate assertion**: The CertBundle fields `largest_tensor_shape`, `segment_sum_k`, `alloc_bytes_est` (see §2.2) **must** be asserted in tests against declared budget limits. This is where “fixed-cost” is verified in practice.

#### 5.7.7 Maintenance and certificates

- **Cull/forget/merge-reduce**: Unchanged in nature but **tile-scoped** for active tiles. Inactive tiles are untouched (certificate records “inactive tiles not updated”).
- **Certificates (required)**:
  - Active set: `n_active_tiles`, `tile_ids_active` (or hash), tile cache hits/misses.
  - Association: `candidate_tiles_per_meas`, `candidate_primitives_per_meas_mean` / p95, OT marginal defect if applicable; `sum_a`, `sum_m`, `sum_novel`.
  - Insert: `insert_count_total` = N_ACTIVE_TILES * K_INSERT_TILE (constant), `insert_mass_total`, `insert_mass_p95`, `evicted_mass_total`; optional `insert_mean_novelty`, `insert_p95_novelty`.
  - Stability: e.g. `evict_mass_p95`, `coverage_growth_proxy` (for auditing).
- **Event log (optional)**: Append-only per-scan log (e.g. scan timestamp, active tile ids, inserted primitive payloads) for post-run replay without re-running the pipeline.

#### 5.7.8 Invariants (recap)

#### 5.7.8 Invariants (recap)

- **No "insert only when empty"**: Insertion is additive every scan, with fixed budget per tile. Initialization is the degenerate case of empty tiles in the atlas.
- **Fixed-cost**: Active set size, candidate pool cap, K_ASSOC, K_INSERT_TILE, K_SINKHORN are constants. No superlinear materialization (§1.7).
- **No gating**: Active set and novelty are fixed-window and continuous; no branch on "did we observe" to decide whether to run an operator.
- **Determinism**: Tile ID ordering and eviction tie-breaks (primitive_id) ensure reproducible replay and audits.

### 5.8 Atlas = charted manifold (concept → implementable contract)

**Principle:** Tiles enforce bounded compute. Charts enforce bounded linearization error. The certificate system is the glue: it quantifies when those bounds are strained, and adaptation responds only via continuous scaling (never gates).

#### 5.8.1 Chart type (what a chart is in code)

- **Chart** = `(anchor_id, X_anchor, local_tangent_ordering, valid_radius, overlap_policy)`.
- `valid_radius` is **not gating**; it only affects **transport accuracy certificates** (how far the retraction is trusted), never whether an operator runs.

#### 5.8.2 Chart transition operator (mandatory primitive)

- **Transport(z, chart_a → chart_b)** with:
  - declared retraction (e.g. SE(3) ⊕ with right-increment),
  - certificate fields: `transport_delta`, `2nd_order_error_proxy`, `chart_overlap_frac`.

#### 5.8.3 Tile ↔ chart relationship

- **State charts**: Pose, velocity, bias, dt, extrinsics are charted around `X_anchor` (already in §1.2). Belief and evidence live in these charts; transport between charts is via the chart transition operator.
- **Map tiles**: Tiles are **indexing budgets**, not coordinate charts (primitives stored in world frame). In addition:
  - Each tile may store a **local linearization handle** for association costs (e.g. cached `X_world←anchor` at last update) so association costs can be computed with bounded error and certified.
- This split is mandatory: charts are for state linearization and transport; tiles are for bounded map compute and locality.

### 5.9 Per-scan map update objective (internal objective only)

The per-scan map update is an **approximate coordinate-ascent step on a declared free-energy / surrogate objective** under fixed compute budgets. Expected vs realized is internal-objective only (never ATE/RPE).

- **Objective name** for map update (even if approximate): e.g. `F = (expected log-likelihood under soft association) − (OT regularization terms) − (budget penalties as ApproxOp)`.
- **Contract:** `ExpectedEffect.objective_name` for map-related operators must be one of a declared set, e.g. `scan_free_energy`, `association_dual_gap_proxy`, `map_negloglik_proxy`, or another named internal objective. This makes the emergence of SLAM from fixed-cost steps on a declared objective explicit and auditable.

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

#### 6.3.1 DeskewConstantTwistSufficientStats (recommended path)

If a downstream operator consumes LiDAR primarily through voxel/primitive sufficient statistics (mass, first/second moments, plane-fit covariance), deskew must operate on those sufficient statistics using fixed time bins, not per-point warping, unless declared otherwise. Time-binning count `B_TIME` is a hard constant and must be reported in the manifest. The approximation magnitude must be certified via `deskew_bin_ang_max` and `deskew_bin_trans_max`. This aligns with the “no hidden iteration, fixed budget” design.

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

#### 6.4.1 Consistency evidence operators (cross-sensor; branch-free)

Cross-sensor consistency is first-class evidence, not diagnostics-only. **Consistency evidence operators** (branch-free):

- **Inputs**: Dyaw (or pose/velocity) estimates from gyro, odom, LiDAR MF, and (when available) camera.
- **Output**: An information-form Gaussian factor (or vMF/MF factor projected at `z_lin`) that penalizes disagreement between these estimates.
- **Uncertainty**: When dependence cannot be modeled exactly, use **own IW inflation** on the residual (learn inflation covariance on the SPD cone); aligns with §4.2.1 and the dependence policy. No gating: continuous penalty and continuous conservatism.

This prevents “two sensors silently fighting” while staying faithful to no-gates and the “disagreement is information” rule.

### 6.5 Pose evidence backend (explicit, single path only)

The only allowed pose evidence backend is `primitives`. It must be declared in the runtime manifest.

- `pose_evidence_backend="primitives"`: primitive alignment / visual+LiDAR pose evidence (Laplace at `z_lin`)
  - Implementation: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/visual_pose_evidence.py`

Contract (hard): The bin-based pose evidence backend and any bin-map implementation are **forbidden**. They must not exist in the codebase. Retaining them is a spec violation; they must be removed, not deprecated.

### 6.6 Adaptive noise (Inverse-Wishart; implemented)

Noise is modeled as IW random variables with per-scan commutative sufficient-stat updates.

Implementation anchors:
- Process noise IW: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/inverse_wishart_jax.py`
- Measurement noise IW: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/measurement_noise_iw_jax.py`
- Priors/hyperpriors: `fl_ws/src/fl_slam_poc/fl_slam_poc/common/constants.py`

### 6.6.0 Noise and uncertainty roles (normative)

- **Noise modeling roles (normative):**
  - Measurement noise affects evidence strength within a likelihood (first-order uncertainty).
  - Wishart / Inverse-Wishart models represent uncertainty *about* that noise (second-order uncertainty).
  - **Wishart** distributions parameterize uncertainty over **information matrices** (precision, Λ).
  - **Inverse-Wishart** distributions parameterize uncertainty over **covariance matrices** (Σ).
  - All adaptive noise behavior in GC v2 must be expressible using these two layers only.

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

### 6.6.4 Mandatory IW adaptation triggers

- **Mandatory IW adaptation triggers (non-exhaustive):**
  1. Any ApproxOp that projects out-of-family MUST introduce or update an IW factor (and log it in CertBundle).
  2. Any operator acting on a weakly observable subspace (as indicated by certificates, e.g. overconfidence.z_to_xy_ratio, near_null_count) MUST update IW for that subspace.
  3. Any evidence fusion where independence cannot be justified MUST inflate IW (or use explicit cross-covariance) rather than down-weight evidence heuristically.
  4. Any buffer saturation or scan-window truncation MUST be logged and MAY update IW continuously.

### 6.6.5 Scope and lifetime of IW factors

- **Scope and lifetime of IW factors:**
  - IW factors may be attached at the level of: individual evidence operators, sensor streams, tiles/charts, or declared subspaces (e.g. Z, yaw, dt). The chosen scope must be declared and logged.
  - IW updates are persistent across scans unless explicitly reset by a declared ExactOp (e.g. re-initialization from prior).
  - IW adaptation is continuous and monotonic in the sense that evidence weakens confidence (inflates covariance) or strengthens it (tightens covariance); ad hoc resets or "inflate once and forget" are forbidden unless justified by a declared operator with certificate.

### 6.7 `FusionScaleFromCertificates` (ExactOp)

Computes continuous trust `alpha` from certificates and manifest constants.

Implementation anchor: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/fusion.py`.

Control-law contract (hard):
- Treat this mapping as a control law, not glue code.
- Must be deterministic, continuous, and bounded; must never gate factors.
- Must be monotone in “risk” proxies (worse conditioning/support/excitation/overconfidence ⇒ no larger alpha).
- Must use certificates including `influence.power_beta` when tempered posteriors apply; β contributes to Frobenius trigger magnitude via |1−β|.
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

### 6.12 Loop closure (contract: late evidence + budgeted recomposition)

Loop closure in GC is **late evidence** plus **budgeted recomposition** only. No iterative global optimization.

- **LoopFactor**: A loop-closure constraint is just another evidence object in the same information form (e.g. Gaussian factor on pose difference or relative constraint). It enters the same fusion path as other evidence.
- **BudgetedRecomposition**: The only allowed ApproxOp for large-scope (e.g. loop-corrected) state updates is an explicit **budgeted recomposition**: declared objective, fixed-cost scope reduction, Frobenius correction when applicable, and expected vs realized effect logged. Any scope reduction must be an explicit approximation with certificate.
- **No optimizer creep**: No “run optimizer until convergence”; no hidden iteration. Loop closure does not change the operator set or introduce new solver loops.

---

## 7) Runtime Manifest (Required)

The runtime must emit a manifest listing:
- chart id, dimensions, budgets, epsilons
- explicit backend selections (`pose_evidence_backend`, `map_backend`)
- `bev_backend_enabled: bool`, `bev_views_n: int` (0 if disabled)
- unbalanced OT parameters (ε, τ_a, τ_b or equivalent) when association is active (§5.7.3.1)
- a short list of "resolved operators/backends" used at runtime
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

## 9) Forbidden / No Retained Legacy

The following must not exist in the implementation. Legacy must not be retained; deprecated paths cannot be used. If present, remove them.

- Bin-based pose evidence backend and bin-map: **forbidden**. One canonical map backend (PrimitiveMap); one pose evidence path (primitives).
- UT-based deskew moment matching and UT-cache-driven LiDAR quadratic regression evidence: not canonical; must not be active or retained as code paths.
- Wahba SVD / Translation WLS as primary pose extraction path: not canonical; must not be retained.
- Fixed measurement covariances as truths (non-IW): adaptive IW is canonical; fixed covariances only as priors/hyperpriors. No retained "constant noise" code paths.

---

## 10) Declared Non-Idealities (sync with implementation and roadmap)

The spec is "current as implemented in code." Where the implementation or roadmap knowingly diverges from the ideal, the spec must list it so the document remains the single source of truth. Each non-ideality states: **current behavior** (what the code does now), **end state** (what the spec requires), and **evidence/uncertainty/certificate** mapping. Code anchors are given where useful.

### 10.1 Table: Non-idealities and resolution

| Non-ideality | Current (what code does now) | End state (spec / target) | Planned resolution |
|--------------|------------------------------|---------------------------|--------------------|
| **Pose–twist as joint observation** | **Partial:** Odom pose evidence (6D) and odom twist evidence (velocity, yaw-rate) are separate factors; a third factor `pose_twist_kinematic_consistency` penalizes pose-change vs integrated twist (`odom_twist_evidence.py`, `pipeline.py`). No declared joint observation or pose–twist cross-covariance. | Per §4.2.1: odom pose and odom twist must be modeled as a single joint observation or with explicit cross-covariance. | Explicit joint factor on [pose; twist] or cross-covariance block; reflect kinematics in evidence. |
| **Cross-sensor consistency as evidence** | **Diagnostics only:** `dyaw_gyro`, `dyaw_odom` (and optionally LiDAR/visual yaw) are computed and logged in `ScanDiagnostics` and pipeline NPZ (`diagnostics.py`, `pipeline.py`). Not fused as evidence. | Cross-sensor agreement (e.g. dyaw from gyro / odom / LiDAR) must enter as evidence with own IW inflation when dependence is uncertain. | Consistency evidence operators (§6.4.1); info-form factor penalizing disagreement; no gating. |
| **Use more raw message info** | **IMU:** Backend reads `angular_velocity` and `linear_acceleration` only; message covariances and orientation are not consumed (`backend_node.py`; `use_imu_message_covariance` exists but is not wired). **LiDAR:** Intensity is optional in PointCloud2 schema and used only for rendering κ modulation (`rendering.py`); not used in evidence or IW priors. | IMU message covariances (and orientation when present) must feed IW priors or evidence construction. LiDAR intensity may feed weighting/features in a declared way. No hidden constants. | Wire IMU covariances/orientation into backend evidence or IW; wire LiDAR intensity into evidence or declared prior; no new gates. |
| **Z diffusion / observability** | **Partial:** Planar z and v_z priors exist (`planar_prior.py`); overconfidence sentinels `z_to_xy_ratio`, `dt_asymmetry` in certs (§2.2); fusion uses z quality for scaling. **Gap:** Process noise Q trans block uses a single scalar `GC_PROCESS_TRANS_DIFFUSION` for x,y,z (`inverse_wishart_jax.py`). `GC_PROCESS_Z_DIFFUSION` is defined in `constants.py` but **never used** in Q. | Q trans block must use distinct z diffusion (e.g. `GC_PROCESS_Z_DIFFUSION`) for planar robots so prediction does not add full trans diffusion to z. Z observability remains partially covered by sentinels and planar priors; optional planar/height constraint factor. | Use `GC_PROCESS_Z_DIFFUSION` for z in process-noise trans block (IW structure: trans block diagonal [TRANS, TRANS, Z_DIFFUSION]). Keep sentinels; add Z diffusion discipline to manifest/certs. |
| **Atlas / map growth** | **Current:** Single global `PrimitiveMap`; insert runs only when `map_view.count == 0` (`pipeline.py`). After first scan, only fuse/cull/forget; map does not grow. | Per §5.7: Atlas of tiles; additive insertion every scan with fixed K_INSERT_TILE per tile; no "insert only when empty." | Implement AtlasMap, tile-local fuse/insert, novelty-driven insertion; remove insert-only-when-empty condition. |
| **Rotation / tilt observability** | Planar priors and odom 6×6 covariance (including roll/pitch) are used. No explicit tilt prior; no systematic audit of gravity/IMU alignment or odom roll/pitch inflation. | For planar robots: optional tilt prior (roll≈0, pitch≈0); odom roll/pitch variance inflation when bag publishes tight tilt; gravity/IMU alignment auditable. | Document in PIPELINE_DESIGN_GAPS; optional tilt prior and roll/pitch inflation; startup audit of gravity vs IMU. |

### 10.2 Other roadmap items (
- **Loop closure:** Contract in §6.12 (late evidence + budgeted recomposition); not yet implemented; no "Declared Non-Ideality" row until we add a loop detector.

### 10.3 Maintenance

This section shall be updated when gaps are closed or new non-idealities are accepted. When a row is resolved, move it to a "Resolved" subsection or remove it and note in CHANGELOG so the spec does not drift from "ideal" while the repo is "real."

---

## 11) Statistical–Topological Glue and Renderable Atlas

### 11.1 Motivation and principle

GC v2 operates on a **statistical manifold**: beliefs, measurements, and map primitives are elements of exponential families with natural parameters, dual coordinates, and well-defined information geometry.

A statistical manifold is still a manifold. As such, it admits:

- an **atlas** (local chart domains),
- **overlap structure** between charts,
- a **nerve** (simplicial/topological object encoding overlap),
- and derived **geometric realizations** (visualizable surfaces, connectivity, and structure).

**Principle:** Statistical glue (likelihoods, responsibilities, natural-parameter fusion) must induce a corresponding topological glue that can be visualized, audited, and reasoned about — *without feeding back discrete decisions into inference*. Inference remains continuous, branch-free, and fixed-cost. Topology and rendering are **derived**, output-side constructions.

### 11.2 Statistical overlap ⇒ topological overlap

#### 11.2.1 Overlap is defined by transported mass (π), not thresholds

Let tiles/charts be indexed by α, β; measurement primitives by i; map primitives by k; and π(i,k) ≥ 0 be the transported mass from measurement i to map primitive k (output of soft association).

Define the **tile–tile overlap weight**:

`W_αβ := Σ_i Σ_{k∈α} Σ_{k'∈β} π(i,k) π(i,k')`

Interpretation: If a measurement distributes mass across two tiles, those tiles overlap statistically. This is the direct statistical analog of overlapping coordinate charts in a manifold atlas.

Note: Statistical overlap defined by transported mass need not coincide with Euclidean spatial overlap; this is intentional, as topology reflects inferential coupling rather than raw distance alone.

Properties: Continuous (no gates); symmetric; derived entirely from inference outputs; zero if and only if there is no statistical coupling. No hard assignment (“which tile owns this measurement”) is ever introduced.

### 11.3 The atlas nerve (topological object)

From the overlap weights W_αβ, define a **weighted nerve**:

- **Vertices**: tiles/charts
- **Edges**: (α, β) with weight W_αβ
- **Higher simplices**: defined by multi-way overlap (e.g. triple products of π)

This nerve encodes connectivity, loops, and branching in the map; it evolves continuously as inference progresses; it can be thresholded or filtered **only for visualization or analysis**. Filtrations over W_αβ (e.g. persistent homology) are allowed **only as derived views** and must not alter inference.

The atlas nerve is a derived object, computed per scan or over a bounded sliding window, from current or cached association outputs; it is not a persistent state variable of the inference backend.

### 11.4 From statistical atlas to renderable geometry

#### 11.4.1 Canonical renderable primitive (mandatory)

Every map primitive **must** be renderable as a continuous surface element. Each primitive SHALL expose:

- Mean position: μ_world ∈ ℝ³
- Covariance or precision: Σ_world or Λ_world (3×3)
- Mass / support: m ≥ 0
- Color payload: RGB ∈ [0,1]³ (if available)
- **Directional distribution: vMF natural parameters η ∈ ℝ³ (MANDATORY)**

vMF is not optional. Every primitive carries a directional distribution, even if initially weak or isotropic. This makes the map a **mixture of oriented, colored Gaussian surface elements**.

Any canonical map publisher (e.g. PrimitiveMapPublisher) SHALL be able to export all fields required by §11.4.1; point-only exports are permitted only as explicitly labeled debugging views.

#### 11.4.2 Rendering semantics (output-side, non-inferential)

Rendering is a **pushforward** of statistical quantities into image space. Canonical rendering modes (any may be implemented; all respect the same contract):

**(A) Gaussian splatting (EWA)**  
Project μ_world into the view; project Σ_world into image space; draw an elliptical kernel with weight α ∝ f(m, support); accumulate using alpha compositing.

**(B) Oriented surfels**  
Use Σ eigenstructure to define tangent-plane extent; use vMF mean direction as surface normal; render as oriented disks.

Point rendering (μ only) is permitted **only** as a debugging view and is not canonical.

### 11.5 Viewpoint-dependent rendering via vMF (no spherical harmonics)

This section is **non-negotiable** and intentionally excludes spherical harmonics (SH).

#### 11.5.1 vMF as the directional radiance primitive

Each primitive carries a vMF distribution on S²: `p(n) ∝ exp(ηᵀ n)`, with η = κ μ, |μ| = 1. Under a view rotation R ∈ SO(3): η′ = R η. This pushforward is exact, closed-form, and in-family.

#### 11.5.2 View-dependent intensity / shading

Given view direction v: `I(v) ∝ exp(η′ᵀ v)`. This yields smooth, anisotropic appearance; view-dependent highlights; no basis truncation; no SH ringing or bandwidth artifacts. Multiple vMF lobes per primitive are allowed and encouraged (e.g. B=3), with additive natural parameters. When multiple vMF lobes are present, view-dependent intensity is computed from the resultant natural parameter or via additive superposition of lobe contributions; both are closed-form and in-family. This enables **novel view synthesis** directly from information geometry.

### 11.6 Statistical–topological consistency invariants (hard)

1. **No topological object may gate inference.** Nerve structure, loops, components, etc. are diagnostics or proposal generators only. Any feedback must be continuous (e.g. influence scaling), never discrete.
2. **No rendering-driven pruning.** Visibility, opacity, or shading must not remove primitives from inference.
3. **Charts/tiles remain inference-side.** Topological constructs do not define active sets or operator selection.
4. **vMF is mandatory.** Any primitive lacking a directional distribution is incomplete. Isotropic vMF (κ≈0) is allowed; absence is not.

### 11.7 Why this is principled (not ad hoc)

- The atlas nerve is the **nerve of a statistical cover**, not a heuristic graph.
- Overlap is defined by **likelihood transport**, not distance thresholds.
- Rendering is a **pushforward of exponential-family structure**, not a separate model.
- View dependence emerges from **SO(3) action on natural parameters**, not basis tricks.

Coherent rendered geometry is therefore a necessary but not sufficient indicator of belief correctness; it reflects internal consistency under the current model and evidence.

This completes the translation: **Information Geometry ⇒ Atlas ⇒ Topology ⇒ Renderable Geometry** without violating GC’s core constraints.

---

## 12) JAX JIT + Host/Device Runtime Discipline (Fixed-Cost Implementation Contract)

### 12.1 Purpose

GC v2's modeling contract requires fixed-cost, branch-free operator execution with auditable compute. To make this true in practice, the implementation must obey strict JAX discipline:

- no Python control-flow in jitted paths,
- no repeated host↔device transfers,
- no dynamic shapes that create unbounded compilation churn,
- no silent fallbacks to non-jitted code.

This section is normative: violations are spec violations.

### 12.2 Single-device, single-path execution (no hidden fallbacks)

Hard rules:

1. Operators that are declared JAX/JIT-backed SHALL have exactly one canonical implementation path.
2. "Try JAX, else NumPy" or "if tracing fails, run Python" is forbidden.
3. If a JAX operator cannot be compiled under declared budgets/shapes, it must **fail fast** and emit a certificate explaining why (e.g. shape mismatch, non-static loop bound).

### 12.3 Shape discipline (static shapes, fixed budgets)

Hard rules:

1. All arrays consumed by jitted operators must have **static shapes** determined by compile-time budgets: `N_POINTS_CAP`, `MAX_IMU_PREINT_LEN`, `K_ASSOC`, `B_PAIR`, etc.
2. Variable-sized inputs must be represented as: padded arrays + `valid_mask`, or fixed-size chunked streams with deterministic chunk sizes.

Forbidden: dynamic-length concatenations inside jitted functions; Python lists of arrays; shape-dependent branches.

**Certificate requirement:** Each operator must record `largest_tensor_shape`, `alloc_bytes_est`, and key budget values used in the call. These must be asserted in tests.

### 12.4 Loop discipline (no Python loops, only JAX control flow)

Hard rules:

1. No Python `for`/`while` loops in any code path intended for JIT compilation.
2. All iteration inside JIT must use: `jax.lax.fori_loop`, `jax.lax.while_loop` (only if bounded by a compile-time constant), or `jax.lax.scan` for fixed unrolled structure.
3. "Until tolerance / until convergence" loops are forbidden (§1.5). Only fixed-iteration solvers (e.g. fixed Sinkhorn iterations) are allowed.

**Certificate requirement:** Any fixed-iteration operator must report the exact iteration count used (compile-time constant) and any convergence diagnostics as *diagnostic-only* outputs (must not affect control flow).

### 12.5 Host↔device transfer discipline (no NumPy thrash)

Hard rules:

1. Within a scan/hypothesis update, inputs shall be transferred to device **at most once** per stream and reused.
2. Operators must not repeatedly convert between `np.ndarray` ↔ `jnp.ndarray`, host scalars ↔ device scalars, Python objects ↔ traced values.
3. All intermediate computation for JIT operators must remain in JAX device arrays until the operator boundary returns.

Forbidden: `np.array(jnp_array)` inside the pipeline; `jnp.array(np_array)` inside inner loops; `.item()` / `float(x)` / shape queries that force host synchronization.

Recommended: A `DeviceBatch` wrapper type that holds all per-scan arrays as `jnp.ndarray` plus masks and metadata. Explicit "boundary conversion" only at: ROS I/O boundaries, logging/serialization boundaries, visualization/output-side exports.

**Certificate requirement:** Each operator must record a lightweight proxy for transfer/sync risk (e.g. `host_sync_count_est`, `device_to_host_bytes_est`). Exact measurement is optional; nonzero values must be explainable.

### 12.6 JIT caching discipline (avoid recompilation churn)

Hard rules:

1. JIT compilation must be cached and amortized across scans.
2. Operators must not introduce scan-dependent polymorphism in shapes or dtypes that triggers recompiles.
3. All static arguments must be flagged as static (`static_argnums`/`static_argnames`) and must be truly constant.

Forbidden: passing Python dicts/objects into jitted functions; passing variable-length lists as arguments; switching dtypes based on data.

**Certificate requirement:** Operators should optionally record `jit_recompile_count` (if instrumentation exists) or a proxy warning when shapes/dtypes differ from prior calls.

### 12.7 Streaming reduce-by-key in JAX (map fuse contract)

Map fuse must respect §1.7 (no superlinear materialization) *and* be JIT-safe: use fixed blocks `B_PAIR` and tile-local accumulators; use scatter-add into fixed-size buffers; avoid building a full (N_meas, K_assoc) contribution tensor. This is mandatory for atlas/tile fusion (§5.7.6) and must be implemented using JAX primitives compatible with JIT.

### 12.8 Boundary rule: what may remain in NumPy/Python

Allowed (explicitly outside inference-critical JIT core): ROS message parsing and serialization; visualization exports (RenderablePrimitiveBatch, MarkerArray, point debug views); diagnostics formatting (strings, logs); offline analytics (topology persistence, etc.).

Forbidden: performing inference math in NumPy "because it's easier"; computing association/likelihood core in Python loops.

If a component is outside JIT, it must be labeled **Output-side** or **I/O-side** and must not affect inference operators except via continuous scalars already declared in the manifest.

### 12.9 Testing requirements (non-negotiable)

1. **Budget tests:** Enforce that allocations and shapes remain within declared budgets.
2. **No recompilation tests:** Run multiple scans and assert JIT cache stability (if tooling available).
3. **No host sync tests:** Assert absence of known sync triggers in critical paths (best-effort).
4. **Determinism tests:** Repeated runs with same inputs produce identical outputs (to floating tolerance) and identical certificates.

### 12.10 Pipeline concurrency and scan clock (multi-threading contract)

GC v2 is permitted to use multi-threading for I/O, buffering, and operator execution, but concurrency must not change the mathematical program. The system SHALL operate on a single deterministic **scan clock** defined by the slowest sensor stream used by the backend (typically LiDAR). All other sensor streams are treated as asynchronous evidence sources that are **sampled/aggregated** into fixed-budget windows aligned to this scan clock.

**Scan clock contract (hard):**

- A "scan" is the atomic unit of inference update.
- One scan triggers exactly one execution of the canonical per-scan operator order (§8).
- The scan rate is the effective update rate of the slowest required stream.
- Faster streams may update internal buffers between scans but must not trigger partial pipeline runs.

This ensures fixed-cost, reproducible updates and avoids implicit multi-rate "modes".

### 12.11 Multi-threaded buffering model (single subscription per stream, deterministic ring buffers)

Each sensor stream (IMU, odom, camera, LiDAR, etc.) is consumed by exactly one subscriber/consumer that writes into a bounded ring buffer. Buffers are read **only** at scan boundaries.

**Hard rules:**

1. **One subscription per stream** (no duplicate consumers).
2. **Bounded buffers** with compile-time maximum capacity per stream: IMU: fixed maximum samples per scan window (`MAX_IMU_PREINT_LEN`) with padding + mask; Odom: fixed maximum samples per scan window; Camera: fixed maximum feature batches per scan (or one fused batch per scan).
3. **Deterministic selection** of the buffer slice used for a scan: selection rule uses timestamps, not arrival order; ties are resolved deterministically.
4. **Missing data is represented by masks**, not branching: if fewer than capacity samples exist, pad with zeros and set `valid_mask=0`; the operator still runs with continuous influence scaling (never "skip IMU factor").

**Required timestamping:** Each buffered message must carry a source stamp. When combining streams, the scan boundary is defined by the slow stream's stamp, and all other streams are windowed relative to it.

**Concurrency rule:** Buffer writes may happen concurrently. The per-scan read must observe a consistent snapshot (e.g. via lock-free versioning or a single short lock) such that the slice used for the scan is immutable during the scan computation.

### 12.12 Parallel operator execution ( allowed and encouraged but ONLY if it preserves semantics)

Operators may execute in parallel across hypotheses or across independent subcomputations **only if** doing so is semantics-preserving and does not introduce data-dependent scheduling.

Allowed examples: run per-hypothesis pipelines in parallel (same operator order, same budgets); run map association and pose evidence preparation in parallel if they consume the same scan snapshot and produce deterministic outputs.

Forbidden: parallel execution that changes the set of operators run; adaptive "race-to-finish" updates; executing an operator conditionally because another thread "didn't finish."

**Determinism contract:** Given identical buffered inputs and manifest constants, the pipeline must produce identical outputs (within floating tolerance), regardless of thread scheduling. Any nondeterminism from parallel reductions must be bounded and, where possible, avoided by deterministic reduction order or fixed reduction schemes.

**Certificates (required):** Per scan: record `scan_stamp`, `buffer_window_bounds` for each stream, and `valid_counts` (IMU samples used, odom samples used, etc.). Record whether buffers hit capacity caps (saturation), as this is a declared approximation (not a gate).

### 12.13 Why this matters

This makes the intended behavior explicit: **The backend is scan-driven**, not "sensor-event driven." **Multi-threading improves throughput**, but does not change the inference program. Faster sensors contribute more information inside the scan window, but never trigger extra pipeline runs.

---

## 13) Planned Extensions: Categorical and Hierarchical Semantic Payloads (Roadmap)

### 13.1 Motivation

Once the geometric core of GC v2 is stabilized and validated, the next natural extension is **semantic understanding**: objects, rooms, places, and functional regions. This is not treated as a separate system layered on top of SLAM, but as an **additional belief payload** integrated into the same information-geometric framework.

The guiding principle remains unchanged: Semantic understanding is not imposed by discrete decisions or post-hoc classification; it must **emerge from continuous, probabilistic evidence**, composed additively and audited via certificates, just like geometry.

### 13.2 Semantics as exponential-family belief payloads

Semantic information will be represented using **categorical exponential-family distributions**, attached to the same primitives that already carry geometry, direction (vMF), and appearance.

Canonical choices include: **Categorical / Multinomial likelihoods**; **Dirichlet priors and posteriors** (natural parameters = pseudo-counts).

Each map primitive may carry one or more categorical belief vectors:

- **SemanticPayload**: `level_id` (e.g. "object", "room", "function"); `eta_cat` (K_level,) Dirichlet natural parameters (pseudo-counts); `valid_mask` (K_level,) optional, for fixed-budget vocabularies.

Fusion is **closed-form and exact**: evidence contributes additive pseudo-counts in natural coordinates; multiple semantic evidence sources compose via simple addition (product-of-experts in probability space). This preserves associativity, order-robustness, and fixed-cost computation under declared category budgets.

### 13.3 Hierarchical categorical structure (objects → rooms → functions)

GC explicitly targets **hierarchical semantics**, not flat labels. Examples: object → furniture → room type → functional area; surface → structural element → navigational affordance.

Hierarchy is represented via **multiple coupled categorical payloads**, one per level, with declared parent–child mappings.

Two supported hierarchy mechanisms (future):

1. **Deterministic aggregation (ExactOp):** Parent-level pseudo-counts are computed as declared linear aggregations of child-level counts (e.g. sum of object evidence induces room evidence).
2. **Probabilistic coupling (ApproxOp):** Cross-level projections using a declared conditional table or learned mapping, certified as an approximation.

Hard rule (consistent with GC invariants): Hierarchical coupling must never introduce gating ("only classify room if object confidence > τ" is forbidden). All cross-level influence must be continuous and logged via certificates.

### 13.4 Semantic evidence sources (non-exhaustive)

Planned semantic evidence operators may include:

- **Vision classifiers:** Per-feature or per-primitive soft class probabilities (logits → pseudo-counts).
- **Appearance consistency:** Color/texture similarity across primitives contributes weak semantic evidence.
- **Geometric affordances:** Shape, orientation, and spatial configuration (e.g. vertical planes → walls; horizontal planes → floors).
- **Contextual priors:** Co-occurrence and adjacency patterns (e.g. chairs near tables), expressed as weak categorical coupling.

All semantic evidence must obey the same contract as geometric evidence: soft (no thresholds), fixed-budget, additive in natural parameters, certifiable.

### 13.5 Interaction with geometry, topology, and rendering

Semantic payloads are **co-located** with geometric primitives and tiles. This enables: **Semantic rendering** (colorization, highlighting, or transparency driven by categorical posteriors; output-side only); **Semantic topology** (atlas nerve nodes and overlaps annotated with dominant semantic mass, e.g. "this connected region is a corridor"); **View-dependent semantics** (vMF-based directional payloads can modulate semantic appearance without spherical harmonics, e.g. façade vs interior).

Important invariant: Semantic information must not redefine the atlas, tile boundaries, or active sets. It annotates belief; it does not control inference structure.

### 13.6 Certificates and overconfidence for semantics (future)

Semantic reasoning introduces new failure modes (confirmation bias, label lock-in). GC anticipates this by extending the certificate system.

Planned semantic certificates: `semantic_support_frac` (effective ESS per category); `semantic_entropy` (uncertainty vs overconfidence); `semantic_conflict_score` (disagreement between evidence sources); `semantic_growth_rate` (rate of pseudo-count accumulation).

As with geometry: rising confidence without corresponding new evidence must trigger **continuous conservatism**, never gating; semantic payloads may be tempered (power-EP style) exactly like geometric likelihoods.

### 13.7 Fixed-cost and no-gates discipline (non-negotiable)

All semantic extensions must respect GC's core invariants: fixed category vocabularies per level (compile-time budgets); padding + masks for variable evidence; no data-dependent branching; no discrete "recognized / not recognized" states; no semantic backends that run only when geometry is "good enough". Semantics enrich understanding; they must not destabilize geometry.

### 13.8 Status and integration plan

**Current status:** Semantic/categorical operators are **not active** in the runtime pipeline.

**Integration plan:**

1. Stabilize and validate geometric + topological core (§0–§12).
2. Add categorical payload data structures (inactive by default).
3. Introduce semantic evidence operators as optional, declared components.
4. Extend rendering and diagnostics to surface semantic belief.
5. Only then consider semantic-aware planning or affordances.

This staged approach ensures that semantics are built on a solid, auditable geometric foundation.

### 13.9 Why this section is valuable now

Including because it does three critical things: (1) It **signals architectural intent** without over-committing implementation. (2) It reassures future contributors that semantics are not an afterthought or bolt-on. (3) It shows that GC's information-geometric design *scales naturally* from geometry → topology → meaning.

Most importantly, it preserves the central philosophy: Understanding is not classification; it is the gradual alignment of belief with the structure of the world — geometric, topological, and eventually semantic.
