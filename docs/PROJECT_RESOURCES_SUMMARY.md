# Information Geometry & Active Inference Project Resources

A comprehensive summary of reusable components from two key projects in this workspace.

---

## Table of Contents

1. [Info_Geo-FORK1](#info_geo-fork1)
   - [Core Information Geometry](#core-information-geometry)
   - [Cognitive Agent Pipeline](#cognitive-agent-pipeline)
   - [V2V Collaborative Perception](#v2v-collaborative-perception)
2. [AIF2 (Frobenius Active Inference Navigator)](#aif2-frobenius-active-inference-navigator)
   - [Exponential Family Beliefs](#exponential-family-beliefs)
   - [Hierarchical Precision Learning](#hierarchical-precision-learning)
   - [EFE-Based Planning](#efe-based-planning)
   - [ROS2 Integration](#ros2-integration)
3. [Comparison & Integration Opportunities](#comparison--integration-opportunities)

---

## Info_Geo-FORK1

**Location:** [`/home/will/Documents/Coding/Info_Geo-FORK1/`](./Info_Geo-FORK1/)

A comprehensive information geometry framework for cognitive agents, V2V perception, and uncertainty-aware reasoning.

### Core Information Geometry

| Module | Location | Key Features |
|--------|----------|--------------|
| **Geodesics** | [`cognitive_agent/ig_geodesics.py`](./Info_Geo-FORK1/cognitive_agent/ig_geodesics.py) | Exponential/logarithmic maps on Gaussian manifold, geodesic interpolation, Fisher metric distance |
| **Parallel Transport** | [`cognitive_agent/ig_parallel_transport.py`](./Info_Geo-FORK1/cognitive_agent/ig_parallel_transport.py) | Transport vectors along geodesics |
| **α-Connections** | [`cognitive_agent/ig_connections.py`](./Info_Geo-FORK1/cognitive_agent/ig_connections.py) | m/e-geodesics for different exponential families |
| **Bregman Divergences** | [`cognitive_agent/ig_bregman.py`](./Info_Geo-FORK1/cognitive_agent/ig_bregman.py) | KL as Bregman divergence, efficient projections |

#### Key Functions

```python
# From ig_geodesics.py
exponential_map_gaussian(p_mean, p_cov, v_mean, v_cov, t)  # Follow geodesic from p in direction v
logarithmic_map_gaussian(p_mean, p_cov, q_mean, q_cov)      # Tangent vector from p to q
geodesic_interpolation(p_mean, p_cov, q_mean, q_cov, n_points)  # Path between distributions
compute_fisher_metric_distance(p_mean, p_cov, q_mean, q_cov)    # Riemannian distance
slerp(v1, v2, t)  # Spherical linear interpolation for unit vectors
```

#### Mathematical Reference

See [`docs/IG_FORMULAS_REFERENCE.md`](./Info_Geo-FORK1/docs/IG_FORMULAS_REFERENCE.md) for:
- Gaussian KL Divergence (full vs trace-only)
- Natural Gradient Descent (Amari's dual coordinates)
- vMF Normalization Constant
- Hellinger Distance on Gaussian Manifold
- Fisher Information Matrix

---

### Cognitive Agent Pipeline

**Location:** [`SRC/cognitive_agent.py`](./Info_Geo-FORK1/SRC/cognitive_agent.py) (~2570 lines)

A complete **video-to-action** pipeline with information geometry throughout.

#### Core Primitives

| Class | Description |
|-------|-------------|
| `GaussianSplat` | 3D primitive: position (μ, Σ) + vMF motion (direction, κ) + SH color |
| `ExponentialFamilyAttention` | Change detection with natural gradient learning |
| `SelectiveUpdater` | Classify static/dynamic, update only what changes |
| `VolumetricMemory` | k-d tree spatial indexing, separate static/dynamic indices |

#### Reasoning Modules

| Class | Location | Description |
|-------|----------|-------------|
| `PermanenceReasoner` | Lines 1171-1232 | Object persistence with motion coherence |
| `MemoryUpdater` | Lines 1235-1270 | Selective dynamic handling |
| `AbsenceTracker` | Lines 1273-1299 | Track searched regions |
| `CausalMemory` | Lines 1348-1435 | Causal links with cycle detection, counterfactual simulation |
| `UncertaintyTracker` | Lines 1438-1498 | Cramér-Rao bounds, confidence regions |
| `HypothesisTracker` | Lines 1518-1583 | Multi-hypothesis tracking with ELBO pruning |
| `PredictiveMemory` | Lines 1586-1645 | Future predictions, surprise measurement |
| `GroundedPlanner` | Lines 1648-1715 | Geodesic path planning using Hellinger distance |

#### Key Distance Functions

```python
# From cognitive_agent.py
hellinger_distance(splat1, splat2)  # Geodesic distance between Gaussians
vmf_similarity(splat1, splat2)       # Motion-aware directional similarity
motion_similarity(dir1, dir2, kappa1, kappa2)  # Directional similarity with concentration
```

#### Integrated Pipeline

[`IntegratedCognitivePipeline`](./Info_Geo-FORK1/SRC/cognitive_agent.py) (Lines 1736-2203):
- `process_frame(rgb, depth, ...)` → splats + metrics
- `predict_future(horizon)` → predictions with uncertainty
- `plan_to_target(label)` → geodesic waypoints
- `query_memory(position, radius)` → nearby splats with permanence check
- `get_causes_of(splat)` / `get_effects_of(splat)` → causal queries
- `find_similar_splats(query)` → Hellinger-based similarity search

---

### V2V Collaborative Perception

**Location:** [`Other_test/`](./Info_Geo-FORK1/Other_test/)

Validated +10.3% mAP improvement on real OPV2V data.

#### Key Files

| File | Lines | Description |
|------|-------|-------------|
| [`v2v_dataset.py`](./Info_Geo-FORK1/Other_test/v2v_dataset.py) | ~725 | OPV2V/V2X-Sim data loaders with real data verification |
| [`opencda_bridge.py`](./Info_Geo-FORK1/Other_test/opencda_bridge.py) | ~1133 | HIS computation, selection logic, mAP evaluation |
| [`gpu_information_geometry.py`](./Info_Geo-FORK1/Other_test/gpu_information_geometry.py) | ~1090 | GPU-accelerated Hellinger/vMF/Gamma kernels |
| [`run_opencda_validation.py`](./Info_Geo-FORK1/Other_test/run_opencda_validation.py) | ~1544 | Comprehensive validation suite |

#### Validation Results

See [`FINAL_VALIDATION_REPORT.md`](./Info_Geo-FORK1/Other_test/FINAL_VALIDATION_REPORT.md):

| Bandwidth | HIS | Random | Gain |
|-----------|-----|--------|------|
| 0.5 KB | 0.530 | 0.480 | **+10.3%** |
| 1.0 KB | 0.648 | 0.587 | **+10.4%** |
| 2.0 KB | 0.809 | 0.741 | **+9.1%** |

**Component Validation (HIS-MI Correlation):**
- Gaussian: ρ = 0.979 ✅
- vMF: ρ = 0.9997 ✅
- Gamma: ρ = 0.9992 ✅

---

## AIF2 (Frobenius Active Inference Navigator)

**Location:** [`/home/will/Documents/Coding/Phantom Fellowship MIT/AIF2/`](./Phantom%20Fellowship%20MIT/AIF2/)

A complete Active Inference robot navigation system for TurtleBot3 in ROS2/Gazebo.

**Philosophy:** Everything is either inferred online or emerges from free energy minimization — only physical motor limits are configured.

### Exponential Family Beliefs

**Location:** [`frobenius_nav_v2/frobenius_nav/core.py`](./Phantom%20Fellowship%20MIT/AIF2/frobenius_nav_v2/frobenius_nav/core.py) (Lines 48-465)

| Class | Natural Parameters | Use Case |
|-------|-------------------|----------|
| `GaussianBelief` | η₁ = Λμ, η₂ = -½Λ (information form) | Pose, position |
| `GammaBelief` | α, β | Precision inference |
| `DirichletBelief` | α vector | Policy priors, object types |
| `VonMisesBelief` | c = κ·cos(μ), s = κ·sin(μ) | Orientation |
| `BetaBelief` | α, β | Success/collision rates |

#### Key Methods (all beliefs)

```python
belief.free_energy_from_prior(prior)  # D_KL(self || prior)
belief.entropy()                       # H[q]
belief.fuse(other)                     # Bayesian fusion via sufficient statistics
belief.copy()                          # Deep copy
```

#### GaussianBelief Specifics

```python
GaussianBelief.from_moments(mu, Sigma)           # Create from mean/covariance
GaussianBelief.from_precision(Lambda_mu, Lambda) # Create from precision form
GaussianBelief.from_observation(z, precision)    # Likelihood from observation
belief.mean, belief.covariance, belief.precision # Properties
belief.mahalanobis_sq(observation)               # Mahalanobis distance squared
belief.marginal(indices)                         # Marginalize to subset
```

---

### Hierarchical Precision Learning

**Location:** [`core.py`](./Phantom%20Fellowship%20MIT/AIF2/frobenius_nav_v2/frobenius_nav/core.py) (Lines 469-622)

```
Level 3: PrecisionHyperpriors
         ├── Motion precisions (linear, angular)
         ├── Odometry precisions (position, orientation)
         ├── Lidar precisions (range, bearing)
         ├── Visual precisions (position, ARD features)
         ├── Goal precisions (position, orientation)
         ├── EFE weights (epistemic, pragmatic, instrumental) ← LEARNED
         ├── Safety precision
         └── Object persistence precision
```

#### Automatic Relevance Determination (ARD)

[`FeaturePrecisions`](./Phantom%20Fellowship%20MIT/AIF2/frobenius_nav_v2/frobenius_nav/core.py) (Lines 472-506):
- Per-dimension precision for shape (Hu moments)
- Hue, saturation, value precisions
- `get_shape_weights()` → relative importance
- `weighted_shape_distance()` → precision-weighted distance

---

### Policy Priors (Habits)

**Location:** [`core.py`](./Phantom%20Fellowship%20MIT/AIF2/frobenius_nav_v2/frobenius_nav/core.py) (Lines 629-750)

[`PolicyPriors`](./Phantom%20Fellowship%20MIT/AIF2/frobenius_nav_v2/frobenius_nav/core.py):
- Dirichlet over discretized (v, ω) control space
- Successful actions accumulate evidence
- Blends learned habits with EFE selection

```python
policy.discretize_action(v, omega)  # Map to bin
policy.continuous_action(bin_idx)   # Map back
policy.observe_successful_action(v, omega)  # Update from success
policy.blend_with_efe(efe_values, temperature)  # Combine with EFE
policy.sample_action(efe_values, temperature)   # Sample from blended
```

---

### EFE-Based Planning

**Location:** [`core.py`](./Phantom%20Fellowship%20MIT/AIF2/frobenius_nav_v2/frobenius_nav/core.py) (Lines 1507-1602)

```
G = π_p·E[KL(q(o)||p(o|g))]   # Pragmatic (goal-seeking)
  - π_e·E[IG]                 # Epistemic (information gain)
  + π_i·E[Risk]               # Instrumental (safety)
  - E[log π(a)]               # Habit prior
```

[`compute_expected_free_energy()`](./Phantom%20Fellowship%20MIT/AIF2/frobenius_nav_v2/frobenius_nav/core.py):
- Top-down predictions via sensor model
- Information gain from expected observations
- Collision risk from object existence probabilities
- Returns `EFEComponents` with decomposition

---

### Scene State with Structural Adaptation

**Location:** [`core.py`](./Phantom%20Fellowship%20MIT/AIF2/frobenius_nav_v2/frobenius_nav/core.py) (Lines 917-1384)

| Class | Description |
|-------|-------------|
| `ObjectBelief` | Position + visual features + **existence probability** (Beta) |
| `PoseIndexedMemory` | View-specific priors indexed by discretized pose |
| `SensorModel` | Forward model: predict observations from state |
| `SceneState` | Full predictive coding loop with Bayesian model reduction |

#### Predictive Coding Loop

1. `predict_expected_observations(pose)` → top-down predictions
2. Receive actual observations (bottom-up)
3. `update(observations, expected, precisions, timestamp)` → prediction errors
4. Update beliefs to minimize free energy
5. Prediction errors update precisions (meta-learning)
6. Objects with low existence probability are pruned

---

### ROS2 Integration

**Location:** [`frobenius_nav_v2/frobenius_nav/navigator_node.py`](./Phantom%20Fellowship%20MIT/AIF2/frobenius_nav_v2/frobenius_nav/navigator_node.py) (~1360 lines)

#### Topics

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribe | `/odom` | `nav_msgs/Odometry` |
| Subscribe | `/scan` | `sensor_msgs/LaserScan` |
| Subscribe | `/camera/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/camera/depth/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/goal_pose` | `geometry_msgs/PoseStamped` |
| Publish | `/cmd_vel` | `geometry_msgs/TwistStamped` |
| Publish | `/navigator/belief_pose` | `PoseWithCovarianceStamped` |
| Publish | `/navigator/diagnostics` | `std_msgs/String` (JSON) |
| Publish | `/navigator/markers` | `visualization_msgs/MarkerArray` |

#### Sensor Processors

| Class | Description |
|-------|-------------|
| `LidarProcessor` | Clusters points, projects to world frame |
| `VisualProcessor` | Canny edges, Hu moments, color extraction, depth projection |

#### Brain Persistence

Learned parameters saved to `~/.frobenius_aif_v2/brain.json`:
- All precision beliefs
- Policy priors (action tendencies)
- Self-efficacy, success/collision rates
- Pose-indexed memory

---

## Comparison & Integration Opportunities

| Aspect | Info_Geo-FORK1 | AIF2 |
|--------|----------------|------|
| **Gaussian Form** | Moment form (μ, Σ) | Information form (η₁, η₂) |
| **Geometry** | Geodesics, parallel transport | KL divergence, fusion |
| **Distributions** | Gaussian, vMF | Gaussian, vMF, Gamma, Dirichlet, Beta |
| **Distance** | Hellinger, Fisher metric | Mahalanobis, KL |
| **Planning** | Geodesic path planning | EFE trajectory optimization |
| **Memory** | k-d tree spatial | Pose-indexed semantic |
| **Learning** | Natural gradient EFA | Precision/EFE weights/habits |
| **Validation** | V2V +10.3% mAP | TurtleBot3 navigation |

### Potential Integration Points

1. **Replace Hellinger with Fisher metric distance** in AIF2's collision checking
2. **Add geodesic interpolation** to AIF2's trajectory optimization
3. **Use AIF2's hierarchical precision learning** for Info_Geo's sensor fusion
4. **Port Info_Geo's V2V framework** to AIF2 for multi-robot coordination
5. **Combine pose-indexed memory** with Info_Geo's causal reasoning
6. **Use AIF2's policy priors** to bias Info_Geo's grounded planner

### Shared Mathematical Foundations

Both projects implement:
- Exponential family distributions
- Natural parameters and sufficient statistics
- Bayesian fusion via parameter addition
- KL divergence for complexity cost
- Entropy for uncertainty quantification

---

## Quick Reference

### Running Info_Geo Validation

```bash
cd Info_Geo-FORK1/Other_test
python run_opencda_validation.py --validate 50
```

### Running AIF2 Navigation

```bash
cd "Phantom Fellowship MIT/AIF2/frobenius_nav_v2"
./tools/run_and_evaluate.sh
```

### Key Imports

```python
# Info_Geo
from cognitive_agent import (
    GaussianSplat, VolumetricMemory, IntegratedCognitivePipeline,
    hellinger_distance, vmf_similarity
)
from ig_geodesics import exponential_map_gaussian, geodesic_interpolation

# AIF2
from frobenius_nav.core import (
    GaussianBelief, GammaBelief, DirichletBelief, VonMisesBelief, BetaBelief,
    PrecisionHyperpriors, PolicyPriors, ActiveInferenceAgent,
    compute_expected_free_energy
)
```

---

*Last updated: January 2026*
