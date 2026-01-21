# Order Invariance in FL-SLAM: Design & Verification

## Principle

The system is designed to be **order-invariant** (associative and commutative) at all levels where information fusion occurs. This is a **fundamental property** of information-form fusion that replaces traditional EKF/optimization-based SLAM.

### Mathematical Foundation

For exponential families in natural parameter space:
```
θ_fused = θ_A + θ_B  (exact, closed-form)
```

Properties:
- **Commutative**: `θ_A + θ_B = θ_B + θ_A`
- **Associative**: `(θ_A + θ_B) + θ_C = θ_A + (θ_B + θ_C)`
- **No Jacobians**: Addition in flat natural coordinates (no relinearization)

---

## Implementation Guarantees

### 1. Gaussian Position Fusion ✅

**Location**: `operators/gaussian_info.py`

```python
def fuse_info(L1, h1, L2, h2, weight=1.0, rho=1.0):
    """Order-invariant additive fusion."""
    return (rho * L1 + weight * L2, rho * h1 + weight * h2)
```

**Verification**:
- Test: `test_rgbd_multimodal.py::TestGaussianInfoAssociativity`
- `(A + B) + C == A + (B + C)` ✅
- `A + B == B + A` ✅

**Shape Convention**: All `h` vectors are **flat (n,)** to prevent silent NumPy broadcasting errors.

---

### 2. vMF Normal Fusion ✅

**Location**: `operators/vmf_geometry.py`

```python
def vmf_barycenter(thetas, weights, d=3):
    """Closed-form Bregman barycenter (exact via Bessel)."""
    # Dual-space averaging
    eta_star = sum(w_i * A_d(κ_i) * μ_i) / W
    # Primal recovery via special function inversion
    kappa_star = A_d_inverse_exact(||eta_star||)
```

**Key Fix (2024-01)**: Replaced hand-rolled series with **bracketed root solve** of the defining Bessel equation `A_d(κ) = r` using `scipy.optimize.brentq`. This gives:
- **Idempotence**: `barycenter([θ], [1.0]) == θ` (exact to machine precision)
- **Associativity**: `(A+B)+C == A+(B+C)` (no truncation-induced drift)

**Verification**:
- Test: `test_rgbd_multimodal.py::TestVMFBarycenter::test_barycenter_associativity` ✅
- Test: `test_rgbd_multimodal.py::TestBesselInversion::test_inverse_series_roundtrip` ✅

---

### 3. RGB-D Evidence→Anchor Association ✅

**Location**: `nodes/fl_backend_node.py::process_rgbd_evidence()`

**OLD (Order-Dependent)** ❌:
```python
for evidence in evidence_list:
    for anchor_id in self.anchors:  # Insertion order matters!
        if distance < threshold:
            fuse_at_anchor(anchor_id, evidence)
            break  # First anchor wins
```

**Problem**: If two anchors are equidistant, **insertion order** determines assignment. Evidence order also matters (greedy association).

**NEW (Order-Invariant)** ✅:
```python
# Phase 1: Deterministic assignment (before any updates)
evidence_for_anchor = {}
for evidence in evidence_list:
    candidates = [(dist, anchor_id) for anchor_id, ... if dist < radius]
    candidates.sort(key=lambda x: (x[0], x[1]))  # Distance, then anchor_id
    if candidates:
        nearest = candidates[0][1]
        evidence_for_anchor[nearest].append(evidence)

# Phase 2: Apply all updates (order doesn't matter due to info addition)
for anchor_id, evidence_batch in evidence_for_anchor.items():
    for ev in evidence_batch:
        fuse_at_anchor(anchor_id, ev)  # θ += θ_ev (commutative)
```

**Guarantee**:
```
Evidence_A + Evidence_B + Anchor_1 + Anchor_2 
== Evidence_B + Evidence_A + Anchor_2 + Anchor_1
```

---

### 4. Loop Closure Updates ✅

**Location**: `nodes/fl_backend_node.py::on_loop()`

Loop closures use **joint Gaussian update** (information addition in augmented state):
```python
L_joint = block_diag(L_current, L_anchor)
L_joint += H.T @ Λ_obs @ H  # Information from relative observation
```

**Order Invariance**: Loop closures are processed **one at a time** from a buffer, but:
- Each update is additive in information form
- No relinearization between loops (flat coordinates)
- Result independent of processing order

**Edge Case**: If two loop factors arrive before their anchors, they're buffered and processed **in anchor creation order** (deterministic).

---

## True RGB-D Colors/Normals ✅

**Fixed**: Frontend now extracts **true RGB colors and surface normals** from synchronized RGB-D pairs.

**Pipeline**:
1. `SensorIO` buffers raw RGB + depth arrays
2. `get_synchronized_rgbd()` finds temporally aligned pairs
3. `rgbd_processor.depth_to_pointcloud()` extracts:
   - 3D positions via pinhole back-projection
   - Surface normals via depth gradients
   - True RGB colors from image
   - Depth-dependent covariances
4. `rgbd_to_evidence()` converts to information form (L, h, θ)
5. `transform_evidence_to_global()` transforms to odom frame
6. Frontend publishes JSON payload to `/sim/rgbd_evidence`
7. Backend fuses via `process_rgbd_evidence()` (order-invariant)

**Verification**: Visual inspection in Foxglove `/cdwm/map` (colored points with normals).

---

## Testing Strategy

### Unit Tests

**Test**: `test/test_rgbd_multimodal.py`

Coverage:
- Gaussian information fusion associativity/commutativity
- vMF barycenter associativity/commutativity
- Laser 2D + RGB-D 3D fusion exactness
- Fisher-Rao distance symmetry/triangle inequality

All tests pass: `25 passed` ✅

### Integration Tests

**Rosbag Playback**:
```bash
## Phase 2 note
Alternative launch files are stored under `phase2/` and are not installed by the MVP package by default.
See: `phase2/fl_ws/src/fl_slam_poc/launch/poc_tb3_rosbag.launch.py`

ros2 launch fl_slam_poc poc_tb3_rosbag.launch.py \
  bag:=rosbags/tb3_slam3d_small_ros2 \
  play_bag:=true \
  publish_rgbd_evidence:=true
```

**Check**:
- Backend status: `"dense_modules": N > 0`
- Foxglove `/cdwm/map`: Colored points visible
- Backend logs: "Fused RGB-D at anchor X" messages

---

## Failure Modes Prevented

1. **Silent NumPy Broadcasting** ❌→✅
   - **Old**: `h` vectors mixed shapes (n,) and (n,1), silent broadcast errors
   - **New**: All `h` normalized to (n,) in `_as_vector()`

2. **vMF Truncation Drift** ❌→✅
   - **Old**: Hand-rolled series approximation, idempotence violated
   - **New**: Root solve of exact Bessel equation

3. **Evidence Assignment Ambiguity** ❌→✅
   - **Old**: First anchor in dict iteration order wins (insertion-dependent)
   - **New**: Deterministic tiebreak on `(distance, anchor_id)`

4. **RGB Colors/Normals Not Extracted** ❌→✅
   - **Old**: Only depth→points, no color/normal extraction
   - **New**: Full `rgbd_processor` pipeline with synchronized RGB-D

---

## Design Principles

### P1: Exact Fusion (Closed-Form)
All fusion operations are **algebraically exact** (no optimization, no iterative solvers):
- Gaussian: `L + L_obs`, `h + h_obs` (matrix/vector addition)
- vMF: Dual averaging + special function inversion (Bessel)

### P2: Associativity/Commutativity
Updates **commute and associate** because:
- Natural parameter addition is commutative
- No relinearization (flat coordinates)
- Deterministic data association (no greedy order-dependence)

### P3: No Silent Failures
- All evidence explicitly published to topics
- Backend logs evidence reception/fusion
- Shape/type errors caught (no silent broadcasting)

---

## References

- **Gaussian Info Form**: Barndorff-Nielsen (1978), Amari (2016)
- **vMF Geometry**: Mardia & Jupp (2000), Miyamoto et al. (2024)
- **Associativity (WDVV)**: Combe (2022-2025) pre-Frobenius manifolds
- **Closed-Form Distances**: "On Closed-Form Expressions for Fisher-Rao Distance" (2024)
