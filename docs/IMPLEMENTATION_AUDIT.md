# IMU Integration Implementation Audit

**Date**: 2026-01-21  
**Plans Audited**:
- `phase_2_maximal_by_construction_imu.plan.md`
- `hellinger-dirichlet_imu_integration_88f41249.plan.md`

---

## ✅ **Post-Audit Fixes (Recovery/Consistency)**

These issues were found after recovery from accidental deletions and have been fixed:

- **Frontend IMU wiring missing**: `IMUPreintegrator` and `/sim/imu_factor` publishing were absent after refactor. Restored full IMU factor publishing at anchor keyframes, with buffer clearing and OpReports.
- **Backend IMU race drops**: IMU factors arriving before anchor creation were dropped. Added per-anchor buffering and processing on anchor creation.
- **Rotation stability near π**: Added deterministic axis extraction in `rotmat_to_rotvec` and switched delta conversions to `quat_to_rotvec` to reduce antipodal flip artifacts.
- **Gravity configurability**: Added `gravity` parameter to frontend/backend and launch to allow dataset-specific frame conventions.
- **Kernel compliance**: Added `@jax.jit`, removed `S + R_imu` inflation, and wired `bias_ref` into the JAX kernel signature.

---

## ✅ **ADHERENCE: Correctly Implemented**

### 1. State Extension (15DOF)
- ✅ **State initialization**: `mu0 = np.zeros(15)`, `cov0 = 15×15` with correct priors
- ✅ **Process noise**: Extended to 15D with velocity and bias random walks
- ✅ **Anchor storage**: Anchors embedded into 15D with velocity=0, bias=bias_ref
- ✅ **Compatibility**: `on_odom`, `on_loop`, `_publish_state` correctly handle 15D

### 2. Manifold Operations (SE(3) ⊕/⊖)
- ✅ **JAX Lie operators**: `so3_exp`, `so3_log`, `se3_plus`, `se3_minus` implemented
- ✅ **Manifold retraction**: Uses `se3_plus` for pose components, not raw vector addition
- ✅ **Tangent space moment matching**: Correctly computes deltas in tangent space, then retracts

### 3. Dirichlet Routing
- ✅ **Frobenius retention**: `retention_base ** 3` (cubic contraction) applied before pseudo-counts
- ✅ **Softmax as mean-map**: Correctly uses softmax to generate pseudo-counts, not as final belief
- ✅ **Hellinger shift monitoring**: Computes `H²(π_t, π_{t-1}) = 1 - Σ√(π_t · π_{t-1})`
- ✅ **Dirichlet update**: `α ← α' + c` where `α' = t³ · α`

### 4. Global Moment Matching
- ✅ **No per-anchor Schur**: Correctly eliminated per-anchor marginalization
- ✅ **No natural-param weighted sum**: Uses moment matching in expectation space, not `Λ ← Σ w_i Λ_i`
- ✅ **Single e-projection**: Global moment match across all (anchor, sigma) pairs
- ✅ **Correct formula**: `μ = Σ w(i,s) · ξ_j(i,s)`, `Σ = Σ w(i,s) · (ξ_j - μ)(ξ_j - μ)ᵀ`

### 5. Hellinger-Tilted Likelihood
- ✅ **Formula**: `ω_i ∝ exp(-½ r^T R⁻¹ r) · exp(-2 D_H²)`
- ✅ **Hellinger distance**: Closed-form via Bhattacharyya coefficient
- ✅ **Definition**: `p_nom = N(0, R_nom)`, `p̂_i = N(r̄_i, S_i)` where `r̄_i, S_i` from UT

### 6. Batched Architecture
- ✅ **Single kernel call**: One JAX kernel call per IMU packet (not per anchor)
- ✅ **Batched sigma points**: All anchors processed simultaneously via `jax.vmap`
- ✅ **No Python loops**: Core computation is vectorized

---

## ⚠️ **DEVIATIONS: Issues Found**

### **CRITICAL DEVIATION #1: Missing JIT Compilation**

**Plan Requirement** (line 265 in phase_2 plan):
```python
@jax.jit
def imu_batched_projection_kernel(...):
```

**Actual Implementation**:
```python
def imu_batched_projection_kernel(...):  # NO @jax.jit decorator
```

**Impact**: 
- Kernel is not JIT-compiled, missing performance benefits
- No compilation warmup at startup
- Violates "JAX as reference implementation" requirement

**Fix Required**: Add `@jax.jit` decorator and implement warmup.

---

### **CRITICAL DEVIATION #2: Incorrect Residual Covariance S**

**Plan Requirement** (line 294 in hellinger plan):
> "For each anchor i, compute UT predicted residual mean r̄_i and covariance S_i (from its sigma points)"

**Actual Implementation** (line 294-295 in `imu_jax_kernel.py`):
```python
S = jnp.einsum("s,msi,msj->mij", W, r_centered, r_centered)
S = S + R_imu[None, :, :]  # ❌ WRONG: Adding measurement cov to predicted cov
```

**Problem**: 
- `S` should be the **predicted residual covariance** from sigma points only
- Adding `R_imu` (measurement covariance) to `S` (predicted covariance) is mathematically incorrect
- `R_imu` is already used in the likelihood term; adding it to `S` double-counts measurement uncertainty

**Correct Implementation**:
```python
S = jnp.einsum("s,msi,msj->mij", W, r_centered, r_centered)
# S is already the predicted residual covariance from UT - DO NOT add R_imu
S = 0.5 * (S + jnp.swapaxes(S, 1, 2)) + jnp.eye(9, dtype=S.dtype)[None, :, :] * COV_REGULARIZATION
```

**Impact**: 
- Hellinger distance computation uses incorrect `S`, leading to wrong robustness weights
- May cause overconfidence or underconfidence in anchor matching

---

### **CRITICAL DEVIATION #3: Missing bias_ref Parameter**

**Plan Requirement** (line 277 in phase_2 plan):
```python
bias_ref: jnp.ndarray,       # (6,)
```

**Actual Implementation** (line 186 in `imu_jax_kernel.py`):
```python
def imu_batched_projection_kernel(
    ...
    # ❌ MISSING: bias_ref parameter
    gravity: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
```

**Problem**:
- `bias_ref` is extracted in `backend_node.py` (line 877) but not passed to kernel
- Kernel signature doesn't match plan specification
- Bias reference is used for anchor embedding but not available in kernel

**Impact**: 
- Kernel cannot use bias reference if needed for future enhancements
- Signature mismatch with plan

---

### **MODERATE DEVIATION #4: Sigma Point Scheme**

**Plan Requirement** (line 373 in phase_2 plan):
> "Sigma point generation (2n+1 = 61 points for n=30)"

**Plan Requirement** (line 138 in hellinger plan):
> "Generate sigma points for ALL anchors: `(M, 2*30+1, 30)`"

**Actual Implementation** (lines 240-244 in `imu_jax_kernel.py`):
```python
# Uses spherical-radial cubature weights:
#   - center point has 0 weight
#   - 2n symmetric points each have weight 1/(2n)
W = jnp.concatenate([
    jnp.array([0.0], dtype=anchor_mus.dtype),
    jnp.full((2 * joint_dim,), 1.0 / (2.0 * joint_dim), dtype=anchor_mus.dtype),
])  # (61,)
```

**Analysis**:
- **Count**: Correctly generates 61 points (1 center + 60 symmetric)
- **Weights**: Uses cubature weights (1/(2n) for symmetric points) instead of unscented transform weights
- **Rationale**: Comment says "non-negative sigma-support weights" for mixture support

**Impact**: 
- Different weight distribution than plan specifies
- May affect moment matching accuracy
- Cubature is valid but deviates from plan's unscented transform specification

**Plan Spec** (unscented transform):
- Center weight: `W₀ = λ/(n+λ)` where `λ = α²(n+κ) - n`
- Symmetric weights: `W_i = 1/(2(n+λ))` for i=1,...,2n

**Current Implementation** (cubature):
- Center weight: `W₀ = 0`
- Symmetric weights: `W_i = 1/(2n)` for i=1,...,2n

---

### **MODERATE DEVIATION #5: Missing Kernel Warmup**

**Plan Requirement** (line 328 in phase_2 plan):
```python
# In FLBackend.__init__
self._warmup_imu_kernel()
```

**Actual Implementation**: 
- ❌ No warmup method implemented
- ❌ No kernel compilation at startup

**Impact**: 
- First IMU factor will pay compilation cost (latency spike)
- Violates plan's performance optimization

---

### **MODERATE DEVIATION #6: Missing Fisher-Rao Tangent Decomposition**

**Plan Requirement** (line 208 in phase_2 plan):
> "Define fixed decomposition of 15D chart into blocks: pose / velocity / biases"
> "Track energy in each block"
> "Log as geometry-defined 'approximation stress' scalar"

**Actual Implementation**: 
- ❌ No Fisher-Rao decomposition implemented
- ❌ No energy tracking per block
- ❌ No approximation stress logging

**Impact**: 
- Missing diagnostic for numerical stability monitoring
- Cannot detect "dominant anchor collapse" via geometry-defined metrics

---

### **MINOR DEVIATION #7: Missing Bias Sensitivity in Residual**

**Plan Note** (line 68-72 in `imu_jax_kernel.py`):
> "Biases are part of the 15D state, but this residual currently does not include bias sensitivity because `IMUFactor.msg` does not carry preintegration Jacobians"

**Plan Requirement**: 
- Plan doesn't explicitly require bias sensitivity, but notes it as an approximation
- Current implementation correctly logs this as `approximation_triggers=["BiasFrozen"]`

**Status**: ✅ **ACCEPTABLE** - Explicitly logged as approximation, matches plan's auditability requirement

---

### **MINOR DEVIATION #8: Enhanced Anchor Matching (Not in Plan)**

**Actual Implementation** (lines 931-1001 in `backend_node.py`):
- Added keyframe_to_anchor mapping lookup
- Added Hellinger distance fallback for nearest-neighbor matching

**Plan Requirement**: 
- Plan doesn't specify anchor matching strategy
- Plan assumes anchors are already correctly associated

**Status**: ✅ **ENHANCEMENT** - Adds robustness beyond plan requirements

---

### **MINOR DEVIATION #9: Missing IMUCompositionalEvidence Structure**

**Plan Requirement** (line 223 in phase_2 plan):
```python
IMUCompositionalEvidence = {
    "keyframe_j": int,
    "natural_params": {"L": 15x15, "h": 15D},
    "routing_posterior": {"w": [w_1, ..., w_M]},
    "retention_scalar": t,
    "hellinger_shift": H²,
    "projection_metadata": {...}
}
```

**Actual Implementation**: 
- Uses `OpReport` instead of structured `IMUCompositionalEvidence`
- All required information is present in `OpReport.metrics` and diagnostics

**Status**: ⚠️ **ACCEPTABLE** - Information is present but in different format

---

## 📊 **Summary of Deviations**

| Severity | Issue | Location | Impact |
|----------|-------|----------|--------|
| **CRITICAL** | Missing `@jax.jit` | `imu_jax_kernel.py:176` | Performance, violates plan |
| **CRITICAL** | Incorrect `S = S + R_imu` | `imu_jax_kernel.py:295` | Wrong Hellinger distance, robustness weights |
| **CRITICAL** | Missing `bias_ref` parameter | `imu_jax_kernel.py:176` | Signature mismatch, missing functionality |
| **MODERATE** | Cubature vs Unscented Transform | `imu_jax_kernel.py:240-244` | Different weight distribution |
| **MODERATE** | Missing kernel warmup | `backend_node.py` | First-call latency |
| **MODERATE** | Missing Fisher-Rao decomposition | N/A | No stability diagnostics |
| **MINOR** | Missing bias sensitivity | `imu_jax_kernel.py:68-72` | Logged as approximation ✅ |
| **MINOR** | Enhanced anchor matching | `backend_node.py:931-1001` | Enhancement beyond plan ✅ |
| **MINOR** | OpReport vs IMUCompositionalEvidence | `backend_node.py:990-1020` | Different format, same info ✅ |

---

## 🔧 **Required Fixes (Priority Order)**

### **Fix #1: Correct Residual Covariance S** (CRITICAL)
**File**: `imu_jax_kernel.py:295`
```python
# REMOVE this line:
S = S + R_imu[None, :, :]  # ❌ DELETE

# S is already correct from UT - just symmetrize and regularize:
S = 0.5 * (S + jnp.swapaxes(S, 1, 2)) + jnp.eye(9, dtype=S.dtype)[None, :, :] * COV_REGULARIZATION
```

### **Fix #2: Add JIT Compilation** (CRITICAL)
**File**: `imu_jax_kernel.py:176`
```python
@jax.jit
def imu_batched_projection_kernel(...):
```

**File**: `backend_node.py` (add warmup method)
```python
def _warmup_imu_kernel(self):
    """Warm up JAX kernel compilation at startup."""
    if not self.enable_imu_fusion:
        return
    # ... create dummy inputs and call kernel once
```

### **Fix #3: Add bias_ref Parameter** (CRITICAL)
**File**: `imu_jax_kernel.py:176`
```python
def imu_batched_projection_kernel(
    ...
    gravity: jnp.ndarray,
    bias_ref: jnp.ndarray,  # ADD THIS
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
```

**File**: `backend_node.py:1019`
```python
mu_new_jax, cov_new_jax, diagnostics = imu_batched_projection_kernel(
    ...
    gravity=jnp.array(self.gravity),
    bias_ref=jnp.array(bias_ref),  # ADD THIS
)
```

### **Fix #4: Switch to Unscented Transform Weights** (MODERATE)
**File**: `imu_jax_kernel.py:240-244`
Replace cubature weights with unscented transform weights as specified in plan.

### **Fix #5: Add Fisher-Rao Decomposition** (MODERATE)
Implement block-wise energy tracking and approximation stress logging.

---

## ✅ **What Was Done Correctly**

1. ✅ **15D state extension**: Complete and correct
2. ✅ **Manifold operations**: SE(3) ⊕/⊖ correctly implemented
3. ✅ **Dirichlet routing**: Frobenius retention, Hellinger shift all correct
4. ✅ **Global moment matching**: Correct formula, no per-anchor Schur
5. ✅ **Hellinger distance**: Formula matches plan (Bhattacharyya coefficient)
6. ✅ **Batched architecture**: Single kernel call, no Python loops
7. ✅ **No natural-param weighted sum**: Correctly uses moment matching

---

## 🎯 **Overall Assessment**

**Adherence Score**: ~85%

**Strengths**:
- Core mathematical structure is correct
- Global moment matching correctly implemented
- Dirichlet routing with Frobenius retention is correct
- Manifold operations are correct

**Critical Issues**:
- **S = S + R_imu is mathematically wrong** - must fix
- **Missing JIT compilation** - violates performance requirement
- **Missing bias_ref** - signature mismatch

**Recommendation**: Fix the 3 critical issues before considering implementation complete. The S covariance error is the most serious as it affects core robustness logic.
