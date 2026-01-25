# Data Flow Audit Report - Vectorized Operators

**Date**: 2026-01-25  
**Status**: ✅ ALL CHECKS PASSED

## Executive Summary

All vectorized operators are correctly wired into the pipeline. The data flow is complete and all operators use the new vectorized implementations. No Python for-loops remain in hot paths, and all operators use Cholesky-based inverses.

## 1. Data Flow Path

### Complete Pipeline Flow:
1. **Livox CustomMsg** → `livox_converter.py` → **PointCloud2** (with `time_offset` field)
2. **PointCloud2** → `backend_node.parse_pointcloud2_vectorized()` → JAX arrays
3. **Pipeline.process_scan_single_hypothesis()**:
   - `translation_wls()` → uses `_translation_wls_core()` (vectorized)
   - `pos_cov_inflation_pushforward()` → uses `_map_update_core()` (vectorized)
4. **Pipeline.combine_hypotheses()**:
   - `hypothesis_barycenter_projection()` → uses `_hypothesis_barycenter_core()` (vectorized)
5. **Results** → trajectory export → evaluation

## 2. Operator Status

### ✅ translation_wls (translation.py)
- **Status**: Fully vectorized
- **Core Function**: `_translation_wls_core()` - JIT-compiled
- **Vectorization**: Uses `jax.vmap` and `jnp.einsum` over bins
- **Inverse Method**: `spd_cholesky_inverse_lifted_core()` (no `jnp.linalg.inv`)
- **Python Loops**: 0 (verified)
- **Pipeline Usage**: Step 8 in `process_scan_single_hypothesis()`

### ✅ hypothesis_barycenter_projection (hypothesis.py)
- **Status**: Fully vectorized
- **Core Function**: `_hypothesis_barycenter_core()` - JIT-compiled
- **Vectorization**: Uses `jax.vmap` and `jnp.einsum` over hypotheses
- **Python Loops**: 0 (verified)
- **Pipeline Usage**: Called via `combine_hypotheses()` in `backend_node.py`

### ✅ pos_cov_inflation_pushforward (map_update.py)
- **Status**: Fully vectorized
- **Core Function**: `_map_update_core()` - JIT-compiled
- **Vectorization**: Uses `jax.vmap` and `jnp.einsum` over bins
- **Python Loops**: 0 (verified)
- **Pipeline Usage**: Step 13 in `process_scan_single_hypothesis()`

## 3. Runtime Manifest

The `PipelineConfig.backends` dictionary now includes:
- `"translation_wls": "fl_slam_poc.backend.operators.translation (vectorized over bins; Cholesky inverse)"`
- `"hypothesis_barycenter": "fl_slam_poc.backend.operators.hypothesis (vectorized over hypotheses)"`
- `"map_update": "fl_slam_poc.backend.operators.map_update (vectorized over bins)"`
- `"lifted_spd_inverse": "fl_slam_poc.common.primitives.spd_cholesky_inverse_lifted_core"`

## 4. SE(3) Log Stability

### ✅ se3_log (se3_jax.py)
- **Status**: Stabilized
- **Method**: Uses `_se3_V_inv()` closed-form inverse (no linear solve)
- **Stability**: Taylor expansion for small angles, exact formula for larger angles
- **No `jnp.linalg.solve`**: Verified removed

## 5. Livox Converter

### ✅ time_offset Field
- **Status**: Always included
- **Implementation**: 
  - If `offset_time` present in source: uses actual value
  - If not (livox_ros_driver2): computes synthetically from point indices (uniform 100ms scan)
- **Backend Compatibility**: Backend parser now receives required field

## 6. Operator Exports

All operators correctly exported in `fl_slam_poc/backend/operators/__init__.py`:
- `translation_wls`, `TranslationWLSResult`
- `hypothesis_barycenter_projection`, `HypothesisProjectionResult`
- `pos_cov_inflation_pushforward`, `MapUpdateResult`

## 7. Test Files

### ✅ test_operators.py
- Tests `hypothesis_barycenter_projection` (updated to expect `exact=False` for kappa)
- All tests use correct operator imports

### ✅ test_audit_invariants.py (NEW)
- Comprehensive audit invariant tests
- Order-invariance, no-gates, units/dt, SO(3)/SE(3) roundtrip, IW commutative updates
- Vectorized operator correctness tests

## 8. Evaluation Script

### ✅ run_and_evaluate_gc.sh
- **Stage 5**: Audit Invariants Check added
- Runs `pytest test/test_audit_invariants.py`
- Reports pass/fail counts
- Results saved to `audit_invariants.log`

## 9. No Multiple Math Paths

**Verified**: Single implementation per operator
- ✅ No duplicate implementations
- ✅ No fallback paths in operators
- ✅ No environment-based selection
- ✅ No try/except ImportError backends
- ✅ All operators exported in `__init__.py` (single import path)

## 10. No Heuristics/Gates

**Verified**: All operations are continuous
- ✅ No `if threshold` gates in vectorized operators
- ✅ No `if reject` or `if skip` branches
- ✅ All operations use continuous weighting
- ✅ Math approximations explicitly documented

## 11. Documentation Status

### ✅ kappa.py
- Comprehensive module docstring explaining low-R approximation
- Error bounds documented (1% for R<0.53, 5% for R<0.85, >10% for R>0.90)
- Certificate correctly marked as `create_approx` with trigger

### ✅ imu_evidence.py
- Comprehensive module docstring explaining vMF Hessian approximation
- Exact Fisher information derivation documented
- Error characteristics and justification provided

## 12. Verification Checklist

- [x] All operators import successfully
- [x] Pipeline uses vectorized operators
- [x] No Python for-loops in hot paths
- [x] No `jnp.linalg.inv` in translation.py
- [x] SE(3) log uses `_se3_V_inv` (no linear solve)
- [x] Livox converter always includes `time_offset`
- [x] All core functions are JIT-compiled
- [x] Runtime manifest includes new operators
- [x] Test files updated
- [x] Evaluation script includes audit checks
- [x] No multiple math paths
- [x] No heuristics/gates introduced
- [x] All approximations documented

## Conclusion

**All wiring is correct.** The vectorized operators are:
1. Properly imported and used in the pipeline
2. JIT-compiled for performance
3. Free of Python loops in hot paths
4. Using Cholesky-based inverses
5. Documented in runtime manifest
6. Tested in audit invariant suite
7. Verified in evaluation script

The system is ready for evaluation runs.
