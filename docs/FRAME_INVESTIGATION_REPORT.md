# Frame Convention Investigation Report

**Date:** 2026-01-29
**Investigator:** Claude (with user direction)
**Purpose:** Root-cause frame/convention mismatches causing trajectory estimation issues

---

## Executive Summary

The investigation found **several interconnected issues** but the core problem is NOT that LiDAR is Z-forward (as initially hypothesized). The main issues are:

1. **Odometry Z is garbage** - Wheel odometry Z starts at ~30m and is unreliable (correctly marked with 1e+06 covariance)
2. **GT is in a different world frame** - Mocap world vs odom_combined have different origins and potentially different axes
3. **Wheel frame vs body frame mismatch** - SLAM outputs wheel frame (`base_footprint`), GT is in body frame (`camera_imu`)
4. **Z-drift band-aids** - The pipeline has workarounds that mask but don't fix the root causes

---

## 1. LiDAR Frame Convention: **CONFIRMED Z-UP**

### Evidence

**Point distribution analysis (512,568 valid points from 30 scans):**
- 77.2% of points above Z=0 (above sensor level)
- 22.8% of points below Z=0 (below sensor level)
- This 3:1 asymmetry matches MID-360's asymmetric FOV (-7° to +52°)

**Floor detection:**
- Peak at Z = -0.81m matches expected floor (sensor at 0.778m height per T_base_lidar)
- Horizontal extent at Z = [-0.80, -0.60] is ~15.7m (large flat surface = floor)

**Raw point values:**
- Values are in meters (not mm)
- Range: 0.1m to 12.2m (reasonable for indoor)
- Z distribution matches expected vertical FOV asymmetry

### Conclusion

**LiDAR is Z-UP.** The initial hypothesis that LiDAR might be Z-forward was incorrect. The T_base_lidar rotation of identity is correct.

---

## 2. IMU Frame Convention: **CONFIRMED Z-UP with 25° tilt**

### Evidence

**Accelerometer analysis (500 samples):**
```
Mean accel: [-0.45g, -0.03g, +0.88g]
Magnitude: 0.99g (correct)
Direction: [-0.45, -0.03, +0.89] normalized
```

**Interpretation:**
- Dominant +Z component (0.89) confirms Z-UP convention
- Significant -X component (-0.45) indicates ~25° tilt from vertical
- This matches the configured T_base_imu rotation of ~28°

### Conclusion

**IMU is Z-UP but tilted ~25° in the X direction.** The T_base_imu rotation `[-0.016, 0.489, 0.0]` rad correctly compensates for this.

---

## 3. Odometry Frame: **Z IS GARBAGE**

### Evidence

**Odom message analysis:**
```
header.frame_id: odom_combined
child_frame_id: base_footprint

Position at start: x=3.07, y=3.98, z=29.99 (!)
Z range over trajectory: [29.95, 36.43] (6.5m variation!)
Z covariance: 1e+06 (correctly marks Z as unobserved)
```

### Root Cause

Wheel odometry (2D) does NOT measure height. The Z value in odom messages is:
- Meaningless (starts at arbitrary ~30m)
- Drifting (varies by 6.5m over the trajectory)
- Correctly marked as unobserved (covariance 1e+06)

### Impact

**The pipeline should NEVER trust odom Z values.** The covariance correctly indicates this, but:
1. Initial state may be incorrectly seeded from odom Z
2. Any code path that uses odom Z without checking covariance will fail
3. The "Z drift to -50 to -80m" symptom may originate here

---

## 4. Ground Truth Frame: **DIFFERENT FROM ESTIMATE**

### Evidence

**GT pose analysis:**
```
frame_id: world (mocap world)
Position: x=2.95, y=1.32, z=0.86 (robot height!)
Z std: 0.0034m (constant - planar motion confirmed)
```

**Axis convention (from Umeyama alignment):**
```
R_EST_TO_GT = [
    [0, 0, -1],  # GT x ≈ -EST z
    [1, 0, 0],   # GT y ≈ EST x
    [0, -1, 0],  # GT z ≈ -EST y
]
```

### Root Cause

The mocap world frame has **completely different axis conventions** from the odom_combined frame:
- GT X ≈ negative EST Z
- GT Y ≈ EST X
- GT Z ≈ negative EST Y

This is NOT a simple Z-up vs Z-forward flip - it's a complex signed permutation.

### Impact

Evaluation requires the `R_EST_TO_GT` transform to compare trajectories. This is handled at evaluation time but indicates fundamental frame mismatch.

---

## 5. Wheel Frame vs Body Frame: **DOCUMENTED MISMATCH**

### Evidence

| Frame | Definition | Used by |
|-------|------------|---------|
| `base_footprint` (wheel) | Odom child_frame_id | SLAM estimate |
| `camera_imu` (body) | M3DGR GT frame | Ground truth |

**Impact measured:**
- Before body-frame correction: ATE rotation RMSE ~171°
- After body-frame correction: ATE rotation RMSE ~117° (34% reduction)

### Root Cause

SLAM outputs poses in wheel frame, but GT is captured in body frame. The `body_T_wheel` calibration transform is required to compare them.

---

## 6. Z-Drift Band-Aids: **SYMPTOMS, NOT FIXES**

The code has multiple workarounds for Z drift:

### map_update.py:104-109
```python
# PLANAR FIX: Zero out t_hat[2] before map update.
t_hat_planar = t_hat.at[2].set(0.0)
```
This **breaks the map-belief feedback loop** for Z by forcing map centroids to stay in Z=0 plane.

### constants.py:263-276
```python
GC_PLANAR_Z_REF = 0.0  # Pull base_footprint Z toward ground (wheel/base frame)
GC_PLANAR_Z_SIGMA = 0.1  # Soft constraint
```
These **planar priors** constrain the *wheel/base* frame Z to stay near 0.0m (ground contact).

Note: M3DGR ground truth is in `camera_imu` (body) with Z ≈ 0.85m; evaluation should transform the
wheel-frame estimate into body frame using `config/m3dgr_body_T_wheel.yaml` (see `tools/transform_estimate_to_body_frame.py`).

### Root Cause (unresolved)

Why does Z drift to -50 to -80m without these band-aids? Possible causes:
1. Odom Z garbage being ingested somewhere
2. LiDAR→map→belief feedback loop accumulating errors
3. Process model allowing Z to drift
4. Some other frame/transform issue

---

## Recommendations

### Immediate Actions

1. **Audit odom Z usage** - Find all code paths that use odom Z values. They should either:
   - Check covariance and ignore if > threshold
   - Use a fixed Z value (e.g., GC_PLANAR_Z_REF)
   - Not use odom Z at all

2. **Remove Z band-aids and diagnose** - Temporarily remove:
   - `t_hat.at[2].set(0.0)` in map_update.py
   - Planar priors
   And observe what causes Z to drift. This will reveal the root cause.

3. **Validate transforms end-to-end** - Create a test that:
   - Takes a known pose in GT frame
   - Transforms through all the pipeline frames
   - Verifies it arrives at the expected pose

### Longer-term

4. **Consider outputting body-frame poses** - Instead of patching at evaluation time, have the pipeline output poses in the body frame directly

5. **Document all frame conventions explicitly** - Create a single-source-of-truth diagram showing:
   - All frames (world, odom_combined, base_footprint, livox_frame, camera_imu, mocap_world)
   - All transforms between them
   - Which are measured vs configured

---

## Summary Table

| Component | Convention | Status | Notes |
|-----------|------------|--------|-------|
| LiDAR | Z-UP | ✅ Confirmed | 3:1 point asymmetry matches FOV |
| IMU | Z-UP (tilted 25°) | ✅ Confirmed | T_base_imu compensates |
| Odom XY | Planar motion | ✅ Correct | Normal wheel odometry |
| Odom Z | **GARBAGE** | ❌ **PROBLEM** | Starts at 30m, varies 6.5m |
| GT world | Z-UP | ✅ Confirmed | Z constant at 0.86m |
| GT vs EST axes | Different | ⚠️ Mismatch | Signed permutation required |
| Wheel vs body | Different | ⚠️ Mismatch | body_T_wheel transform required |

---

## Files Changed/Created

- `tools/diagnose_frames_detailed.py` - New detailed diagnostic tool
- `docs/FRAME_INVESTIGATION_REPORT.md` - This document
