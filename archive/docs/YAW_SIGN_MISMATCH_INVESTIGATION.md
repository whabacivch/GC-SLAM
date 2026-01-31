# Yaw Sign Mismatch Deep Dive Investigation

**Date**: 2026-01-27  
**Issue**: Gyro and **Matrix Fisher (LiDAR)** yaw increments have opposite signs 95.7% of the time (legacy test)  
**Evidence**: Invariant test shows correlation -0.271 between gyro and LiDAR rotation increments (legacy Wahba test)

## Executive Summary

The invariant test reveals a **clear sign inversion** in gyro processing:
- **Gyro ↔ Matrix Fisher**: Only 4.3% same sign (67/70 scans opposite) *(legacy Wahba run)*
- **Correlation**: -0.271 (strong negative correlation = opposite directions)
- **Mean magnitudes**: Gyro 37.80°, LiDAR 41.99° (similar magnitudes, opposite signs)

This indicates the gyro is rotating in the **opposite direction** from what LiDAR observes.  
**Update:** The pipeline now uses **Matrix Fisher rotation** instead of Wahba; rerun the invariant test and treat “Wahba” below as “LiDAR rotation” unless otherwise stated.

---

## Investigation: Three Potential Causes

### A) IMU→Base Rotation (`T_base_imu`) - Wrong Rotation Matrix

#### Current Configuration
```yaml
T_base_imu: [0.0, 0.0, 0.0, -0.015586, 0.489293, 0.0]
```
- Rotation vector: `[-0.015586, 0.489293, 0.0]` radians
- Rotation angle: **28.05°**
- Rotation axis: `[-0.0318, 0.9995, 0.0]` (mostly Y-axis, slight X)

#### How It's Applied
**Location**: `backend_node.py:489`
```python
gyro = self.R_base_imu @ gyro  # Transform from IMU frame to base frame
```

**Math**: `gyro_base = R_base_imu @ gyro_imu`

#### Analysis of Current Rotation Matrix

The rotation matrix `R_base_imu` is:
```
[[ 0.8827  -0.0037   0.4700]
 [-0.0037   0.9999   0.0150]
 [-0.4700  -0.0150   0.8825]]
```

**Effect on Z-axis gyro (yaw rotation)**:
- IMU z-axis `[0, 0, 1]` → Base frame `[0.470, 0.015, 0.883]`
- **Z-component in base: 0.883** (88.3% preserved, but mixed with X/Y)

**Key Finding**: The rotation **mixes** Z-axis gyro into X and Y components:
- Z-axis gyro in IMU → 47% X + 1.5% Y + 88.3% Z in base
- This is a **28° rotation about Y-axis**, which tilts the Z-axis

#### Potential Issues

1. **Wrong rotation angle**: The 28° rotation might be correct for gravity alignment (pitch), but could be **inverting the yaw axis** if the IMU is mounted differently than expected.

2. **Wrong rotation axis**: The rotation is about `[-0.0318, 0.9995, 0.0]` (mostly Y-axis). If the IMU's yaw axis (Z) is actually aligned differently, this rotation could map Z→Z incorrectly.

3. **Frame convention mismatch**: The rotation assumes `T_base_imu` means "transform from IMU to base", but if the convention is reversed, we'd need `R_imu_base = R_base_imu.T`.

#### Verification Needed

Check if the rotation **preserves yaw direction**:
- If IMU Z-axis is the yaw axis, and base Z-axis is the yaw axis, then `R_base_imu[2,2]` should be close to 1.0
- Current: `R_base_imu[2,2] = 0.8825` (88% preserved, but 12% leakage to X/Y)
- **This leakage could cause sign issues if the leakage components have wrong signs**

---

### B) Gyro Axis Sign Flip - Gyro Reading Convention

#### Current Code
**Location**: `backend_node.py:476-479`
```python
gyro = np.array(
    [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
    dtype=np.float64,
)
```

#### Convention Documentation
**Location**: `docs/FRAME_AND_QUATERNION_CONVENTIONS.md:95-98`
- **Right-hand rule**: Positive rotation about +Z axis is **counter-clockwise** when viewed from above
- **After extrinsic**: `gyro_base = R_base_imu @ gyro_imu`

#### Potential Issues

1. **ROS IMU message convention**: The `sensor_msgs/Imu` message might use a different sign convention than expected.
   - Check: Does positive `angular_velocity.z` mean counter-clockwise or clockwise?
   - If ROS convention is **clockwise** but we assume **counter-clockwise**, we need to negate Z.

2. **Livox IMU convention**: The Livox Mid-360 IMU might output angular velocity with a different sign convention.
   - Check datasheet or test: Rotate robot clockwise, check if `angular_velocity.z` is positive or negative.

3. **Frame ID interpretation**: The IMU message has a `frame_id`. If the frame is defined with a different handedness, the sign could be wrong.

#### Verification Needed

**Test**: Rotate robot **clockwise** (when viewed from above):
- If `angular_velocity.z > 0` → Convention matches (positive = counter-clockwise)
- If `angular_velocity.z < 0` → Convention is flipped (positive = clockwise) → **Need to negate Z**

**Current behavior**: The invariant test shows gyro going **opposite** to LiDAR, suggesting either:
- Gyro Z is negated (should be `-gyro[2]`)
- Or the rotation matrix is inverting the sign

---

### C) Left/Right Hand Convention Mismatch

#### Coordinate Frame Conventions

**Base frame** (`base_footprint`):
- X: Forward
- Y: Left
- Z: Up
- **Right-handed**: X × Y = Z

**IMU frame** (`livox_frame`):
- Unknown convention (need to verify)
- Could be left-handed or have different axis ordering

#### Potential Issues

1. **Handedness mismatch**: If IMU frame is **left-handed** but base frame is **right-handed**, the rotation matrix would need special handling.

2. **Axis ordering mismatch**: If IMU uses `[X, Y, Z]` but base uses `[X, -Y, Z]` or similar, the rotation would be wrong.

3. **Yaw axis definition**: If IMU's "yaw" axis is actually the X or Y axis (not Z), the rotation would map to the wrong base axis.

#### Verification Needed

**Check frame definitions**:
- What is the `frame_id` of the IMU messages? (`livox_frame`?)
- What is the axis convention for `livox_frame`?
- Does `livox_frame` use right-handed or left-handed coordinates?

**Test**: If we negate the **entire gyro vector** (all three components), does the sign mismatch go away?
- If yes → Handedness issue
- If no → Specific axis issue

---

## Mathematical Analysis

### Current Transformation Chain

1. **Raw gyro** (IMU frame): `gyro_imu = [wx, wy, wz]`
2. **Transform to base**: `gyro_base = R_base_imu @ gyro_imu`
3. **Preintegration**: `delta_R = Exp(gyro_base * dt)` (integrated over scan)
4. **Evidence**: Compare `R_start @ delta_R` vs `R_wahba` (LiDAR)

### Sign Mismatch Hypothesis

If gyro and LiDAR rotation (Matrix Fisher) have **opposite signs**, one of these is wrong:

**Option 1**: Gyro Z-axis sign is flipped
```python
gyro[2] = -gyro[2]  # Negate yaw component
```

**Option 2**: Rotation matrix inverts yaw
```python
# If R_base_imu[2,2] < 0, or if the rotation axis is wrong
```

**Option 3**: Frame convention mismatch
```python
# If IMU frame is left-handed, need special transform
```

### Test: What Happens If We Negate Gyro Z?

If we change:
```python
gyro = self.R_base_imu @ gyro
gyro[2] = -gyro[2]  # Negate yaw after rotation
```

**Expected result**: Gyro and LiDAR rotation (Matrix Fisher) should have **same sign** (correlation becomes positive).

---

## Code Flow Analysis

### Gyro Processing Path

1. **`backend_node.py:on_imu()`** (line 469-492):
   - Read `msg.angular_velocity.{x,y,z}`
   - Transform: `gyro = R_base_imu @ gyro`
   - Store in buffer

2. **`pipeline.py:process_scan_single_hypothesis()`** (line 254):
   - Receive `imu_gyro` (already transformed to base frame)

3. **`imu_preintegration.py:preintegrate_imu_relative_pose_jax()`** (line 94-95):
   - `omega = gyro_i - gyro_bias`
   - `dR = so3_exp(omega * dt_eff)`
   - **This integrates angular velocity → rotation**

4. **`imu_gyro_evidence.py:imu_gyro_rotation_evidence()`** (line 66-68):
   - `R_delta = so3_exp(delta_rotvec_meas)`
   - `R_end_imu = R_start @ R_delta`
   - Compare with `R_end_pred` from state

### Key Question: Where Does Sign Get Flipped?

**Possibility 1**: `R_base_imu` has wrong sign in Z-component
- Check: `R_base_imu[2,2]` should be positive and close to 1.0
- Current: `0.8825` (positive, but less than 1.0 due to 28° tilt)

**Possibility 2**: Gyro Z reading has wrong sign
- Check: ROS/Livox convention for `angular_velocity.z`
- Test: Rotate clockwise, check sign

**Possibility 3**: `so3_exp` convention
- Check: Does `so3_exp([0,0,θ])` rotate counter-clockwise or clockwise?
- Standard: `so3_exp([0,0,θ])` should give counter-clockwise rotation for positive θ

---

## Recommended Tests

### Test 1: Verify Gyro Z Sign Convention
```python
# In backend_node.py, add logging:
if self.imu_count < 10:
    self.get_logger().info(f"Gyro raw: {gyro_raw}, after R_base_imu: {gyro}")
    # Manually rotate robot and check signs
```

### Test 2: Negate Gyro Z After Rotation
```python
# Temporary test in backend_node.py:489
gyro = self.R_base_imu @ gyro
gyro[2] = -gyro[2]  # TEST: Negate yaw
```
**Expected**: If this fixes the sign mismatch, the issue is in gyro Z convention.

### Test 3: Check Rotation Matrix Z-Component
```python
# Verify R_base_imu[2,2] is correct
# If IMU Z should map to base Z, this should be close to 1.0
# Current: 0.8825 (reasonable for 28° tilt, but verify it's not negative)
```

### Test 4: Verify Frame Conventions
- Check `frame_id` of IMU messages
- Verify `livox_frame` axis convention (right/left-handed)
- Check if `base_footprint` and `livox_frame` use same handedness

---

## Most Likely Root Cause

Based on the analysis:

1. **The 28° rotation is for gravity alignment** (pitch), which is correct.
2. **The rotation matrix preserves 88% of Z-axis** (yaw), which is reasonable.
3. **BUT**: The sign mismatch is **consistent and strong** (-0.271 correlation).

**Hypothesis**: The gyro Z-axis **sign convention is flipped**. Either:
- ROS `angular_velocity.z` uses clockwise-positive (we assume counter-clockwise)
- Or Livox IMU outputs clockwise-positive

**Fix**: Negate gyro Z after rotation:
```python
gyro = self.R_base_imu @ gyro
gyro[2] = -gyro[2]  # Fix sign convention
```

**Alternative**: If the rotation matrix itself is wrong, we might need to recompute `T_base_imu` with a different convention or verify the frame definitions.

---

## Next Steps

1. **Run Test 2** (negate gyro Z) to confirm if this fixes the sign mismatch
2. **Verify gyro convention** by manually rotating robot and checking `angular_velocity.z` sign
3. **Check frame definitions** for `livox_frame` and `base_footprint`
4. **Re-run invariant test** after fix to verify correlation becomes positive
