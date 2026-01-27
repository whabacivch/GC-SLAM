# Yaw Drift Root Cause Analysis

## Problem Statement

Rotation ATE errors of 100-130° indicate severe yaw drift. This document analyzes the root causes based on gyro and odom limitations.

## Key Insights

### 1. Gyro Limitations

**Gyro measures angular velocity (relative yaw change), not absolute yaw.**

- ✅ **What gyro does**: Integrates angular velocity → relative rotation change
- ❌ **What gyro cannot do**: Fix absolute yaw (no magnetometer/global heading)
- ⚠️ **Critical issue**: If `T_base_imu` rotation is wrong, we inject **rotated angular velocity**

**Impact of wrong `T_base_imu`**:
```
gyro_base = R_base_imu @ gyro_imu  (line 470 in backend_node.py)
```

If `R_base_imu` is incorrect:
- Gyro measurements are rotated into wrong frame
- Angular velocity appears in wrong axes
- Integration accumulates constant yaw drift
- **This drift cannot be corrected by gyro alone** (no absolute reference)

### 2. Odom Limitations

**Odom provides relative yaw, but only if:**
1. Odom frame is correctly defined
2. Yaw covariance is reasonable (not too weak/strong)
3. Odom is in the same base frame we're estimating

**Current Implementation**:
- Odom is transformed relative to first odom pose (line 508):
  ```python
  odom_relative = first_odom^{-1} ∘ odom_absolute
  ```
- This makes first odom effectively at origin
- **Issue**: If first odom has wrong yaw, all subsequent odom is biased

**Odom Evidence** (line 591 in pipeline.py):
- Computes pose error: `T_err = belief^{-1} ∘ odom`
- Uses odom covariance (permuted from ROS to GC ordering)
- **If yaw covariance is wrong**, odom cannot correct yaw drift

## Current Configuration Issues

### 1. T_base_imu Rotation

**Config values** (FIXED):
- `gc_unified.yaml`: `[0.169063, -2.692032, 0.0]` (~154.5°)
- `gc_rosbag.launch.py`: `[0.169063, -2.692032, 0.0]` (~154.5°) ✓ **NOW MATCHES**

**Previous issue**: Launch file had `[1.475086, -0.813463, -0.957187]` (111.01°) which was wrong!

**Verification needed**:
- Which value is actually used at runtime?
- Is the rotation angle correct?
- Is the rotation axis correct?

### 2. Odom Frame Alignment

**Frames**:
- Odom message: `header.frame_id = odom_combined`, `child_frame_id = base_footprint`
- Backend config: `odom_frame = odom`, `base_frame = base_footprint`

**Potential issues**:
- Is `odom_combined` the same as `odom`?
- Is odom yaw in the same frame as our base frame?
- First odom pose yaw might be wrong → all subsequent odom biased

### 3. Odom Yaw Covariance

**Need to verify**:
- What is the actual yaw covariance value in the bag?
- Is it too small (overconfident) → odom dominates but is wrong
- Is it too large (underconfident) → odom doesn't help correct drift

## Diagnostic Steps

### Step 1: Verify T_base_imu is Applied Correctly

**Check**: Is gyro rotation actually being applied?
- Location: `backend_node.py:470`
- Code: `gyro = self.R_base_imu @ gyro`
- **Action**: Log first few gyro measurements before/after rotation

### Step 2: Check Odom Yaw Covariance

**Check**: What is the actual yaw covariance?
- Extract from rosbag
- Verify it's reasonable (not 0, not huge)
- Check if it matches 2D robot pattern (low x,y,yaw, high z,roll,pitch)

### Step 3: Verify First Odom Pose

**Check**: What is the first odom pose yaw?
- Location: `backend_node.py:499-501`
- **Action**: Log first odom pose rotation
- If first odom yaw is wrong, all subsequent odom is biased

### Step 4: Check Frame Consistency

**Check**: Are odom and base frames aligned?
- Verify `odom_combined` vs `odom` naming
- Verify `base_footprint` is consistent
- Check if odom yaw convention matches our base frame

## Potential Fixes

### Fix 1: Correct T_base_imu Rotation

**If rotation is wrong**:
1. Re-run `diagnose_coordinate_frames.py` to get correct rotation
2. Update both `gc_unified.yaml` and `gc_rosbag.launch.py` to match
3. Verify rotation is applied correctly (log before/after)

### Fix 2: Fix Odom Yaw Covariance

**If covariance is wrong**:
1. Extract actual covariance from bag
2. If too small: scale it up (make odom less confident)
3. If too large: scale it down (make odom more confident)
4. Or: use adaptive covariance (IW updates)

### Fix 3: Fix First Odom Reference

**If first odom yaw is wrong**:
1. Don't use first odom as reference
2. Use ground truth first pose as reference
3. Or: Initialize from odom with known-good yaw

### Fix 4: Add Absolute Yaw Reference

**If no fix works**:
1. Use magnetometer (if available)
2. Use visual features for absolute heading
3. Use loop closures to correct accumulated drift

## Verification Checklist

- [ ] `T_base_imu` values match in config and launch file
- [ ] `T_base_imu` rotation is applied to gyro (verified in logs)
- [ ] Odom yaw covariance is reasonable (checked from bag)
- [ ] First odom pose yaw is logged and verified
- [ ] Odom frame names are consistent (`odom_combined` vs `odom`)
- [ ] Base frame names are consistent (`base_footprint`)
- [ ] Odom evidence is actually being used (check fusion weights)

## References

- Gyro rotation: `fl_slam_poc/backend/backend_node.py:470`
- Odom reference: `fl_slam_poc/backend/backend_node.py:497-508`
- Odom evidence: `fl_slam_poc/backend/operators/odom_evidence.py:36`
- Frame conventions: `docs/FRAME_AND_QUATERNION_CONVENTIONS.md`
