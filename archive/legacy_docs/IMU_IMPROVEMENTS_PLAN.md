# IMU Integration Improvements Plan

## Priority #1: Rotation Frame Issue (Critical)

**Problem**: ATE rotation error is ~135° (should be <20°). This indicates a systematic frame convention mismatch.

**Root Cause**: Quaternion/rotation vector conversions near π (180°) are numerically unstable and may have sign/axis ambiguities.

**Solution**:
1. ✅ Created comprehensive unit tests (`test_rotation_conversions.py`) covering:
   - Roundtrip conversions near π
   - Exactly π rotations
   - Large rotation vectors (>2π)
   - Quaternion sign consistency
   - Frame convention verification

2. **Next Steps**:
   - Run tests to identify specific failures
   - Fix `rotmat_to_rotvec` near π (eigenvalue decomposition path)
   - Fix `quat_to_rotvec` to handle π rotations correctly
   - Add explicit frame convention documentation
   - Verify all conversion paths use consistent conventions

**Expected Outcome**: ATE rotation error drops from ~135° to <20°.

---

## Priority #2: IMU Anchor Matching Enhancement

**Problem**: IMU factors may not match to correct anchors if `keyframe_to_anchor` mapping is missing.

**Solution**: Enhanced matching with fallback:
1. **Primary**: Use `keyframe_to_anchor[keyframe_i]` if mapping exists
2. **Fallback**: Nearest-neighbor search using Hellinger distance on pose distributions
3. **Leverage**: Use existing Hellinger tilt logic from kernel

**Implementation**:
- Compute Hellinger distance between IMU factor's predicted anchor pose and all candidate anchors
- Select anchor with minimum Hellinger distance
- Use this distance as initial routing logit (stronger match = higher logit)

**Expected Outcome**: More accurate IMU factor association, better velocity/bias constraints.

---

## Priority #3: Trajectory Termination Handling

**Problem**: Last odom/IMU messages may be lost when rosbag playback ends, causing incomplete trajectory.

**Solution**: Post-rosbag processing queue:
1. Buffer last N odom/IMU messages in a "post-rosbag" queue
2. On node shutdown, process remaining messages
3. Ensure trajectory file is flushed and closed properly

**Implementation**:
- Add `post_rosbag_queue` for odom/IMU messages
- Add shutdown handler that processes queue
- Add explicit trajectory file flush on shutdown

**Expected Outcome**: Complete trajectory coverage, no lost end-of-trajectory data.

---

## Priority #4: Performance Optimization

**Current State**: 
- ATE Translation: 2.30 m
- ATE Rotation: 135.54°
- RPE @ 1m: 0.81 m/m

**Expected with More IMU**:
- More IMU factors = tighter velocity/bias constraints
- Better observability of rotation (gyro) and translation (accel)
- **Target**: Sub-1m ATE translation

**Key Metrics to Monitor**:
- IMU factor rate (should match keyframe rate)
- Bias convergence (should stabilize over time)
- Whitened residual norms (should be stable, ~1-3)
- Hellinger shift diagnostics (should be small for good matches)

---

## Implementation Checklist

### Rotation Frame Fixes
- [x] Create unit tests for rotation conversions near π
- [ ] Fix `rotmat_to_rotvec` near π singularity
- [ ] Fix `quat_to_rotvec` for π rotations
- [ ] Add frame convention documentation
- [ ] Verify all conversion paths
- [ ] Run tests and verify ATE rotation <20°

### IMU Anchor Matching
- [ ] Enhance `on_imu_factor` to use `keyframe_to_anchor` mapping
- [ ] Add nearest-neighbor fallback with Hellinger distance
- [ ] Integrate Hellinger distance into routing logits
- [ ] Test with missing keyframe mappings

### Trajectory Termination
- [ ] Add post-rosbag message queue
- [ ] Add shutdown handler
- [ ] Process queue on shutdown
- [ ] Ensure trajectory file flush

### Performance Monitoring
- [ ] Add IMU factor rate logging
- [ ] Add bias norm tracking
- [ ] Add whitened residual norm logging
- [ ] Add Hellinger shift diagnostics

---

## Files Modified/Created

1. **`test/test_rotation_conversions.py`** (NEW) - Comprehensive rotation conversion tests
2. **`backend/backend_node.py`** - Enhanced IMU anchor matching, trajectory termination
3. **`docs/IMU_IMPROVEMENTS_PLAN.md`** (THIS FILE) - Implementation plan

---

## Testing Strategy

1. **Rotation Tests**: Run `pytest test_rotation_conversions.py -v` to identify failures
2. **IMU Matching**: Test with/without keyframe mappings, verify Hellinger fallback
3. **Trajectory**: Run full rosbag, verify complete trajectory coverage
4. **Performance**: Compare metrics before/after fixes

---

## Success Criteria

- ✅ ATE rotation <20° (from ~135°)
- ✅ IMU factors correctly matched to anchors
- ✅ Complete trajectory coverage (no missing end data)
- ✅ ATE translation <1m (with more IMU factors)
