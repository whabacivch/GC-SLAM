# Diagnostic Tools for Coordinate Frames & Extrinsics

This document lists all available tools to diagnose coordinate frame issues.

## Quick Start

Run the comprehensive diagnostic:
```bash
# Use venv Python directly (recommended)
.venv/bin/python tools/diagnose_coordinate_frames.py rosbags/m3dgr/Dynamic01_ros2

# Or activate venv first, then python3 will use venv Python
source .venv/bin/activate
python3 tools/diagnose_coordinate_frames.py rosbags/m3dgr/Dynamic01_ros2
```

This will tell you:
1. **LiDAR frame convention** (Z-up vs Z-down) from raw point cloud analysis
2. **IMU gravity direction** in sensor frame vs expected base frame
3. **Odom covariance ordering** (ROS standard; GC now matches ROS)

## Available Tools

### 1. `diagnose_coordinate_frames.py` (NEW - Comprehensive)

**Purpose**: First-principles diagnostic of all coordinate frame conventions.

**What it checks**:
- LiDAR Z-convention by analyzing ground plane normal
- IMU gravity direction vs expected base frame
- Odom covariance ordering (ROS [x,y,z,roll,pitch,yaw]; GC uses the same `[trans, rot]` ordering)

**Usage**:
```bash
.venv/bin/python tools/diagnose_coordinate_frames.py <bag_path> [--n-scans 20]
```

**Output**: Clear interpretation of what conventions are actually used.

---

### 2. `check_lidar_mount_angle.py`

**Purpose**: Check if LiDAR is mounted horizontally or at an angle.

**What it checks**:
- Principal component analysis of point clouds
- Angle between principal components and Z-axis
- Z-distribution statistics

**Usage**:
```bash
.venv/bin/python tools/check_lidar_mount_angle.py <bag_path> [--n-scans 10]
```

**Output**: Tells you if lidar is horizontal (rotation [0,0,0] likely correct) or angled.

---

### 3. `estimate_imu_base_extrinsic_rotation.py`

**Purpose**: Estimate IMU rotation from gravity alignment.

**What it checks**:
- IMU accelerometer direction when stationary
- Compares with expected gravity in base frame
- Estimates rotation vector to align them

**Usage**:
```bash
.venv/bin/python tools/estimate_imu_base_extrinsic_rotation.py <bag_path>
```

**Output**: Suggested `T_base_imu` rotation vector.

---

### 4. `estimate_lidar_base_extrinsic.py`

**Purpose**: Estimate LiDAR extrinsics using hand-eye calibration.

**What it checks**:
- ICP-based scan-to-scan alignment
- Compares with odometry/ground truth motion
- Solves hand-eye calibration problem

**Usage**:
```bash
.venv/bin/python tools/estimate_lidar_base_extrinsic.py <bag_path> --base-source odom
```

**Output**: Estimated `T_base_lidar` transform.

---

### 5. `inspect_rosbag_deep.py`

**Purpose**: Deep inspection of rosbag structure.

**What it checks**:
- All topics, message types, frame IDs
- Message counts, timestamps
- QoS settings
- TF presence/absence

**Usage**:
```bash
.venv/bin/python tools/inspect_rosbag_deep.py <bag_path> [--json output.json]
```

**Output**: Complete bag inventory.

---

### 6. `inspect_odom_source.py`

**Purpose**: Inspect odometry messages.

**What it checks**:
- Frame IDs
- Covariance values
- Pose values

**Usage**:
```bash
.venv/bin/python tools/inspect_odom_source.py <bag_path>
```

---

### 7. `diagnose_frame_offset.py`

**Purpose**: Diagnose constant rotation offset in trajectory.

**What it checks**:
- Mean rotation error
- Per-axis rotation errors
- Constant offset detection

**Usage**:
```bash
.venv/bin/python tools/diagnose_frame_offset.py <gt_trajectory.tum> <est_trajectory.tum>
```

---

## Current Known Issues

### LiDAR Frame Convention
- **Documentation says**: `livox_frame` is Z-down, needs `[π, 0, 0]` rotation
- **Code has**: `[0, 0, 0]` (identity) - reverted pending verification
- **Action**: Run `diagnose_coordinate_frames.py` to verify

### IMU Frame Convention
- **Current**: `T_base_imu = [0, 0, 0, -0.015586, 0.489293, 0.0]`
- **Source**: Estimated from gravity alignment (~28° misalignment)
- **Action**: Run `estimate_imu_base_extrinsic_rotation.py` to re-verify

### Odom Covariance Ordering
- **ROS convention**: `[x, y, z, roll, pitch, yaw]` = `[trans, rot]`
- **GC convention**: `[tx, ty, tz, rx, ry, rz]` = `[trans, rot]` (matches ROS)
- **Status**: Unified ordering; no permutation applied
- **Action**: Run `diagnose_coordinate_frames.py` to verify odom covariance is ROS-standard

## Transformation Chain

### LiDAR Points
1. **Raw from bag**: Points in `livox_frame` (sensor frame)
2. **After `livox_converter`**: Still in `livox_frame` (frame_id preserved)
3. **In `backend_node.on_lidar()`** (line 556):
   ```python
   pts_base = (self.R_base_lidar @ pts_np.T).T + self.t_base_lidar[None, :]
   ```
   - Transforms to `base_footprint` frame
   - Formula: `p_base = R_base_lidar @ p_lidar + t_base_lidar`
4. **In pipeline**: All processing happens in `base_footprint` frame

### IMU Data
1. **Raw from bag**: IMU data in `livox_frame`
2. **In `imu_normalizer`**: Frame ID preserved (no transform applied)
3. **In `backend_node`**: IMU samples are rotated using `R_base_imu` before preintegration
   - Location: Check `backend_node.py` for IMU transformation

## Verification Checklist

Before trusting any extrinsics, verify:

- [ ] Run `diagnose_coordinate_frames.py` and review all interpretations
- [ ] Verify LiDAR Z-convention matches your `T_base_lidar` rotation
- [ ] Verify IMU gravity direction matches expected base frame
- [ ] Verify odom covariance ordering is `[trans, rot]` (ROS standard, matches GC)
- [ ] Check that transformations are applied in the right places
- [ ] Verify frame IDs are consistent across the pipeline

## Next Steps After Diagnosis

1. **If LiDAR is Z-down**: Set `T_base_lidar = [-0.011, 0.0, 0.778, 3.141593, 0.0, 0.0]`
2. **If LiDAR is Z-up**: Keep `T_base_lidar = [-0.011, 0.0, 0.778, 0.0, 0.0, 0.0]`
3. **If IMU gravity is wrong**: Re-run `estimate_imu_base_extrinsic_rotation.py`
4. **If odom ordering is wrong**: Verify the incoming `/odom` covariance is ROS-standard (no permutation in `odom_evidence.py`)
