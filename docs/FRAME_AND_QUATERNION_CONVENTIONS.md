# Frame and Quaternion Conventions (Canonical Reference)

This document is the **single source of truth** for coordinate frame and quaternion conventions used throughout the Golden Child SLAM v2 codebase.

## Quaternion Conventions

### ROS Message Format
- **ROS quaternion order**: `[x, y, z, w]` (xyzw)
- **ROS message fields**: `orientation.x`, `orientation.y`, `orientation.z`, `orientation.w`
- **Location**: `fl_slam_poc/backend/backend_node.py:487`
  ```python
  quat = [ori.x, ori.y, ori.z, ori.w]  # xyzw format
  R = Rotation.from_quat(quat)  # scipy expects xyzw
  ```

### scipy.spatial.transform.Rotation
- **Input format**: `from_quat([x, y, z, w])` - **xyzw order**
- **Output format**: `as_quat()` returns `[x, y, z, w]` - **xyzw order**
- **Verification**: scipy documentation confirms xyzw is the standard

### Internal Representation
- **NOT quaternions**: Internal state uses **rotation vectors (rotvec)** in radians
- **Format**: `[rx, ry, rz]` where `||[rx, ry, rz]||` is the rotation angle in radians
- **Conversion**: Always via `Rotation.from_quat()` → `as_rotvec()` or `from_rotvec()` → `as_quat()`

### TUM Format Export
- **TUM format**: `timestamp x y z qx qy qz qw` (xyzw)
- **Location**: `fl_slam_poc/backend/backend_node.py:936`
  ```python
  f"{stamp_sec:.9f} {trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} "
  f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
  # quat[0:3] = [x, y, z], quat[3] = w
  ```

## Coordinate Frame Conventions

### Frame Names (from rosbag)
- **base_footprint**: Body/base frame (Z-up convention)
- **livox_frame**: Livox MID-360 sensor frame (Z-down convention)
- **camera_imu_optical_frame**: RealSense D435i IMU frame
- **odom_combined**: Odometry frame (parent of base_footprint)

### Frame Transformations

#### LiDAR Frame (Livox MID-360)
- **Sensor frame**: `livox_frame` (Z-down)
- **Base frame**: `base_footprint` (Z-up)
- **Extrinsic**: `T_base_lidar = [x, y, z, rx, ry, rz]` (rotvec in radians)
- **Current config**: `[-0.011, 0.0, 0.778, 0.0, 0.0, 0.0]`
  - Translation: `[-0.011, 0.0, 0.778]` meters
  - Rotation: `[0, 0, 0]` (identity - Z-up confirmed via ground plane analysis)
- **Transformation**: `p_base = R_base_lidar @ p_lidar + t_base_lidar`
- **Location**: `fl_slam_poc/backend/backend_node.py:556`

#### IMU Frame
- **Current sensor**: `/livox/mid360/imu` → `livox_frame`
- **Alternative**: `/camera/imu` → `camera_imu_optical_frame` (not currently used)
- **Extrinsic**: `T_base_imu = [x, y, z, rx, ry, rz]` (rotvec in radians)
- **Current config**: `[0.0, 0.0, 0.0, -0.026, 0.488, 0.0]`
  - Translation: `[0.0, 0.0, 0.0]` (co-located with LiDAR)
  - Rotation: `[-0.026, 0.488, 0.0]` radians (~28° to align gravity with +Z)
- **Transformation**: `accel_base = R_base_imu @ accel_imu`
- **Location**: `fl_slam_poc/backend/backend_node.py:470-471`

## Gravity and IMU Conventions (CRITICAL)

### Gravity Vector
- **World frame convention**: Z-up (gravity points DOWN in -Z direction)
- **GC_GRAVITY_W**: `[0.0, 0.0, -9.81]` m/s² (gravity acceleration vector)
- **Location**: `fl_slam_poc/common/constants.py:38`

### Accelerometer Convention
- **Standard IMU convention**: Accelerometer measures **reaction to gravity** (specific force)
- **Level sensor reading**: When level and stationary, accelerometer reads `[0, 0, +g]` (pointing UP)
- **Physical interpretation**: Accelerometer measures the force preventing freefall
- **NOT gravity itself**: The reading is opposite to gravity direction

### IMU Evidence Expected Direction
- **g_hat**: Normalized gravity direction = `[0, 0, -1]` (pointing DOWN)
- **minus_g_hat**: Expected accel direction = `[0, 0, +1]` (pointing UP)
- **mu0**: Expected accel in body frame = `R_body^T @ minus_g_hat`
- **xbar**: Measured accel direction in body frame (should match mu0 when level)
- **Alignment check**: `xbar @ mu0` should be ~+1.0 for correct gravity alignment
- **Location**: `fl_slam_poc/backend/operators/imu_evidence.py:148-170`

### IMU Units
- **Livox MID-360 raw output**: Acceleration in **g's** (1g ≈ 9.81 m/s²)
- **GC_IMU_ACCEL_SCALE**: `9.81` (conversion factor from g's to m/s²)
- **Internal units**: All acceleration in **m/s²**
- **Location**: `fl_slam_poc/backend/backend_node.py:466`
  ```python
  accel = accel_raw * constants.GC_IMU_ACCEL_SCALE  # g → m/s²
  ```

### Gyroscope Convention
- **Gyro measures**: Angular velocity `[wx, wy, wz]` in rad/s
- **Right-hand rule**: Positive rotation about +Z axis is counter-clockwise when viewed from above
- **After extrinsic**: `gyro_base = R_base_imu @ gyro_imu`
- **Location**: `fl_slam_poc/backend/backend_node.py:470`

### IMU Extrinsic Verification
To verify IMU extrinsic is correct:
1. Read raw IMU accel when sensor is level: expect mostly +Z in IMU frame
2. Apply `R_base_imu @ accel_imu`
3. Result should have Z-component ~+9.81 m/s² (reaction to gravity pointing UP)
4. If Z-component is negative, extrinsic is inverting gravity

```python
# Verification code
accel_imu = np.array([ax, ay, az]) * 9.81  # raw IMU in m/s²
accel_base = R_base_imu @ accel_imu
# accel_base[2] should be positive ~+9.81 for level sensor
```

## SE(3) Pose Representation

### Internal Format
- **6D pose**: `[trans(3), rotvec(3)]` = `[x, y, z, rx, ry, rz]`
- **Units**: Translation in meters, rotation in radians
- **Functions**: `se3_from_rotvec_trans(rotvec, trans)` and `se3_to_rotvec_trans(pose)`
- **Location**: `fl_slam_poc/common/belief.py:89-102`

### State Vector (22D)
- **Ordering**: `[trans(3), rot(3), vel(3), bg(3), ba(3), dt(1), ex(6)]`
- **Pose slice**: `[trans(0:3), rot(3:6)]` = `[tx, ty, tz, rx, ry, rz]`
- **Note**: State and SE(3) now share the same `[trans, rot]` ordering
- **Conversion**: `pose_se3_to_z_delta()` and `pose_z_to_se3_delta()` are identity (kept for compatibility)

## Covariance Ordering

### ROS Odometry Covariance
- **ROS convention**: `[x, y, z, roll, pitch, yaw]` = `[trans(0:3), rot(3:6)]`
- **Location**: `fl_slam_poc/backend/backend_node.py:511`
- **Comment**: "Pose covariance is row-major 6x6: [x,y,z,roll,pitch,yaw]"

### GC Internal Covariance
- **GC convention**: `[x, y, z, roll, pitch, yaw]` = `[trans(0:3), rot(3:6)]`
- **Permutation**: **None** (GC now matches ROS ordering)
- **Location**: `fl_slam_poc/backend/operators/odom_evidence.py`
  ```python
  # ROS pose covariance: [x, y, z, roll, pitch, yaw] = [trans(0:3), rot(3:6)]
  # GC pose ordering:    [tx, ty, tz, rx, ry, rz]    = [trans(0:3), rot(3:6)]
  cov = cov_ros  # no permutation needed
  ```

## Rotation Representations

### Rotation Matrix (R)
- **Format**: 3×3 orthogonal matrix ∈ SO(3)
- **Properties**: `R @ R.T = I`, `det(R) = +1`
- **Conversion**: `R = so3_exp(rotvec)` and `rotvec = so3_log(R)`
- **Location**: `fl_slam_poc/common/geometry/se3_jax.py`

### Rotation Vector (rotvec)
- **Format**: `[rx, ry, rz]` in radians
- **Angle**: `θ = ||[rx, ry, rz]||`
- **Axis**: `[rx, ry, rz] / θ` (normalized)
- **Rodrigues formula**: `R = I + sin(θ)/θ · [ω]× + (1-cos(θ))/θ² · [ω]×²`

### Quaternion (q)
- **Format**: `[x, y, z, w]` (xyzw)
- **Only used for**: ROS message I/O, TUM export
- **NOT used internally**: All internal calculations use rotvec

## SE(3) Operations

### Composition
- **Function**: `se3_compose(a, b)` → `T_a ∘ T_b`
- **Formula**: `t_out = t_a + R_a @ t_b`, `R_out = R_a @ R_b`
- **Location**: `fl_slam_poc/common/geometry/se3_jax.py:421`

### Inverse
- **Function**: `se3_inverse(a)` → `T_a^{-1}`
- **Formula**: `R_inv = R.T`, `t_inv = -R_inv @ t`
- **Location**: `fl_slam_poc/common/geometry/se3_jax.py:442`

### Relative Transform
- **Function**: `se3_relative(a, b)` → `a ⊖ b = b^{-1} ∘ a`
- **Used for**: Pose error computation in odom evidence
- **Location**: `fl_slam_poc/common/geometry/se3_jax.py:457`

## Verification Checklist

### Quaternion I/O
- [x] ROS messages: `[ori.x, ori.y, ori.z, ori.w]` = xyzw ✓
- [x] scipy.from_quat: expects xyzw ✓
- [x] scipy.as_quat: returns xyzw ✓
- [x] TUM export: `qx qy qz qw` = xyzw ✓

### Frame Transformations
- [x] LiDAR: `p_base = R_base_lidar @ p_lidar + t_base_lidar` ✓
- [x] IMU: `accel_base = R_base_imu @ accel_imu` ✓
- [x] Extrinsics format: `[x, y, z, rx, ry, rz]` (rotvec) ✓

### Covariance Ordering
- [x] ROS odom: `[x, y, z, roll, pitch, yaw]` = `[trans, rot]` ✓
- [x] GC internal: `[tx, ty, tz, rx, ry, rz]` = `[trans, rot]` ✓
- [x] No permutation applied (orderings match) ✓

### SE(3) Representation
- [x] Internal pose: `[trans(3), rotvec(3)]` = `[x, y, z, rx, ry, rz]` ✓
- [x] State vector: `[trans(3), rot(3), ...]` = `[tx, ty, tz, rx, ry, rz, ...]` ✓
- [x] Conversion functions are identity (ordering is unified) ✓

## Potential Issues

### 1. Quaternion Convention Consistency
- **Status**: ✓ Consistent - all use xyzw
- **Verification**: scipy.spatial.transform.Rotation uses xyzw by default

### 2. Frame Convention Documentation
- **Status**: ⚠️ Partially documented
- **Issue**: LiDAR Z-up vs Z-down was historically ambiguous
- **Resolution**: Confirmed Z-up via ground plane analysis (diagnose_coordinate_frames.py)

### 3. Covariance Permutation
- **Status**: ✓ Removed (no longer needed)
- **Location**: `odom_evidence.py` uses ROS covariance directly

### 4. SE(3) vs State Ordering
- **Status**: ✓ Unified
- **Note**: SE(3) and state both use `[trans, rot]`
- **Conversion**: `pose_se3_to_z_delta()` and `pose_z_to_se3_delta()` are identity

## References

- **scipy.spatial.transform.Rotation**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
- **ROS geometry_msgs/Quaternion**: http://docs.ros.org/en/api/geometry_msgs/html/msg/Quaternion.html
- **TUM format**: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
- **SE(3) Lie group**: Barfoot (2017), Forster et al. (2017)
