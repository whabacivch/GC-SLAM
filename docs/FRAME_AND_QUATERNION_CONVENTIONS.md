# Frame and Quaternion Conventions (Canonical Reference)

This document is the **single source of truth** for coordinate frame and quaternion conventions used throughout the Golden Child SLAM v2 codebase.

## Status Labels (Read This First)

Each convention below is marked with one of:

- **CONFIRMED**: Empirically validated on the current dataset (or by code-level invariants).
- **ASSUMED-CONTRACT**: Declared as a required contract, but not yet empirically validated.
- **TO-CONFIRM**: We believe it is true, but it still needs an explicit empirical check.

When something is **TO-CONFIRM**, we should either:
1) Add a deterministic validation script/log (preferred), or
2) Treat violations as a hard error at runtime (fail-fast), or
3) Downgrade it to an explicitly declared assumption with consequences.

## Quaternion Conventions

### ROS Message Format
- **CONFIRMED**: ROS quaternion order is `[x, y, z, w]` (xyzw)
- **ROS message fields**: `orientation.x`, `orientation.y`, `orientation.z`, `orientation.w`
- **Location**: `fl_slam_poc/backend/backend_node.py:487`
  ```python
  quat = [ori.x, ori.y, ori.z, ori.w]  # xyzw format
  R = Rotation.from_quat(quat)  # scipy expects xyzw
  ```

### scipy.spatial.transform.Rotation
- **CONFIRMED**: `Rotation.from_quat([x, y, z, w])` expects xyzw and `as_quat()` returns xyzw.
- **Verification**: scipy documentation confirms xyzw is the standard.

### Internal Representation
- **CONFIRMED**: Internal state uses **rotation vectors (rotvec)** in radians (not quaternions).
- **Format**: `[rx, ry, rz]` where `||[rx, ry, rz]||` is the rotation angle in radians
- **Conversion**: Always via `Rotation.from_quat()` → `as_rotvec()` or `from_rotvec()` → `as_quat()`

### TUM Format Export
- **CONFIRMED**: TUM format is `timestamp x y z qx qy qz qw` (xyzw).
- **Location**: `fl_slam_poc/backend/backend_node.py:936`
  ```python
  f"{stamp_sec:.9f} {trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} "
  f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
  # quat[0:3] = [x, y, z], quat[3] = w
  ```

## Coordinate Frame Conventions

## Global Frame Model (What Is "World" in GC v2?)

GC v2 runs without `/tf` as a dependency, so we must declare a single, stable
"world" frame by construction.

- **ASSUMED-CONTRACT**: We treat the **world / map frame** as the frame used by the odometry message's
  `header.frame_id` (in this bag: `odom_combined`).
- **CONFIRMED (Dynamic01_ros2)**: `header.frame_id` is stable across the entire rosbag (0 changes observed).
- **ASSUMED-CONTRACT**: The state pose is interpreted as **pose of the body in the world** as an SE(3)
  element `T_{world<-body}`.
- **ASSUMED-CONTRACT**: Runtime should fail-fast if `header.frame_id` changes mid-run (add explicit audit if/when needed).

Body axes convention (GC assumes a standard mobile base convention):

- **CONFIRMED (Dynamic01_ros2)**: `base_footprint` behaves as a standard planar mobile base frame:
  linear motion is along +X and yaw is about +Z (right-handed, ENU-like).

- **CONFIRMED (Dynamic01_ros2)**: The odom twist is purely planar (`v_x` and `ω_z` nonzero; `v_y=v_z=ω_x=ω_y=0`)
  and the pose+twist satisfy `dp_parent ≈ R_parent_child @ v_child * dt` with cosine similarity ~0.9998 in XY.
  This confirms the intended base planar axes usage in this dataset.

### Frame Names (from rosbag)
- **CONFIRMED (Dynamic01_ros2)**: `base_footprint` is the body/base frame (planar base; Z-up convention implied by covariance and gravity alignment).
- **CONFIRMED (Dynamic01_ros2)**: `livox_frame` is the Livox MID-360 sensor frame and is **Z-up** for this dataset (ground normal points +Z).
- **camera_imu_optical_frame**: RealSense D435i IMU frame
- **odom_combined**: Odometry frame (parent of base_footprint)

**Dataset-specific frame mapping:** For Kimera bags, frame names and axis verification are documented in [KIMERA_FRAME_MAPPING.md](KIMERA_FRAME_MAPPING.md).

### Odometry Message Semantics (nav_msgs/Odometry) - MUST MATCH SE(3) MATH

This project does not rely on `/tf` at runtime, so we must be explicit about
how we interpret `nav_msgs/Odometry` by construction.

Canonical ROS meaning (per message comments and common TF usage):

- **CONFIRMED (bag)**: `header.frame_id` is used as the **parent** frame (e.g., `odom_combined`).
- **CONFIRMED (bag)**: `child_frame_id` is used as the **child** frame (e.g., `base_footprint`).
- **CONFIRMED (code+bag)**: `msg.pose.pose` is interpreted as the rigid transform `T_{parent<-child}` in the usual
  rigid-body form:

  - Rotation: `R_parent_child` (applied to child-frame vectors to express them in parent)
  - Translation: `t_parent_child` (child origin expressed in parent coordinates)

  For any point `p_child` expressed in the child frame:

  ```text
  p_parent = R_parent_child * p_child + t_parent_child
  ```

This is exactly the convention used throughout our internal SE(3) operators
(`se3_compose`, `se3_inverse`) which implement:

```text
T_out = T_a ∘ T_b
t_out = t_a + R_a * t_b
R_out = R_a * R_b
```

Therefore:

- **CONFIRMED (pipeline tracing)**: When ingesting odometry, we must treat the pose as
  `T_{header.frame_id<-child_frame_id}` and **must not invert it** at the ingestion boundary.
- If we ever need the opposite transform (child<-parent), we compute it explicitly via
  `se3_inverse()` / matrix transpose rules. No silent flips.

Twist semantics:

- **CONFIRMED (Dynamic01_ros2)**: `msg.twist` is specified in the coordinate frame given by `child_frame_id`
  (linear/angular velocity expressed in the child/body frame). Empirically, the finite-difference
  position increments match `dp_parent ≈ R_parent_child @ v_child * dt` with cosine similarity ~0.9998 in XY.
- GC v2 currently uses odom as a pose observation (not a twist constraint).

### Frame Transformations

### Transform Chain (One Table, No Ambiguity)

All transforms in this system are applied in the standard rigid-body form:

```text
p_parent = R_parent_child * p_child + t_parent_child
```

| Symbol / Parameter | Meaning | Direction | Where applied |
|---|---|---|---|
| `T_base_lidar` | base<-lidar extrinsic | `T_{base<-lidar}` | LiDAR point ingest (`backend_node.py`) |
| `T_base_imu` | base<-imu extrinsic | `T_{base<-imu}` | IMU gyro/accel ingest (`backend_node.py`) |
| `/odom` pose | parent<-child from message | `T_{header.frame_id<-child_frame_id}` | Odom ingest (`backend_node.py`) |
| `X_WL` state pose | world<-body | `T_{world<-body}` | Belief/pipeline pose (`belief.mean_world_pose`) |

Practical note for this codebase today:

- **CONFIRMED (code)**: We transform incoming LiDAR points into **base frame** before any inference.
  Therefore, the "LiDAR body" used by the state and LiDAR evidence effectively
  coincides with `base_footprint` for geometry. The LiDAR origin offset is still
  used for ray directions via `t_base_lidar`.

### Extrinsics Direction Convention (No Silent Inverses)

All configured extrinsics in this repo follow the same rule:

- `T_base_sensor` means `T_{base<-sensor}`.
- Points/vectors are transformed via:

  ```text
  p_base = R_base_sensor * p_sensor + t_base_sensor
  ```

If the inverse is needed, it must be computed explicitly:

```text
R_sensor_base = R_base_sensor^T
t_sensor_base = -R_base_sensor^T * t_base_sensor
```

No swapping of parameter names and no implicit "maybe it's the other direction".

#### LiDAR Frame (Livox MID-360)
- **Sensor frame**: `livox_frame` (**CONFIRMED (Dynamic01_ros2): Z-up**)
- **Base frame**: `base_footprint` (Z-up)
- **Extrinsic**: `T_base_lidar = [x, y, z, rx, ry, rz]` (rotvec in radians)
- **TO-CONFIRM**: `T_base_lidar` translation values are correct for this dataset:
  `t_base_lidar = [-0.011, 0.0, 0.778]` meters (requires physical/calibration confirmation).

  Note: We attempted to infer sensor height from raw LiDAR scans, but in this dataset the dominant planar
  structures are not consistently the ground plane (many points lie near z≈0 and there are large negative
  z outliers), so a purely bag-based "ground height" estimate is not stable enough to certify `tz=0.778`.
  See `results/confirm_remaining_dynamic01.json` for the attempted estimates and why they are inconclusive.
- **CONFIRMED (Dynamic01_ros2)**: `T_base_lidar` rotation is identity (`rotvec=[0,0,0]`) and consistent with the bag's
  point cloud Z-up convention (ground normal points +Z in `livox_frame`).
- **If evaluation shows ~180° roll offset vs ground truth:** Run `tools/diagnose_coordinate_frames.py` on the bag. If it reports Z-down, set `T_base_lidar` rotation to `[π, 0, 0]` and update this subsection to record dataset-specific Z-down. See `docs/PIPELINE_DESIGN_GAPS.md` §5.5.
- **Transformation**: `p_base = R_base_lidar @ p_lidar + t_base_lidar`
- **Location**: `fl_slam_poc/backend/backend_node.py:556`

#### IMU Frame
- **Current sensor**: `/livox/mid360/imu` → `livox_frame`
- **Alternative**: `/camera/imu` → `camera_imu_optical_frame` (not currently used)
- **Extrinsic**: `T_base_imu = [x, y, z, rx, ry, rz]` (rotvec in radians)
- **CONFIRMED (Dynamic01_ros2)**: `T_base_imu` rotation is gravity-aligned for this dataset:
  applying `R_base_imu` to the mean specific force yields `a_base_mean ≈ [0, 0, +g]`.
  Current value used in runs: `[0.0, 0.0, 0.0, -0.015586, 0.489293, 0.0]` (~28°).
  - Translation: `[0.0, 0.0, 0.0]` (co-located with LiDAR)
  - Rotation: `[-0.015586, 0.489293, 0.0]` radians (~28° to align specific force with +Z)
- **Transformation**: `accel_base = R_base_imu @ accel_imu`
- **Location**: `fl_slam_poc/backend/backend_node.py:470-471`

## Gravity and IMU Conventions (CRITICAL)

### Gravity Vector
- **ASSUMED-CONTRACT**: World frame convention is Z-up (gravity points DOWN in -Z direction).
- **GC_GRAVITY_W**: `[0.0, 0.0, -9.81]` m/s² (gravity acceleration vector)
- **Location**: `fl_slam_poc/common/constants.py:38`

### Accelerometer Convention
- **ASSUMED-CONTRACT**: Accelerometer measures **reaction to gravity** (specific force).
- **Level sensor reading**: When level and stationary, accelerometer reads `[0, 0, +g]` (pointing UP)
- **Physical interpretation**: Accelerometer measures the force preventing freefall
- **NOT gravity itself**: The reading is opposite to gravity direction

#### Historical Note (Tooling Bug, Now Fixed)

Earlier versions of `tools/diagnose_coordinate_frames.py` incorrectly compared the mean accelerometer
direction against the **gravity direction** `[0, 0, -1]`. This is physically wrong for IMUs:
accelerometers measure **specific force**, so the stationary expectation in a Z-up base is
`[0, 0, +1]` (up).

Symptom of the bug:
- It reported an apparent ~180°/large "IMU misalignment" (e.g., ~154°) even when the IMU extrinsic
  was gravity-aligned.

Fix:
- The tool now compares against expected **specific force** (+Z) and reports the true misalignment
  (~25-30° for this dataset with the current `T_base_imu`).

### IMU Evidence Expected Direction
- **g_hat**: Normalized gravity direction = `[0, 0, -1]` (pointing DOWN)
- **minus_g_hat**: Expected accel direction = `[0, 0, +1]` (pointing UP)
- **mu0**: Expected accel in body frame = `R_body^T @ minus_g_hat`
- **xbar**: Measured accel direction in body frame (should match mu0 when level)
- **Alignment check**: `xbar @ mu0` should be ~+1.0 for correct gravity alignment
- **Location**: `fl_slam_poc/backend/operators/imu_evidence.py:148-170`

### IMU Units
- **CONFIRMED (Dynamic01_ros2)**: Livox MID-360 raw output acceleration units are **g's** for this dataset
  (mean ||a_raw|| ≈ 0.998 g; after scaling by 9.81, mean ||a|| ≈ 9.79 m/s²).
- **GC_IMU_ACCEL_SCALE**: `9.81` (conversion factor from g's to m/s²)
- **Internal units**: All acceleration in **m/s²**
- **Location**: `fl_slam_poc/backend/backend_node.py:466`
  ```python
  accel = accel_raw * constants.GC_IMU_ACCEL_SCALE  # g → m/s²
  ```

### Gyroscope Convention
- **CONFIRMED (Dynamic01_ros2, base frame)**: Gyro angular velocity magnitudes are consistent with rad/s
  (mean ||ω|| ≈ 0.157 rad/s, max ≈ 0.649 rad/s), and the sign of `ωz` after applying `R_base_imu`
  agrees with odom yaw-rate sign on average (positive sign-product mean).
- **TO-CONFIRM (native IMU frame)**: Exact native-frame axis and sign conventions of the IMU driver
  (the base-frame convention is what GC uses after applying `T_base_imu`).
  We can, however, empirically confirm that after applying `R_base_imu`, `omega_base.z` correlates with
  odom yaw-rate with positive sign and high correlation (see `results/confirm_remaining_dynamic01.json`).
- **Right-hand rule**: Positive rotation about +Z axis is counter-clockwise when viewed from above
- **After extrinsic**: `gyro_base = R_base_imu @ gyro_imu`
- **Location**: `fl_slam_poc/backend/backend_node.py:470`

## Yaw / Rotation Sign Convention (Audit-Ready)

We must be explicit about yaw extraction because "yaw sign" bugs are almost
always "different yaw extraction" or "different frame convention" bugs.

Yaw extraction used by the pipeline invariant test (**CONFIRMED (code)**):

```text
yaw(R) = atan2(R[1,0], R[0,0])
```

Interpretation:

- `R` is treated as a body->world (child->parent) rotation when used for state/world pose.
- Positive yaw corresponds to right-hand rotation about +Z (counter-clockwise when looking down +Z).

When comparing yaw increments from different sources (gyro, odom, **Matrix Fisher**), the
same `yaw(R)` function must be used.

## Time / Stamp Semantics (No Hidden Alignment Heuristics)

LiDAR:

- **CONFIRMED (code)**: `t_scan` is `PointCloud2.header.stamp` (scan reference time).
- Per-point time offsets (if present) are only for **within-scan** deskew. For this
  bag (`livox_ros_driver2`), offsets are often all zeros.

IMU:

- **CONFIRMED (code)**: `dt_int` is the sum of actual IMU sample intervals within the integration window
  `(t_last_scan, t_scan)`.
- Membership weights are soft/continuous; there is no hard gating window.

Odometry:

- Odometry is treated as a pose observation at `msg.header.stamp`.
- GC **does** use odom twist as constraints (velocity factor, yaw‑rate factor, pose–twist kinematic consistency).

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
- [x] Odom pose is `T_{header.frame_id<-child_frame_id}` (parent<-child) ✓

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
- **Status**: ✓ Documented explicitly
- **LiDAR**: Z-up vs Z-down was historically ambiguous; confirmed Z-up via ground plane analysis (diagnose_coordinate_frames.py)
- **Odometry**: Documented `nav_msgs/Odometry` pose semantics explicitly as `T_{parent<-child}` to match our SE(3) operators.

### 3. Covariance Permutation
- **Status**: ✓ Removed (no longer needed)
- **Location**: `odom_evidence.py` uses ROS covariance directly

### 4. SE(3) vs State Ordering
- **Status**: ✓ Unified
- **Note**: SE(3) and state both use `[trans, rot]`
- **Conversion**: `pose_se3_to_z_delta()` and `pose_z_to_se3_delta()` are identity

### 5. Common Failure Modes (Fast Debug Checklist)

These are not "heuristics"; they are deterministic symptoms of convention mismatches.

1) Symptom: `dyaw_gyro` has opposite sign to both `dyaw_odom` and `dyaw_wahba`
   - Likely cause: IMU axis convention mismatch (wrong `T_base_imu` rotation) or IMU driver axis sign.
   - Verify: log raw gyro + transformed gyro in base frame and compare against odom yaw direction.

2) Symptom: `dyaw_odom` has opposite sign to both `dyaw_gyro` and `dyaw_wahba`
   - Likely cause: odom pose interpreted as the inverse transform (parent/child swapped).
   - Verify: check `header.frame_id`, `child_frame_id`, and confirm `p_parent = R p_child + t`.

3) Symptom: gravity alignment dot `xbar @ mu0` is near -1 (instead of +1) when stationary
   - Likely cause: IMU extrinsic inverts gravity or accel units are wrong.
   - Verify: apply `R_base_imu @ accel_imu` on stationary data; Z should be ~+9.81 m/s^2 in base frame.

4) Symptom: `R_hat` appears to be consistently inverted relative to gyro (matches `R_hat.T`)
  - Likely cause: mismatched direction pairing in scan‑to‑map inputs or a frame reflection.
  - Verify: ensure Matrix Fisher inputs use consistent frames for scan/map directions and the rotation is interpreted with the correct parent/child convention.

## References

- **scipy.spatial.transform.Rotation**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
- **ROS geometry_msgs/Quaternion**: http://docs.ros.org/en/api/geometry_msgs/html/msg/Quaternion.html
- **TUM format**: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
- **SE(3) Lie group**: Barfoot (2017), Forster et al. (2017)

## Empirical Validation Artifacts (Dynamic01_ros2)

The following files are produced by deterministic inspection of the rosbag and
serve as audit evidence for the **CONFIRMED (Dynamic01_ros2)** labels above:

- `results/frame_validation_dynamic01.json`
  - Confirms: `/odom` frame_id stability, `/livox/mid360/imu` frame_id, accel units (~1 g),
    odom twist semantics (dp ≈ R v dt), and gyro-vs-odom yaw-rate sign agreement.
- `results/frame_diagnose_dynamic01.txt`
  - Confirms: LiDAR point cloud Z-up convention and that `T_base_lidar` rotation can be identity for this bag;
  also confirms odom covariance ordering is ROS-standard for this dataset.
- `results/confirm_remaining_dynamic01.json`
  - Confirms: IMU yaw-axis sign consistency after applying `R_base_imu` (regression and correlation vs odom yaw-rate).
  - Attempts: LiDAR height/translation inference from raw scans; currently inconclusive for certifying `t_base_lidar.z`.
- `results/turn_invariant_dynamic01.json`
  - Method: left-turn invariant test on a CCW segment selected by `/odom` yaw-rate (`> 0.05 rad/s`).
  - Result: `gyro_base_z_mean` and `accel_base_y_mean` are both **positive** in the selected turn window
    (`t_start=1732437253.3068836`, `t_end=1732437260.0567005`), so the “gyro Z flipped vs accel” hypothesis
    is **not supported** for this bag segment.
