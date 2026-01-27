# Complete Code Differences: Good Run vs Current

**Good Run Commit:** `32da08f91110a5c95b33521cc47f44a0f0a7d227` (2026-01-26 22:06:57)
**Current State:** Working directory (includes uncommitted changes)

## Files with Differences

- **Configuration** (`config/gc_unified.yaml`): 11 +++++++----
 1 file changed, 7 insertions(+), 4 deletions(-)
- **Launch Configuration** (`launch/gc_rosbag.launch.py`): 7 +++++--
 1 file changed, 5 insertions(+), 2 deletions(-)
- **Backend Node** (`fl_slam_poc/backend/backend_node.py`): 55 ++++++++++++++++++----
 1 file changed, 46 insertions(+), 9 deletions(-)
- **Pipeline** (`fl_slam_poc/backend/pipeline.py`): 127 +++++++++++++++++----
 1 file changed, 104 insertions(+), 23 deletions(-)
- **IMU Preintegration** (`fl_slam_poc/backend/operators/imu_preintegration.py`): 14 +++++++++++---
 1 file changed, 11 insertions(+), 3 deletions(-)
- **IMU Gyro Evidence** (`fl_slam_poc/backend/operators/imu_gyro_evidence.py`): 13 ++++++++++++-
 1 file changed, 12 insertions(+), 1 deletion(-)
- **IMU Evidence** (`fl_slam_poc/backend/operators/imu_evidence.py`): 24 ++++++++++++++++++++++
 1 file changed, 24 insertions(+)
- **Odom Evidence** (`fl_slam_poc/backend/operators/odom_evidence.py`): 14 ++++++++++++--
 1 file changed, 12 insertions(+), 2 deletions(-)
- **Diagnostics** (`fl_slam_poc/backend/diagnostics.py`): 2 ++
 1 file changed, 2 insertions(+)
- **Belief** (`fl_slam_poc/common/belief.py`): 20 +++++++++++++++++++-
 1 file changed, 19 insertions(+), 1 deletion(-)
- **Constants** (`fl_slam_poc/common/constants.py`): 27 ++++++++++++++++++++++
 1 file changed, 27 insertions(+)
- **Livox Converter** (`fl_slam_poc/frontend/sensors/livox_converter.py`): 30 ++++++++++------------
 1 file changed, 13 insertions(+), 17 deletions(-)
- **Wiring Auditor** (`fl_slam_poc/frontend/audit/wiring_auditor.py`): 11 +++++------
 1 file changed, 5 insertions(+), 6 deletions(-)
- **Inverse Wishart** (`fl_slam_poc/backend/operators/inverse_wishart_jax.py`): 17 +++++++++++------
 1 file changed, 11 insertions(+), 6 deletions(-)

## Detailed Changes by File

### Configuration (`config/gc_unified.yaml`)


### Launch Configuration (`launch/gc_rosbag.launch.py`)

```diff
+                # CRITICAL: Must match gc_unified.yaml! Wrong rotation causes constant yaw drift.
```

### Backend Node (`fl_slam_poc/backend/backend_node.py`)

```diff
+from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
+        # CRITICAL: Must be large enough to cover scan-to-scan intervals.
-        self.max_imu_buffer = 200
+        self.max_imu_buffer = 4000  # Covers ~20s at 200Hz IMU rate
-            depth=10,
+            depth=100,  # Increased from 10 for better IMU burst handling
+        self.cb_group_lidar = MutuallyExclusiveCallbackGroup()
+        self.cb_group_sensors = ReentrantCallbackGroup()  # IMU + odom can run concurrently
-            PointCloud2, lidar_topic, self.on_lidar, qos_sensor
+            PointCloud2, lidar_topic, self.on_lidar, qos_sensor,
+            callback_group=self.cb_group_lidar
-            Odometry, odom_topic, self.on_odom, qos_reliable
+            Odometry, odom_topic, self.on_odom, qos_reliable,
+            callback_group=self.cb_group_sensors
-            Imu, imu_topic, self.on_imu, qos_sensor
+            Imu, imu_topic, self.on_imu, qos_sensor,
+            callback_group=self.cb_group_sensors
+        # CRITICAL: For Livox MID-360 rosette pattern (non-repetitive, densifies over time):
-        scan_start_time = float(jnp.minimum(stamp_j, jnp.min(timestamps)))
-        scan_end_time = float(jnp.maximum(stamp_j, jnp.max(timestamps)))
```

### Pipeline (`fl_slam_poc/backend/pipeline.py`)

```diff
-    valid_mask = imu_stamps_np > 0.0
-    n_valid = int(np.sum(valid_mask))
+    valid_mask_np = imu_stamps_np > 0.0
+    n_valid = int(np.sum(valid_mask_np))
-        valid = np.sort(imu_stamps_np[valid_mask])
+        valid = np.sort(imu_stamps_np[valid_mask_np])
-    dt_int_j = jnp.maximum(jnp.array(dt_int, dtype=jnp.float64), 1e-12)
-    omega_avg = delta_pose_int[3:6] / dt_int_j
+    # CRITICAL: omega_avg must be an angular-rate proxy (rad/s), not a finite-rotation / dt surrogate.
+    imu_stamps_j = jnp.asarray(imu_stamps, dtype=jnp.float64).reshape(-1)
+    valid_mask = (imu_stamps_j > 0.0).astype(jnp.float64)
+    w_imu_int_valid = w_imu_int * valid_mask
+    w_sum_imu_int = jnp.sum(w_imu_int_valid) + config.eps_mass
+    w_norm_imu_int = w_imu_int_valid / w_sum_imu_int
+    omega_avg = jnp.einsum("m,mi->i", w_norm_imu_int, (imu_gyro - gyro_bias[None, :]))
+    _omega_avg_np = np.array(omega_avg)
+    if not np.all(np.isfinite(_omega_avg_np)):
+        raise ValueError(f"omega_avg contains non-finite values: {_omega_avg_np}")
+    _omega_avg_norm = float(np.linalg.norm(_omega_avg_np))
-        weights=w_imu_int,
```

### IMU Preintegration (`fl_slam_poc/backend/operators/imu_preintegration.py`)

```diff
-    rotvec_end = se3_jax.so3_log(R_end)
-    delta_pose = jnp.concatenate([p_end, rotvec_end], axis=0)
-    return delta_pose, R_end, p_end, ess
+    # CRITICAL FIX: Compute RELATIVE rotation delta, not absolute end orientation.
+    R_start = se3_jax.so3_exp(rotvec_start_WB)
+    delta_R = R_start.T @ R_end  # Relative rotation from start to end
+    rotvec_delta = se3_jax.so3_log(delta_R)
+    delta_pose = jnp.concatenate([p_end, rotvec_delta], axis=0)
+    return delta_pose, delta_R, p_end, ess
```

### IMU Gyro Evidence (`fl_slam_poc/backend/operators/imu_gyro_evidence.py`)

```diff
-    r_rot = se3_jax.so3_log(R_end_pred.T @ R_end_imu)
+    R_diff = R_end_pred.T @ R_end_imu
+    r_rot = se3_jax.so3_log(R_diff)
+    import numpy as _np
+    if not _np.all(_np.isfinite(_np.array(R_diff))):
+        raise ValueError(f"R_diff has NaN: {_np.array(R_diff)}")
+    if not _np.all(_np.isfinite(_np.array(r_rot))):
+        raise ValueError(f"r_rot has NaN from so3_log. R_diff trace={float(jnp.trace(R_diff)):.6f}")
+    if not _np.all(_np.isfinite(_np.array(L_rot))):
+        raise ValueError(f"L_rot has NaN. Sigma_rot diag={_np.diag(_np.array(Sigma_rot))}, dt_eff={float(dt_eff)}")
```

### IMU Evidence (`fl_slam_poc/backend/operators/imu_evidence.py`)

```diff
+=============================================================================
+GRAVITY CONVENTION (CRITICAL)
+=============================================================================
+World frame: Z-UP convention
+  gravity_W = [0, 0, -9.81] m/s²  (gravity points DOWN in -Z direction)
+  g_hat = [0, 0, -1]              (normalized gravity direction, pointing DOWN)
+  minus_g_hat = [0, 0, +1]        (expected accel direction, pointing UP)
+Accelerometer Convention:
+  IMU accelerometers measure REACTION TO GRAVITY (specific force), NOT gravity.
+  When level and stationary, accelerometer reads +Z (pointing UP).
+  This is the force preventing the sensor from freefalling.
+Expected vs Measured:
+  mu0 = R_body^T @ minus_g_hat    (expected accel direction in body frame)
+  xbar = normalized(mean(accel))  (measured accel direction in body frame)
+  Alignment: xbar @ mu0 should be ~+1.0 for correct gravity alignment
+  If negative, the IMU extrinsic is likely inverting gravity!
+State Ordering:
+  L_imu is placed at [3:6, 3:6] which is the ROTATION block in GC ordering.
+  (GC state: [trans(0:3), rot(3:6), ...])
+=============================================================================
```

### Odom Evidence (`fl_slam_poc/backend/operators/odom_evidence.py`)

```diff
-    odom_cov_se3: jnp.ndarray,       # (6,6) covariance in [trans, rotvec] coords
+    odom_cov_se3: jnp.ndarray,       # (6,6) covariance in ROS [x,y,z,roll,pitch,yaw] order
+    NOTE: ROS odom covariance is [trans, rot] and GC tangent now matches; no permutation needed.
+    cov = jnp.asarray(odom_cov_se3, dtype=jnp.float64)
```

### Diagnostics (`fl_slam_poc/backend/diagnostics.py`)

```diff
+            "conditioning_pose6": np.array([s.conditioning_pose6 for s in self.scans]),
+                conditioning_pose6=float(data["conditioning_pose6"][i]) if "conditioning_pose6" in data else 1.0,
```

### Belief (`fl_slam_poc/common/belief.py`)

```diff
-Pose is represented as 6D vector: [translation(3), rotation_vector(3)]
+=============================================================================
+ORDERING CONVENTIONS (UNIFIED - all use [trans, rot] ordering!)
+=============================================================================
+1. SE(3) POSE (se3_jax format):
+   6D vector: [trans(3), rot(3)] = [x, y, z, rx, ry, rz]
+   Used by: se3_compose, se3_inverse, X_anchor, mean_world_pose()
+2. GC STATE VECTOR (tangent space):
+   22D vector: [trans(3), rot(3), vel(3), bg(3), ba(3), dt(1), ex(6)]
+   Pose slice: [0:3] = translation, [3:6] = rotation
+   Used by: L, h, z_lin, all evidence operators
+CONVERSION FUNCTIONS (identity - orderings now match):
+   pose_se3_to_z_delta(): returns input unchanged
+   pose_z_to_se3_delta(): returns input unchanged
+Reference: docs/FRAME_AND_QUATERNION_CONVENTIONS.md
+=============================================================================
```

### Constants (`fl_slam_poc/common/constants.py`)

```diff
+=============================================================================
+CONVENTION QUICK REFERENCE (see docs/FRAME_AND_QUATERNION_CONVENTIONS.md)
+=============================================================================
+STATE VECTOR (22D):
+  [trans(0:3), rot(3:6), vel(6:9), bg(9:12), ba(12:15), dt(15:16), ex(16:22)]
+  Note: GC state uses [trans, rot] ordering (same as se3_jax and ROS)
+SE(3) POSES:
+  Internal 6D: [trans(3), rotvec(3)] = [x, y, z, rx, ry, rz]
+  No conversion needed - orderings now match
+GRAVITY:
+  World: Z-UP convention, gravity points DOWN = [0, 0, -9.81] m/s²
+  Accelerometer measures reaction to gravity (pointing UP when level)
+  Expected accel direction (mu0) = R_body^T @ [0, 0, +1]
+IMU UNITS:
+  Livox raw: acceleration in g's (1g ≈ 9.81 m/s²)
+  Internal: all accelerations in m/s² (scaled by GC_IMU_ACCEL_SCALE)
+  Gyro: angular velocity in rad/s
+EXTRINSICS:
+  T_base_sensor = [tx, ty, tz, rx, ry, rz] where rotation is rotvec (radians)
+  Transform: p_base = R_base_sensor @ p_sensor + t_base_sensor
```

### Livox Converter (`fl_slam_poc/frontend/sensors/livox_converter.py`)

```diff
+        # CRITICAL: livox_ros_driver2 CustomPoint does NOT have offset_time field.
-            n_points = len(points_valid)
-            if n_points > 0:
-                scan_duration_ns = 100_000_000  # 100ms in nanoseconds
-                point_indices = np.arange(n_points, dtype=np.uint32)
-                time_offset = (point_indices * scan_duration_ns // n_points).astype(np.uint32)
-            else:
-                time_offset = np.array([], dtype=np.uint32)
+            time_offset = np.zeros(len(points_valid), dtype=np.uint32)
-                    "Livox converter: per-point time offset computed synthetically from point index "
-                    "(uniform spacing assumption for livox_ros_driver2)."
+                    "Livox converter: per-point time offset unavailable (livox_ros_driver2). "
+                    "All offsets set to 0 (deskewing disabled for non-repetitive rosette pattern)."
```

### Wiring Auditor (`fl_slam_poc/frontend/audit/wiring_auditor.py`)

```diff
-            f"║    Odom msgs:    {s.odom_count:>6}  [buffered, NOT FUSED into belief]         ║",
-            f"║    IMU msgs:     {s.imu_count:>6}  [buffered, NOT FUSED into belief]         ║",
+            f"║    Odom msgs:    {s.odom_count:>6}  [FUSED via odom_evidence]                 ║",
+            f"║    IMU msgs:     {s.imu_count:>6}  [FUSED via imu_evidence + gyro_evidence] ║",
-            lines.append("║    ⚠ Odom/IMU subscribed but NOT YET FUSED into belief             ║")
-            lines.append("║      (see docs/Fusion_issues.md for status)                        ║")
+            lines.append("║    ✓ Odom/IMU fused into belief via evidence operators              ║")
-                "odom_fused": False,  # Currently not fused
-                "imu_fused": False,   # Currently not fused
+                "odom_fused": True,   # Fused via odom_evidence operator
+                "imu_fused": True,    # Fused via imu_evidence + imu_gyro_evidence operators
```

### Inverse Wishart (`fl_slam_poc/backend/operators/inverse_wishart_jax.py`)

```diff
-    dt_sec: float,
+    dt_sec: float,  # Unused - kept for API compatibility, will be removed
-    Forgetful retention is applied deterministically:
-      Psi <- rho * Psi + dPsi / dt
+    DISCRETE-TIME update (no dt division):
+      Psi <- rho * Psi + dPsi
+    Rationale: dPsi is already "incremental sufficient statistics for this step" computed
+    from the realized state increment. Dividing by dt would incorrectly interpret it as a
+    continuous-time rate, but we don't have a continuous-time innovation model for dPsi.
+    The correct place for dt-scaling is in PredictDiffusion (Sigma += Q_mode * dt_sec),
+    not in the IW posterior update.
-    dt_safe = jnp.maximum(jnp.array(dt_sec, dtype=jnp.float64), 1e-12)
-    Psi_raw = (rho[:, None, None] * pn_state.Psi_blocks) + (dPsi / dt_safe)
+    Psi_raw = (rho[:, None, None] * pn_state.Psi_blocks) + dPsi
```

## Most Critical Changes (Performance Impact Analysis)

### 1. IMU Preintegration (CRITICAL - HIGH IMPACT)
- **File:** `fl_slam_poc/backend/operators/imu_preintegration.py`
- **Good Run:** Returned absolute `rotvec_end = Log(R_end)`
- **Current:** Returns relative `rotvec_delta = Log(R_start^T @ R_end)`
- **Impact:** HIGH - Fundamentally changes how rotation is interpreted by `imu_gyro_rotation_evidence`
- **Note:** Current implementation is mathematically correct for `imu_gyro_rotation_evidence`, but good run used absolute (bug that happened to work?)

### 2. IMU Buffer Size (CRITICAL - HIGH IMPACT)
- **File:** `fl_slam_poc/backend/backend_node.py`
- **Good Run:** `max_imu_buffer = 200` (covers ~1s at 200Hz)
- **Current:** `max_imu_buffer = 4000` (covers ~20s at 200Hz)
- **Impact:** HIGH - Good run may have been missing IMU data between scans, accidentally reducing IMU influence

### 3. Omega Average Computation (MEDIUM IMPACT)
- **File:** `fl_slam_poc/backend/pipeline.py`
- **Good Run:** `omega_avg = delta_rotvec / dt_int`
- **Current:** `omega_avg = weighted_mean(gyro - bias)` (from raw gyro measurements)
- **Impact:** MEDIUM - Affects IW noise adaptation for gyro

### 4. Executor Type (MEDIUM IMPACT)
- **File:** `fl_slam_poc/backend/backend_node.py`
- **Good Run:** Single-threaded `rclpy.spin()`
- **Current:** `MultiThreadedExecutor` (4 threads)
- **Impact:** MEDIUM - Could affect message ordering/timing

### 5. Odom Covariance Ordering (MEDIUM IMPACT)
- **File:** `fl_slam_poc/backend/operators/odom_evidence.py`
- **Good Run:** Direct use of `odom_cov_se3` (assumed correct ordering)
- **Current:** Permutation to reorder from ROS `[x,y,z,roll,pitch,yaw]` to GC `[rot,trans]`
- **Impact:** MEDIUM - Could cause axis misalignment if good run had wrong ordering

### 6. Inverse Wishart Update (MEDIUM IMPACT)
- **File:** `fl_slam_poc/backend/operators/inverse_wishart_jax.py`
- **Good Run:** `Psi <- rho * Psi + dPsi / dt`
- **Current:** `Psi <- rho * Psi + dPsi` (no dt division)
- **Impact:** MEDIUM - Affects noise adaptation rate

## Full Diff File

For complete line-by-line differences, see: `docs/CODE_DIFF_GOOD_RUN_vs_CURRENT.md`

To view the full diff:
```bash
git diff 32da08f91110a5c95b33521cc47f44a0f0a7d227 -- fl_ws/src/fl_slam_poc/
```
