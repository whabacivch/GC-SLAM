=== fl_ws/src/fl_slam_poc/config/gc_unified.yaml ===
diff --git a/fl_ws/src/fl_slam_poc/config/gc_unified.yaml b/fl_ws/src/fl_slam_poc/config/gc_unified.yaml
index 9b3ff7a..17b55d9 100644
--- a/fl_ws/src/fl_slam_poc/config/gc_unified.yaml
+++ b/fl_ws/src/fl_slam_poc/config/gc_unified.yaml
@@ -71,11 +71,14 @@ gc_backend:
     base_frame: base_footprint
 
     # No-TF extrinsics (T_{base<-sensor}) in [x, y, z, rx, ry, rz] (rotvec, radians).
-    # Estimated from Dynamic01 bag gravity alignment (tools/estimate_imu_base_extrinsic_rotation.py):
-    #   IMU accel direction in livox_frame: [-0.47, -0.015, 0.88]
-    #   Expected in base_footprint (Z-up):   [0, 0, 1]
-    #   Rotation: ~28° misalignment → rotvec [-0.015586, 0.489293, 0.0] rad
+    # LiDAR: Z-up confirmed (diagnose_coordinate_frames.py) - rotation [0,0,0] is correct.
+    # IMU: 154.5° rotation from gravity alignment tool. UNDER INVESTIGATION - there may be
+    #      sign convention issues elsewhere in the pipeline that need to be resolved first.
+    # See: tools/estimate_imu_base_extrinsic_rotation.py
     T_base_lidar: [-0.011, 0.0, 0.778, 0.0, 0.0, 0.0]
+    # IMU extrinsic: 28° rotation to align IMU gravity with base +Z
+    # (Previous 154.5° rotation was inverting both gravity and yaw axis)
+    # Restored to match good run (2026-01-26 22:06:57) values
     T_base_imu:   [0.0,   0.0, 0.0,   -0.015586, 0.489293, 0.0]
 
     # Trajectory export

=== fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py ===
diff --git a/fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py b/fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py
index 826b650..aca66bb 100644
--- a/fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py
+++ b/fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py
@@ -131,9 +131,12 @@ def generate_launch_description():
                 # Bag truth for M3DGR Dynamic01: odom child_frame_id is base_footprint.
                 "base_frame": "base_footprint",
                 # No-TF extrinsics (T_{base<-sensor}) in [x,y,z,rx,ry,rz] rotvec (rad).
-                # IMU rotation estimated from Dynamic01 bag gravity alignment (~28° misalignment).
-                # See: tools/estimate_imu_base_extrinsic_rotation.py
+                # LiDAR: Z-up confirmed (diagnose_coordinate_frames.py) - rotation [0,0,0] is correct.
+                # IMU: 154.5° rotation from gravity alignment. UNDER INVESTIGATION.
+                # CRITICAL: Must match gc_unified.yaml! Wrong rotation causes constant yaw drift.
                 "T_base_lidar": [-0.011, 0.0, 0.778, 0.0, 0.0, 0.0],
+                # IMU extrinsic: 28° rotation to align IMU gravity with base +Z
+                # Restored to match good run (2026-01-26 22:06:57) values
                 "T_base_imu": [0.0, 0.0, 0.0, -0.015586, 0.489293, 0.0],
                 "status_check_period_sec": 5.0,
                 "forgetting_factor": 0.99,

=== fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_preintegration.py ===
diff --git a/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_preintegration.py b/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_preintegration.py
index f14b998..24449ec 100644
--- a/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_preintegration.py
+++ b/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/operators/imu_preintegration.py
@@ -99,7 +99,15 @@ def preintegrate_imu_relative_pose_jax(
     carry0 = (R, jnp.zeros((3,), dtype=jnp.float64), jnp.zeros((3,), dtype=jnp.float64))
     (R_end, v_end, p_end), _ = jax.lax.scan(step, carry0, (imu_gyro, imu_accel, dt, w))
 
-    rotvec_end = se3_jax.so3_log(R_end)
-    delta_pose = jnp.concatenate([p_end, rotvec_end], axis=0)
-    return delta_pose, R_end, p_end, ess
+    # CRITICAL FIX: Compute RELATIVE rotation delta, not absolute end orientation.
+    # R_end = R_start @ dR_0 @ dR_1 @ ... (absolute)
+    # delta_R = R_start^T @ R_end = dR_0 @ dR_1 @ ... (relative)
+    # Before: returned rotvec_end = Log(R_end) which is absolute orientation.
+    # This caused imu_gyro_rotation_evidence to compute R_end_imu = R_start @ Exp(rotvec_end)
+    # = R_start @ R_end (WRONG!) instead of just R_end.
+    R_start = se3_jax.so3_exp(rotvec_start_WB)
+    delta_R = R_start.T @ R_end  # Relative rotation from start to end
+    rotvec_delta = se3_jax.so3_log(delta_R)
+    delta_pose = jnp.concatenate([p_end, rotvec_delta], axis=0)
+    return delta_pose, delta_R, p_end, ess
 

=== fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py ===
diff --git a/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py b/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py
index 01abf32..d29ff9b 100644
--- a/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py
+++ b/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py
@@ -17,6 +17,7 @@ import rclpy
 from rclpy.clock import Clock, ClockType
 from rclpy.node import Node
 from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
+from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
 
 import tf2_ros
 from geometry_msgs.msg import PoseStamped, TransformStamped
@@ -335,8 +336,11 @@ class GoldenChildBackend(Node):
         self.first_odom_pose = None  # (6,) SE3 pose [trans, rotvec]
         
         # IMU buffer for high-rate prediction
+        # CRITICAL: Must be large enough to cover scan-to-scan intervals.
+        # At 200Hz IMU and up to 20s scan intervals, need ~4000 samples.
+        # Previous value of 200 only covered 1s, causing dt_int ≈ 0 for most scans!
         self.imu_buffer: List[Tuple[float, jnp.ndarray, jnp.ndarray]] = []
-        self.max_imu_buffer = 200
+        self.max_imu_buffer = 4000  # Covers ~20s at 200Hz IMU rate
         
         # Tracking
         self.imu_count = 0
@@ -363,7 +367,7 @@ class GoldenChildBackend(Node):
         qos_sensor = QoSProfile(
             reliability=ReliabilityPolicy.BEST_EFFORT,
             history=HistoryPolicy.KEEP_LAST,
-            depth=10,
+            depth=100,  # Increased from 10 for better IMU burst handling
             durability=DurabilityPolicy.VOLATILE,
         )
         
@@ -378,14 +382,23 @@ class GoldenChildBackend(Node):
         odom_topic = str(self.get_parameter("odom_topic").value)
         imu_topic = str(self.get_parameter("imu_topic").value)
         
+        # Separate callback groups so IMU/odom can be buffered while lidar pipeline runs.
+        # Without this, single-threaded or MutuallyExclusive groups would block IMU during
+        # the ~1-2s pipeline processing, causing 200Hz IMU messages to overflow the queue.
+        self.cb_group_lidar = MutuallyExclusiveCallbackGroup()
+        self.cb_group_sensors = ReentrantCallbackGroup()  # IMU + odom can run concurrently
+        
         self.sub_lidar = self.create_subscription(
-            PointCloud2, lidar_topic, self.on_lidar, qos_sensor
+            PointCloud2, lidar_topic, self.on_lidar, qos_sensor,
+            callback_group=self.cb_group_lidar
         )
         self.sub_odom = self.create_subscription(
-            Odometry, odom_topic, self.on_odom, qos_reliable
+            Odometry, odom_topic, self.on_odom, qos_reliable,
+            callback_group=self.cb_group_sensors
         )
         self.sub_imu = self.create_subscription(
-            Imu, imu_topic, self.on_imu, qos_sensor
+            Imu, imu_topic, self.on_imu, qos_sensor,
+            callback_group=self.cb_group_sensors
         )
         
         self.get_logger().info(f"LiDAR: {lidar_topic} (PIPELINE ACTIVE)")
@@ -585,10 +598,27 @@ class GoldenChildBackend(Node):
         # This must be captured BEFORE updating self.last_scan_stamp.
         t_prev_scan = float(self.last_scan_stamp) if self.last_scan_stamp > 0.0 else 0.0
         
-        # Derive scan bounds from per-point timestamps for deskew (within-scan only)
+        # Derive scan bounds for deskew (within-scan only).
+        # CRITICAL: For Livox MID-360 rosette pattern (non-repetitive, densifies over time):
+        # - timebase_sec = start of accumulation window
+        # - header.stamp = end of accumulation window (when message published)
+        # - All time_offset = 0 (no per-point timestamps available)
+        # So we MUST use timebase_sec as scan_start_time and header.stamp as scan_end_time
+        # to capture the accumulation window, even though we can't deskew individual points.
         stamp_j = jnp.array(stamp_sec, dtype=jnp.float64)
-        scan_start_time = float(jnp.minimum(stamp_j, jnp.min(timestamps)))
-        scan_end_time = float(jnp.maximum(stamp_j, jnp.max(timestamps)))
+        timestamps_min = float(jnp.min(timestamps))
+        timestamps_max = float(jnp.max(timestamps))
+        
+        # If all timestamps are the same (rosette pattern with time_offset=0),

