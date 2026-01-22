"""
Sensor I/O Layer - NO MATH.

Handles:
- Sensor subscriptions and buffering
- TF lookups
- Point cloud conversions (depth→3D, scan→3D)
- Timestamp-based data retrieval

All geometric transforms use geometry.se3 (exact operations).
"""

from typing import Optional, Tuple
import time
import numpy as np
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan, PointCloud2, PointField
from nav_msgs.msg import Odometry
from tf2_ros import TransformException
try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

from fl_slam_poc.common import constants
from fl_slam_poc.common.se3 import quat_to_rotmat, se3_compose, rotmat_to_rotvec, rotvec_to_rotmat


def pointcloud2_to_array(msg: PointCloud2) -> np.ndarray:
    """
    Convert PointCloud2 message to numpy array of XYZ points.
    
    Handles common point cloud formats (XYZ, XYZRGB, etc.)
    Returns array of shape (N, 3) containing [x, y, z] coordinates.
    """
    # Get field offsets
    field_names = [f.name for f in msg.fields]
    field_offsets = {f.name: f.offset for f in msg.fields}
    
    # Check for XYZ fields
    if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
        raise ValueError("PointCloud2 message missing x, y, or z fields")
    
    # Get data type info
    dtype_map = {
        PointField.FLOAT32: np.float32,
        PointField.FLOAT64: np.float64,
        PointField.INT32: np.int32,
        PointField.UINT32: np.uint32,
    }
    
    # Assume float32 for XYZ (most common)
    x_offset = field_offsets['x']
    y_offset = field_offsets['y']
    z_offset = field_offsets['z']
    
    point_step = msg.point_step
    n_points = msg.width * msg.height
    
    if n_points == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    # Parse raw data
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)
    
    x = data[:, x_offset:x_offset+4].view(np.float32).flatten()
    y = data[:, y_offset:y_offset+4].view(np.float32).flatten()
    z = data[:, z_offset:z_offset+4].view(np.float32).flatten()
    
    points = np.stack([x, y, z], axis=1)
    
    # Filter invalid points (NaN, Inf)
    valid = np.isfinite(points).all(axis=1)
    return points[valid].astype(np.float32)


def stamp_to_sec(stamp) -> float:
    """Convert ROS timestamp to seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SensorIO:
    """
    Sensor I/O manager - buffering, TF, and point cloud conversion.
    
    Pure I/O layer - NO mathematical inference.
    """
    
    def __init__(self, node: Node, config: dict):
        """
        Args:
            node: ROS node for subscriptions and logging
            config: Dict with keys: scan_topic, odom_topic, camera_topic, depth_topic,
                    camera_info_topic, enable_image, enable_depth, enable_camera_info,
                    odom_is_delta, odom_frame, base_frame, camera_frame, scan_frame,
                    tf_timeout_sec, feature_buffer_len, depth_stride
        """
        self.node = node
        self.config = config
        
        # Buffers (timestamp, data)
        self.odom_buffer = []
        self.image_buffer = []  # (timestamp, rgb_array, frame_id)
        self.depth_buffer = []  # (timestamp, depth_array, points, frame_id)
        self.pointcloud_buffer = []  # (timestamp, points_array, frame_id)
        self.imu_buffer = []  # (timestamp, accel_xyz, gyro_xyz) - event-driven clearing
        self.last_imu_frame_id = None
        
        # State
        self.last_pose = None
        self.depth_intrinsics = None
        self._last_msg_keys = {}
        
        # 3D point cloud mode
        self.use_3d_pointcloud = config.get("use_3d_pointcloud", False)
        self.enable_pointcloud = config.get("enable_pointcloud", False)
        self._last_pointcloud_time = 0.0
        self._pointcloud_rate_limit = 1.0 / config.get("pointcloud_rate_limit_hz", 30.0) if config.get("pointcloud_rate_limit_hz", 30.0) > 0 else 0.0
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        # Rosbag playback frequently publishes static transforms on /tf_static.
        # Use TRANSIENT_LOCAL for static TF so late-joining subscribers receive it.
        tf_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        tf_static_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            node,
            qos=tf_qos,
            static_qos=tf_static_qos,
        )
        
        # CV Bridge (handle NumPy 2.x incompatibility gracefully)
        if CvBridge is not None:
            try:
                self.cv_bridge = CvBridge()
            except (ImportError, AttributeError, RuntimeError) as e:
                self.node.get_logger().warn(
                    f"cv_bridge initialization failed (NumPy 2.x compatibility issue): {e}. "
                    "Image processing will be disabled."
                )
                self.cv_bridge = None
        else:
            self.cv_bridge = None
        
        # Subscribe to sensors
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        """Create ROS subscriptions for all enabled sensors."""
        qos_profiles, qos_names = self._resolve_qos_profiles()
        for qos in qos_profiles:
            # Always subscribe to odometry
            self.node.create_subscription(
                Odometry, self.config["odom_topic"], self._on_odom, qos)
            
            # Subscribe to LaserScan (2D mode) or PointCloud2 (3D mode)
            if self.use_3d_pointcloud:
                # 3D mode: Subscribe to PointCloud2 instead of LaserScan
                pointcloud_topic = self.config.get("pointcloud_topic", constants.POINTCLOUD_TOPIC_DEFAULT)
                self.node.create_subscription(
                    PointCloud2, pointcloud_topic, self._on_pointcloud, qos)
                self.node.get_logger().info(
                    f"SensorIO: 3D point cloud mode enabled, subscribing to {pointcloud_topic}"
                )
            else:
                # 2D mode: Subscribe to LaserScan
                self.node.create_subscription(
                    LaserScan, self.config["scan_topic"], self._on_scan_internal, qos)
            
            # Optionally subscribe to both PointCloud2 AND LaserScan
            if self.enable_pointcloud and not self.use_3d_pointcloud:
                pointcloud_topic = self.config.get("pointcloud_topic", constants.POINTCLOUD_TOPIC_DEFAULT)
                self.node.create_subscription(
                    PointCloud2, pointcloud_topic, self._on_pointcloud, qos)
            
            if self.config.get("enable_image", False):
                self.node.create_subscription(
                    Image, self.config["camera_topic"], self._on_image, qos)
            
            # Depth images are optional and may be used for RGB-D evidence/appearance
            # even when the primary sparse geometry source is PointCloud2.
            if self.config.get("enable_depth", False):
                self.node.create_subscription(
                    Image, self.config["depth_topic"], self._on_depth, qos)
            
            if self.config.get("enable_camera_info", False):
                self.node.create_subscription(
                    CameraInfo, self.config["camera_info_topic"], self._on_camera_info, qos)

            # IMU subscription (high-rate sensor for preintegration)
            if self.config.get("enable_imu", False):
                imu_topic = self.config.get("imu_topic", constants.IMU_TOPIC_DEFAULT)
                self.node.create_subscription(Imu, imu_topic, self._on_imu, qos)
                self.node.get_logger().info(f"SensorIO: IMU subscription enabled on {imu_topic}")

        mode_str = "3D PointCloud" if self.use_3d_pointcloud else "2D LaserScan"
        self.node.get_logger().info(
            f"SensorIO ({mode_str} mode) subscribed to {self.config['odom_topic']} "
            f"with QoS reliability: {', '.join(qos_names)}"
        )

    def _resolve_qos_profiles(self):
        """
        Resolve sensor QoS profiles from config.

        Supported values:
          - reliable
          - best_effort
          - system_default
          - both (subscribe twice: RELIABLE + BEST_EFFORT)
        """
        reliability = str(self.config.get("sensor_qos_reliability", "reliable")).lower()
        rel_map = {
            "reliable": ReliabilityPolicy.RELIABLE,
            "best_effort": ReliabilityPolicy.BEST_EFFORT,
            "system_default": ReliabilityPolicy.SYSTEM_DEFAULT,
        }
        if reliability == "both":
            rels = [ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT]
            names = ["reliable", "best_effort"]
        elif reliability in rel_map:
            rels = [rel_map[reliability]]
            names = [reliability]
        else:
            rels = [ReliabilityPolicy.RELIABLE]
            names = ["reliable"]
        
        profiles = [
            QoSProfile(
                reliability=rel,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            for rel in rels
        ]
        
        return profiles, names

    def _is_duplicate(self, key: str, stamp, frame_id: str) -> bool:
        """Prevent double-processing when subscribing with multiple QoS profiles."""
        if stamp is None:
            return False
        stamp_key = (stamp.sec, stamp.nanosec, frame_id or "")
        if self._last_msg_keys.get(key) == stamp_key:
            return True
        self._last_msg_keys[key] = stamp_key
        return False
    
    def set_scan_callback(self, callback):
        """Set external callback for scan processing."""
        self._scan_callback = callback

    def set_odom_callback(self, callback):
        """Set external callback for odom processing."""
        self._odom_callback = callback

    def set_image_callback(self, callback):
        """Set external callback for image processing."""
        self._image_callback = callback

    def set_depth_callback(self, callback):
        """Set external callback for depth processing."""
        self._depth_callback = callback

    def set_pointcloud_callback(self, callback):
        """Set external callback for point cloud processing."""
        self._pointcloud_callback = callback

    def set_imu_callback(self, callback):
        """Set external callback for IMU processing."""
        self._imu_callback = callback
    
    def _on_scan_internal(self, msg: LaserScan):
        """Internal scan handler - calls external callback if set."""
        if self._is_duplicate("scan", msg.header.stamp, msg.header.frame_id):
            return
        # Debug: Log first scan received
        if not hasattr(self, '_first_scan_logged'):
            self._first_scan_logged = True
            self.node.get_logger().info(
                f"SensorIO: First scan received, frame_id={msg.header.frame_id}, "
                f"ranges={len(msg.ranges)}, last_pose={'SET' if self.last_pose is not None else 'NONE'}"
            )
        
        if hasattr(self, '_scan_callback'):
            self._scan_callback(msg)
    
    def _on_pointcloud(self, msg: PointCloud2):
        """Buffer 3D point cloud data."""
        if self._is_duplicate("pointcloud", msg.header.stamp, msg.header.frame_id):
            return
        
        # Rate limiting for high-frequency point clouds
        current_time = time.time()
        if self._pointcloud_rate_limit > 0 and (current_time - self._last_pointcloud_time) < self._pointcloud_rate_limit:
            return
        self._last_pointcloud_time = current_time
        
        try:
            # Convert PointCloud2 to numpy array
            points = pointcloud2_to_array(msg)
            
            if points.shape[0] == 0:
                return
            
            # Debug: Log first point cloud received
            if not hasattr(self, '_first_pointcloud_logged'):
                self._first_pointcloud_logged = True
                self.node.get_logger().info(
                    f"SensorIO: First point cloud received, frame_id={msg.header.frame_id}, "
                    f"points={points.shape[0]}, last_pose={'SET' if self.last_pose is not None else 'NONE'}"
                )
            
            stamp = stamp_to_sec(msg.header.stamp)
            frame_id = msg.header.frame_id or self.config.get("camera_frame", "camera_link")
            
            # Transform to base_frame if needed
            base_frame = self.config.get("base_frame", "base_link")
            if frame_id != base_frame:
                transform = self._lookup_transform(base_frame, frame_id, msg.header.stamp)
                if transform is not None:
                    points = self._transform_points(points, transform)
                    frame_id = base_frame
                else:
                    if not hasattr(self, '_logged_pc_tf_fail'):
                        self._logged_pc_tf_fail = True
                        self.node.get_logger().error(
                            f"CRITICAL: Cannot transform pointcloud from '{frame_id}' to '{base_frame}'. "
                            f"TF lookup failed! Points will be in wrong frame, anchors/loops may fail."
                        )
            else:
                if not hasattr(self, '_logged_pc_no_tf'):
                    self._logged_pc_no_tf = True
                    self.node.get_logger().info(
                        f"PointCloud frame '{frame_id}' matches base_frame, no TF transform needed"
                    )
            
            # Buffer management
            buffer_len = self.config.get("feature_buffer_len", 10)
            self.pointcloud_buffer.append((stamp, points, frame_id))
            if len(self.pointcloud_buffer) > buffer_len:
                self.pointcloud_buffer.pop(0)
            
            # Call external callback if set
            if hasattr(self, '_pointcloud_callback'):
                self._pointcloud_callback(msg, points)
                
        except Exception as e:
            self.node.get_logger().warn(
                f"PointCloud2 conversion failed: {e}", throttle_duration_sec=5.0)
    
    def _on_odom(self, msg: Odometry):
        """Buffer odometry (pose only)."""
        if self._is_duplicate("odom", msg.header.stamp, msg.header.frame_id):
            return
        stamp = stamp_to_sec(msg.header.stamp)
        
        # Debug logging for first odom message
        if self.last_pose is None:
            self.node.get_logger().info(
                f"SensorIO: First odom received at stamp {stamp:.3f}, frame_id={msg.header.frame_id}"
            )
        
        if self.config.get("odom_is_delta", False):
            # Accumulate deltas using proper SE(3) composition
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            
            # CRITICAL FIX: Convert quaternion to rotation vector (not just qx,qy,qz!)
            # The odometry message contains a quaternion, not a rotation vector
            R_delta = quat_to_rotmat(ori.x, ori.y, ori.z, ori.w)
            rotvec_delta = rotmat_to_rotvec(R_delta)
            delta = np.array([pos.x, pos.y, pos.z, rotvec_delta[0], rotvec_delta[1], rotvec_delta[2]], dtype=float)
            
            if self.last_pose is None:
                self.last_pose = np.zeros(6, dtype=float)
            
            # CRITICAL FIX: Use proper SE(3) composition instead of addition
            # pose_new = pose_old ∘ delta (compose in the body frame)
            self.last_pose = se3_compose(self.last_pose, delta)
            pose = self.last_pose.copy()
        else:
            # Absolute pose - convert quaternion to rotation vector
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            
            # Use geometry.se3 for exact conversion
            R = quat_to_rotmat(ori.x, ori.y, ori.z, ori.w)
            rotvec = rotmat_to_rotvec(R)
            
            pose = np.array([pos.x, pos.y, pos.z, rotvec[0], rotvec[1], rotvec[2]], dtype=float)
            self.last_pose = pose.copy()
        
        # Buffer management
        buffer_len = self.config.get("feature_buffer_len", 10)
        self.odom_buffer.append((stamp, pose))
        if len(self.odom_buffer) > buffer_len:
            self.odom_buffer.pop(0)

        if hasattr(self, "_odom_callback"):
            self._odom_callback(msg)
    
    def _on_image(self, msg: Image):
        """Buffer RGB image array for RGB-D evidence extraction."""
        if self._is_duplicate("image", msg.header.stamp, msg.header.frame_id):
            return
        if self.cv_bridge is None:
            return
        
        try:
            # Convert to numpy array (RGB8 format)
            rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            rgb = np.asarray(rgb, dtype=np.uint8)
        except Exception as e:
            self.node.get_logger().warn(f"RGB conversion failed: {e}", throttle_duration_sec=5.0)
            return
        
        stamp = stamp_to_sec(msg.header.stamp)
        frame_id = msg.header.frame_id or self.config.get("camera_frame", "camera_link")
        self.image_buffer.append((stamp, rgb, frame_id))
        
        buffer_len = self.config.get("feature_buffer_len", 10)
        if len(self.image_buffer) > buffer_len:
            self.image_buffer.pop(0)

        if hasattr(self, "_image_callback"):
            self._image_callback(msg)
    
    def _on_depth(self, msg: Image):
        """Buffer depth array (and optionally 3D points) for RGB-D evidence extraction."""
        if self._is_duplicate("depth", msg.header.stamp, msg.header.frame_id):
            return
        if self.cv_bridge is None:
            return
        
        try:
            depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth = np.asarray(depth, dtype=np.float32)
        except Exception as e:
            self.node.get_logger().warn(f"Depth conversion failed: {e}", throttle_duration_sec=5.0)
            return
        
        # Optionally compute points for legacy descriptor/ICP use (2D scan mode only).
        # In 3D PointCloud mode, depth points are not required and often cannot be TF-transformed
        # during rosbag playback (missing TF), so we avoid triggering TF warnings.
        points = None
        if not self.use_3d_pointcloud and self.depth_intrinsics is not None:
            points = self._depth_to_points(depth, msg.header)
        
        stamp = stamp_to_sec(msg.header.stamp)
        frame_id = msg.header.frame_id or self.config.get("camera_frame", "camera_link")
        # Store depth array AND points (depth needed for normals, points for legacy)
        self.depth_buffer.append((stamp, depth, points, frame_id))
        
        buffer_len = self.config.get("feature_buffer_len", 10)
        if len(self.depth_buffer) > buffer_len:
            self.depth_buffer.pop(0)

        if hasattr(self, "_depth_callback"):
            self._depth_callback(msg)
    
    def _on_camera_info(self, msg: CameraInfo):
        """Extract camera intrinsics."""
        if self._is_duplicate("camera_info", msg.header.stamp, msg.header.frame_id):
            return
        self.depth_intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy
        
        # Log first camera info received
        if not hasattr(self, '_first_camera_info_logged'):
            self._first_camera_info_logged = True
            fx, fy, cx, cy = self.depth_intrinsics
            self.node.get_logger().info(
                f"SensorIO: Camera intrinsics received from CameraInfo: "
                f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"
            )

    def _on_imu(self, msg: Imu):
        """
        Buffer IMU measurements for preintegration.

        Uses event-driven clearing: measurements accumulate until explicitly
        cleared after preintegration (no automatic sliding window).
        """
        if self._is_duplicate("imu", msg.header.stamp, msg.header.frame_id):
            return

        stamp = stamp_to_sec(msg.header.stamp)
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=float)
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=float)

        self.last_imu_frame_id = msg.header.frame_id
        self.imu_buffer.append((stamp, accel, gyro))

        # Debug: Log first IMU received
        if not hasattr(self, '_first_imu_logged'):
            self._first_imu_logged = True
            self.node.get_logger().info(
                f"SensorIO: First IMU received, frame_id={msg.header.frame_id}, "
                f"accel=({accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}), "
                f"gyro=({gyro[0]:.4f}, {gyro[1]:.4f}, {gyro[2]:.4f})"
            )

        # Call external callback if set
        if hasattr(self, '_imu_callback'):
            self._imu_callback(msg)

    def _depth_to_points(self, depth: np.ndarray, header) -> Optional[np.ndarray]:
        """Convert depth image to 3D points in base_frame."""
        if self.depth_intrinsics is None:
            return None
        
        fx, fy, cx, cy = self.depth_intrinsics
        stride = self.config.get("depth_stride", 4)
        h, w = depth.shape
        
        ys, xs = np.arange(0, h, stride, dtype=float), np.arange(0, w, stride, dtype=float)
        grid_x, grid_y = np.meshgrid(xs, ys)
        zs = depth[::stride, ::stride]
        
        valid = np.isfinite(zs) & (zs > 0.0)
        if not np.any(valid):
            return None
        
        z = zs[valid]
        x = (grid_x[valid] - cx) * z / fx
        y = (grid_y[valid] - cy) * z / fy
        points_cam = np.stack([x, y, z], axis=1)
        
        # Transform to base_frame
        frame_id = header.frame_id or self.config.get("camera_frame", "camera_link")
        transform = self._lookup_transform(self.config["base_frame"], frame_id, header.stamp)
        if transform is None:
            return None
        
        return self._transform_points(points_cam, transform)
    
    def scan_to_points(self, msg: LaserScan) -> Optional[np.ndarray]:
        """Convert LaserScan to 3D points in base_frame."""
        ranges = np.asarray(msg.ranges, dtype=float).reshape(-1)
        if ranges.size == 0:
            return None
        
        angles = msg.angle_min + np.arange(ranges.size, dtype=float) * msg.angle_increment
        valid = np.isfinite(ranges)
        valid &= (ranges >= float(msg.range_min))
        valid &= (ranges <= float(msg.range_max))
        
        if not np.any(valid):
            return None
        
        r = ranges[valid]
        a = angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)
        z = np.zeros_like(x)
        points_scan = np.stack([x, y, z], axis=1)
        
        # Transform to base_frame
        frame_id = msg.header.frame_id or self.config.get("scan_frame", "base_link")
        base_frame = self.config["base_frame"]
        
        # If frames are the same, no transform needed
        if frame_id == base_frame:
            if not hasattr(self, '_logged_scan_no_tf'):
                self._logged_scan_no_tf = True
                self.node.get_logger().info(
                    f"Scan frame '{frame_id}' matches base_frame, no TF transform needed"
                )
            return points_scan
        
        transform = self._lookup_transform(base_frame, frame_id, msg.header.stamp)
        if transform is None:
            if not hasattr(self, '_logged_scan_tf_fail'):
                self._logged_scan_tf_fail = True
                self.node.get_logger().error(
                    f"CRITICAL: Cannot transform scan from '{frame_id}' to '{base_frame}'. "
                    f"TF lookup failed! Check that TF tree is published in rosbag or by robot. "
                    f"Without TF, scan points cannot be used for anchors/loops."
                )
            return None
        
        return self._transform_points(points_scan, transform)

    def _coerce_time(self, stamp):
        """
        Coerce various stamp types into `rclpy.time.Time`.

        `tf2_ros.Buffer.lookup_transform()` expects an `rclpy.time.Time` object.
        Passing `builtin_interfaces.msg.Time` can raise `TypeError` (and crash the node)
        during rosbag playback.
        """
        from rclpy.time import Time

        if stamp is None:
            return Time()

        # Already an rclpy Time
        if isinstance(stamp, Time):
            return stamp

        # Likely builtin_interfaces.msg.Time
        try:
            return Time.from_msg(stamp)
        except Exception:
            # Fallback: "latest available"
            return Time()
    
    def _lookup_transform(self, target_frame: str, source_frame: str, stamp) -> Optional[np.ndarray]:
        """Lookup TF transform as SE(3) pose."""
        try:
            from rclpy.duration import Duration
            from rclpy.time import Time
            timeout = Duration(seconds=self.config.get("tf_timeout_sec", 0.05))
            
            query_time = self._coerce_time(stamp)
            try:
                t = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, query_time, timeout=timeout
                )
            except (TransformException, TypeError, ValueError) as e_first:
                # Bag playback often contains static transforms; if a time-specific lookup
                # fails (extrapolation), fall back to "latest available".
                try:
                    t = self.tf_buffer.lookup_transform(
                        target_frame, source_frame, Time(), timeout=timeout
                    )
                except (TransformException, TypeError, ValueError):
                    raise e_first
            
            trans = t.transform.translation
            rot = t.transform.rotation
            
            # Convert to rotation vector using geometry.se3
            R = quat_to_rotmat(rot.x, rot.y, rot.z, rot.w)
            rotvec = rotmat_to_rotvec(R)
            
            return np.array([trans.x, trans.y, trans.z, rotvec[0], rotvec[1], rotvec[2]], dtype=float)
        
        except (TransformException, TypeError, ValueError) as e:
            self.node.get_logger().warn(
                f"TF lookup failed ({target_frame} ← {source_frame}): {e}",
                throttle_duration_sec=5.0)
            return None
        except Exception as e:
            self.node.get_logger().warn(
                f"TF lookup unexpected error ({target_frame} ← {source_frame}): {type(e).__name__}: {e}",
                throttle_duration_sec=5.0,
            )
            return None
    
    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply SE(3) transform to points."""
        R = rotvec_to_rotmat(transform[3:6])
        t = transform[:3]
        
        # points_new = R @ points^T + t
        return (R @ points.T).T + t
    
    def get_nearest_pose(self, stamp_sec: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get pose nearest to timestamp. Returns (pose, dt)."""
        if not self.odom_buffer:
            return None, None
        closest = min(self.odom_buffer, key=lambda x: abs(x[0] - stamp_sec))
        return closest[1], float(stamp_sec - closest[0])
    
    def get_nearest_image(self, stamp_sec: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get RGB image array nearest to timestamp. Returns (rgb_array, dt)."""
        if not self.image_buffer:
            return None, None
        closest = min(self.image_buffer, key=lambda x: abs(x[0] - stamp_sec))
        return closest[1], float(stamp_sec - closest[0])
    
    def get_nearest_depth(self, stamp_sec: float) -> Optional[Tuple[float, Optional[np.ndarray], Optional[np.ndarray], str]]:
        """Get depth data nearest to timestamp. Returns (timestamp, depth_array, points, frame_id)."""
        if not self.depth_buffer:
            return None
        return min(self.depth_buffer, key=lambda x: abs(x[0] - stamp_sec))
    
    def get_synchronized_rgbd(self, stamp_sec: float, max_dt: float = 0.05) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[str]]:
        """
        Get synchronized RGB + depth pair nearest to timestamp.
        
        Args:
            stamp_sec: Target timestamp
            max_dt: Maximum time offset for synchronization (seconds)
        
        Returns:
            (rgb_array, depth_array, dt) or (None, None, None) if no sync found
        """
        if not self.image_buffer or not self.depth_buffer:
            return None, None, None, None
        
        # Find closest depth
        depth_item = min(self.depth_buffer, key=lambda x: abs(x[0] - stamp_sec))
        depth_stamp, depth_array, _, depth_frame = depth_item
        
        # Find closest RGB to the depth timestamp
        rgb_item = min(self.image_buffer, key=lambda x: abs(x[0] - depth_stamp))
        rgb_stamp, rgb_array, _rgb_frame = rgb_item
        
        # Check sync quality
        dt_rgb_depth = abs(rgb_stamp - depth_stamp)
        if dt_rgb_depth > max_dt:
            return None, None, None, None
        
        dt = float(stamp_sec - depth_stamp)
        return rgb_array, depth_array, dt, depth_frame
    
    def get_nearest_pointcloud(self, stamp_sec: float) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
        """
        Get point cloud nearest to timestamp.
        
        Args:
            stamp_sec: Target timestamp in seconds
            
        Returns:
            (points_array, dt, frame_id) or (None, None, None) if no data
        """
        if not self.pointcloud_buffer:
            return None, None, None
        closest = min(self.pointcloud_buffer, key=lambda x: abs(x[0] - stamp_sec))
        stamp, points, frame_id = closest
        return points, float(stamp_sec - stamp), frame_id
    
    def get_latest_pointcloud(self) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
        """
        Get the most recent point cloud.
        
        Returns:
            (points_array, timestamp, frame_id) or (None, None, None) if no data
        """
        if not self.pointcloud_buffer:
            return None, None, None
        stamp, points, frame_id = self.pointcloud_buffer[-1]
        return points, stamp, frame_id
    
    def is_3d_mode(self) -> bool:
        """Check if running in 3D point cloud mode."""
        return self.use_3d_pointcloud

    def get_imu_measurements(self, start_sec: float, end_sec: float) -> list:
        """
        Retrieve IMU measurements in time interval [start_sec, end_sec].

        Args:
            start_sec: Start of interval (seconds)
            end_sec: End of interval (seconds)

        Returns:
            List of (timestamp, accel, gyro) tuples sorted by timestamp.
            Each accel/gyro is a numpy array of shape (3,).
        """
        measurements = [
            (t, a.copy(), g.copy())
            for t, a, g in self.imu_buffer
            if start_sec <= t <= end_sec
        ]
        return sorted(measurements, key=lambda x: x[0])

    def clear_imu_buffer(self, before_sec: float) -> int:
        """
        Clear IMU measurements before the specified timestamp.

        Called after preintegration to remove consumed measurements.
        This is the event-driven clearing approach (no automatic sliding window).

        Args:
            before_sec: Clear all measurements with timestamp < before_sec

        Returns:
            Number of measurements cleared
        """
        original_len = len(self.imu_buffer)
        self.imu_buffer = [
            (t, a, g) for t, a, g in self.imu_buffer if t >= before_sec
        ]
        cleared = original_len - len(self.imu_buffer)
        return cleared

    def get_imu_buffer_size(self) -> int:
        """Get current number of buffered IMU measurements."""
        return len(self.imu_buffer)
