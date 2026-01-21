"""
Frobenius-Legendre SLAM Frontend Node (REFACTORED).

This is pure ORCHESTRATION - all math is in operators/ and models/.

Modular structure:
- frontend.sensor_io: Sensor I/O, buffering, TF, point clouds (NO MATH)
- frontend.descriptor_builder: Descriptor extraction (uses models.nig)
- frontend.anchor_manager: Anchor lifecycle (uses models.birth, operators.third_order_correct)
- frontend.loop_processor: Loop detection (uses operators.icp, operators.information_distances)

Data association using information-geometric distances:
- Fisher-Rao distance on Student-t predictive (NIG model)
- Product manifold distance for multi-channel descriptors
- Probabilistic domain constraints (no hard gates)

Loop Factor Convention (EXPLICIT):
    Z = T_anchor^{-1} ∘ T_current
    Backend reconstruction: T_current = T_anchor ∘ Z

Covariance Convention:
    se(3) tangent space at identity, [δx, δy, δz, δωx, δωy, δωz]
    Transported via adjoint representation.

Observability:
    Publishes /cdwm/frontend_status (JSON) with sensor connection status.

Reference: Miyamoto et al. (2024), Combe (2022-2025), Barfoot (2017)
"""

import json
import time
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String

# Import modular helpers (orchestration only, call operators/models)
from fl_slam_poc.frontend.processing import SensorIO, StatusMonitor, SensorStatus
from fl_slam_poc.frontend.anchors import DescriptorBuilder, AnchorManager
from fl_slam_poc.frontend.loops import LoopProcessor
from fl_slam_poc.backend.parameters import (
    AdaptiveParameter,
    TimeAlignmentModel,
    StochasticBirthModel,
)
from fl_slam_poc.backend.fusion.gaussian_info import make_evidence
from fl_slam_poc.frontend.loops.vmf_geometry import vmf_make_evidence
from fl_slam_poc.common import constants
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common.transforms.se3 import rotmat_to_quat, rotvec_to_rotmat, se3_compose, se3_relative
from fl_slam_poc.msg import AnchorCreate, LoopFactor, IMUFactor
from fl_slam_poc.operators.imu_preintegration import IMUPreintegrator


def stamp_to_sec(stamp) -> float:
    """Convert ROS timestamp to seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class Frontend(Node):
    """
    Refactored frontend - orchestration only.
    
    All math operations call operators/ and models/ via helper modules.
    """
    
    def __init__(self):
        super().__init__("fl_frontend")
        self._declare_parameters()
        self._init_modules()
        self._init_publishers()
        self._init_timer()
        
        # Startup banner (after all modules initialized)
        self.get_logger().info("=" * 60)
        self.get_logger().info("FL-SLAM Frontend initialized")
        self.get_logger().info("=" * 60)
        mode = "3D PointCloud" if self.sensor_io.use_3d_pointcloud else "2D LaserScan + RGB-D"
        self.get_logger().info(f"Mode: {mode}")
        self.get_logger().info(f"Birth intensity: {float(self.get_parameter('birth_intensity').value)}")
        self.get_logger().info(f"Using GPU: {bool(self.get_parameter('use_gpu').value)}")
        if self.sensor_io.depth_intrinsics:
            fx, fy, cx, cy = self.sensor_io.depth_intrinsics
            self.get_logger().info(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        else:
            self.get_logger().warn("NO camera intrinsics - RGB-D evidence DISABLED until set")
        self.get_logger().info("")
        self.get_logger().info("Waiting for sensor data to start processing...")
        self.get_logger().info("=" * 60)
    
    def _declare_parameters(self):
        """Declare all ROS parameters with defaults."""
        # Topics
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_is_delta", False)
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("enable_image", True)
        self.declare_parameter("enable_depth", True)
        self.declare_parameter("enable_camera_info", True)
        # If no CameraInfo is available in the bag, you can provide intrinsics directly.
        # Set all four > 0 to enable.
        self.declare_parameter("camera_fx", 0.0)
        self.declare_parameter("camera_fy", 0.0)
        self.declare_parameter("camera_cx", 0.0)
        self.declare_parameter("camera_cy", 0.0)
        self.declare_parameter("publish_rgbd_evidence", True)
        self.declare_parameter("rgbd_evidence_topic", "/sim/rgbd_evidence")
        self.declare_parameter("rgbd_publish_every_n_scans", 5)
        self.declare_parameter("rgbd_max_points_per_msg", 500)
        # RGB-D synchronization + depth descriptor range (used even in 3D PointCloud mode)
        self.declare_parameter("rgbd_sync_max_dt_sec", 0.1)
        self.declare_parameter("rgbd_min_depth_m", 0.1)
        self.declare_parameter("rgbd_max_depth_m", 10.0)
        
        # Budgets
        self.declare_parameter("descriptor_bins", 60)
        self.declare_parameter("anchor_budget", 0)
        self.declare_parameter("loop_budget", 0)
        self.declare_parameter("anchor_id_offset", 0)
        
        # ICP
        self.declare_parameter("icp_max_iter_prior", 15)
        self.declare_parameter("icp_tol_prior", 1e-4)
        self.declare_parameter("icp_prior_strength", 10.0)
        self.declare_parameter("icp_n_ref", 100.0)
        self.declare_parameter("icp_sigma_mse", 0.2)  # 20cm tolerance for real robot data
        
        # Sensor
        self.declare_parameter("depth_stride", 4)
        self.declare_parameter("feature_buffer_len", 10)
        self.declare_parameter("sensor_qos_reliability", "reliable")
        
        # Timestamp alignment
        self.declare_parameter("alignment_sigma_prior", 0.1)
        self.declare_parameter("alignment_prior_strength", 5.0)
        self.declare_parameter("alignment_sigma_floor", 0.001)
        
        # Birth model
        self.declare_parameter("birth_intensity", 10.0)
        self.declare_parameter("scan_period", 0.1)
        self.declare_parameter("base_component_weight", 1.0)
        
        # IMU Integration
        self.declare_parameter("enable_imu", False)
        self.declare_parameter("imu_topic", constants.IMU_TOPIC_DEFAULT)
        self.declare_parameter("imu_gyro_noise_density", constants.IMU_GYRO_NOISE_DENSITY_DEFAULT)
        self.declare_parameter("imu_accel_noise_density", constants.IMU_ACCEL_NOISE_DENSITY_DEFAULT)
        self.declare_parameter("imu_gyro_random_walk", constants.IMU_GYRO_RANDOM_WALK_DEFAULT)
        self.declare_parameter("imu_accel_random_walk", constants.IMU_ACCEL_RANDOM_WALK_DEFAULT)
        self.declare_parameter("keyframe_translation_threshold", 0.5)
        self.declare_parameter("keyframe_rotation_threshold", 0.26)
        self.declare_parameter("gravity", list(constants.GRAVITY_DEFAULT))
        
        # Frames
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("scan_frame", "base_link")
        self.declare_parameter("tf_timeout_sec", 0.05)
        
        # Fisher-Rao
        self.declare_parameter("fr_distance_scale_prior", 1.0)
        self.declare_parameter("fr_scale_prior_strength", 5.0)
        
        # 3D Point Cloud Mode
        self.declare_parameter("use_3d_pointcloud", False)  # Enable 3D point cloud mode
        self.declare_parameter("enable_pointcloud", False)  # Subscribe to PointCloud2
        self.declare_parameter("pointcloud_topic", constants.POINTCLOUD_TOPIC_DEFAULT)
        self.declare_parameter("pointcloud_range_min", 0.1)
        self.declare_parameter("pointcloud_range_max", 50.0)
        
        # Point Cloud Processing
        self.declare_parameter("voxel_size", 0.05)  # Voxel grid filter size (meters)
        self.declare_parameter("max_points_after_filter", 50000)
        self.declare_parameter("min_points_for_icp", 100)
        self.declare_parameter("icp_max_correspondence_distance", 0.5)
        self.declare_parameter("normal_estimation_radius", 0.1)
        self.declare_parameter("pointcloud_rate_limit_hz", 30.0)
        
        # GPU Acceleration
        self.declare_parameter("use_gpu", False)  # Enable GPU processing
        self.declare_parameter("gpu_device_index", 0)
        self.declare_parameter("gpu_fallback_to_cpu", True)
    
    def _init_modules(self):
        """Initialize modular components."""
        # Frames
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        
        # Sensor I/O configuration
        sensor_config = {
            "scan_topic": str(self.get_parameter("scan_topic").value),
            "odom_topic": str(self.get_parameter("odom_topic").value),
            "camera_topic": str(self.get_parameter("camera_topic").value),
            "depth_topic": str(self.get_parameter("depth_topic").value),
            "camera_info_topic": str(self.get_parameter("camera_info_topic").value),
            "enable_image": bool(self.get_parameter("enable_image").value),
            "enable_depth": bool(self.get_parameter("enable_depth").value),
            "enable_camera_info": bool(self.get_parameter("enable_camera_info").value),
            "odom_is_delta": bool(self.get_parameter("odom_is_delta").value),
            "odom_frame": self.odom_frame,
            "base_frame": str(self.get_parameter("base_frame").value),
            "camera_frame": str(self.get_parameter("camera_frame").value),
            "scan_frame": str(self.get_parameter("scan_frame").value),
            "tf_timeout_sec": float(self.get_parameter("tf_timeout_sec").value),
            "feature_buffer_len": int(self.get_parameter("feature_buffer_len").value),
            "depth_stride": int(self.get_parameter("depth_stride").value),
            "sensor_qos_reliability": str(self.get_parameter("sensor_qos_reliability").value),
            # IMU
            "enable_imu": bool(self.get_parameter("enable_imu").value),
            "imu_topic": str(self.get_parameter("imu_topic").value),
            # 3D Point Cloud Mode
            "use_3d_pointcloud": bool(self.get_parameter("use_3d_pointcloud").value),
            "enable_pointcloud": bool(self.get_parameter("enable_pointcloud").value),
            "pointcloud_topic": str(self.get_parameter("pointcloud_topic").value),
            # Point Cloud Processing
            "voxel_size": float(self.get_parameter("voxel_size").value),
            "max_points_after_filter": int(self.get_parameter("max_points_after_filter").value),
            "min_points_for_icp": int(self.get_parameter("min_points_for_icp").value),
            "icp_max_correspondence_distance": float(self.get_parameter("icp_max_correspondence_distance").value),
            # GPU Acceleration
            "use_gpu": bool(self.get_parameter("use_gpu").value),
            "gpu_device_index": int(self.get_parameter("gpu_device_index").value),
            "gpu_fallback_to_cpu": bool(self.get_parameter("gpu_fallback_to_cpu").value),
        }
        
        # Initialize SensorIO (pure I/O, no math)
        self.sensor_io = SensorIO(self, sensor_config)
        if sensor_config["use_3d_pointcloud"]:
            self.sensor_io.set_pointcloud_callback(self._on_pointcloud)
        else:
            self.sensor_io.set_scan_callback(self._on_scan)
        self.sensor_io.set_odom_callback(lambda _msg: self.status_monitor.mark_received("odom"))
        if sensor_config["enable_image"]:
            self.sensor_io.set_image_callback(lambda _msg: self.status_monitor.mark_received("camera"))
        if sensor_config["enable_depth"]:
            self.sensor_io.set_depth_callback(lambda _msg: self.status_monitor.mark_received("depth"))

        # Optional: set intrinsics from parameters if CameraInfo is absent.
        fx = float(self.get_parameter("camera_fx").value)
        fy = float(self.get_parameter("camera_fy").value)
        cx = float(self.get_parameter("camera_cx").value)
        cy = float(self.get_parameter("camera_cy").value)
        if fx > 0.0 and fy > 0.0 and cx > 0.0 and cy > 0.0:
            self.sensor_io.depth_intrinsics = (fx, fy, cx, cy)
            self.get_logger().info(
                f"Frontend: Using camera intrinsics from parameters: "
                f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"
            )
        else:
            self.get_logger().warn(
                "Frontend: No camera intrinsics set (all values must be > 0). "
                "RGB-D evidence will NOT be published until CameraInfo is received or parameters are set."
            )
        
        # Initialize DescriptorBuilder (uses models.nig)
        descriptor_bins = int(self.get_parameter("descriptor_bins").value)
        self.rgbd_sync_max_dt_sec = float(self.get_parameter("rgbd_sync_max_dt_sec").value)
        self.rgbd_min_depth_m = float(self.get_parameter("rgbd_min_depth_m").value)
        self.rgbd_max_depth_m = float(self.get_parameter("rgbd_max_depth_m").value)
        self.descriptor_builder = DescriptorBuilder(
            descriptor_bins,
            depth_range_m=(self.rgbd_min_depth_m, self.rgbd_max_depth_m),
        )
        
        # Adaptive models (from models.adaptive, models.timestamp)
        align_prior = float(self.get_parameter("alignment_sigma_prior").value)
        align_strength = float(self.get_parameter("alignment_prior_strength").value)
        align_floor = float(self.get_parameter("alignment_sigma_floor").value)
        
        self.align_pose = TimeAlignmentModel(align_prior, align_strength, align_floor)
        self.align_image = TimeAlignmentModel(align_prior, align_strength, align_floor)
        self.align_depth = TimeAlignmentModel(align_prior, align_strength, align_floor)
        
        fr_prior = float(self.get_parameter("fr_distance_scale_prior").value)
        fr_strength = float(self.get_parameter("fr_scale_prior_strength").value)
        fr_distance_scale = AdaptiveParameter(fr_prior, fr_strength, floor=0.01)
        
        icp_iter_prior = int(self.get_parameter("icp_max_iter_prior").value)
        icp_tol_prior = float(self.get_parameter("icp_tol_prior").value)
        icp_strength = float(self.get_parameter("icp_prior_strength").value)
        icp_max_iter = AdaptiveParameter(float(icp_iter_prior), icp_strength, floor=3.0)
        icp_tol = AdaptiveParameter(icp_tol_prior, icp_strength, floor=1e-6)
        
        # Initialize AnchorManager (uses models.birth, operators.third_order_correct)
        birth_intensity = float(self.get_parameter("birth_intensity").value)
        scan_period = float(self.get_parameter("scan_period").value)
        birth_model = StochasticBirthModel(birth_intensity, scan_period)
        anchor_id_offset = int(self.get_parameter("anchor_id_offset").value)
        self.anchor_manager = AnchorManager(birth_model, anchor_id_offset)
        
        # Initialize base weight
        self.anchor_manager.base_weight = float(self.get_parameter("base_component_weight").value)
        
        # Initialize LoopProcessor (uses operators.icp, operators.information_distances)
        icp_n_ref = float(self.get_parameter("icp_n_ref").value)
        icp_sigma_mse = float(self.get_parameter("icp_sigma_mse").value)
        
        # GPU configuration for LoopProcessor
        use_gpu = bool(self.get_parameter("use_gpu").value)
        gpu_device_index = int(self.get_parameter("gpu_device_index").value)
        gpu_fallback_to_cpu = bool(self.get_parameter("gpu_fallback_to_cpu").value)
        voxel_size = float(self.get_parameter("voxel_size").value)
        max_correspondence_distance = float(self.get_parameter("icp_max_correspondence_distance").value)
        use_3d_pointcloud = bool(self.get_parameter("use_3d_pointcloud").value)
        
        self.loop_processor = LoopProcessor(
            fr_distance_scale=fr_distance_scale,
            icp_max_iter=icp_max_iter,
            icp_tol=icp_tol,
            icp_n_ref=icp_n_ref,
            icp_sigma_mse=icp_sigma_mse,
            use_gpu=use_gpu,
            gpu_device_index=gpu_device_index,
            gpu_fallback_to_cpu=gpu_fallback_to_cpu,
            voxel_size=voxel_size,
            max_correspondence_distance=max_correspondence_distance,
            use_3d_pointcloud=use_3d_pointcloud
        )

        # IMU preintegration setup
        self.enable_imu = bool(self.get_parameter("enable_imu").value)
        self.keyframe_translation_threshold = float(self.get_parameter("keyframe_translation_threshold").value)
        self.keyframe_rotation_threshold = float(self.get_parameter("keyframe_rotation_threshold").value)
        self.gravity = np.array(self.get_parameter("gravity").value, dtype=float)
        if self.enable_imu:
            self.imu_preintegrator = IMUPreintegrator(
                gyro_noise_density=float(self.get_parameter("imu_gyro_noise_density").value),
                accel_noise_density=float(self.get_parameter("imu_accel_noise_density").value),
                gyro_random_walk=float(self.get_parameter("imu_gyro_random_walk").value),
                accel_random_walk=float(self.get_parameter("imu_accel_random_walk").value),
                gravity=self.gravity,
            )
        else:
            self.imu_preintegrator = None
        self._last_keyframe_id: Optional[int] = None
        self._last_keyframe_stamp: Optional[float] = None
        self._last_keyframe_pose: Optional[np.ndarray] = None
        
        # Log GPU status
        if use_gpu:
            if self.loop_processor.is_using_gpu():
                self.get_logger().info("LoopProcessor: GPU acceleration enabled")
            else:
                self.get_logger().warning("LoopProcessor: GPU requested but not available, using CPU")
        
        # Status monitoring
        self.status_monitor = StatusMonitor(self)
        if use_3d_pointcloud:
            self.status_monitor.add_sensor("pointcloud", sensor_config["pointcloud_topic"])
        else:
            self.status_monitor.add_sensor("scan", str(self.get_parameter("scan_topic").value))
        self.status_monitor.add_sensor("odom", str(self.get_parameter("odom_topic").value))
        if sensor_config["enable_image"]:
            self.status_monitor.add_sensor("camera", sensor_config["camera_topic"])
        if sensor_config["enable_depth"]:
            self.status_monitor.add_sensor("depth", sensor_config["depth_topic"])
        if sensor_config.get("enable_imu", False):
            self.status_monitor.add_sensor("imu", sensor_config.get("imu_topic"))
    
    def _init_publishers(self):
        """Initialize ROS publishers."""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        self.pub_loop = self.create_publisher(LoopFactor, "/sim/loop_factor", qos)
        self.pub_anchor = self.create_publisher(AnchorCreate, "/sim/anchor_create", qos)
        if bool(self.get_parameter("enable_imu").value):
            self.pub_imu_factor = self.create_publisher(IMUFactor, "/sim/imu_factor", qos)
        self.pub_report = self.create_publisher(String, "/cdwm/op_report", qos)
        self.pub_status = self.create_publisher(String, "/cdwm/frontend_status", qos)

        # RGB-D dense evidence (JSON payload) for backend mapping.
        self.pub_rgbd = self.create_publisher(
            String,
            str(self.get_parameter("rgbd_evidence_topic").value),
            qos,
        )
        self._scan_count = 0
    
    def _init_timer(self):
        """Initialize status publishing timer."""
        self.create_timer(1.0, self._publish_status)
    
    def _on_scan(self, msg: LaserScan):
        """
        Main scan processing callback (orchestration only).

        All math operations call operators/ and models/ via helper modules.
        """
        self.status_monitor.mark_received("scan")

        scan_stamp = stamp_to_sec(msg.header.stamp)

        # Get aligned sensor data with probabilistic weights
        pose, pose_dt = self.sensor_io.get_nearest_pose(scan_stamp)

        # If no pose available yet, buffer the scan for later processing
        if pose is None:
            if not hasattr(self, '_scan_buffer'):
                self._scan_buffer = []
            self._scan_buffer.append(msg)

            # Log buffer status occasionally
            if len(self._scan_buffer) % 10 == 1:
                self.get_logger().info(
                    f"Frontend: Buffered {len(self._scan_buffer)} scans waiting for odometry. "
                    f"Odom buffer size: {len(self.sensor_io.odom_buffer)}"
                )
            return

        # Process any buffered scans first
        if hasattr(self, '_scan_buffer') and self._scan_buffer:
            self.get_logger().info(f"Frontend: Processing {len(self._scan_buffer)} buffered scans")
            for buffered_msg in self._scan_buffer:
                self._process_scan(buffered_msg)
            self._scan_buffer.clear()

        # Process current scan
        self._process_scan(msg)

    def _process_scan(self, msg: LaserScan):
        """Process a single scan message."""
        scan_stamp = stamp_to_sec(msg.header.stamp)

        # Get aligned sensor data with probabilistic weights
        pose, pose_dt = self.sensor_io.get_nearest_pose(scan_stamp)
        if pose is None:
            # This shouldn't happen since we check before calling
            return

        self.align_pose.update(pose_dt)
        pose_weight = self.align_pose.weight(pose_dt)

        # Get image features if enabled
        image_rgb, image_dt = self.sensor_io.get_nearest_image(scan_stamp)
        if image_dt is not None:
            self.align_image.update(image_dt)
        image_weight = self.align_image.weight(image_dt)

        # Get synchronized RGB-D pair for true evidence extraction
        rgb_array, depth_array, rgbd_dt, depth_frame = self.sensor_io.get_synchronized_rgbd(
            scan_stamp, max_dt=self.rgbd_sync_max_dt_sec
        )
        if rgbd_dt is not None:
            self.align_depth.update(rgbd_dt)
        depth_weight = self.align_depth.weight(rgbd_dt)

        # Convert scan to points (LaserScan is the sparse anchor / loop source).
        # Prefer LaserScan points for ICP; if TF is missing, fall back to depth points
        # but cap the number of points to keep ICP tractable.
        scan_points = self.sensor_io.scan_to_points(msg)
        depth_item = self.sensor_io.get_nearest_depth(scan_stamp)
        depth_points = depth_item[2] if depth_item else None

        points = scan_points
        point_source = "scan"
        if points is None and depth_points is not None:
            max_pts = 1000
            if len(depth_points) > max_pts:
                idx = np.linspace(0, len(depth_points) - 1, max_pts, dtype=int)
                points = depth_points[idx]
            else:
                points = depth_points
            point_source = "depth_fallback"

        if points is None:
            point_source = "none"

        # Publish dense RGB-D evidence with TRUE colors/normals (no silent pipeline)
        self._scan_count += 1
        if (
            bool(self.get_parameter("publish_rgbd_evidence").value)
            and rgb_array is not None
            and depth_array is not None
            and (self._scan_count % int(self.get_parameter("rgbd_publish_every_n_scans").value) == 0)
        ):
            self._publish_rgbd_evidence(
                rgb=rgb_array,
                depth=depth_array,
                pose_odom_base=pose,
                camera_frame=depth_frame or str(self.get_parameter("camera_frame").value),
                stamp=msg.header.stamp,
            )

        # Build descriptor (uses models.nig via descriptor_builder)
        scan_desc = self.descriptor_builder.scan_descriptor(msg)
        image_feat_desc = self.descriptor_builder.image_descriptor(rgb_array if rgb_array is not None else image_rgb)
        depth_feat_desc = self.descriptor_builder.depth_descriptor(depth_array)
        desc = self.descriptor_builder.compose_descriptor(scan_desc, image_feat_desc, depth_feat_desc)

        obs_weight = pose_weight * image_weight * depth_weight

        # Initialize global model if needed
        self.descriptor_builder.init_global_model(desc)

        # Compute responsibilities (uses models.nig.fisher_rao_distance - exact)
        anchors = self.anchor_manager.get_all_anchors()
        global_model = self.descriptor_builder.get_global_model()
        base_weight = self.anchor_manager.get_base_weight()

        responsibilities, r_new = self.loop_processor.compute_responsibilities(
            desc, anchors, global_model, base_weight)

        # Debug: log responsibilities
        if not hasattr(self, '_resp_debug_count'):
            self._resp_debug_count = 0
        if self._resp_debug_count < 5 and len(anchors) > 0:  # Reduced from 10 to 5
            self._resp_debug_count += 1
            resp_str = ", ".join([f"a{aid}={r:.4f}" for aid, r in responsibilities.items()])
            self.get_logger().info(
                f"Responsibilities: n_anchors={len(anchors)}, r_new={r_new:.4f}, [{resp_str}]"
            )

        # Update anchors (uses models.nig.update - exact)
        r_new_eff = self.anchor_manager.update_anchors(desc, responsibilities, r_new, obs_weight)

        # Update global model (uses models.nig.update - exact)
        self.descriptor_builder.update_global_model(desc, obs_weight)

        # Stochastic birth decision (uses models.birth.sample_birth - exact Poisson)
        birth_prob = self.anchor_manager.get_birth_probability(r_new_eff)
        should_birth = self.anchor_manager.should_birth_anchor(r_new_eff)

        # Debug logging for first few scans
        if not hasattr(self, '_birth_debug_count'):
            self._birth_debug_count = 0
        if self._birth_debug_count < 10:
            self._birth_debug_count += 1
            self.get_logger().info(
                f"Scan #{self._birth_debug_count} processed: r_new_eff={r_new_eff:.6f}, "
                f"birth_prob={birth_prob:.4f}, should_birth={should_birth}, "
                f"obs_weight={obs_weight:.4f}, points={'OK' if points is not None else 'NONE'}, "
                f"anchors={len(anchors)}"
            )

            if points is None:
                self.get_logger().warn(
                    f"Scan #{self._birth_debug_count}: NO POINTS available! "
                    f"Check TF transforms between scan/pointcloud frame and base_link. "
                    f"Anchors CANNOT be created without points."
                )

        if should_birth and points is not None:
            global_model_copy = self.descriptor_builder.copy_global_model()
            anchor_id = self.anchor_manager.create_anchor(
                stamp_sec=scan_stamp,
                pose=pose,
                descriptor=desc,
                desc_model=global_model_copy,
                r_new_eff=r_new_eff,
                points=points,
                frame_id=self.odom_frame
            )

            self._publish_anchor_create(anchor_id, msg.header.stamp, points)
            self.get_logger().info(f"✓ Created anchor {anchor_id} with {len(points)} points")

            # Publish IMU factor between keyframes (anchors)
            self._publish_imu_factor(anchor_id, msg.header.stamp, scan_stamp, pose)

            self._publish_report(OpReport(
                name="StochasticAnchorBirth",
                exact=True,
                family_in="Poisson",
                family_out="Anchor",
                closed_form=True,
                metrics={"anchor_id": anchor_id, "r_new_eff": r_new_eff, "birth_probability": birth_prob},
                notes="Poisson birth with intensity λ = λ₀ * r_new.",
            ))

        # Apply anchor budget if needed (uses operators.third_order_correct)
        anchor_budget = int(self.get_parameter("anchor_budget").value)
        if anchor_budget > 0:
            budget_report = self.anchor_manager.apply_budget(anchor_budget)
            if budget_report is not None:
                self._publish_report(budget_report)

        # Publish loop factors
        self._publish_loop_factors(responsibilities, msg, points, obs_weight, point_source)

    def _on_pointcloud(self, msg: PointCloud2, points: np.ndarray):
        """
        Main 3D point cloud processing callback.

        Mirrors `_on_scan` but uses PointCloud2 as the triggering sensor.
        """
        self.status_monitor.mark_received("pointcloud")

        pc_stamp = stamp_to_sec(msg.header.stamp)

        # Get aligned odometry pose (absolute)
        pose, pose_dt = self.sensor_io.get_nearest_pose(pc_stamp)

        # If no pose available yet, buffer the point cloud for later processing
        if pose is None:
            if not hasattr(self, '_pointcloud_buffer'):
                self._pointcloud_buffer = []
            self._pointcloud_buffer.append((msg, points))

            # Log buffer status occasionally
            if len(self._pointcloud_buffer) % 5 == 1:
                self.get_logger().info(
                    f"Frontend: Buffered {len(self._pointcloud_buffer)} point clouds waiting for odometry. "
                    f"Odom buffer size: {len(self.sensor_io.odom_buffer)}"
                )
            return

        # Process any buffered point clouds first
        if hasattr(self, '_pointcloud_buffer') and self._pointcloud_buffer:
            self.get_logger().info(f"Frontend: Processing {len(self._pointcloud_buffer)} buffered point clouds")
            for buffered_msg, buffered_points in self._pointcloud_buffer:
                self._process_pointcloud(buffered_msg, buffered_points)
            self._pointcloud_buffer.clear()

        # Process current point cloud
        self._process_pointcloud(msg, points)

    def _process_pointcloud(self, msg: PointCloud2, points: np.ndarray):
        """Process a single point cloud message."""
        pc_stamp = stamp_to_sec(msg.header.stamp)

        # Get aligned odometry pose (absolute)
        pose, pose_dt = self.sensor_io.get_nearest_pose(pc_stamp)
        if pose is None:
            # This shouldn't happen since we check before calling
            return

        self.align_pose.update(pose_dt)
        pose_weight = self.align_pose.weight(pose_dt)

        # Get image features if enabled
        image_rgb, image_dt = self.sensor_io.get_nearest_image(pc_stamp)
        if image_dt is not None:
            self.align_image.update(image_dt)
        image_weight = self.align_image.weight(image_dt)

        # Get synchronized RGB-D pair for appearance/depth descriptors (and optional dense evidence)
        rgb_array, depth_array, rgbd_dt, _depth_frame = self.sensor_io.get_synchronized_rgbd(
            pc_stamp, max_dt=self.rgbd_sync_max_dt_sec
        )
        if rgbd_dt is not None:
            self.align_depth.update(rgbd_dt)
        depth_weight = self.align_depth.weight(rgbd_dt)

        # Build a scan-like range histogram descriptor from point ranges.
        rmin = float(self.get_parameter("pointcloud_range_min").value)
        rmax = float(self.get_parameter("pointcloud_range_max").value)
        ranges = np.linalg.norm(points.astype(np.float64), axis=1)
        valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
        if np.any(valid):
            hist, _ = np.histogram(
                ranges[valid],
                bins=self.descriptor_builder.descriptor_bins,
                range=(rmin, rmax),
            )
            scan_desc = np.asarray(hist, dtype=float)
            desc_sum = float(np.sum(scan_desc))
            if desc_sum > 1e-12:
                scan_desc = scan_desc / desc_sum
        else:
            scan_desc = np.zeros(self.descriptor_builder.descriptor_bins, dtype=float)

        image_feat_desc = self.descriptor_builder.image_descriptor(rgb_array if rgb_array is not None else image_rgb)
        depth_feat_desc = self.descriptor_builder.depth_descriptor(depth_array)
        desc = self.descriptor_builder.compose_descriptor(scan_desc, image_feat_desc, depth_feat_desc)

        obs_weight = pose_weight * image_weight * depth_weight

        # Initialize global model if needed
        self.descriptor_builder.init_global_model(desc)

        anchors = self.anchor_manager.get_all_anchors()
        global_model = self.descriptor_builder.get_global_model()
        base_weight = self.anchor_manager.get_base_weight()

        responsibilities, r_new = self.loop_processor.compute_responsibilities(
            desc, anchors, global_model, base_weight
        )

        r_new_eff = self.anchor_manager.update_anchors(desc, responsibilities, r_new, obs_weight)
        self.descriptor_builder.update_global_model(desc, obs_weight)

        should_birth = self.anchor_manager.should_birth_anchor(r_new_eff)

        # Debug logging for first few pointcloud scans
        if not hasattr(self, '_pc_birth_debug_count'):
            self._pc_birth_debug_count = 0
        if self._pc_birth_debug_count < 10:
            self._pc_birth_debug_count += 1
            self.get_logger().info(
                f"PointCloud #{self._pc_birth_debug_count} processed: r_new_eff={r_new_eff:.6f}, "
                f"should_birth={should_birth}, points={len(points) if points is not None else 0}, "
                f"anchors={len(anchors)}"
            )

            if points is None or len(points) == 0:
                self.get_logger().warn(
                    f"PointCloud #{self._pc_birth_debug_count}: NO POINTS! "
                    f"Check TF transforms and point cloud validity."
                )

        if should_birth and points is not None and len(points) > 0:
            global_model_copy = self.descriptor_builder.copy_global_model()
            anchor_id = self.anchor_manager.create_anchor(
                stamp_sec=pc_stamp,
                pose=pose,
                descriptor=desc,
                desc_model=global_model_copy,
                r_new_eff=r_new_eff,
                points=points,
                frame_id=self.odom_frame,
            )
            self._publish_anchor_create(anchor_id, msg.header.stamp, points)
            self.get_logger().info(f"✓ Created anchor {anchor_id} with {len(points)} points (PointCloud2)")

        # Apply anchor budget if needed (uses operators.third_order_correct)
        anchor_budget = int(self.get_parameter("anchor_budget").value)
        if anchor_budget > 0:
            budget_report = self.anchor_manager.apply_budget(anchor_budget)
            if budget_report is not None:
                self._publish_report(budget_report)

        # Publish loop factors
        self._publish_loop_factors(responsibilities, msg, points, obs_weight, "pointcloud")
    
    def _publish_loop_factors(self, responsibilities: dict, msg,
                              points: Optional[np.ndarray], obs_weight: float, point_source: str):
        """
        Publish loop factors with ICP registration.
        
        Uses operators.icp, operators.transport_covariance_to_frame, operators.gaussian_frobenius_correction.
        """
        if points is None or len(self.anchor_manager.get_all_anchors()) == 0:
            return
        
        # Apply loop budget if needed (uses operators.third_order_correct)
        loop_budget = int(self.get_parameter("loop_budget").value)
        truncation_applied = False
        
        if loop_budget > 0 and len(responsibilities) > loop_budget:
            responsibilities, budget_report = self.loop_processor.apply_loop_budget(
                responsibilities, loop_budget)
            truncation_applied = True
            if budget_report is not None:
                self._publish_report(budget_report)
        
        # Process each anchor with non-zero responsibility
        for anchor in self.anchor_manager.get_all_anchors():
            weight = responsibilities.get(anchor.anchor_id, 0.0)
            if weight < 1e-12:
                continue
            
            # Debug first few loop attempts
            if not hasattr(self, '_loop_debug_count'):
                self._loop_debug_count = 0
            if self._loop_debug_count < 5:
                self._loop_debug_count += 1
                self.get_logger().info(
                    f"Loop attempt: anchor={anchor.anchor_id}, resp={weight:.6f}, "
                    f"src_pts={len(points) if points is not None else 0}, "
                    f"tgt_pts={len(anchor.depth_points) if anchor.depth_points is not None else 0}"
                )
            
            # Run ICP (uses operators.icp_3d - exact solver)
            icp_result = self.loop_processor.run_icp(points, anchor.depth_points)
            
            if self._loop_debug_count <= 5 and icp_result is not None:
                self.get_logger().info(
                    f"ICP result: converged={icp_result.converged}, mse={icp_result.mse:.6f}, "
                    f"iters={icp_result.iterations}"
                )
            
            if icp_result is None or not icp_result.converged:
                continue
            
            # Compute loop factor (uses operators exact formulas)
            rel_pose, cov_transported, final_weight = self.loop_processor.compute_loop_factor(
                icp_result, anchor.pose, obs_weight, weight)
            
            if self._loop_debug_count <= 3:  # Reduced from 5 to 3
                self.get_logger().info(
                    f"Loop factor computed: anchor={anchor.anchor_id}, final_weight={final_weight:.6f}, "
                    f"rel_pose_norm={np.linalg.norm(rel_pose) if rel_pose is not None else 0:.4f}"
                )
            
            if rel_pose is None or final_weight < 1e-12:
                if self._loop_debug_count <= 5:
                    self.get_logger().warn(f"Loop factor DROPPED: rel_pose is None or weight too low")
                continue
            
            # Update adaptive ICP parameters
            self.loop_processor.update_adaptive_params(icp_result)
            
            # Publish loop factor
            self._publish_loop(rel_pose, cov_transported, msg.header.stamp,
                              anchor.anchor_id, final_weight, icp_result,
                              truncation_applied, point_source)
            
            if not hasattr(self, '_loop_published_count'):
                self._loop_published_count = 0
            self._loop_published_count += 1
            if self._loop_published_count <= 5:
                self.get_logger().info(f"✓ Published loop factor to anchor {anchor.anchor_id}")
            
            # Apply Frobenius correction for linearization (uses operators.gaussian_frobenius_correction)
            _, frob_stats = self.loop_processor.apply_frobenius_correction(rel_pose)
            
            # Publish OpReport
            self._publish_report(OpReport(
                name="LoopFactorPublished",
                exact=False,
                approximation_triggers=["Linearization"] + (["BudgetTruncation"] if truncation_applied else []),
                family_in="PointCloud",
                family_out="Gaussian",
                closed_form=False,
                solver_used="ICP",
                frobenius_applied=True,
                frobenius_operator="gaussian_identity_third_order",
                frobenius_delta_norm=float(frob_stats["delta_norm"]),
                frobenius_input_stats=dict(frob_stats["input_stats"]),
                frobenius_output_stats=dict(frob_stats["output_stats"]),
                metrics={
                    "anchor_id": anchor.anchor_id,
                    "weight": final_weight,
                    "mse": icp_result.mse,
                    "iterations": icp_result.iterations,
                    "converged": icp_result.converged,
                    "point_source": point_source,
                },
                notes="ICP linearization at sensor layer per Jacobian policy.",
            ))
    
    def _publish_loop(self, rel_pose: np.ndarray, cov: np.ndarray, stamp,
                     anchor_id: int, weight: float, icp_result, truncation_applied: bool, point_source: str):
        """Publish LoopFactor message."""
        loop = LoopFactor()
        loop.header.stamp = stamp
        loop.header.frame_id = self.odom_frame
        loop.anchor_id = int(anchor_id)
        loop.weight = float(weight)
        loop.rel_pose.position.x = float(rel_pose[0])
        loop.rel_pose.position.y = float(rel_pose[1])
        loop.rel_pose.position.z = float(rel_pose[2])
        
        # Convert rotation vector to quaternion
        R = rotvec_to_rotmat(rel_pose[3:6])
        qx, qy, qz, qw = rotmat_to_quat(R)
        loop.rel_pose.orientation.x = qx
        loop.rel_pose.orientation.y = qy
        loop.rel_pose.orientation.z = qz
        loop.rel_pose.orientation.w = qw
        
        loop.covariance = cov.reshape(-1).tolist()
        loop.approximation_triggers = ["Linearization"] + (["BudgetTruncation"] if truncation_applied else [])
        loop.solver_name = "ICP_SCAN" if point_source == "scan" else "ICP"
        loop.solver_objective = icp_result.final_objective
        loop.solver_tolerance = icp_result.tolerance
        loop.solver_iterations = icp_result.iterations
        loop.solver_max_iterations = icp_result.max_iterations
        loop.information_weight = icp_result.final_objective  # Placeholder
        
        self.pub_loop.publish(loop)
    
    def _publish_anchor_create(self, anchor_id: int, stamp, points: np.ndarray):
        """Publish AnchorCreate message with point cloud."""
        from geometry_msgs.msg import Point
        
        msg = AnchorCreate()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.anchor_id = int(anchor_id)
        
        # Publish point cloud (subsample for message size)
        # Limit to 1000 points to keep message size reasonable
        max_points = 1000
        if len(points) > max_points:
            indices = np.linspace(0, len(points)-1, max_points, dtype=int)
            points_sub = points[indices]
        else:
            points_sub = points
        
        msg.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) 
                      for p in points_sub]
        
        self.pub_anchor.publish(msg)

    def _publish_imu_factor(self, keyframe_j: int, stamp, stamp_sec: float, pose: np.ndarray):
        """
        Publish IMU preintegration factor between successive keyframes (anchors).
        """
        if not self.enable_imu or self.imu_preintegrator is None:
            return

        # Initialize first keyframe without publishing a factor
        if self._last_keyframe_stamp is None or self._last_keyframe_id is None:
            self._last_keyframe_stamp = float(stamp_sec)
            self._last_keyframe_id = int(keyframe_j)
            self._last_keyframe_pose = pose.copy()
            return

        start_sec = float(self._last_keyframe_stamp)
        end_sec = float(stamp_sec)
        if end_sec <= start_sec:
            return

        imu_measurements = self.sensor_io.get_imu_measurements(start_sec, end_sec)
        bias_gyro = np.zeros(3, dtype=float)
        bias_accel = np.zeros(3, dtype=float)
        delta_rotvec, delta_v, delta_p, cov_preint, op_report = self.imu_preintegrator.integrate(
            start_stamp=start_sec,
            end_stamp=end_sec,
            imu_measurements=imu_measurements,
            bias_gyro=bias_gyro,
            bias_accel=bias_accel,
        )

        imu_msg = IMUFactor()
        imu_msg.header.stamp = stamp
        imu_msg.keyframe_i = int(self._last_keyframe_id)
        imu_msg.keyframe_j = int(keyframe_j)
        imu_msg.dt = float(end_sec - start_sec)
        imu_msg.delta_p = [float(x) for x in delta_p]
        imu_msg.delta_v = [float(x) for x in delta_v]
        imu_msg.delta_rotvec = [float(x) for x in delta_rotvec]
        imu_msg.bias_gyro = [float(x) for x in bias_gyro]
        imu_msg.bias_accel = [float(x) for x in bias_accel]
        imu_msg.cov_preint = cov_preint.reshape(-1).astype(float).tolist()
        imu_msg.n_measurements = int(len(imu_measurements))

        self.pub_imu_factor.publish(imu_msg)

        # Clear IMU buffer up to current keyframe
        cleared = self.sensor_io.clear_imu_buffer(end_sec)

        # Motion diagnostics between keyframes
        rel = se3_relative(pose, self._last_keyframe_pose)
        trans_norm = float(np.linalg.norm(rel[:3]))
        rot_norm = float(np.linalg.norm(rel[3:]))

        self._publish_report(op_report)
        self._publish_report(OpReport(
            name="IMUKeyframe",
            exact=True,
            family_in="SE3",
            family_out="SE3",
            closed_form=True,
            metrics={
                "keyframe_i": int(self._last_keyframe_id),
                "keyframe_j": int(keyframe_j),
                "translation_norm": trans_norm,
                "rotation_norm": rot_norm,
                "translation_threshold": self.keyframe_translation_threshold,
                "rotation_threshold": self.keyframe_rotation_threshold,
                "imu_measurements": imu_msg.n_measurements,
                "imu_cleared": cleared,
            },
            notes="IMU factor published between anchor keyframes.",
        ))
        if imu_msg.n_measurements <= 2:
            self.get_logger().warn(
                f"IMU factor {imu_msg.keyframe_i}->{imu_msg.keyframe_j}: "
                f"insufficient measurements (n={imu_msg.n_measurements}, cleared={cleared})"
            )

        # Update last keyframe trackers
        self._last_keyframe_stamp = end_sec
        self._last_keyframe_id = int(keyframe_j)
        self._last_keyframe_pose = pose.copy()

    def _publish_rgbd_evidence(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose_odom_base: np.ndarray,
        camera_frame: str,
        stamp,
    ):
        """
        Publish dense RGB-D evidence with TRUE colors and normals.
        
        This is the EXPLICIT pipeline: frontend extracts evidence → backend processes.
        No silent data flow.
        
        Args:
            rgb: (H, W, 3) RGB image (uint8)
            depth: (H, W) depth image (float32, meters)
            pose_odom_base: (6,) SE(3) pose of base_link in odom frame
            camera_frame: Camera frame ID for TF lookup
        """
        try:
            from fl_slam_poc.frontend.processing.rgbd_processor import (
                depth_to_pointcloud,
                rgbd_to_evidence,
                transform_evidence_to_global,
                subsample_evidence_spatially,
            )
            from rclpy.time import Time
            
            # Get camera intrinsics
            if self.sensor_io.depth_intrinsics is None:
                return
            fx, fy, cx, cy = self.sensor_io.depth_intrinsics
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            
            # Extract 3D points + colors + normals in camera frame
            points_cam, colors, normals, covs = depth_to_pointcloud(
                depth, K, rgb=rgb, subsample=10, min_depth=0.1, max_depth=10.0
            )
            
            if len(points_cam) == 0:
                return
            
            # Convert to evidence (camera frame)
            evidence_cam = rgbd_to_evidence(
                points_cam, colors, normals, covs,
                kappa_normal=10.0,  # High confidence in normals
                color_var=0.01,     # Low color noise
                alpha_mean=1.0,
                alpha_var=0.1
            )
            
            # Transform evidence to odom frame
            # Camera pose = (odom<-base) ∘ (base<-camera)
            # Use TF to look up base<-camera at this timestamp.
            # Try time-specific TF first (works for dynamic frames), then fall back to "latest"
            # which is correct for static camera extrinsics stored in /tf_static.
            T_base_camera = self.sensor_io._lookup_transform(  # pylint: disable=protected-access
                self.sensor_io.config["base_frame"],
                camera_frame,
                stamp,
            )
            if T_base_camera is None:
                T_base_camera = self.sensor_io._lookup_transform(  # pylint: disable=protected-access
                    self.sensor_io.config["base_frame"],
                    camera_frame,
                    Time(),  # latest
                )
            if T_base_camera is None:
                return

            T_odom_camera = se3_compose(pose_odom_base, T_base_camera)
            
            evidence_odom = transform_evidence_to_global(evidence_cam, T_odom_camera)
            
            # Spatial subsampling to reduce message size
            max_pts = int(self.get_parameter("rgbd_max_points_per_msg").value)
            evidence_subsampled = subsample_evidence_spatially(evidence_odom, grid_size=0.1, max_points=max_pts)
            
            if len(evidence_subsampled) == 0:
                return
            
            # Serialize to JSON
            evidence_json = []
            for ev in evidence_subsampled:
                evidence_json.append({
                    "position_L": ev["position_L"].tolist(),
                    "position_h": ev["position_h"].tolist(),
                    "color_L": ev["color_L"].tolist(),
                    "color_h": ev["color_h"].tolist(),
                    "normal_theta": ev["normal_theta"].tolist(),
                    "alpha_mean": float(ev["alpha_mean"]),
                    "alpha_var": float(ev["alpha_var"]),
                })
            
            payload = {"evidence": evidence_json}
            msg = String()
            msg.data = json.dumps(payload)
            self.pub_rgbd.publish(msg)
            
            self.get_logger().debug(
                f"Published {len(evidence_subsampled)} RGB-D evidence with true colors/normals",
                throttle_duration_sec=2.0
            )
        except Exception as e:
            self.get_logger().warn(
                f"RGB-D evidence publish failed: {e}",
                throttle_duration_sec=5.0,
            )
    
    def _publish_report(self, report: OpReport):
        """Publish OpReport as JSON."""
        report.validate()
        msg = String()
        msg.data = report.to_json()
        self.pub_report.publish(msg)
    
    def _publish_status(self):
        """Publish frontend status (observability)."""
        status = self.status_monitor.get_status_dict()
        status["anchors"] = self.anchor_manager.get_anchor_count()
        
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)


def main():
    rclpy.init()
    node = Frontend()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
