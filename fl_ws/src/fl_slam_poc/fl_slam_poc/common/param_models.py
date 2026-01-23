"""Pydantic parameter models for FL-SLAM nodes."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from fl_slam_poc.common import constants


class BaseSLAMParams(BaseModel):
    """Shared parameter base with common priors and alignment settings."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    use_sim_time: bool = False
    alignment_sigma_prior: float = Field(0.1, gt=0.0)
    alignment_prior_strength: float = Field(5.0, gt=0.0)
    alignment_sigma_floor: float = Field(0.001, gt=0.0)

    gravity: List[float] = Field(default_factory=lambda: list(constants.GRAVITY_DEFAULT), min_length=3, max_length=3)

    imu_gyro_noise_density: float = Field(constants.IMU_GYRO_NOISE_DENSITY_DEFAULT, gt=0.0)
    imu_accel_noise_density: float = Field(constants.IMU_ACCEL_NOISE_DENSITY_DEFAULT, gt=0.0)


class FrontendParams(BaseSLAMParams):
    """Frontend node parameter model."""

    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    odom_is_delta: bool = False
    camera_topic: str = "/camera/image_raw"
    depth_topic: str = "/camera/depth/image_raw"
    camera_info_topic: str = "/camera/depth/camera_info"
    enable_image: bool = True
    enable_depth: bool = True
    enable_camera_info: bool = True

    camera_fx: float = 0.0
    camera_fy: float = 0.0
    camera_cx: float = 0.0
    camera_cy: float = 0.0

    publish_rgbd_evidence: bool = True
    rgbd_evidence_topic: str = "/sim/rgbd_evidence"
    rgbd_publish_every_n_scans: int = Field(5, ge=1)
    rgbd_max_points_per_msg: int = Field(500, ge=1)
    rgbd_sync_max_dt_sec: float = Field(0.1, gt=0.0)
    rgbd_min_depth_m: float = Field(constants.DEPTH_MIN_VALID, gt=0.0)
    rgbd_max_depth_m: float = Field(constants.DEPTH_MAX_VALID, gt=0.0)
    rgbd_spatial_grid_size: float = Field(constants.RGBD_SPATIAL_GRID_SIZE, gt=0.0)
    rgbd_kappa_normal: float = Field(constants.RGBD_KAPPA_NORMAL_DEFAULT, gt=0.0)
    rgbd_color_variance: float = Field(constants.RGBD_COLOR_VARIANCE_DEFAULT, gt=0.0)
    rgbd_alpha_mean: float = Field(constants.RGBD_ALPHA_MEAN_DEFAULT, gt=0.0)
    rgbd_alpha_var: float = Field(constants.RGBD_ALPHA_VAR_DEFAULT, gt=0.0)
    rgbd_rng_seed: int = Field(-1, ge=-1)

    descriptor_bins: int = Field(constants.DESCRIPTOR_BINS_DEFAULT, ge=1)
    anchor_budget: int = Field(0, ge=0)
    loop_budget: int = Field(0, ge=0)
    anchor_id_offset: int = Field(0, ge=0)
    anchor_create_max_points: int = Field(constants.ANCHOR_CREATE_MAX_POINTS, ge=1)
    depth_points_fallback_max_points: int = Field(constants.DEPTH_POINTS_FALLBACK_MAX, ge=1)

    icp_max_iter_prior: int = Field(constants.ICP_MAX_ITER_DEFAULT, ge=1)
    icp_tol_prior: float = Field(constants.ICP_TOLERANCE_DEFAULT, gt=0.0)
    icp_prior_strength: float = Field(10.0, gt=0.0)
    icp_n_ref: float = Field(constants.ICP_N_REF_DEFAULT, gt=0.0)
    icp_sigma_mse: float = Field(0.2, gt=0.0)

    depth_stride: int = Field(constants.DEPTH_STRIDE_DEFAULT, ge=1)
    feature_buffer_len: int = Field(constants.FEATURE_BUFFER_MAX_LENGTH, ge=1)
    sensor_qos_reliability: str = constants.QOS_IMU_RELIABILITY_DEFAULT
    imu_qos_reliability: str = constants.QOS_IMU_RELIABILITY_DEFAULT

    birth_intensity: float = Field(constants.BIRTH_INTENSITY_DEFAULT, gt=0.0)
    scan_period: float = Field(constants.SCAN_PERIOD_DEFAULT, gt=0.0)
    base_component_weight: float = Field(constants.BASE_COMPONENT_WEIGHT_DEFAULT, gt=0.0)
    birth_rng_seed: int = Field(-1, ge=-1)

    enable_imu: bool = False
    imu_topic: str = constants.IMU_TOPIC_DEFAULT
    imu_accel_scale: float = Field(constants.IMU_ACCEL_SCALE_DEFAULT, gt=0.0)
    keyframe_translation_threshold: float = Field(constants.KEYFRAME_TRANSLATION_THRESHOLD_DEFAULT, ge=0.0)
    keyframe_rotation_threshold: float = Field(constants.KEYFRAME_ROTATION_THRESHOLD_DEFAULT, ge=0.0)
    imu_min_measurements_publish: int = Field(constants.IMU_MIN_MEASUREMENTS_PUBLISH, ge=1)
    imu_min_measurements_warning: int = Field(constants.IMU_MIN_MEASUREMENTS_WARNING, ge=1)
    imu_buffer_size_warning: int = Field(constants.IMU_BUFFER_SIZE_WARNING, ge=1)

    odom_frame: str = "odom"
    base_frame: str = "base_link"
    camera_frame: str = "camera_link"
    scan_frame: str = "base_link"
    tf_timeout_sec: float = Field(0.05, gt=0.0)
    status_publish_interval_sec: float = Field(constants.STATUS_PUBLISH_INTERVAL_SEC, gt=0.0)
    lidar_base_extrinsic: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        min_length=6,
        max_length=6,
    )

    fr_distance_scale_prior: float = Field(constants.FR_DISTANCE_SCALE_PRIOR, gt=0.0)
    fr_scale_prior_strength: float = Field(constants.FR_SCALE_PRIOR_STRENGTH, gt=0.0)

    use_3d_pointcloud: bool = False
    enable_pointcloud: bool = False
    pointcloud_topic: str = constants.POINTCLOUD_TOPIC_DEFAULT
    pointcloud_range_min: float = Field(constants.DEPTH_MIN_VALID, ge=0.0)
    pointcloud_range_max: float = Field(50.0, gt=0.0)

    voxel_size: float = Field(constants.VOXEL_SIZE_DEFAULT, gt=0.0)
    max_points_after_filter: int = Field(constants.MAX_POINTS_AFTER_FILTER, ge=1)
    min_points_for_icp: int = Field(constants.MIN_POINTS_FOR_3D_ICP, ge=1)
    icp_max_correspondence_distance: float = Field(constants.ICP_MAX_CORRESPONDENCE_DIST_DEFAULT, gt=0.0)
    normal_estimation_radius: float = Field(constants.NORMAL_ESTIMATION_RADIUS, gt=0.0)
    pointcloud_rate_limit_hz: float = Field(constants.POINTCLOUD_RATE_LIMIT_HZ, gt=0.0)

    use_gpu: bool = False
    gpu_device_index: int = Field(constants.CUDA_DEVICE_INDEX, ge=0)
    gpu_fallback_to_cpu: bool = True


class BackendParams(BaseSLAMParams):
    """Backend node parameter model."""

    odom_frame: str = "odom"
    rgbd_evidence_topic: str = "/sim/rgbd_evidence"

    process_noise_trans_prior: float = Field(constants.PROCESS_NOISE_TRANS_PRIOR, gt=0.0)
    process_noise_rot_prior: float = Field(constants.PROCESS_NOISE_ROT_PRIOR, gt=0.0)
    process_noise_prior_strength: float = Field(constants.PROCESS_NOISE_PRIOR_STRENGTH, gt=0.0)

    # NOTE: enable_imu_fusion removed - 15D state is always enabled
    imu_gyro_random_walk: float = Field(constants.IMU_GYRO_RANDOM_WALK_DEFAULT, gt=0.0)
    imu_accel_random_walk: float = Field(constants.IMU_ACCEL_RANDOM_WALK_DEFAULT, gt=0.0)

    trajectory_export_path: str = "/tmp/fl_slam_trajectory.tum"
    trajectory_path_max_length: int = Field(constants.TRAJECTORY_PATH_MAX_LENGTH, ge=1)
    status_check_period_sec: float = Field(constants.STATUS_CHECK_PERIOD, gt=0.0)

    dense_association_radius: float = Field(constants.DENSE_ASSOCIATION_RADIUS_DEFAULT, gt=0.0)
    max_dense_modules: int = Field(constants.DENSE_MODULE_COMPUTE_BUDGET, ge=1)
    dense_module_keep_fraction: float = Field(constants.DENSE_MODULE_KEEP_FRACTION, gt=0.0, le=1.0)
    max_pending_loops_per_anchor: int = Field(constants.LOOP_PENDING_BUFFER_BUDGET, ge=1)
    max_pending_imu_per_anchor: int = Field(constants.IMU_PENDING_BUFFER_BUDGET, ge=1)
    state_buffer_max_length: int = Field(constants.STATE_BUFFER_MAX_LENGTH, ge=1)
