"""
Geometric Compositional SLAM v2 Operators.

All operators are total functions that always run.
Each returns (result, CertBundle, ExpectedEffect).
No gates, no conditional branching.

Reference: docs/GC_SLAM.md Section 5
"""

from fl_slam_poc.backend.operators.point_budget import (
    point_budget_resample,
    PointBudgetResult,
)

from fl_slam_poc.backend.operators.predict import (
    predict_diffusion,
)

from fl_slam_poc.backend.operators.imu_preintegration import (
    smooth_window_weights,
    preintegrate_imu_relative_pose_jax,
)

from fl_slam_poc.backend.operators.deskew_constant_twist import (
    deskew_constant_twist,
    DeskewConstantTwistResult,
)

from fl_slam_poc.backend.operators.kappa import (
    kappa_from_resultant_v2,
    KappaResult,
)

from fl_slam_poc.backend.operators.odom_evidence import (
    odom_quadratic_evidence,
    OdomEvidenceResult,
)

from fl_slam_poc.backend.operators.odom_twist_evidence import (
    odom_velocity_evidence,
    odom_yawrate_evidence,
    pose_twist_kinematic_consistency,
)

from fl_slam_poc.backend.operators.imu_evidence import (
    imu_vmf_gravity_evidence,
    imu_vmf_gravity_evidence_time_resolved,
    ImuEvidenceResult,
    TimeResolvedImuResult,
)

from fl_slam_poc.backend.operators.fusion import (
    fusion_scale_from_certificates,
    info_fusion_additive,
    FusionScaleResult,
)

from fl_slam_poc.backend.operators.recompose import (
    pose_update_frobenius_recompose,
    RecomposeResult,
)

from fl_slam_poc.backend.operators.measurement_noise_iw_jax import (
    imu_gyro_meas_iw_suffstats_from_avg_rate_jax,
    imu_accel_meas_iw_suffstats_from_gravity_dir_jax,
    measurement_noise_mean_jax,
    measurement_noise_apply_suffstats_jax,
)

from fl_slam_poc.backend.operators.planar_prior import (
    planar_z_prior,
    velocity_z_prior,
)

from fl_slam_poc.backend.operators.imu_gyro_evidence import (
    imu_gyro_rotation_evidence,
)

from fl_slam_poc.backend.operators.imu_preintegration_factor import (
    imu_preintegration_factor,
)

from fl_slam_poc.backend.operators.excitation import (
    compute_excitation_scales_jax,
    apply_excitation_prior_scaling_jax,
)

from fl_slam_poc.backend.operators.inverse_wishart_jax import (
    process_noise_iw_suffstats_from_info_jax,
    process_noise_state_to_Q_jax,
    process_noise_iw_apply_suffstats_jax,
)

from fl_slam_poc.backend.operators.lidar_surfel_extraction import (
    extract_lidar_surfels,
    SurfelExtractionConfig,
)

from fl_slam_poc.backend.operators.primitive_association import (
    associate_primitives_ot,
    AssociationConfig,
    block_associations_for_fuse,
    PrimitiveAssociationResult,
)

from fl_slam_poc.backend.operators.visual_pose_evidence import (
    visual_pose_evidence,
    build_visual_pose_evidence_22d,
)

from fl_slam_poc.backend.operators.anchor_drift import (
    anchor_drift_update,
    AnchorDriftResult,
)

from fl_slam_poc.backend.operators.hypothesis import (
    hypothesis_barycenter_projection,
    HypothesisProjectionResult,
)


__all__ = [
    # Point budget
    "point_budget_resample",
    "PointBudgetResult",
    # Predict
    "predict_diffusion",
    # IMU preintegration
    "smooth_window_weights",
    "preintegrate_imu_relative_pose_jax",
    # Deskew
    "deskew_constant_twist",
    "DeskewConstantTwistResult",
    # Kappa
    "kappa_from_resultant_v2",
    "KappaResult",
    # Odom evidence
    "odom_quadratic_evidence",
    "OdomEvidenceResult",
    "odom_velocity_evidence",
    "odom_yawrate_evidence",
    "pose_twist_kinematic_consistency",
    # IMU evidence
    "imu_vmf_gravity_evidence",
    "imu_vmf_gravity_evidence_time_resolved",
    "ImuEvidenceResult",
    "TimeResolvedImuResult",
    # Fusion
    "fusion_scale_from_certificates",
    "info_fusion_additive",
    "FusionScaleResult",
    # Recompose
    "pose_update_frobenius_recompose",
    "RecomposeResult",
    # Measurement noise IW
    "imu_gyro_meas_iw_suffstats_from_avg_rate_jax",
    "imu_accel_meas_iw_suffstats_from_gravity_dir_jax",
    "measurement_noise_mean_jax",
    "measurement_noise_apply_suffstats_jax",
    # Planar priors
    "planar_z_prior",
    "velocity_z_prior",
    # Gyro evidence
    "imu_gyro_rotation_evidence",
    # IMU preintegration factor
    "imu_preintegration_factor",
    # Excitation
    "compute_excitation_scales_jax",
    "apply_excitation_prior_scaling_jax",
    # Process noise IW
    "process_noise_iw_suffstats_from_info_jax",
    "process_noise_state_to_Q_jax",
    "process_noise_iw_apply_suffstats_jax",
    # LiDAR surfels
    "extract_lidar_surfels",
    "SurfelExtractionConfig",
    # Primitive association
    "associate_primitives_ot",
    "AssociationConfig",
    "block_associations_for_fuse",
    "PrimitiveAssociationResult",
    # Visual pose evidence
    "visual_pose_evidence",
    "build_visual_pose_evidence_22d",
    # Anchor drift
    "anchor_drift_update",
    "AnchorDriftResult",
    # Hypothesis
    "hypothesis_barycenter_projection",
    "HypothesisProjectionResult",
]
