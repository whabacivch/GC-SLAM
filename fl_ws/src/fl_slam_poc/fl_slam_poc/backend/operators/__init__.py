"""
Golden Child SLAM v2 Operators.

All operators are total functions that always run.
Each returns (result, CertBundle, ExpectedEffect).
No gates, no conditional branching.

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5
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

from fl_slam_poc.backend.operators.binning import (
    bin_soft_assign,
    scan_bin_moment_match,
    create_bin_atlas,
    BinSoftAssignResult,
    ScanBinStats,
)

from fl_slam_poc.backend.operators.kappa import (
    kappa_from_resultant_v2,
    KappaResult,
)

from fl_slam_poc.backend.operators.odom_evidence import (
    odom_quadratic_evidence,
    OdomEvidenceResult,
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

from fl_slam_poc.backend.operators.map_update import (
    pos_cov_inflation_pushforward,
    MapUpdateResult,
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
    # Binning
    "bin_soft_assign",
    "scan_bin_moment_match",
    "create_bin_atlas",
    "BinSoftAssignResult",
    "ScanBinStats",
    # Kappa
    "kappa_from_resultant_v2",
    "KappaResult",
    # Odom evidence
    "odom_quadratic_evidence",
    "OdomEvidenceResult",
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
    # Map update
    "pos_cov_inflation_pushforward",
    "MapUpdateResult",
    # Anchor drift
    "anchor_drift_update",
    "AnchorDriftResult",
    # Hypothesis
    "hypothesis_barycenter_projection",
    "HypothesisProjectionResult",
]
