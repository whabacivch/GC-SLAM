"""
FL-SLAM Operators - Legacy compatibility module.

This module exists to preserve older import paths (e.g. `fl_slam_poc.operators.*`)
while avoiding eager imports that bloat the MVP runtime closure.

New code should import from specific subpackages directly:
- `fl_slam_poc.backend.fusion.*`
- `fl_slam_poc.frontend.loops.*`
- `fl_slam_poc.common.*`
- `fl_slam_poc.operators.dirichlet_geom` (Dirichlet geometry; experimental)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # Gaussian info
    "make_evidence",
    "fuse_info",
    "mean_cov",
    "log_partition",
    "kl_divergence",
    "hellinger_distance",
    "bhattacharyya_coefficient",
    "fisher_information",
    "natural_gradient",
    "marginalize",
    "condition",
    "product_of_experts",
    # Dirichlet
    "dirichlet_log_partition",
    "psi_potential",
    "g_fisher",
    "c_contract_uv",
    "frob_product",
    "third_order_correct",
    "target_E_log_p_from_mixture",
    "residual_f",
    "iproject_dirichlet_from_mixture",
    # Information distances
    "hellinger_sq_expfam",
    "hellinger_expfam",
    "hellinger_gaussian",
    "fisher_rao_gaussian_1d",
    "fisher_rao_student_t",
    "fisher_rao_student_t_vec",
    "fisher_rao_spd",
    "product_distance",
    "product_distance_weighted",
    "gaussian_kl",
    "gaussian_kl_symmetric",
    "wishart_bregman",
    "bhattacharyya_coefficient_gaussian",
    "bhattacharyya_distance_gaussian",
    # ICP
    "ICPResult",
    "best_fit_se3",
    "icp_3d",
    "icp_information_weight",
    "icp_covariance_tangent",
    "transport_covariance_to_frame",
    "N_MIN_SE3_DOF",
    "K_SIGMOID",
    # Gaussian Frobenius
    "gaussian_frobenius_correction",
    # vMF geometry
    "A_d",
    "A_d_inverse_series",
    "vmf_make_evidence",
    "vmf_mean_param",
    "vmf_barycenter",
    "vmf_fisher_rao_distance",
    "vmf_third_order_correction",
    "vmf_hellinger_distance",
    # GPU point cloud processing
    "GPUPointCloudProcessor",
    "is_gpu_available",
    "voxel_filter_gpu",
    "icp_gpu",
    # OpReport
    "OpReport",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Gaussian info
    "make_evidence": ("fl_slam_poc.backend.fusion.gaussian_info", "make_evidence"),
    "fuse_info": ("fl_slam_poc.backend.fusion.gaussian_info", "fuse_info"),
    "mean_cov": ("fl_slam_poc.backend.fusion.gaussian_info", "mean_cov"),
    "log_partition": ("fl_slam_poc.backend.fusion.gaussian_info", "log_partition"),
    "kl_divergence": ("fl_slam_poc.backend.fusion.gaussian_info", "kl_divergence"),
    "hellinger_distance": ("fl_slam_poc.backend.fusion.gaussian_info", "hellinger_distance"),
    "bhattacharyya_coefficient": ("fl_slam_poc.backend.fusion.gaussian_info", "bhattacharyya_coefficient"),
    "fisher_information": ("fl_slam_poc.backend.fusion.gaussian_info", "fisher_information"),
    "natural_gradient": ("fl_slam_poc.backend.fusion.gaussian_info", "natural_gradient"),
    "marginalize": ("fl_slam_poc.backend.fusion.gaussian_info", "marginalize"),
    "condition": ("fl_slam_poc.backend.fusion.gaussian_info", "condition"),
    "product_of_experts": ("fl_slam_poc.backend.fusion.gaussian_info", "product_of_experts"),
    # Dirichlet geometry (experimental)
    "dirichlet_log_partition": ("fl_slam_poc.operators.dirichlet_geom", "dirichlet_log_partition"),
    "psi_potential": ("fl_slam_poc.operators.dirichlet_geom", "psi_potential"),
    "g_fisher": ("fl_slam_poc.operators.dirichlet_geom", "g_fisher"),
    "c_contract_uv": ("fl_slam_poc.operators.dirichlet_geom", "c_contract_uv"),
    "frob_product": ("fl_slam_poc.operators.dirichlet_geom", "frob_product"),
    "third_order_correct": ("fl_slam_poc.operators.dirichlet_geom", "third_order_correct"),
    "target_E_log_p_from_mixture": ("fl_slam_poc.operators.dirichlet_geom", "target_E_log_p_from_mixture"),
    "residual_f": ("fl_slam_poc.operators.dirichlet_geom", "residual_f"),
    "iproject_dirichlet_from_mixture": ("fl_slam_poc.operators.dirichlet_geom", "iproject_dirichlet_from_mixture"),
    # Information distances
    "hellinger_sq_expfam": ("fl_slam_poc.backend.fusion.information_distances", "hellinger_sq_expfam"),
    "hellinger_expfam": ("fl_slam_poc.backend.fusion.information_distances", "hellinger_expfam"),
    "hellinger_gaussian": ("fl_slam_poc.backend.fusion.information_distances", "hellinger_gaussian"),
    "fisher_rao_gaussian_1d": ("fl_slam_poc.backend.fusion.information_distances", "fisher_rao_gaussian_1d"),
    "fisher_rao_student_t": ("fl_slam_poc.backend.fusion.information_distances", "fisher_rao_student_t"),
    "fisher_rao_student_t_vec": ("fl_slam_poc.backend.fusion.information_distances", "fisher_rao_student_t_vec"),
    "fisher_rao_spd": ("fl_slam_poc.backend.fusion.information_distances", "fisher_rao_spd"),
    "product_distance": ("fl_slam_poc.backend.fusion.information_distances", "product_distance"),
    "product_distance_weighted": ("fl_slam_poc.backend.fusion.information_distances", "product_distance_weighted"),
    "gaussian_kl": ("fl_slam_poc.backend.fusion.information_distances", "gaussian_kl"),
    "gaussian_kl_symmetric": ("fl_slam_poc.backend.fusion.information_distances", "gaussian_kl_symmetric"),
    "wishart_bregman": ("fl_slam_poc.backend.fusion.information_distances", "wishart_bregman"),
    "bhattacharyya_coefficient_gaussian": (
        "fl_slam_poc.backend.fusion.information_distances",
        "bhattacharyya_coefficient_gaussian",
    ),
    "bhattacharyya_distance_gaussian": (
        "fl_slam_poc.backend.fusion.information_distances",
        "bhattacharyya_distance_gaussian",
    ),
    # ICP (frontend/loops)
    "ICPResult": ("fl_slam_poc.frontend.loops.icp", "ICPResult"),
    "best_fit_se3": ("fl_slam_poc.frontend.loops.icp", "best_fit_se3"),
    "icp_3d": ("fl_slam_poc.frontend.loops.icp", "icp_3d"),
    "icp_information_weight": ("fl_slam_poc.frontend.loops.icp", "icp_information_weight"),
    "icp_covariance_tangent": ("fl_slam_poc.frontend.loops.icp", "icp_covariance_tangent"),
    "transport_covariance_to_frame": ("fl_slam_poc.frontend.loops.icp", "transport_covariance_to_frame"),
    "N_MIN_SE3_DOF": ("fl_slam_poc.frontend.loops.icp", "N_MIN_SE3_DOF"),
    "K_SIGMOID": ("fl_slam_poc.frontend.loops.icp", "K_SIGMOID"),
    # Gaussian Frobenius
    "gaussian_frobenius_correction": ("fl_slam_poc.backend.fusion.gaussian_geom", "gaussian_frobenius_correction"),
    # vMF geometry (frontend/loops)
    "A_d": ("fl_slam_poc.frontend.loops.vmf_geometry", "A_d"),
    "A_d_inverse_series": ("fl_slam_poc.frontend.loops.vmf_geometry", "A_d_inverse_series"),
    "vmf_make_evidence": ("fl_slam_poc.frontend.loops.vmf_geometry", "vmf_make_evidence"),
    "vmf_mean_param": ("fl_slam_poc.frontend.loops.vmf_geometry", "vmf_mean_param"),
    "vmf_barycenter": ("fl_slam_poc.frontend.loops.vmf_geometry", "vmf_barycenter"),
    "vmf_fisher_rao_distance": ("fl_slam_poc.frontend.loops.vmf_geometry", "vmf_fisher_rao_distance"),
    "vmf_third_order_correction": ("fl_slam_poc.frontend.loops.vmf_geometry", "vmf_third_order_correction"),
    "vmf_hellinger_distance": ("fl_slam_poc.frontend.loops.vmf_geometry", "vmf_hellinger_distance"),
    # GPU point cloud processing
    "GPUPointCloudProcessor": ("fl_slam_poc.frontend.loops.pointcloud_gpu", "GPUPointCloudProcessor"),
    "is_gpu_available": ("fl_slam_poc.frontend.loops.pointcloud_gpu", "is_gpu_available"),
    "voxel_filter_gpu": ("fl_slam_poc.frontend.loops.pointcloud_gpu", "voxel_filter_gpu"),
    "icp_gpu": ("fl_slam_poc.frontend.loops.pointcloud_gpu", "icp_gpu"),
    # IMU preintegration
    "IMUPreintegrator": ("fl_slam_poc.operators.imu_preintegration", "IMUPreintegrator"),
    # OpReport
    "OpReport": ("fl_slam_poc.common.op_report", "OpReport"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))
