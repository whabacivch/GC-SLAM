"""
Backend fusion operators for FL-SLAM.

This package intentionally avoids eager imports to keep the MVP runtime
dependency surface small. Import specific submodules (recommended), or rely on
lazy attribute access via this package for backwards compatibility.
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
    "mixture_moment_match",
    "embed_info_form",
    "hellinger_squared_from_moments",
    # Gaussian geom
    "gaussian_frobenius_correction",
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
    # Lie operators (JAX - no NumPy fallback)
    "skew",
    "so3_exp",
    "so3_log",
    "se3_plus",
    "se3_minus",
    "so3_adjoint",
    "se3_adjoint",
    "so3_exp_batch",
    "so3_log_batch",
    "se3_plus_batch",
    "se3_minus_batch",
    # IMU kernel (JAX)
    "imu_batched_projection_kernel",
    "unscented_sigma_points",
    "imu_prediction_residual",
    "hellinger_squared_gaussian",
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
    "mixture_moment_match": ("fl_slam_poc.backend.fusion.gaussian_info", "mixture_moment_match"),
    "embed_info_form": ("fl_slam_poc.backend.fusion.gaussian_info", "embed_info_form"),
    "hellinger_squared_from_moments": ("fl_slam_poc.backend.fusion.gaussian_info", "hellinger_squared_from_moments"),
    # Gaussian geom
    "gaussian_frobenius_correction": ("fl_slam_poc.backend.fusion.gaussian_geom", "gaussian_frobenius_correction"),
    # Lie operators (JAX - no NumPy fallback)
    "skew": ("fl_slam_poc.backend.fusion.lie_jax", "skew"),
    "so3_exp": ("fl_slam_poc.backend.fusion.lie_jax", "so3_exp"),
    "so3_log": ("fl_slam_poc.backend.fusion.lie_jax", "so3_log"),
    "se3_plus": ("fl_slam_poc.backend.fusion.lie_jax", "se3_plus"),
    "se3_minus": ("fl_slam_poc.backend.fusion.lie_jax", "se3_minus"),
    "so3_adjoint": ("fl_slam_poc.backend.fusion.lie_jax", "so3_adjoint"),
    "se3_adjoint": ("fl_slam_poc.backend.fusion.lie_jax", "se3_adjoint"),
    "so3_exp_batch": ("fl_slam_poc.backend.fusion.lie_jax", "so3_exp_batch"),
    "so3_log_batch": ("fl_slam_poc.backend.fusion.lie_jax", "so3_log_batch"),
    "se3_plus_batch": ("fl_slam_poc.backend.fusion.lie_jax", "se3_plus_batch"),
    "se3_minus_batch": ("fl_slam_poc.backend.fusion.lie_jax", "se3_minus_batch"),
    # IMU kernel (JAX)
    "imu_batched_projection_kernel": ("fl_slam_poc.backend.fusion.imu_jax_kernel", "imu_batched_projection_kernel"),
    "unscented_sigma_points": ("fl_slam_poc.backend.fusion.imu_jax_kernel", "unscented_sigma_points"),
    "imu_prediction_residual": ("fl_slam_poc.backend.fusion.imu_jax_kernel", "imu_prediction_residual"),
    "hellinger_squared_gaussian": ("fl_slam_poc.backend.fusion.imu_jax_kernel", "hellinger_squared_gaussian"),
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
