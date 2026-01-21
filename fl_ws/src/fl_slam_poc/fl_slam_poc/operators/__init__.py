"""
FL-SLAM Operators.

This module contains operator implementations:
- dirichlet_geom: Dirichlet geometry operators (Frobenius correction, etc.)
- imu_preintegration: IMU preintegration operator (Forster et al. 2017)

Import directly from submodules:
    from fl_slam_poc.operators.dirichlet_geom import third_order_correct
    from fl_slam_poc.operators.imu_preintegration import IMUPreintegrator
"""

from fl_slam_poc.operators.dirichlet_geom import (
    dirichlet_log_partition,
    psi_potential,
    g_fisher,
    c_contract_uv,
    frob_product,
    third_order_correct,
    target_E_log_p_from_mixture,
    residual_f,
    iproject_dirichlet_from_mixture,
)

from fl_slam_poc.operators.imu_preintegration import IMUPreintegrator

__all__ = [
    # Dirichlet geometry
    "dirichlet_log_partition",
    "psi_potential",
    "g_fisher",
    "c_contract_uv",
    "frob_product",
    "third_order_correct",
    "target_E_log_p_from_mixture",
    "residual_f",
    "iproject_dirichlet_from_mixture",
    # IMU preintegration
    "IMUPreintegrator",
]
