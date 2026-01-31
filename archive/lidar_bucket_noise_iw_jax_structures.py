"""
LiDAR (line,tag) bucket measurement-noise IW state for GC v2 (ARCHIVED).

ARCHIVED: Bin-based LiDAR evidence path removed. PrimitiveMap is canonical;
measurement noise is handled by primitive precisions / measurement_noise_iw_jax.
"""

from __future__ import annotations

from typing import NamedTuple

from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common import constants as C


class LidarBucketNoiseIWState(NamedTuple):
    """Arrays-only IW state per (line,tag) bucket."""

    nu: jnp.ndarray  # (K,) float64
    Psi: jnp.ndarray  # (K,3,3) float64


def create_datasheet_lidar_bucket_noise_state(
    lidar_sigma_meas: float | None = None,
) -> LidarBucketNoiseIWState:
    """
    Initialize K bucket IW states with the same weak prior.

    Uses: nu = p + 1 + nu_extra and Psi = Sigma_prior * nu_extra so mean exists.

    lidar_sigma_meas: optional scalar (m²) for isotropic LiDAR prior. If None, uses C.GC_LIDAR_SIGMA_MEAS.
    """
    K = int(C.GC_LIDAR_N_BUCKETS)
    p = 3.0
    nu_extra = float(C.GC_IW_NU_WEAK_ADD)
    nu0 = p + 1.0 + nu_extra

    sigma0 = float(lidar_sigma_meas if lidar_sigma_meas is not None else C.GC_LIDAR_SIGMA_MEAS)
    Sigma0 = sigma0 * jnp.eye(3, dtype=jnp.float64)
    Psi0 = Sigma0 * nu_extra

    nu = jnp.full((K,), nu0, dtype=jnp.float64)
    Psi = jnp.tile(Psi0[None, :, :], (K, 1, 1))
    return LidarBucketNoiseIWState(nu=nu, Psi=Psi)
