"""
LiDAR bucket IW operators (ARCHIVED).

ARCHIVED: Bin-based LiDAR evidence path removed. See archive/lidar_bucket_noise_iw_jax_structures.py.
"""

from __future__ import annotations

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants as C
from fl_slam_poc.common.primitives import (
    domain_projection_psd_core,
    spd_cholesky_inverse_lifted_core,
)
# Import from structures snapshot in same archive (path may need adjustment if run).
# LidarBucketNoiseIWState = from lidar_bucket_noise_iw_jax_structures
