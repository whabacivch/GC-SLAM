"""
Inverse-Wishart state containers (arrays-only) for Golden Child SLAM v2.

This module defines JAX-pytree-friendly containers for adaptive noise.
All structures here are arrays-only and safe to pass through @jax.jit codepaths.

Process noise (Q) is maintained as blockwise InvWishart states over the 22D tangent blocks.
"""

from __future__ import annotations

from typing import NamedTuple

from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common import constants as C


# Process-noise blocks in the 22D tangent ordering:
# [trans(3), rot(3), vel(3), bg(3), ba(3), dt(1), ex(6)]
PROCESS_BLOCK_DIMS = jnp.array([3, 3, 3, 3, 3, 1, 6], dtype=jnp.int32)  # (7,)
PROCESS_BLOCK_STARTS = jnp.array([0, 3, 6, 9, 12, 15, 16], dtype=jnp.int32)  # (7,)

# Fixed (7,6,6) masks selecting each padded block's active submatrix.
_rows = (jnp.arange(6, dtype=jnp.int32)[None, :] < PROCESS_BLOCK_DIMS[:, None])  # (7,6) bool
PROCESS_BLOCK_MASKS = (jnp.logical_and(_rows[:, :, None], _rows[:, None, :])).astype(jnp.float64)  # (7,6,6)


class ProcessNoiseIWState(NamedTuple):
    """
    Arrays-only InvWishart state for process noise.

    Each block i maintains:
      Sigma_i ~ InvWishart(Psi_i, nu_i)

    Stored as padded 6x6 blocks for fixed-shape vectorized ops.
    """

    nu: jnp.ndarray  # (7,) float64
    Psi_blocks: jnp.ndarray  # (7, 6, 6) float64 (padded; zeros outside block dim)
    block_dims: jnp.ndarray  # (7,) int32


def create_datasheet_process_noise_state() -> ProcessNoiseIWState:
    """
    Initialize process-noise IW states from constants.py (Contract 4 compliant).

    We choose nu so the IW mean exists and equals the intended prior:
      nu = p + 1 + nu_extra
      E[Sigma] = Psi / (nu - p - 1) = Psi / nu_extra

    With nu_extra = GC_IW_NU_WEAK_ADD (default 0.5), set Psi = Sigma_prior * nu_extra.
    """
    block_dims = PROCESS_BLOCK_DIMS
    p = block_dims.astype(jnp.float64)
    nu_extra = jnp.array(C.GC_IW_NU_WEAK_ADD, dtype=jnp.float64)
    nu = p + 1.0 + nu_extra  # (7,)

    # Per-block diffusion-rate priors (z^2 / s), compatible with `cov += dt * Q`.
    # NOTE: these are weak priors; the IW update adapts them from innovations.
    sigma_diag = jnp.array(
        [
            C.GC_PROCESS_TRANS_DIFFUSION,      # trans (m^2 / s)
            C.GC_PROCESS_ROT_DIFFUSION,        # rot   (rad^2 / s)
            C.GC_PROCESS_VEL_DIFFUSION,        # vel   (m^2 / s^3)
            C.GC_PROCESS_BG_DIFFUSION,         # bg
            C.GC_PROCESS_BA_DIFFUSION,         # ba
            C.GC_PROCESS_DT_DIFFUSION,         # dt
            C.GC_PROCESS_EXTRINSIC_DIFFUSION,  # ex (6D)
        ],
        dtype=jnp.float64,
    )

    Psi_blocks = jnp.zeros((7, 6, 6), dtype=jnp.float64)
    for i in range(7):
        dim_i = int(block_dims[i])
        Sigma_i = jnp.eye(dim_i, dtype=jnp.float64) * sigma_diag[i]
        Psi_i = Sigma_i * nu_extra  # Psi = Sigma_prior * (nu - p - 1)
        Psi_blocks = Psi_blocks.at[i, :dim_i, :dim_i].set(Psi_i)

    return ProcessNoiseIWState(nu=nu, Psi_blocks=Psi_blocks, block_dims=block_dims)
