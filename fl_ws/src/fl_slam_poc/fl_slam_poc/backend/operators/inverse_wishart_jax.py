"""
Inverse-Wishart adaptive noise operators (arrays-only JAX).

This module provides:
- Q construction from process-noise IW state (fixed cost, block-diagonal)
- commutative sufficient-statistics extraction for IW updates (per hypothesis)
- deterministic IW state update from aggregated sufficient statistics (once per scan)
"""

from __future__ import annotations

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants as C
from fl_slam_poc.common.primitives import (
    domain_projection_psd_core,
    spd_cholesky_solve_lifted_core,
    spd_cholesky_inverse_lifted_core,
)
from fl_slam_poc.backend.structures.inverse_wishart_jax import (
    ProcessNoiseIWState,
    PROCESS_BLOCK_DIMS,
    PROCESS_BLOCK_STARTS,
    PROCESS_BLOCK_MASKS,
)


@jax.jit
def _positive_part_softplus(x: jnp.ndarray, eps: float = 1e-12, beta: float = 50.0) -> jnp.ndarray:
    """Smooth projection to (0,∞): softplus(x) + eps."""
    x = jnp.asarray(x, dtype=jnp.float64)
    b = jnp.array(beta, dtype=jnp.float64)
    return (jax.nn.softplus(b * x) / b) + jnp.array(eps, dtype=jnp.float64)


@jax.jit
def process_noise_state_to_Q_jax(
    pn_state: ProcessNoiseIWState,
    eps_psd: float = C.GC_EPS_PSD,
) -> jnp.ndarray:
    """
    Assemble full 22x22 process diffusion matrix Q from blockwise IW posterior means.

    Uses the IW mean:
        E[Sigma] = Psi / (nu - p - 1)
    where nu is stored as total degrees-of-freedom.

    Returns:
        Q_psd: (22,22) PSD matrix (DomainProjectionPSD always applied).
    """
    dims_f = pn_state.block_dims.astype(jnp.float64)
    denom_raw = pn_state.nu - dims_f - 1.0
    denom = _positive_part_softplus(denom_raw, eps=1e-12)

    Q_blocks = pn_state.Psi_blocks / denom[:, None, None]  # (7,6,6)
    Q_blocks = Q_blocks * PROCESS_BLOCK_MASKS

    Q = jnp.zeros((C.GC_D_Z, C.GC_D_Z), dtype=jnp.float64)

    def place_one(i, Q_accum):
        start = PROCESS_BLOCK_STARTS[i]
        block = Q_blocks[i]  # (6,6) padded
        # Place a 6x6 patch; overlaps are safe because later blocks overwrite their own diagonal.
        return jax.lax.dynamic_update_slice(Q_accum, block, (start, start))

    Q = jax.lax.fori_loop(0, 7, place_one, Q)

    Q_psd, _cert = domain_projection_psd_core(Q, eps_psd)
    return Q_psd


@jax.jit
def process_noise_iw_suffstats_from_info_jax(
    L_pred: jnp.ndarray,
    h_pred: jnp.ndarray,
    L_post: jnp.ndarray,
    h_post: jnp.ndarray,
    eps_lift: float = C.GC_EPS_LIFT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute commutative IW sufficient statistics for process-noise updates.

    Decision (#6): use posterior expected outer product:
        r := mu_post - mu_pred
        E[rr^T] ≈ rr^T + Sigma_post

    Returns:
        dPsi: (7,6,6) padded block updates (masked)
        dnu:  (7,)   per-block dof increments (default 1.0 each)
    """
    mu_pred, _ = spd_cholesky_solve_lifted_core(L_pred, h_pred, eps_lift)
    mu_post, _ = spd_cholesky_solve_lifted_core(L_post, h_post, eps_lift)
    Sigma_post, _ = spd_cholesky_inverse_lifted_core(L_post, eps_lift)

    r = mu_post - mu_pred  # (22,)

    r_pad = jnp.zeros((7, 6), dtype=jnp.float64)
    r_pad = r_pad.at[0, :3].set(r[0:3])       # rot
    r_pad = r_pad.at[1, :3].set(r[3:6])       # trans
    r_pad = r_pad.at[2, :3].set(r[6:9])       # vel
    r_pad = r_pad.at[3, :3].set(r[9:12])      # bg
    r_pad = r_pad.at[4, :3].set(r[12:15])     # ba
    r_pad = r_pad.at[5, 0].set(r[15])         # dt
    r_pad = r_pad.at[6, :6].set(r[16:22])     # ex

    rrT = jnp.einsum("bi,bj->bij", r_pad, r_pad)  # (7,6,6)

    Sigma_blocks = jnp.zeros((7, 6, 6), dtype=jnp.float64)
    Sigma_blocks = Sigma_blocks.at[0, :3, :3].set(Sigma_post[0:3, 0:3])
    Sigma_blocks = Sigma_blocks.at[1, :3, :3].set(Sigma_post[3:6, 3:6])
    Sigma_blocks = Sigma_blocks.at[2, :3, :3].set(Sigma_post[6:9, 6:9])
    Sigma_blocks = Sigma_blocks.at[3, :3, :3].set(Sigma_post[9:12, 9:12])
    Sigma_blocks = Sigma_blocks.at[4, :3, :3].set(Sigma_post[12:15, 12:15])
    Sigma_blocks = Sigma_blocks.at[5, :1, :1].set(Sigma_post[15:16, 15:16])
    Sigma_blocks = Sigma_blocks.at[6, :6, :6].set(Sigma_post[16:22, 16:22])

    dPsi = (rrT + Sigma_blocks) * PROCESS_BLOCK_MASKS
    dnu = jnp.ones((7,), dtype=jnp.float64)
    return dPsi, dnu


@jax.jit
def process_noise_iw_apply_suffstats_jax(
    pn_state: ProcessNoiseIWState,
    dPsi: jnp.ndarray,   # (7,6,6)
    dnu: jnp.ndarray,    # (7,)
    dt_sec: float,  # Unused - kept for API compatibility, will be removed
    eps_psd: float = C.GC_EPS_PSD,
    nu_max: float = 1000.0,
) -> tuple[ProcessNoiseIWState, jnp.ndarray]:
    """
    Apply aggregated commutative sufficient statistics to update the IW state (once per scan).

    DISCRETE-TIME update (no dt division):
      Psi <- rho * Psi + dPsi
      nu  <- rho * nu  + dnu

    Rationale: dPsi is already "incremental sufficient statistics for this step" computed
    from the realized state increment. Dividing by dt would incorrectly interpret it as a
    continuous-time rate, but we don't have a continuous-time innovation model for dPsi.
    The correct place for dt-scaling is in PredictDiffusion (Sigma += Q_mode * dt_sec),
    not in the IW posterior update.

    ν is clipped to remain > p+1 (mean exists) and to a fixed nu_max.
    """
    rho = jnp.array(
        [
            C.GC_IW_RHO_ROT,
            C.GC_IW_RHO_TRANS,
            C.GC_IW_RHO_VEL,
            C.GC_IW_RHO_BG,
            C.GC_IW_RHO_BA,
            C.GC_IW_RHO_DT,
            C.GC_IW_RHO_EX,
        ],
        dtype=jnp.float64,
    )

    # Discrete-time sufficient statistics update - NO division by dt
    Psi_raw = (rho[:, None, None] * pn_state.Psi_blocks) + dPsi
    Psi_raw = Psi_raw * PROCESS_BLOCK_MASKS

    # Project each padded block to PSD for numerical stability (always applied).
    def proj_block(P):
        P_psd, cert_vec = domain_projection_psd_core(P, eps_psd)
        return P_psd, cert_vec

    Psi_psd, Psi_cert = jax.vmap(proj_block)(Psi_raw)
    psd_proj_delta = jnp.sum(Psi_cert[:, 0])

    # Update nu with retention and clipping
    nu_raw = rho * pn_state.nu + dnu
    dims_f = pn_state.block_dims.astype(jnp.float64)
    nu_min = dims_f + 1.0 + C.GC_IW_NU_WEAK_ADD  # enforce mean existence baseline
    # Smooth projection of ν to [nu_min, nu_max] (no hard clip kink).
    nu_floor = nu_min + jax.nn.softplus(nu_raw - nu_min)
    nu = jnp.array(nu_max, dtype=jnp.float64) - jax.nn.softplus(jnp.array(nu_max, dtype=jnp.float64) - nu_floor)
    nu_proj_delta = jnp.sum(jnp.abs(nu - nu_raw))

    cert_vec = jnp.array([psd_proj_delta, nu_proj_delta], dtype=jnp.float64)
    return ProcessNoiseIWState(nu=nu, Psi_blocks=Psi_psd, block_dims=pn_state.block_dims), cert_vec
