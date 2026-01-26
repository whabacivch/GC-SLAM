"""
LiDAR bucket IW operators (arrays-only JAX).

Provides:
- bucket index (ring,tag) -> [0,K)
- per-bucket posterior mean + precision + scalar reliability
- per-scan commutative sufficient-statistics updates from bin residuals apportioned to buckets
"""

from __future__ import annotations

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants as C
from fl_slam_poc.common.primitives import (
    domain_projection_psd_core,
    spd_cholesky_inverse_lifted_core,
)
from fl_slam_poc.backend.structures.lidar_bucket_noise_iw_jax import LidarBucketNoiseIWState


@jax.jit
def lidar_bucket_index_jax(ring: jnp.ndarray, tag: jnp.ndarray) -> jnp.ndarray:
    """Vectorized (ring,tag) -> bucket index (int32)."""
    ring_i = jnp.asarray(ring, dtype=jnp.int32)
    tag_i = jnp.asarray(tag, dtype=jnp.int32)
    return ring_i * jnp.int32(C.GC_LIDAR_N_TAGS) + tag_i


@jax.jit
def lidar_bucket_mean_precision_jax(
    state: LidarBucketNoiseIWState,
    eps_psd: float = C.GC_EPS_PSD,
    eps_lift: float = C.GC_EPS_LIFT,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute per-bucket Σ_mean, Λ_mean, and scalar reliability λ = tr(Λ)/3.
    """
    p = 3.0
    denom = state.nu - p - 1.0
    denom = jnp.maximum(denom, 1e-12)
    Sigma = state.Psi / denom[:, None, None]  # (K,3,3)

    def proj_and_inv(S):
        S_psd, _ = domain_projection_psd_core(S, eps_psd)
        L_inv, _lift = spd_cholesky_inverse_lifted_core(S_psd, eps_lift)
        return S_psd, L_inv

    Sigma_psd, Lambda = jax.vmap(proj_and_inv)(Sigma)  # (K,3,3), (K,3,3)
    lam = jnp.trace(Lambda, axis1=1, axis2=2) / 3.0
    return Sigma_psd, Lambda, lam


@jax.jit
def lidar_point_reliability_from_buckets_jax(
    state: LidarBucketNoiseIWState,
    ring: jnp.ndarray,
    tag: jnp.ndarray,
) -> jnp.ndarray:
    """Per-point scalar reliability λ_i from bucket precisions."""
    _Sigma, _Lambda, lam_b = lidar_bucket_mean_precision_jax(state)
    idx = lidar_bucket_index_jax(ring, tag)
    idx = jnp.clip(idx, 0, jnp.int32(C.GC_LIDAR_N_BUCKETS - 1))
    return lam_b[idx]


@jax.jit
def lidar_bucket_iw_suffstats_from_bin_residuals_jax(
    residuals_bin: jnp.ndarray,     # (B,3)
    responsibilities: jnp.ndarray,  # (N,B)
    weights: jnp.ndarray,           # (N,)
    ring: jnp.ndarray,              # (N,)
    tag: jnp.ndarray,               # (N,)
    eps_mass: float = C.GC_EPS_MASS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build commutative IW sufficient statistics for each bucket from bin translation residuals.

    Each point contributes to each bin via responsibilities; bin residual is shared.
    Bucket/bin mass:
      m[k,b] = Σ_i w_i * 1[bucket(i)=k] * r_ib

    Bucket update:
      dPsi_k = Σ_b (m[k,b]/(Σ_b m[k,b]+eps)) * (r_b r_b^T)
      dnu_k  = (Σ_b m[k,b]) / (Σ_{k,b} m[k,b] + eps)   (ESS-weighted, sums to 1)
    """
    residuals_bin = jnp.asarray(residuals_bin, dtype=jnp.float64)
    responsibilities = jnp.asarray(responsibilities, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)

    idx = lidar_bucket_index_jax(ring, tag)
    idx = jnp.clip(idx, 0, jnp.int32(C.GC_LIDAR_N_BUCKETS - 1))

    # Weighted responsibilities: (N,B)
    W = weights[:, None] * responsibilities

    # Precompute per-bin rrT: (B,3,3)
    rrT_bin = jnp.einsum("bi,bj->bij", residuals_bin, residuals_bin)

    K = int(C.GC_LIDAR_N_BUCKETS)  # Must be Python int for jnp.arange (concrete, not traced)
    B = responsibilities.shape[1]

    def one_bucket(k):
        mask = (idx == k).astype(jnp.float64)  # (N,)
        m_kb = jnp.sum(mask[:, None] * W, axis=0)  # (B,)
        m_k = jnp.sum(m_kb)
        m_kb_norm = m_kb / (m_k + eps_mass)
        dPsi_k = jnp.einsum("b,bij->ij", m_kb_norm, rrT_bin)
        dPsi_k = 0.5 * (dPsi_k + dPsi_k.T)
        dPsi_k_psd, _ = domain_projection_psd_core(dPsi_k, C.GC_EPS_PSD)
        return dPsi_k_psd, m_k

    dPsi, m = jax.vmap(one_bucket)(jnp.arange(K, dtype=jnp.int32))
    m_sum = jnp.sum(m) + eps_mass
    dnu = m / m_sum
    return dPsi, dnu


@jax.jit
def lidar_bucket_iw_apply_suffstats_jax(
    state: LidarBucketNoiseIWState,
    dPsi: jnp.ndarray,    # (K,3,3)
    dnu: jnp.ndarray,     # (K,)
    eps_psd: float = C.GC_EPS_PSD,
    nu_max: float = 1000.0,
) -> LidarBucketNoiseIWState:
    """
    Apply commutative sufficient statistics to bucket IW state (once per scan).
    """
    rho = jnp.array(C.GC_IW_RHO_MEAS_LIDAR, dtype=jnp.float64)
    Psi_raw = rho * state.Psi + dPsi
    Psi_raw = 0.5 * (Psi_raw + jnp.swapaxes(Psi_raw, -1, -2))

    def proj(P):
        P_psd, _ = domain_projection_psd_core(P, eps_psd)
        return P_psd

    Psi_psd = jax.vmap(proj)(Psi_raw)

    nu_raw = rho * state.nu + dnu
    nu_min = (3.0 + 1.0 + C.GC_IW_NU_WEAK_ADD)
    nu = jnp.clip(jnp.maximum(nu_raw, nu_min), nu_min, nu_max)
    return LidarBucketNoiseIWState(nu=nu, Psi=Psi_psd)

