"""
Bin atlas and map structures for Geometric Compositional SLAM v2 (ARCHIVED).

ARCHIVED: Not importable by fl_slam_poc. PrimitiveMap is the canonical map;
bin-based pose evidence and MapBinStats are no longer used.

Reference: docs/GC_SLAM.md Sections 4.1, 4.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.primitives import (
    domain_projection_psd_batch,
    inv_mass_core,
    _safe_normalize_jax,
)
from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_batch


# =============================================================================
# Bin Atlas
# =============================================================================


@dataclass
class BinAtlas:
    """
    Fixed bin atlas for directional binning.

    Contains B_BINS unit vectors uniformly distributed on S^2.
    """
    dirs: jnp.ndarray  # (B_BINS, 3) unit vectors


def _create_fibonacci_atlas_jax(n_bins: int) -> jnp.ndarray:
    """
    Create Fibonacci lattice on sphere using JAX.

    Produces quasi-uniform distribution of points on S^2.
    Note: Not JIT compiled since this is only called once at init time.
    """
    indices = jnp.arange(n_bins, dtype=jnp.float64) + 0.5
    phi = jnp.arccos(1 - 2 * indices / n_bins)
    theta = jnp.pi * (1 + jnp.sqrt(5)) * indices

    x = jnp.sin(phi) * jnp.cos(theta)
    y = jnp.sin(phi) * jnp.sin(theta)
    z = jnp.cos(phi)

    dirs = jnp.stack([x, y, z], axis=1)

    # Normalize to ensure unit vectors
    norms = jnp.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (norms + constants.GC_EPS_MASS)

    return dirs


def create_fibonacci_atlas(n_bins: int = constants.GC_B_BINS) -> BinAtlas:
    """
    Create fixed bin atlas with Fibonacci lattice directions.

    Args:
        n_bins: Number of bins (default from constants)

    Returns:
        BinAtlas with uniformly distributed directions
    """
    dirs = _create_fibonacci_atlas_jax(n_bins)
    return BinAtlas(dirs=dirs)


# =============================================================================
# Map Bin Statistics
# =============================================================================


@dataclass
class MapBinStats:
    """
    Map bin sufficient statistics (additive, associative).

    All statistics are stored as sufficient statistics that can be
    additively updated without losing information.

    Reference: docs/GC_SLAM.md Section 4.2
    """
    # Directional sufficient stats
    S_dir: jnp.ndarray  # (B_BINS, 3) resultant vectors Σ w u
    S_dir_scatter: jnp.ndarray  # (B_BINS, 3, 3) directional scatter Σ w u u^T
    N_dir: jnp.ndarray  # (B_BINS,) mass / ESS

    # Spatial sufficient stats
    N_pos: jnp.ndarray  # (B_BINS,)
    sum_p: jnp.ndarray  # (B_BINS, 3) Σ w p
    sum_ppT: jnp.ndarray  # (B_BINS, 3, 3) Σ w p p^T


@dataclass
class MapDerivedStats:
    """
    Derived statistics from map sufficient stats (single pytree for JIT).

    All fields are JAX arrays; no host pulls inside the JIT'd core.
    """
    mu_dir: jnp.ndarray   # (B_BINS, 3) mean directions
    kappa: jnp.ndarray   # (B_BINS,) concentration parameters
    centroid: jnp.ndarray  # (B_BINS, 3) centroids
    Sigma_c: jnp.ndarray  # (B_BINS, 3, 3) centroid covariances


def create_empty_map_stats(n_bins: int = constants.GC_B_BINS) -> MapBinStats:
    """
    Create empty map statistics (zero initialized).

    Args:
        n_bins: Number of bins

    Returns:
        Empty MapBinStats
    """
    return MapBinStats(
        S_dir=jnp.zeros((n_bins, 3), dtype=jnp.float64),
        S_dir_scatter=jnp.zeros((n_bins, 3, 3), dtype=jnp.float64),
        N_dir=jnp.zeros(n_bins, dtype=jnp.float64),
        N_pos=jnp.zeros(n_bins, dtype=jnp.float64),
        sum_p=jnp.zeros((n_bins, 3), dtype=jnp.float64),
        sum_ppT=jnp.zeros((n_bins, 3, 3), dtype=jnp.float64),
    )


def update_map_stats(
    map_stats: MapBinStats,
    increments_S_dir: jnp.ndarray,
    increments_S_dir_scatter: jnp.ndarray,
    increments_N_dir: jnp.ndarray,
    increments_N_pos: jnp.ndarray,
    increments_sum_p: jnp.ndarray,
    increments_sum_ppT: jnp.ndarray,
) -> MapBinStats:
    """
    Update map statistics additively.

    Args:
        map_stats: Current map statistics
        increments_*: Increments to add

    Returns:
        Updated MapBinStats
    """
    return MapBinStats(
        S_dir=map_stats.S_dir + increments_S_dir,
        S_dir_scatter=map_stats.S_dir_scatter + increments_S_dir_scatter,
        N_dir=map_stats.N_dir + increments_N_dir,
        N_pos=map_stats.N_pos + increments_N_pos,
        sum_p=map_stats.sum_p + increments_sum_p,
        sum_ppT=map_stats.sum_ppT + increments_sum_ppT,
    )


@jax.jit
def _compute_map_derived_stats_core(
    S_dir: jnp.ndarray,
    N_dir: jnp.ndarray,
    N_pos: jnp.ndarray,
    sum_p: jnp.ndarray,
    sum_ppT: jnp.ndarray,
    eps_mass: jnp.ndarray,
    eps_psd: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT'd batched computation of derived stats over all bins.

    Takes map_stats fields as arrays so JIT sees only arrays (no dataclass).
    No Python loop; no host pulls. Returns (mu_dir, kappa, centroid, Sigma_c).
    """
    # Directional: batched safe_normalize over rows of S_dir
    mu_dir, _ = jax.vmap(_safe_normalize_jax, (0, None))(S_dir, eps_mass)

    # Kappa from resultant length (batched)
    inv_N_dir, _ = inv_mass_core(N_dir, eps_mass)
    S_norms = jnp.linalg.norm(S_dir, axis=1)
    Rbar = S_norms * inv_N_dir
    kappa = kappa_from_resultant_batch(Rbar, eps_r=constants.GC_EPS_R)

    # Spatial: batched centroid and covariance
    inv_N_pos, _ = inv_mass_core(N_pos, eps_mass)
    centroid = sum_p * inv_N_pos[:, None]
    Sigma_raw = (
        sum_ppT * inv_N_pos[:, None, None]
        - jnp.einsum("bi,bj->bij", centroid, centroid)
    )
    Sigma_c = domain_projection_psd_batch(Sigma_raw, eps_psd)

    return mu_dir, kappa, centroid, Sigma_c


def compute_map_derived_stats(
    map_stats: MapBinStats,
    eps_mass: float = constants.GC_EPS_MASS,
    eps_psd: float = constants.GC_EPS_PSD,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute derived statistics from sufficient stats.

    Uses a single JIT'd batched path; no Python loop, no per-bin host pulls.
    Returns (mu_dir, kappa, centroid, Sigma_c) for backward compatibility.
    """
    eps_mass_j = jnp.asarray(eps_mass, dtype=jnp.float64)
    eps_psd_j = jnp.asarray(eps_psd, dtype=jnp.float64)
    return _compute_map_derived_stats_core(
        map_stats.S_dir,
        map_stats.N_dir,
        map_stats.N_pos,
        map_stats.sum_p,
        map_stats.sum_ppT,
        eps_mass_j,
        eps_psd_j,
    )


# =============================================================================
# Forgetting for Map Statistics
# =============================================================================


def apply_forgetting(
    map_stats: MapBinStats,
    forgetting_factor: float = 0.99,
) -> MapBinStats:
    """
    Apply exponential forgetting to map statistics.

    This implements the "all accumulators must use forgetting" requirement.

    Args:
        map_stats: Current map statistics
        forgetting_factor: Factor in (0, 1), closer to 1 = slower forgetting

    Returns:
        Map statistics with forgetting applied
    """
    gamma = float(forgetting_factor)

    return MapBinStats(
        S_dir=gamma * map_stats.S_dir,
        S_dir_scatter=gamma * map_stats.S_dir_scatter,
        N_dir=gamma * map_stats.N_dir,
        N_pos=gamma * map_stats.N_pos,
        sum_p=gamma * map_stats.sum_p,
        sum_ppT=gamma * map_stats.sum_ppT,
    )
