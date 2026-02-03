"""
MA hex web: hex cell keys, fixed bucket, stencil K=64, PoE denoise, candidate generation.

Hex: a1=(1,0), a2=(1/2,√3/2); s_k = a_k·y, cell = floor(s/h), h = hex_scale_factor × median(√λ_max(Σ_bev)).
Fixed [num_cells, max_occupants]; stencil K=64. generate_candidates_ma_hex_web_jax replaces kNN for OT association.
PoE denoise and convexity_weight available for optional per-cell denoise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from fl_slam_poc.common.jax_init import jax, jnp

# Hex basis (plan): a1=(1,0), a2=(1/2, √3/2), a3 = a2 - a1
_A1 = np.array([1.0, 0.0], dtype=np.float64)
_A2 = np.array([0.5, 0.5 * np.sqrt(3.0)], dtype=np.float64)
_A3 = _A2 - _A1  # (-0.5, √3/2)


# -----------------------------------------------------------------------------
# Config (no magic numbers)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MAHexWebConfig:
    """Configuration for MA hex web."""

    num_cells_1: int = 128  # grid extent in s1 (cell_1 in [-n1/2, n1/2) or [0, n1))
    num_cells_2: int = 128  # grid extent in s2
    max_occupants: int = 32  # max splats per cell (fixed bucket)
    K_STENCIL: int = 64  # fixed stencil size (e.g. 8×8)
    hex_scale_factor: float = 2.5  # h = hex_scale_factor * median(sqrt(λ_max(Σ)))
    # Stencil radius in cell units (e.g. 4 → 8×8 = 64 cells if symmetric)
    stencil_radius: int = 4


# -----------------------------------------------------------------------------
# Hex cell key: s_k = a_k·y, cell_k = floor(s_k / h)
# -----------------------------------------------------------------------------


def hex_cell_key(y: np.ndarray, h: float) -> Tuple[int, int]:
    """
    Map 2D BEV point y to hex cell (cell_1, cell_2). s1 = a1·y, s2 = a2·y; cell_k = floor(s_k/h).

    Args:
        y: (2,) BEV coordinates (e.g. x, y in world or body).
        h: cell scale (m); from config and median(√λ_max(Σ)).

    Returns:
        (cell_1, cell_2) integer cell indices.
    """
    y = np.asarray(y, dtype=np.float64).ravel()[:2]
    s1 = float(np.dot(_A1, y))
    s2 = float(np.dot(_A2, y))
    h = max(float(h), 1e-12)
    cell_1 = int(np.floor(s1 / h))
    cell_2 = int(np.floor(s2 / h))
    return (cell_1, cell_2)


def hex_cell_key_batch(Y: np.ndarray, h: float) -> np.ndarray:
    """(N, 2) BEV points -> (N, 2) cell keys (cell_1, cell_2)."""
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 2)
    s1 = Y @ _A1
    s2 = Y @ _A2
    h = max(float(h), 1e-12)
    cell_1 = np.floor(s1 / h).astype(np.int64)
    cell_2 = np.floor(s2 / h).astype(np.int64)
    return np.column_stack([cell_1, cell_2])


def hex_cell_key_batch_jax(Y: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """JAX: (N, 2) BEV points -> (N, 2) cell keys (cell_1, cell_2)."""
    Y = jnp.asarray(Y, dtype=jnp.float64).reshape(-1, 2)
    h = jnp.maximum(jnp.asarray(h, dtype=jnp.float64), 1e-12)
    s1 = Y @ jnp.asarray(_A1, dtype=jnp.float64)
    s2 = Y @ jnp.asarray(_A2, dtype=jnp.float64)
    cell_1 = jnp.floor(s1 / h).astype(jnp.int32)
    cell_2 = jnp.floor(s2 / h).astype(jnp.int32)
    return jnp.stack([cell_1, cell_2], axis=1)


# -----------------------------------------------------------------------------
# Scale h from BEV covariances: h = scale_factor * median(√λ_max(Σ))
# -----------------------------------------------------------------------------


def compute_hex_scale_h(
    Sigma_bev: np.ndarray,
    scale_factor: float = 2.5,
) -> float:
    """
    Compute cell scale h from BEV covariances. h = scale_factor * median(√λ_max(Σ)).

    Sigma_bev: (N, 2, 2) BEV covariances.
    """
    Sigma_bev = np.asarray(Sigma_bev, dtype=np.float64)
    if Sigma_bev.size == 0:
        return 1.0
    if Sigma_bev.ndim == 2:
        Sigma_bev = Sigma_bev.reshape(1, 2, 2)
    N = Sigma_bev.shape[0]
    sqrt_lam_max = np.zeros(N, dtype=np.float64)
    for i in range(N):
        eigvals = np.linalg.eigvalsh(Sigma_bev[i])
        eigvals = np.maximum(eigvals, 1e-12)
        sqrt_lam_max[i] = np.sqrt(float(np.max(eigvals)))
    med = float(np.median(sqrt_lam_max))
    return max(scale_factor * med, 1e-6)


def compute_hex_scale_h_jax(
    Sigma_bev: jnp.ndarray,
    scale_factor: float = 2.5,
) -> jnp.ndarray:
    """
    JAX version: h = scale_factor * median(sqrt(lambda_max(Sigma_bev))).
    Sigma_bev: (N, 2, 2) or (2, 2)
    """
    Sigma_bev = jnp.asarray(Sigma_bev, dtype=jnp.float64)
    Sigma_bev = jnp.reshape(Sigma_bev, (-1, 2, 2))
    eigvals = jax.vmap(jnp.linalg.eigvalsh)(Sigma_bev)
    eigvals = jnp.maximum(eigvals, 1e-12)
    sqrt_lam_max = jnp.sqrt(jnp.max(eigvals, axis=1))
    med = jnp.median(sqrt_lam_max)
    return jnp.maximum(scale_factor * med, 1e-6)


# -----------------------------------------------------------------------------
# Fixed bucket: [num_cells, max_occupants]; cell index from (cell_1, cell_2)
# -----------------------------------------------------------------------------


def cell_key_to_linear(cell_1: int, cell_2: int, n1: int, n2: int) -> int:
    """Map (cell_1, cell_2) to linear index in [0, n1*n2) with modulo wrapping."""
    i1 = cell_1 % n1
    i2 = cell_2 % n2
    if i1 < 0:
        i1 += n1
    if i2 < 0:
        i2 += n2
    return int(i1 * n2 + i2)


def cell_key_to_linear_jax(cell_1: jnp.ndarray, cell_2: jnp.ndarray, n1: int, n2: int) -> jnp.ndarray:
    """JAX: map (cell_1, cell_2) to linear index in [0, n1*n2) with modulo wrapping."""
    i1 = jnp.mod(cell_1, n1)
    i2 = jnp.mod(cell_2, n2)
    return i1 * n2 + i2


def stencil_linear_indices(
    cell_linear: int,
    n1: int,
    n2: int,
    radius: int,
    max_size: int = 64,
) -> np.ndarray:
    """
    Return linear indices of cells in a square stencil around the given cell.
    Center at (i1, i2) from cell_linear; stencil = (i1+di, i2+dj) for di,dj in [-radius, radius).
    Up to max_size cells (fixed K). Wrapping in [0, n1) x [0, n2).
    """
    n_cells = n1 * n2
    i1 = cell_linear // n2
    i2 = cell_linear % n2
    out: List[int] = []
    side = 2 * radius
    for di in range(-radius, radius):
        for dj in range(-radius, radius):
            if len(out) >= max_size:
                break
            ni1 = (i1 + di) % n1
            ni2 = (i2 + dj) % n2
            if ni1 < 0:
                ni1 += n1
            if ni2 < 0:
                ni2 += n2
            idx = ni1 * n2 + ni2
            out.append(idx)
        if len(out) >= max_size:
            break
    return np.array(out[:max_size], dtype=np.int64)


# -----------------------------------------------------------------------------
# PoE denoise: combine cell + stencil experts (Gaussian PoE)
# -----------------------------------------------------------------------------


def poe_denoise_2d(
    mu: np.ndarray,
    Lambda: np.ndarray,
    stencil_mus: np.ndarray,
    stencil_Lambdas: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Product of experts: Lambda_poe = Lambda + sum(stencil_Lambdas), theta_poe = Lambda@mu + sum(Lambda_j@mu_j).
    mu_poe = inv(Lambda_poe) @ theta_poe. Fixed-cost; no iteration.

    Args:
        mu: (2,) cell mean. Lambda: (2,2) cell precision.
        stencil_mus: (K, 2) neighbor means. stencil_Lambdas: (K, 2, 2) neighbor precisions.

    Returns:
        (mu_poe (2,), Lambda_poe (2,2)).
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()[:2]
    Lambda = np.asarray(Lambda, dtype=np.float64).reshape(2, 2)
    stencil_mus = np.asarray(stencil_mus, dtype=np.float64).reshape(-1, 2)
    stencil_Lambdas = np.asarray(stencil_Lambdas, dtype=np.float64).reshape(-1, 2, 2)
    theta = Lambda @ mu
    Lambda_poe = Lambda.copy()
    for k in range(stencil_Lambdas.shape[0]):
        Lambda_poe += stencil_Lambdas[k]
        theta += stencil_Lambdas[k] @ stencil_mus[k]
    Lambda_poe += eps * np.eye(2)
    mu_poe = np.linalg.solve(Lambda_poe, theta)
    return mu_poe, Lambda_poe


def convexity_weight(
    mu: np.ndarray,
    mu_poe: np.ndarray,
    tau: float = 10.0,
) -> float:
    """
    Continuous convexity weight from shift after PoE: w = 1/(1 + tau * |mu_poe - mu|^2).
    No gate; smooth downweight when denoised mean moves far from cell mean.
    """
    diff = np.asarray(mu_poe, dtype=np.float64).ravel()[:2] - np.asarray(mu, dtype=np.float64).ravel()[:2]
    d2 = float(np.dot(diff, diff))
    return 1.0 / (1.0 + tau * d2)


#
# Candidate generation is JAX-only in this project to avoid host transfers and to
# enforce a single runtime implementation.


# -----------------------------------------------------------------------------
# JAX Candidate Generation
# -----------------------------------------------------------------------------


def generate_candidates_ma_hex_web_jax(
    meas_positions: jnp.ndarray,
    map_positions: jnp.ndarray,
    map_covariances: jnp.ndarray,
    k_assoc: int,
    config: MAHexWebConfig,
) -> jnp.ndarray:
    """
    JAX candidate generation via MA hex web (fixed budgets, no branches).

    Returns:
        candidate_indices: (N_meas, k_assoc) indices into map
    """
    meas_positions = jnp.asarray(meas_positions, dtype=jnp.float64).reshape(-1, 3)
    map_positions = jnp.asarray(map_positions, dtype=jnp.float64).reshape(-1, 3)
    map_covariances = jnp.asarray(map_covariances, dtype=jnp.float64).reshape(-1, 3, 3)
    N_meas = meas_positions.shape[0]
    M_map = map_positions.shape[0]

    if M_map == 0 or N_meas == 0:
        return jnp.zeros((N_meas, k_assoc), dtype=jnp.int32)

    meas_bev = meas_positions[:, :2]
    map_bev = map_positions[:, :2]
    Sigma_bev = map_covariances[:, :2, :2]
    h = compute_hex_scale_h_jax(Sigma_bev, config.hex_scale_factor)
    h = jnp.maximum(h, 1e-12)

    n1, n2 = int(config.num_cells_1), int(config.num_cells_2)
    max_occ = int(config.max_occupants)

    # Bucket: fixed [n_cells, max_occupants]
    n_cells = n1 * n2
    bucket = jnp.full((n_cells, max_occ), -1, dtype=jnp.int32)
    count = jnp.zeros((n_cells,), dtype=jnp.int32)

    map_cells = hex_cell_key_batch_jax(map_bev, h)
    map_linear = cell_key_to_linear_jax(map_cells[:, 0], map_cells[:, 1], n1, n2).astype(jnp.int32)

    def add_one(j, state):
        bucket_i, count_i = state
        cell = map_linear[j]
        c = count_i[cell]
        can_add = c < max_occ
        pos = jnp.minimum(c, max_occ - 1)
        cell_bucket = bucket_i[cell]
        added = cell_bucket.at[pos].set(j)
        shifted = jnp.concatenate([cell_bucket[1:], jnp.array([j], dtype=jnp.int32)], axis=0)
        new_cell = jnp.where(can_add, added, shifted)
        bucket_i = bucket_i.at[cell].set(new_cell)
        count_i = count_i.at[cell].set(jnp.minimum(c + 1, max_occ))
        return bucket_i, count_i

    bucket, count = jax.lax.fori_loop(0, M_map, add_one, (bucket, count))

    # Stencil offsets
    radius = int(config.stencil_radius)
    grid = jnp.arange(-radius, radius, dtype=jnp.int32)
    di, dj = jnp.meshgrid(grid, grid, indexing="ij")
    di = di.reshape(-1)
    dj = dj.reshape(-1)
    if di.shape[0] > int(config.K_STENCIL):
        di = di[: int(config.K_STENCIL)]
        dj = dj[: int(config.K_STENCIL)]

    def candidates_for_meas(y):
        cell = hex_cell_key_batch_jax(y[None, :2], h)[0]
        cell_lin = cell_key_to_linear_jax(cell[0], cell[1], n1, n2)
        i1 = cell_lin // n2
        i2 = cell_lin % n2
        ni1 = jnp.mod(i1 + di, n1)
        ni2 = jnp.mod(i2 + dj, n2)
        stencil_lin = ni1 * n2 + ni2
        cand = bucket[stencil_lin]  # (K_STENCIL, max_occ)
        cand_flat = cand.reshape(-1)
        valid = cand_flat >= 0
        idx_safe = jnp.where(valid, cand_flat, 0)
        diff = map_positions[idx_safe] - y[None, :]
        dists = jnp.sum(diff * diff, axis=1)
        dists = jnp.where(valid, dists, 1e12)
        order = jnp.argsort(dists)[:k_assoc]
        return idx_safe[order].astype(jnp.int32)

    candidate_indices = jax.vmap(candidates_for_meas)(meas_positions)
    return candidate_indices

# -----------------------------------------------------------------------------
# MA HEX 3D (JAX): fixed bucket for LiDAR surfel extraction
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MAHex3DConfig:
    """
    3D MA hex binning configuration (JAX-friendly).

    Note: This is used for *per-scan* LiDAR surfel extraction. The grid is a fixed-size hash grid
    (via modulo wrapping) to keep compute bounded; collisions are an explicit approximation.
    """

    num_cells_1: int = 32
    num_cells_2: int = 32
    num_cells_z: int = 8
    max_occupants: int = 32
    voxel_size: float = 0.1

    @property
    def n_cells(self) -> int:
        return int(self.num_cells_1 * self.num_cells_2 * self.num_cells_z)


@dataclass
class MAHex3DBucket:
    """
    Fixed bucket:
      - bucket: (n_cells, max_occupants) point indices, -1 = empty
      - count:  (n_cells,) occupancy (clipped to max_occupants for fixed gather)
    """

    bucket: jnp.ndarray
    count: jnp.ndarray

    @staticmethod
    def create(config: MAHex3DConfig) -> "MAHex3DBucket":
        return MAHex3DBucket(
            bucket=jnp.full((config.n_cells, config.max_occupants), -1, dtype=jnp.int32),
            count=jnp.zeros((config.n_cells,), dtype=jnp.int32),
        )


def hex_cell_3d_batch(points: jnp.ndarray, h: float) -> jnp.ndarray:
    """
    Vectorized 3D cell assignment.

    Hex for x,y:
      s1 = x
      s2 = 0.5 x + (sqrt(3)/2) y
      cell_k = floor(s_k / h)
    Linear for z:
      cell_z = floor(z / h)
    """
    points = jnp.asarray(points, dtype=jnp.float64).reshape(-1, 3)
    h = jnp.maximum(jnp.asarray(h, dtype=jnp.float64), 1e-12)
    s1 = points[:, 0]
    s2 = points[:, 0] * 0.5 + points[:, 1] * (jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float64)) * 0.5)
    sz = points[:, 2]
    cell_1 = jnp.floor(s1 / h).astype(jnp.int32)
    cell_2 = jnp.floor(s2 / h).astype(jnp.int32)
    cell_z = jnp.floor(sz / h).astype(jnp.int32)
    return jnp.stack([cell_1, cell_2, cell_z], axis=1)


def bin_points_3d(
    points: jnp.ndarray,
    point_mask: jnp.ndarray,
    config: MAHex3DConfig,
) -> MAHex3DBucket:
    """
    Bin N points into a fixed 3D MA-hex grid. Fully vectorized, fixed output sizes.

    Args:
        points: (N, 3) point positions (any frame; typically base frame).
        point_mask: (N,) bool/int mask; points with mask=0 are ignored.
        config: MAHex3DConfig
    """
    points = jnp.asarray(points, dtype=jnp.float64).reshape(-1, 3)
    point_mask = jnp.asarray(point_mask).reshape(-1)
    N = int(points.shape[0])
    n_cells = int(config.n_cells)
    max_occ = int(config.max_occupants)

    cells = hex_cell_3d_batch(points, config.voxel_size)
    wrap = jnp.asarray([config.num_cells_1, config.num_cells_2, config.num_cells_z], dtype=jnp.int32)
    cells = jnp.mod(cells, wrap[None, :])

    linear = (
        cells[:, 0] * (config.num_cells_2 * config.num_cells_z)
        + cells[:, 1] * config.num_cells_z
        + cells[:, 2]
    ).astype(jnp.int32)

    mask_i32 = point_mask.astype(jnp.int32)
    linear = jnp.where(mask_i32 > 0, linear, jnp.int32(0))

    # Sort by (is_masked, linear) so masked points are last (do not affect ranks for real cells).
    key = linear + (jnp.int32(1) - mask_i32) * jnp.int32(n_cells)
    order = jnp.argsort(key)
    linear_s = linear[order]
    mask_s = mask_i32[order]
    idx_s = jnp.arange(N, dtype=jnp.int32)[order]
    pos = jnp.arange(N, dtype=jnp.int32)

    # Count per cell (includes all points; clipped later for fixed gathers).
    count = jnp.zeros((n_cells,), dtype=jnp.int32).at[linear_s].add(mask_s)

    # Start position per cell in the sorted list.
    start = jnp.full((n_cells,), N, dtype=jnp.int32).at[linear_s].min(pos)
    start = jnp.where(count > 0, start, jnp.int32(0))

    rank = pos - start[linear_s]
    keep = (mask_s == 1) & (rank < max_occ)

    # Scatter all points into an extended bucket; dropped points go to a dummy cell/slot.
    cell_t = jnp.where(keep, linear_s, jnp.int32(n_cells))
    rank_t = jnp.where(keep, rank, jnp.int32(max_occ))
    idx_t = jnp.where(keep, idx_s, jnp.int32(-1))

    bucket_ext = jnp.full((n_cells + 1, max_occ + 1), -1, dtype=jnp.int32)
    bucket_ext = bucket_ext.at[cell_t, rank_t].set(idx_t)

    bucket = bucket_ext[:n_cells, :max_occ]
    count_clipped = jnp.minimum(count, jnp.int32(max_occ))
    return MAHex3DBucket(bucket=bucket, count=count_clipped)
