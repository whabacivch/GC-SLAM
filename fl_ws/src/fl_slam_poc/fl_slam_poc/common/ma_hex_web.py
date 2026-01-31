"""
MA hex web: hex cell keys, fixed bucket, stencil K=64, PoE denoise, candidate generation.

Hex: a1=(1,0), a2=(1/2,√3/2); s_k = a_k·y, cell = floor(s/h), h = hex_scale_factor × median(√λ_max(Σ_bev)).
Fixed [num_cells, max_occupants]; stencil K=64. generate_candidates_ma_hex_web replaces kNN for OT association.
PoE denoise and convexity_weight available for optional per-cell denoise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

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


# -----------------------------------------------------------------------------
# Fixed bucket structure (for assignment and stencil lookup)
# -----------------------------------------------------------------------------


@dataclass
class MAHexBucket:
    """
    Fixed bucket [num_cells, max_occupants]. Each slot holds a splat index (int >= 0) or -1 for empty.
    No edits to existing binning; this is a standalone structure for BEV splat organization.
    """

    n1: int
    n2: int
    max_occupants: int
    # bucket[i, j] = splat index (>=0) or -1 if empty. Shape (n1*n2, max_occupants).
    bucket: np.ndarray
    # Count per cell (for fast lookup). Shape (n1*n2,).
    count: np.ndarray

    def __init__(self, n1: int, n2: int, max_occupants: int):
        self.n1 = n1
        self.n2 = n2
        self.max_occupants = max_occupants
        n_cells = n1 * n2
        self.bucket = np.full((n_cells, max_occupants), -1, dtype=np.int64)
        self.count = np.zeros(n_cells, dtype=np.int64)

    def clear(self) -> None:
        """Reset bucket and count."""
        self.bucket.fill(-1)
        self.count.fill(0)

    def add(self, cell_1: int, cell_2: int, splat_index: int) -> bool:
        """
        Add splat_index to cell (cell_1, cell_2). Returns True if added, False if cell full.
        When full, overwrite oldest (first slot); no hidden iteration.
        """
        idx = cell_key_to_linear(cell_1, cell_2, self.n1, self.n2)
        c = self.count[idx]
        if c < self.max_occupants:
            self.bucket[idx, c] = splat_index
            self.count[idx] = c + 1
            return True
        # Full: drop oldest (first slot), add new at end
        self.bucket[idx, 0:-1] = self.bucket[idx, 1:]
        self.bucket[idx, -1] = splat_index
        return True

    def get_occupants(self, cell_linear: int) -> np.ndarray:
        """Return array of splat indices in cell (length count[cell_linear]); no -1 padding."""
        c = int(self.count[cell_linear])
        return self.bucket[cell_linear, :c].copy()


# -----------------------------------------------------------------------------
# Candidate generation: fixed K_ASSOC per measurement from hex stencil (replaces kNN)
# -----------------------------------------------------------------------------


def generate_candidates_ma_hex_web(
    meas_positions: np.ndarray,
    map_positions: np.ndarray,
    map_covariances: np.ndarray,
    k_assoc: int,
    config: MAHexWebConfig,
) -> np.ndarray:
    """
    Generate K_ASSOC candidates per measurement via MA hex web (fixed topology).

    BEV (x, y) for hex cells; h = hex_scale_factor * median(sqrt(λ_max(Σ_bev))).
    Map primitives are binned by hex cell; per measurement we take the stencil of
    neighbor cells, collect all map indices in those cells, then take nearest
    k_assoc by squared distance (fixed-cost: sort capped at stencil_cells * max_occupants).

    Args:
        meas_positions: (N_meas, 3) measurement positions
        map_positions: (M_map, 3) map primitive positions
        map_covariances: (M_map, 3, 3) map covariances (BEV = top-left 2x2)
        k_assoc: number of candidates per measurement
        config: MAHexWebConfig (num_cells, max_occupants, stencil, scale)

    Returns:
        candidate_indices: (N_meas, k_assoc) indices into map
    """
    N_meas = meas_positions.shape[0]
    M_map = map_positions.shape[0]
    meas_positions = np.asarray(meas_positions, dtype=np.float64).reshape(-1, 3)
    map_positions = np.asarray(map_positions, dtype=np.float64).reshape(-1, 3)
    map_covariances = np.asarray(map_covariances, dtype=np.float64).reshape(-1, 3, 3)

    if M_map == 0 or N_meas == 0:
        return np.zeros((N_meas, k_assoc), dtype=np.int64)

    meas_bev = meas_positions[:, :2]
    map_bev = map_positions[:, :2]
    Sigma_bev = map_covariances[:, :2, :2]
    h = compute_hex_scale_h(Sigma_bev, config.hex_scale_factor)
    h = max(float(h), 1e-12)

    n1, n2 = config.num_cells_1, config.num_cells_2
    bucket = MAHexBucket(n1, n2, config.max_occupants)

    map_cells = hex_cell_key_batch(map_bev, h)
    for j in range(M_map):
        c1, c2 = int(map_cells[j, 0]), int(map_cells[j, 1])
        bucket.add(c1, c2, j)

    max_candidates = config.K_STENCIL * config.max_occupants
    candidate_indices = np.zeros((N_meas, k_assoc), dtype=np.int64)

    for i in range(N_meas):
        cell_i = hex_cell_key_batch(meas_bev[i : i + 1], h)[0]
        cell_linear = cell_key_to_linear(int(cell_i[0]), int(cell_i[1]), n1, n2)
        stencil = stencil_linear_indices(
            cell_linear, n1, n2, config.stencil_radius, max_size=config.K_STENCIL
        )
        collected: List[int] = []
        for idx in stencil:
            occ = bucket.get_occupants(int(idx))
            collected.extend(occ.tolist())
        arr = np.array(collected, dtype=np.int64)
        if arr.size == 0:
            candidate_indices[i, :] = 0
            continue
        if arr.size > k_assoc:
            diff = map_positions[arr] - meas_positions[i]
            dists_sq = np.sum(diff * diff, axis=1)
            order = np.argsort(dists_sq)[:k_assoc]
            candidate_indices[i, :] = arr[order]
        else:
            k_fill = min(k_assoc, arr.size)
            candidate_indices[i, :k_fill] = arr[:k_fill]
            if k_fill < k_assoc:
                candidate_indices[i, k_fill:] = arr[0] if arr.size > 0 else 0

    return candidate_indices
