"""
Packed splat batch for OT: fixed caps N_max, M_max, masks, neighbor indices.

Standardizes internal representation so pipeline wiring uses fixed-size arrays
and avoids dynamic shapes (recompiles / slow paths). JIT-friendly when used
with fixed N_max, M_max, K_max. Even without JIT, this prevents future rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Fixed caps (compile-time constants; tune per deployment)
# -----------------------------------------------------------------------------

# Default caps for OT: camera splats N, LiDAR splats M, stencil K (e.g. MA hex)
GC_OT_N_MAX = 2048  # Max camera BEV splats per batch
GC_OT_M_MAX = 2048  # Max LiDAR BEV splats per batch
GC_OT_K_STENCIL = 64  # Max stencil size (e.g. MA hex web)
GC_OT_K_MAX = GC_OT_K_STENCIL  # Alias for fixed cap K_max (shape consistency)


# -----------------------------------------------------------------------------
# Packed batch structure
# -----------------------------------------------------------------------------


@dataclass
class PackedSplatBatch:
    """
    Fixed-size packed batch for Sinkhorn OT and fusion.
    Padded to (N_max, M_max); masks indicate valid rows/columns.
    Neighbor indices (e.g. MA hex stencil) padded to (N_max, K_max); -1 = invalid.
    """

    N_max: int
    M_max: int
    K_max: int  # Max stencil size (e.g. MA hex neighbors)
    n_cam: int  # Actual number of camera splats (n_cam <= N_max)
    n_lidar: int  # Actual number of LiDAR splats (n_lidar <= M_max)

    # Camera BEV splats (padded to N_max)
    mu_cam: np.ndarray  # (N_max, 2)
    Sigma_cam: np.ndarray  # (N_max, 2, 2)
    mask_cam: np.ndarray  # (N_max,) 1.0 valid, 0.0 padding
    mu_n_cam: Optional[np.ndarray]  # (N_max, 3) or None
    kappa_cam: np.ndarray  # (N_max,)

    # LiDAR BEV splats (padded to M_max)
    mu_lidar: np.ndarray  # (M_max, 2)
    Sigma_lidar: np.ndarray  # (M_max, 2, 2)
    mask_lidar: np.ndarray  # (M_max,)
    mu_n_lidar: Optional[np.ndarray]  # (M_max, 3) or None
    kappa_lidar: np.ndarray  # (M_max,)

    # Neighbor indices for MA hex (camera side): (N_max, K_max); -1 = invalid. Always allocated for shape consistency.
    neighbor_indices_cam: np.ndarray


def pack_splat_batch(
    mu_cam: np.ndarray,
    Sigma_cam: np.ndarray,
    mu_n_cam: Optional[np.ndarray],
    kappa_cam: np.ndarray,
    mu_lidar: np.ndarray,
    Sigma_lidar: np.ndarray,
    mu_n_lidar: Optional[np.ndarray],
    kappa_lidar: np.ndarray,
    N_max: int = GC_OT_N_MAX,
    M_max: int = GC_OT_M_MAX,
    K_max: int = GC_OT_K_STENCIL,
    neighbor_indices_cam: Optional[np.ndarray] = None,
) -> PackedSplatBatch:
    """
    Pack variable-length camera and LiDAR BEV splats into fixed-size arrays with masks.
    Pads with zeros; mask_cam[i] = 1 for i < n_cam else 0, mask_lidar[j] = 1 for j < n_lidar else 0.
    neighbor_indices_cam: optional (n_cam, K) integer array (e.g. MA hex stencil); padded to (N_max, K_max) with -1.
    """
    mu_cam = np.asarray(mu_cam, dtype=np.float64)
    Sigma_cam = np.asarray(Sigma_cam, dtype=np.float64)
    mu_lidar = np.asarray(mu_lidar, dtype=np.float64)
    Sigma_lidar = np.asarray(Sigma_lidar, dtype=np.float64)
    kappa_cam = np.asarray(kappa_cam, dtype=np.float64).ravel()
    kappa_lidar = np.asarray(kappa_lidar, dtype=np.float64).ravel()

    n_cam = min(mu_cam.shape[0], N_max)
    n_lidar = min(mu_lidar.shape[0], M_max)

    # Allocate fixed-size
    mu_cam_pad = np.zeros((N_max, 2), dtype=np.float64)
    Sigma_cam_pad = np.zeros((N_max, 2, 2), dtype=np.float64)
    mask_cam_pad = np.zeros(N_max, dtype=np.float64)
    kappa_cam_pad = np.zeros(N_max, dtype=np.float64)
    mu_cam_pad[:n_cam] = mu_cam[:n_cam].reshape(-1, 2)[:n_cam]
    Sigma_cam_pad[:n_cam] = Sigma_cam[:n_cam].reshape(-1, 2, 2)[:n_cam]
    mask_cam_pad[:n_cam] = 1.0
    kappa_cam_pad[:n_cam] = kappa_cam[:n_cam] if kappa_cam.size >= n_cam else np.full(n_cam, 1.0)

    mu_lidar_pad = np.zeros((M_max, 2), dtype=np.float64)
    Sigma_lidar_pad = np.zeros((M_max, 2, 2), dtype=np.float64)
    mask_lidar_pad = np.zeros(M_max, dtype=np.float64)
    kappa_lidar_pad = np.zeros(M_max, dtype=np.float64)
    mu_lidar_pad[:n_lidar] = mu_lidar[:n_lidar].reshape(-1, 2)[:n_lidar]
    Sigma_lidar_pad[:n_lidar] = Sigma_lidar[:n_lidar].reshape(-1, 2, 2)[:n_lidar]
    mask_lidar_pad[:n_lidar] = 1.0
    kappa_lidar_pad[:n_lidar] = kappa_lidar[:n_lidar] if kappa_lidar.size >= n_lidar else np.full(n_lidar, 1.0)

    mu_n_cam_pad: Optional[np.ndarray] = None
    if mu_n_cam is not None:
        mu_n_cam_pad = np.zeros((N_max, 3), dtype=np.float64)
        mu_n_cam_pad[:n_cam] = np.asarray(mu_n_cam, dtype=np.float64).reshape(-1, 3)[:n_cam]
    mu_n_lidar_pad: Optional[np.ndarray] = None
    if mu_n_lidar is not None:
        mu_n_lidar_pad = np.zeros((M_max, 3), dtype=np.float64)
        mu_n_lidar_pad[:n_lidar] = np.asarray(mu_n_lidar, dtype=np.float64).reshape(-1, 3)[:n_lidar]

    # Neighbor indices: always (N_max, K_max), -1 = invalid (shape consistency for JIT)
    neigh_pad = np.full((N_max, K_max), -1, dtype=np.int64)
    if neighbor_indices_cam is not None:
        neigh = np.asarray(neighbor_indices_cam, dtype=np.int64)
        if neigh.ndim == 1:
            neigh = neigh.reshape(1, -1)
        n_rows = min(neigh.shape[0], n_cam, N_max)
        n_cols = min(neigh.shape[1], K_max)
        if n_rows > 0 and n_cols > 0:
            neigh_pad[:n_rows, :n_cols] = neigh[:n_rows, :n_cols]

    return PackedSplatBatch(
        N_max=N_max,
        M_max=M_max,
        K_max=K_max,
        n_cam=n_cam,
        n_lidar=n_lidar,
        mu_cam=mu_cam_pad,
        Sigma_cam=Sigma_cam_pad,
        mask_cam=mask_cam_pad,
        mu_n_cam=mu_n_cam_pad,
        kappa_cam=kappa_cam_pad,
        mu_lidar=mu_lidar_pad,
        Sigma_lidar=Sigma_lidar_pad,
        mask_lidar=mask_lidar_pad,
        mu_n_lidar=mu_n_lidar_pad,
        kappa_lidar=kappa_lidar_pad,
        neighbor_indices_cam=neigh_pad,
    )


def unpack_coupling_masked(
    pi: np.ndarray,
    mask_cam: np.ndarray,
    mask_lidar: np.ndarray,
    n_cam: int,
    n_lidar: int,
) -> np.ndarray:
    """
    Return coupling restricted to valid region (n_cam x n_lidar).
    pi is (N_max, M_max); output is (n_cam, n_lidar) with invalid region zeroed.
    """
    pi = np.asarray(pi, dtype=np.float64)
    out = np.zeros((n_cam, n_lidar), dtype=np.float64)
    if n_cam > 0 and n_lidar > 0:
        out = pi[:n_cam, :n_lidar].copy()
    return out
