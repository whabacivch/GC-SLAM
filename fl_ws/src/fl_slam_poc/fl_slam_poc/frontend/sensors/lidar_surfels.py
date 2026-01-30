"""
LiDAR surfels: voxel downsample, plane fit per voxel, Gaussian + vMF B=3, Wishart.

Plan: lidar-camera_splat_fusion_and_bev_ot. Voxel downsample (e.g. 0.1 m grid);
local plane fit per voxel; Gaussian mean from points, Σ from residuals + sensor noise;
vMF from fitted normals, multi-lobe B=3. Wishart on Λ; same natural-param interface
as camera splats. No edits to livox_converter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from fl_slam_poc.frontend.sensors.lidar_camera_depth_fusion import _fit_plane_weighted


# -----------------------------------------------------------------------------
# Config (no magic numbers)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LidarSurfelConfig:
    """Configuration for LiDAR surfel extraction."""

    voxel_size_m: float = 0.1  # voxel grid size (m)
    min_points_per_voxel: int = 3  # min points for plane fit (fixed; no RANSAC)
    sensor_noise_var_per_axis: float = 1e-6  # additive variance per axis (m²) for Σ
    # Wishart: Λ_reg = Λ_data + nu * Psi^{-1}, Psi = psi_scale * I
    wishart_nu: float = 5.0
    wishart_psi_scale: float = 0.1
    # vMF B=3: main lobe (normal) kappa from planarity; in-plane lobes smaller
    vmf_kappa_main_scale: float = 10.0  # kappa_main ∝ 1/sigma_perp
    vmf_kappa_plane_scale: float = 0.5  # in-plane lobes (B=1,2) smaller
    eig_min: float = 1e-12  # SPD clamp for cov


# -----------------------------------------------------------------------------
# Voxel assignment (fixed-cost)
# -----------------------------------------------------------------------------


def voxel_key(x: float, y: float, z: float, voxel_size: float) -> Tuple[int, int, int]:
    """Map 3D point to voxel cell key (i, j, k)."""
    voxel_size = max(float(voxel_size), 1e-12)
    return (
        int(np.floor(x / voxel_size)),
        int(np.floor(y / voxel_size)),
        int(np.floor(z / voxel_size)),
    )


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
) -> List[np.ndarray]:
    """
    Group points by voxel key. Returns list of arrays of point indices per voxel
    (each element is indices into points). Fixed-cost: one pass.
    """
    from collections import defaultdict
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    n = points.shape[0]
    if n == 0:
        return []
    voxel_size = max(float(voxel_size), 1e-12)
    groups: dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in range(n):
        k = (
            int(np.floor(points[i, 0] / voxel_size)),
            int(np.floor(points[i, 1] / voxel_size)),
            int(np.floor(points[i, 2] / voxel_size)),
        )
        groups[k].append(i)
    return [np.array(indices, dtype=np.int64) for indices in groups.values()]


# -----------------------------------------------------------------------------
# Per-voxel plane fit → Gaussian + vMF B=3
# -----------------------------------------------------------------------------


def _cov_from_plane_residuals(
    centroid: np.ndarray,
    normal: np.ndarray,
    points: np.ndarray,
    sigma_perp_sq: float,
    sensor_var: float,
    eig_min: float,
) -> np.ndarray:
    """
    Build 3x3 covariance: along-plane from point spread, normal direction from sigma_perp_sq.
    Σ = V diag(σ_plane^2, σ_plane^2, σ_perp^2) V' + sensor_var*I; V = [e1, e2, n].
    """
    n = np.asarray(normal, dtype=np.float64).ravel()[:3]
    n = n / (np.linalg.norm(n) + 1e-12)
    # Two in-plane directions (arbitrary orthonormal)
    if abs(n[2]) < 0.9:
        e1 = np.array([-n[1], n[0], 0.0], dtype=np.float64)
    else:
        e1 = np.array([-n[2], 0.0, n[0]], dtype=np.float64)
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    # In-plane variance from point spread
    centered = np.asarray(points, dtype=np.float64) - np.asarray(centroid, dtype=np.float64)
    var_e1 = float(np.var(centered @ e1)) + sensor_var
    var_e2 = float(np.var(centered @ e2)) + sensor_var
    var_perp = max(sigma_perp_sq, 0.0) + sensor_var
    V = np.column_stack([e1, e2, n])
    D = np.diag([max(var_e1, eig_min), max(var_e2, eig_min), max(var_perp, eig_min)])
    Sigma = V @ D @ V.T
    Sigma = 0.5 * (Sigma + Sigma.T) + eig_min * np.eye(3)
    return Sigma


def _vmf_lobes_from_plane(
    normal: np.ndarray,
    eigvals: np.ndarray,
    sigma_perp_sq: float,
    kappa_main_scale: float,
    kappa_plane_scale: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """B=3 vMF: lobe 0 = normal (kappa from 1/sigma_perp); lobes 1,2 = in-plane (smaller kappa)."""
    n = np.asarray(normal, dtype=np.float64).ravel()[:3]
    n = n / (np.linalg.norm(n) + eps)
    if abs(n[2]) < 0.9:
        e1 = np.array([-n[1], n[0], 0.0], dtype=np.float64)
    else:
        e1 = np.array([-n[2], 0.0, n[0]], dtype=np.float64)
    e1 = e1 / (np.linalg.norm(e1) + eps)
    e2 = np.cross(n, e1)
    e2 = e2 / (np.linalg.norm(e2) + eps)
    sigma_perp_sq = max(sigma_perp_sq, eps)
    kappa_main = min(kappa_main_scale / np.sqrt(sigma_perp_sq), 100.0)
    kappa_plane = max(kappa_plane_scale, 0.1)
    mus = np.stack([n, e1, e2], axis=0)  # (3, 3)
    kappas = np.array([kappa_main, kappa_plane, kappa_plane], dtype=np.float64)
    return mus, kappas


# -----------------------------------------------------------------------------
# Wishart regularization: Λ_reg = Λ + nu * Psi^{-1}, Psi = psi_scale * I
# -----------------------------------------------------------------------------


def wishart_regularize_3d(
    Lambda: np.ndarray,
    nu: float,
    psi_scale: float,
    eig_min: float = 1e-12,
) -> np.ndarray:
    """Λ_reg = Λ + nu/psi_scale * I. Ensures Λ_reg is PD."""
    Lambda = np.asarray(Lambda, dtype=np.float64).reshape(3, 3)
    Lambda = 0.5 * (Lambda + Lambda.T)
    s = max(float(psi_scale), 1e-12)
    Lambda_reg = Lambda + (nu / s) * np.eye(3)
    eigvals = np.linalg.eigvalsh(Lambda_reg)
    if eigvals.min() < eig_min:
        Lambda_reg = Lambda_reg + (eig_min - eigvals.min()) * np.eye(3)
    return Lambda_reg


# -----------------------------------------------------------------------------
# Surfel dataclass (same natural-param interface as camera splats)
# -----------------------------------------------------------------------------


@dataclass
class LidarSurfel:
    """LiDAR surfel: same natural-param interface as Feature3D for OT/splat fusion."""

    xyz: np.ndarray  # (3,)
    cov_xyz: np.ndarray  # (3, 3)
    info_xyz: np.ndarray  # (3, 3)
    canonical_theta: np.ndarray  # (3,)
    logdet_cov: float
    # Multi-lobe vMF B=3
    mu_app: np.ndarray  # (3, 3) B=3 lobes
    kappa_app: np.ndarray  # (3,)
    weight: float = 1.0


def points_to_surfels(
    points: np.ndarray,
    config: LidarSurfelConfig,
) -> List[LidarSurfel]:
    """
    Voxel downsample, plane fit per voxel, Gaussian + vMF B=3, Wishart.
    Returns list of LidarSurfel (same natural-param interface as camera splats).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[0] < config.min_points_per_voxel:
        return []

    voxel_groups = voxel_downsample(points, config.voxel_size_m)
    surfels: List[LidarSurfel] = []
    eig_min = config.eig_min
    sensor_var = config.sensor_noise_var_per_axis

    for indices in voxel_groups:
        if len(indices) < config.min_points_per_voxel:
            continue
        pts = points[indices]
        centroid, normal, eigvals, sigma_perp_sq = _fit_plane_weighted(pts, None)
        Sigma = _cov_from_plane_residuals(
            centroid, normal, pts, sigma_perp_sq, sensor_var, eig_min,
        )
        Lambda = np.linalg.inv(Sigma + eig_min * np.eye(3))
        Lambda = 0.5 * (Lambda + Lambda.T)
        Lambda_reg = wishart_regularize_3d(
            Lambda, config.wishart_nu, config.wishart_psi_scale, eig_min,
        )
        Sigma_reg = np.linalg.inv(Lambda_reg)
        Sigma_reg = 0.5 * (Sigma_reg + Sigma_reg.T) + eig_min * np.eye(3)
        info_xyz = np.linalg.inv(Sigma_reg)
        info_xyz = 0.5 * (info_xyz + info_xyz.T)
        sign, logdet_cov = np.linalg.slogdet(Sigma_reg)
        canonical_theta = info_xyz @ centroid
        mu_app, kappa_app = _vmf_lobes_from_plane(
            normal, eigvals, sigma_perp_sq,
            config.vmf_kappa_main_scale, config.vmf_kappa_plane_scale,
        )
        surfels.append(
            LidarSurfel(
                xyz=centroid.copy(),
                cov_xyz=Sigma_reg.copy(),
                info_xyz=info_xyz.copy(),
                canonical_theta=canonical_theta.copy(),
                logdet_cov=float(logdet_cov),
                mu_app=mu_app,
                kappa_app=kappa_app,
                weight=1.0,
            )
        )
    return surfels
