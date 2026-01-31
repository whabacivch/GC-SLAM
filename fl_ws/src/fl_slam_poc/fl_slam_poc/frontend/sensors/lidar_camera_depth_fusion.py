"""
LiDAR–camera depth fusion via natural-parameter addition on depth along camera rays.

I0.2: One LiDAR depth evidence function lidar_depth_evidence(u,v,...) -> (Λ_ell, θ_ell).
Internally computes Route A (project + robust sample) and Route B (ray–plane);
outputs always defined, continuous; Λℓ, θℓ → 0 when not applicable. PoE: Λf = Λc + Λℓ, θf = θc + θℓ.

§1: Return natural params; Route A/B as mixture-of-experts (Λℓ = ΛA + ΛB, θℓ = θA + θB).
Reliability: wA = w_cnt * w_mad * w_repr; wB = w_angle * w_planar * w_res * w_z. Λ = w σ^{-2}, θ = Λ ẑ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# MAD scale for Gaussian: 1.4826 * median(|x - med|) estimates sigma
MAD_NORMAL_SCALE = 1.4826


# -----------------------------------------------------------------------------
# Config (no magic numbers)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LidarCameraDepthFusionConfig:
    """Configuration for LiDAR–camera depth fusion and unified lidar_depth_evidence (Route A + B)."""

    # Depth fusion: reliability scaling (continuous w ∈ (0,1]); apply as Λ←wΛ, θ←wθ.
    depth_fusion_weight_camera: float = 1.0
    depth_fusion_weight_lidar: float = 1.0
    # Global modality dial for LiDAR evidence (optional): Λ_ell *= gamma_lidar
    gamma_lidar: float = 1.0

    # Route A: project LiDAR to image, robust sample
    lidar_projection_radius_pix: float = 3.0
    lidar_plane_fit_min_points: int = 3
    lidar_depth_base_sigma_m: float = 0.02
    lidar_incidence_sigma_scale: float = 1.0
    depth_var_min_m2: float = 1e-8
    # Route A reliability (continuous): w_cnt = σ(α(n−n0)), w_mad = exp(−β σ²_A), w_repr = exp(−γ r²)
    point_support_n0: float = 3.0
    point_support_alpha: float = 1.0
    spread_mad_beta: float = 10.0  # w_mad = exp(−β σ²_A)
    repr_gamma: float = 10.0  # w_repr = exp(−γ r²), r² = depth variance in neighborhood

    # Route B: ray–plane intersection
    lidar_ray_plane_fit_max_points: int = 64
    plane_intersection_delta: float = 1e-6
    plane_angle_sigmoid_alpha: float = 10.0
    plane_angle_sigmoid_t: float = 0.1
    plane_planarity_sigmoid_beta: float = 5.0
    plane_planarity_rho0: float = 0.3
    plane_residual_exp_gamma: float = 100.0
    plane_fit_eps: float = 1e-12
    depth_min_m: float = 0.05
    depth_sigma_max_sq: float = 1e4
    # w_z = σ(α_z(z* − z_min)) for behind/min-depth (continuous)
    depth_min_sigmoid_alpha_z: float = 20.0


def _sigmoid(x: float) -> float:
    """Continuous sigmoid 1/(1+exp(-x)); no gate."""
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    t = math.exp(x)
    return t / (1.0 + t)


# -----------------------------------------------------------------------------
# Route A: project LiDAR to image, per-(u,v) → (ΛA, θA)
# -----------------------------------------------------------------------------


def _project_camera(
    points_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points in camera frame to (u, v) and depth z. Returns (uv (N,2), z (N,))."""
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.ndim == 1:
        points_cam = points_cam.reshape(1, -1)
    n = points_cam.shape[0]
    z = points_cam[:, 2]
    u = fx * (points_cam[:, 0] / (z + 1e-12)) + cx
    v = fy * (points_cam[:, 1] / (z + 1e-12)) + cy
    uv = np.column_stack([u, v])
    return uv, z


def _route_a_natural_params(
    points_camera_frame: np.ndarray,
    uv_query: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: LidarCameraDepthFusionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Route A: per-query (u,v) → (ΛA, θA). Robust sample + reliability wA = w_cnt * w_mad * w_repr.
    ΛA = wA/σ²_A, θA = ΛA*z^A. Invalid/weak → (0, 0). Always defined, continuous.
    """
    uv_query = np.asarray(uv_query, dtype=np.float64)
    if uv_query.ndim == 1:
        uv_query = uv_query.reshape(1, 2)
    M = uv_query.shape[0]
    Lambda_A = np.zeros(M, dtype=np.float64)
    theta_A = np.zeros(M, dtype=np.float64)

    if points_camera_frame.size == 0:
        return Lambda_A, theta_A

    uv, z = _project_camera(points_camera_frame, fx, fy, cx, cy)
    valid = z > 0.0
    uv = uv[valid]
    z = z[valid]
    if uv.size == 0:
        return Lambda_A, theta_A

    r = config.lidar_projection_radius_pix
    base_sigma = config.lidar_depth_base_sigma_m
    var_min = config.depth_var_min_m2
    n0 = config.point_support_n0
    alpha_cnt = config.point_support_alpha
    beta_mad = config.spread_mad_beta
    gamma_repr = config.repr_gamma

    for i in range(M):
        uq, vq = uv_query[i, 0], uv_query[i, 1]
        dist_sq = (uv[:, 0] - uq) ** 2 + (uv[:, 1] - vq) ** 2
        in_radius = dist_sq <= (r * r)
        if not np.any(in_radius):
            continue
        z_near = z[in_radius]
        n_pt = z_near.size
        if n_pt < config.lidar_plane_fit_min_points:
            continue
        z_med = float(np.median(z_near))
        mad = float(np.median(np.abs(z_near - z_med)))
        sigma_A = MAD_NORMAL_SCALE * mad
        sigma_A_sq = sigma_A * sigma_A
        var_spread = float(np.var(z_near)) if n_pt >= 2 else 0.0
        sigma_ell_sq = base_sigma * base_sigma + max(sigma_A_sq, var_spread)
        sigma_ell_sq = max(sigma_ell_sq, var_min)

        w_cnt = _sigmoid(alpha_cnt * (float(n_pt) - n0))
        w_mad = math.exp(-beta_mad * sigma_A_sq)
        w_repr = math.exp(-gamma_repr * var_spread)
        wA = w_cnt * w_mad * w_repr
        if wA <= 0.0 or not np.isfinite(z_med) or z_med <= 0.0:
            continue
        Lambda_A[i] = wA / sigma_ell_sq
        theta_A[i] = Lambda_A[i] * z_med

    return Lambda_A, theta_A


# -----------------------------------------------------------------------------
# Route B: ray–plane intersection → (z*, σ², w); _route_b_natural_params → (ΛB, θB)
# -----------------------------------------------------------------------------


def ray_from_pixel(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Unit ray direction in camera frame for pixel (u,v). Origin at camera center (0,0,0)."""
    dx = (float(u) - cx) / fx
    dy = (float(v) - cy) / fy
    r = np.array([dx, dy, 1.0], dtype=np.float64)
    nrm = np.linalg.norm(r)
    if nrm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return r / nrm


def _distance_point_to_ray(points: np.ndarray, ray_dir: np.ndarray) -> np.ndarray:
    """Squared distance of each point to the ray through origin with direction ray_dir. (N,)"""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    ray_dir = np.asarray(ray_dir, dtype=np.float64).ravel()[:3]
    ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-12)
    # proj = (points @ ray_dir) * ray_dir; dist_sq = |points - proj|^2
    proj_len = points @ ray_dir
    proj = proj_len[:, np.newaxis] * ray_dir
    diff = points - proj
    return np.sum(diff * diff, axis=1)


def _softplus(x: float) -> float:
    """Continuous softplus: log(1+exp(x)). Stable for large |x|."""
    x = float(x)
    if x > 20.0:
        return x
    if x < -20.0:
        return 0.0
    return math.log(1.0 + math.exp(x))


def _fit_plane_weighted(
    points: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Weighted PCA plane: min_{n,d} sum_k w_k (n'x_k + d)^2 s.t. ||n||=1.
    Weighted centroid x̄, weighted cov S = sum w_k (x_k - x̄)(x_k - x̄)'; n = eigenvector of λ_min.
    Returns (centroid (3,), normal (3,) unit, eigvals ascending (3,), sigma_perp_sq).
    sigma_perp_sq = (1/sum w_k) sum w_k (n'(x_k - x̄))^2. Fixed-cost; no iteration.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    n_pt = points.shape[0]
    if weights is None:
        weights = np.ones(n_pt, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).ravel()[:n_pt]
    if weights.size != n_pt:
        weights = np.ones(n_pt, dtype=np.float64)
    w_sum = float(np.sum(weights)) + 1e-300
    centroid = np.average(points, axis=0, weights=weights)
    centered = points - centroid
    # S = sum w_k (x_k - x̄)(x_k - x̄)'
    S = (centered.T * weights) @ centered / w_sum
    S = 0.5 * (S + S.T) + 1e-12 * np.eye(3)
    eigvals, eigvecs = np.linalg.eigh(S)
    # Ascending: λ1 ≤ λ2 ≤ λ3; normal = eigenvector of λ1 (smallest)
    normal = eigvecs[:, 0]
    if normal[2] < 0:
        normal = -normal
    res_sq = (centered @ normal) ** 2
    sigma_perp_sq = float(np.average(res_sq, weights=weights))
    return centroid, normal, eigvals, max(sigma_perp_sq, 0.0)


def _route_b_ray_plane(
    points_camera_frame: np.ndarray,
    uv_query: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: LidarCameraDepthFusionConfig,
    *,
    point_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Route B internal: per-query (u,v) → (z*, σ², w). Ray–plane intersection + w = w_angle * w_planar * w_res * w_z.
    _route_b_natural_params converts to (ΛB, θB).
    """
    uv_query = np.asarray(uv_query, dtype=np.float64)
    if uv_query.ndim == 1:
        uv_query = uv_query.reshape(1, 2)
    M = uv_query.shape[0]
    z_out = np.full(M, np.nan, dtype=np.float64)
    sigma_sq_out = np.full(M, np.nan, dtype=np.float64)
    weight_out = np.zeros(M, dtype=np.float64)

    if points_camera_frame.size == 0:
        return z_out, sigma_sq_out, weight_out

    uv, z = _project_camera(points_camera_frame, fx, fy, cx, cy)
    valid = z > 0.0
    points = np.asarray(points_camera_frame, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = points[valid]
    uv = uv[valid]
    z = z[valid]
    if points.shape[0] < config.lidar_plane_fit_min_points:
        return z_out, sigma_sq_out, weight_out

    r_pix = config.lidar_projection_radius_pix
    K = min(config.lidar_ray_plane_fit_max_points, points.shape[0])
    base_sigma = config.lidar_depth_base_sigma_m
    var_min = config.depth_var_min_m2
    delta = config.plane_intersection_delta
    eps = config.plane_fit_eps
    alpha = config.plane_angle_sigmoid_alpha
    t_angle = config.plane_angle_sigmoid_t
    beta = config.plane_planarity_sigmoid_beta
    rho0 = config.plane_planarity_rho0
    gamma = config.plane_residual_exp_gamma

    for i in range(M):
        uq, vq = uv_query[i, 0], uv_query[i, 1]
        ray_dir = ray_from_pixel(uq, vq, fx, fy, cx, cy)
        dist_sq_pix = (uv[:, 0] - uq) ** 2 + (uv[:, 1] - vq) ** 2
        in_radius = dist_sq_pix <= (r_pix * r_pix)
        if not np.any(in_radius):
            continue
        pts_near = points[in_radius]
        dist_to_ray = _distance_point_to_ray(pts_near, ray_dir)
        n_cand = pts_near.shape[0]
        if n_cand < config.lidar_plane_fit_min_points:
            continue
        k_use = min(K, n_cand)
        idx = np.argpartition(dist_to_ray, k_use - 1)[:k_use]
        pts_fit = pts_near[idx]
        w_fit = None
        if point_weights is not None:
            pw = np.asarray(point_weights, dtype=np.float64).ravel()
            if pw.size == points_camera_frame.shape[0]:
                w_valid = pw[valid]
                w_near = w_valid[in_radius]
                w_fit = w_near[idx]

        centroid, normal, eigvals, sigma_perp_sq = _fit_plane_weighted(pts_fit, w_fit)
        # Ray origin C = 0; plane n'(x − x̄) = 0 ⇒ z* = n'x̄/(n'r̂+δ) (continuous)
        ndotr = float(np.dot(normal, ray_dir)) + delta
        z_ell_raw = float(np.dot(normal, centroid)) / ndotr
        # Continuous depth: softplus(z_raw − z_min) + z_min (no reject; behind-camera → near z_min)
        z_min = config.depth_min_m
        z_ell_i = _softplus(z_ell_raw - z_min) + z_min
        # Downweight when behind camera (z_raw < z_min): multiply w_plane by sigmoid(z_raw − z_min)
        w_behind = _sigmoid(z_ell_raw - z_min) if z_ell_raw < z_min else 1.0

        # σz,ℓ² = σ⊥²/((n'r̂)²+δ); cap so precision doesn't underflow (float safety)
        ndotr_sq = float(np.dot(normal, ray_dir)) ** 2 + delta
        sigma_ell_sq_i = base_sigma * base_sigma + sigma_perp_sq / max(ndotr_sq, delta)
        sigma_ell_sq_i = max(sigma_ell_sq_i, var_min)
        sigma_ell_sq_i = min(sigma_ell_sq_i, config.depth_sigma_max_sq)

        # Continuous reliability (no gating)
        ndotr_abs = abs(float(np.dot(normal, ray_dir)))
        w_angle = _sigmoid(alpha * (ndotr_abs - t_angle))
        lam1, lam2, lam3 = eigvals[0], eigvals[1], eigvals[2]
        lam2 = max(lam2, eps)
        lam3 = max(lam3, eps)
        rho = float(lam2 / (lam3 + eps))
        w_planar = _sigmoid(beta * (rho - rho0))
        w_res = math.exp(-gamma * sigma_perp_sq)
        w_plane = w_angle * w_planar * w_res * w_behind
        # w_z = σ(α_z(z* − z_min)) for behind/min-depth (continuous)
        alpha_z = config.depth_min_sigmoid_alpha_z
        w_z = _sigmoid(alpha_z * (z_ell_i - z_min))
        w_plane = w_plane * w_z

        z_out[i] = z_ell_i
        sigma_sq_out[i] = sigma_ell_sq_i
        weight_out[i] = max(w_plane, 0.0)
    return z_out, sigma_sq_out, weight_out


def _route_b_natural_params(
    points_camera_frame: np.ndarray,
    uv_query: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: LidarCameraDepthFusionConfig,
    *,
    point_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Route B: per-query (u,v) → (ΛB, θB). Ray–plane intersection + wB = w_angle * w_planar * w_res * w_z.
    ΛB = wB/σ²_B, θB = ΛB*z*. Invalid/weak → (0, 0). Always defined, continuous.
    """
    z_out, sigma_sq_out, weight_out = _route_b_ray_plane(
        points_camera_frame,
        uv_query,
        fx, fy, cx, cy,
        config,
        point_weights=point_weights,
    )
    M = z_out.shape[0]
    Lambda_B = np.zeros(M, dtype=np.float64)
    theta_B = np.zeros(M, dtype=np.float64)
    var_min = config.depth_var_min_m2
    for i in range(M):
        z_i = z_out[i]
        sigma_sq_i = sigma_sq_out[i]
        w_i = weight_out[i]
        if not np.isfinite(z_i) or z_i <= 0.0 or not np.isfinite(sigma_sq_i) or sigma_sq_i <= 0.0 or w_i <= 0.0:
            continue
        sigma_sq_i = max(sigma_sq_i, var_min)
        Lambda_B[i] = w_i / sigma_sq_i
        theta_B[i] = Lambda_B[i] * z_i
    return Lambda_B, theta_B


def lidar_depth_evidence(
    points_camera_frame: np.ndarray,
    uv_query: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: LidarCameraDepthFusionConfig,
    *,
    point_weights: Optional[np.ndarray] = None,
    return_diag: bool = False,
) -> Tuple[np.ndarray, np.ndarray, ...]:
    """
    One LiDAR depth evidence function (I0.2). Returns (Λ_ell, θ_ell) per query.

    Internally computes Route A (project + robust sample) and Route B (ray–plane);
    combines as mixture-of-experts: Λ_ell = ΛA + ΛB, θ_ell = θA + θB.
    Always defined, continuous; Λ_ell → 0, θ_ell → 0 when not applicable (camera-only behavior).

    Args:
        points_camera_frame: (N, 3) in camera frame.
        uv_query: (M, 2) query pixel coordinates (u, v).
        fx, fy, cx, cy: pinhole intrinsics.
        config: fusion config (gamma_lidar optional modality dial).
        point_weights: optional (N,) weights for Route B plane fit; None → uniform.
        return_diag: if True, return (Lambda_ell, theta_ell, diag) with diag dict.

    Returns:
        Lambda_ell: (M,) precision (1/σ²) along ray; 0 when no LiDAR support.
        theta_ell: (M,) natural param θ = Λ*ẑ; 0 when no LiDAR support.
        If return_diag: diag with keys Lambda_A, theta_A, Lambda_B, theta_B (optional).
    """
    Lambda_A, theta_A = _route_a_natural_params(
        points_camera_frame, uv_query, fx, fy, cx, cy, config
    )
    Lambda_B, theta_B = _route_b_natural_params(
        points_camera_frame, uv_query, fx, fy, cx, cy, config,
        point_weights=point_weights,
    )
    Lambda_ell = Lambda_A + Lambda_B
    theta_ell = theta_A + theta_B
    gamma = config.gamma_lidar
    if gamma != 1.0:
        Lambda_ell = Lambda_ell * gamma
        theta_ell = theta_ell * gamma
    if return_diag:
        diag: Dict[str, Any] = {
            "Lambda_A": Lambda_A.copy(),
            "theta_A": theta_A.copy(),
            "Lambda_B": Lambda_B.copy(),
            "theta_B": theta_B.copy(),
        }
        return Lambda_ell, theta_ell, diag
    return Lambda_ell, theta_ell


# -----------------------------------------------------------------------------
# Backproject and closed-form cov (for splat_prep: fused depth -> 3D + cov)
# -----------------------------------------------------------------------------


def backproject_camera(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Backproject pixel (u,v) and depth z to 3D in camera frame. Returns (3,) xyz."""
    x = (float(u) - cx) * float(z) / fx
    y = (float(v) - cy) * float(z) / fy
    return np.array([x, y, float(z)], dtype=np.float64)


def backprojection_cov_camera(
    u: float,
    v: float,
    z: float,
    var_u: float,
    var_v: float,
    var_z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Closed-form 3x3 covariance for pinhole backprojection with independent Gaussian (u,v,z).
    Same formula as visual_feature_extractor._backprojection_cov_closed_form.
    """
    du = float(u - cx)
    dv = float(v - cy)
    z = float(z)
    vu, vv, vz = max(var_u, 0.0), max(var_v, 0.0), max(var_z, 0.0)
    var_x = (z * z * vu + du * du * vz + vu * vz) / (fx * fx)
    var_y = (z * z * vv + dv * dv * vz + vv * vz) / (fy * fy)
    cov_xy = (du * dv * vz) / (fx * fy)
    cov_xz = (du * vz) / fx
    cov_yz = (dv * vz) / fy
    return np.array(
        [
            [var_x, cov_xy, cov_xz],
            [cov_xy, var_y, cov_yz],
            [cov_xz, cov_yz, vz],
        ],
        dtype=np.float64,
    )
