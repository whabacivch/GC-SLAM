"""
LiDAR–camera depth fusion via natural-parameter addition on depth along camera rays.

Plan: lidar-camera_splat_fusion_and_bev_ot. Part A: depth latent z along ray;
Λ = 1/σ², θ = Λ·ẑ; fuse Λf = Λc + Λℓ, θf = θc + θℓ; ẑf = θf/Λf, σf² = 1/Λf.

All parameters from config; no magic numbers. Optional Student-t weight scaling
is documented as a robustness option (continuous weight w ∈ (0,1], scale Λ by w
before adding); not a gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Config (no magic numbers)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LidarCameraDepthFusionConfig:
    """Configuration for LiDAR–camera depth fusion and LiDAR ray depth (Route A)."""

    # Depth fusion: optional Student-t–style weight scaling (continuous w ∈ (0,1])
    # Scale Λc, Λℓ by w before adding; document as robustness option. Default 1.0 = no scaling.
    depth_fusion_weight_camera: float = 1.0
    depth_fusion_weight_lidar: float = 1.0

    # LiDAR ray depth (Route A): per-pixel aggregation
    lidar_projection_radius_pix: float = 3.0  # radius in pixel space for nearest/median
    lidar_plane_fit_min_points: int = 3  # min points for local stats (fixed; not RANSAC)
    lidar_depth_base_sigma_m: float = 0.02  # base depth std (m) from range noise model
    lidar_incidence_sigma_scale: float = 1.0  # scale σ by 1/cos(incidence) if desired (1 = off)
    # Minimum depth variance (m²) for numerical stability
    depth_var_min_m2: float = 1e-8

    # Route B: ray–plane intersection (fixed-cost; continuous; no reject)
    lidar_ray_plane_fit_max_points: int = 64  # fixed K: max points per ray for plane fit (no RANSAC)
    plane_intersection_delta: float = 1e-6  # δ in z* = n'(x̄−C)/(n'r̂+δ); continuous, no division-by-zero
    # Continuous reliability (no gating): w_angle, w_planar, w_res; Λℓ ← w_angle * w_planar * w_res * Λℓ
    plane_angle_sigmoid_alpha: float = 10.0  # w_angle = sigmoid(α(|n'r̂| − t))
    plane_angle_sigmoid_t: float = 0.1  # threshold for angle sigmoid
    plane_planarity_sigmoid_beta: float = 5.0  # w_planar = sigmoid(β(ρ − ρ0)), ρ = λ2/(λ3+ε)
    plane_planarity_rho0: float = 0.3  # planarity threshold
    plane_residual_exp_gamma: float = 100.0  # w_res = exp(−γ σ⊥²)
    plane_fit_eps: float = 1e-12  # ε for ρ denominator and numerics
    # Continuous depth handling (no reject): behind-camera / negative z
    depth_min_m: float = 0.05  # z_min for softplus clamp: z ← softplus(z_raw − z_min) + z_min
    # Cap σz² for float safety: when |n'r̂| is very small, σz² = σ⊥²/((n'r̂)²+δ) becomes huge
    # (precision → 0); correct behavior—depth_sigma_max_sq prevents float blow-up.
    depth_sigma_max_sq: float = 1e4  # cap σz,ℓ² so precision doesn't underflow (m²)


# -----------------------------------------------------------------------------
# Depth natural params (1D Gaussian along ray)
# -----------------------------------------------------------------------------


def depth_natural_params(
    z_hat: float,
    sigma_sq: float,
    var_min: float = 1e-8,
) -> Tuple[float, float]:
    """
    Scalar natural parameters for 1D depth Gaussian: Λ = 1/σ², θ = Λ·ẑ.

    Args:
        z_hat: depth estimate along ray (m).
        sigma_sq: depth variance (m²). Must be positive.
        var_min: minimum variance for stability (default 1e-8).

    Returns:
        (Lambda, theta) with Lambda = 1 / max(sigma_sq, var_min), theta = Lambda * z_hat.
    """
    sigma_sq = max(float(sigma_sq), var_min)
    Lambda = 1.0 / sigma_sq
    theta = Lambda * float(z_hat)
    return (Lambda, theta)


def fuse_depth_natural_params(
    Lambda_c: float,
    theta_c: float,
    Lambda_ell: float,
    theta_ell: float,
    w_c: float = 1.0,
    w_ell: float = 1.0,
) -> Tuple[float, float]:
    """
    Fuse camera and LiDAR depth by natural-parameter addition (same ray).

    Λf = w_c·Λc + w_ell·Λℓ, θf = w_c·θc + w_ell·θℓ; ẑf = θf/Λf, σf² = 1/Λf.

    Optional weights w_c, w_ell ∈ (0, 1] scale the information (e.g. Student-t
    robustness); no gate. If both Λ are 0, returns (0.0, float('inf')) and
    caller should treat as no observation.

    Args:
        Lambda_c, theta_c: camera depth natural params.
        Lambda_ell, theta_ell: LiDAR depth natural params.
        w_c, w_ell: optional continuous weights (default 1.0).

    Returns:
        (z_f, sigma_f_sq) in m and m². sigma_f_sq = inf if Lambda_f <= 0.
    """
    Lambda_f = w_c * Lambda_c + w_ell * Lambda_ell
    theta_f = w_c * theta_c + w_ell * theta_ell
    if Lambda_f <= 0.0:
        return (0.0, float("inf"))
    z_f = theta_f / Lambda_f
    sigma_f_sq = 1.0 / Lambda_f
    return (z_f, sigma_f_sq)


# -----------------------------------------------------------------------------
# LiDAR ray depth (Route A): project points to camera, per-(u,v) z_ell, σ_ℓ²
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


def lidar_ray_depth_route_a(
    points_camera_frame: np.ndarray,
    uv_query: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: LidarCameraDepthFusionConfig,
    *,
    use_median: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-query (u,v) get LiDAR depth z_ell and variance sigma_ell_sq (Route A).

    Points are in camera frame (N,3). Project to pixels; for each query (u,v)
    find points within lidar_projection_radius_pix, take median (or nearest)
    depth. sigma_ell_sq from config (base + optional incidence); no hidden iteration.

    Args:
        points_camera_frame: (N, 3) in camera frame.
        uv_query: (M, 2) query pixel coordinates (u, v).
        fx, fy, cx, cy: pinhole intrinsics.
        config: fusion/config (radius, base sigma, var_min).
        use_median: if True use median depth in radius, else nearest by reprojection.

    Returns:
        z_ell: (M,) depth in m; NaN where no points in radius.
        sigma_ell_sq: (M,) variance in m²; same shape.
    """
    uv_query = np.asarray(uv_query, dtype=np.float64)
    if uv_query.ndim == 1:
        uv_query = uv_query.reshape(1, 2)
    M = uv_query.shape[0]
    if points_camera_frame.size == 0:
        return np.full(M, np.nan, dtype=np.float64), np.full(M, np.nan, dtype=np.float64)

    uv, z = _project_camera(points_camera_frame, fx, fy, cx, cy)
    # Filter to valid depth
    valid = z > 0.0
    uv = uv[valid]
    z = z[valid]
    if uv.size == 0:
        return np.full(M, np.nan, dtype=np.float64), np.full(M, np.nan, dtype=np.float64)

    r = config.lidar_projection_radius_pix
    z_out = np.full(M, np.nan, dtype=np.float64)
    sigma_sq_out = np.full(M, np.nan, dtype=np.float64)
    base_sigma = config.lidar_depth_base_sigma_m
    var_min = config.depth_var_min_m2

    for i in range(M):
        uq, vq = uv_query[i, 0], uv_query[i, 1]
        dist_sq = (uv[:, 0] - uq) ** 2 + (uv[:, 1] - vq) ** 2
        in_radius = dist_sq <= (r * r)
        if not np.any(in_radius):
            continue
        z_near = z[in_radius]
        if use_median:
            z_ell_i = float(np.median(z_near))
            # Variance: spread of depths in neighborhood + base sensor variance
            n_pt = z_near.size
            if n_pt >= config.lidar_plane_fit_min_points:
                var_spread = float(np.var(z_near))
            else:
                var_spread = 0.0
            sigma_ell_sq_i = base_sigma * base_sigma + var_spread
        else:
            idx = np.argmin(np.sqrt(dist_sq[in_radius]))
            z_ell_i = float(z_near.flat[idx])
            sigma_ell_sq_i = base_sigma * base_sigma
        sigma_ell_sq_i = max(sigma_ell_sq_i, var_min)
        z_out[i] = z_ell_i
        sigma_sq_out[i] = sigma_ell_sq_i

    return z_out, sigma_sq_out


# -----------------------------------------------------------------------------
# LiDAR ray depth (Route B): ray–plane intersection (fixed-cost; continuous weight)
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


def _sigmoid(x: float) -> float:
    """Continuous sigmoid 1/(1+exp(-x)); no gate."""
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    t = math.exp(x)
    return t / (1.0 + t)


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


def lidar_ray_depth_route_b(
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
    Per-query (u,v) get LiDAR depth z_ell and sigma_ell_sq via ray–plane intersection (Route B).

    Weighted PCA plane: min sum w_k (n'x_k+d)^2; n = eigenvector of smallest eigenvalue.
    Intersection (continuous): z* = n'(x̄−C)/(n'r̂+δ), C=0. σz,ℓ² = σ⊥²/((n'r̂)²+δ).
    Continuous reliability: w_angle = sigmoid(α(|n'r̂|−t)), w_planar = sigmoid(β(ρ−ρ0)),
    w_res = exp(−γ σ⊥²); weight_plane = w_angle * w_planar * w_res. No gate.

    Args:
        points_camera_frame: (N, 3) in camera frame.
        uv_query: (M, 2) query pixel coordinates (u, v).
        fx, fy, cx, cy: pinhole intrinsics.
        config: fusion config (Route B: plane_intersection_delta, angle/planarity/residual params).
        point_weights: optional (N,) weights for plane fit; None → uniform.

    Returns:
        z_ell: (M,) depth in m; NaN where no points in radius.
        sigma_ell_sq: (M,) variance in m².
        weight_plane: (M,) continuous weight for scaling Λℓ (w_angle * w_planar * w_res).
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

        z_out[i] = z_ell_i
        sigma_sq_out[i] = sigma_ell_sq_i
        weight_out[i] = max(w_plane, 0.0)
    return z_out, sigma_sq_out, weight_out


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
