"""
Splat preparation: orchestrate extractor + LiDAR depth fusion + backproject + 3DGS splat.

Plan: lidar-camera_splat_fusion_and_bev_ot. Run extractor -> (u,v,z_c,sigma_c_sq);
get (z_ell, sigma_ell_sq) from LiDAR; fuse -> (z_f, sigma_f_sq); backproject + cov;
output Feature3D list (caller can call feature_to_splat_3dgs on each). No backend_node/pipeline edits.
"""

from __future__ import annotations

from typing import List

import numpy as np

from fl_slam_poc.frontend.sensors.lidar_camera_depth_fusion import (
    LidarCameraDepthFusionConfig,
    backproject_camera,
    backprojection_cov_camera,
    depth_natural_params,
    fuse_depth_natural_params,
    lidar_ray_depth_route_a,
    lidar_ray_depth_route_b,
)
from fl_slam_poc.frontend.sensors.visual_feature_extractor import (
    ExtractionResult,
    Feature3D,
    PinholeIntrinsics,
)

_LOG_2PI = np.log(2.0 * np.pi)


def _precision_and_logdet(cov_xyz: np.ndarray, reg: float = 1e-9) -> tuple[np.ndarray, float]:
    """(3,3) cov -> info, logdet. Regularize for invertibility."""
    cov = np.asarray(cov_xyz, dtype=np.float64).reshape(3, 3) + reg * np.eye(3)
    info = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    return info, float(logdet)


def splat_prep_fused(
    extraction_result: ExtractionResult,
    points_camera_frame: np.ndarray,
    intrinsics: PinholeIntrinsics,
    config: LidarCameraDepthFusionConfig,
    *,
    pixel_sigma: float = 1.0,
    use_route_b: bool = False,
) -> List[Feature3D]:
    """
    Fused depth splat path: for each feature get z_ell from LiDAR, fuse with z_c, backproject.

    If use_route_b=True, uses ray–plane intersection (Route B) and scales Λℓ by plane-fit weight.
    Returns list of Feature3D with fused xyz/cov/info/canonical_theta; same descriptor/weight/mu_app
    as input. Caller can pass each to extractor.feature_to_splat_3dgs() for 3DGS export.
    """
    features_in = extraction_result.features
    if not features_in:
        return []

    uv = np.array([[f.u, f.v] for f in features_in], dtype=np.float64)
    if use_route_b:
        z_ell, sigma_ell_sq, weight_plane = lidar_ray_depth_route_b(
            points_camera_frame,
            uv,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            config,
        )
    else:
        z_ell, sigma_ell_sq = lidar_ray_depth_route_a(
            points_camera_frame,
            uv,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            config,
            use_median=True,
        )
        weight_plane = None

    fused: List[Feature3D] = []
    var_min = config.depth_var_min_m2
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy

    for i, feat in enumerate(features_in):
        Lambda_c = feat.meta.get("depth_Lambda_c", 0.0)
        theta_c = feat.meta.get("depth_theta_c", 0.0)
        z_c = feat.meta.get("depth_m", np.nan)
        if not np.isfinite(z_c):
            z_c = 0.0

        z_ell_i = z_ell[i]
        sigma_ell_sq_i = sigma_ell_sq[i]
        if np.isfinite(z_ell_i) and np.isfinite(sigma_ell_sq_i) and sigma_ell_sq_i > 0:
            Lambda_ell, theta_ell = depth_natural_params(
                z_ell_i, sigma_ell_sq_i, var_min=var_min
            )
            w_ell = config.depth_fusion_weight_lidar
            if weight_plane is not None and np.isfinite(weight_plane[i]):
                w_ell = w_ell * float(weight_plane[i])
            z_f, sigma_f_sq = fuse_depth_natural_params(
                Lambda_c, theta_c, Lambda_ell, theta_ell,
                w_c=config.depth_fusion_weight_camera,
                w_ell=w_ell,
            )
        else:
            z_f = z_c if Lambda_c > 0 else 0.0
            sigma_f_sq = 1.0 / Lambda_c if Lambda_c > 0 else float("inf")

        if not np.isfinite(z_f) or z_f <= 0 or not np.isfinite(sigma_f_sq) or sigma_f_sq <= 0:
            fused.append(feat)
            continue

        var_z = max(sigma_f_sq, var_min)
        xyz = backproject_camera(feat.u, feat.v, z_f, fx, fy, cx, cy)
        cov_xyz = backprojection_cov_camera(
            feat.u, feat.v, z_f,
            pixel_sigma ** 2, pixel_sigma ** 2, var_z,
            fx, fy, cx, cy,
        )
        info_xyz, logdet_cov = _precision_and_logdet(cov_xyz)
        canonical_theta = info_xyz @ xyz
        canonical_log_partition = (
            0.5 * float(xyz @ info_xyz @ xyz)
            + 0.5 * float(logdet_cov)
            + 1.5 * _LOG_2PI
        )

        meta_fused = dict(feat.meta)
        meta_fused["depth_m"] = float(z_f)
        meta_fused["depth_sigma_c_sq"] = float(sigma_f_sq)
        meta_fused["depth_Lambda_c"] = 1.0 / sigma_f_sq
        meta_fused["depth_theta_c"] = (1.0 / sigma_f_sq) * z_f

        fused.append(
            Feature3D(
                u=feat.u,
                v=feat.v,
                xyz=xyz,
                cov_xyz=cov_xyz,
                info_xyz=info_xyz,
                logdet_cov=float(logdet_cov),
                canonical_theta=canonical_theta,
                canonical_log_partition=float(canonical_log_partition),
                desc=feat.desc.copy(),
                weight=feat.weight,
                meta=meta_fused,
                mu_app=feat.mu_app.copy() if feat.mu_app is not None else None,
                kappa_app=feat.kappa_app,
            )
        )
    return fused
