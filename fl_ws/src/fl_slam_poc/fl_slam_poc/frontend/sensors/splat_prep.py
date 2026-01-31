"""
Splat preparation: orchestrate extractor + LiDAR depth fusion + backproject + 3DGS splat.

I0.2: One LiDAR depth evidence lidar_depth_evidence(u,v,...) -> (Λ_ell, θ_ell).
PoE: Λf = Λc + Λ_ell, θf = θc + θ_ell; z_f = θf/Λf, σf² = 1/Λf. Camera-only when Λ_ell → 0.
"""

from __future__ import annotations

from typing import List

import numpy as np

from fl_slam_poc.frontend.sensors.lidar_camera_depth_fusion import (
    LidarCameraDepthFusionConfig,
    backproject_camera,
    backprojection_cov_camera,
    lidar_depth_evidence,
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
    _, logdet = np.linalg.slogdet(cov)
    return info, float(logdet)


def splat_prep_fused(
    extraction_result: ExtractionResult,
    points_camera_frame: np.ndarray,
    intrinsics: PinholeIntrinsics,
    config: LidarCameraDepthFusionConfig,
    *,
    pixel_sigma: float = 1.0,
) -> List[Feature3D]:
    """
    Fused depth splat path: one LiDAR depth evidence API; PoE Λf = Λc + Λ_ell, θf = θc + θ_ell.

    Uses lidar_depth_evidence(u,v,...) -> (Λ_ell, θ_ell); fuse with camera (Λc, θc) from feature meta.
    Returns list of Feature3D with fused xyz/cov/info/canonical_theta; same descriptor/weight/mu_app
    as input. Caller can pass each to extractor.feature_to_splat_3dgs() for 3DGS export.
    """
    features_in = extraction_result.features
    if not features_in:
        return []

    uv = np.array([[f.u, f.v] for f in features_in], dtype=np.float64)
    Lambda_ell, theta_ell = lidar_depth_evidence(
        points_camera_frame,
        uv,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        config,
    )

    fused: List[Feature3D] = []
    var_min = config.depth_var_min_m2
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    w_c = config.depth_fusion_weight_camera
    w_ell = config.depth_fusion_weight_lidar

    for i, feat in enumerate(features_in):
        Lambda_c = feat.meta.get("depth_Lambda_c", 0.0)
        theta_c = feat.meta.get("depth_theta_c", 0.0)
        # Reliability-aware: Λ←wΛ, θ←wθ (I0.2, §2.3)
        Lambda_c_scaled = w_c * Lambda_c
        theta_c_scaled = w_c * theta_c
        Lambda_ell_i = w_ell * Lambda_ell[i]
        theta_ell_i = w_ell * theta_ell[i]

        Lambda_f = Lambda_c_scaled + Lambda_ell_i
        theta_f = theta_c_scaled + theta_ell_i

        if Lambda_f <= 0.0 or not np.isfinite(Lambda_f) or not np.isfinite(theta_f):
            fused.append(feat)
            continue

        z_f = theta_f / Lambda_f
        sigma_f_sq = 1.0 / Lambda_f
        if not np.isfinite(z_f) or z_f <= 0.0 or not np.isfinite(sigma_f_sq) or sigma_f_sq <= 0.0:
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
        meta_fused["depth_Lambda_c"] = Lambda_f
        meta_fused["depth_theta_c"] = float(theta_f)

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
