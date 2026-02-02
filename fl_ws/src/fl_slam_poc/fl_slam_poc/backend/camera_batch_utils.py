"""
Build MeasurementBatch from camera splats (Feature3D list from VisualFeatureExtractor + splat_prep).

Used by backend_node to wire camera into the pipeline. Converts List[Feature3D] to the arrays
required by measurement_batch_from_camera_splats.
"""

from __future__ import annotations

from typing import List

import numpy as np

from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common import constants
from fl_slam_poc.backend.structures.measurement_batch import (
    measurement_batch_from_camera_splats,
)
from fl_slam_poc.frontend.sensors.visual_feature_extractor import Feature3D


def feature_list_to_camera_batch(
    features: List[Feature3D],
    timestamp_sec: float,
    n_feat: int = constants.GC_N_FEAT,
    n_surfel: int = constants.GC_N_SURFEL,
    eps_lift: float = constants.GC_EPS_LIFT,
):
    """
    Convert List[Feature3D] (from splat_prep_fused) to MeasurementBatch (camera slice only).

    Caller merges with LiDAR surfels via extract_lidar_surfels(..., base_batch=batch).
    """
    if not features:
        from fl_slam_poc.backend.structures.measurement_batch import create_empty_measurement_batch
        return create_empty_measurement_batch(n_feat=n_feat, n_surfel=n_surfel)

    n = min(len(features), n_feat)
    positions = np.array([f.xyz for f in features[:n]], dtype=np.float64)
    covariances = np.array([f.cov_xyz for f in features[:n]], dtype=np.float64)
    weights = np.array([f.weight for f in features[:n]], dtype=np.float64)
    kappas = np.array([f.kappa_app for f in features[:n]], dtype=np.float64)
    timestamps = np.full(n, timestamp_sec, dtype=np.float64)

    # Colors: from Feature3D.color (RGB [0,1]) when present, else default gray
    default_gray = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    colors_list = []
    for f in features[:n]:
        c = getattr(f, "color", None)
        if c is not None and np.size(c) >= 3:
            colors_list.append(np.asarray(c, dtype=np.float64).ravel()[:3])
        else:
            colors_list.append(default_gray)
    colors = np.array(colors_list, dtype=np.float64)

    # Directions: mu_app if present (unit), else view direction xyz / norm(xyz)
    directions = np.zeros((n, 3), dtype=np.float64)
    for i, f in enumerate(features[:n]):
        if f.mu_app is not None:
            d = np.asarray(f.mu_app, dtype=np.float64)
        else:
            d = np.asarray(f.xyz, dtype=np.float64)
        norm_d = np.linalg.norm(d)
        directions[i] = d / (norm_d + 1e-12)

    return measurement_batch_from_camera_splats(
        positions=jnp.array(positions),
        covariances=jnp.array(covariances),
        directions=jnp.array(directions),
        kappas=jnp.array(kappas),
        weights=jnp.array(weights),
        timestamps=jnp.array(timestamps),
        colors=jnp.array(colors),
        n_feat=n_feat,
        n_surfel=n_surfel,
        eps_lift=eps_lift,
    )
