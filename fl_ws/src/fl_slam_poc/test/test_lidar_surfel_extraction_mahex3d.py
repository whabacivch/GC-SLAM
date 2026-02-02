"""
LiDAR surfel extraction (MA-hex 3D, JAX-only) smoke tests.

These tests use a *small* MA-hex grid to keep runtime bounded in unit tests.
"""

import numpy as np

from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.backend.operators.lidar_surfel_extraction import (
    SurfelExtractionConfig,
    extract_lidar_surfels,
)


def test_extract_lidar_surfels_mahex3d_produces_valid_lidar_slice():
    # Two tight clusters, each should form at least one valid cell when voxel_size_m is large.
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.01, size=(20, 3))
    cluster_b = rng.normal(loc=[1.0, 1.0, 0.0], scale=0.01, size=(20, 3))
    points = np.vstack([cluster_a, cluster_b]).astype(np.float64)
    timestamps = np.linspace(0.0, 1.0, points.shape[0], dtype=np.float64)
    weights = np.ones((points.shape[0],), dtype=np.float64)

    cfg = SurfelExtractionConfig(
        n_surfel=8,
        n_feat=4,
        voxel_size_m=0.5,
        min_points_per_voxel=5,
        hex3d_num_cells_1=8,
        hex3d_num_cells_2=8,
        hex3d_num_cells_z=2,
        hex3d_max_occupants=32,
    )

    batch, cert, _ = extract_lidar_surfels(
        points=jnp.array(points),
        timestamps=jnp.array(timestamps),
        weights=jnp.array(weights),
        config=cfg,
        base_batch=None,
        chart_id="GC-RIGHT-01",
        anchor_id="test_surfel_extraction",
    )

    assert batch.n_surfel == 8
    assert batch.n_feat == 4
    assert 0 <= batch.n_lidar_valid <= batch.n_surfel

    lidar_mask = batch.valid_mask[batch.lidar_slice]
    assert int(jnp.sum(lidar_mask)) == batch.n_lidar_valid

    # All populated precisions should be finite
    if batch.n_lidar_valid > 0:
        Lambdas = batch.Lambdas[batch.lidar_slice][: batch.n_lidar_valid]
        assert bool(jnp.all(jnp.isfinite(Lambdas)))
        assert bool(jnp.all(jnp.isfinite(batch.weights[batch.lidar_slice][: batch.n_lidar_valid])))

    # Cert is approximate and has triggers (single-path, no fallback)
    assert cert.exact is False
    assert cert.approximation_triggers
