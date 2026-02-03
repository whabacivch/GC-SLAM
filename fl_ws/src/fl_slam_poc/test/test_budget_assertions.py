"""
Budget assertion tests for fixed-cost GC v2.

Reference: docs/GC_SLAM.md §5.7.7 and plan Phase 3.2.
"""

import numpy as np

from fl_slam_poc.common import constants
from fl_slam_poc.backend.structures import (
    create_empty_tile,
    extract_primitive_map_view,
)
from fl_slam_poc.backend.structures.measurement_batch import create_empty_measurement_batch, MeasurementBatch
from fl_slam_poc.backend.operators.primitive_association import (
    associate_primitives_ot,
    AssociationConfig,
)


def _build_measurement_batch(n_feat: int, n_surfel: int, n_valid: int) -> MeasurementBatch:
    batch = create_empty_measurement_batch(n_feat=n_feat, n_surfel=n_surfel)
    n_total = batch.n_total
    n_valid = min(n_valid, n_total)

    Lambdas = batch.Lambdas.at[:n_valid].set(np.eye(3, dtype=np.float64))
    thetas = batch.thetas.at[:n_valid].set(np.zeros((n_valid, 3), dtype=np.float64))
    etas = batch.etas.at[:n_valid, 0].set(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    weights = batch.weights.at[:n_valid].set(1.0)
    valid_mask = batch.valid_mask.at[:n_valid].set(True)
    timestamps = batch.timestamps.at[:n_valid].set(0.0)
    colors = batch.colors.at[:n_valid].set(np.zeros((n_valid, 3), dtype=np.float64))

    return MeasurementBatch(
        Lambdas=Lambdas,
        thetas=thetas,
        etas=etas,
        weights=weights,
        sources=batch.sources,
        source_indices=batch.source_indices,
        valid_mask=valid_mask,
        timestamps=timestamps,
        colors=colors,
        n_feat=batch.n_feat,
        n_surfel=batch.n_surfel,
        n_camera_valid=n_valid if n_surfel == 0 else 0,
        n_lidar_valid=0 if n_surfel == 0 else n_valid,
    )


def _build_tile_view(m_tile: int, n_valid: int):
    tile = create_empty_tile(tile_id=0, m_tile=m_tile)
    n_valid = min(n_valid, m_tile)
    idx = np.arange(n_valid, dtype=np.int32)

    Lambdas = tile.Lambdas.at[idx].set(np.eye(3, dtype=np.float64))
    thetas = tile.thetas.at[idx].set(np.zeros((n_valid, 3), dtype=np.float64))
    etas = tile.etas.at[idx, 0].set(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    weights = tile.weights.at[idx].set(1.0)
    timestamps = tile.timestamps.at[idx].set(0.0)
    created_timestamps = tile.created_timestamps.at[idx].set(0.0)
    primitive_ids = tile.primitive_ids.at[idx].set(np.arange(n_valid, dtype=np.int64))
    valid_mask = tile.valid_mask.at[idx].set(True)
    colors = tile.colors.at[idx].set(np.zeros((n_valid, 3), dtype=np.float64))

    tile = tile.__class__(
        tile_id=tile.tile_id,
        Lambdas=Lambdas,
        thetas=thetas,
        etas=etas,
        weights=weights,
        timestamps=timestamps,
        created_timestamps=created_timestamps,
        primitive_ids=primitive_ids,
        valid_mask=valid_mask,
        colors=colors,
        next_local_id=tile.next_local_id,
        count=n_valid,
    )
    return extract_primitive_map_view(tile=tile)


def test_budget_assertions_association():
    k_assoc = int(constants.GC_K_ASSOC)
    n_feat = max(1, k_assoc)
    n_surfel = 0
    n_valid = k_assoc
    m_tile = max(k_assoc, 1)

    batch = _build_measurement_batch(n_feat=n_feat, n_surfel=n_surfel, n_valid=n_valid)
    map_view = _build_tile_view(m_tile=m_tile, n_valid=k_assoc)

    config = AssociationConfig(k_assoc=k_assoc, k_sinkhorn=constants.GC_K_SINKHORN)
    _, cert, _ = associate_primitives_ot(
        measurement_batch=batch,
        map_view=map_view,
        config=config,
    )

    # Budget assertions (Phase 3.2)
    assert cert.compute.largest_tensor_shape[0] <= batch.n_total
    assert cert.compute.largest_tensor_shape[1] <= k_assoc
    assert cert.compute.segment_sum_k == k_assoc

    bytes_per_f64 = 8
    alloc_budget = int(batch.n_total * k_assoc * bytes_per_f64 * 4)
    assert cert.compute.alloc_bytes_est <= alloc_budget

    # PSD projection count: at most one per primitive per scan
    assert cert.compute.psd_projection_count <= map_view.m_tile
