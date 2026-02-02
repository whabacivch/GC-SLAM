"""
LiDAR Surfel Extraction for Geometric Compositional SLAM v2.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md Section 4

Extracts surfels from deskewed LiDAR points and populates MeasurementBatch.
Uses voxel downsampling + plane fitting per voxel.

Operator: extract_lidar_surfels(points, timestamps, weights, ...) -> (MeasurementBatch, CertBundle, ExpectedEffect)

Key constraints:
- Fixed N_SURFEL budget (pad/truncate)
- Output in info form (Lambda, theta) + vMF natural params (eta = kappa * mu)
- No gates, no branching on data counts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import CertBundle, ExpectedEffect, InfluenceCert, SupportCert
from fl_slam_poc.backend.structures.measurement_batch import (
    MeasurementBatch,
    measurement_batch_from_lidar_only,
    measurement_batch_add_lidar_surfels,
)
from fl_slam_poc.frontend.sensors.lidar_surfels import (
    voxel_downsample,
    _cov_from_plane_residuals,
    wishart_regularize_3d,
    LidarSurfelConfig,
)
from fl_slam_poc.frontend.sensors.lidar_camera_depth_fusion import _fit_plane_weighted


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SurfelExtractionConfig:
    """Configuration for LiDAR surfel extraction."""
    n_surfel: int = constants.GC_N_SURFEL
    n_feat: int = constants.GC_N_FEAT  # For MeasurementBatch sizing
    voxel_size_m: float = 0.1
    min_points_per_voxel: int = 3
    sensor_noise_var_per_axis: float = 1e-6
    wishart_nu: float = 5.0
    wishart_psi_scale: float = 0.1
    kappa_main_scale: float = 10.0  # kappa = scale / sqrt(sigma_perp^2)
    kappa_min: float = 0.1
    kappa_max: float = 100.0
    eig_min: float = 1e-12
    eps_lift: float = constants.GC_EPS_LIFT


# =============================================================================
# Core Extraction
# =============================================================================


def _extract_surfels_core(
    points: np.ndarray,
    timestamps: np.ndarray,
    weights: np.ndarray,
    config: SurfelExtractionConfig,
) -> Tuple[
    np.ndarray,  # positions (N, 3)
    np.ndarray,  # covariances (N, 3, 3)
    np.ndarray,  # normals (N, 3)
    np.ndarray,  # kappas (N,)
    np.ndarray,  # weights (N,)
    np.ndarray,  # timestamps (N,)
    int,         # n_extracted
]:
    """
    Core surfel extraction: voxel downsample + plane fit per voxel.

    Returns arrays of extracted surfels (may be fewer than N_SURFEL).
    """
    points = np.asarray(points, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Drop padded points (weight==0) before voxelization to avoid zero-mass planes.
    if weights.size > 0:
        mask = weights > 0.0
        points = points[mask]
        timestamps = timestamps[mask]
        weights = weights[mask]

    if points.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            0,
        )

    # Voxel downsample
    voxel_groups = voxel_downsample(points, config.voxel_size_m)

    positions_list = []
    covariances_list = []
    normals_list = []
    kappas_list = []
    weights_list = []
    timestamps_list = []

    for indices in voxel_groups:
        if len(indices) < config.min_points_per_voxel:
            continue

        pts = points[indices]
        wts = weights[indices]
        ts = timestamps[indices]

        # Plane fit
        centroid, normal, eigvals, sigma_perp_sq = _fit_plane_weighted(pts, wts)

        # Covariance from plane residuals
        Sigma = _cov_from_plane_residuals(
            centroid, normal, pts, sigma_perp_sq,
            config.sensor_noise_var_per_axis, config.eig_min,
        )

        # Wishart regularization
        Lambda = np.linalg.inv(Sigma + config.eig_min * np.eye(3))
        Lambda = 0.5 * (Lambda + Lambda.T)
        Lambda_reg = wishart_regularize_3d(
            Lambda, config.wishart_nu, config.wishart_psi_scale, config.eig_min,
        )
        Sigma_reg = np.linalg.inv(Lambda_reg)
        Sigma_reg = 0.5 * (Sigma_reg + Sigma_reg.T) + config.eig_min * np.eye(3)

        # vMF kappa from planarity (main lobe only for single-lobe vMF)
        sigma_perp_sq = max(sigma_perp_sq, config.eig_min)
        kappa = config.kappa_main_scale / np.sqrt(sigma_perp_sq)
        kappa = np.clip(kappa, config.kappa_min, config.kappa_max)

        # Weight = sum of point weights in voxel
        w_surfel = float(np.sum(wts))

        # Timestamp = weighted mean
        w_sum = np.sum(wts) + 1e-12
        t_surfel = float(np.sum(wts * ts) / w_sum)

        positions_list.append(centroid)
        covariances_list.append(Sigma_reg)
        normals_list.append(normal)
        kappas_list.append(kappa)
        weights_list.append(w_surfel)
        timestamps_list.append(t_surfel)

    n_extracted = len(positions_list)

    if n_extracted == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            0,
        )

    return (
        np.array(positions_list, dtype=np.float64),
        np.array(covariances_list, dtype=np.float64),
        np.array(normals_list, dtype=np.float64),
        np.array(kappas_list, dtype=np.float64),
        np.array(weights_list, dtype=np.float64),
        np.array(timestamps_list, dtype=np.float64),
        n_extracted,
    )


# =============================================================================
# Main Operator
# =============================================================================


def extract_lidar_surfels(
    points: jnp.ndarray,
    timestamps: jnp.ndarray,
    weights: jnp.ndarray,
    config: Optional[SurfelExtractionConfig] = None,
    base_batch: Optional[MeasurementBatch] = None,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "surfel_extraction",
) -> Tuple[MeasurementBatch, CertBundle, ExpectedEffect]:
    """
    Extract surfels from deskewed LiDAR points and build MeasurementBatch.

    If base_batch is provided (e.g. camera splats), adds LiDAR surfels to it.
    Otherwise returns lidar-only batch.

    Fixed-cost operator: always outputs fixed-size arrays (N_SURFEL).
    Pads unused slots with zeros.

    Args:
        points: Deskewed LiDAR points (N, 3)
        timestamps: Per-point timestamps (N,)
        weights: Per-point weights (N,)
        config: Extraction configuration
        base_batch: Optional existing batch (e.g. camera) to add surfels to
        chart_id: Chart identifier
        anchor_id: Anchor identifier

    Returns:
        (MeasurementBatch, CertBundle, ExpectedEffect)
    """
    if config is None:
        config = SurfelExtractionConfig()

    # Convert to numpy for extraction
    points_np = np.asarray(points, dtype=np.float64)
    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)

    # Extract surfels
    (
        positions,
        covariances,
        normals,
        kappas,
        surfel_weights,
        surfel_timestamps,
        n_extracted,
    ) = _extract_surfels_core(points_np, timestamps_np, weights_np, config)

    # Truncate to budget if needed
    n_use = min(n_extracted, config.n_surfel)

    if n_use > 0:
        positions = positions[:n_use]
        covariances = covariances[:n_use]
        normals = normals[:n_use]
        kappas = kappas[:n_use]
        surfel_weights = surfel_weights[:n_use]
        surfel_timestamps = surfel_timestamps[:n_use]
    else:
        # Empty case: create minimal arrays
        positions = np.zeros((0, 3), dtype=np.float64)
        covariances = np.zeros((0, 3, 3), dtype=np.float64)
        normals = np.zeros((0, 3), dtype=np.float64)
        kappas = np.zeros((0,), dtype=np.float64)
        surfel_weights = np.zeros((0,), dtype=np.float64)
        surfel_timestamps = np.zeros((0,), dtype=np.float64)

    # Build MeasurementBatch: merge into base_batch (camera) or lidar-only
    if base_batch is not None:
        batch = measurement_batch_add_lidar_surfels(
            batch=base_batch,
            positions=jnp.array(positions, dtype=jnp.float64),
            covariances=jnp.array(covariances, dtype=jnp.float64),
            normals=jnp.array(normals, dtype=jnp.float64),
            kappas=jnp.array(kappas, dtype=jnp.float64),
            weights=jnp.array(surfel_weights, dtype=jnp.float64),
            timestamps=jnp.array(surfel_timestamps, dtype=jnp.float64),
            ring_indices=None,
            eps_lift=config.eps_lift,
        )
    else:
        batch = measurement_batch_from_lidar_only(
            positions=jnp.array(positions, dtype=jnp.float64),
            covariances=jnp.array(covariances, dtype=jnp.float64),
            normals=jnp.array(normals, dtype=jnp.float64),
            kappas=jnp.array(kappas, dtype=jnp.float64),
            weights=jnp.array(surfel_weights, dtype=jnp.float64),
            timestamps=jnp.array(surfel_timestamps, dtype=jnp.float64),
            ring_indices=None,
            n_feat=config.n_feat,
            n_surfel=config.n_surfel,
            eps_lift=config.eps_lift,
        )

    # Certificate
    support_frac = float(n_use) / float(max(config.n_surfel, 1))
    n_input = points_np.shape[0]

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["voxel_downsample", "plane_fit", "wishart_regularization"],
        support=SupportCert(
            ess_total=float(n_use),
            support_frac=support_frac,
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    effect = ExpectedEffect(
        objective_name="surfel_extraction",
        predicted=float(n_use),
        realized=float(n_use),
    )

    return batch, cert, effect


# =============================================================================
# Utility: Convert Surfels to PrimitiveMapView for Association
# =============================================================================


def measurement_batch_to_arrays(
    batch: MeasurementBatch,
    eps_lift: float = constants.GC_EPS_LIFT,
    eps_mass: float = constants.GC_EPS_MASS,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Extract arrays from MeasurementBatch for PrimitiveMapView.

    Returns:
        positions: (N_valid, 3) mean positions
        covariances: (N_valid, 3, 3) covariance matrices
        directions: (N_valid, 3) mean directions
        kappas: (N_valid,) vMF concentrations
        weights: (N_valid,) reliability weights
    """
    from fl_slam_poc.backend.structures.measurement_batch import (
        measurement_batch_mean_positions,
        measurement_batch_mean_directions,
        measurement_batch_kappas,
    )

    # Get all positions/directions/kappas
    positions_all = measurement_batch_mean_positions(batch, eps_lift=eps_lift)
    directions_all = measurement_batch_mean_directions(batch, eps_mass=eps_mass)
    kappas_all = measurement_batch_kappas(batch)

    # Filter to valid entries
    valid_mask = batch.valid_mask
    valid_indices = jnp.where(valid_mask)[0]

    n_valid = batch.n_valid
    if n_valid == 0:
        return (
            jnp.zeros((0, 3), dtype=jnp.float64),
            jnp.zeros((0, 3, 3), dtype=jnp.float64),
            jnp.zeros((0, 3), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
        )

    # Compute covariances from info form
    Lambda_reg = batch.Lambdas + eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]
    covariances_all = jax.vmap(jnp.linalg.inv)(Lambda_reg)

    positions = positions_all[valid_indices]
    covariances = covariances_all[valid_indices]
    directions = directions_all[valid_indices]
    kappas = kappas_all[valid_indices]
    weights = batch.weights[valid_indices]

    return positions, covariances, directions, kappas, weights
