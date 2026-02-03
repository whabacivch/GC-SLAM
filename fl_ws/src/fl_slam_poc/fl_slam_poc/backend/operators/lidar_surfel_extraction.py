"""
LiDAR Surfel Extraction for Geometric Compositional SLAM v2.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md Section 4

Extracts surfels from deskewed LiDAR points and populates MeasurementBatch.

Implementation (single-path, JAX-only):
- Fixed-size MA-hex 3D bucketing (hash grid via modulo wrapping; bounded compute).
- Batched weighted plane fit per cell (jax.vmap + eigh on 3×3 covariance).
- Gaussian covariance from in-plane spread + perpendicular residual; Wishart-regularized precision.

Operator: extract_lidar_surfels(points, timestamps, weights, ...) -> (MeasurementBatch, CertBundle, ExpectedEffect)

Key constraints:
- Fixed N_SURFEL budget (pad/truncate)
- Output in info form (Lambda, theta) + vMF natural params (eta = kappa * mu)
- No heuristic gating: validity is a mask used to pad/truncate to budget; no accept/reject branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common.runtime_counters import record_device_to_host
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import CertBundle, ExpectedEffect, InfluenceCert, SupportCert
from fl_slam_poc.common.ma_hex_web import MAHex3DConfig, bin_points_3d
from fl_slam_poc.backend.structures.measurement_batch import (
    MeasurementBatch,
    measurement_batch_from_lidar_only,
    measurement_batch_add_lidar_surfels,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SurfelExtractionConfig:
    """Configuration for LiDAR surfel extraction."""
    n_surfel: int = constants.GC_N_SURFEL
    n_feat: int = constants.GC_N_FEAT  # For MeasurementBatch sizing
    voxel_size_m: float = 0.1
    # Fixed MA-hex 3D hash grid (bounded compute; collisions are explicit approximation)
    hex3d_num_cells_1: int = 32
    hex3d_num_cells_2: int = 32
    hex3d_num_cells_z: int = 8
    hex3d_max_occupants: int = 32
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

def _normalize(v: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return v / (jnp.linalg.norm(v) + eps)

def _orthonormal_basis_from_normal(n: jnp.ndarray, eps: float = 1e-12) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (e1, e2) orthonormal to unit normal n; branch-free selection of a stable e1."""
    n = _normalize(n, eps)
    e1_a = jnp.array([-n[1], n[0], 0.0], dtype=jnp.float64)
    e1_b = jnp.array([-n[2], 0.0, n[0]], dtype=jnp.float64)
    use_a = jnp.abs(n[2]) < 0.9
    e1 = jnp.where(use_a, e1_a, e1_b)
    e1 = _normalize(e1, eps)
    e2 = _normalize(jnp.cross(n, e1), eps)
    return e1, e2


def _fit_one_cell(
    points_centered: jnp.ndarray,    # (N, 3)
    timestamps: jnp.ndarray,         # (N,)
    weights: jnp.ndarray,            # (N,)
    idx_vec: jnp.ndarray,            # (max_occ,)
    count_used: jnp.ndarray,         # ()
    *,
    min_points: int,
    sensor_var: float,
    wishart_nu: float,
    wishart_psi_scale: float,
    kappa_scale: float,
    kappa_min: float,
    kappa_max: float,
    eig_min: float,
    eps: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fit a weighted plane to one cell (gathered/padded indices), and return surfel params.

    Returns:
        centroid_c: (3,) in centered coordinates
        Sigma_reg: (3,3)
        normal: (3,)
        kappa: ()
        w_surfel: ()
        t_surfel: ()
        valid: () bool
    """
    idx_vec = jnp.asarray(idx_vec, dtype=jnp.int32).reshape(-1)
    idx_safe = jnp.maximum(idx_vec, 0)
    present = (idx_vec >= 0).astype(jnp.float64)

    pts = points_centered[idx_safe, :]  # (max_occ, 3)
    w = weights[idx_safe] * present
    t = timestamps[idx_safe] * present

    w_sum = jnp.sum(w) + eps
    centroid = jnp.sum(pts * w[:, None], axis=0) / w_sum

    centered = pts - centroid[None, :]
    cov = (centered * w[:, None]).T @ centered / w_sum
    cov = 0.5 * (cov + cov.T) + eig_min * jnp.eye(3, dtype=jnp.float64)

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    normal = normal * jnp.where(normal[2] < 0.0, -1.0, 1.0)  # deterministic sign
    normal = _normalize(normal, eps)

    e1, e2 = _orthonormal_basis_from_normal(normal, eps)
    proj1 = centered @ e1
    proj2 = centered @ e2

    var_e1 = jnp.sum(w * (proj1 * proj1)) / w_sum + sensor_var
    var_e2 = jnp.sum(w * (proj2 * proj2)) / w_sum + sensor_var
    sigma_perp_sq = jnp.maximum(eigvals[0], eig_min)
    var_perp = sigma_perp_sq + sensor_var

    V = jnp.stack([e1, e2, normal], axis=1)  # (3,3)
    D = jnp.diag(jnp.array([jnp.maximum(var_e1, eig_min), jnp.maximum(var_e2, eig_min), jnp.maximum(var_perp, eig_min)], dtype=jnp.float64))
    Sigma = V @ D @ V.T
    Sigma = 0.5 * (Sigma + Sigma.T) + eig_min * jnp.eye(3, dtype=jnp.float64)

    # Wishart regularization in precision space
    Lambda = jnp.linalg.inv(Sigma + eig_min * jnp.eye(3, dtype=jnp.float64))
    Lambda = 0.5 * (Lambda + Lambda.T)
    psi = jnp.maximum(jnp.asarray(wishart_psi_scale, dtype=jnp.float64), eps)
    Lambda_reg = Lambda + (jnp.asarray(wishart_nu, dtype=jnp.float64) / psi) * jnp.eye(3, dtype=jnp.float64)
    Lambda_reg = 0.5 * (Lambda_reg + Lambda_reg.T) + eig_min * jnp.eye(3, dtype=jnp.float64)
    Sigma_reg = jnp.linalg.inv(Lambda_reg)
    Sigma_reg = 0.5 * (Sigma_reg + Sigma_reg.T) + eig_min * jnp.eye(3, dtype=jnp.float64)

    kappa = jnp.asarray(kappa_scale, dtype=jnp.float64) / jnp.sqrt(jnp.maximum(sigma_perp_sq, eig_min))
    kappa = jnp.clip(kappa, jnp.asarray(kappa_min, dtype=jnp.float64), jnp.asarray(kappa_max, dtype=jnp.float64))

    w_surfel = jnp.sum(w)
    t_surfel = jnp.sum(t) / w_sum

    valid = (count_used >= min_points) & (w_surfel > 0.0)
    return centroid, Sigma_reg, normal, kappa, w_surfel, t_surfel, valid


def _extract_surfels_mahex3d_jax(
    points: jnp.ndarray,
    timestamps: jnp.ndarray,
    weights: jnp.ndarray,
    config: SurfelExtractionConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    JAX-only surfel extraction with fixed-size bucketing.

    Returns fixed-size arrays of length config.n_surfel, plus n_valid (python int) for how many slots are valid.
    """

    (
        positions_sel,
        covs_sel,
        normals_sel,
        kappas_sel,
        weights_sel,
        timestamps_sel,
        n_valid_i32,
    ) = _extract_surfels_mahex3d_jax_jit(
        points=points,
        timestamps=timestamps,
        weights=weights,
        n_surfel=int(config.n_surfel),
        voxel_size_m=float(config.voxel_size_m),
        min_points_per_voxel=int(config.min_points_per_voxel),
        hex3d_num_cells_1=int(config.hex3d_num_cells_1),
        hex3d_num_cells_2=int(config.hex3d_num_cells_2),
        hex3d_num_cells_z=int(config.hex3d_num_cells_z),
        hex3d_max_occupants=int(config.hex3d_max_occupants),
        sensor_noise_var_per_axis=float(config.sensor_noise_var_per_axis),
        wishart_nu=float(config.wishart_nu),
        wishart_psi_scale=float(config.wishart_psi_scale),
        kappa_main_scale=float(config.kappa_main_scale),
        kappa_min=float(config.kappa_min),
        kappa_max=float(config.kappa_max),
        eig_min=float(config.eig_min),
    )

    n_valid = int(jax.device_get(n_valid_i32))
    record_device_to_host(n_valid_i32, syncs=1)
    return (
        positions_sel,
        covs_sel,
        normals_sel,
        kappas_sel,
        weights_sel,
        timestamps_sel,
        n_valid,
    )


@jax.jit(static_argnames=(
    "n_surfel",
    "min_points_per_voxel",
    "hex3d_num_cells_1",
    "hex3d_num_cells_2",
    "hex3d_num_cells_z",
    "hex3d_max_occupants",
))
def _extract_surfels_mahex3d_jax_jit(
    *,
    points: jnp.ndarray,
    timestamps: jnp.ndarray,
    weights: jnp.ndarray,
    n_surfel: int,
    voxel_size_m: float,
    min_points_per_voxel: int,
    hex3d_num_cells_1: int,
    hex3d_num_cells_2: int,
    hex3d_num_cells_z: int,
    hex3d_max_occupants: int,
    sensor_noise_var_per_axis: float,
    wishart_nu: float,
    wishart_psi_scale: float,
    kappa_main_scale: float,
    kappa_min: float,
    kappa_max: float,
    eig_min: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled core for MA-hex 3D surfel extraction.

    Contract:
    - Inputs should be fixed-shape arrays (e.g. (N_POINTS_CAP,3)).
    - All outputs are fixed-size (n_surfel, ...).
    - n_valid returned as int32 scalar.
    """
    points = jnp.asarray(points, dtype=jnp.float64).reshape(-1, 3)
    timestamps = jnp.asarray(timestamps, dtype=jnp.float64).reshape(-1)
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)

    # Mask out non-finite sentinels (parse_pointcloud2_vlp16 replaces NaN/Inf with ±GC_NONFINITE_SENTINEL).
    sentinel = jnp.asarray(constants.GC_NONFINITE_SENTINEL, dtype=jnp.float64)
    point_mask = jnp.all(jnp.abs(points) < (0.1 * sentinel), axis=1)
    weights_eff = weights * point_mask.astype(jnp.float64)

    # Center points for per-scan hashing stability (translation does not affect covariances).
    w_sum = jnp.sum(weights_eff) + jnp.asarray(eig_min, dtype=jnp.float64)
    center = jnp.sum(points * weights_eff[:, None], axis=0) / w_sum
    points_c = points - center[None, :]

    hex_cfg = MAHex3DConfig(
        num_cells_1=hex3d_num_cells_1,
        num_cells_2=hex3d_num_cells_2,
        num_cells_z=hex3d_num_cells_z,
        max_occupants=hex3d_max_occupants,
        voxel_size=voxel_size_m,
    )
    bucket = bin_points_3d(points_c, point_mask, hex_cfg)

    fit_fn = lambda idx, cnt: _fit_one_cell(
        points_centered=points_c,
        timestamps=timestamps,
        weights=weights_eff,
        idx_vec=idx,
        count_used=cnt,
        min_points=min_points_per_voxel,
        sensor_var=sensor_noise_var_per_axis,
        wishart_nu=wishart_nu,
        wishart_psi_scale=wishart_psi_scale,
        kappa_scale=kappa_main_scale,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        eig_min=eig_min,
    )

    centroids_c, covs, normals, kappas, surfel_w, surfel_t, valid = jax.vmap(fit_fn)(bucket.bucket, bucket.count)
    centroids = centroids_c + center[None, :]

    # Select up to n_surfel valid cells in a deterministic order (valid first, then increasing cell id).
    n_cells = int(hex_cfg.n_cells)
    cell_ids = jnp.arange(n_cells, dtype=jnp.int32)
    key = cell_ids + (jnp.int32(1) - valid.astype(jnp.int32)) * jnp.int32(n_cells)
    order = jnp.argsort(key)
    take = order[:n_surfel]

    slot_valid = valid[take]
    n_valid_i32 = jnp.sum(slot_valid.astype(jnp.int32))

    positions_sel = centroids[take, :]
    covs_sel = covs[take, :, :]
    normals_sel = normals[take, :]
    kappas_sel = kappas[take]
    weights_sel = surfel_w[take]
    timestamps_sel = surfel_t[take]

    # Zero out the invalid tail for safety (measurement_batch_add_lidar_surfels uses n_valid_override).
    slot_mask = (jnp.arange(n_surfel, dtype=jnp.int32) < n_valid_i32).astype(jnp.float64)
    positions_sel = positions_sel * slot_mask[:, None]
    covs_sel = covs_sel * slot_mask[:, None, None] + (1.0 - slot_mask)[:, None, None] * jnp.eye(3, dtype=jnp.float64)[None, :, :]
    normals_sel = normals_sel * slot_mask[:, None]
    weights_sel = weights_sel * slot_mask
    kappas_sel = kappas_sel * slot_mask
    timestamps_sel = timestamps_sel * slot_mask

    return (
        positions_sel,
        covs_sel,
        normals_sel,
        kappas_sel,
        weights_sel,
        timestamps_sel,
        n_valid_i32,
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

    (
        positions,
        covariances,
        normals,
        kappas,
        surfel_weights,
        surfel_timestamps,
        n_use,
    ) = _extract_surfels_mahex3d_jax(points, timestamps, weights, config)

    # Build MeasurementBatch: merge into base_batch (camera) or lidar-only
    if base_batch is not None:
        batch = measurement_batch_add_lidar_surfels(
            batch=base_batch,
            positions=positions,
            covariances=covariances,
            normals=normals,
            kappas=kappas,
            weights=surfel_weights,
            timestamps=surfel_timestamps,
            ring_indices=None,
            n_valid_override=n_use,
            eps_lift=config.eps_lift,
        )
    else:
        batch = measurement_batch_from_lidar_only(
            positions=positions,
            covariances=covariances,
            normals=normals,
            kappas=kappas,
            weights=surfel_weights,
            timestamps=surfel_timestamps,
            ring_indices=None,
            n_feat=config.n_feat,
            n_surfel=config.n_surfel,
            n_valid_override=n_use,
            eps_lift=config.eps_lift,
        )

    # Certificate
    support_frac = float(n_use) / float(max(config.n_surfel, 1))

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ma_hex3d_binning", "plane_fit_batched", "wishart_regularization"],
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
