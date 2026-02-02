"""
MeasurementBatch: Unified measurement primitive structure for Geometric Compositional SLAM v2.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md Section 4

All sensor outputs become measurement primitives packed to fixed sizes:
- Camera splats: N_FEAT (fixed)
- LiDAR surfels: N_SURFEL (fixed)
- Optional: IMU/odom-derived motion features (reliability scaling)

Each primitive has:
- 3D Gaussian info form (Lambda, theta)
- vMF params (multi-lobe; mandatory)
- Reliability weight
- Source metadata (camera_idx, lidar_ring, etc.)

Natural-parameter discipline:
- 3D Gaussian: store (Lambda, theta) as primary; invert only at boundaries
- vMF: store etas[b] = kappa_b * mu_b for B fixed lobes; resultant eta = Σ_b etas[b]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants


# =============================================================================
# Single Measurement Primitive
# =============================================================================


@dataclass
class MeasurementPrimitive:
    """
    Single measurement primitive in info form.

    Represents a feature extracted from camera or LiDAR.

    Geometry (3D Gaussian in info form):
        Lambda: (3, 3) precision matrix
        theta: (3,) information vector (= Lambda @ position_mean)

    Orientation/appearance (multi-lobe vMF, B=GC_VMF_N_LOBES):
        etas: (B, 3) natural parameters (= kappa_b * mu_b). Resultant eta is sum over lobes.

    Metadata:
        weight: Reliability/confidence weight
        source: 0=camera, 1=lidar
        source_idx: Index within source (camera frame idx, lidar ring, etc.)
    """
    Lambda: jnp.ndarray  # (3, 3) precision
    theta: jnp.ndarray   # (3,) info vector
    etas: jnp.ndarray    # (B, 3) vMF natural params (B lobes)
    weight: float        # Reliability weight
    source: int          # 0=camera, 1=lidar
    source_idx: int      # Index within source


# =============================================================================
# MeasurementBatch: Fixed-Size Packed Batch
# =============================================================================


@dataclass
class MeasurementBatch:
    """
    Fixed-size batch of measurement primitives.

    All arrays have shape (N_total,) where N_total = N_FEAT + N_SURFEL.
    Camera splats occupy indices [0, N_FEAT).
    LiDAR surfels occupy indices [N_FEAT, N_total).

    Valid entries marked by valid_mask.

    Attributes:
        Lambdas: (N_total, 3, 3) precision matrices
        thetas: (N_total, 3) information vectors
        etas: (N_total, B, 3) vMF natural parameters (B lobes; mandatory)
        weights: (N_total,) reliability weights
        sources: (N_total,) source identifiers (0=camera, 1=lidar)
        source_indices: (N_total,) indices within source
        valid_mask: (N_total,) bool mask for valid entries
        timestamps: (N_total,) measurement timestamps
        colors: (N_total, 3) RGB colors; camera slice from camera splats, LiDAR slice from colors_lidar or default from normals (never black)

        # Fixed budgets (compile-time constants)
        n_feat: int = N_FEAT budget
        n_surfel: int = N_SURFEL budget
        n_camera_valid: int = actual camera features
        n_lidar_valid: int = actual LiDAR surfels
    """
    # Geometry (Gaussian info form)
    Lambdas: jnp.ndarray      # (N_total, 3, 3)
    thetas: jnp.ndarray       # (N_total, 3)

    # Orientation/appearance (multi-lobe vMF)
    # etas: (N_total, B, 3) where B=GC_VMF_N_LOBES. Resultant eta = sum_b etas[:,b,:].
    etas: jnp.ndarray         # (N_total, B, 3)

    # Metadata
    weights: jnp.ndarray      # (N_total,)
    sources: jnp.ndarray      # (N_total,) int: 0=camera, 1=lidar
    source_indices: jnp.ndarray  # (N_total,)
    valid_mask: jnp.ndarray   # (N_total,) bool
    timestamps: jnp.ndarray   # (N_total,)
    colors: jnp.ndarray       # (N_total, 3)

    # Budgets
    n_feat: int
    n_surfel: int
    n_camera_valid: int
    n_lidar_valid: int

    @property
    def n_total(self) -> int:
        return self.n_feat + self.n_surfel

    @property
    def n_valid(self) -> int:
        return self.n_camera_valid + self.n_lidar_valid

    @property
    def camera_slice(self) -> slice:
        """Slice for camera features [0, n_feat)."""
        return slice(0, self.n_feat)

    @property
    def lidar_slice(self) -> slice:
        """Slice for LiDAR surfels [n_feat, n_total)."""
        return slice(self.n_feat, self.n_total)


def create_empty_measurement_batch(
    n_feat: int = constants.GC_N_FEAT,
    n_surfel: int = constants.GC_N_SURFEL,
) -> MeasurementBatch:
    """Create empty measurement batch with fixed-size arrays."""
    n_total = n_feat + n_surfel
    return MeasurementBatch(
        Lambdas=jnp.zeros((n_total, 3, 3), dtype=jnp.float64),
        thetas=jnp.zeros((n_total, 3), dtype=jnp.float64),
        etas=jnp.zeros((n_total, constants.GC_VMF_N_LOBES, 3), dtype=jnp.float64),
        weights=jnp.zeros((n_total,), dtype=jnp.float64),
        sources=jnp.zeros((n_total,), dtype=jnp.int32),
        source_indices=jnp.zeros((n_total,), dtype=jnp.int32),
        valid_mask=jnp.zeros((n_total,), dtype=bool),
        timestamps=jnp.zeros((n_total,), dtype=jnp.float64),
        colors=jnp.zeros((n_total, 3), dtype=jnp.float64),
        n_feat=n_feat,
        n_surfel=n_surfel,
        n_camera_valid=0,
        n_lidar_valid=0,
    )


# =============================================================================
# Builder Functions
# =============================================================================


def measurement_batch_from_camera_splats(
    positions: jnp.ndarray,           # (N, 3) mean positions
    covariances: jnp.ndarray,         # (N, 3, 3) covariances
    directions: jnp.ndarray,          # (N, 3) mean directions
    kappas: jnp.ndarray,              # (N,) vMF concentrations
    weights: jnp.ndarray,             # (N,) reliability weights
    timestamps: jnp.ndarray,          # (N,) timestamps
    colors: Optional[jnp.ndarray] = None,  # (N, 3) RGB
    n_feat: int = constants.GC_N_FEAT,
    n_surfel: int = constants.GC_N_SURFEL,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> MeasurementBatch:
    """
    Build MeasurementBatch from camera splat outputs.

    Converts covariance form to info form.
    Pads/truncates to fixed N_FEAT budget.

    Args:
        positions: Mean positions (N, 3)
        covariances: Covariance matrices (N, 3, 3)
        directions: Mean directions (N, 3)
        kappas: vMF concentrations (N,)
        weights: Reliability weights (N,)
        timestamps: Measurement timestamps (N,)
        colors: Optional RGB colors (N, 3)
        n_feat: Camera feature budget
        n_surfel: LiDAR surfel budget (for sizing)
        eps_lift: Regularization for matrix inversion

    Returns:
        MeasurementBatch with camera features populated
    """
    N = positions.shape[0]
    n_valid = min(N, n_feat)
    n_total = n_feat + n_surfel

    # Convert covariance to info form: Lambda = Sigma^{-1}, theta = Lambda @ mu
    Sigma_reg = covariances[:n_valid] + eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]
    Lambdas_cam = jax.vmap(jnp.linalg.inv)(Sigma_reg)
    thetas_cam = jax.vmap(lambda L, mu: L @ mu)(Lambdas_cam, positions[:n_valid])

    # vMF natural params (B lobes): default is a single resultant lobe in slot 0.
    # Contract: vMF is not optional; all primitives carry B lobes even if only one is populated.
    B = int(constants.GC_VMF_N_LOBES)
    etas_cam = jnp.zeros((n_valid, B, 3), dtype=jnp.float64)
    etas_cam = etas_cam.at[:, 0, :].set(kappas[:n_valid, None] * directions[:n_valid])

    # Build full batch arrays
    Lambdas = jnp.zeros((n_total, 3, 3), dtype=jnp.float64)
    Lambdas = Lambdas.at[:n_valid].set(Lambdas_cam)

    thetas = jnp.zeros((n_total, 3), dtype=jnp.float64)
    thetas = thetas.at[:n_valid].set(thetas_cam)

    etas = jnp.zeros((n_total, B, 3), dtype=jnp.float64)
    etas = etas.at[:n_valid].set(etas_cam)

    weights_full = jnp.zeros((n_total,), dtype=jnp.float64)
    weights_full = weights_full.at[:n_valid].set(weights[:n_valid])

    sources = jnp.zeros((n_total,), dtype=jnp.int32)  # 0 = camera

    source_indices = jnp.zeros((n_total,), dtype=jnp.int32)
    source_indices = source_indices.at[:n_valid].set(jnp.arange(n_valid, dtype=jnp.int32))

    valid_mask = jnp.zeros((n_total,), dtype=bool)
    valid_mask = valid_mask.at[:n_valid].set(True)

    timestamps_full = jnp.zeros((n_total,), dtype=jnp.float64)
    timestamps_full = timestamps_full.at[:n_valid].set(timestamps[:n_valid])

    # Colors: use provided camera colors, or default neutral gray (never black)
    colors_full = jnp.zeros((n_total, 3), dtype=jnp.float64)
    if colors is not None:
        colors_full = colors_full.at[:n_valid].set(jnp.clip(colors[:n_valid], 0.0, 1.0))
    else:
        default_gray = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float64)
        colors_full = colors_full.at[:n_valid].set(default_gray)

    return MeasurementBatch(
        Lambdas=Lambdas,
        thetas=thetas,
        etas=etas,
        weights=weights_full,
        sources=sources,
        source_indices=source_indices,
        valid_mask=valid_mask,
        timestamps=timestamps_full,
        colors=colors_full,
        n_feat=n_feat,
        n_surfel=n_surfel,
        n_camera_valid=n_valid,
        n_lidar_valid=0,
    )


def _lidar_default_colors_from_normals(normals: jnp.ndarray) -> jnp.ndarray:
    """
    Default RGB for LiDAR surfels when no intensity: grayscale from normal.z.
    normal.z in [-1, 1] -> gray in [0.25, 0.75]; RGB = (g, g, g). Visible, not black.
    """
    nz = jnp.clip(normals[:, 2:3], -1.0, 1.0)
    g = 0.25 + 0.5 * (nz + 1.0) / 2.0  # (M, 1)
    return jnp.broadcast_to(g, (normals.shape[0], 3))


def measurement_batch_add_lidar_surfels(
    batch: MeasurementBatch,
    positions: jnp.ndarray,           # (M, 3) surfel centers
    covariances: jnp.ndarray,         # (M, 3, 3) surfel covariances
    normals: jnp.ndarray,             # (M, 3) surface normals
    kappas: jnp.ndarray,              # (M,) normal concentrations
    weights: jnp.ndarray,             # (M,) reliability weights
    timestamps: jnp.ndarray,          # (M,) timestamps
    ring_indices: Optional[jnp.ndarray] = None,  # (M,) LiDAR ring
    colors_lidar: Optional[jnp.ndarray] = None,  # (M, 3) optional RGB; when None use default from normals
    n_valid_override: Optional[int] = None,  # When set, treat only the first n_valid_override entries as valid.
    eps_lift: float = constants.GC_EPS_LIFT,
) -> MeasurementBatch:
    """
    Add LiDAR surfels to existing measurement batch.

    Converts covariance form to info form.
    Pads/truncates to fixed N_SURFEL budget.
    LiDAR slice colors: from colors_lidar if provided, else grayscale from normal.z (never black).
    """
    M = int(positions.shape[0])
    if n_valid_override is None:
        n_valid = min(M, batch.n_surfel)
    else:
        n_valid = min(int(n_valid_override), M, batch.n_surfel)

    # Convert covariance to info form
    Sigma_reg = covariances[:n_valid] + eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]
    Lambdas_lidar = jax.vmap(jnp.linalg.inv)(Sigma_reg)
    thetas_lidar = jax.vmap(lambda L, mu: L @ mu)(Lambdas_lidar, positions[:n_valid])

    # vMF natural params (B lobes) for normals: default is a single resultant lobe in slot 0.
    B = int(constants.GC_VMF_N_LOBES)
    etas_lidar = jnp.zeros((n_valid, B, 3), dtype=jnp.float64)
    etas_lidar = etas_lidar.at[:, 0, :].set(kappas[:n_valid, None] * normals[:n_valid])

    # LiDAR slice colors: explicit or default from normals (no black default)
    if colors_lidar is not None and colors_lidar.shape[0] >= n_valid:
        rgb_lidar = jnp.clip(colors_lidar[:n_valid], 0.0, 1.0)
    else:
        rgb_lidar = _lidar_default_colors_from_normals(normals[:n_valid])

    # Update LiDAR slice [n_feat, n_feat + n_surfel)
    start = batch.n_feat
    end = start + n_valid

    Lambdas = batch.Lambdas.at[start:end].set(Lambdas_lidar)
    thetas = batch.thetas.at[start:end].set(thetas_lidar)
    etas = batch.etas.at[start:end].set(etas_lidar)
    weights_full = batch.weights.at[start:end].set(weights[:n_valid])
    sources = batch.sources.at[start:end].set(1)  # 1 = lidar

    if ring_indices is not None:
        source_indices = batch.source_indices.at[start:end].set(ring_indices[:n_valid].astype(jnp.int32))
    else:
        source_indices = batch.source_indices.at[start:end].set(jnp.arange(n_valid, dtype=jnp.int32))

    valid_mask = batch.valid_mask.at[start:end].set(True)
    timestamps_full = batch.timestamps.at[start:end].set(timestamps[:n_valid])
    colors = batch.colors.at[start:end].set(rgb_lidar)

    return MeasurementBatch(
        Lambdas=Lambdas,
        thetas=thetas,
        etas=etas,
        weights=weights_full,
        sources=sources,
        source_indices=source_indices,
        valid_mask=valid_mask,
        timestamps=timestamps_full,
        colors=colors,
        n_feat=batch.n_feat,
        n_surfel=batch.n_surfel,
        n_camera_valid=batch.n_camera_valid,
        n_lidar_valid=n_valid,
    )


def measurement_batch_from_lidar_only(
    positions: jnp.ndarray,           # (M, 3) surfel centers
    covariances: jnp.ndarray,         # (M, 3, 3) surfel covariances
    normals: jnp.ndarray,             # (M, 3) surface normals
    kappas: jnp.ndarray,              # (M,) normal concentrations
    weights: jnp.ndarray,             # (M,) reliability weights
    timestamps: jnp.ndarray,          # (M,) timestamps
    ring_indices: Optional[jnp.ndarray] = None,
    colors_lidar: Optional[jnp.ndarray] = None,  # (M, 3) optional; when None use default from normals
    n_valid_override: Optional[int] = None,
    n_feat: int = constants.GC_N_FEAT,
    n_surfel: int = constants.GC_N_SURFEL,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> MeasurementBatch:
    """
    Build MeasurementBatch from LiDAR surfels only (no camera).
    LiDAR colors: from colors_lidar if provided, else grayscale from normal.z (never black).
    """
    batch = create_empty_measurement_batch(n_feat=n_feat, n_surfel=n_surfel)
    return measurement_batch_add_lidar_surfels(
        batch=batch,
        positions=positions,
        covariances=covariances,
        normals=normals,
        kappas=kappas,
        weights=weights,
        timestamps=timestamps,
        ring_indices=ring_indices,
        colors_lidar=colors_lidar,
        n_valid_override=n_valid_override,
        eps_lift=eps_lift,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def measurement_batch_mean_positions(
    batch: MeasurementBatch,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> jnp.ndarray:
    """Extract mean positions from info form: mu = Lambda^{-1} @ theta."""
    Lambda_reg = batch.Lambdas + eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]
    return jax.vmap(jnp.linalg.solve)(Lambda_reg, batch.thetas)


def measurement_batch_mean_directions(
    batch: MeasurementBatch,
    eps_mass: float = constants.GC_EPS_MASS,
) -> jnp.ndarray:
    """Extract mean directions from multi-lobe vMF resultant: mu = eta_sum / ||eta_sum||."""
    eta_sum = jnp.sum(batch.etas, axis=1)  # (N_total, 3)
    norms = jnp.linalg.norm(eta_sum, axis=1, keepdims=True)
    return eta_sum / (norms + eps_mass)


def measurement_batch_kappas(batch: MeasurementBatch) -> jnp.ndarray:
    """Extract vMF concentrations from resultant: kappa = ||eta_sum||."""
    eta_sum = jnp.sum(batch.etas, axis=1)  # (N_total, 3)
    return jnp.linalg.norm(eta_sum, axis=1)


def measurement_batch_lobe_directions(
    batch: MeasurementBatch,
    lobe: int,
    eps_mass: float = constants.GC_EPS_MASS,
) -> jnp.ndarray:
    """Extract lobe mean directions (N_total,3): mu_b = eta_b / ||eta_b||."""
    eta_b = batch.etas[:, int(lobe), :]
    norms = jnp.linalg.norm(eta_b, axis=1, keepdims=True)
    return eta_b / (norms + eps_mass)


def measurement_batch_lobe_kappas(batch: MeasurementBatch, lobe: int) -> jnp.ndarray:
    """Extract lobe kappas (N_total,): kappa_b = ||eta_b||."""
    eta_b = batch.etas[:, int(lobe), :]
    return jnp.linalg.norm(eta_b, axis=1)
