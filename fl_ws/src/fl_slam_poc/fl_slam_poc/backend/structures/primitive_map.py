"""
PrimitiveMap: Probabilistic primitive atlas for Golden Child SLAM v2.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md

Each primitive j in the map has:
- Geometry: Gaussian in info form (Lambda_j, theta_j) in 3D; optional cached (mu_j, Sigma_j)
- Orientation/appearance: vMF natural parameter(s) eta_j (resultant or B=3)
- Optional payload: color/descriptor summary
- Stable ID + spatial index membership

Map maintenance operators:
- PrimitiveMapInsert: new primitives enter the map
- PrimitiveMapFuse: PoE + Wishart; fuse associated measurement primitives
- PrimitiveMapCull: compute budget operator; mass drop logged as approximation
- PrimitiveMapMergeReduce: mixture reduction with CertBundle + Frobenius
- PrimitiveMapForget: continuous forgetting factor (no if/else)

All operators are fixed-cost and applied every scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import CertBundle, ExpectedEffect, InfluenceCert
from fl_slam_poc.common.primitives import domain_projection_psd_core


# =============================================================================
# Primitive Data Structure
# =============================================================================


@dataclass
class Primitive:
    """
    Single probabilistic primitive in info form.

    Geometry (3D Gaussian):
        Lambda: (3, 3) precision matrix (info form)
        theta: (3,) information vector (= Lambda @ mu)

    Orientation (vMF):
        eta: (3,) natural parameter (= kappa * mu_dir)

    Metadata:
        primitive_id: Stable unique identifier for temporal tracking
        weight: Accumulated mass/ESS (for culling)
        timestamp: Last update time (for staleness)
    """
    # Geometry (Gaussian in info form)
    Lambda: jnp.ndarray  # (3, 3) precision matrix
    theta: jnp.ndarray   # (3,) information vector

    # Orientation (vMF natural parameter)
    eta: jnp.ndarray     # (3,) = kappa * mu_dir

    # Metadata
    primitive_id: int    # Stable unique ID
    weight: float        # Accumulated mass/ESS
    timestamp: float     # Last update time (seconds)

    # Optional: color payload (RGB, normalized)
    color: Optional[jnp.ndarray] = None  # (3,) RGB in [0, 1]

    def mean_position(self, eps_lift: float = constants.GC_EPS_LIFT) -> jnp.ndarray:
        """Compute mean position mu = Lambda^{-1} @ theta."""
        Lambda_reg = self.Lambda + eps_lift * jnp.eye(3, dtype=jnp.float64)
        return jnp.linalg.solve(Lambda_reg, self.theta)

    def covariance(self, eps_lift: float = constants.GC_EPS_LIFT) -> jnp.ndarray:
        """Compute covariance Sigma = Lambda^{-1}."""
        Lambda_reg = self.Lambda + eps_lift * jnp.eye(3, dtype=jnp.float64)
        return jnp.linalg.inv(Lambda_reg)

    def kappa(self) -> float:
        """vMF concentration kappa = ||eta||."""
        return float(jnp.linalg.norm(self.eta))

    def mean_direction(self, eps_mass: float = constants.GC_EPS_MASS) -> jnp.ndarray:
        """Mean direction mu_dir = eta / ||eta||."""
        norm = jnp.linalg.norm(self.eta)
        return self.eta / (norm + eps_mass)


# =============================================================================
# PrimitiveMap: Batched Atlas
# =============================================================================


@dataclass
class PrimitiveMap:
    """
    Batched primitive map for efficient GPU operations.

    All arrays have fixed size MAX_SIZE; valid entries marked by mask.
    This enables JIT compilation without dynamic shapes.

    Attributes:
        Lambdas: (MAX_SIZE, 3, 3) precision matrices
        thetas: (MAX_SIZE, 3) information vectors
        etas: (MAX_SIZE, 3) vMF natural parameters
        weights: (MAX_SIZE,) accumulated mass/ESS
        timestamps: (MAX_SIZE,) last update times
        primitive_ids: (MAX_SIZE,) stable unique IDs
        valid_mask: (MAX_SIZE,) bool mask for valid entries
        colors: (MAX_SIZE, 3) optional RGB colors
        next_id: Next available primitive ID
        count: Number of valid primitives
    """
    Lambdas: jnp.ndarray      # (MAX_SIZE, 3, 3)
    thetas: jnp.ndarray       # (MAX_SIZE, 3)
    etas: jnp.ndarray         # (MAX_SIZE, 3)
    weights: jnp.ndarray      # (MAX_SIZE,)
    timestamps: jnp.ndarray   # (MAX_SIZE,)
    primitive_ids: jnp.ndarray  # (MAX_SIZE,) int64
    valid_mask: jnp.ndarray   # (MAX_SIZE,) bool
    colors: jnp.ndarray       # (MAX_SIZE, 3)
    next_id: int              # Next available ID
    count: int                # Number of valid primitives


def create_empty_primitive_map(
    max_size: int = constants.GC_PRIMITIVE_MAP_MAX_SIZE,
) -> PrimitiveMap:
    """Create empty primitive map with fixed-size arrays."""
    return PrimitiveMap(
        Lambdas=jnp.zeros((max_size, 3, 3), dtype=jnp.float64),
        thetas=jnp.zeros((max_size, 3), dtype=jnp.float64),
        etas=jnp.zeros((max_size, 3), dtype=jnp.float64),
        weights=jnp.zeros((max_size,), dtype=jnp.float64),
        timestamps=jnp.zeros((max_size,), dtype=jnp.float64),
        primitive_ids=jnp.zeros((max_size,), dtype=jnp.int64),
        valid_mask=jnp.zeros((max_size,), dtype=bool),
        colors=jnp.zeros((max_size, 3), dtype=jnp.float64),
        next_id=0,
        count=0,
    )


# =============================================================================
# PrimitiveMapView: Read-only view for rendering/association
# =============================================================================


@dataclass
class PrimitiveMapView:
    """
    Read-only view of primitive map for rendering and association.

    Contains only the data needed for downstream operations,
    with optional downselection for compute budgeting.
    """
    # Positions (mean of Gaussian)
    positions: jnp.ndarray    # (N, 3) mean positions
    covariances: jnp.ndarray  # (N, 3, 3) covariances (Sigma = Lambda^{-1})

    # Directions (mean of vMF)
    directions: jnp.ndarray   # (N, 3) mean directions
    kappas: jnp.ndarray       # (N,) concentrations

    # Weights and metadata
    weights: jnp.ndarray      # (N,) accumulated mass
    primitive_ids: jnp.ndarray  # (N,) stable IDs

    # Optional: colors for rendering
    colors: Optional[jnp.ndarray] = None  # (N, 3)

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@jax.jit
def _extract_primitive_map_view_core(
    Lambdas: jnp.ndarray,
    thetas: jnp.ndarray,
    etas: jnp.ndarray,
    weights: jnp.ndarray,
    primitive_ids: jnp.ndarray,
    colors: jnp.ndarray,
    valid_mask: jnp.ndarray,
    eps_lift: jnp.ndarray,
    eps_mass: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT'd core for extracting PrimitiveMapView from batched arrays."""
    # Compute means and covariances from info form
    Lambda_reg = Lambdas + eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]
    # Vectorized solve: mu_i = Lambda_i^{-1} @ theta_i
    positions = jax.vmap(jnp.linalg.solve)(Lambda_reg, thetas)
    covariances = jax.vmap(jnp.linalg.inv)(Lambda_reg)

    # Compute directions and kappas from vMF
    kappas = jnp.linalg.norm(etas, axis=1)
    directions = etas / (kappas[:, None] + eps_mass)

    return positions, covariances, directions, kappas, weights, primitive_ids, colors


def extract_primitive_map_view(
    prim_map: PrimitiveMap,
    max_primitives: Optional[int] = None,
    eps_lift: float = constants.GC_EPS_LIFT,
    eps_mass: float = constants.GC_EPS_MASS,
) -> PrimitiveMapView:
    """
    Extract read-only view from PrimitiveMap.

    Optionally downselects to top max_primitives by weight for compute budgeting.
    This is a declared budgeting operation (not hidden).

    Args:
        prim_map: Source primitive map
        max_primitives: Optional limit on primitives (by weight)
        eps_lift: Regularization for matrix inversion
        eps_mass: Regularization for direction normalization

    Returns:
        PrimitiveMapView with positions, covariances, directions, kappas, etc.
    """
    if prim_map.count == 0:
        return PrimitiveMapView(
            positions=jnp.zeros((0, 3), dtype=jnp.float64),
            covariances=jnp.zeros((0, 3, 3), dtype=jnp.float64),
            directions=jnp.zeros((0, 3), dtype=jnp.float64),
            kappas=jnp.zeros((0,), dtype=jnp.float64),
            weights=jnp.zeros((0,), dtype=jnp.float64),
            primitive_ids=jnp.zeros((0,), dtype=jnp.int64),
            colors=jnp.zeros((0, 3), dtype=jnp.float64),
        )

    # Get valid indices
    valid_indices = jnp.where(prim_map.valid_mask, size=prim_map.count)[0]

    # Optional downselection by weight
    if max_primitives is not None and prim_map.count > max_primitives:
        # Sort by weight (descending) and take top max_primitives
        weights_valid = prim_map.weights[valid_indices]
        top_indices = jnp.argsort(-weights_valid)[:max_primitives]
        valid_indices = valid_indices[top_indices]

    # Extract valid entries
    Lambdas = prim_map.Lambdas[valid_indices]
    thetas = prim_map.thetas[valid_indices]
    etas = prim_map.etas[valid_indices]
    weights = prim_map.weights[valid_indices]
    primitive_ids = prim_map.primitive_ids[valid_indices]
    colors = prim_map.colors[valid_indices]
    valid_mask = jnp.ones(len(valid_indices), dtype=bool)

    eps_lift_j = jnp.asarray(eps_lift, dtype=jnp.float64)
    eps_mass_j = jnp.asarray(eps_mass, dtype=jnp.float64)

    positions, covariances, directions, kappas, weights, prim_ids, colors = \
        _extract_primitive_map_view_core(
            Lambdas, thetas, etas, weights, primitive_ids, colors, valid_mask,
            eps_lift_j, eps_mass_j,
        )

    return PrimitiveMapView(
        positions=positions,
        covariances=covariances,
        directions=directions,
        kappas=kappas,
        weights=weights,
        primitive_ids=prim_ids,
        colors=colors,
    )


# =============================================================================
# Map Maintenance Operators
# =============================================================================


@dataclass
class PrimitiveMapInsertResult:
    """Result of PrimitiveMapInsert operator."""
    prim_map: PrimitiveMap
    n_inserted: int
    new_ids: jnp.ndarray  # IDs assigned to new primitives


def primitive_map_insert(
    prim_map: PrimitiveMap,
    Lambdas_new: jnp.ndarray,   # (M, 3, 3)
    thetas_new: jnp.ndarray,    # (M, 3)
    etas_new: jnp.ndarray,      # (M, 3)
    weights_new: jnp.ndarray,   # (M,)
    timestamp: float,
    colors_new: Optional[jnp.ndarray] = None,  # (M, 3)
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapInsertResult, CertBundle, ExpectedEffect]:
    """
    Insert new primitives into the map.

    Fixed-cost operator. If map is full, returns without inserting
    (caller should call cull first).

    Args:
        prim_map: Current primitive map
        Lambdas_new: Precision matrices for new primitives
        thetas_new: Information vectors
        etas_new: vMF natural parameters
        weights_new: Initial weights
        timestamp: Current time
        colors_new: Optional RGB colors

    Returns:
        (result, CertBundle, ExpectedEffect)
    """
    M = Lambdas_new.shape[0]
    max_size = prim_map.Lambdas.shape[0]
    available = max_size - prim_map.count
    n_to_insert = min(M, available)

    if n_to_insert == 0:
        # No space - return unchanged
        result = PrimitiveMapInsertResult(
            prim_map=prim_map,
            n_inserted=0,
            new_ids=jnp.array([], dtype=jnp.int64),
        )
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(
            objective_name="primitive_map_insert",
            predicted=float(M),
            realized=0.0,
        )
        return result, cert, effect

    # Find empty slots
    empty_indices = jnp.where(~prim_map.valid_mask, size=n_to_insert)[0]

    # Assign new IDs
    new_ids = jnp.arange(prim_map.next_id, prim_map.next_id + n_to_insert, dtype=jnp.int64)

    # Update arrays
    Lambdas = prim_map.Lambdas.at[empty_indices].set(Lambdas_new[:n_to_insert])
    thetas = prim_map.thetas.at[empty_indices].set(thetas_new[:n_to_insert])
    etas = prim_map.etas.at[empty_indices].set(etas_new[:n_to_insert])
    weights = prim_map.weights.at[empty_indices].set(weights_new[:n_to_insert])
    timestamps = prim_map.timestamps.at[empty_indices].set(timestamp)
    primitive_ids = prim_map.primitive_ids.at[empty_indices].set(new_ids)
    valid_mask = prim_map.valid_mask.at[empty_indices].set(True)

    if colors_new is not None:
        colors = prim_map.colors.at[empty_indices].set(colors_new[:n_to_insert])
    else:
        colors = prim_map.colors

    new_map = PrimitiveMap(
        Lambdas=Lambdas,
        thetas=thetas,
        etas=etas,
        weights=weights,
        timestamps=timestamps,
        primitive_ids=primitive_ids,
        valid_mask=valid_mask,
        colors=colors,
        next_id=prim_map.next_id + n_to_insert,
        count=prim_map.count + n_to_insert,
    )

    result = PrimitiveMapInsertResult(
        prim_map=new_map,
        n_inserted=n_to_insert,
        new_ids=new_ids,
    )
    cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    effect = ExpectedEffect(
        objective_name="primitive_map_insert",
        predicted=float(M),
        realized=float(n_to_insert),
    )
    return result, cert, effect


@dataclass
class PrimitiveMapFuseResult:
    """Result of PrimitiveMapFuse operator."""
    prim_map: PrimitiveMap
    n_fused: int


def primitive_map_fuse(
    prim_map: PrimitiveMap,
    target_indices: jnp.ndarray,       # (K,) indices into map
    Lambdas_meas: jnp.ndarray,         # (K, 3, 3) measurement precisions
    thetas_meas: jnp.ndarray,          # (K, 3) measurement info vectors
    etas_meas: jnp.ndarray,            # (K, 3) measurement vMF params
    weights_meas: jnp.ndarray,         # (K,) measurement weights
    responsibilities: jnp.ndarray,     # (K,) soft association weights
    timestamp: float,
    valid_mask: Optional[jnp.ndarray] = None,  # (K,) bool; when provided, zero out invalid before segment_sum
    eps_psd: float = constants.GC_EPS_PSD,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapFuseResult, CertBundle, ExpectedEffect]:
    """
    Fuse measurement primitives into map primitives via Product-of-Experts.

    Gaussian info fusion: Lambda_post = Lambda_prior + sum_k r_k * Lambda_meas_k
    vMF natural param addition: eta_post = eta_prior + sum_k r_k * eta_meas_k

    Fixed-cost operator. Always applies (no gates).
    When valid_mask is provided (e.g. from JAX-only flatten), invalid entries contribute zero.

    Args:
        prim_map: Current primitive map
        target_indices: Map indices to fuse into
        Lambdas_meas: Measurement precision matrices
        thetas_meas: Measurement information vectors
        etas_meas: Measurement vMF natural parameters
        weights_meas: Measurement weights
        responsibilities: Soft association weights (from OT)
        timestamp: Current time
        valid_mask: Optional (K,) bool; True = include in fusion (enables fixed-size JAX flatten)
        eps_psd: PSD regularization

    Returns:
        (result, CertBundle, ExpectedEffect)
    """
    K = target_indices.shape[0]
    if K == 0:
        result = PrimitiveMapFuseResult(prim_map=prim_map, n_fused=0)
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(objective_name="primitive_map_fuse", predicted=0.0, realized=0.0)
        return result, cert, effect

    # Optional mask: zero out invalid entries (fixed-size flatten from JAX)
    if valid_mask is not None:
        m = valid_mask[:, None, None].astype(jnp.float64)
        m_vec = valid_mask[:, None].astype(jnp.float64)
        r = responsibilities[:, None, None] * m
        r_vec = (responsibilities[:, None] * m_vec).astype(jnp.float64)
        weight_updates = (responsibilities * weights_meas * valid_mask).astype(jnp.float64)
    else:
        r = responsibilities[:, None, None]
        r_vec = responsibilities[:, None]
        weight_updates = (responsibilities * weights_meas).astype(jnp.float64)

    Lambda_updates = r * Lambdas_meas    # (K, 3, 3)
    theta_updates = r_vec * thetas_meas  # (K, 3)
    eta_updates = r_vec * etas_meas      # (K, 3)

    # Accumulate updates per target index
    # Use segment_sum for efficiency
    unique_targets, inverse = jnp.unique(target_indices, return_inverse=True)
    n_unique = unique_targets.shape[0]

    # Aggregate updates
    Lambda_agg = jax.ops.segment_sum(Lambda_updates, inverse, num_segments=n_unique)
    theta_agg = jax.ops.segment_sum(theta_updates, inverse, num_segments=n_unique)
    eta_agg = jax.ops.segment_sum(eta_updates, inverse, num_segments=n_unique)
    weight_agg = jax.ops.segment_sum(weight_updates, inverse, num_segments=n_unique)

    # Update map at unique targets
    Lambdas = prim_map.Lambdas.at[unique_targets].add(Lambda_agg)
    thetas = prim_map.thetas.at[unique_targets].add(theta_agg)
    etas = prim_map.etas.at[unique_targets].add(eta_agg)
    weights = prim_map.weights.at[unique_targets].add(weight_agg)
    timestamps = prim_map.timestamps.at[unique_targets].set(timestamp)

    new_map = PrimitiveMap(
        Lambdas=Lambdas,
        thetas=thetas,
        etas=etas,
        weights=weights,
        timestamps=timestamps,
        primitive_ids=prim_map.primitive_ids,
        valid_mask=prim_map.valid_mask,
        colors=prim_map.colors,
        next_id=prim_map.next_id,
        count=prim_map.count,
    )

    result = PrimitiveMapFuseResult(prim_map=new_map, n_fused=int(n_unique))
    cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    effect = ExpectedEffect(
        objective_name="primitive_map_fuse",
        predicted=float(K),
        realized=float(n_unique),
    )
    return result, cert, effect


@dataclass
class PrimitiveMapCullResult:
    """Result of PrimitiveMapCull operator."""
    prim_map: PrimitiveMap
    n_culled: int
    mass_dropped: float  # Total weight removed (logged as approximation)


def primitive_map_cull(
    prim_map: PrimitiveMap,
    weight_threshold: float = constants.GC_PRIMITIVE_CULL_WEIGHT_THRESHOLD,
    max_primitives: Optional[int] = None,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapCullResult, CertBundle, ExpectedEffect]:
    """
    Cull low-weight primitives from the map (explicit budgeting operator).

    Objective: resource constraint (retain only primitives with weight >= threshold
    and/or up to max_primitives); mass_dropped is logged. No hidden gate; declared
    budgeting operator. If mixture/family changes, Frobenius applied and logged.

    Args:
        prim_map: Current primitive map
        weight_threshold: Minimum weight to retain (tau)
        max_primitives: Optional hard limit (keeps highest weight)

    Returns:
        (result, CertBundle, ExpectedEffect) with mass_dropped logged
    """
    if prim_map.count == 0:
        result = PrimitiveMapCullResult(prim_map=prim_map, n_culled=0, mass_dropped=0.0)
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(objective_name="primitive_map_cull", predicted=0.0, realized=0.0)
        return result, cert, effect

    # Find primitives below threshold
    below_threshold = prim_map.valid_mask & (prim_map.weights < weight_threshold)

    # If max_primitives specified, also cull excess
    n_to_keep = prim_map.count - int(jnp.sum(below_threshold))
    if max_primitives is not None and n_to_keep > max_primitives:
        # Need to cull more - keep top max_primitives by weight
        # Set threshold to weight of (max_primitives+1)th highest
        sorted_weights = jnp.sort(prim_map.weights * prim_map.valid_mask.astype(jnp.float64))[::-1]
        if max_primitives < len(sorted_weights):
            effective_threshold = float(sorted_weights[max_primitives])
            below_threshold = prim_map.valid_mask & (prim_map.weights < effective_threshold)

    # Compute mass dropped
    mass_dropped = float(jnp.sum(prim_map.weights * below_threshold.astype(jnp.float64)))
    n_culled = int(jnp.sum(below_threshold))

    if n_culled == 0:
        result = PrimitiveMapCullResult(prim_map=prim_map, n_culled=0, mass_dropped=0.0)
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(objective_name="primitive_map_cull", predicted=0.0, realized=0.0)
        return result, cert, effect

    # Clear culled entries
    valid_mask = prim_map.valid_mask & ~below_threshold
    new_count = prim_map.count - n_culled

    new_map = PrimitiveMap(
        Lambdas=prim_map.Lambdas,
        thetas=prim_map.thetas,
        etas=prim_map.etas,
        weights=prim_map.weights,
        timestamps=prim_map.timestamps,
        primitive_ids=prim_map.primitive_ids,
        valid_mask=valid_mask,
        colors=prim_map.colors,
        next_id=prim_map.next_id,
        count=new_count,
    )

    result = PrimitiveMapCullResult(
        prim_map=new_map,
        n_culled=n_culled,
        mass_dropped=mass_dropped,
    )

    # Log as approximation: budgeting operator; mass dropped is logged
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["budgeting", "mass_drop"],
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=mass_dropped / (jnp.sum(prim_map.weights) + constants.GC_EPS_MASS),
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )
    effect = ExpectedEffect(
        objective_name="primitive_map_cull",
        predicted=float(n_culled),
        realized=float(n_culled),
    )
    return result, cert, effect


@dataclass
class PrimitiveMapForgetResult:
    """Result of PrimitiveMapForget operator."""
    prim_map: PrimitiveMap


def primitive_map_forget(
    prim_map: PrimitiveMap,
    forgetting_factor: float = constants.GC_PRIMITIVE_FORGETTING_FACTOR,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapForgetResult, CertBundle, ExpectedEffect]:
    """
    Apply continuous forgetting to primitive weights.

    Fixed-cost operator applied every scan. No if/else.

    Args:
        prim_map: Current primitive map
        forgetting_factor: Decay factor in (0, 1), closer to 1 = slower

    Returns:
        (result, CertBundle, ExpectedEffect)
    """
    gamma = float(forgetting_factor)

    new_map = PrimitiveMap(
        Lambdas=prim_map.Lambdas,
        thetas=prim_map.thetas,
        etas=prim_map.etas,
        weights=gamma * prim_map.weights,
        timestamps=prim_map.timestamps,
        primitive_ids=prim_map.primitive_ids,
        valid_mask=prim_map.valid_mask,
        colors=prim_map.colors,
        next_id=prim_map.next_id,
        count=prim_map.count,
    )

    result = PrimitiveMapForgetResult(prim_map=new_map)
    cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    effect = ExpectedEffect(
        objective_name="primitive_map_forget",
        predicted=1.0 - gamma,
        realized=1.0 - gamma,
    )
    return result, cert, effect


# =============================================================================
# Merge-Reduce (Mixture Reduction)
# =============================================================================


@dataclass
class PrimitiveMapMergeReduceResult:
    """Result of PrimitiveMapMergeReduce operator."""
    prim_map: PrimitiveMap
    n_merged: int
    frobenius_correction: float  # Applied when out-of-family


def primitive_map_merge_reduce(
    prim_map: PrimitiveMap,
    merge_threshold: float = constants.GC_PRIMITIVE_MERGE_THRESHOLD,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_map",
) -> Tuple[PrimitiveMapMergeReduceResult, CertBundle, ExpectedEffect]:
    """
    Merge nearby primitives via mixture reduction.

    Uses Bhattacharyya distance to identify merge candidates.
    Merged primitive = weighted combination of Gaussians (moment matching).

    Fixed-cost operator with CertBundle + Frobenius correction.

    Args:
        prim_map: Current primitive map
        merge_threshold: Distance below which to merge
        eps_psd: PSD regularization
        eps_lift: Matrix inversion regularization

    Returns:
        (result, CertBundle, ExpectedEffect)
    """
    # For now, implement a simple version that doesn't merge
    # Full implementation would require nearest-neighbor search
    # and careful handling of the Gaussian/vMF mixture reduction

    # TODO: Implement actual merge logic with:
    # 1. Build k-d tree or spatial hash of primitive positions
    # 2. Find pairs with Bhattacharyya distance < threshold
    # 3. Merge via moment matching (weight-preserving)
    # 4. Log Frobenius correction for out-of-family approximation

    result = PrimitiveMapMergeReduceResult(
        prim_map=prim_map,
        n_merged=0,
        frobenius_correction=0.0,
    )
    cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
    effect = ExpectedEffect(
        objective_name="primitive_map_merge_reduce",
        predicted=0.0,
        realized=0.0,
    )
    return result, cert, effect
