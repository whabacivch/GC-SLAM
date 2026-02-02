"""
Primitive Association via Optimal Transport for Geometric Compositional SLAM v2.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md Section 6

OT is the canonical association operator. Single association path (no gates).

Operator: associate_primitives_ot(MeasurementBatch, PrimitiveMapView) -> (pi, Cert, Effect)

Key constraints:
- Candidate generation is MA hex web (hex cells + stencil, nearest k_assoc); inside the operator
- Output pi is always shape [N_meas, K_ASSOC] (sparse-by-design)
- Fixed-cost: K_SINKHORN iterations, K_ASSOC candidates per measurement
- Responsibilities are the only association mechanism (no nearest-neighbor, no "if residual small")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import CertBundle, ExpectedEffect, InfluenceCert
from fl_slam_poc.common.ma_hex_web import MAHexWebConfig, generate_candidates_ma_hex_web
from fl_slam_poc.backend.structures.measurement_batch import MeasurementBatch
from fl_slam_poc.backend.structures.primitive_map import PrimitiveMapView



# =============================================================================
# Association Result
# =============================================================================


@dataclass
class PrimitiveAssociationResult:
    """Result of primitive association via OT."""
    # Sparse responsibilities: (N_meas, K_ASSOC)
    # pi[i, k] = responsibility of measurement i to candidate k
    responsibilities: jnp.ndarray  # (N_meas, K_ASSOC)

    # Candidate indices: (N_meas, K_ASSOC)
    # candidate_indices[i, k] = map primitive index for measurement i, candidate k
    candidate_indices: jnp.ndarray  # (N_meas, K_ASSOC) int

    # Per-measurement total mass (for diagnostics)
    row_masses: jnp.ndarray  # (N_meas,)

    # Cost matrix for diagnostics
    cost_matrix: jnp.ndarray  # (N_meas, K_ASSOC)


# =============================================================================
# Candidate Generation (MA hex web; fixed topology, JIT-friendly)
# =============================================================================


# =============================================================================
# Cost Computation
# =============================================================================


def _sinkhorn_unbalanced_fixed_k(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    tau_a: float,
    tau_b: float,
    K: int,
) -> np.ndarray:
    """
    Unbalanced Sinkhorn: fixed K iterations. KL relaxation on marginals (continuous; no threshold).
    min_π <π,C> + ε KL(π|a⊗b) + τ_a KL(π1|a) + τ_b KL(πᵀ1|b).
    Updates: u = (a / (K v))^(1/(1+τ_a/ε)), v = (b / (Kᵀ u))^(1/(1+τ_b/ε)).
    τ_a=τ_b=0 recovers balanced. Returns coupling π (N,M).
    """
    C = np.asarray(C, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    N, M = C.shape
    eps = max(float(epsilon), 1e-12)
    K_mat = np.exp(-C / eps)
    u = np.ones(N, dtype=np.float64)
    v = np.ones(M, dtype=np.float64)
    ua = 1.0 / (1.0 + float(tau_a) / eps)
    vb = 1.0 / (1.0 + float(tau_b) / eps)
    for _ in range(int(K)):
        Kv = K_mat @ v
        u = (a / (Kv + 1e-12)) ** ua
        KTu = K_mat.T @ u
        v = (b / (KTu + 1e-12)) ** vb
    pi = u.reshape(-1, 1) * K_mat * v.reshape(1, -1)
    return pi


def _compute_sparse_cost_matrix(
    meas_positions: np.ndarray,      # (N_meas, 3)
    meas_covariances: np.ndarray,    # (N_meas, 3, 3)
    meas_directions: np.ndarray,     # (N_meas, 3)
    meas_kappas: np.ndarray,         # (N_meas,)
    map_positions: np.ndarray,       # (M_map, 3)
    map_covariances: np.ndarray,     # (M_map, 3, 3)
    map_directions: np.ndarray,      # (M_map, 3)
    map_kappas: np.ndarray,          # (M_map,)
    candidate_indices: np.ndarray,   # (N_meas, K_ASSOC)
    beta: float = 0.5,
    eig_min: float = 1e-12,
) -> np.ndarray:
    """
    Compute sparse cost matrix for (measurement, candidate) pairs.

    Cost = W2^2(position) + beta * H^2_vMF(direction)

    Fully vectorized (no Python loops).

    Args:
        meas_*: Measurement batch attributes
        map_*: Map primitive attributes
        candidate_indices: (N_meas, K_ASSOC) map indices per measurement
        beta: Direction weight
        eig_min: SPD clamp for W2^2

    Returns:
        cost_matrix: (N_meas, K_ASSOC) costs
    """
    N_meas, K_assoc = candidate_indices.shape

    if N_meas == 0 or K_assoc == 0:
        return np.zeros((N_meas, K_assoc), dtype=np.float64)

    # Gather all candidate data: (N_meas, K_assoc, 3) and (N_meas, K_assoc)
    map_pos_all = map_positions[candidate_indices]  # (N_meas, K_assoc, 3)
    map_dir_all = map_directions[candidate_indices]  # (N_meas, K_assoc, 3)
    map_kappa_all = map_kappas[candidate_indices]  # (N_meas, K_assoc)

    # Position cost: ||meas_pos[i] - map_pos[i,k]||^2
    # (N_meas, 1, 3) - (N_meas, K_assoc, 3) -> (N_meas, K_assoc, 3)
    diff = meas_positions[:, None, :] - map_pos_all
    d_pos = np.sum(diff ** 2, axis=-1)  # (N_meas, K_assoc)

    # Direction cost: Hellinger^2 for vMF (vectorized)
    # eta1 = kappa1 * mu1, eta2 = kappa2 * mu2
    # km = 0.5 * ||eta1 + eta2||
    # H^2 = 1 - exp(A(km) - 0.5*(A(k1) + A(k2)))
    meas_eta = meas_kappas[:, None, None] * meas_directions[:, None, :]  # (N_meas, 1, 3)
    map_eta = map_kappa_all[:, :, None] * map_dir_all  # (N_meas, K_assoc, 3)
    eta_sum = meas_eta + map_eta  # (N_meas, K_assoc, 3)
    km = 0.5 * np.linalg.norm(eta_sum, axis=-1)  # (N_meas, K_assoc)

    # A_vmf(k) = log(4*pi) + log(sinh(k)) - log(k)
    # For stability: log(sinh(k)) = k - log(2) for large k
    def _A_vmf_vec(k):
        """Vectorized A_vmf function."""
        k = np.maximum(k, eig_min)
        # For small k: log(sinh(k)) ≈ log(k)
        # For large k: log(sinh(k)) ≈ k - log(2)
        log_sinh_k = np.where(
            k > 20.0,
            k - np.log(2.0),
            np.where(k >= 1e-2, np.log(np.sinh(k)), np.log(k + (k ** 3) / 6.0))
        )
        return np.log(4.0 * np.pi) + log_sinh_k - np.log(k)

    km_safe = np.maximum(km, eig_min)
    k1_safe = np.maximum(meas_kappas[:, None], eig_min)  # (N_meas, 1)
    k2_safe = np.maximum(map_kappa_all, eig_min)  # (N_meas, K_assoc)

    A_km = _A_vmf_vec(km_safe)
    A_k1 = _A_vmf_vec(k1_safe)
    A_k2 = _A_vmf_vec(k2_safe)

    bc = np.exp(A_km - 0.5 * (A_k1 + A_k2))
    d_dir = np.maximum(0.0, 1.0 - bc)  # (N_meas, K_assoc)

    # Mask out direction cost where either kappa is zero
    valid_dir = (meas_kappas[:, None] > 0) & (map_kappa_all > 0)
    d_dir = np.where(valid_dir, d_dir, 0.0)

    # Total cost
    cost_matrix = d_pos + beta * d_dir

    return cost_matrix


# =============================================================================
# Main Association Operator
# =============================================================================


@dataclass
class AssociationConfig:
    """Configuration for primitive association."""
    k_assoc: int = constants.GC_K_ASSOC
    k_sinkhorn: int = constants.GC_K_SINKHORN
    beta: float = 0.5  # Direction weight
    epsilon: float = 0.1  # Entropy regularization
    tau_a: float = 0.5  # Unbalanced KL for measurement marginal
    tau_b: float = 0.5  # Unbalanced KL for map marginal
    cost_subtract_row_min: bool = True
    cost_scale_by_median: bool = False


def associate_primitives_ot(
    measurement_batch: MeasurementBatch,
    map_view: PrimitiveMapView,
    config: AssociationConfig = None,
    eps_lift: float = constants.GC_EPS_LIFT,
    eps_mass: float = constants.GC_EPS_MASS,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "primitive_ot",
) -> Tuple[PrimitiveAssociationResult, CertBundle, ExpectedEffect]:
    """
    Associate measurement primitives to map primitives via Sinkhorn OT.

    Fixed-cost operator. Candidate generation is inside this operator.
    Output shape is always [N_meas, K_ASSOC].

    Args:
        measurement_batch: Measurement primitives (camera + lidar)
        map_view: Map primitive view
        config: Association configuration
        eps_lift: Matrix regularization
        eps_mass: Mass regularization

    Returns:
        (result, CertBundle, ExpectedEffect)
    """
    if config is None:
        config = AssociationConfig()

    N_meas = measurement_batch.n_valid
    N_total = measurement_batch.n_total
    M_map = map_view.count

    # Handle empty cases (return fixed shape N_total x K_assoc for JAX-only flatten)
    if N_meas == 0 or M_map == 0:
        result = PrimitiveAssociationResult(
            responsibilities=jnp.zeros((N_total, config.k_assoc), dtype=jnp.float64),
            candidate_indices=jnp.zeros((N_total, config.k_assoc), dtype=jnp.int64),
            row_masses=jnp.zeros((N_total,), dtype=jnp.float64),
            cost_matrix=jnp.zeros((N_total, config.k_assoc), dtype=jnp.float64),
        )
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(
            objective_name="primitive_association_ot",
            predicted=0.0,
            realized=0.0,
        )
        return result, cert, effect

    # Extract numpy arrays for computation
    # Measurement batch
    from fl_slam_poc.backend.structures.measurement_batch import (
        measurement_batch_mean_positions,
        measurement_batch_mean_directions,
        measurement_batch_kappas,
    )

    meas_positions = np.array(measurement_batch_mean_positions(measurement_batch, eps_lift=eps_lift))
    meas_directions = np.array(measurement_batch_mean_directions(measurement_batch, eps_mass=eps_mass))
    meas_kappas = np.array(measurement_batch_kappas(measurement_batch))

    # Covariances from info form
    Lambda_reg = measurement_batch.Lambdas + eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]
    meas_covariances = np.array(jax.vmap(jnp.linalg.inv)(Lambda_reg))

    # Filter to valid measurements
    valid_mask = np.array(measurement_batch.valid_mask)
    valid_indices = np.where(valid_mask)[0][:N_meas]

    meas_positions = meas_positions[valid_indices]
    meas_directions = meas_directions[valid_indices]
    meas_kappas = meas_kappas[valid_indices]
    meas_covariances = meas_covariances[valid_indices]

    # Map view
    map_positions = np.array(map_view.positions)
    map_directions = np.array(map_view.directions)
    map_kappas = np.array(map_view.kappas)
    map_covariances = np.array(map_view.covariances)

    # Generate candidates (MA hex web: hex cells + stencil, nearest k_assoc per measurement)
    hex_config = MAHexWebConfig()
    candidate_indices = generate_candidates_ma_hex_web(
        meas_positions=meas_positions,
        map_positions=map_positions,
        map_covariances=map_covariances,
        k_assoc=config.k_assoc,
        config=hex_config,
    )

    # Compute sparse cost matrix
    cost_matrix = _compute_sparse_cost_matrix(
        meas_positions=meas_positions,
        meas_covariances=meas_covariances,
        meas_directions=meas_directions,
        meas_kappas=meas_kappas,
        map_positions=map_positions,
        map_covariances=map_covariances,
        map_directions=map_directions,
        map_kappas=map_kappas,
        candidate_indices=candidate_indices,
        beta=config.beta,
    )

    # Cost normalization
    if config.cost_subtract_row_min:
        row_min = np.min(cost_matrix, axis=1, keepdims=True)
        cost_matrix = cost_matrix - row_min

    if config.cost_scale_by_median:
        C_flat = cost_matrix[np.isfinite(cost_matrix)]
        med = float(np.median(C_flat)) if C_flat.size > 0 else 1.0
        cost_matrix = cost_matrix / (med + 1e-12)

    # Uniform marginals
    a = np.ones(N_meas, dtype=np.float64) / N_meas
    b = np.ones(config.k_assoc, dtype=np.float64) / config.k_assoc

    # Run Sinkhorn (unbalanced only; no balanced path)
    pi = _sinkhorn_unbalanced_fixed_k(
        C=cost_matrix,
        a=a,
        b=b,
        epsilon=config.epsilon,
        tau_a=config.tau_a,
        tau_b=config.tau_b,
        K=config.k_sinkhorn,
    )
    triggers = ["sinkhorn_fixed_iter", "sinkhorn_unbalanced_kl_relax"]

    # Normalize responsibilities per row (soft assignment)
    row_masses = np.sum(pi, axis=1)
    row_masses_safe = np.maximum(row_masses, 1e-12)
    responsibilities = pi / row_masses_safe[:, None]

    # Pad to fixed shape (N_total, K_assoc) so flatten can be JAX-only (no host round-trip).
    # Invalid rows get responsibilities=0, candidate_indices=0.
    N_total = measurement_batch.n_total
    responsibilities_full = np.zeros((N_total, config.k_assoc), dtype=np.float64)
    responsibilities_full[valid_indices, :] = responsibilities
    candidate_indices_full = np.zeros((N_total, config.k_assoc), dtype=np.int64)
    candidate_indices_full[valid_indices, :] = candidate_indices
    row_masses_full = np.zeros(N_total, dtype=np.float64)
    row_masses_full[valid_indices] = row_masses
    cost_matrix_full = np.zeros((N_total, config.k_assoc), dtype=np.float64)
    cost_matrix_full[valid_indices, :] = cost_matrix

    # Build result (fixed shape for JAX-only flatten)
    result = PrimitiveAssociationResult(
        responsibilities=jnp.array(responsibilities_full, dtype=jnp.float64),
        candidate_indices=jnp.array(candidate_indices_full, dtype=jnp.int64),
        row_masses=jnp.array(row_masses_full, dtype=jnp.float64),
        cost_matrix=jnp.array(cost_matrix_full, dtype=jnp.float64),
    )

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=triggers,
        frobenius_applied=False,
    )

    total_cost = float(np.sum(pi * cost_matrix))
    effect = ExpectedEffect(
        objective_name="primitive_association_ot",
        predicted=total_cost,
        realized=total_cost,
    )

    return result, cert, effect


# =============================================================================
# Utility: Flatten Sparse Associations for Map Update (JAX-only, no threshold)
# =============================================================================


def flatten_associations_for_fuse(
    result: PrimitiveAssociationResult,
    measurement_batch: MeasurementBatch,
    n_valid: int,
    responsibility_threshold: float = 0.0,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Flatten associations to fixed-size flat arrays for primitive_map_fuse.

    Pure JAX (no host round-trip). Result has fixed shape (N_total, K_ASSOC);
    invalid rows have responsibilities=0. We include all pairs (no hard gate);
    responsibility_threshold=0 excludes only exactly-zero mass.
    valid_flat marks rows that correspond to valid measurements (row < n_valid).

    Returns:
        target_indices: (N_total * K_assoc,) map primitive indices
        Lambdas_meas: (N_total * K_assoc, 3, 3)
        thetas_meas: (N_total * K_assoc, 3)
        etas_meas: (N_total * K_assoc, B, 3)
        weights_meas: (N_total * K_assoc,)
        responsibilities: (N_total * K_assoc,)
        valid_flat: (N_total * K_assoc,) bool — True where row < n_valid
        colors_meas: (N_total * K_assoc, 3) measurement RGB per flat row (for fuse color blend)
    """
    N_total, K_assoc = result.responsibilities.shape
    flat_size = N_total * K_assoc

    flat_idx = jnp.arange(flat_size, dtype=jnp.int32)
    row = flat_idx // K_assoc
    col = flat_idx % K_assoc

    # Continuous: include all pairs; invalid rows have resp=0 (no arbitrary threshold)
    resp_flat = result.responsibilities[row, col]
    target_flat = result.candidate_indices[row, col]

    # Optional: exclude exactly-zero responsibility (threshold=0) or use tiny eps
    eps = jnp.maximum(responsibility_threshold, 1e-12)
    include = resp_flat > eps
    resp_flat = jnp.where(include, resp_flat, 0.0)

    # Measurement data: row indexes batch (fixed layout)
    Lambdas_flat = measurement_batch.Lambdas[row]
    thetas_flat = measurement_batch.thetas[row]
    etas_flat = measurement_batch.etas[row]
    weights_flat = measurement_batch.weights[row]
    colors_flat = measurement_batch.colors[row]  # (flat_size, 3) for fuse color blend

    # Valid mask: only rows with row < n_valid correspond to real measurements
    valid_flat = row < n_valid

    return (
        target_flat,
        Lambdas_flat,
        thetas_flat,
        etas_flat,
        weights_flat,
        resp_flat,
        valid_flat,
        colors_flat,
    )
