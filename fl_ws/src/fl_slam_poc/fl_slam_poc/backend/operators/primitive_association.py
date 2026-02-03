"""
Primitive Association via Optimal Transport for Geometric Compositional SLAM v2.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md Section 6

OT is the canonical association operator. Single association path (no gates).

Operator: associate_primitives_ot(MeasurementBatch, PrimitiveMapView) -> (pi, Cert, Effect)

Key constraints:
- Candidate generation is MA hex web (hex cells + stencil, nearest k_assoc); inside the operator
- Output pi is always shape [N_total, K_ASSOC] (fixed-cost, sparse-by-design)
- Fixed-cost: K_SINKHORN iterations, K_ASSOC candidates per measurement
- Responsibilities are the only association mechanism (no nearest-neighbor, no "if residual small")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    InfluenceCert,
    SupportCert,
    ComputeCert,
)


# =============================================================================
# Mass Policy Enums (explicit declaration of OT marginal semantics)
# =============================================================================


class MeasurementMassPolicy(Enum):
    """
    Policy for measurement-side marginal 'a' in OT.

    Spec §5.7.2: Mass budgets are declared explicitly, not derived from data.
    """
    UNIFORM = "uniform"  # a[i] = 1/N_valid (uniform over valid measurements)
    WEIGHT_PROPORTIONAL = "weight_proportional"  # a[i] ∝ measurement_batch.weights[i]
    FEATURE_CONFIDENCE = "feature_confidence"  # a[i] ∝ feature detection confidence (future)


class MapMassPolicy(Enum):
    """
    Policy for map-side marginal 'b' in OT.

    Spec §5.7.2: Mass budgets are declared explicitly.
    """
    UNIFORM = "uniform"  # b[k] = 1/K_assoc (uniform over candidates)
    PRIMITIVE_MASS = "primitive_mass"  # b[k] ∝ map primitive weight (accumulated ESS)
    MASS_TEMPERED = "mass_tempered"  # b[k] ∝ primitive_weight^beta (future)
from fl_slam_poc.common.ma_hex_web import (
    MAHexWebConfig,
    generate_candidates_ma_hex_web_jax,
)
from fl_slam_poc.backend.structures.measurement_batch import MeasurementBatch
from fl_slam_poc.backend.structures.primitive_map import PrimitiveMapView



# =============================================================================
# Association Result
# =============================================================================


@dataclass
class PrimitiveAssociationResult:
    """Result of primitive association via OT."""
    # Sparse responsibilities: (N_total, K_ASSOC)
    # pi[i, k] = responsibility of measurement i to candidate k
    responsibilities: jnp.ndarray  # (N_total, K_ASSOC)

    # Candidate addressing: (N_total, K_ASSOC)
    # candidate_tile_ids[i, k] = tile ID for measurement i, candidate k
    # candidate_slots[i, k] = slot index (tile-local) for measurement i, candidate k
    candidate_tile_ids: jnp.ndarray  # (N_total, K_ASSOC) int
    candidate_slots: jnp.ndarray  # (N_total, K_ASSOC) int

    # Per-measurement total mass (for diagnostics)
    row_masses: jnp.ndarray  # (N_total,)

    # Cost matrix for diagnostics
    cost_matrix: jnp.ndarray  # (N_total, K_ASSOC)


# =============================================================================
# Candidate Generation (MA hex web; fixed topology, JIT-friendly)
# =============================================================================


# =============================================================================
# Cost Computation
# =============================================================================


def _sinkhorn_unbalanced_fixed_k_jax(
    C: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    epsilon: float,
    tau_a: float,
    tau_b: float,
    K: int,
) -> jnp.ndarray:
    """
    Unbalanced Sinkhorn (JAX): fixed K iterations, no convergence check.
    """
    C = jnp.asarray(C, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64).reshape(-1)
    b = jnp.asarray(b, dtype=jnp.float64).reshape(-1)
    N, M = C.shape
    eps = jnp.maximum(jnp.asarray(epsilon, dtype=jnp.float64), 1e-12)
    K_mat = jnp.exp(-C / eps)
    u0 = jnp.ones((N,), dtype=jnp.float64)
    v0 = jnp.ones((M,), dtype=jnp.float64)
    ua = 1.0 / (1.0 + jnp.asarray(tau_a, dtype=jnp.float64) / eps)
    vb = 1.0 / (1.0 + jnp.asarray(tau_b, dtype=jnp.float64) / eps)

    def one_iter(_, uv):
        u, v = uv
        Kv = K_mat @ v
        u = (a / (Kv + 1e-12)) ** ua
        KTu = K_mat.T @ u
        v = (b / (KTu + 1e-12)) ** vb
        return (u, v)

    u, v = jax.lax.fori_loop(0, int(K), one_iter, (u0, v0))
    pi = u.reshape(-1, 1) * K_mat * v.reshape(1, -1)
    return pi


def _A_vmf_vec_jax(k: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """A_vmf(k) = log(4*pi) + log(sinh(k)) - log(k), with stable log-sinh."""
    k = jnp.maximum(jnp.asarray(k, dtype=jnp.float64), eps)
    log_sinh_k = jnp.where(
        k > 20.0,
        k - jnp.log(2.0),
        jnp.where(k >= 1e-2, jnp.log(jnp.sinh(k)), jnp.log(k + (k ** 3) / 6.0)),
    )
    return jnp.log(4.0 * jnp.pi) + log_sinh_k - jnp.log(k)


def _compute_sparse_cost_matrix_jax(
    meas_positions: jnp.ndarray,      # (N_total, 3)
    meas_directions: jnp.ndarray,     # (N_total, 3)
    meas_kappas: jnp.ndarray,         # (N_total,)
    map_positions: jnp.ndarray,       # (M_map, 3)
    map_directions: jnp.ndarray,      # (M_map, 3)
    map_kappas: jnp.ndarray,          # (M_map,)
    candidate_indices: jnp.ndarray,   # (N_total, K_ASSOC)
    beta: float = 0.5,
    eig_min: float = 1e-12,
) -> jnp.ndarray:
    """
    Sparse cost C[i,k] for candidate pairs, device-side.

    Cost = ||x_i - x_j||^2 + beta * H^2_vMF(dir_i, dir_j).
    """
    meas_positions = jnp.asarray(meas_positions, dtype=jnp.float64)
    meas_directions = jnp.asarray(meas_directions, dtype=jnp.float64)
    meas_kappas = jnp.asarray(meas_kappas, dtype=jnp.float64)
    map_positions = jnp.asarray(map_positions, dtype=jnp.float64)
    map_directions = jnp.asarray(map_directions, dtype=jnp.float64)
    map_kappas = jnp.asarray(map_kappas, dtype=jnp.float64)
    candidate_indices = jnp.asarray(candidate_indices, dtype=jnp.int32)

    map_pos_all = map_positions[candidate_indices]
    map_dir_all = map_directions[candidate_indices]
    map_kappa_all = map_kappas[candidate_indices]

    diff = meas_positions[:, None, :] - map_pos_all
    d_pos = jnp.sum(diff * diff, axis=-1)

    meas_eta = meas_kappas[:, None, None] * meas_directions[:, None, :]
    map_eta = map_kappa_all[:, :, None] * map_dir_all
    km = 0.5 * jnp.linalg.norm(meas_eta + map_eta, axis=-1)

    km_safe = jnp.maximum(km, eig_min)
    k1_safe = jnp.maximum(meas_kappas[:, None], eig_min)
    k2_safe = jnp.maximum(map_kappa_all, eig_min)
    A_km = _A_vmf_vec_jax(km_safe, eps=eig_min)
    A_k1 = _A_vmf_vec_jax(k1_safe, eps=eig_min)
    A_k2 = _A_vmf_vec_jax(k2_safe, eps=eig_min)
    bc = jnp.exp(A_km - 0.5 * (A_k1 + A_k2))
    d_dir = jnp.maximum(0.0, 1.0 - bc)
    valid_dir = (meas_kappas[:, None] > 0.0) & (map_kappa_all > 0.0)
    d_dir = jnp.where(valid_dir, d_dir, 0.0)
    return d_pos + float(beta) * d_dir


# =============================================================================
# Main Association Operator
# =============================================================================


@dataclass
class AssociationConfig:
    """
    Configuration for primitive association.

    OT parameters (spec §5.7):
    - epsilon: Entropic regularization (larger = smoother)
    - tau_a, tau_b: Unbalanced KL relaxation (smaller = more unbalanced)
    - k_sinkhorn: Fixed iteration count (no convergence check)

    Mass policies (spec §5.7.2): Explicit declaration of OT marginal semantics.
    """
    k_assoc: int = constants.GC_K_ASSOC
    k_sinkhorn: int = constants.GC_K_SINKHORN
    beta: float = 0.5  # Direction weight in cost
    epsilon: float = 0.1  # Entropic regularization
    tau_a: float = 0.5  # Unbalanced KL for measurement marginal
    tau_b: float = 0.5  # Unbalanced KL for map marginal
    cost_subtract_row_min: bool = True
    cost_scale_by_median: bool = False
    # Explicit mass policies (spec §5.7.2)
    a_policy: MeasurementMassPolicy = MeasurementMassPolicy.UNIFORM
    b_policy: MapMassPolicy = MapMassPolicy.UNIFORM
    # Mass regularization (replaces magic 1e-12 constants)
    eps_mass: float = constants.GC_EPS_MASS


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

    N_total = measurement_batch.n_total
    M_map = map_view.count

    # Handle empty cases (return fixed shape N_total x K_assoc for JAX-only flatten)
    if measurement_batch.n_valid == 0 or M_map == 0:
        result = PrimitiveAssociationResult(
            responsibilities=jnp.zeros((N_total, config.k_assoc), dtype=jnp.float64),
            candidate_tile_ids=jnp.zeros((N_total, config.k_assoc), dtype=jnp.int64),
            candidate_slots=jnp.zeros((N_total, config.k_assoc), dtype=jnp.int64),
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

    # Extract device arrays
    from fl_slam_poc.backend.structures.measurement_batch import (
        measurement_batch_mean_positions,
        measurement_batch_mean_directions,
        measurement_batch_kappas,
    )

    meas_positions = measurement_batch_mean_positions(measurement_batch, eps_lift=eps_lift)  # (N_total, 3)
    meas_directions = measurement_batch_mean_directions(measurement_batch, eps_mass=eps_mass)  # (N_total, 3)
    meas_kappas = measurement_batch_kappas(measurement_batch)  # (N_total,)
    valid_mask = measurement_batch.valid_mask.astype(jnp.float64)  # (N_total,)

    map_positions = map_view.positions
    map_directions = map_view.directions
    map_kappas = map_view.kappas
    map_covariances = map_view.covariances

    # Generate candidates (MA hex web: hex cells + stencil, nearest k_assoc per measurement)
    hex_config = MAHexWebConfig()
    candidate_view_indices = generate_candidates_ma_hex_web_jax(
        meas_positions=meas_positions,
        map_positions=map_positions,
        map_covariances=map_covariances,
        k_assoc=config.k_assoc,
        config=hex_config,
    )
    candidate_view_indices = jnp.where(
        valid_mask[:, None] > 0.0, candidate_view_indices, 0
    ).astype(jnp.int32)
    candidate_slots = map_view.slot_indices[candidate_view_indices]
    candidate_tile_ids = jnp.full(
        candidate_slots.shape, int(map_view.tile_id), dtype=jnp.int32
    )

    # Compute sparse cost matrix (device-side)
    cost_matrix = _compute_sparse_cost_matrix_jax(
        meas_positions=meas_positions,
        meas_directions=meas_directions,
        meas_kappas=meas_kappas,
        map_positions=map_positions,
        map_directions=map_directions,
        map_kappas=map_kappas,
        candidate_indices=candidate_view_indices,
        beta=config.beta,
    )

    # Cost normalization
    if config.cost_subtract_row_min:
        row_min = jnp.min(cost_matrix, axis=1, keepdims=True)
        cost_matrix = cost_matrix - row_min

    if config.cost_scale_by_median:
        med = jnp.median(cost_matrix)
        cost_matrix = cost_matrix / (med + 1e-12)

    # Build marginals according to declared mass policies (spec §5.7.2)
    eps_m = config.eps_mass

    # Measurement marginal 'a' based on policy
    if config.a_policy == MeasurementMassPolicy.UNIFORM:
        sum_a = jnp.maximum(jnp.sum(valid_mask), eps_m)
        a = valid_mask / sum_a
    elif config.a_policy == MeasurementMassPolicy.WEIGHT_PROPORTIONAL:
        weighted = valid_mask * measurement_batch.weights.astype(jnp.float64)
        sum_a = jnp.maximum(jnp.sum(weighted), eps_m)
        a = weighted / sum_a
    else:  # FEATURE_CONFIDENCE - fallback to uniform for now
        sum_a = jnp.maximum(jnp.sum(valid_mask), eps_m)
        a = valid_mask / sum_a

    # Map marginal 'b' based on policy
    if config.b_policy == MapMassPolicy.UNIFORM:
        b = jnp.ones((config.k_assoc,), dtype=jnp.float64) / float(config.k_assoc)
    elif config.b_policy == MapMassPolicy.PRIMITIVE_MASS:
        # Note: candidate_indices selects per-measurement candidates; b applies globally
        # For now, use uniform; full implementation requires per-measurement b
        b = jnp.ones((config.k_assoc,), dtype=jnp.float64) / float(config.k_assoc)
    else:  # MASS_TEMPERED - fallback to uniform for now
        b = jnp.ones((config.k_assoc,), dtype=jnp.float64) / float(config.k_assoc)

    sum_b = float(jnp.sum(b))

    # Run Sinkhorn (unbalanced only; fixed iter; device-side)
    pi = _sinkhorn_unbalanced_fixed_k_jax(
        C=cost_matrix,
        a=a,
        b=b,
        epsilon=config.epsilon,
        tau_a=config.tau_a,
        tau_b=config.tau_b,
        K=int(config.k_sinkhorn),
    )
    triggers = ["sinkhorn_fixed_iter", "sinkhorn_unbalanced_kl_relax"]

    # Per spec §5.7.3: Use π directly as responsibilities (no row-normalization).
    # Row-normalization would destroy the unbalanced OT semantics where
    # row_masses = sum_k pi[i,k] encodes the transported mass per measurement.
    # Novelty (spec §5.7.5) = declared budget - transported mass.
    # Only mask invalid rows; do not divide by row_masses.
    row_masses = jnp.sum(pi, axis=1)
    responsibilities = pi * (valid_mask[:, None] > 0.0)  # Only mask invalid rows

    result = PrimitiveAssociationResult(
        responsibilities=responsibilities.astype(jnp.float64),
        candidate_tile_ids=candidate_tile_ids.astype(jnp.int64),
        candidate_slots=candidate_slots.astype(jnp.int64),
        row_masses=row_masses.astype(jnp.float64),
        cost_matrix=cost_matrix.astype(jnp.float64),
    )

    # Compute OT diagnostics for cert (spec §5.7.4)
    # Marginal defects: ||π·1 - a|| and ||π^T·1 - b||
    col_masses = jnp.sum(pi, axis=0)  # sum over rows (measurements) for each candidate
    marginal_defect_a = float(jnp.linalg.norm(row_masses - a))
    marginal_defect_b = float(jnp.linalg.norm(col_masses - b))
    transport_mass_total = float(jnp.sum(pi))

    # Mass statistics for cert (spec §5.7.2)
    n_nonzero_a = int(jnp.sum(a > eps_m))
    n_nonzero_b = int(jnp.sum(b > eps_m))
    # p95 of mass distributions
    a_sorted = jnp.sort(a)
    b_sorted = jnp.sort(b)
    p95_idx_a = int(0.95 * len(a_sorted))
    p95_idx_b = int(0.95 * len(b_sorted))
    p95_a = float(a_sorted[min(p95_idx_a, len(a_sorted) - 1)])
    p95_b = float(b_sorted[min(p95_idx_b, len(b_sorted) - 1)])

    # ESS from transported mass: effective sample size of assignments
    ess_ot = float(jnp.sum(row_masses) ** 2 / (jnp.sum(row_masses ** 2) + eps_m))

    bytes_per_f64 = 8
    alloc_bytes_est = int(N_total * config.k_assoc * bytes_per_f64 * 4)
    compute = ComputeCert(
        alloc_bytes_est=alloc_bytes_est,
        largest_tensor_shape=(int(N_total), int(config.k_assoc)),
        segment_sum_k=int(config.k_assoc),
        psd_projection_count=0,
        chol_solve_count=0,
    )

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=triggers,
        frobenius_applied=False,
        support=SupportCert(
            ess_total=ess_ot,
            support_frac=float(n_nonzero_a) / float(max(N_total, 1)),
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=float(eps_m) / (transport_mass_total + eps_m),
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
        compute=compute,
    )
    # Store OT-specific diagnostics in cert metadata (custom fields via approximation_triggers for now)
    # TODO: Add dedicated OTCert dataclass in future refactor
    cert.approximation_triggers.append(f"ot_defect_a={marginal_defect_a:.6f}")
    cert.approximation_triggers.append(f"ot_defect_b={marginal_defect_b:.6f}")
    cert.approximation_triggers.append(f"ot_transport_mass={transport_mass_total:.6f}")
    cert.approximation_triggers.append(f"ot_sum_a={float(sum_a):.6f}")
    cert.approximation_triggers.append(f"ot_sum_b={sum_b:.6f}")
    cert.approximation_triggers.append(f"ot_p95_a={p95_a:.6f}")
    cert.approximation_triggers.append(f"ot_p95_b={p95_b:.6f}")
    cert.approximation_triggers.append(f"ot_nonzero_a={n_nonzero_a}")
    cert.approximation_triggers.append(f"ot_nonzero_b={n_nonzero_b}")

    total_cost = float(jnp.sum(pi * cost_matrix))
    effect = ExpectedEffect(
        objective_name="primitive_association_ot",
        predicted=total_cost,
        realized=total_cost,
    )

    return result, cert, effect


# =============================================================================
# Utility: Flatten Sparse Associations for Map Update (JAX-only, no threshold)
# =============================================================================


def block_associations_for_fuse(
    result: PrimitiveAssociationResult,
    valid_mask: jnp.ndarray,
    block_size: int = constants.GC_ASSOC_BLOCK_SIZE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Blocked sparse representation for fuse. Fixed block size; no full flatten.

    Returns:
        meas_idx: (N_blocks, block_size) measurement indices (clipped to N_total-1)
        candidate_tile_ids: (N_blocks, block_size, K_assoc)
        candidate_slots: (N_blocks, block_size, K_assoc)
        responsibilities: (N_blocks, block_size, K_assoc) (invalid rows zeroed)
        valid_rows: (N_blocks, block_size) True where row < n_valid
    """
    N_total, K_assoc = result.responsibilities.shape
    block = int(max(1, block_size))
    n_blocks = (N_total + block - 1) // block
    idx = jnp.arange(n_blocks * block, dtype=jnp.int32)
    meas_idx = idx.reshape(n_blocks, block)
    meas_idx_clipped = jnp.minimum(meas_idx, N_total - 1)
    in_range = meas_idx < N_total
    valid_mask = jnp.asarray(valid_mask, dtype=bool).reshape(-1)
    valid_rows = in_range & valid_mask[meas_idx_clipped]
    candidate_tile_ids = result.candidate_tile_ids[meas_idx_clipped]
    candidate_slots = result.candidate_slots[meas_idx_clipped]
    responsibilities = result.responsibilities[meas_idx_clipped] * valid_rows[:, :, None]
    return meas_idx_clipped, candidate_tile_ids, candidate_slots, responsibilities, valid_rows
