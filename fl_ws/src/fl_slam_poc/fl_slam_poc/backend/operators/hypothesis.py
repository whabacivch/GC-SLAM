"""
HypothesisBarycenterProjection operator for Geometric Compositional SLAM v2.

Combines K hypotheses into a single belief for publishing.

Fully vectorized over hypotheses - no Python for-loops in hot path.

Reference: docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md Section 5.15
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import BeliefGaussianInfo, D_Z, CHART_ID_GC_RIGHT_01
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    SupportCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    domain_projection_psd_core,
    spd_cholesky_solve_lifted,
    spd_cholesky_solve_lifted_core,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class HypothesisProjectionResult:
    """Result of HypothesisBarycenterProjection operator."""
    belief_out: BeliefGaussianInfo
    floor_adjustment: float  # Total weight floor adjustment


# =============================================================================
# Vectorized Core (JIT-safe)
# =============================================================================


@jax.jit
def _hypothesis_barycenter_core(
    L_stack: jnp.ndarray,
    h_stack: jnp.ndarray,
    z_lin_stack: jnp.ndarray,
    weights: jnp.ndarray,
    HYP_WEIGHT_FLOOR: float,
    eps_psd: float,
    eps_lift: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vectorized core computation for hypothesis barycenter (JIT-compiled).

    Args:
        L_stack: Stacked information matrices (K, D_Z, D_Z)
        h_stack: Stacked information vectors (K, D_Z)
        z_lin_stack: Stacked linearization points (K, D_Z)
        weights: Hypothesis weights (K,)
        HYP_WEIGHT_FLOOR: Minimum weight floor
        eps_psd: PSD projection epsilon
        eps_lift: Cholesky lift epsilon

    Returns:
        L_out: Fused information matrix (D_Z, D_Z)
        h_out: Fused information vector (D_Z,)
        z_lin_out: Fused linearization point (D_Z,)
        floor_adjustment: Total weight floor adjustment
        weights_normalized: Normalized weights (K,)
        psd_cert_vec: PSD projection certificate vector
        spread_proxy: Spread proxy scalar
    """
    # Step 1: Enforce weight floor (continuous, no branching)
    weights_floored = jnp.maximum(weights, HYP_WEIGHT_FLOOR)
    floor_adjustment = jnp.sum(jnp.abs(weights_floored - weights))

    # Step 2: Renormalize
    weights_normalized = weights_floored / jnp.sum(weights_floored)

    # Step 3: Barycenter in information form (vectorized)
    # L_out = sum_j w_j L_j, h_out = sum_j w_j h_j
    # weights_normalized: (K,), L_stack: (K, D_Z, D_Z)
    L_out_raw = jnp.einsum("k,kij->ij", weights_normalized, L_stack)
    h_out = jnp.einsum("k,ki->i", weights_normalized, h_stack)

    # Weighted mean linearization point (vectorized)
    z_lin_out = jnp.einsum("k,ki->i", weights_normalized, z_lin_stack)

    # Step 4: Apply DomainProjectionPSD (always)
    L_out, psd_cert_vec = domain_projection_psd_core(L_out_raw, eps_psd)

    # Compute spread proxy for expected effect (vectorized)
    # Solve for means: mu_j = L_j^{-1} h_j
    def solve_one(L_j, h_j):
        mu_j, _ = spd_cholesky_solve_lifted_core(L_j, h_j, eps_lift)
        return mu_j

    means_stack = jax.vmap(solve_one)(L_stack, h_stack)  # (K, D_Z)

    # Weighted mean of means
    mean_of_means = jnp.einsum("k,ki->i", weights_normalized, means_stack)  # (D_Z,)

    # Weighted variance: sum_j w_j ||mu_j - mean||^2
    deltas = means_stack - mean_of_means[None, :]  # (K, D_Z)
    delta_sq = jnp.sum(deltas * deltas, axis=1)  # (K,)
    spread_proxy = jnp.sum(weights_normalized * delta_sq)

    return L_out, h_out, z_lin_out, floor_adjustment, weights_normalized, psd_cert_vec, spread_proxy


# =============================================================================
# Main Operator (Wrapper)
# =============================================================================


def hypothesis_barycenter_projection(
    hypotheses: List[BeliefGaussianInfo],
    weights: jnp.ndarray,
    K_HYP: int = constants.GC_K_HYP,
    HYP_WEIGHT_FLOOR: float = constants.GC_HYP_WEIGHT_FLOOR,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
) -> Tuple[HypothesisProjectionResult, CertBundle, ExpectedEffect]:
    """
    Combine K hypotheses into a single belief via barycenter projection.

    Always:
    1. Enforce weight floor continuously
    2. Renormalize weights
    3. Barycenter in information form
    4. Apply DomainProjectionPSD

    Fully vectorized over hypotheses (no Python for-loops).

    Args:
        hypotheses: List of K_HYP beliefs
        weights: Hypothesis weights (K_HYP,)
        K_HYP: Number of hypotheses (default from constants)
        HYP_WEIGHT_FLOOR: Minimum weight (default from constants)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon

    Returns:
        Tuple of (HypothesisProjectionResult, CertBundle, ExpectedEffect)

    Spec ref: Section 5.15
    """
    weights = jnp.asarray(weights, dtype=jnp.float64)

    if len(hypotheses) != K_HYP:
        raise ValueError(f"Expected {K_HYP} hypotheses, got {len(hypotheses)}")
    if weights.shape != (K_HYP,):
        raise ValueError(f"Expected weights shape ({K_HYP},), got {weights.shape}")

    # Stack hypothesis data into batched arrays (done once, outside JIT)
    L_stack = jnp.stack([h.L for h in hypotheses], axis=0)  # (K, D_Z, D_Z)
    h_stack = jnp.stack([h.h for h in hypotheses], axis=0)  # (K, D_Z)
    z_lin_stack = jnp.stack([h.z_lin for h in hypotheses], axis=0)  # (K, D_Z)

    # Call vectorized core
    L_out, h_out, z_lin_out, floor_adjustment, weights_normalized, psd_cert_vec, spread_proxy = (
        _hypothesis_barycenter_core(
            L_stack,
            h_stack,
            z_lin_stack,
            weights,
            HYP_WEIGHT_FLOOR,
            eps_psd,
            eps_lift,
        )
    )

    # Use the first hypothesis as template for anchor/chart info
    template = hypotheses[0]

    # Extract PSD certificate values
    psd_projection_delta = float(psd_cert_vec[0])
    eig_min = float(psd_cert_vec[2])
    eig_max = float(psd_cert_vec[3])
    cond = float(psd_cert_vec[4])
    near_null_count = int(psd_cert_vec[5])

    # Build output belief
    cert_out = CertBundle.create_approx(
        chart_id=CHART_ID_GC_RIGHT_01,
        anchor_id=template.anchor_id,
        triggers=["HypothesisProjection", "I-projection-info-barycenter"],
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=near_null_count,
        ),
        support=SupportCert(
            ess_total=float(1.0 / jnp.sum(weights_normalized**2)),  # ESS
            support_frac=float(jnp.sum(weights_normalized > HYP_WEIGHT_FLOOR) / K_HYP),
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=psd_projection_delta,
            mass_epsilon_ratio=float(floor_adjustment) / K_HYP,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    belief_out = BeliefGaussianInfo(
        chart_id=CHART_ID_GC_RIGHT_01,
        anchor_id=template.anchor_id,
        X_anchor=template.X_anchor,
        stamp_sec=template.stamp_sec,
        z_lin=z_lin_out,
        L=L_out,
        h=h_out,
        cert=cert_out,
    )

    # Build result
    result = HypothesisProjectionResult(
        belief_out=belief_out,
        floor_adjustment=float(floor_adjustment),
    )

    expected_effect = ExpectedEffect(
        objective_name="predicted_projection_spread_proxy",
        predicted=float(spread_proxy),
        realized=None,
    )

    return result, cert_out, expected_effect
