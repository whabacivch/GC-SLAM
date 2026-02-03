"""
PointBudgetResample operator for Geometric Compositional SLAM v2.

Deterministically resample points to enforce N_POINTS_CAP.
All points contribute through mass redistribution.

Reference: docs/GC_SLAM.md Section 5.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    SupportCert,
    InfluenceCert,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PointBudgetResult:
    """Result of PointBudgetResample operator."""
    points: jnp.ndarray  # (N_POINTS_CAP, 3) resampled points; padded rows zero
    timestamps: jnp.ndarray  # (N_POINTS_CAP,) timestamps
    weights: jnp.ndarray  # (N_POINTS_CAP,) adjusted weights; padded rows zero
    ring: jnp.ndarray  # (N_POINTS_CAP,) uint8 (0 if unavailable)
    tag: jnp.ndarray   # (N_POINTS_CAP,) uint8 (0 if unavailable)
    n_input: int
    n_output: int  # actual count selected (≤ N_POINTS_CAP)
    total_mass_in: float
    total_mass_out: float


# =============================================================================
# JIT'd core (fixed N_budget output; wrapper builds PointBudgetResult and cert)
# =============================================================================


@jax.jit(static_argnames=("n_input", "n_points_cap", "stride"))
def _point_budget_resample_core(
    points: jnp.ndarray,
    timestamps: jnp.ndarray,
    weights: jnp.ndarray,
    ring: jnp.ndarray,
    tag: jnp.ndarray,
    n_input: int,
    n_points_cap: int,
    stride: int,
    eps_mass: jnp.ndarray,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
]:
    """
    Deterministic stride-based subsampling; always output (n_points_cap,) arrays (pad unused).
    stride must be concrete (computed in wrapper). Returns (points_out, timestamps_out, ...).
    """
    total_mass_in = jnp.sum(weights)
    indices = jnp.arange(0, n_input, stride)
    n_selected = indices.shape[0]

    points_selected = points[indices]
    points_out = jnp.zeros((n_points_cap, 3), dtype=jnp.float64)
    points_out = points_out.at[:n_selected].set(points_selected)

    timestamps_out = jnp.zeros((n_points_cap,), dtype=jnp.float64)
    timestamps_out = timestamps_out.at[:n_selected].set(timestamps[indices])

    weights_raw = weights[indices]
    total_mass_selected = jnp.sum(weights_raw)
    mass_scale = total_mass_in / (total_mass_selected + eps_mass)
    weights_out = jnp.zeros((n_points_cap,), dtype=jnp.float64)
    weights_out = weights_out.at[:n_selected].set(weights_raw * mass_scale)
    total_mass_out = total_mass_in

    ring_out = jnp.zeros((n_points_cap,), dtype=jnp.uint8)
    ring_out = ring_out.at[:n_selected].set(ring[indices])
    tag_out = jnp.zeros((n_points_cap,), dtype=jnp.uint8)
    tag_out = tag_out.at[:n_selected].set(tag[indices])

    n_input_arr = jnp.array(n_input, dtype=jnp.int32)
    n_selected_arr = jnp.array(n_selected, dtype=jnp.int32)

    weights_normalized = weights_out / (total_mass_out + eps_mass)
    ess = 1.0 / (jnp.sum(weights_normalized ** 2 + eps_mass))

    return (
        points_out,
        timestamps_out,
        weights_out,
        ring_out,
        tag_out,
        n_input_arr,
        n_selected_arr,
        total_mass_in,
        total_mass_out,
        ess,
    )


# =============================================================================
# Main Operator
# =============================================================================


def point_budget_resample(
    points: jnp.ndarray,
    timestamps: jnp.ndarray,
    weights: jnp.ndarray,
    ring: jnp.ndarray | None = None,
    tag: jnp.ndarray | None = None,
    n_points_cap: int = constants.GC_N_POINTS_CAP,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[PointBudgetResult, CertBundle, ExpectedEffect]:
    """
    Deterministically resample points to enforce budget.
    
    Uses deterministic subsampling with mass preservation.
    All operations are continuous and total.
    
    Args:
        points: Input point positions (N, 3)
        timestamps: Per-point timestamps (N,)
        weights: Per-point weights (N,)
        n_points_cap: Maximum number of points (default from constants)
        chart_id: Chart identifier
        anchor_id: Anchor identifier
        
    Returns:
        Tuple of (PointBudgetResult, CertBundle, ExpectedEffect)
        
    Spec ref: Section 5.1
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    timestamps = jnp.asarray(timestamps, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    if ring is None:
        ring = jnp.zeros((points.shape[0],), dtype=jnp.uint8)
    else:
        ring = jnp.asarray(ring, dtype=jnp.uint8).reshape(-1)
    if tag is None:
        tag = jnp.zeros((points.shape[0],), dtype=jnp.uint8)
    else:
        tag = jnp.asarray(tag, dtype=jnp.uint8).reshape(-1)

    n_input = points.shape[0]
    eps_mass = jnp.array(constants.GC_EPS_MASS, dtype=jnp.float64)
    stride = max(1, int(math.ceil(n_input / n_points_cap)))

    (
        points_out,
        timestamps_out,
        weights_out,
        ring_out,
        tag_out,
        n_input_arr,
        n_selected_arr,
        total_mass_in,
        total_mass_out,
        ess,
    ) = _point_budget_resample_core(
        points,
        timestamps,
        weights,
        ring,
        tag,
        n_input=n_input,
        n_points_cap=n_points_cap,
        stride=stride,
        eps_mass=eps_mass,
    )

    n_selected = int(n_selected_arr)
    support_frac = float(
        jnp.minimum(1.0, n_points_cap / (n_input + constants.GC_EPS_MASS))
    )
    # Return fixed-size arrays (N_POINTS_CAP) so downstream JITs see static shapes; padded rows have zero weight.
    result = PointBudgetResult(
        points=points_out,
        timestamps=timestamps_out,
        weights=weights_out,
        ring=ring_out,
        tag=tag_out,
        n_input=int(n_input_arr),
        n_output=n_selected,
        total_mass_in=float(total_mass_in),
        total_mass_out=float(total_mass_out),
    )

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["PointBudgetResample"],
        support=SupportCert(
            ess_total=float(ess),
            support_frac=support_frac,
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=constants.GC_EPS_MASS / (float(total_mass_in) + constants.GC_EPS_MASS),
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    expected_effect = ExpectedEffect(
        objective_name="predicted_ess",
        predicted=float(ess),
        realized=None,
    )

    return result, cert, expected_effect
