"""
ARCHIVED: WahbaSVD operator — replaced by Matrix Fisher rotation evidence.

Pipeline uses matrix_fisher_evidence.matrix_fisher_rotation_evidence only.
This file is kept for reference; not importable by installed entrypoints.

Original: fl_slam_poc/backend/operators/wahba.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    SupportCert,
    MismatchCert,
)


@dataclass
class WahbaResult:
    """Result of WahbaSVD operator."""
    R_hat: jnp.ndarray  # (3, 3) optimal rotation
    cost: float  # Wahba cost
    det_sign: float  # Sign of determinant (for reflection handling)


def _wahba_svd_core(
    mu_map: jnp.ndarray,
    mu_scan: jnp.ndarray,
    weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B = jnp.einsum('b,bi,bj->ij', weights, mu_map, mu_scan)
    U, S, Vt = jnp.linalg.svd(B, full_matrices=True)
    det_UVt = jnp.linalg.det(U @ Vt)
    det_sign = jnp.sign(det_UVt)
    diag_correction = jnp.array([1.0, 1.0, det_sign], dtype=jnp.float64)
    R_hat = U @ jnp.diag(diag_correction) @ Vt
    total_weight = jnp.sum(weights)
    cost = total_weight - jnp.trace(R_hat @ B.T)
    return R_hat, cost, det_sign


def wahba_svd(
    mu_map: jnp.ndarray,
    mu_scan: jnp.ndarray,
    weights: jnp.ndarray,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "initial",
) -> Tuple[WahbaResult, CertBundle, ExpectedEffect]:
    mu_map = jnp.asarray(mu_map, dtype=jnp.float64)
    mu_scan = jnp.asarray(mu_scan, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    n_bins = weights.shape[0]
    R_hat, cost_jax, det_sign_jax = _wahba_svd_core(mu_map, mu_scan, weights)
    cost = float(cost_jax)
    det_sign = float(det_sign_jax)
    result = WahbaResult(R_hat=R_hat, cost=cost, det_sign=det_sign)
    total_weight = float(jnp.sum(weights))
    nonzero_bins = float(jnp.sum(weights > constants.GC_EPS_MASS))
    normalized_cost = cost / (total_weight + constants.GC_EPS_MASS)
    cert = CertBundle.create_exact(
        chart_id=chart_id,
        anchor_id=anchor_id,
        support=SupportCert(
            ess_total=total_weight,
            support_frac=nonzero_bins / n_bins,
        ),
        mismatch=MismatchCert(
            nll_per_ess=normalized_cost,
            directional_score=1.0 - normalized_cost,
        ),
    )
    expected_effect = ExpectedEffect(
        objective_name="wahba_cost",
        predicted=cost,
        realized=None,
    )
    return result, cert, expected_effect
