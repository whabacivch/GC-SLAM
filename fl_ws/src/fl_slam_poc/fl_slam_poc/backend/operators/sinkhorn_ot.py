"""
Sinkhorn OT for BEV splat association: fixed-K iterations, cost W2² + β H²_vMF.

Plan: lidar-camera_splat_fusion_and_bev_ot. Part C. No external OT library.
Cost c_ij = d_pos² + β d_dir²; W2² between 2D Gaussians; Hellinger² vMF.
Fixed number of Sinkhorn steps (no convergence check). Returns (π, CertBundle, ExpectedEffect).
"""

# FUTURE/EXPERIMENTAL:
# This module is intentionally not wired into the runtime pipeline. It is preserved
# as a future BEV15 view-layer association scaffold. Do not import from core ops.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from fl_slam_poc.common import constants
from fl_slam_poc.common.certificates import CertBundle, ExpectedEffect


# -----------------------------------------------------------------------------
# Config (no magic numbers)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SinkhornOTConfig:
    """Configuration for Sinkhorn OT operator."""

    K_SINKHORN: int = 50  # Fixed iteration count (compile-time constant)
    beta: float = 0.5  # Direction weight: cost = W2² + beta * H²_vMF
    epsilon: float = 0.1  # Entropy regularization (m² scale for position)
    # Unbalanced: τ_a, τ_b KL relaxation on marginals (continuous; no threshold)
    # τ_a=τ_b=0 → balanced; τ_a, τ_b > 0 → allow mass creation/destruction (partial overlap).
    tau_a: float = 0.5
    tau_b: float = 0.5
    # SPD guardrails for W2² (not gating): symmetrize + eigen clamp
    w2_eig_min: float = 1e-12
    # Cost normalization (numerical conditioning; not gating): τ and ε interpretable across cost scales
    cost_subtract_row_min: bool = True  # C ← C - min_j C_ij per row (stabilizes exp(-C/ε))
    cost_scale_by_median: bool = False  # C ← C / (median(C)+ε) so ε is interpretable


# -----------------------------------------------------------------------------
# vMF Hellinger² (self-contained for backend; no frontend dependency)
# -----------------------------------------------------------------------------

_VMF_LOG_4PI = math.log(4.0 * math.pi)


def _log_sinh_stable(k: float, eps: float = 1e-12) -> float:
    k = max(float(k), eps)
    if k > 20.0:
        return k - math.log(2.0) + math.log1p(-math.exp(-2.0 * k))
    if k >= 1e-2:
        return math.log(math.sinh(k))
    return math.log(k + (k ** 3) / 6.0)


def _A_vmf(k: float, eps: float = 1e-12) -> float:
    k = max(float(k), eps)
    return _VMF_LOG_4PI + _log_sinh_stable(k, eps) - math.log(k)


def _hellinger2_vmf(
    mu1: np.ndarray,
    k1: float,
    mu2: np.ndarray,
    k2: float,
    eps: float = 1e-12,
) -> float:
    """Squared Hellinger between two vMF on S^2."""
    eta1 = k1 * np.asarray(mu1, dtype=np.float64).ravel()[:3]
    eta2 = k2 * np.asarray(mu2, dtype=np.float64).ravel()[:3]
    eta_sum = eta1 + eta2
    km = 0.5 * float(np.linalg.norm(eta_sum))
    km = max(km, eps)
    k1 = max(float(k1), eps)
    k2 = max(float(k2), eps)
    bc = math.exp(_A_vmf(km, eps) - 0.5 * (_A_vmf(k1, eps) + _A_vmf(k2, eps)))
    return max(0.0, 1.0 - bc)


# -----------------------------------------------------------------------------
# W2² between 2D Gaussians (closed form)
# -----------------------------------------------------------------------------


def _matrix_sqrt_2d(S: np.ndarray, eig_min: float = 1e-12) -> np.ndarray:
    """Symmetric 2x2 SPD: symmetrize, eigen clamp, return S^{1/2} via eigendecomposition."""
    S = np.asarray(S, dtype=np.float64).reshape(2, 2)
    S = 0.5 * (S + S.T)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, eig_min)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def w2_sq_2d(
    m1: np.ndarray,
    S1: np.ndarray,
    m2: np.ndarray,
    S2: np.ndarray,
    eig_min: float = 1e-12,
) -> float:
    """
    Squared 2D Wasserstein-2: W2²(N(m1,S1), N(m2,S2)) = ||m1-m2||² + tr(S1+S2-2(S2^{1/2}S1S2^{1/2})^{1/2}).
    SPD-safe: symmetrize S1,S2 and clamp eigenvalues to eig_min (domain projection, not gating).
    """
    m1 = np.asarray(m1, dtype=np.float64).ravel()[:2]
    m2 = np.asarray(m2, dtype=np.float64).ravel()[:2]
    S1 = np.asarray(S1, dtype=np.float64).reshape(2, 2)
    S2 = np.asarray(S2, dtype=np.float64).reshape(2, 2)
    S1 = 0.5 * (S1 + S1.T) + eig_min * np.eye(2)
    S2 = 0.5 * (S2 + S2.T) + eig_min * np.eye(2)
    dmu = m1 - m2
    term_mean = float(np.dot(dmu, dmu))
    sqrt_S2 = _matrix_sqrt_2d(S2, eig_min)
    M = sqrt_S2 @ S1 @ sqrt_S2
    M = 0.5 * (M + M.T) + eig_min * np.eye(2)
    sqrt_M = _matrix_sqrt_2d(M, eig_min)
    cross = 2.0 * float(np.trace(sqrt_M))
    return term_mean + float(np.trace(S1)) + float(np.trace(S2)) - cross


# -----------------------------------------------------------------------------
# Cost matrix and Sinkhorn
# -----------------------------------------------------------------------------


def cost_matrix_bev(
    mu_cam: np.ndarray,
    Sigma_cam: np.ndarray,
    mu_n_cam: Optional[np.ndarray],
    kappa_cam: np.ndarray,
    mu_lidar: np.ndarray,
    Sigma_lidar: np.ndarray,
    mu_n_lidar: Optional[np.ndarray],
    kappa_lidar: np.ndarray,
    beta: float = 0.5,
    eig_min: float = 1e-12,
) -> np.ndarray:
    """
    Cost matrix C[i,j] = W2²(cam_i, lidar_j) + beta * H²_vMF(cam_i, lidar_j).
    mu_cam (N,2), Sigma_cam (N,2,2), kappa_cam (N,); mu_lidar (M,2), Sigma_lidar (M,2,2), kappa_lidar (M,).
    eig_min: SPD clamp for W2² (domain projection, not gating).
    """
    N = mu_cam.shape[0]
    M = mu_lidar.shape[0]
    kappa_cam = np.asarray(kappa_cam, dtype=np.float64).ravel()
    kappa_lidar = np.asarray(kappa_lidar, dtype=np.float64).ravel()
    if kappa_cam.size != N:
        kappa_cam = np.full(N, 1.0, dtype=np.float64)
    if kappa_lidar.size != M:
        kappa_lidar = np.full(M, 1.0, dtype=np.float64)
    C = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            d_pos = w2_sq_2d(
                mu_cam[i], Sigma_cam[i],
                mu_lidar[j], Sigma_lidar[j],
                eig_min=eig_min,
            )
            d_dir = 0.0
            if (
                mu_n_cam is not None
                and mu_n_lidar is not None
                and kappa_cam[i] > 0
                and kappa_lidar[j] > 0
            ):
                d_dir = _hellinger2_vmf(
                    mu_n_cam[i], float(kappa_cam[i]),
                    mu_n_lidar[j], float(kappa_lidar[j]),
                )
            C[i, j] = d_pos + beta * d_dir
    return C


def sinkhorn_balanced_fixed_k(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    K: int,
) -> np.ndarray:
    """
    Deprecated; use unbalanced only (sinkhorn_unbalanced_fixed_k).
    Balanced Sinkhorn: fixed K iterations. K is compile-time constant; no convergence check.
    min_π <π,C> + ε KL(π | a⊗b); row marginals a, column marginals b.
    Returns coupling π (N,M).
    """
    C = np.asarray(C, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    N, M = C.shape
    # Kernel K_ij = exp(-C_ij/epsilon)
    K_mat = np.exp(-C / max(epsilon, 1e-12))
    u = np.ones(N, dtype=np.float64)
    v = np.ones(M, dtype=np.float64)
    for _ in range(K):
        u = a / (K_mat @ v)
        v = b / (K_mat.T @ u)
    pi = u.reshape(-1, 1) * K_mat * v.reshape(1, -1)
    return pi


def sinkhorn_unbalanced_fixed_k(
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
    eps = max(epsilon, 1e-12)
    K_mat = np.exp(-C / eps)
    u = np.ones(N, dtype=np.float64)
    v = np.ones(M, dtype=np.float64)
    # Powers for unbalanced: 1/(1+τ/ε); when τ=0 this is 1 (balanced update)
    pow_a = 1.0 / (1.0 + tau_a / eps)
    pow_b = 1.0 / (1.0 + tau_b / eps)
    for _ in range(K):
        Kv = K_mat @ v
        Kv = np.maximum(Kv, 1e-300)
        u = np.power(a / Kv, pow_a)
        Ktu = K_mat.T @ u
        Ktu = np.maximum(Ktu, 1e-300)
        v = np.power(b / Ktu, pow_b)
    pi = u.reshape(-1, 1) * K_mat * v.reshape(1, -1)
    return pi


# -----------------------------------------------------------------------------
# Operator: (result, CertBundle, ExpectedEffect)
# -----------------------------------------------------------------------------


@dataclass
class SinkhornOTResult:
    """Result of Sinkhorn OT operator."""

    pi: np.ndarray  # (N, M) coupling
    cost_matrix: np.ndarray  # (N, M) for audit


def sinkhorn_ot_bev(
    mu_cam: np.ndarray,
    Sigma_cam: np.ndarray,
    mu_n_cam: Optional[np.ndarray],
    kappa_cam: np.ndarray,
    mu_lidar: np.ndarray,
    Sigma_lidar: np.ndarray,
    mu_n_lidar: Optional[np.ndarray],
    kappa_lidar: np.ndarray,
    config: SinkhornOTConfig,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "bev_ot",
) -> Tuple[SinkhornOTResult, CertBundle, ExpectedEffect]:
    """
    Fixed-K Sinkhorn OT for BEV camera vs LiDAR splats. Returns (π, CertBundle, ExpectedEffect).

    No hidden iteration: exactly K_SINKHORN steps. Certificate records
    approximation_triggers=["sinkhorn_fixed_iter"].
    """
    N = mu_cam.shape[0]
    M = mu_lidar.shape[0]
    if N == 0 or M == 0:
        pi = np.zeros((N, M), dtype=np.float64)
        result = SinkhornOTResult(pi=pi, cost_matrix=np.zeros((N, M)))
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(objective_name="sinkhorn_ot", predicted=0.0, realized=0.0)
        return result, cert, effect

    C = cost_matrix_bev(
        mu_cam, Sigma_cam, mu_n_cam, kappa_cam,
        mu_lidar, Sigma_lidar, mu_n_lidar, kappa_lidar,
        beta=config.beta,
        eig_min=config.w2_eig_min,
    )
    # Cost normalization (numerical conditioning; not gating)
    if config.cost_subtract_row_min:
        C = C - np.min(C, axis=1, keepdims=True)
    if config.cost_scale_by_median:
        C_flat = C[np.isfinite(C)]
        med = float(np.median(C_flat)) if C_flat.size > 0 else 1.0
        C = C / (med + 1e-12)
    a = np.ones(N, dtype=np.float64) / N
    b = np.ones(M, dtype=np.float64) / M
    use_unbalanced = config.tau_a > 0.0 or config.tau_b > 0.0
    if use_unbalanced:
        pi = sinkhorn_unbalanced_fixed_k(
            C, a, b, config.epsilon, config.tau_a, config.tau_b, config.K_SINKHORN,
        )
        triggers = ["sinkhorn_fixed_iter", "sinkhorn_unbalanced_kl_relax"]
    else:
        pi = sinkhorn_balanced_fixed_k(
            C, a, b, config.epsilon, config.K_SINKHORN,
        )
        triggers = ["sinkhorn_fixed_iter"]
    result = SinkhornOTResult(pi=pi, cost_matrix=C)
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=triggers,
        frobenius_applied=False,
    )
    effect = ExpectedEffect(
        objective_name="sinkhorn_ot",
        predicted=float(np.sum(pi * C)),
        realized=float(np.sum(pi * C)),
    )
    return result, cert, effect
