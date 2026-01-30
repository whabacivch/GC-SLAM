"""
OT-weighted fusion: w_ij from coupling π; Λ/θ (Gaussian) and η (vMF) fusion.

Plan: lidar-camera_splat_fusion_and_bev_ot. Part C. w_ij = π_ij / (Σ_j π_ij + ε)
(continuous; no threshold). Λ_i(ℓ) = Σ_j w_ij Λ_j, θ_i(ℓ) = Σ_j w_ij θ_j;
Λ_i(f) = Λ_i(c) + γ_i Λ_i(ℓ), θ_i(f) = θ_i(c) + γ_i θ_i(ℓ). vMF: η_i(f) = η_i(c) + γ_i Σ_j w_ij η_j.
All parameters from config (gamma, epsilon).

Recommended usage: use confidence_tempered_gamma(π, γ, α, m0) and pass the result as
gamma_per_row to weighted_fusion_gaussian_bev / weighted_fusion_vmf_bev so that
γ_i = γ·σ(α(m_i−m0)) (row-mass confidence). Prevents tiny accidental couplings from
noticeable fusion while staying continuous (no thresholds).

Wishart and temporal_smooth_lambda: apply on Λ in a consistent coordinate chart
(e.g. BEV map frame or world frame), not a frame that changes every timestep.
Temporal smoothing requires stable feature IDs (e.g. map primitives); if applied
on per-frame features without re-association, it blurs unrelated covariances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Config (no magic numbers)
# -----------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Continuous sigmoid 1/(1+exp(-x)); no gate."""
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    t = np.exp(x)
    return t / (1.0 + t)


@dataclass(frozen=True)
class OTFusionConfig:
    """Configuration for OT-weighted fusion."""

    gamma: float = 1.0  # Modality trust: fuse Λ_i(f) = Λ_i(c) + gamma_i * Λ_i(ℓ)
    epsilon: float = 1e-12  # Denominator regularization for w_ij = π_ij / (Σ_j π_ij + ε)
    # Continuous confidence tempering: γ_i = γ · σ(α(m_i − m0)); m_i = Σ_j π_ij (no threshold)
    confidence_alpha: float = 10.0  # steepness of sigmoid
    confidence_m0: float = 0.2  # center of sigmoid (row-mass scale)
    # Post-fusion: Wishart regularization and temporal smoothing (continuous; no gate)
    wishart_nu: float = 5.0  # Λ_reg = Λ + nu * Psi^{-1}, Psi = psi_scale * I
    wishart_psi_scale: float = 0.1  # s in Psi = s*I
    temporal_alpha: float = 0.3  # Λ_t ← Λ_t + alpha * Λ_{t-1}; caller holds Λ_prev


# -----------------------------------------------------------------------------
# Weights from coupling (continuous; no threshold)
# -----------------------------------------------------------------------------


def coupling_to_weights(pi: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    w_ij = π_ij / (Σ_j π_ij + ε). Continuous; no gate on Σ_j π_ij.
    pi (N, M) -> w (N, M).
    """
    pi = np.asarray(pi, dtype=np.float64)
    row_sum = np.sum(pi, axis=1, keepdims=True) + epsilon
    return pi / row_sum


def confidence_tempered_gamma(
    pi: np.ndarray,
    gamma: float,
    alpha: float,
    m0: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Row-mass confidence tempering: m_i = Σ_j π_ij, γ_i = γ · σ(α(m_i − m0)).
    Continuous; no threshold. Prevents tiny accidental couplings from noticeable fusion.
    pi (N, M) -> gamma_i (N,).
    """
    pi = np.asarray(pi, dtype=np.float64)
    m_i = np.sum(pi, axis=1) + eps
    gamma_i = gamma * np.array([_sigmoid(alpha * (float(m) - m0)) for m in m_i], dtype=np.float64)
    return gamma_i


# -----------------------------------------------------------------------------
# Gaussian (2D BEV) fusion: Λ, θ natural params
# -----------------------------------------------------------------------------


def weighted_fusion_gaussian_bev(
    Lambda_cam: np.ndarray,
    theta_cam: np.ndarray,
    Lambda_lidar: np.ndarray,
    theta_lidar: np.ndarray,
    w: np.ndarray,
    gamma: float,
    gamma_per_row: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each camera index i: Λ_i(ℓ) = Σ_j w_ij Λ_j, θ_i(ℓ) = Σ_j w_ij θ_j;
    Λ_i(f) = Λ_i(c) + γ_i Λ_i(ℓ), θ_i(f) = θ_i(c) + γ_i θ_i(ℓ).
    If gamma_per_row (N,) is provided, use γ_i per row; else use scalar gamma.

    Lambda_cam (N, 2, 2), theta_cam (N, 2); Lambda_lidar (M, 2, 2), theta_lidar (M, 2);
    w (N, M). Returns Lambda_f (N, 2, 2), theta_f (N, 2).
    """
    N = Lambda_cam.shape[0]
    Lambda_cam = np.asarray(Lambda_cam, dtype=np.float64)
    theta_cam = np.asarray(theta_cam, dtype=np.float64)
    Lambda_lidar = np.asarray(Lambda_lidar, dtype=np.float64)
    theta_lidar = np.asarray(theta_lidar, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    Lambda_ell = np.einsum("ij,jkl->ikl", w, Lambda_lidar)
    theta_ell = np.einsum("ij,jk->ik", w, theta_lidar)
    if gamma_per_row is not None:
        gamma_per_row = np.asarray(gamma_per_row, dtype=np.float64).ravel()[:N]
        if gamma_per_row.size == N:
            Lambda_f = Lambda_cam + (gamma_per_row[:, np.newaxis, np.newaxis] * Lambda_ell)
            theta_f = theta_cam + (gamma_per_row[:, np.newaxis] * theta_ell)
        else:
            Lambda_f = Lambda_cam + gamma * Lambda_ell
            theta_f = theta_cam + gamma * theta_ell
    else:
        Lambda_f = Lambda_cam + gamma * Lambda_ell
        theta_f = theta_cam + gamma * theta_ell
    return Lambda_f, theta_f


# -----------------------------------------------------------------------------
# vMF fusion: η = κ μ; η_i(f) = η_i(c) + γ Σ_j w_ij η_j; then κ = ‖η‖, μ = η/‖η‖
# -----------------------------------------------------------------------------


def weighted_fusion_vmf_bev(
    eta_cam: np.ndarray,
    eta_lidar: np.ndarray,
    w: np.ndarray,
    gamma: float,
    gamma_per_row: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    η_i(f) = η_i(c) + γ_i Σ_j w_ij η_j; κ_i = ‖η_i(f)‖, μ_i = η_i(f) / ‖η_i(f)‖.
    If gamma_per_row (N,) is provided, use γ_i per row; else use scalar gamma.

    eta_cam (N, 3), eta_lidar (M, 3); w (N, M). Returns mu_f (N, 3), kappa_f (N,).
    """
    N = eta_cam.shape[0]
    eta_cam = np.asarray(eta_cam, dtype=np.float64)
    eta_lidar = np.asarray(eta_lidar, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    eta_ell = np.einsum("ij,jk->ik", w, eta_lidar)
    if gamma_per_row is not None:
        gamma_per_row = np.asarray(gamma_per_row, dtype=np.float64).ravel()[:N]
        if gamma_per_row.size == N:
            eta_f = eta_cam + (gamma_per_row[:, np.newaxis] * eta_ell)
        else:
            eta_f = eta_cam + gamma * eta_ell
    else:
        eta_f = eta_cam + gamma * eta_ell
    kappa_f = np.linalg.norm(eta_f, axis=1)
    kappa_f = np.maximum(kappa_f, eps)
    mu_f = eta_f / kappa_f[:, np.newaxis]
    return mu_f, kappa_f


# -----------------------------------------------------------------------------
# Natural params to mean/cov (2D)
# -----------------------------------------------------------------------------


def natural_to_mean_cov_2d(Lambda: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """μ = Λ^{-1} θ, Σ = Λ^{-1}. Lambda (2,2), theta (2,). Returns mu (2,), Sigma (2,2)."""
    Lambda = np.asarray(Lambda, dtype=np.float64).reshape(2, 2)
    theta = np.asarray(theta, dtype=np.float64).ravel()[:2]
    Lambda = Lambda + 1e-12 * np.eye(2)
    Sigma = np.linalg.inv(Lambda)
    mu = Sigma @ theta
    return mu, Sigma


# -----------------------------------------------------------------------------
# Post-fusion: Wishart regularization (Λ_reg = Λ + ν Ψ^{-1}, Ψ = sI)
# -----------------------------------------------------------------------------


def wishart_regularize_2d(
    Lambda: np.ndarray,
    nu: float,
    psi_scale: float,
    eig_min: float = 1e-12,
) -> np.ndarray:
    """
    Λ_reg = Λ + nu * Psi^{-1}, Psi = psi_scale * I. Ensures Λ_reg is PD.
    Continuous; no gate. Returns (2,2) Lambda_reg.
    Apply in a consistent coordinate chart (e.g. BEV/map frame), not camera frame.
    """
    Lambda = np.asarray(Lambda, dtype=np.float64).reshape(2, 2)
    Lambda = 0.5 * (Lambda + Lambda.T)
    s = max(float(psi_scale), 1e-12)
    Lambda_reg = Lambda + (nu / s) * np.eye(2)
    eigvals = np.linalg.eigvalsh(Lambda_reg)
    if eigvals.min() < eig_min:
        Lambda_reg = Lambda_reg + (eig_min - eigvals.min()) * np.eye(2)
    return Lambda_reg


# -----------------------------------------------------------------------------
# Temporal smoothing: Λ_t ← Λ_t + alpha * Λ_{t-1} (caller holds Λ_prev)
# -----------------------------------------------------------------------------


def temporal_smooth_lambda(
    Lambda_t: np.ndarray,
    Lambda_prev: Optional[np.ndarray],
    alpha: float,
    eig_min: float = 1e-12,
) -> np.ndarray:
    """
    Λ_smoothed = Λ_t + alpha * Λ_prev. Continuous; no gate.
    If Lambda_prev is None, returns Lambda_t (no smoothing). Returns (2,2) or (N,2,2).
    Apply in a consistent frame (e.g. BEV/map). Requires stable feature IDs across
    frames (e.g. map primitives); do not use on per-frame features without re-association.
    """
    Lambda_t = np.asarray(Lambda_t, dtype=np.float64)
    if Lambda_prev is None:
        return Lambda_t
    Lambda_prev = np.asarray(Lambda_prev, dtype=np.float64)
    alpha = max(0.0, min(1.0, float(alpha)))
    Lambda_smoothed = Lambda_t + alpha * Lambda_prev
    if Lambda_smoothed.ndim == 2:
        Lambda_smoothed = 0.5 * (Lambda_smoothed + Lambda_smoothed.T) + eig_min * np.eye(2)
    else:
        for i in range(Lambda_smoothed.shape[0]):
            Lambda_smoothed[i] = 0.5 * (Lambda_smoothed[i] + Lambda_smoothed[i].T) + eig_min * np.eye(2)
    return Lambda_smoothed
