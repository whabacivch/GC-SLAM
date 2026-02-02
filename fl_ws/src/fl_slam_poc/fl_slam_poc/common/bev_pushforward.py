"""
BEV pushforward: exact linear pushforward of 3D Gaussians to 2D BEV.

Linear P ∈ R^{2×3}:
μ_bev = P μ, Σ_bev = P Σ Pᵀ. Oblique P(φ) = [[1,0,0],[0,cos φ, sin φ]].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BEVPushforwardConfig:
    """Configuration for BEV pushforward."""

    # Single-view oblique angle (deg): P(φ) = [[1,0,0],[0,cos φ, sin φ]].
    oblique_phi_deg: float = 10.0

    # Multi-view BEV15: generate N views along a 1D geodesic in the oblique angle parameter.
    # This is an output-side/view-side construct for rendering/association diagnostics.
    n_views: int = 15
    phi_center_deg: float = 10.0
    phi_span_deg: float = 14.0  # total span across views (deg)


def _oblique_P(phi_deg: float) -> np.ndarray:
    """P(φ) ∈ R^{2×3}: [[1,0,0],[0,cos φ, sin φ]]. phi_deg in degrees."""
    phi = np.deg2rad(float(phi_deg))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi), np.sin(phi)],
        ],
        dtype=np.float64,
    )


def pushforward_gaussian_3d_to_2d(
    mu: np.ndarray,
    Sigma: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Push 3D Gaussian to 2D BEV: μ_bev = P μ, Σ_bev = P Σ Pᵀ.

    Args:
        mu: (3,) mean in 3D (e.g. world or body).
        Sigma: (3,3) covariance.
        P: (2,3) linear map (e.g. oblique P(φ) or identity drop (x,y)).

    Returns:
        mu_bev: (2,) mean in BEV.
        Sigma_bev: (2,2) covariance in BEV.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()[:3]
    Sigma = np.asarray(Sigma, dtype=np.float64).reshape(3, 3)
    P = np.asarray(P, dtype=np.float64).reshape(2, 3)
    mu_bev = P @ mu
    Sigma_bev = P @ Sigma @ P.T
    return mu_bev, Sigma_bev


def oblique_P_from_config(config: BEVPushforwardConfig) -> np.ndarray:
    """Build oblique P(φ) from config."""
    return _oblique_P(config.oblique_phi_deg)


def oblique_Ps_bev15(config: BEVPushforwardConfig) -> np.ndarray:
    """
    Build BEV15 projection matrices P_k (N,2,3) by sweeping φ along a 1D geodesic in angle space.

    For 1D parameter φ, the geodesic is linear interpolation. We center at phi_center_deg and span phi_span_deg.
    """
    n = int(config.n_views)
    n = max(1, n)
    center = float(config.phi_center_deg)
    span = float(config.phi_span_deg)
    if n == 1:
        phis = np.array([center], dtype=np.float64)
    else:
        offsets = np.linspace(-0.5, 0.5, n, dtype=np.float64) * span
        phis = center + offsets
    Ps = np.stack([_oblique_P(phi) for phi in phis], axis=0)  # (N,2,3)
    return Ps


def rotate_vmf_eta(R: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    Exact SO(3) pushforward for vMF natural parameters.

    vMF: f(u) ∝ exp(ηᵀ u), η = κ μ. Under rotation R, pushforward is η' = R η.
    This is closed-form and in-family (κ unchanged, μ' = R μ).
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    eta = np.asarray(eta, dtype=np.float64).ravel()[:3]
    return R @ eta


def rotate_vmf_etas(R: np.ndarray, etas: np.ndarray) -> np.ndarray:
    """
    Vectorized SO(3) pushforward for multi-lobe vMF etas: (B,3) -> (B,3).
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    etas = np.asarray(etas, dtype=np.float64).reshape(-1, 3)
    return (R @ etas.T).T


__all__ = [
    "BEVPushforwardConfig",
    "pushforward_gaussian_3d_to_2d",
    "oblique_P_from_config",
    "oblique_Ps_bev15",
    "rotate_vmf_eta",
    "rotate_vmf_etas",
]
