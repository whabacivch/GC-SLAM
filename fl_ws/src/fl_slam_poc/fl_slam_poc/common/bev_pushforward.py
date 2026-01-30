"""
BEV pushforward: 3D Gaussian (and optional vMF) to 2D BEV for OT/fusion.

Plan: lidar-camera_splat_fusion_and_bev_ot. Part B. Linear P ∈ R^{2×3}:
μ_bev = P μ, Σ_bev = P Σ Pᵀ. Oblique P(φ) = [[1,0,0],[0,cos φ, sin φ]]; config oblique_phi_deg.
vMF to map: project direction to horizontal (e.g. azimuth); pushforward to S¹ is approximate (stub).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BEVPushforwardConfig:
    """Configuration for BEV pushforward."""

    # Oblique angle (deg): P(φ) = [[1,0,0],[0,cos φ, sin φ]], φ ≈ 10°
    oblique_phi_deg: float = 10.0


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


def pushforward_vmf_to_s1_stub(
    mu_n: np.ndarray,
    kappa: float,
) -> Tuple[float, float]:
    """
    Stub: push vMF on S² to S¹ (e.g. azimuth). Approximate (moment-match or sample).

    Plan: e_u=(1,0,0), e_v=(0,cos φ, sin φ); d_b = (μ_n·e_u, μ_n·e_v); document
    that pushforward of vMF to S¹ is approximate. Returns (theta_or_eta, kappa_map)
    placeholder; implement moment-match when needed.
    """
    mu_n = np.asarray(mu_n, dtype=np.float64).ravel()[:3]
    # Placeholder: azimuth in [-pi, pi] from (x, y) component
    theta = float(np.arctan2(mu_n[1], mu_n[0]))
    kappa_map = float(kappa) * float(np.sqrt(mu_n[0] ** 2 + mu_n[1] ** 2 + 1e-12))
    return (theta, kappa_map)
