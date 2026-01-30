"""
Optional splat rendering: EWA splatting, multi-lobe vMF shading, fBm.

Plan: lidar-camera_splat_fusion_and_bev_ot. EWA: w_i(p) = α_i exp(−½(p−μ_i)' Σ_i^{-1} (p−μ_i));
normalized. Multi-lobe vMF: s = Σ_b π_b exp(κ_b(μ_{n,b}' v − 1)); rgb(v) = c·a_base + c_spec·a_spec·s.
fBm: value noise, O=5 octaves, gain=0.5. Tiling 32×32; fixed cap. No JAX requirement; fixed loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Config (no magic numbers)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SplatRenderingConfig:
    """Configuration for splat rendering."""

    tile_size: int = 32  # Tiling 32×32 (fixed cap)
    fbm_octaves: int = 5  # O=5 octaves for fBm
    fbm_gain: float = 0.5
    opacity_gamma: float = 1.0  # α ∝ σ(γ tr(Λ)); from plan
    logdet0: float = 0.0
    eps: float = 1e-12


# -----------------------------------------------------------------------------
# EWA splatting: w_i(p) = α_i exp(−½(p−μ_i)' Σ_i^{-1} (p−μ_i))
# -----------------------------------------------------------------------------


def ewa_splat_weight(
    p: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
    alpha: float,
) -> float:
    """Weight at pixel p from one splat: α * exp(−½(p−μ)' Σ^{-1} (p−μ))."""
    p = np.asarray(p, dtype=np.float64).ravel()[:2]
    mu = np.asarray(mu, dtype=np.float64).ravel()[:2]
    Sigma_inv = np.asarray(Sigma_inv, dtype=np.float64).reshape(2, 2)
    d = p - mu
    q = float(d @ Sigma_inv @ d)
    return float(alpha) * np.exp(-0.5 * max(q, 0.0))


def ewa_splat_weights_at_point(
    p: np.ndarray,
    means: np.ndarray,
    Sigmas_inv: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Weights (N,) at pixel p from N splats. Not normalized (caller normalizes)."""
    p = np.asarray(p, dtype=np.float64).ravel()[:2]
    N = means.shape[0]
    means = np.asarray(means, dtype=np.float64).reshape(N, 2)
    Sigmas_inv = np.asarray(Sigmas_inv, dtype=np.float64).reshape(N, 2, 2)
    alphas = np.asarray(alphas, dtype=np.float64).ravel()[:N]
    w = np.zeros(N, dtype=np.float64)
    for i in range(N):
        d = p - means[i]
        q = float(d @ Sigmas_inv[i] @ d)
        w[i] = alphas[i] * np.exp(-0.5 * max(q, 0.0))
    return w


# -----------------------------------------------------------------------------
# Multi-lobe vMF shading: s = Σ_b π_b exp(κ_b(μ_{n,b}' v − 1))
# -----------------------------------------------------------------------------


def vmf_shading_multi_lobe(
    v: np.ndarray,
    mu_app: np.ndarray,
    kappa_app: np.ndarray,
    pi_b: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """
    Shading factor s = Σ_b π_b exp(κ_b(μ_{n,b}' v − 1)).
    v: view direction (3,); mu_app (B, 3) lobes; kappa_app (B,); pi_b (B,) weights, default uniform.
    """
    v = np.asarray(v, dtype=np.float64).ravel()[:3]
    v = v / (np.linalg.norm(v) + eps)
    B = mu_app.shape[0]
    mu_app = np.asarray(mu_app, dtype=np.float64).reshape(B, 3)
    kappa_app = np.asarray(kappa_app, dtype=np.float64).ravel()[:B]
    for b in range(B):
        mu_app[b] = mu_app[b] / (np.linalg.norm(mu_app[b]) + eps)
    if pi_b is None:
        pi_b = np.ones(B, dtype=np.float64) / B
    else:
        pi_b = np.asarray(pi_b, dtype=np.float64).ravel()[:B]
        pi_b = pi_b / (np.sum(pi_b) + eps)
    s = 0.0
    for b in range(B):
        s += pi_b[b] * np.exp(kappa_app[b] * (float(np.dot(mu_app[b], v)) - 1.0))
    return float(s)


# -----------------------------------------------------------------------------
# fBm: value noise, O octaves, gain (fixed; no hidden iteration)
# -----------------------------------------------------------------------------


def _hash_float(h: int) -> float:
    """Deterministic hash to [0,1). No global RNG."""
    h = (h * 1103515245 + 12345) & 0x7FFFFFFF
    return float(h) / float(0x7FFFFFFF + 1)


def _value_noise_2d(x: float, y: float, seed: int = 0) -> float:
    """Deterministic value noise at (x,y). Hash-based; bilinear interpolation."""
    ix, iy = int(np.floor(x)), int(np.floor(y))
    fx = x - np.floor(x)
    fy = y - np.floor(y)
    fx = max(0.0, min(1.0, fx))
    fy = max(0.0, min(1.0, fy))
    v00 = _hash_float(((seed * 31 + ix) * 31 + iy) & 0x7FFFFFFF)
    v10 = _hash_float(((seed * 31 + ix + 1) * 31 + iy) & 0x7FFFFFFF)
    v01 = _hash_float(((seed * 31 + ix) * 31 + iy + 1) & 0x7FFFFFFF)
    v11 = _hash_float(((seed * 31 + ix + 1) * 31 + iy + 1) & 0x7FFFFFFF)
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    v0 = v00 * (1 - sx) + v10 * sx
    v1 = v01 * (1 - sx) + v11 * sx
    return v0 * (1 - sy) + v1 * sy


def fbm_value_noise(
    x: float,
    y: float,
    octaves: int = 5,
    gain: float = 0.5,
    seed: int = 0,
) -> float:
    """fBm value noise at (x,y). O octaves, gain. Fixed-cost."""
    x, y = float(x), float(y)
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0
    for _ in range(octaves):
        value += amplitude * _value_noise_2d(x * frequency, y * frequency, seed)
        max_amplitude += amplitude
        amplitude *= gain
        frequency *= 2.0
    return value / (max_amplitude + 1e-12)


# -----------------------------------------------------------------------------
# Opacity from precision: α ∝ σ(γ (logdet0 − logdet)) (from plan)
# -----------------------------------------------------------------------------


def opacity_from_logdet(
    logdet_cov: float,
    gamma: float = 1.0,
    logdet0: float = 0.0,
) -> float:
    """α = 1/(1+exp(-gamma*(logdet0 - logdet))). Continuous."""
    return 1.0 / (1.0 + np.exp(-gamma * (logdet0 - logdet_cov)))


# -----------------------------------------------------------------------------
# Tiled render (fixed tile size; fixed cap on splats per tile for cost)
# -----------------------------------------------------------------------------


def render_tile_ewa(
    tile_origin: Tuple[int, int],
    tile_size: int,
    means: np.ndarray,
    Sigmas_inv: np.ndarray,
    alphas: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """
    Render one tile (tile_size × tile_size) with EWA splats. Returns RGB (tile_size, tile_size, 3).
    means (N, 2), Sigmas_inv (N, 2, 2), alphas (N,), colors (N, 3).
    """
    tile_size = min(max(int(tile_size), 1), 32)
    out = np.zeros((tile_size, tile_size, 3), dtype=np.float64)
    w_sum = np.zeros((tile_size, tile_size), dtype=np.float64)
    N = means.shape[0]
    if N == 0:
        return out
    oy, ox = tile_origin[0], tile_origin[1]
    for py in range(tile_size):
        for px in range(tile_size):
            p = np.array([ox + px + 0.5, oy + py + 0.5], dtype=np.float64)
            w = ewa_splat_weights_at_point(p, means, Sigmas_inv, alphas)
            total = np.sum(w) + 1e-12
            for i in range(N):
                out[py, px, :] += w[i] * colors[i, :3]
            w_sum[py, px] = total
    for py in range(tile_size):
        for px in range(tile_size):
            if w_sum[py, px] > 1e-12:
                out[py, px, :] /= w_sum[py, px]
    return np.clip(out, 0.0, 1.0)
