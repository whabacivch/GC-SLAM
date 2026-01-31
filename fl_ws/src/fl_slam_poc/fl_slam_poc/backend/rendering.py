"""
Splat rendering: EWA splatting, multi-lobe vMF shading, fBm.

Backend output: given state/map (splats) and view, produce image. Not sensor input.
Plan: lidar-camera_splat_fusion_and_bev_ot. EWA: w_i(p) = α_i exp(−½(p−μ_i)' Σ_i^{-1} (p−μ_i));
normalized. Multi-lobe vMF: s = Σ_b π_b exp(κ_b(μ_{n,b}' v − 1)); energy-normalized by (1+κ̄).
Intensity (reflectivity) can modulate κ. fBm: value noise in world space for view-stable texture.

By-construction upgrades: (1) Tile binning with fixed cap (splat_indices_for_tile) for O(pixels×cap).
(2) EWA log-domain clipping (ewa_log_clip) to avoid underflow. (3) World-space fBm (fbm_at_splat_positions,
means_world_xy in render_tile_ewa) so texture does not shimmer. (4) vMF energy normalization.
(5) Opacity soft floor (alpha_min). No gates; closed-form; fixed loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# -----------------------------------------------------------------------------
# Config (no magic numbers)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SplatRenderingConfig:
    """Configuration for splat rendering."""

    tile_size: int = 32  # Tiling 32×32 (fixed cap)
    max_splats_per_tile: int = 64  # Tile binning: cap splats per tile for O(pixels × cap) cost
    fbm_octaves: int = 5  # O=5 octaves for fBm
    fbm_gain: float = 0.5
    opacity_gamma: float = 1.0  # α ∝ σ(γ tr(Λ)); from plan
    logdet0: float = 0.0
    eps: float = 1e-12
    ewa_log_clip: float = 25.0  # Clip -0.5*q to [-L, 0] to avoid underflow for tight covariances
    alpha_min: float = 0.02  # Soft floor so very uncertain splats don't vanish completely
    # vMF κ modulation by intensity (reflectivity): higher intensity → shinier → higher κ
    vmf_intensity_scale: float = 0.5  # 0 = off; effective κ = κ_base * (1 + scale * intensity_norm)
    vmf_intensity_max: float = 255.0  # normalize uint8 intensity to [0,1]
    vmf_kappa_max: float = 100.0  # cap modulated κ for stability


# -----------------------------------------------------------------------------
# EWA splatting: w_i(p) = α_i exp(−½(p−μ_i)' Σ_i^{-1} (p−μ_i))
# -----------------------------------------------------------------------------


def ewa_splat_weight(
    p: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
    alpha: float,
    log_clip: float = 25.0,
) -> float:
    """Weight at pixel p from one splat: α * exp(−½(p−μ)' Σ^{-1} (p−μ)). Log-domain clipped for stability."""
    p = np.asarray(p, dtype=np.float64).ravel()[:2]
    mu = np.asarray(mu, dtype=np.float64).ravel()[:2]
    Sigma_inv = np.asarray(Sigma_inv, dtype=np.float64).reshape(2, 2)
    d = p - mu
    q = float(d @ Sigma_inv @ d)
    exp_arg = np.clip(-0.5 * max(q, 0.0), -log_clip, 0.0)
    return float(alpha) * np.exp(exp_arg)


def ewa_splat_weights_at_point(
    p: np.ndarray,
    means: np.ndarray,
    Sigmas_inv: np.ndarray,
    alphas: np.ndarray,
    log_clip: float = 25.0,
) -> np.ndarray:
    """Weights (N,) at pixel p from N splats. Not normalized (caller normalizes). Log-domain clipped."""
    p = np.asarray(p, dtype=np.float64).ravel()[:2]
    N = means.shape[0]
    means = np.asarray(means, dtype=np.float64).reshape(N, 2)
    Sigmas_inv = np.asarray(Sigmas_inv, dtype=np.float64).reshape(N, 2, 2)
    alphas = np.asarray(alphas, dtype=np.float64).ravel()[:N]
    w = np.zeros(N, dtype=np.float64)
    for i in range(N):
        d = p - means[i]
        q = float(d @ Sigmas_inv[i] @ d)
        exp_arg = np.clip(-0.5 * max(q, 0.0), -log_clip, 0.0)
        w[i] = alphas[i] * np.exp(exp_arg)
    return w


# -----------------------------------------------------------------------------
# Multi-lobe vMF shading: s = Σ_b π_b exp(κ_b(μ_{n,b}' v − 1))
# -----------------------------------------------------------------------------


def kappa_modulated_by_intensity(
    kappa_base: float,
    intensity: float,
    intensity_scale: float,
    intensity_max: float = 255.0,
    kappa_max: float = 100.0,
    eps: float = 1e-12,
) -> float:
    """
    Modulate vMF concentration κ by intensity (reflectivity): shinier → higher κ.
    κ_eff = κ_base * (1 + intensity_scale * (intensity / intensity_max)), capped at kappa_max.
    Use when splats carry intensity (e.g. LiDAR reflectivity) as an indicator of shininess.
    """
    if intensity_scale <= 0.0 or intensity_max <= eps:
        return float(kappa_base)
    intensity_norm = float(intensity) / max(float(intensity_max), eps)
    intensity_norm = max(0.0, min(1.0, intensity_norm))
    kappa_eff = float(kappa_base) * (1.0 + float(intensity_scale) * intensity_norm)
    return min(kappa_eff, float(kappa_max))


def vmf_shading_multi_lobe(
    v: np.ndarray,
    mu_app: np.ndarray,
    kappa_app: np.ndarray,
    pi_b: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    intensity_scale: float = 0.0,
    intensity_max: float = 255.0,
    kappa_max: float = 100.0,
    eps: float = 1e-12,
) -> float:
    """
    Shading factor s = Σ_b π_b exp(κ_b(μ_{n,b}' v − 1)).
    v: view direction (3,); mu_app (B, 3) lobes; kappa_app (B,); pi_b (B,) weights, default uniform.
    When intensity is provided and intensity_scale > 0, κ is modulated by intensity (reflectivity)
    so that shinier lobes get higher effective κ: κ_eff = κ * (1 + scale * intensity_norm).
    intensity: optional (B,) or scalar; same units as intensity_max (e.g. uint8 0–255).
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
    kappa_sum = 0.0
    for b in range(B):
        k_b = float(kappa_app[b])
        if intensity_scale > 0.0 and intensity is not None:
            int_b = float(np.asarray(intensity).ravel()[b]) if np.asarray(intensity).size >= B else float(np.asarray(intensity).ravel()[0])
            k_b = kappa_modulated_by_intensity(k_b, int_b, intensity_scale, intensity_max, kappa_max, eps)
        s += pi_b[b] * np.exp(k_b * (float(np.dot(mu_app[b], v)) - 1.0))
        kappa_sum += k_b
    # Energy-normalize so shading is comparable across splats with different κ (prevents blow-out highlights)
    mean_kappa = kappa_sum / max(B, 1)
    s = float(s) / (1.0 + mean_kappa)
    return s


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


def fbm_at_splat_positions(
    means_world_xy: np.ndarray,
    octaves: int = 5,
    gain: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """
    fBm value noise at each splat's world (x,y). Use for view-stable texture: modulate
    albedo/κ/opacity by this so noise does not shimmer when the camera moves.
    means_world_xy (N, 2). Returns (N,) in [0,1].
    """
    means_world_xy = np.asarray(means_world_xy, dtype=np.float64).reshape(-1, 2)
    N = means_world_xy.shape[0]
    out = np.zeros(N, dtype=np.float64)
    for i in range(N):
        out[i] = fbm_value_noise(means_world_xy[i, 0], means_world_xy[i, 1], octaves=octaves, gain=gain, seed=seed)
    return out


# -----------------------------------------------------------------------------
# Opacity from precision: α ∝ σ(γ (logdet0 − logdet)) (from plan)
# -----------------------------------------------------------------------------


def opacity_from_logdet(
    logdet_cov: float,
    gamma: float = 1.0,
    logdet0: float = 0.0,
    alpha_min: float = 0.02,
) -> float:
    """α = alpha_min + (1 - alpha_min) * σ(γ(logdet0 - logdet)). Continuous; soft floor avoids holes."""
    raw = 1.0 / (1.0 + np.exp(-gamma * (logdet0 - logdet_cov)))
    return alpha_min + (1.0 - alpha_min) * raw


# -----------------------------------------------------------------------------
# Tile binning: conservative 2σ bbox per splat → splat indices per tile (fixed cap)
# -----------------------------------------------------------------------------


def splat_indices_for_tile(
    tile_origin: tuple[int, int],
    tile_size: int,
    means: np.ndarray,
    Sigmas_inv: np.ndarray,
    max_splats_per_tile: int = 64,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Splats overlapping this tile, capped at max_splats_per_tile.
    Conservative 2σ ellipse: radius in pixel space = 2/sqrt(λ_min(Σ_inv)).
    Overlap test: splat bbox [μ±r] vs tile rect; order by distance to tile center, take first cap.
    Returns 1D array of splat indices (length ≤ max_splats_per_tile).
    """
    oy, ox = int(tile_origin[0]), int(tile_origin[1])
    tile_size = min(max(int(tile_size), 1), 32)
    N = means.shape[0]
    if N == 0:
        return np.array([], dtype=np.int64)
    means = np.asarray(means, dtype=np.float64).reshape(N, 2)
    Sigmas_inv = np.asarray(Sigmas_inv, dtype=np.float64).reshape(N, 2, 2)
    tile_cx = ox + 0.5 * tile_size
    tile_cy = oy + 0.5 * tile_size
    tile_x_max = ox + tile_size
    tile_y_max = oy + tile_size
    overlapping: list[tuple[float, int]] = []
    for i in range(N):
        eigvals = np.linalg.eigvalsh(Sigmas_inv[i])
        lam_min = max(float(np.min(eigvals)), eps)
        r = 2.0 / np.sqrt(lam_min)
        mx, my = float(means[i, 0]), float(means[i, 1])
        if mx + r < ox or mx - r > tile_x_max or my + r < oy or my - r > tile_y_max:
            continue
        dist_sq = (mx - tile_cx) ** 2 + (my - tile_cy) ** 2
        overlapping.append((dist_sq, i))
    overlapping.sort(key=lambda x: x[0])
    indices = np.array([idx for _, idx in overlapping[:max_splats_per_tile]], dtype=np.int64)
    return indices


# -----------------------------------------------------------------------------
# Tiled render (fixed tile size; fixed cap on splats per tile for cost)
# -----------------------------------------------------------------------------


def render_tile_ewa(
    tile_origin: tuple[int, int],
    tile_size: int,
    means: np.ndarray,
    Sigmas_inv: np.ndarray,
    alphas: np.ndarray,
    colors: np.ndarray,
    splat_indices: np.ndarray | None = None,
    log_clip: float = 25.0,
    means_world_xy: np.ndarray | None = None,
    fbm_octaves: int = 5,
    fbm_gain: float = 0.5,
    fbm_seed: int = 0,
    fbm_modulate_scale: float = 0.0,
) -> np.ndarray:
    """
    Render one tile (tile_size × tile_size) with EWA splats. Returns RGB (tile_size, tile_size, 3).
    means (N, 2), Sigmas_inv (N, 2, 2), alphas (N,), colors (N, 3).
    splat_indices: optional; when set, only these splats are considered (tile binning).
    log_clip: clip -0.5*q to [-log_clip, 0] for numerical stability.
    means_world_xy: optional (N, 2) world x,y; when set with fbm_modulate_scale > 0, fBm at world
    position modulates color (view-stable texture, no shimmer).
    """
    tile_size = min(max(int(tile_size), 1), 32)
    out = np.zeros((tile_size, tile_size, 3), dtype=np.float64)
    w_sum = np.zeros((tile_size, tile_size), dtype=np.float64)
    N = means.shape[0]
    if N == 0:
        return out
    if splat_indices is not None and splat_indices.size == 0:
        return np.clip(out, 0.0, 1.0)
    idx = splat_indices if splat_indices is not None else np.arange(N, dtype=np.int64)
    means_sub = means[idx]
    Sigmas_inv_sub = Sigmas_inv[idx]
    alphas_sub = alphas[idx]
    colors_sub = np.asarray(colors, dtype=np.float64).reshape(-1, 3)[idx]
    if fbm_modulate_scale > 0.0 and means_world_xy is not None:
        world = np.asarray(means_world_xy, dtype=np.float64).reshape(-1, 2)[idx]
        n_sub = world.shape[0]
        fbm_val = np.zeros(n_sub, dtype=np.float64)
        for i in range(n_sub):
            fbm_val[i] = fbm_value_noise(world[i, 0], world[i, 1], octaves=fbm_octaves, gain=fbm_gain, seed=fbm_seed)
        mod = (1.0 - fbm_modulate_scale) + fbm_modulate_scale * fbm_val
        colors_sub = colors_sub * mod[:, np.newaxis]
    oy, ox = tile_origin[0], tile_origin[1]
    n_sub = means_sub.shape[0]
    for py in range(tile_size):
        for px in range(tile_size):
            p = np.array([ox + px + 0.5, oy + py + 0.5], dtype=np.float64)
            w = ewa_splat_weights_at_point(p, means_sub, Sigmas_inv_sub, alphas_sub, log_clip=log_clip)
            total = np.sum(w) + 1e-12
            for i in range(n_sub):
                out[py, px, :] += w[i] * colors_sub[i, :3]
            w_sum[py, px] = total
    for py in range(tile_size):
        for px in range(tile_size):
            if w_sum[py, px] > 1e-12:
                out[py, px, :] /= w_sum[py, px]
    return np.clip(out, 0.0, 1.0)
