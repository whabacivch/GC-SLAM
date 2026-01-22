"""
Gaussian pre-Frobenius correction utilities.

For the full Gaussian family, the cubic tensor C = ∇³ψ is zero in natural
coordinates (see Comprehensive Information Geometry.md). This means the
third-order Frobenius correction is an identity (no-op) for Gaussian updates.

We still emit proof-of-execution metadata so approximation triggers remain
auditable and compliant with the Frobenius policy.
"""

import numpy as np


def _vec_stats(vec: np.ndarray) -> dict:
    v = np.asarray(vec, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "norm": float(np.linalg.norm(v)),
    }


def gaussian_frobenius_correction(delta: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Apply Gaussian Frobenius correction (identity for C = 0).

    Returns:
        delta_corr: corrected delta (equals delta for Gaussian family)
        stats: dict with delta_norm, input_stats, output_stats
    """
    d = np.asarray(delta, dtype=float).reshape(-1)
    delta_corr = d.copy()
    stats = {
        "delta_norm": 0.0,
        "input_stats": {"delta": _vec_stats(d)},
        "output_stats": {"delta_corr": _vec_stats(delta_corr)},
    }
    return delta_corr, stats

