"""
Independent weight combination utilities.

Weights represent independent likelihood factors, so combination is the
product of weights (equivalently, sum of log-likelihoods).
"""

import math
from typing import Iterable


def combine_independent_weights(weights: Iterable[float], eps: float = 1e-12) -> float:
    """
    Combine independent likelihood weights via product.

    Uses log-space accumulation for numerical stability:
        log w_total = sum(log(max(w_i, eps)))

    Args:
        weights: iterable of weights in [0, 1]
        eps: small floor to avoid log(0) underflow

    Returns:
        Combined weight in (0, 1]
    """
    log_sum = 0.0
    for w in weights:
        w_clamped = max(float(w), eps)
        log_sum += math.log(w_clamped)
    return math.exp(log_sum)

