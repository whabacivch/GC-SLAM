"""
EXPERIMENTAL: Dirichlet geometry operators for semantic category fusion.

This module is EXPERIMENTAL and not part of the main FL-SLAM pipeline.
It provides information-geometric operations on Dirichlet distributions
for the experimental semantic SLAM components.

Status: Experimental - kept for reference and future development.
"""

import numpy as np
from scipy.special import digamma, gammaln, polygamma

from fl_slam_poc.common.op_report import OpReport

EPS = 1e-9  # numerical interior safeguard (domain constraint)


def _vec_stats(vec: np.ndarray) -> dict:
    v = np.asarray(vec, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "norm": float(np.linalg.norm(v)),
    }


def dirichlet_log_partition(alpha: np.ndarray) -> float:
    """
    Log-partition for Dirichlet in alpha-coordinates:
      psi(alpha) = sum log Gamma(alpha_i) - log Gamma(sum alpha)
    """
    a = np.asarray(alpha, dtype=float).reshape(-1)
    s = float(np.sum(a))
    return float(np.sum(gammaln(a)) - gammaln(s))


def psi_potential(alpha: np.ndarray) -> float:
    """Alias for the Dirichlet log-partition potential."""
    return dirichlet_log_partition(alpha)


def g_fisher(alpha: np.ndarray) -> np.ndarray:
    """
    Fisher metric in alpha-coordinates:
      g_ij = trigamma(alpha_i) * delta_ij - trigamma(sum alpha)
    where trigamma(x) = polygamma(1, x)
    """
    a = np.asarray(alpha, dtype=float).reshape(-1)
    s = float(np.sum(a))
    trigamma_sum = float(polygamma(1, s))
    trigamma = polygamma(1, a).astype(float)
    ones = np.ones((a.size, 1), dtype=float)
    g = np.diag(trigamma) - trigamma_sum * (ones @ ones.T)
    return g


def c_contract_uv(alpha: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute b_i = sum_{j,k} C_{ijk} u_j v_k without forming C explicitly.

    For Dirichlet:
      C_{ijk} = polygamma(2, alpha_i) * delta_ij * delta_ik - polygamma(2, sum a)

    Then:
      b_i = polygamma(2, alpha_i) * u_i * v_i - polygamma(2, sum a) * (sum u) * (sum v)
    """
    a = np.asarray(alpha, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)

    s = float(np.sum(a))
    poly_sum = float(polygamma(2, s))
    poly = polygamma(2, a).astype(float)

    su = float(np.sum(u))
    sv = float(np.sum(v))

    b = poly * u * v - poly_sum * su * sv * np.ones_like(a)
    return b


def frob_product(alpha: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Frobenius-induced tangent product u ∘ v defined by:
      g(u ∘ v, w) = C(u, v, w) for all w
    """
    b = c_contract_uv(alpha, u, v)
    g = g_fisher(alpha)
    return np.linalg.solve(g, b)


def third_order_correct(alpha: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Apply third-order Frobenius correction to a tangent update delta:
      delta_corr = delta + 0.5 * (delta ∘ delta)
    """
    d = np.asarray(delta, dtype=float).reshape(-1)
    d2 = frob_product(alpha, d, d)
    return d + 0.5 * d2


def target_E_log_p_from_mixture(alphas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute target t_i = E_mix[log p_i] for a mixture of Dirichlets.
    """
    A = np.asarray(alphas, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / np.sum(w)

    sums = np.sum(A, axis=1).reshape(-1, 1)
    t = np.sum(w.reshape(-1, 1) * (digamma(A) - digamma(sums)), axis=0)
    return t


def residual_f(alpha: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """
    f_i(alpha) = digamma(alpha_i) - digamma(sum alpha) - target_t_i
    Root f(alpha) = 0 defines the information projection matching E[log p].
    """
    a = np.maximum(np.asarray(alpha, dtype=float).reshape(-1), EPS)
    s = float(np.sum(a))
    return (digamma(a) - digamma(s)) - target_t


def iproject_dirichlet_from_mixture(
    alphas: np.ndarray,
    weights: np.ndarray,
    alpha_init: np.ndarray,
    max_iter: int = 5,
    tol: float = 1e-9,
    use_third_order: bool = True,
) -> tuple[np.ndarray, OpReport]:
    """
    Information projection of a Dirichlet mixture to a single Dirichlet by solving:
      digamma(alpha_i) - digamma(sum alpha) = target_t_i

    Newton step uses Jacobian J = g_fisher (Fisher metric).
    Optional third-order Frobenius correction applied to the Newton delta.
    """
    target_t = target_E_log_p_from_mixture(alphas, weights)
    a = np.asarray(alpha_init, dtype=float).reshape(-1)
    projection_hit = bool(np.any(a <= EPS))
    a = np.maximum(a, EPS)

    report = OpReport(
        name="DirichletMixtureProjection",
        exact=False,
        approximation_triggers=["MixtureReduction"],
        family_in="DirichletMixture",
        family_out="Dirichlet",
        closed_form=False,
        solver_used="Newton",
        frobenius_applied=bool(use_third_order),
        frobenius_operator="dirichlet_third_order" if use_third_order else None,
        # Only allow ablation when explicitly requested by the caller.
        allow_ablation=False,
        domain_projection=projection_hit,
    )
    report.metrics = {"iters": 0, "final_norm": None}
    if use_third_order:
        zeros = np.zeros_like(a, dtype=float)
        report.frobenius_delta_norm = 0.0
        report.frobenius_input_stats = {"alpha": _vec_stats(a), "delta": _vec_stats(zeros)}
        report.frobenius_output_stats = {"delta_corr": _vec_stats(zeros)}

    for it in range(max_iter):
        f = residual_f(a, target_t)
        norm = float(np.linalg.norm(f))
        report.metrics["iters"] = it + 1
        report.metrics["final_norm"] = norm
        if norm < tol:
            break

        J = g_fisher(a)
        delta_raw = np.linalg.solve(J, -f)
        delta = delta_raw
        if use_third_order:
            delta = third_order_correct(a, delta_raw)
            report.frobenius_delta_norm = float(np.linalg.norm(delta - delta_raw))
            report.frobenius_input_stats = {
                "alpha": _vec_stats(a),
                "delta": _vec_stats(delta_raw),
            }
            report.frobenius_output_stats = {
                "delta_corr": _vec_stats(delta),
            }

        a_next = a + delta
        if np.any(a_next <= EPS):
            report.domain_projection = True
        a = np.maximum(a_next, EPS)

    return a, report
