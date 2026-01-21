"""
Dirichlet-Categorical Routing Module.

Implements uncertainty-aware routing for IMU factor fusion with:
- Dirichlet conjugate prior over categorical routing probabilities
- Frobenius retention (cubic contraction) for belief persistence
- Hellinger shift monitoring for stability diagnostics

This is NOT "normal softmax" as the end belief. Softmax is used as the
categorical mean map to generate pseudo-count evidence, which is then
used to update the Dirichlet posterior.

Key principle (from CL framework):
    "Softmax IS the categorical exponential family mean map"
    The correct uncertainty-bearing object is Dirichlet-categorical conjugacy.

Reference: Compositional Legendre framework, Hellinger hierarchical construction
"""

import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import cholesky, solve_triangular
from typing import Dict
import numpy as np  # For interface compatibility with ROS callbacks


# =============================================================================
# Numerical Constants
# =============================================================================

# Minimum probability/weight for numerical stability
# Prevents log(0) and division by zero
MIN_PROB = 1e-15

# Default Frobenius retention base (cubic contraction applied)
# t = 0.95^3 ≈ 0.857, meaning ~14% belief decay per update
DEFAULT_RETENTION_BASE = 0.95

# Default Dirichlet prior concentration (symmetric)
# α = 1.0 corresponds to uniform prior (non-informative)
DEFAULT_ALPHA_PRIOR = 1.0

# Default evidence budget per update
# B = 1.0 means each observation contributes 1 total pseudo-count
DEFAULT_EVIDENCE_BUDGET = 1.0

# Prior strength for E[log θ] term in combined logits
DEFAULT_LAMBDA_PRIOR = 1.0


class DirichletRoutingModule:
    """
    Dirichlet-categorical routing with Frobenius retention.
    
    Maintains a Dirichlet posterior over anchor routing probabilities.
    Updates are Bayesian with a Frobenius cubic retention factor.
    
    The Dirichlet-categorical conjugacy means:
    - Prior: Dir(α)
    - Likelihood: Cat(θ) with θ ~ Dir(α)
    - Posterior: Dir(α + counts)
    
    Frobenius retention applies a cubic contraction α' = t³ * α before
    each update, implementing a principled "forgetting" that prevents
    belief collapse to a single anchor.
    """
    
    def __init__(
        self,
        n_anchors: int,
        alpha_prior: float = DEFAULT_ALPHA_PRIOR,
        retention_base: float = DEFAULT_RETENTION_BASE,
        evidence_budget: float = DEFAULT_EVIDENCE_BUDGET,
        lambda_prior: float = DEFAULT_LAMBDA_PRIOR,
    ):
        """
        Initialize Dirichlet routing module.
        
        Args:
            n_anchors: Number of anchors to route between
            alpha_prior: Initial Dirichlet concentration (symmetric)
            retention_base: Base for Frobenius cubic contraction (0 < t < 1)
            evidence_budget: Total pseudo-count per update (B)
            lambda_prior: Weight for E[log θ] term in combined logits
        
        Raises:
            ValueError: If parameters are out of valid range
        """
        if n_anchors <= 0:
            raise ValueError(f"n_anchors must be positive, got {n_anchors}")
        if alpha_prior <= 0:
            raise ValueError(f"alpha_prior must be positive, got {alpha_prior}")
        if not 0 < retention_base < 1:
            raise ValueError(f"retention_base must be in (0, 1), got {retention_base}")
        if evidence_budget <= 0:
            raise ValueError(f"evidence_budget must be positive, got {evidence_budget}")
        
        self.n_anchors = n_anchors
        self.alpha_prior = alpha_prior
        self.retention_base = retention_base
        self.evidence_budget = evidence_budget
        self.lambda_prior = lambda_prior
        
        # Initialize Dirichlet parameters (symmetric prior)
        self.alpha = np.full(n_anchors, alpha_prior, dtype=np.float64)
        
        # State for Hellinger shift monitoring
        self._prev_resp: np.ndarray | None = None
        self._last_hellinger_shift: float = 0.0
        self._update_count: int = 0
    
    def update(self, logits: np.ndarray) -> np.ndarray:
        """
        Update routing belief with new evidence.
        
        Algorithm:
        1. Frobenius retention: α' = t³ * α (cubic contraction)
        2. Combined logits: s_i = ω_i + λ * E[log θ_i]
           where E[log θ_i] = ψ(α_i) - ψ(Σα) ≈ log(α_i) - log(Σα)
        3. Softmax → pseudo-counts: c = B * softmax(s)
        4. Dirichlet update: α = α' + c
        5. Hellinger shift: H² = 1 - Σ√(π_t · π_{t-1})
        6. Return responsibilities: w = E[θ] = α / Σα
        
        Args:
            logits: Per-anchor log-weights from likelihood (M,)
        
        Returns:
            responsibilities: Dirichlet mean (normalized) (M,)
        
        Raises:
            ValueError: If logits dimension doesn't match n_anchors
        """
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        
        if len(logits) != self.n_anchors:
            raise ValueError(
                f"logits dimension {len(logits)} != n_anchors {self.n_anchors}"
            )
        
        # Step 1: Frobenius retention (cubic contraction)
        retention = self.retention_base ** 3
        alpha_retained = retention * self.alpha
        
        # Step 2: Combined logits with Dirichlet prior term
        # E[log θ_i] ≈ log(α_i) - log(Σα) for practical α values
        alpha_sum = np.sum(alpha_retained)
        expected_log_theta = np.log(alpha_retained + MIN_PROB) - np.log(alpha_sum + MIN_PROB)
        
        combined_logits = logits + self.lambda_prior * expected_log_theta
        
        # Step 3: Numerically stable softmax → pseudo-counts
        logits_shifted = combined_logits - np.max(combined_logits)
        exp_logits = np.exp(logits_shifted)
        softmax_probs = exp_logits / np.sum(exp_logits)
        
        pseudo_counts = self.evidence_budget * softmax_probs
        
        # Step 4: Dirichlet update
        self.alpha = alpha_retained + pseudo_counts
        
        # Step 5: Responsibilities (Dirichlet mean)
        responsibilities = self.alpha / np.sum(self.alpha)
        
        # Step 6: Hellinger shift diagnostic
        if self._prev_resp is not None:
            self._last_hellinger_shift = self._hellinger_squared(
                responsibilities, self._prev_resp
            )
        
        self._prev_resp = responsibilities.copy()
        self._update_count += 1
        
        return responsibilities
    
    def _hellinger_squared(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Squared Hellinger distance between discrete distributions.
        
        H²(p, q) = 1 - Σᵢ √(pᵢ · qᵢ)
        
        Properties:
        - H² ∈ [0, 1]
        - H² = 0 iff p = q
        - H² = 1 iff p and q have disjoint support
        """
        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(np.maximum(p * q, 0.0)))
        return float(max(0.0, 1.0 - bc))
    
    def get_responsibilities(self) -> np.ndarray:
        """Get current responsibilities (Dirichlet mean)."""
        return self.alpha / np.sum(self.alpha)
    
    def get_retention_scalar(self) -> float:
        """Get Frobenius retention factor (t³)."""
        return self.retention_base ** 3
    
    def get_hellinger_shift(self) -> float:
        """Get last computed Hellinger shift."""
        return self._last_hellinger_shift
    
    def get_alpha(self) -> np.ndarray:
        """Get current Dirichlet concentration parameters."""
        return self.alpha.copy()
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information for logging/monitoring."""
        responsibilities = self.get_responsibilities()
        return {
            "alpha": self.alpha.tolist(),
            "alpha_sum": float(np.sum(self.alpha)),
            "responsibilities": responsibilities.tolist(),
            "max_responsibility": float(np.max(responsibilities)),
            "retention_factor": self.get_retention_scalar(),
            "hellinger_shift": self._last_hellinger_shift,
            "update_count": self._update_count,
            "entropy": float(-np.sum(responsibilities * np.log(responsibilities + MIN_PROB))),
        }
    
    def resize(self, new_n_anchors: int) -> None:
        """
        Resize module for different number of anchors.
        
        New anchors receive the prior concentration.
        Removed anchors' mass is lost (not redistributed).
        
        Args:
            new_n_anchors: New number of anchors
        
        Raises:
            ValueError: If new_n_anchors <= 0
        """
        if new_n_anchors <= 0:
            raise ValueError(f"new_n_anchors must be positive, got {new_n_anchors}")
        
        if new_n_anchors > self.n_anchors:
            # Add new anchors with prior concentration
            extra = np.full(new_n_anchors - self.n_anchors, self.alpha_prior, dtype=np.float64)
            self.alpha = np.concatenate([self.alpha, extra])
        elif new_n_anchors < self.n_anchors:
            # Truncate (mass of removed anchors is lost)
            self.alpha = self.alpha[:new_n_anchors]
        
        self.n_anchors = new_n_anchors
        self._prev_resp = None  # Reset Hellinger tracking
    
    def reset(self) -> None:
        """Reset to prior state (symmetric Dirichlet)."""
        self.alpha = np.full(self.n_anchors, self.alpha_prior, dtype=np.float64)
        self._prev_resp = None
        self._last_hellinger_shift = 0.0
        self._update_count = 0


# =============================================================================
# IMU Logit Computation (Hellinger-tilted)
# =============================================================================

@jit
def _hellinger_squared_9d(
    r_bar: jnp.ndarray,
    S: jnp.ndarray,
    R_nom: jnp.ndarray,
) -> float:
    """
    Squared Hellinger distance H²(N(r̄, S), N(0, R_nom)) in 9D.
    
    Uses Cholesky-based computation for numerical stability.
    """
    # Covariance average
    cov_avg = 0.5 * (S + R_nom)
    
    # Regularize
    reg = 1e-8
    cov_avg_reg = cov_avg + jnp.eye(9) * reg
    S_reg = S + jnp.eye(9) * reg
    R_nom_reg = R_nom + jnp.eye(9) * reg
    
    # Cholesky factorizations
    L_avg = cholesky(cov_avg_reg, lower=True)
    L_S = cholesky(S_reg, lower=True)
    L_R = cholesky(R_nom_reg, lower=True)
    
    # Log determinants
    logdet_avg = 2.0 * jnp.sum(jnp.log(jnp.diag(L_avg)))
    logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S)))
    logdet_R = 2.0 * jnp.sum(jnp.log(jnp.diag(L_R)))
    
    # Mahalanobis distance
    y = solve_triangular(L_avg, r_bar, lower=True)
    mahal_sq = jnp.dot(y, y)
    
    # Bhattacharyya coefficient
    log_bc = 0.25 * logdet_S + 0.25 * logdet_R - 0.5 * logdet_avg - 0.125 * mahal_sq
    bc = jnp.exp(log_bc)
    
    return jnp.clip(1.0 - bc, 0.0, 1.0)


def compute_imu_logits(
    anchor_residuals: np.ndarray,
    residual_covs: np.ndarray,
    nominal_cov: np.ndarray,
    hellinger_weight: float = 2.0,
) -> np.ndarray:
    """
    Compute per-anchor logits for IMU factor routing.
    
    Each anchor's logit combines:
    1. Gaussian log-likelihood: -½ r̄ᵀ R⁻¹ r̄
    2. Hellinger tilt: -γ · H²(N(r̄, S), N(0, R_nom))
    
    where γ = hellinger_weight (default 2.0 from Hellinger hierarchical construction).
    
    Args:
        anchor_residuals: Per-anchor residual means (M, 9)
        residual_covs: Per-anchor residual covariances (M, 9, 9)
        nominal_cov: Nominal residual covariance R_nom (9, 9)
        hellinger_weight: Weight for Hellinger term (default 2.0)
    
    Returns:
        logits: Per-anchor log-weights (M,)
    """
    M = len(anchor_residuals)
    logits = np.zeros(M, dtype=np.float64)
    
    # Convert to JAX arrays
    anchor_residuals_jax = jnp.array(anchor_residuals)
    residual_covs_jax = jnp.array(residual_covs)
    R_nom_jax = jnp.array(nominal_cov)
    
    # Compute R_nom inverse via Cholesky
    R_nom_reg = R_nom_jax + jnp.eye(9) * 1e-8
    L_R = cholesky(R_nom_reg, lower=True)
    
    for i in range(M):
        r_i = anchor_residuals_jax[i]
        S_i = residual_covs_jax[i]
        
        # Gaussian log-likelihood: -½ r̄ᵀ R⁻¹ r̄
        y = solve_triangular(L_R, r_i, lower=True)
        log_lik = -0.5 * float(jnp.dot(y, y))
        
        # Hellinger distance
        h_sq = float(_hellinger_squared_9d(r_i, S_i, R_nom_jax))
        
        # Combined logit
        logits[i] = log_lik - hellinger_weight * h_sq
    
    return logits
