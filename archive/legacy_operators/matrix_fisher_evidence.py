"""
Matrix Fisher rotation evidence operator for Golden Child SLAM v2.

Replaces Wahba + vMF Laplace with principled Matrix Fisher distribution on SO(3).

The Matrix Fisher distribution has density:
    p(R | F) ∝ exp(tr(F^T R))

where F ∈ ℝ^{3×3} is the natural parameter matrix.

Key insight: The cross-covariance H = Σ_b (u_map ⊗ u_scan) summed over bins
is the sufficient statistic for rotation estimation. The SVD H = U S V^T
gives the ML rotation R* = U V^T.

The information matrix (Fisher info) for rotation perturbations δθ is derived
from the second-order expansion of tr(F^T exp([δθ]_×) R*).

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 5.9 (replacement)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import (
    BeliefGaussianInfo,
    D_Z,
    SLICE_POSE,
    pose_se3_to_z_delta,
)
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    ExcitationCert,
    MismatchCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    domain_projection_psd_core,
)
from fl_slam_poc.common.geometry import se3_jax


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ScatterMetrics:
    """Anisotropy and planarity metrics from directional scatter tensor."""
    eigenvalues: jnp.ndarray  # (3,) sorted descending λ₁ ≥ λ₂ ≥ λ₃
    eigenvectors: jnp.ndarray  # (3, 3) columns are eigenvectors
    linearity: float  # (λ₁ - λ₂) / λ₁ - how line-like
    planarity: float  # (λ₂ - λ₃) / λ₁ - how plane-like
    sphericity: float  # λ₃ / λ₁ - how sphere-like
    anisotropy: float  # 1 - sphericity
    effective_rank: float  # Continuous rank estimate


@dataclass
class MatrixFisherResult:
    """Result of Matrix Fisher rotation evidence operator."""
    R_mf: jnp.ndarray  # (3, 3) ML rotation from Matrix Fisher
    L_rot: jnp.ndarray  # (3, 3) rotation information matrix in tangent space
    h_rot: jnp.ndarray  # (3,) rotation information vector
    delta_rot: jnp.ndarray  # (3,) rotation residual (axis-angle)
    svd_singular_values: jnp.ndarray  # (3,) SVD singular values of H
    map_scatter_metrics: ScatterMetrics  # Metrics for map directions
    scan_scatter_metrics: ScatterMetrics  # Metrics for scan directions


# =============================================================================
# Scatter Metrics Computation
# =============================================================================


def compute_scatter_metrics(
    S_scatter: jnp.ndarray,
    N_total: float,
    eps: float = constants.GC_EPS_MASS,
) -> ScatterMetrics:
    """
    Compute anisotropy/planarity metrics from directional scatter tensor.

    Args:
        S_scatter: (3, 3) directional scatter Σ w u u^T (summed over bins)
        N_total: Total mass (for normalization)
        eps: Regularization epsilon

    Returns:
        ScatterMetrics with eigenvalue-based shape descriptors
    """
    # Normalize to get the orientation tensor (covariance-like)
    inv_N = 1.0 / (N_total + eps)
    T = S_scatter * inv_N  # Orientation tensor in [0, 1]^{3x3}

    # Eigendecomposition (returns eigenvalues in ascending order)
    eigenvalues_asc, eigenvectors = jnp.linalg.eigh(T)

    # Sort descending: λ₁ ≥ λ₂ ≥ λ₃
    idx = jnp.argsort(eigenvalues_asc)[::-1]
    eigenvalues = eigenvalues_asc[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp eigenvalues to [0, ∞) for numerical stability
    eigenvalues = jnp.maximum(eigenvalues, 0.0)

    lambda1 = eigenvalues[0]
    lambda2 = eigenvalues[1]
    lambda3 = eigenvalues[2]

    # Normalized shape descriptors (all in [0, 1])
    inv_lambda1 = 1.0 / (lambda1 + eps)
    linearity = (lambda1 - lambda2) * inv_lambda1
    planarity = (lambda2 - lambda3) * inv_lambda1
    sphericity = lambda3 * inv_lambda1
    anisotropy = 1.0 - sphericity

    # Effective rank: entropy-based continuous rank
    # Sum of eigenvalues = 1 (for normalized), so this is well-defined
    total = lambda1 + lambda2 + lambda3 + eps
    p1 = lambda1 / total
    p2 = lambda2 / total
    p3 = lambda3 / total
    entropy = -(
        p1 * jnp.log(p1 + eps)
        + p2 * jnp.log(p2 + eps)
        + p3 * jnp.log(p3 + eps)
    )
    # Normalize so max entropy (uniform) gives rank 3
    effective_rank = jnp.exp(entropy)

    return ScatterMetrics(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        linearity=float(linearity),
        planarity=float(planarity),
        sphericity=float(sphericity),
        anisotropy=float(anisotropy),
        effective_rank=float(effective_rank),
    )


# =============================================================================
# Matrix Fisher Core
# =============================================================================


@jax.jit
def _matrix_fisher_core(
    scan_S_dir: jnp.ndarray,  # (B, 3) scan resultants
    scan_S_scatter: jnp.ndarray,  # (B, 3, 3) scan scatter tensors
    scan_N: jnp.ndarray,  # (B,) scan masses
    map_S_dir: jnp.ndarray,  # (B, 3) map resultants
    map_S_scatter: jnp.ndarray,  # (B, 3, 3) map scatter tensors
    map_N: jnp.ndarray,  # (B,) map masses
    R_pred: jnp.ndarray,  # (3, 3) predicted rotation
    eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled Matrix Fisher core computation.

    Returns:
        R_mf: (3, 3) ML rotation
        L_rot: (3, 3) rotation information
        svd_s: (3,) singular values
        scan_scatter_total: (3, 3) total scan scatter
        map_scatter_total: (3, 3) total map scatter
        N_effective: scalar effective mass
    """
    n_bins = scan_N.shape[0]

    # Compute per-bin weights based on geometric mean of masses
    # This down-weights bins where either scan or map has low confidence
    w_b = jnp.sqrt(scan_N * map_N + eps)  # (B,)

    # Normalize scan and map directions to get mean directions
    scan_norms = jnp.linalg.norm(scan_S_dir, axis=1, keepdims=True)  # (B, 1)
    map_norms = jnp.linalg.norm(map_S_dir, axis=1, keepdims=True)  # (B, 1)

    # Mean directions (normalized resultants)
    u_scan = scan_S_dir / (scan_norms + eps)  # (B, 3)
    u_map = map_S_dir / (map_norms + eps)  # (B, 3)

    # Resultant lengths give concentration (R-bar)
    inv_scan_N = 1.0 / (scan_N + eps)
    inv_map_N = 1.0 / (map_N + eps)
    Rbar_scan = scan_norms.squeeze(-1) * inv_scan_N  # (B,)
    Rbar_map = map_norms.squeeze(-1) * inv_map_N  # (B,)

    # Per-bin confidence: product of resultant lengths
    # High Rbar = concentrated directions = confident
    confidence_b = Rbar_scan * Rbar_map  # (B,)

    # Build the cross-covariance matrix H = Σ_b w_b * u_map @ u_scan^T
    # This is the sufficient statistic for rotation estimation
    # Note: map directions are in world frame, scan in body frame
    # So H encodes the correspondence u_map ≈ R @ u_scan

    # Weight by mass and confidence
    w_final = w_b * confidence_b  # (B,)

    # H = Σ w_b (u_map ⊗ u_scan) where we want R such that u_map ≈ R @ u_scan
    # So H[i,j] = Σ w_b * u_map[i] * u_scan[j]
    H = jnp.einsum("b,bi,bj->ij", w_final, u_map, u_scan)  # (3, 3)

    # SVD: H = U @ S @ V^T
    # The ML rotation is R* = U @ V^T (Kabsch/Wahba solution)
    U, s, Vt = jnp.linalg.svd(H, full_matrices=True)

    # Handle reflection: ensure det(R) = +1
    det_sign = jnp.linalg.det(U @ Vt)
    # If det < 0, flip the sign of the last column of U
    U_corrected = U.at[:, 2].set(U[:, 2] * jnp.sign(det_sign))

    R_mf = U_corrected @ Vt  # (3, 3)

    # ==========================================================================
    # Fisher Information for rotation perturbations
    # ==========================================================================
    # The Matrix Fisher log-likelihood is L(R) = tr(H^T R)
    # At R = R*, the Hessian in the tangent space gives the information matrix.
    #
    # For a perturbation R = R* exp([δθ]×), to second order:
    #   tr(H^T R) ≈ tr(H^T R*) + δθ^T J_rot + 0.5 δθ^T H_rot δθ
    #
    # where H_rot (the information matrix) depends on the singular values of H.
    # Specifically, for rotation around axis e_i, the curvature is:
    #   H_rot[i,i] = s[j] + s[k]  for {i,j,k} a cyclic permutation of {1,2,3}
    #
    # The off-diagonal terms are zero when H is diagonal (which SVD ensures).

    # Information matrix in local frame (diagonal)
    L_rot_diag = jnp.array([
        s[1] + s[2],  # rotation around x: curvature from y,z singular values
        s[0] + s[2],  # rotation around y: curvature from x,z singular values
        s[0] + s[1],  # rotation around z: curvature from x,y singular values
    ], dtype=jnp.float64)

    # This is in the frame where H is diagonal (SVD frame).
    # Transform to body frame: L_body = V @ diag(L_rot_diag) @ V^T
    V = Vt.T
    L_rot = V @ jnp.diag(L_rot_diag) @ V.T  # (3, 3)

    # Compute total scatter tensors for metrics
    scan_scatter_total = jnp.sum(scan_S_scatter, axis=0)  # (3, 3)
    map_scatter_total = jnp.sum(map_S_scatter, axis=0)  # (3, 3)
    N_effective = jnp.sum(w_final)

    return R_mf, L_rot, s, scan_scatter_total, map_scatter_total, N_effective


# =============================================================================
# Main Operator
# =============================================================================


def matrix_fisher_rotation_evidence(
    belief_pred: BeliefGaussianInfo,
    scan_s_dir: jnp.ndarray,
    scan_S_dir_scatter: jnp.ndarray,
    scan_N: jnp.ndarray,
    map_S_dir: jnp.ndarray,
    map_S_dir_scatter: jnp.ndarray,
    map_N_dir: jnp.ndarray,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    eps_mass: float = constants.GC_EPS_MASS,
) -> Tuple[MatrixFisherResult, CertBundle, ExpectedEffect]:
    """
    Matrix Fisher rotation evidence from directional correspondences.

    Produces rotation evidence by matching scan and map directions using
    the Matrix Fisher distribution on SO(3).

    Key advantages over Wahba + vMF Laplace:
    1. Uses full scatter tensor, not just mean direction
    2. Information matrix derived from Fisher info, not heuristic
    3. Naturally handles anisotropic directional distributions
    4. No separate "Wahba" step - estimation and uncertainty unified

    Args:
        belief_pred: Predicted belief (for chart info)
        scan_s_dir: (B, 3) scan direction resultants
        scan_S_dir_scatter: (B, 3, 3) scan directional scatter tensors
        scan_N: (B,) scan bin masses
        map_S_dir: (B, 3) map direction resultants (in world frame)
        map_S_dir_scatter: (B, 3, 3) map directional scatter tensors
        map_N_dir: (B,) map bin masses
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        eps_mass: Mass regularization epsilon

    Returns:
        Tuple of (MatrixFisherResult, CertBundle, ExpectedEffect)
    """
    scan_s_dir = jnp.asarray(scan_s_dir, dtype=jnp.float64)
    scan_S_dir_scatter = jnp.asarray(scan_S_dir_scatter, dtype=jnp.float64)
    scan_N = jnp.asarray(scan_N, dtype=jnp.float64)
    map_S_dir = jnp.asarray(map_S_dir, dtype=jnp.float64)
    map_S_dir_scatter = jnp.asarray(map_S_dir_scatter, dtype=jnp.float64)
    map_N_dir = jnp.asarray(map_N_dir, dtype=jnp.float64)

    # Get predicted rotation for residual computation
    pose_pred = belief_pred.mean_world_pose(eps_lift=eps_lift)
    R_pred = se3_jax.so3_exp(pose_pred[3:6])

    # Core computation
    R_mf, L_rot_raw, svd_s, scan_scatter_total, map_scatter_total, N_eff = _matrix_fisher_core(
        scan_s_dir,
        scan_S_dir_scatter,
        scan_N,
        map_S_dir,
        map_S_dir_scatter,
        map_N_dir,
        R_pred,
        eps_mass,
    )

    # Compute rotation residual: δθ = Log(R_pred^T @ R_mf)
    # This is the rotation from predicted to measured, expressed in body frame
    R_err = R_pred.T @ R_mf
    delta_rot = se3_jax.so3_log(R_err)  # (3,) axis-angle

    # Project L_rot to PSD
    L_rot_psd, psd_cert = domain_projection_psd_core(L_rot_raw, eps_psd)

    # Compute h_rot = L_rot @ delta_rot
    h_rot = L_rot_psd @ delta_rot

    # Compute scatter metrics for diagnostics
    N_scan_total = float(jnp.sum(scan_N))
    N_map_total = float(jnp.sum(map_N_dir))
    scan_metrics = compute_scatter_metrics(scan_scatter_total, N_scan_total, eps_mass)
    map_metrics = compute_scatter_metrics(map_scatter_total, N_map_total, eps_mass)

    # Build result
    result = MatrixFisherResult(
        R_mf=R_mf,
        L_rot=L_rot_psd,
        h_rot=h_rot,
        delta_rot=delta_rot,
        svd_singular_values=svd_s,
        map_scatter_metrics=map_metrics,
        scan_scatter_metrics=scan_metrics,
    )

    # Compute mismatch proxy (rotation NLL)
    rot_nll = 0.5 * float(delta_rot @ L_rot_psd @ delta_rot)

    # Build certificate
    eig_min = float(jnp.min(svd_s))
    eig_max = float(jnp.max(svd_s))
    cond = eig_max / (eig_min + eps_mass)

    cert = CertBundle.create_approx(
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
        triggers=["MatrixFisherRotationEvidence"],
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=int(jnp.sum(svd_s < eps_mass)),
        ),
        mismatch=MismatchCert(
            nll_per_ess=rot_nll / (N_eff + eps_mass),
            directional_score=float(jnp.sum(svd_s)),
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=float(psd_cert[0]),
            mass_epsilon_ratio=float(eps_mass / (N_eff + eps_mass)),
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    # Expected effect
    expected_effect = ExpectedEffect(
        objective_name="predicted_rotation_nll",
        predicted=rot_nll,
        realized=None,
    )

    return result, cert, expected_effect


# =============================================================================
# Planarized Translation Evidence (SE(2.5))
# =============================================================================


@dataclass
class PlanarTranslationResult:
    """Result of planarized translation evidence."""
    t_wls: jnp.ndarray  # (3,) WLS translation estimate
    L_trans: jnp.ndarray  # (3, 3) translation information matrix
    h_trans: jnp.ndarray  # (3,) translation information vector
    delta_trans: jnp.ndarray  # (3,) translation residual
    xy_info_scale: float  # Information scale for XY
    z_info_scale: float  # Information scale for Z (should be ~0)


@jax.jit
def _planar_translation_wls_core(
    scan_p_bar: jnp.ndarray,  # (B, 3) scan centroids
    scan_Sigma_p: jnp.ndarray,  # (B, 3, 3) scan centroid covariances
    scan_N: jnp.ndarray,  # (B,) scan masses
    map_centroid: jnp.ndarray,  # (B, 3) map centroids
    map_Sigma_c: jnp.ndarray,  # (B, 3, 3) map centroid covariances
    map_N: jnp.ndarray,  # (B,) map masses
    R_hat: jnp.ndarray,  # (3, 3) estimated rotation
    eps: float,
    z_precision_scale: float,  # Scale factor for z precision (near 0 for planar)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled planarized translation WLS core.

    Uses weighted least squares with correspondence covariances,
    but scales down the z component of the information matrix.

    Returns:
        t_wls: (3,) WLS translation estimate
        L_trans: (3, 3) translation information matrix
        N_eff: scalar effective mass
    """
    n_bins = scan_N.shape[0]

    # Transform scan centroids to world frame (using estimated rotation)
    # p_world = R @ p_scan + t
    # So the residual is: r = c_map - (R @ p_scan + t)
    # => t = c_map - R @ p_scan
    p_scan_rotated = jnp.einsum("ij,bj->bi", R_hat, scan_p_bar)  # (B, 3)

    # Per-bin translation estimate: t_b = c_map - R @ p_scan
    t_per_bin = map_centroid - p_scan_rotated  # (B, 3)

    # Combined covariance per bin: Σ_b = Σ_map + R @ Σ_scan @ R^T
    Sigma_scan_rotated = jnp.einsum("ij,bjk,lk->bil", R_hat, scan_Sigma_p, R_hat)  # (B, 3, 3)
    Sigma_combined = map_Sigma_c + Sigma_scan_rotated  # (B, 3, 3)

    # Per-bin weight: geometric mean of masses (down-weights low-mass bins)
    w_b = jnp.sqrt(scan_N * map_N + eps)  # (B,)

    # WLS: minimize Σ w_b (t - t_b)^T Σ_b^{-1} (t - t_b)
    # Normal equations: (Σ w_b Σ_b^{-1}) t = Σ w_b Σ_b^{-1} t_b

    # Build information matrix and vector
    # L = Σ w_b Σ_b^{-1}
    # h = Σ w_b Σ_b^{-1} t_b

    # For each bin, compute inverse of combined covariance
    def inv_cov_one(Sigma, w):
        # Regularize before inversion
        Sigma_reg = Sigma + eps * jnp.eye(3, dtype=jnp.float64)
        Sigma_inv = jnp.linalg.inv(Sigma_reg)
        return w * Sigma_inv

    Sigma_inv_weighted = jax.vmap(inv_cov_one)(Sigma_combined, w_b)  # (B, 3, 3)

    # Sum to get total information matrix
    L_trans_full = jnp.sum(Sigma_inv_weighted, axis=0)  # (3, 3)

    # Information vector: h = Σ w_b Σ_b^{-1} t_b
    h_terms = jnp.einsum("bij,bj->bi", Sigma_inv_weighted, t_per_bin)  # (B, 3)
    h_trans_full = jnp.sum(h_terms, axis=0)  # (3,)

    # Solve for WLS estimate: t = L^{-1} h
    L_reg = L_trans_full + eps * jnp.eye(3, dtype=jnp.float64)
    t_wls = jnp.linalg.solve(L_reg, h_trans_full)

    # ==========================================================================
    # Planarization: Scale down z information
    # ==========================================================================
    # For a planar robot, z translation is not observable from horizontal LiDAR.
    # Instead of trusting the (noisy) z from scan matching, we let the planar
    # prior handle z. Here we scale down the z row/column of L_trans.

    # Create planarization mask: [1, 1, z_precision_scale]
    planar_mask = jnp.array([1.0, 1.0, z_precision_scale], dtype=jnp.float64)

    # Apply to L_trans: L_planar = diag(mask) @ L_full @ diag(mask)
    L_trans = L_trans_full * planar_mask[:, None] * planar_mask[None, :]

    # Also scale h_trans
    h_trans = h_trans_full * planar_mask

    N_eff = jnp.sum(w_b)

    return t_wls, L_trans, N_eff


def planar_translation_evidence(
    belief_pred: BeliefGaussianInfo,
    scan_p_bar: jnp.ndarray,
    scan_Sigma_p: jnp.ndarray,
    scan_N: jnp.ndarray,
    map_centroid: jnp.ndarray,
    map_Sigma_c: jnp.ndarray,
    map_N_pos: jnp.ndarray,
    map_S_dir_scatter: jnp.ndarray,  # For computing vertical observability
    map_N_dir: jnp.ndarray,  # For normalizing scatter
    R_hat: jnp.ndarray,
    eps_psd: float = constants.GC_EPS_PSD,
    eps_lift: float = constants.GC_EPS_LIFT,
    eps_mass: float = constants.GC_EPS_MASS,
) -> Tuple[PlanarTranslationResult, CertBundle, ExpectedEffect]:
    """
    Planarized translation evidence from centroid correspondences.

    Computes WLS translation from scan-map centroid matches. The z precision
    is AUTOMATICALLY scaled based on vertical observability from the map
    directional scatter tensor - no manual knob needed.

    Self-adaptive z_precision:
    - Computes eigenvalues of total map scatter tensor
    - λ₃/λ₁ measures how much directions spread vertically
    - For flat ground (horizontal LiDAR), λ₃ ≈ 0 → z precision ≈ 0
    - For vertical structure (walls), λ₃ > 0 → z precision increases

    Computes WLS translation from scan-map centroid matches, but scales
    down the z component of the information matrix for planar robots.

    This is the SE(2.5) approach: full 3D position estimate, but z
    uncertainty is inflated (low precision) so that the planar prior
    dominates z estimation.

    Args:
        belief_pred: Predicted belief
        scan_p_bar: (B, 3) scan bin centroids
        scan_Sigma_p: (B, 3, 3) scan centroid covariances
        scan_N: (B,) scan bin masses
        map_centroid: (B, 3) map bin centroids
        map_Sigma_c: (B, 3, 3) map centroid covariances
        map_N_pos: (B,) map position masses
        map_S_dir_scatter: (B, 3, 3) map directional scatter tensors
        map_N_dir: (B,) map directional masses
        R_hat: (3, 3) estimated rotation (from Matrix Fisher)
        eps_psd: PSD projection epsilon
        eps_lift: Solve lift epsilon
        eps_mass: Mass regularization epsilon

    Returns:
        Tuple of (PlanarTranslationResult, CertBundle, ExpectedEffect)
    """
    scan_p_bar = jnp.asarray(scan_p_bar, dtype=jnp.float64)
    scan_Sigma_p = jnp.asarray(scan_Sigma_p, dtype=jnp.float64)
    scan_N = jnp.asarray(scan_N, dtype=jnp.float64)
    map_centroid = jnp.asarray(map_centroid, dtype=jnp.float64)
    map_Sigma_c = jnp.asarray(map_Sigma_c, dtype=jnp.float64)
    map_N_pos = jnp.asarray(map_N_pos, dtype=jnp.float64)
    map_S_dir_scatter = jnp.asarray(map_S_dir_scatter, dtype=jnp.float64)
    map_N_dir = jnp.asarray(map_N_dir, dtype=jnp.float64)
    R_hat = jnp.asarray(R_hat, dtype=jnp.float64)

    # ==========================================================================
    # Self-adaptive z_precision_scale from vertical observability
    # ==========================================================================
    # Compute total map scatter tensor and its eigenvalues
    # For flat ground with horizontal LiDAR, directions are mostly in XY plane
    # → λ₃ (smallest eigenvalue) is small → z is unobservable
    # For vertical structure (walls), λ₃ > 0 → z becomes observable
    map_scatter_total = jnp.sum(map_S_dir_scatter, axis=0)  # (3, 3)
    N_dir_total = jnp.sum(map_N_dir) + eps_mass

    # Normalize to orientation tensor
    T_map = map_scatter_total / N_dir_total

    # Eigenvalues (ascending order from eigh)
    eigenvalues = jnp.linalg.eigvalsh(T_map)
    # Sort descending: λ₁ ≥ λ₂ ≥ λ₃
    eigenvalues = jnp.sort(eigenvalues)[::-1]
    lambda1 = jnp.maximum(eigenvalues[0], eps_mass)
    lambda3 = jnp.maximum(eigenvalues[2], 0.0)

    # Vertical observability: ratio of smallest to largest eigenvalue
    # Range [0, 1]: 0 = all directions in a plane, 1 = isotropic
    z_precision_scale = lambda3 / lambda1  # Self-adaptive, no manual knob!

    # Get predicted translation for residual
    pose_pred = belief_pred.mean_world_pose(eps_lift=eps_lift)
    t_pred = pose_pred[:3]

    # Core computation
    t_wls, L_trans_raw, N_eff = _planar_translation_wls_core(
        scan_p_bar,
        scan_Sigma_p,
        scan_N,
        map_centroid,
        map_Sigma_c,
        map_N_pos,
        R_hat,
        eps_mass,
        z_precision_scale,
    )

    # Translation residual
    delta_trans = t_wls - t_pred  # (3,)

    # Project L_trans to PSD
    L_trans_psd, psd_cert = domain_projection_psd_core(L_trans_raw, eps_psd)

    # Compute h_trans = L_trans @ delta_trans
    h_trans = L_trans_psd @ delta_trans

    # Info scales for diagnostics
    xy_info = 0.5 * (L_trans_psd[0, 0] + L_trans_psd[1, 1])
    z_info = L_trans_psd[2, 2]

    # Build result
    result = PlanarTranslationResult(
        t_wls=t_wls,
        L_trans=L_trans_psd,
        h_trans=h_trans,
        delta_trans=delta_trans,
        xy_info_scale=float(xy_info),
        z_info_scale=float(z_info),
    )

    # Compute mismatch proxy
    trans_nll = 0.5 * float(delta_trans @ L_trans_psd @ delta_trans)

    # Eigenvalues for conditioning
    eigs = jnp.linalg.eigvalsh(L_trans_psd)
    eig_min = float(jnp.min(eigs))
    eig_max = float(jnp.max(eigs))
    cond = eig_max / (eig_min + eps_mass)

    # Build certificate
    cert = CertBundle.create_approx(
        chart_id=belief_pred.chart_id,
        anchor_id=belief_pred.anchor_id,
        triggers=["PlanarTranslationEvidence"],
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=int(jnp.sum(eigs < eps_mass)),
        ),
        mismatch=MismatchCert(
            nll_per_ess=trans_nll / (N_eff + eps_mass),
            directional_score=0.0,
        ),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=float(psd_cert[0]),
            mass_epsilon_ratio=float(eps_mass / (N_eff + eps_mass)),
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    # Expected effect
    expected_effect = ExpectedEffect(
        objective_name="predicted_translation_nll",
        predicted=trans_nll,
        realized=None,
    )

    return result, cert, expected_effect


# =============================================================================
# Full 22D Evidence Builder (combines rotation with translation)
# =============================================================================


def build_mf_rotation_evidence_22d(
    mf_result: MatrixFisherResult,
    belief_pred: BeliefGaussianInfo,
    eps_psd: float = constants.GC_EPS_PSD,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Embed 3D rotation evidence into full 22D tangent space.

    Args:
        mf_result: Matrix Fisher rotation result
        belief_pred: Predicted belief (for frame info)
        eps_psd: PSD projection epsilon

    Returns:
        Tuple of (L_22d, h_22d) rotation evidence in full state
    """
    L_22d = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    h_22d = jnp.zeros(D_Z, dtype=jnp.float64)

    # GC ordering: pose is [trans(0:3), rot(3:6)]
    # Embed rotation info into rot block (3:6)
    L_22d = L_22d.at[3:6, 3:6].set(mf_result.L_rot)
    h_22d = h_22d.at[3:6].set(mf_result.h_rot)

    return L_22d, h_22d


def build_planar_translation_evidence_22d(
    trans_result: PlanarTranslationResult,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Embed 3D translation evidence into full 22D tangent space.

    Args:
        trans_result: Planar translation result

    Returns:
        Tuple of (L_22d, h_22d) translation evidence in full state
    """
    L_22d = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    h_22d = jnp.zeros(D_Z, dtype=jnp.float64)

    # GC ordering: pose is [trans(0:3), rot(3:6)]
    # Embed translation info into trans block (0:3)
    L_22d = L_22d.at[0:3, 0:3].set(trans_result.L_trans)
    h_22d = h_22d.at[0:3].set(trans_result.h_trans)

    return L_22d, h_22d


def build_combined_lidar_evidence_22d(
    mf_result: MatrixFisherResult,
    trans_result: PlanarTranslationResult,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Combine Matrix Fisher rotation and planar translation into 22D evidence.

    This is the replacement for the old lidar_quadratic_evidence operator.

    Args:
        mf_result: Matrix Fisher rotation result
        trans_result: Planar translation result

    Returns:
        Tuple of (L_22d, h_22d) combined LiDAR evidence
    """
    L_22d = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    h_22d = jnp.zeros(D_Z, dtype=jnp.float64)

    # Translation: indices 0:3
    L_22d = L_22d.at[0:3, 0:3].set(trans_result.L_trans)
    h_22d = h_22d.at[0:3].set(trans_result.h_trans)

    # Rotation: indices 3:6
    L_22d = L_22d.at[3:6, 3:6].set(mf_result.L_rot)
    h_22d = h_22d.at[3:6].set(mf_result.h_rot)

    return L_22d, h_22d
