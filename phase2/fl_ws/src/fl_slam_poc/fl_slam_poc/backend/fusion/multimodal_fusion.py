"""
Multi-modal sensor fusion in information form.

Fuses heterogeneous sensor evidence (laser 2D, RGB-D 3D) via natural
parameter addition. Handles dimensional mismatch by lifting lower-dimensional
constraints to higher-dimensional spaces with appropriate priors.

Core operations:
- laser_2d_to_3d_constraint: Lift 2D laser evidence to 3D with weak Z prior
- fuse_laser_rgbd: Fuse laser 2D and RGB-D 3D position evidence

All operations are EXACT (closed-form information addition) and preserve
the associative/commutative properties required by the design invariants.

Reference: Hybrid Laser + RGB-D Sensor Fusion Architecture (Development Log)
"""

import numpy as np
from typing import Tuple

from fl_slam_poc.backend.fusion.gaussian_info import make_evidence, fuse_info, mean_cov
from fl_slam_poc.common.op_report import OpReport


def laser_2d_to_3d_constraint(
    laser_mu_2d: np.ndarray,
    laser_cov_2d: np.ndarray,
    z_prior_mean: float = 0.0,
    z_prior_var: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 2D laser evidence (x, y) to 3D constraint (x, y, z).
    
    Lifts a 2D position belief to 3D by adding a weak prior on the Z dimension.
    This allows laser (which only measures in XY plane) to fuse with RGB-D
    (which provides full 3D) in information form.
    
    The Z dimension gets a weak prior (large variance) so that RGB-D
    evidence dominates the vertical dimension while laser dominates XY.
    
    Args:
        laser_mu_2d: (2,) [x, y] position
        laser_cov_2d: (2, 2) covariance in XY
        z_prior_mean: Default z value (e.g., 0 for ground plane, robot height)
        z_prior_var: Variance on Z prior (large = weak constraint)
    
    Returns:
        (L_3d, h_3d): 3D information form (Lambda, eta)
    
    Example:
        >>> laser_mu = np.array([1.0, 0.5])
        >>> laser_cov = np.eye(2) * 0.1
        >>> L_3d, h_3d = laser_2d_to_3d_constraint(laser_mu, laser_cov)
        >>> mu_3d, cov_3d = mean_cov(L_3d, h_3d)
        >>> print(mu_3d)  # [1.0, 0.5, 0.0]
        >>> print(cov_3d[2, 2])  # ~10.0 (weak Z constraint)
    """
    laser_mu_2d = np.asarray(laser_mu_2d, dtype=float).reshape(-1)
    laser_cov_2d = np.asarray(laser_cov_2d, dtype=float)
    
    if laser_mu_2d.shape[0] != 2:
        raise ValueError(f"Expected 2D mean, got shape {laser_mu_2d.shape}")
    if laser_cov_2d.shape != (2, 2):
        raise ValueError(f"Expected 2x2 covariance, got shape {laser_cov_2d.shape}")
    
    # Embed 2D in 3D
    mu_3d = np.array([laser_mu_2d[0], laser_mu_2d[1], z_prior_mean], dtype=float)
    
    # Covariance: strong XY (from laser), weak Z (prior)
    cov_3d = np.eye(3, dtype=float)
    cov_3d[:2, :2] = laser_cov_2d
    cov_3d[2, 2] = z_prior_var
    
    return make_evidence(mu_3d, cov_3d)


def fuse_laser_rgbd(
    laser_L_2d: np.ndarray,
    laser_h_2d: np.ndarray,
    rgbd_L_3d: np.ndarray,
    rgbd_h_3d: np.ndarray,
    laser_weight: float = 1.0,
    rgbd_weight: float = 1.0,
    z_prior_mean: float = 0.0,
    z_prior_var: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, OpReport]:
    """
    Fuse laser 2D and RGB-D 3D position evidence.
    
    Process:
    1. Lift laser 2D → 3D with weak Z prior
    2. Fuse via information addition (exact)
    3. Return 3D posterior
    
    This is the core multi-modal fusion operation for hybrid SLAM.
    
    Args:
        laser_L_2d: (2, 2) laser precision matrix
        laser_h_2d: (2,) laser information vector
        rgbd_L_3d: (3, 3) RGB-D precision matrix
        rgbd_h_3d: (3,) RGB-D information vector
        laser_weight: Weight for laser evidence (default 1.0)
        rgbd_weight: Weight for RGB-D evidence (default 1.0)
        z_prior_mean: Z prior mean for laser lifting
        z_prior_var: Z prior variance for laser lifting
    
    Returns:
        (L_fused, h_fused, report): 3D fused information form and OpReport
    
    Example:
        >>> # Laser observes (1.0, 0.5) with small uncertainty
        >>> L_laser, h_laser = make_evidence(np.array([1.0, 0.5]), np.eye(2) * 0.1)
        >>> # RGB-D observes (1.0, 0.5, 0.2) with moderate uncertainty
        >>> L_rgbd, h_rgbd = make_evidence(np.array([1.0, 0.5, 0.2]), np.eye(3) * 0.05)
        >>> L_fused, h_fused, report = fuse_laser_rgbd(L_laser, h_laser, L_rgbd, h_rgbd)
        >>> mu_fused, cov_fused = mean_cov(L_fused, h_fused)
        >>> # XY dominated by laser, Z by RGB-D
    """
    laser_L_2d = np.asarray(laser_L_2d, dtype=float)
    laser_h_2d = np.asarray(laser_h_2d, dtype=float).reshape(-1)
    rgbd_L_3d = np.asarray(rgbd_L_3d, dtype=float)
    rgbd_h_3d = np.asarray(rgbd_h_3d, dtype=float).reshape(-1)
    
    # Validate dimensions
    if laser_L_2d.shape != (2, 2):
        raise ValueError(f"Expected 2x2 laser precision, got {laser_L_2d.shape}")
    if laser_h_2d.shape[0] != 2:
        raise ValueError(f"Expected 2D laser info vector, got shape {laser_h_2d.shape}")
    if rgbd_L_3d.shape != (3, 3):
        raise ValueError(f"Expected 3x3 RGB-D precision, got {rgbd_L_3d.shape}")
    if rgbd_h_3d.shape[0] != 3:
        raise ValueError(f"Expected 3D RGB-D info vector, got shape {rgbd_h_3d.shape}")
    
    # Lift laser to 3D
    laser_mu_2d, laser_cov_2d = mean_cov(laser_L_2d, laser_h_2d)
    L_laser_3d, h_laser_3d = laser_2d_to_3d_constraint(
        laser_mu_2d, laser_cov_2d,
        z_prior_mean=z_prior_mean,
        z_prior_var=z_prior_var
    )
    
    # Fuse in 3D (exact additive)
    L_fused, h_fused = fuse_info(
        L_laser_3d, h_laser_3d,
        rgbd_L_3d, rgbd_h_3d,
        weight=rgbd_weight,
        rho=laser_weight
    )
    
    # Compute fused state for OpReport metrics
    mu_fused, cov_fused = mean_cov(L_fused, h_fused)
    
    # OpReport
    report = OpReport(
        name="LaserRGBDFusion",
        exact=True,  # Information addition is exact
        approximation_triggers=[],
        family_in="Gaussian2D+Gaussian3D",
        family_out="Gaussian3D",
        closed_form=True,
        solver_used=None,
        frobenius_applied=False,
        metrics={
            "laser_weight": float(laser_weight),
            "rgbd_weight": float(rgbd_weight),
            "fused_z_uncertainty": float(np.sqrt(cov_fused[2, 2])),
            "fused_xy_uncertainty": float(np.sqrt(0.5 * (cov_fused[0, 0] + cov_fused[1, 1]))),
            "z_prior_var": float(z_prior_var),
        }
    )
    
    return L_fused, h_fused, report


def fuse_multimodal_3d(
    evidence_list: list,
    weights: list = None
) -> Tuple[np.ndarray, np.ndarray, OpReport]:
    """
    Fuse multiple 3D Gaussian evidence sources.
    
    General-purpose multi-modal fusion for 3D position evidence.
    Each evidence source provides (L, h) in 3D information form.
    
    Args:
        evidence_list: List of (L, h) tuples, each 3D information form
        weights: Optional weights for each evidence source
    
    Returns:
        (L_fused, h_fused, report): Combined information form and OpReport
    """
    if len(evidence_list) == 0:
        raise ValueError("Need at least one evidence source")
    
    if weights is None:
        weights = [1.0] * len(evidence_list)
    
    if len(weights) != len(evidence_list):
        raise ValueError("weights must match evidence_list length")
    
    # Start with first evidence
    L_fused, h_fused = evidence_list[0]
    L_fused = np.asarray(L_fused, dtype=float) * weights[0]
    h_fused = np.asarray(h_fused, dtype=float).reshape(-1) * weights[0]
    
    # Accumulate remaining evidence
    for (L, h), w in zip(evidence_list[1:], weights[1:]):
        L = np.asarray(L, dtype=float)
        h = np.asarray(h, dtype=float).reshape(-1)
        L_fused += w * L
        h_fused += w * h
    
    mu_fused, cov_fused = mean_cov(L_fused, h_fused)
    
    report = OpReport(
        name="MultiModal3DFusion",
        exact=True,
        approximation_triggers=[],
        family_in="Gaussian3D",
        family_out="Gaussian3D",
        closed_form=True,
        solver_used=None,
        frobenius_applied=False,
        metrics={
            "n_sources": len(evidence_list),
            "total_weight": float(sum(weights)),
            "position_uncertainty": float(np.sqrt(np.trace(cov_fused) / 3)),
        }
    )
    
    return L_fused, h_fused, report


def spatial_association_weight(
    mu1: np.ndarray,
    mu2: np.ndarray,
    scale: float = 0.5
) -> float:
    """
    Compute spatial association weight based on Euclidean distance.
    
    Uses Gaussian kernel: w = exp(-d² / (2σ²))
    
    This is used for soft association of RGB-D modules to laser anchors.
    
    Args:
        mu1: First position (2D or 3D)
        mu2: Second position (2D or 3D, truncated to match mu1 dimensions)
        scale: Gaussian kernel scale (σ)
    
    Returns:
        Association weight in [0, 1]
    """
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    mu2 = np.asarray(mu2, dtype=float).reshape(-1)
    
    # Use minimum shared dimensions
    d = min(len(mu1), len(mu2))
    mu1 = mu1[:d]
    mu2 = mu2[:d]
    
    dist_sq = float(np.sum((mu1 - mu2) ** 2))
    weight = float(np.exp(-dist_sq / (2.0 * scale * scale)))
    
    return weight
