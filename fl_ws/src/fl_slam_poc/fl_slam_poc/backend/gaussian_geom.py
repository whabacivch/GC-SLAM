"""
Gaussian and SE(3) Frobenius correction utilities.

For the full Gaussian family, the cubic tensor C = ∇³ψ is zero in natural
coordinates (see Comprehensive Information Geometry.md). This means the
third-order Frobenius correction is an identity (no-op) for pure Gaussian updates.

HOWEVER, for SE(3) pose parameters, the manifold has non-zero curvature and
the Baker-Campbell-Hausdorff (BCH) formula provides the third-order correction.
This is the "pre-Frobenius" correction for manifold retraction.

**Philosophy Alignment:**
- Tangent space operations + retraction is the correct pattern for manifolds
- BCH third-order term corrects linearization error from exp(a)exp(b) ≈ exp(a+b)
- This is NOT a heuristic; it's the geometric definition of manifold operations

We emit proof-of-execution metadata so approximation triggers remain
auditable and compliant with the Frobenius policy.

References:
- Barfoot (2017): State Estimation for Robotics (BCH formula)
- Sola et al. (2018): A micro Lie theory for state estimation
- Combe (2022-2025): Pre-Frobenius manifolds and information geometry
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


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector (hat operator)."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)


def se3_tangent_frobenius_correction(
    delta: np.ndarray,
    state_uncertainty: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """
    Apply Frobenius (BCH third-order) correction for SE(3) tangent deltas.
    
    For SE(3), the manifold retraction exp: se(3) → SE(3) is non-commutative.
    The BCH formula gives:
        log(exp(a)exp(b)) = a + b + ½[a,b] + ¹⁄₁₂([a,[a,b]] + [b,[b,a]]) + ...
    
    The third-order correction term captures the ½[a,b] bracket contribution.
    This is applied before retraction to reduce linearization error.
    
    **Information-Geometric Interpretation:**
    - The bracket [a,b] represents manifold curvature (non-flat geometry)
    - Correction magnitude scales with uncertainty (larger deltas need more correction)
    - This transforms "predict in tangent, then retract" to proper geodesic update
    
    Args:
        delta: 6D SE(3) tangent vector [tx, ty, tz, rx, ry, rz] (translation then rotation)
               OR 9D IMU tangent vector [px, py, pz, vx, vy, vz, rx, ry, rz]
               OR 15D full state tangent vector
        state_uncertainty: Optional covariance diagonal for adaptive correction scaling.
                          If provided, correction is weighted by uncertainty.
    
    Returns:
        delta_corr: corrected tangent delta
        stats: dict with correction magnitude and diagnostics
    """
    d = np.asarray(delta, dtype=float).reshape(-1)
    n_dim = len(d)
    
    # Extract rotation component based on dimensionality
    if n_dim >= 6:
        if n_dim == 6:
            # SE(3): [tx, ty, tz, rx, ry, rz]
            trans = d[:3]
            rot = d[3:6]
        elif n_dim == 9:
            # IMU tangent: [px, py, pz, vx, vy, vz, rx, ry, rz]
            rot = d[6:9]
            trans = d[:3]  # Position delta
        else:
            # Full 15D state: [x, y, z, rx, ry, rz, vx, vy, vz, bg1, bg2, bg3, ba1, ba2, ba3]
            rot = d[3:6]
            trans = d[:3]
    else:
        # Not SE(3) format, return uncorrected
        delta_out = d.copy()
        return delta_out, {
            "delta_norm": float(np.linalg.norm(d)),
            "correction_applied": False,
            "reason": "not_se3_format",
            "input_stats": {"delta": _vec_stats(d)},
            "output_stats": {"delta_corr": _vec_stats(delta_out)},
        }
    
    # Compute BCH second-order bracket term: ½[rot, trans]
    # For se(3), the bracket involves rotation-translation coupling
    # [ω, v] = ω × v (cross product)
    rot_norm = float(np.linalg.norm(rot))
    
    # Only apply correction if rotation is non-trivial
    if rot_norm < 1e-10:
        delta_out = d.copy()
        return delta_out, {
            "delta_norm": float(np.linalg.norm(d)),
            "correction_applied": False,
            "reason": "rotation_too_small",
            "rot_norm": rot_norm,
            "input_stats": {"delta": _vec_stats(d)},
            "output_stats": {"delta_corr": _vec_stats(delta_out)},
        }
    
    # BCH second-order term: ½ [ω, v] = ½ ω × v
    # This captures the non-commutativity of rotation and translation
    bracket_rv = 0.5 * np.cross(rot, trans)
    
    # BCH third-order term (Lie bracket of bracket): ¹⁄₁₂([ω,[ω,v]] + [v,[v,ω]])
    # For SO(3): [ω,[ω,v]] = ω × (ω × v)
    # Simplified: ¹⁄₁₂ * ω × (ω × v)
    bracket_rr_v = (1.0/12.0) * np.cross(rot, np.cross(rot, trans))
    
    # Total correction for translation
    trans_correction = bracket_rv + bracket_rr_v
    
    # Apply uncertainty weighting if provided
    correction_scale = 1.0
    if state_uncertainty is not None:
        sigma = np.asarray(state_uncertainty, dtype=float).reshape(-1)
        if len(sigma) >= 6:
            # Scale correction by ratio of rotation uncertainty to position uncertainty
            # Larger uncertainty → smaller correction (conservative)
            rot_unc = float(np.mean(sigma[3:6])) if len(sigma) >= 6 else 1.0
            trans_unc = float(np.mean(sigma[:3])) if len(sigma) >= 3 else 1.0
            if rot_unc > 1e-12:
                correction_scale = min(1.0, trans_unc / (rot_unc + 1e-12))
    
    trans_correction = trans_correction * correction_scale
    
    # Build corrected delta
    delta_corr = d.copy()
    delta_corr[:3] = trans[:3] + trans_correction
    
    correction_norm = float(np.linalg.norm(trans_correction))
    
    stats = {
        "delta_norm": float(np.linalg.norm(d)),
        "correction_applied": True,
        "correction_norm": correction_norm,
        "correction_scale": correction_scale,
        "rot_norm": rot_norm,
        "trans_norm": float(np.linalg.norm(trans)),
        "input_stats": {"delta": _vec_stats(d)},
        "output_stats": {"delta_corr": _vec_stats(delta_corr)},
    }
    
    return delta_corr, stats


def imu_tangent_frobenius_correction(
    delta_p: np.ndarray,
    delta_v: np.ndarray,
    delta_rotvec: np.ndarray,
    cov_preint: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Apply Frobenius (BCH third-order) correction for IMU preintegration tangent deltas.
    
    This is the specialized form for IMU factors where we have separate
    position, velocity, and rotation deltas. The correction accounts for
    rotation-position and rotation-velocity coupling.
    
    **Tangent-Space Correct Pattern:**
    1. Preintegration computes deltas in tangent space
    2. This function applies BCH correction for manifold curvature
    3. Caller then retracts corrected deltas to manifold
    
    Args:
        delta_p: Position delta (3D) in body frame
        delta_v: Velocity delta (3D) in body frame  
        delta_rotvec: Rotation delta (3D) as rotation vector
        cov_preint: Optional 9x9 preintegration covariance for adaptive scaling
    
    Returns:
        delta_p_corr: Corrected position delta
        delta_v_corr: Corrected velocity delta
        delta_rotvec_corr: Corrected rotation delta (unchanged, correction is in trans/vel)
        stats: Diagnostics dict
    """
    dp = np.asarray(delta_p, dtype=float).reshape(-1)
    dv = np.asarray(delta_v, dtype=float).reshape(-1)
    dr = np.asarray(delta_rotvec, dtype=float).reshape(-1)
    
    rot_norm = float(np.linalg.norm(dr))
    
    # Skip correction if rotation is tiny
    if rot_norm < 1e-10:
        delta_full = np.concatenate([dp, dv, dr])
        return (
            dp.copy(), 
            dv.copy(), 
            dr.copy(), 
            {
                "correction_applied": False,
                "reason": "rotation_too_small",
                "rot_norm": rot_norm,
                # Backward-compatible fields for OpReport
                "delta_norm": 0.0,
                "input_stats": {"delta": _vec_stats(delta_full)},
                "output_stats": {"delta_corr": _vec_stats(delta_full)},
            }
        )
    
    # BCH second-order bracket terms
    # Position: ½ [ω, Δp] = ½ ω × Δp
    bracket_r_p = 0.5 * np.cross(dr, dp)
    
    # Velocity: ½ [ω, Δv] = ½ ω × Δv
    bracket_r_v = 0.5 * np.cross(dr, dv)
    
    # BCH third-order terms (smaller, but included for completeness)
    bracket_rr_p = (1.0/12.0) * np.cross(dr, np.cross(dr, dp))
    bracket_rr_v = (1.0/12.0) * np.cross(dr, np.cross(dr, dv))
    
    # Adaptive scaling based on preintegration covariance
    correction_scale_p = 1.0
    correction_scale_v = 1.0
    if cov_preint is not None:
        cov = np.asarray(cov_preint, dtype=float)
        if cov.shape == (9, 9):
            # Extract diagonal uncertainties
            pos_unc = float(np.mean(np.diag(cov)[:3])) if cov.shape[0] >= 3 else 1.0
            vel_unc = float(np.mean(np.diag(cov)[3:6])) if cov.shape[0] >= 6 else 1.0
            rot_unc = float(np.mean(np.diag(cov)[6:9])) if cov.shape[0] >= 9 else 1.0
            
            # Higher position/velocity uncertainty → more conservative correction
            # (uncertainty in input means correction might be wrong direction)
            if pos_unc > 1e-12:
                correction_scale_p = min(1.0, 1.0 / (1.0 + pos_unc))
            if vel_unc > 1e-12:
                correction_scale_v = min(1.0, 1.0 / (1.0 + vel_unc))
    
    # Apply corrections
    dp_correction = (bracket_r_p + bracket_rr_p) * correction_scale_p
    dv_correction = (bracket_r_v + bracket_rr_v) * correction_scale_v
    
    dp_corr = dp + dp_correction
    dv_corr = dv + dv_correction
    dr_corr = dr.copy()  # Rotation unchanged (correction is in translation/velocity)
    
    # Build concatenated tangent vector for stats
    delta_full = np.concatenate([dp, dv, dr])
    delta_corr_full = np.concatenate([dp_corr, dv_corr, dr_corr])
    
    stats = {
        "correction_applied": True,
        "rot_norm": rot_norm,
        "dp_correction_norm": float(np.linalg.norm(dp_correction)),
        "dv_correction_norm": float(np.linalg.norm(dv_correction)),
        "correction_scale_p": correction_scale_p,
        "correction_scale_v": correction_scale_v,
        # Backward-compatible fields for OpReport
        "delta_norm": float(np.linalg.norm(delta_corr_full - delta_full)),
        "input_stats": {"delta": _vec_stats(delta_full)},
        "output_stats": {"delta_corr": _vec_stats(delta_corr_full)},
    }
    
    return dp_corr, dv_corr, dr_corr, stats

