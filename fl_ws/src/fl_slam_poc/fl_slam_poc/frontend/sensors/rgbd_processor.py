"""
RGB-D evidence extraction for dense 3D Gaussian modules.

Extracts from synchronized RGB-D image pairs:
- 3D positions (depth backprojection using camera intrinsics)
- Surface normals (depth gradients → vMF evidence)
- Colors (RGB values → Gaussian evidence)
- Covariances (depth uncertainty model)

All operations produce exponential family evidence suitable for
information-form fusion with laser anchors.

Reference: Hybrid Laser + RGB-D Sensor Fusion Architecture
"""

import numpy as np
from typing import Tuple, Optional, List, Dict

from fl_slam_poc.backend.fusion.gaussian_info import make_evidence
from fl_slam_poc.common.geometry.vmf import vmf_make_evidence


def depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    subsample: int = 10,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Convert depth image to 3D point cloud with normals.
    
    Back-projects depth pixels to 3D points using the pinhole camera model.
    Computes surface normals from local depth gradients.
    
    Args:
        depth: (H, W) depth image in meters (float32)
        K: (3, 3) camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        rgb: (H, W, 3) RGB image (uint8 or float, optional)
        subsample: Spatial stride for efficiency (default 10 = 1/100th of pixels)
        min_depth: Minimum valid depth in meters
        max_depth: Maximum valid depth in meters
    
    Returns:
        (points, colors, normals, covs): 
            - points: (N, 3) 3D positions in camera frame
            - colors: (N, 3) RGB colors normalized to [0, 1]
            - normals: (N, 3) unit surface normals
            - covs: List of N (3, 3) covariance matrices
    
    Example:
        >>> depth = np.ones((480, 640), dtype=np.float32) * 2.0  # 2m flat wall
        >>> K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        >>> points, colors, normals, covs = depth_to_pointcloud(depth, K, subsample=50)
        >>> print(f"Extracted {len(points)} points")
    """
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)
    
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")
    
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Subsample grid
    u = np.arange(0, W, subsample)
    v = np.arange(0, H, subsample)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Sample depth at grid points
    z = depth[v_grid, u_grid]
    
    # Valid depth mask
    valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    
    if not np.any(valid):
        # No valid points
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            []
        )
    
    u_valid = u_grid[valid].astype(np.float64)
    v_valid = v_grid[valid].astype(np.float64)
    z_valid = z[valid].astype(np.float64)
    
    # Back-project to 3D (pinhole camera model)
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    points = np.stack([x, y, z_valid], axis=-1)
    
    # Extract colors
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.ndim == 3 and rgb.shape[2] >= 3:
            # Handle both uint8 [0, 255] and float [0, 1]
            if rgb.dtype == np.uint8:
                colors = rgb[v_grid[valid], u_grid[valid], :3].astype(np.float64) / 255.0
            else:
                colors = rgb[v_grid[valid], u_grid[valid], :3].astype(np.float64)
                colors = np.clip(colors, 0.0, 1.0)
        else:
            colors = np.ones((len(points), 3), dtype=np.float64) * 0.5
    else:
        colors = np.ones((len(points), 3), dtype=np.float64) * 0.5
    
    # Compute normals via local plane fitting
    normals = compute_normals_from_depth(depth, K, subsample)
    normals_valid = normals[valid.ravel()]
    
    # Covariance model: depth uncertainty scales with z²
    # Stereo/ToF model: σ_depth ∝ z (1% of depth is reasonable)
    sigma_depth = 0.01 * z_valid  # 1% depth uncertainty
    sigma_lateral = sigma_depth * z_valid / fx  # Lateral uncertainty from depth projection
    
    covs = []
    for i in range(len(points)):
        cov = np.diag([
            sigma_lateral[i]**2,
            sigma_lateral[i]**2,
            sigma_depth[i]**2
        ])
        covs.append(cov)
    
    return points, colors, normals_valid, covs


def compute_normals_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    subsample: int = 10
) -> np.ndarray:
    """
    Compute surface normals from depth via local gradients.
    
    Uses the cross-product of tangent vectors derived from depth gradients.
    Normals point toward the camera (positive Z).
    
    Args:
        depth: (H, W) depth in meters
        K: (3, 3) camera intrinsics
        subsample: Spatial stride (must match depth_to_pointcloud)
    
    Returns:
        normals: (H//subsample * W//subsample, 3) unit normals (flattened)
    """
    depth = np.asarray(depth, dtype=np.float64)
    
    # Compute depth gradients (central difference)
    # grad_x[i,j] = (depth[i,j+1] - depth[i,j-1]) / 2
    grad_x = np.gradient(depth, axis=1)
    grad_y = np.gradient(depth, axis=0)
    
    # Subsample gradients
    grad_x = grad_x[::subsample, ::subsample]
    grad_y = grad_y[::subsample, ::subsample]
    depth_sub = depth[::subsample, ::subsample]
    
    fx, fy = K[0, 0], K[1, 1]
    
    # Tangent vectors in camera frame
    # t_x = partial(P) / partial(u) where P = [u*z/fx, v*z/fy, z]
    # For fixed pixel, varying depth: t_x ≈ [1/fx, 0, dz/du]
    t_x = np.stack([
        np.ones_like(grad_x) / fx,
        np.zeros_like(grad_x),
        grad_x
    ], axis=-1)
    
    t_y = np.stack([
        np.zeros_like(grad_y),
        np.ones_like(grad_y) / fy,
        grad_y
    ], axis=-1)
    
    # Normal = t_x × t_y (points toward camera)
    normals = np.cross(t_x, t_y, axis=-1)
    
    # Normalize
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    normals = normals / norms
    
    # Ensure normals point toward camera (positive Z component)
    # For a wall facing the camera, normal should be [0, 0, -1] (pointing at camera)
    # But cross product gives [0, 0, +1], so we flip
    normals = -normals
    
    # Flatten spatial dims (row-major)
    normals_flat = normals.reshape(-1, 3)
    return normals_flat


def rgbd_to_evidence(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    covs: List[np.ndarray],
    kappa_normal: float = 10.0,
    color_var: float = 0.01,
    alpha_mean: float = 1.0,
    alpha_var: float = 0.1,
) -> List[Dict]:
    """
    Convert RGB-D observations to exponential family evidence.
    
    Each point produces evidence containing:
    - Position: 3D Gaussian in information form (L, h)
    - Color: 3D Gaussian for RGB
    - Normal: vMF natural parameter (θ = κμ)
    - Opacity: Scalar Gaussian (α)
    
    Args:
        points: (N, 3) 3D positions
        colors: (N, 3) RGB colors in [0, 1]
        normals: (N, 3) unit surface normals
        covs: List of N (3, 3) position covariances
        kappa_normal: vMF concentration for normals (higher = more confident)
        color_var: Variance for RGB Gaussian (smaller = more confident)
        alpha_mean: Default opacity (1.0 = fully opaque)
        alpha_var: Opacity variance
    
    Returns:
        List of evidence dicts, each containing:
        - "position_L": (3, 3) precision matrix
        - "position_h": (3,) information vector
        - "color_L": (3, 3) precision matrix
        - "color_h": (3,) information vector
        - "normal_theta": (3,) vMF natural parameter
        - "alpha_mean": float
        - "alpha_var": float
        - "point_3d": (3,) original 3D point
    """
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    
    if len(points) != len(colors) or len(points) != len(normals) or len(points) != len(covs):
        raise ValueError(
            f"Array length mismatch: points={len(points)}, colors={len(colors)}, "
            f"normals={len(normals)}, covs={len(covs)}"
        )
    
    evidence_list = []
    
    for i in range(len(points)):
        # Position (3D Gaussian)
        L, h = make_evidence(points[i], covs[i])
        
        # Color (3D Gaussian with uniform uncertainty)
        color_cov = np.eye(3, dtype=np.float64) * color_var
        L_color, h_color = make_evidence(colors[i], color_cov)
        
        # Normal (vMF with specified concentration)
        normal_unit = normals[i]
        norm = np.linalg.norm(normal_unit)
        if norm > 1e-8:
            normal_unit = normal_unit / norm
        else:
            normal_unit = np.array([0.0, 0.0, 1.0])  # Default: pointing up
        
        theta_normal = vmf_make_evidence(normal_unit, kappa_normal, d=3)
        
        evidence_list.append({
            "position_L": L,
            "position_h": h,
            "color_L": L_color,
            "color_h": h_color,
            "normal_theta": theta_normal,
            "alpha_mean": float(alpha_mean),
            "alpha_var": float(alpha_var),
            "point_3d": points[i].copy(),
        })
    
    return evidence_list


def transform_evidence_to_global(
    evidence_list: List[Dict],
    T_camera_to_global: np.ndarray
) -> List[Dict]:
    """
    Transform RGB-D evidence from camera frame to global frame.
    
    Args:
        evidence_list: Evidence from rgbd_to_evidence (in camera frame)
        T_camera_to_global: (4, 4) SE(3) transform or (6,) pose [x,y,z,rx,ry,rz]
    
    Returns:
        Evidence list with positions and normals in global frame
    """
    from fl_slam_poc.common.geometry.se3_numpy import rotvec_to_rotmat
    
    T = np.asarray(T_camera_to_global, dtype=np.float64)
    
    if T.shape == (6,):
        # Convert rotation vector to rotation matrix
        t = T[:3]
        R = rotvec_to_rotmat(T[3:])
    elif T.shape == (4, 4):
        R = T[:3, :3]
        t = T[:3, 3]
    else:
        raise ValueError(f"Expected (6,) or (4,4) transform, got {T.shape}")
    
    transformed = []
    for ev in evidence_list:
        # Transform position
        point_global = R @ ev["point_3d"] + t
        
        # Transform covariance: Σ' = R Σ R^T
        cov_local = np.linalg.inv(ev["position_L"])
        cov_global = R @ cov_local @ R.T
        
        L_global, h_global = make_evidence(point_global, cov_global)
        
        # Transform normal direction
        normal_theta = ev["normal_theta"]
        kappa = np.linalg.norm(normal_theta)
        if kappa > 1e-10:
            normal_dir = normal_theta / kappa
            normal_dir_global = R @ normal_dir
            theta_global = vmf_make_evidence(normal_dir_global, kappa, d=3)
        else:
            theta_global = normal_theta.copy()
        
        transformed.append({
            "position_L": L_global,
            "position_h": h_global,
            "color_L": ev["color_L"],  # Color doesn't depend on frame
            "color_h": ev["color_h"],
            "normal_theta": theta_global,
            "alpha_mean": ev["alpha_mean"],
            "alpha_var": ev["alpha_var"],
            "point_3d": point_global,
        })
    
    return transformed


def subsample_evidence_spatially(
    evidence_list: List[Dict],
    grid_size: float = 0.1,
    max_points: int = 5000,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Spatially subsample evidence to reduce density.
    
    Uses voxel grid filtering: keep one point per grid cell.
    
    Args:
        evidence_list: Full evidence list
        grid_size: Voxel grid size in meters
        max_points: Maximum points to keep
    
    Returns:
        Subsampled evidence list
    """
    if len(evidence_list) == 0:
        return []
    
    # Extract positions
    positions = np.array([ev["point_3d"] for ev in evidence_list])
    
    # Voxel indices
    voxel_indices = np.floor(positions / grid_size).astype(int)
    
    # Use dict to keep one point per voxel (last one wins)
    voxel_to_idx = {}
    for i, vox_idx in enumerate(voxel_indices):
        key = tuple(vox_idx)
        voxel_to_idx[key] = i
    
    # Get unique indices
    unique_indices = list(voxel_to_idx.values())
    
    # Limit to max_points
    if len(unique_indices) > max_points:
        # Random sample
        if rng is None:
            rng = np.random.default_rng(seed)
        rng.shuffle(unique_indices)
        unique_indices = unique_indices[:max_points]
    
    return [evidence_list[i] for i in unique_indices]
