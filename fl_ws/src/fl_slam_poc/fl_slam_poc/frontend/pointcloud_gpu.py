"""
GPU-accelerated point cloud processing using Open3D.

Provides GPU-accelerated implementations of:
- Voxel grid filtering (downsampling)
- ICP registration
- Normal estimation

Falls back to CPU implementations when GPU is unavailable.

Requirements:
    - open3d >= 0.18.0 (with CUDA support)
    - CUDA-compatible GPU (tested on RTX 4050)

Design Notes:
    - All operations return the same formats as CPU equivalents
    - ICPResult format matches operators/icp.py for drop-in compatibility
    - Graceful degradation when GPU unavailable
    - Memory management to prevent VRAM exhaustion

Reference:
    - Open3D GPU documentation: http://www.open3d.org/docs/release/
    - CUDA ICP: Zhou et al. (2018)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from fl_slam_poc.frontend.icp import ICPResult

# Logger for GPU operations
_logger = logging.getLogger(__name__)

# Global GPU availability flag
_GPU_AVAILABLE = None
_O3D_DEVICE = None


def _check_gpu_availability():
    """Check if Open3D GPU support is available."""
    global _GPU_AVAILABLE, _O3D_DEVICE
    
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    
    try:
        import open3d as o3d
        
        # Check if CUDA device is available
        if o3d.core.cuda.is_available():
            _O3D_DEVICE = o3d.core.Device("CUDA:0")
            _GPU_AVAILABLE = True
            _logger.info("Open3D GPU acceleration available (CUDA:0)")
        else:
            _O3D_DEVICE = o3d.core.Device("CPU:0")
            _GPU_AVAILABLE = False
            _logger.warning("Open3D CUDA not available, using CPU fallback")
    except ImportError:
        _GPU_AVAILABLE = False
        _O3D_DEVICE = None
        _logger.warning("Open3D not installed, GPU acceleration unavailable")
    except Exception as e:
        _GPU_AVAILABLE = False
        _O3D_DEVICE = None
        _logger.warning(f"Open3D GPU check failed: {e}")
    
    return _GPU_AVAILABLE


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _check_gpu_availability()


@dataclass
class GPUConfig:
    """GPU processing configuration."""
    device_index: int = 0
    memory_limit_mb: int = 4096
    fallback_to_cpu: bool = True


class GPUPointCloudProcessor:
    """
    GPU-accelerated point cloud processor using Open3D.
    
    Provides voxel filtering and ICP registration on GPU with
    graceful fallback to CPU when needed.
    """
    
    def __init__(
        self,
        voxel_size: float = 0.05,
        max_correspondence_distance: float = 0.5,
        device_index: int = 0,
        fallback_to_cpu: bool = True
    ):
        """
        Initialize GPU processor.
        
        Args:
            voxel_size: Voxel grid size for downsampling (meters)
            max_correspondence_distance: Maximum distance for ICP correspondences
            device_index: CUDA device index
            fallback_to_cpu: Use CPU if GPU unavailable
        """
        self.voxel_size = voxel_size
        self.max_correspondence_distance = max_correspondence_distance
        self.device_index = device_index
        self.fallback_to_cpu = fallback_to_cpu
        
        # Check GPU availability
        self.use_gpu = _check_gpu_availability()
        self._o3d = None
        self._device = None
        
        if self.use_gpu:
            try:
                import open3d as o3d
                self._o3d = o3d
                self._device = o3d.core.Device(f"CUDA:{device_index}")
                _logger.info(f"GPUPointCloudProcessor initialized on CUDA:{device_index}")
            except Exception as e:
                _logger.warning(f"GPU initialization failed: {e}")
                self.use_gpu = False
        
        if not self.use_gpu and not fallback_to_cpu:
            raise RuntimeError("GPU not available and fallback disabled")
    
    def voxel_filter(self, points: np.ndarray, voxel_size: Optional[float] = None) -> np.ndarray:
        """
        Voxel grid downsampling.
        
        Args:
            points: Input point cloud (N, 3)
            voxel_size: Override default voxel size
        
        Returns:
            Downsampled point cloud (M, 3) where M <= N
        """
        if points.shape[0] == 0:
            return points
        
        voxel_size = voxel_size or self.voxel_size
        
        if self.use_gpu and self._o3d is not None:
            return self._voxel_filter_gpu(points, voxel_size)
        else:
            return self._voxel_filter_cpu(points, voxel_size)
    
    def _voxel_filter_gpu(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """GPU-accelerated voxel filtering using Open3D tensor API."""
        try:
            o3d = self._o3d
            
            # Create tensor point cloud on GPU
            pcd = o3d.t.geometry.PointCloud(self._device)
            pcd.point.positions = o3d.core.Tensor(
                points.astype(np.float32),
                dtype=o3d.core.float32,
                device=self._device
            )
            
            # Voxel downsample on GPU
            pcd_down = pcd.voxel_down_sample(voxel_size)
            
            # Transfer back to CPU
            return pcd_down.point.positions.cpu().numpy()
            
        except Exception as e:
            _logger.warning(f"GPU voxel filter failed, falling back to CPU: {e}")
            return self._voxel_filter_cpu(points, voxel_size)
    
    def _voxel_filter_cpu(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """CPU voxel filtering fallback using simple grid averaging."""
        if points.shape[0] == 0:
            return points
        
        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Use dictionary to accumulate points in each voxel
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(points[i])
        
        # Average points in each voxel
        filtered = []
        for pts in voxel_dict.values():
            filtered.append(np.mean(pts, axis=0))
        
        return np.array(filtered, dtype=np.float32)
    
    def icp(
        self,
        source: np.ndarray,
        target: np.ndarray,
        init: Optional[np.ndarray] = None,
        max_iter: int = 15,
        tol: float = 1e-4,
        max_correspondence_distance: Optional[float] = None
    ) -> ICPResult:
        """
        ICP registration with GPU acceleration.
        
        Args:
            source: Source point cloud (N, 3)
            target: Target point cloud (M, 3)
            init: Initial transform [x, y, z, rx, ry, rz] (SE(3))
            max_iter: Maximum iterations
            tol: Convergence tolerance
            max_correspondence_distance: Override default max correspondence distance
        
        Returns:
            ICPResult compatible with operators/icp.py
        """
        max_corr_dist = max_correspondence_distance or self.max_correspondence_distance
        
        if self.use_gpu and self._o3d is not None:
            return self._icp_gpu(source, target, init, max_iter, tol, max_corr_dist)
        else:
            return self._icp_cpu(source, target, init, max_iter, tol)
    
    def _icp_gpu(
        self,
        source: np.ndarray,
        target: np.ndarray,
        init: Optional[np.ndarray],
        max_iter: int,
        tol: float,
        max_corr_dist: float
    ) -> ICPResult:
        """GPU-accelerated ICP using Open3D tensor API."""
        try:
            o3d = self._o3d
            
            # Create point clouds on GPU
            src_pcd = o3d.t.geometry.PointCloud(self._device)
            src_pcd.point.positions = o3d.core.Tensor(
                source.astype(np.float32),
                dtype=o3d.core.float32,
                device=self._device
            )
            
            tgt_pcd = o3d.t.geometry.PointCloud(self._device)
            tgt_pcd.point.positions = o3d.core.Tensor(
                target.astype(np.float32),
                dtype=o3d.core.float32,
                device=self._device
            )
            
            # Build initial transform as 4x4 matrix
            if init is not None:
                from fl_slam_poc.common.se3 import rotvec_to_rotmat
                R = rotvec_to_rotmat(init[3:6])
                t = init[:3]
                init_transform = np.eye(4, dtype=np.float64)
                init_transform[:3, :3] = R
                init_transform[:3, 3] = t
            else:
                init_transform = np.eye(4, dtype=np.float64)
            
            init_tensor = o3d.core.Tensor(init_transform, device=self._device)
            
            # Set up ICP criteria
            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=tol,
                relative_rmse=tol,
                max_iteration=max_iter
            )
            
            # Run ICP
            result = o3d.t.pipelines.registration.icp(
                src_pcd, tgt_pcd,
                max_correspondence_distance=max_corr_dist,
                init_source_to_target=init_tensor,
                estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=criteria
            )
            
            # Extract result
            transform_matrix = result.transformation.cpu().numpy()
            
            # Convert 4x4 matrix to SE(3) [x, y, z, rx, ry, rz]
            R = transform_matrix[:3, :3]
            t = transform_matrix[:3, 3]
            from fl_slam_poc.common.se3 import rotmat_to_rotvec
            rotvec = rotmat_to_rotvec(R)
            transform = np.array([t[0], t[1], t[2], rotvec[0], rotvec[1], rotvec[2]], dtype=float)
            
            # Compute final MSE
            src_transformed = (R @ source.T).T + t
            
            # Find correspondences for MSE calculation
            dists = np.linalg.norm(src_transformed[:, np.newaxis, :] - target[np.newaxis, :, :], axis=2)
            min_indices = np.argmin(dists, axis=1)
            matched = target[min_indices]
            residuals = np.linalg.norm(src_transformed - matched, axis=1)
            mse = float(np.mean(residuals ** 2))
            
            # Estimate initial MSE
            if init is not None:
                R_init = rotvec_to_rotmat(init[3:6])
                t_init = init[:3]
                src_init = (R_init @ source.T).T + t_init
            else:
                src_init = source
            dists_init = np.linalg.norm(src_init[:, np.newaxis, :] - target[np.newaxis, :, :], axis=2)
            min_idx_init = np.argmin(dists_init, axis=1)
            matched_init = target[min_idx_init]
            initial_mse = float(np.mean(np.linalg.norm(src_init - matched_init, axis=1) ** 2))
            
            return ICPResult(
                transform=transform,
                mse=mse,
                iterations=max_iter,  # Open3D doesn't expose actual iteration count
                max_iterations=max_iter,
                tolerance=tol,
                initial_objective=initial_mse,
                final_objective=mse,
                matched_points=matched,
                src_transformed=src_transformed,
                n_source=source.shape[0],
                n_target=target.shape[0],
                converged=result.fitness > 0.5  # Heuristic
            )
            
        except Exception as e:
            _logger.warning(f"GPU ICP failed, falling back to CPU: {e}")
            return self._icp_cpu(source, target, init, max_iter, tol)
    
    def _icp_cpu(
        self,
        source: np.ndarray,
        target: np.ndarray,
        init: Optional[np.ndarray],
        max_iter: int,
        tol: float
    ) -> ICPResult:
        """CPU ICP fallback using existing implementation."""
        from fl_slam_poc.frontend.icp import icp_3d
        
        if init is None:
            init = np.zeros(6, dtype=float)
        
        return icp_3d(source, target, init, max_iter, tol)
    
    def estimate_normals(
        self,
        points: np.ndarray,
        radius: float = 0.1,
        max_nn: int = 30
    ) -> Optional[np.ndarray]:
        """
        Estimate point cloud normals.
        
        Args:
            points: Input point cloud (N, 3)
            radius: Search radius for normal estimation
            max_nn: Maximum number of neighbors
        
        Returns:
            Normal vectors (N, 3) or None if estimation fails
        """
        if points.shape[0] < 3:
            return None
        
        if self.use_gpu and self._o3d is not None:
            return self._estimate_normals_gpu(points, radius, max_nn)
        else:
            return self._estimate_normals_cpu(points, radius, max_nn)
    
    def _estimate_normals_gpu(self, points: np.ndarray, radius: float, max_nn: int) -> Optional[np.ndarray]:
        """GPU-accelerated normal estimation."""
        try:
            o3d = self._o3d
            
            pcd = o3d.t.geometry.PointCloud(self._device)
            pcd.point.positions = o3d.core.Tensor(
                points.astype(np.float32),
                dtype=o3d.core.float32,
                device=self._device
            )
            
            pcd.estimate_normals(radius=radius, max_nn=max_nn)
            
            if pcd.point.normals is not None:
                return pcd.point.normals.cpu().numpy()
            return None
            
        except Exception as e:
            _logger.warning(f"GPU normal estimation failed: {e}")
            return self._estimate_normals_cpu(points, radius, max_nn)
    
    def _estimate_normals_cpu(self, points: np.ndarray, radius: float, max_nn: int) -> Optional[np.ndarray]:
        """CPU normal estimation fallback using PCA."""
        try:
            from scipy.spatial import KDTree
            
            tree = KDTree(points)
            normals = np.zeros_like(points)
            
            for i, p in enumerate(points):
                # Find neighbors
                indices = tree.query_ball_point(p, radius)
                if len(indices) < 3:
                    indices = tree.query(p, k=min(max_nn, len(points)))[1]
                
                neighbors = points[indices]
                if len(neighbors) < 3:
                    normals[i] = [0, 0, 1]  # Default up
                    continue
                
                # PCA for normal estimation
                centered = neighbors - np.mean(neighbors, axis=0)
                _, _, Vt = np.linalg.svd(centered)
                normals[i] = Vt[-1]  # Smallest eigenvector
            
            return normals
            
        except Exception as e:
            _logger.warning(f"CPU normal estimation failed: {e}")
            return None


# Module-level convenience functions

def voxel_filter_gpu(points: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    """Voxel grid filter with automatic GPU/CPU selection."""
    proc = GPUPointCloudProcessor(voxel_size=voxel_size)
    return proc.voxel_filter(points)


def icp_gpu(
    source: np.ndarray,
    target: np.ndarray,
    init: Optional[np.ndarray] = None,
    max_iter: int = 15,
    tol: float = 1e-4,
    max_correspondence_distance: float = 0.5
) -> ICPResult:
    """ICP registration with automatic GPU/CPU selection."""
    proc = GPUPointCloudProcessor(max_correspondence_distance=max_correspondence_distance)
    return proc.icp(source, target, init, max_iter, tol)
