"""
3D Point Cloud Processing Tests

Tests for:
- PointCloud2 message conversion
- GPU/CPU voxel filtering
- GPU/CPU ICP registration
- SensorIO 3D mode
- LoopProcessor GPU integration
"""

import numpy as np
import pytest

# Frontend loops operators
from fl_slam_poc.frontend.icp import (
    ICPResult,
    icp_3d,
    best_fit_se3,
)
from fl_slam_poc.frontend.pointcloud_gpu import (
    GPUPointCloudProcessor,
    is_gpu_available,
    voxel_filter_gpu,
    icp_gpu,
)

# Geometry (now in common/transforms/)
from fl_slam_poc.common.se3 import (
    rotvec_to_rotmat,
    rotmat_to_rotvec,
    se3_compose,
    se3_inverse,
)


# =============================================================================
# PointCloud2 Conversion Tests
# =============================================================================


class TestPointCloud2Conversion:
    """Test PointCloud2 message to numpy conversion."""

    def test_import_conversion_function(self):
        """Verify pointcloud2_to_array is importable."""
        from fl_slam_poc.frontend.sensor_io import pointcloud2_to_array
        assert callable(pointcloud2_to_array)

    def test_empty_cloud_handling(self):
        """Empty point clouds should be handled gracefully."""
        from fl_slam_poc.frontend.sensor_io import pointcloud2_to_array
        from sensor_msgs.msg import PointCloud2, PointField
        
        msg = PointCloud2()
        msg.height = 0
        msg.width = 0
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = 0
        msg.data = bytes()
        
        points = pointcloud2_to_array(msg)
        assert points.shape == (0, 3)

    def test_xyz_extraction(self):
        """Test XYZ point extraction from PointCloud2."""
        from fl_slam_poc.frontend.sensor_io import pointcloud2_to_array
        from sensor_msgs.msg import PointCloud2, PointField
        
        # Create a simple point cloud with 3 points
        n_points = 3
        msg = PointCloud2()
        msg.height = 1
        msg.width = n_points
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        
        # Create data: points at (1,2,3), (4,5,6), (7,8,9)
        import struct
        data = b''
        for i in range(n_points):
            x, y, z = float(i*3 + 1), float(i*3 + 2), float(i*3 + 3)
            data += struct.pack('fff', x, y, z)
        msg.data = data
        
        points = pointcloud2_to_array(msg)
        
        assert points.shape == (3, 3)
        assert np.allclose(points[0], [1.0, 2.0, 3.0])
        assert np.allclose(points[1], [4.0, 5.0, 6.0])
        assert np.allclose(points[2], [7.0, 8.0, 9.0])


# =============================================================================
# Voxel Filtering Tests
# =============================================================================


class TestVoxelFiltering:
    """Test voxel grid filtering (CPU and GPU)."""

    def test_voxel_filter_reduces_points(self):
        """Voxel filtering should reduce point count."""
        # Create a dense point cloud
        np.random.seed(42)
        points = np.random.randn(1000, 3).astype(np.float32)
        
        filtered = voxel_filter_gpu(points, voxel_size=0.5)
        
        assert filtered.shape[0] < points.shape[0]
        assert filtered.shape[1] == 3

    def test_voxel_filter_preserves_dimensions(self):
        """Filtered points should have same dimensionality."""
        points = np.random.randn(100, 3).astype(np.float32)
        filtered = voxel_filter_gpu(points, voxel_size=1.0)
        
        assert filtered.shape[1] == 3
        assert filtered.dtype == np.float32 or filtered.dtype == np.float64

    def test_voxel_filter_empty_cloud(self):
        """Empty cloud should return empty."""
        points = np.empty((0, 3), dtype=np.float32)
        filtered = voxel_filter_gpu(points, voxel_size=0.1)
        
        assert filtered.shape[0] == 0

    def test_voxel_filter_single_point(self):
        """Single point should pass through."""
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        filtered = voxel_filter_gpu(points, voxel_size=0.1)
        
        assert filtered.shape[0] == 1

    def test_voxel_filter_preserves_extent(self):
        """Filtered cloud should have similar extent to original."""
        np.random.seed(42)
        points = np.random.randn(1000, 3).astype(np.float32) * 10
        filtered = voxel_filter_gpu(points, voxel_size=0.5)
        
        orig_min = points.min(axis=0)
        orig_max = points.max(axis=0)
        filt_min = filtered.min(axis=0)
        filt_max = filtered.max(axis=0)
        
        # Filtered extent should be within original extent
        assert np.all(filt_min >= orig_min - 0.5)
        assert np.all(filt_max <= orig_max + 0.5)

    def test_processor_voxel_filter(self):
        """Test GPUPointCloudProcessor.voxel_filter."""
        proc = GPUPointCloudProcessor(voxel_size=0.5, fallback_to_cpu=True)
        
        np.random.seed(42)
        points = np.random.randn(500, 3).astype(np.float32)
        
        filtered = proc.voxel_filter(points)
        
        assert filtered.shape[0] < points.shape[0]
        assert filtered.shape[1] == 3


# =============================================================================
# GPU ICP Tests
# =============================================================================


class TestGPUICP:
    """Test GPU-accelerated ICP registration."""

    def test_icp_identity_transform(self):
        """ICP on identical clouds should give near-identity transform."""
        np.random.seed(42)
        source = np.random.randn(100, 3).astype(np.float32)
        target = source.copy()
        
        result = icp_gpu(source, target, max_iter=10, tol=1e-6)
        
        assert isinstance(result, ICPResult)
        assert np.linalg.norm(result.transform) < 0.1
        assert result.mse < 0.01

    def test_icp_recovers_translation(self):
        """ICP should recover known translation."""
        np.random.seed(42)
        source = np.random.randn(100, 3).astype(np.float32)
        translation = np.array([1.0, 0.0, 0.0])
        target = source + translation
        
        # max_correspondence_distance must be > translation magnitude for ICP to find correspondences
        result = icp_gpu(source, target, max_iter=20, tol=1e-6, max_correspondence_distance=2.0)
        
        # Check translation recovery
        assert abs(result.transform[0] - 1.0) < 0.1
        assert abs(result.transform[1]) < 0.1
        assert abs(result.transform[2]) < 0.1

    def test_icp_recovers_rotation(self):
        """ICP should recover known rotation."""
        np.random.seed(42)
        source = np.random.randn(100, 3).astype(np.float32)
        
        # Small rotation around z-axis
        rotvec_true = np.array([0.0, 0.0, 0.1])
        R_true = rotvec_to_rotmat(rotvec_true)
        target = (R_true @ source.T).T.astype(np.float32)
        
        result = icp_gpu(source, target, max_iter=20, tol=1e-6)
        
        # Check rotation recovery (compare rotation vectors)
        rotvec_est = result.transform[3:6]
        assert np.linalg.norm(rotvec_est - rotvec_true) < 0.2

    def test_icp_result_format(self):
        """ICP result should have all required fields."""
        np.random.seed(42)
        source = np.random.randn(50, 3).astype(np.float32)
        target = source + 0.1 * np.random.randn(50, 3)
        
        result = icp_gpu(source, target, max_iter=10, tol=1e-6)
        
        assert hasattr(result, 'transform')
        assert hasattr(result, 'mse')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'max_iterations')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'n_source')
        assert hasattr(result, 'n_target')
        
        assert result.transform.shape == (6,)
        assert isinstance(result.mse, float)
        assert result.mse >= 0

    def test_icp_improves_objective(self):
        """ICP should improve or maintain objective."""
        np.random.seed(42)
        source = np.random.randn(50, 3).astype(np.float32)
        target = source + 0.2 * np.random.randn(50, 3)
        
        result = icp_gpu(source, target, max_iter=20, tol=1e-6)
        
        assert result.final_objective <= result.initial_objective + 1e-8

    def test_icp_bounded_iterations(self):
        """ICP should respect max_iter bound."""
        np.random.seed(42)
        source = np.random.randn(50, 3).astype(np.float32)
        target = source + np.random.randn(50, 3)  # Large noise
        
        max_iter = 5
        result = icp_gpu(source, target, max_iter=max_iter, tol=1e-10)
        
        assert result.iterations <= max_iter


class TestGPUProcessorICP:
    """Test GPUPointCloudProcessor ICP method."""

    def test_processor_icp(self):
        """Test GPUPointCloudProcessor.icp method."""
        proc = GPUPointCloudProcessor(voxel_size=0.1, fallback_to_cpu=True)
        
        np.random.seed(42)
        source = np.random.randn(100, 3).astype(np.float32)
        target = source + np.array([0.5, 0.0, 0.0])
        
        result = proc.icp(source, target, max_iter=20, tol=1e-6)
        
        assert isinstance(result, ICPResult)
        assert abs(result.transform[0] - 0.5) < 0.1

    def test_processor_reports_gpu_status(self):
        """Processor should report GPU availability."""
        proc = GPUPointCloudProcessor(fallback_to_cpu=True)
        
        # Should not raise, regardless of GPU availability
        assert isinstance(proc.use_gpu, bool)


# =============================================================================
# CPU Fallback Tests
# =============================================================================


class TestCPUFallback:
    """Test CPU fallback when GPU is unavailable."""

    def test_fallback_voxel_filter(self):
        """Voxel filter should work even without GPU."""
        proc = GPUPointCloudProcessor(voxel_size=0.5, fallback_to_cpu=True)
        
        # Force CPU path
        proc.use_gpu = False
        proc._gpu_processor = None
        
        points = np.random.randn(100, 3).astype(np.float32)
        filtered = proc.voxel_filter(points)
        
        assert filtered.shape[0] > 0
        assert filtered.shape[1] == 3

    def test_fallback_icp(self):
        """ICP should work even without GPU."""
        proc = GPUPointCloudProcessor(fallback_to_cpu=True)
        
        # Force CPU path
        proc.use_gpu = False
        proc._gpu_processor = None
        
        np.random.seed(42)
        source = np.random.randn(50, 3).astype(np.float32)
        target = source + np.array([0.5, 0.0, 0.0])
        
        result = proc.icp(source, target, max_iter=20, tol=1e-6)
        
        assert isinstance(result, ICPResult)


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoopProcessorGPUIntegration:
    """Test LoopProcessor with GPU configuration."""

    def test_loop_processor_initialization(self):
        """LoopProcessor should initialize with GPU config."""
        from fl_slam_poc.frontend.loop_processor import LoopProcessor
        from fl_slam_poc.backend import AdaptiveParameter
        
        proc = LoopProcessor(
            fr_distance_scale=AdaptiveParameter(prior_mean=1.0, prior_strength=5.0),
            icp_max_iter=AdaptiveParameter(prior_mean=15.0, prior_strength=10.0),
            icp_tol=AdaptiveParameter(prior_mean=1e-4, prior_strength=10.0),
            icp_n_ref=100.0,
            icp_sigma_mse=0.01,
            use_gpu=True,
            gpu_fallback_to_cpu=True,
            use_3d_pointcloud=True,
            voxel_size=0.05,
        )
        
        assert proc.use_3d_pointcloud == True
        assert proc.voxel_size == 0.05

    def test_loop_processor_preprocess(self):
        """LoopProcessor should preprocess 3D point clouds."""
        from fl_slam_poc.frontend.loop_processor import LoopProcessor
        from fl_slam_poc.backend import AdaptiveParameter
        
        proc = LoopProcessor(
            fr_distance_scale=AdaptiveParameter(prior_mean=1.0, prior_strength=5.0),
            icp_max_iter=AdaptiveParameter(prior_mean=15.0, prior_strength=10.0),
            icp_tol=AdaptiveParameter(prior_mean=1e-4, prior_strength=10.0),
            icp_n_ref=100.0,
            icp_sigma_mse=0.01,
            use_gpu=False,
            gpu_fallback_to_cpu=True,
            use_3d_pointcloud=True,
            voxel_size=0.5,
        )
        
        np.random.seed(42)
        points = np.random.randn(1000, 3).astype(np.float32)
        
        filtered = proc.preprocess_pointcloud(points)
        
        assert filtered.shape[0] < points.shape[0]

    def test_loop_processor_run_icp_3d(self):
        """LoopProcessor should run ICP on 3D point clouds."""
        from fl_slam_poc.frontend.loop_processor import LoopProcessor
        from fl_slam_poc.backend import AdaptiveParameter
        
        proc = LoopProcessor(
            fr_distance_scale=AdaptiveParameter(prior_mean=1.0, prior_strength=5.0),
            icp_max_iter=AdaptiveParameter(prior_mean=15.0, prior_strength=10.0),
            icp_tol=AdaptiveParameter(prior_mean=1e-4, prior_strength=10.0),
            icp_n_ref=100.0,
            icp_sigma_mse=0.01,
            use_gpu=False,
            gpu_fallback_to_cpu=True,
            use_3d_pointcloud=False,  # Disable preprocessing for this test
        )
        
        np.random.seed(42)
        source = np.random.randn(50, 3).astype(np.float32)
        target = source + np.array([0.3, 0.0, 0.0])
        
        result = proc.run_icp(source, target, preprocess=False)
        
        assert result is not None
        assert isinstance(result, ICPResult)
        assert abs(result.transform[0] - 0.3) < 0.1


# =============================================================================
# GPU Availability Tests
# =============================================================================


class TestGPUAvailability:
    """Test GPU availability checking."""

    def test_is_gpu_available_callable(self):
        """is_gpu_available should be callable."""
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_gpu_processor_respects_fallback(self):
        """Processor should respect fallback_to_cpu setting."""
        # Should not raise with fallback enabled
        proc = GPUPointCloudProcessor(fallback_to_cpu=True)
        assert proc is not None


# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
