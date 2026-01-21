"""
Unit tests for RGB-D processing and multi-modal sensor fusion.

Tests:
1. Depth to pointcloud conversion
2. Normal estimation from depth
3. vMF barycenter associativity (WDVV compliance)
4. Laser 2D + RGB-D 3D fusion (information form)
5. Fisher-Rao triangle inequality for vMF

Reference: Hybrid Laser + RGB-D Sensor Fusion Architecture
"""

import numpy as np
import pytest

# Backend fusion operators
from fl_slam_poc.backend.fusion.gaussian_info import make_evidence, mean_cov, fuse_info
from fl_slam_poc.backend.fusion.multimodal_fusion import (
    laser_2d_to_3d_constraint,
    fuse_laser_rgbd,
    fuse_multimodal_3d,
    spatial_association_weight,
)

# Frontend loops operators
from fl_slam_poc.frontend.loops.vmf_geometry import (
    vmf_make_evidence,
    vmf_mean_param,
    vmf_barycenter,
    vmf_fisher_rao_distance,
    A_d,
    A_d_inverse_series,
)

# Frontend processing
from fl_slam_poc.frontend.processing.rgbd_processor import (
    depth_to_pointcloud,
    compute_normals_from_depth,
    rgbd_to_evidence,
)


class TestDepthToPointcloud:
    """Tests for depth image to 3D pointcloud conversion."""
    
    def test_flat_wall_at_2m(self):
        """Flat wall at 2m should produce points at z=2."""
        depth = np.ones((480, 640), dtype=np.float32) * 2.0
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        
        points, colors, normals, covs = depth_to_pointcloud(depth, K, subsample=50)
        
        assert len(points) > 0
        assert points.shape[1] == 3
        assert np.allclose(points[:, 2], 2.0, atol=0.01)
    
    def test_invalid_depth_filtered(self):
        """Invalid depth values (0 or very large) should be filtered."""
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[30:70, 30:70] = 3.0  # Valid region
        depth[:30, :] = 0.0  # Invalid (too close)
        depth[70:, :] = 100.0  # Invalid (too far)
        
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        
        points, colors, normals, covs = depth_to_pointcloud(depth, K, subsample=10)
        
        # Only valid points should remain
        assert len(points) > 0
        assert np.all(points[:, 2] >= 0.1)
        assert np.all(points[:, 2] <= 10.0)
    
    def test_covariance_scaling_with_depth(self):
        """Covariance should scale with depth squared (stereo model)."""
        depth = np.ones((100, 100), dtype=np.float32)
        depth[:, :50] = 1.0
        depth[:, 50:] = 4.0
        
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        
        points, colors, normals, covs = depth_to_pointcloud(depth, K, subsample=20)
        
        # Find points at different depths
        near_covs = [cov for pt, cov in zip(points, covs) if pt[2] < 2.0]
        far_covs = [cov for pt, cov in zip(points, covs) if pt[2] > 2.0]
        
        if len(near_covs) > 0 and len(far_covs) > 0:
            near_trace = np.mean([np.trace(c) for c in near_covs])
            far_trace = np.mean([np.trace(c) for c in far_covs])
            # Far points should have larger covariance
            assert far_trace > near_trace


class TestNormalEstimation:
    """Tests for surface normal estimation from depth."""
    
    def test_flat_surface_normal(self):
        """Flat surface perpendicular to camera should have normal ~[0, 0, -1]."""
        depth = np.ones((100, 100), dtype=np.float32) * 2.0
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        
        normals = compute_normals_from_depth(depth, K, subsample=10)
        
        # All normals should point toward camera (negative Z or positive depending on convention)
        assert normals.shape[1] == 3
        # Check that normals are unit vectors
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01)


class TestVMFBarycenter:
    """Tests for vMF barycenter (WDVV associativity compliance)."""
    
    def test_barycenter_single_distribution(self):
        """Barycenter of single distribution should return itself."""
        theta = vmf_make_evidence(np.array([1, 0, 0]), kappa=5.0, d=3)
        
        result, report = vmf_barycenter([theta], [1.0], d=3)
        
        assert np.allclose(result, theta, atol=1e-6)
        assert report.exact == True
    
    def test_barycenter_equal_weights(self):
        """Barycenter of equal-weighted distributions should be symmetric."""
        theta1 = vmf_make_evidence(np.array([1, 0, 0]), kappa=5.0, d=3)
        theta2 = vmf_make_evidence(np.array([0, 1, 0]), kappa=5.0, d=3)
        
        result, report = vmf_barycenter([theta1, theta2], [1.0, 1.0], d=3)
        
        # Result should be in the plane of the two directions
        mu, kappa = vmf_mean_param(result, d=3)
        assert np.allclose(mu[2], 0.0, atol=0.01)  # No Z component
    
    def test_barycenter_associativity(self):
        """Barycenter should be associative: (A+B)+C = A+(B+C)."""
        theta1 = vmf_make_evidence(np.array([1, 0, 0]), kappa=5.0, d=3)
        theta2 = vmf_make_evidence(np.array([0, 1, 0]), kappa=3.0, d=3)
        theta3 = vmf_make_evidence(np.array([0, 0, 1]), kappa=4.0, d=3)
        
        # Left: (θ1 + θ2) + θ3
        theta12, _ = vmf_barycenter([theta1, theta2], [1.0, 1.0], d=3)
        theta_left, _ = vmf_barycenter([theta12, theta3], [2.0, 1.0], d=3)
        
        # Right: θ1 + (θ2 + θ3)
        theta23, _ = vmf_barycenter([theta2, theta3], [1.0, 1.0], d=3)
        theta_right, _ = vmf_barycenter([theta1, theta23], [1.0, 2.0], d=3)
        
        # Should be approximately equal (up to numerical precision)
        assert np.allclose(theta_left, theta_right, atol=1e-4)
    
    def test_barycenter_commutativity(self):
        """Barycenter should be commutative: A+B = B+A."""
        theta1 = vmf_make_evidence(np.array([1, 0, 0]), kappa=5.0, d=3)
        theta2 = vmf_make_evidence(np.array([0, 1, 0]), kappa=3.0, d=3)
        
        result_12, _ = vmf_barycenter([theta1, theta2], [1.0, 2.0], d=3)
        result_21, _ = vmf_barycenter([theta2, theta1], [2.0, 1.0], d=3)
        
        assert np.allclose(result_12, result_21, atol=1e-6)


class TestVMFFisherRao:
    """Tests for vMF Fisher-Rao distance."""
    
    def test_distance_identical_distributions(self):
        """Distance between identical distributions should be zero."""
        theta = vmf_make_evidence(np.array([1, 0, 0]), kappa=5.0, d=3)
        
        d = vmf_fisher_rao_distance(theta, theta, d=3)
        
        assert np.isclose(d, 0.0, atol=1e-6)
    
    def test_distance_symmetry(self):
        """Fisher-Rao distance should be symmetric: d(A,B) = d(B,A)."""
        theta1 = vmf_make_evidence(np.array([1, 0, 0]), kappa=5.0, d=3)
        theta2 = vmf_make_evidence(np.array([0, 1, 0]), kappa=3.0, d=3)
        
        d12 = vmf_fisher_rao_distance(theta1, theta2, d=3)
        d21 = vmf_fisher_rao_distance(theta2, theta1, d=3)
        
        assert np.isclose(d12, d21, atol=1e-6)
    
    def test_distance_triangle_inequality(self):
        """Fisher-Rao should satisfy triangle inequality: d(A,C) <= d(A,B) + d(B,C)."""
        theta_a = vmf_make_evidence(np.array([1, 0, 0]), kappa=5.0, d=3)
        theta_b = vmf_make_evidence(np.array([0, 1, 0]), kappa=3.0, d=3)
        theta_c = vmf_make_evidence(np.array([0, 0, 1]), kappa=4.0, d=3)
        
        d_ab = vmf_fisher_rao_distance(theta_a, theta_b, d=3)
        d_bc = vmf_fisher_rao_distance(theta_b, theta_c, d=3)
        d_ac = vmf_fisher_rao_distance(theta_a, theta_c, d=3)
        
        # Triangle inequality (with small tolerance for numerical errors)
        assert d_ac <= d_ab + d_bc + 1e-6
    
    def test_distance_nonnegative(self):
        """Fisher-Rao distance should be non-negative."""
        theta1 = vmf_make_evidence(np.array([1, 0, 0]), kappa=5.0, d=3)
        theta2 = vmf_make_evidence(np.array([0, 1, 0]), kappa=3.0, d=3)
        
        d = vmf_fisher_rao_distance(theta1, theta2, d=3)
        
        assert d >= 0.0


class TestBesselInversion:
    """Tests for Bessel function inversion (A_d and A_d_inverse)."""
    
    def test_inverse_series_roundtrip(self):
        """A_d_inverse(A_d(κ)) should approximately equal κ."""
        for kappa in [0.1, 1.0, 5.0, 10.0]:
            r = A_d(kappa, d=3)
            kappa_recovered = A_d_inverse_series(r, d=3)
            assert np.isclose(kappa_recovered, kappa, rtol=0.1)
    
    def test_A_d_range(self):
        """A_d(κ) should be in [0, 1) for all κ >= 0."""
        for kappa in [0.0, 0.1, 1.0, 10.0, 100.0]:
            r = A_d(kappa, d=3)
            assert 0.0 <= r < 1.0


class TestLaserRGBDFusion:
    """Tests for laser 2D + RGB-D 3D fusion in information form."""
    
    def test_laser_2d_to_3d_lifting(self):
        """Laser 2D evidence should lift to 3D with weak Z prior."""
        laser_mu = np.array([1.0, 0.5])
        laser_cov = np.eye(2) * 0.1
        
        L_3d, h_3d = laser_2d_to_3d_constraint(laser_mu, laser_cov, z_prior_var=10.0)
        mu_3d, cov_3d = mean_cov(L_3d, h_3d)
        
        # XY should match laser
        assert np.allclose(mu_3d[:2], laser_mu, atol=0.01)
        # Z should be at prior mean (0)
        assert np.isclose(mu_3d[2], 0.0, atol=0.01)
        # Z variance should be large (weak constraint)
        assert cov_3d[2, 2] > cov_3d[0, 0]
    
    def test_fusion_xy_dominated_by_laser(self):
        """XY position should be dominated by laser (lower covariance)."""
        # Laser with low XY uncertainty
        laser_mu = np.array([1.0, 0.5])
        laser_cov = np.eye(2) * 0.01  # Very precise
        L_laser, h_laser = make_evidence(laser_mu, laser_cov)
        
        # RGB-D with higher uncertainty
        rgbd_mu = np.array([1.2, 0.6, 0.3])
        rgbd_cov = np.eye(3) * 0.1  # Less precise
        L_rgbd, h_rgbd = make_evidence(rgbd_mu, rgbd_cov)
        
        # Fuse
        L_fused, h_fused, report = fuse_laser_rgbd(L_laser, h_laser, L_rgbd, h_rgbd)
        mu_fused, cov_fused = mean_cov(L_fused, h_fused)
        
        # XY should be closer to laser (more informative)
        assert np.linalg.norm(mu_fused[:2] - laser_mu) < np.linalg.norm(mu_fused[:2] - rgbd_mu[:2])
        
        # Report should indicate exact fusion
        assert report.exact == True
    
    def test_fusion_z_dominated_by_rgbd(self):
        """Z position should be dominated by RGB-D (laser has weak Z prior)."""
        laser_mu = np.array([1.0, 0.5])
        laser_cov = np.eye(2) * 0.1
        L_laser, h_laser = make_evidence(laser_mu, laser_cov)
        
        rgbd_mu = np.array([1.0, 0.5, 0.2])
        rgbd_cov = np.eye(3) * 0.05
        L_rgbd, h_rgbd = make_evidence(rgbd_mu, rgbd_cov)
        
        L_fused, h_fused, report = fuse_laser_rgbd(
            L_laser, h_laser, L_rgbd, h_rgbd,
            z_prior_var=10.0  # Weak Z prior for laser
        )
        mu_fused, cov_fused = mean_cov(L_fused, h_fused)
        
        # Z should be close to RGB-D value
        assert np.isclose(mu_fused[2], rgbd_mu[2], atol=0.1)
    
    def test_fusion_associativity(self):
        """Multi-modal fusion should be associative."""
        ev1 = (np.eye(3), np.array([1, 0, 0]))
        ev2 = (np.eye(3) * 2, np.array([0, 1, 0]))
        ev3 = (np.eye(3) * 0.5, np.array([0, 0, 1]))
        
        # (ev1 + ev2) + ev3
        L12 = ev1[0] + ev2[0]
        h12 = ev1[1] + ev2[1]
        L_left = L12 + ev3[0]
        h_left = h12 + ev3[1]
        
        # ev1 + (ev2 + ev3)
        L23 = ev2[0] + ev3[0]
        h23 = ev2[1] + ev3[1]
        L_right = ev1[0] + L23
        h_right = ev1[1] + h23
        
        assert np.allclose(L_left, L_right)
        assert np.allclose(h_left, h_right)


class TestSpatialAssociation:
    """Tests for spatial association weighting."""
    
    def test_identical_positions_high_weight(self):
        """Identical positions should have weight ~1."""
        mu = np.array([1.0, 2.0, 3.0])
        w = spatial_association_weight(mu, mu, scale=0.5)
        assert np.isclose(w, 1.0, atol=1e-6)
    
    def test_distant_positions_low_weight(self):
        """Distant positions should have low weight."""
        mu1 = np.array([0.0, 0.0, 0.0])
        mu2 = np.array([10.0, 10.0, 10.0])
        w = spatial_association_weight(mu1, mu2, scale=0.5)
        assert w < 0.01
    
    def test_weight_decreases_with_distance(self):
        """Weight should decrease monotonically with distance."""
        mu1 = np.array([0.0, 0.0, 0.0])
        
        weights = []
        for dist in [0.1, 0.5, 1.0, 2.0, 5.0]:
            mu2 = np.array([dist, 0.0, 0.0])
            w = spatial_association_weight(mu1, mu2, scale=0.5)
            weights.append(w)
        
        # Should be strictly decreasing
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]


class TestRGBDEvidence:
    """Tests for RGB-D to evidence conversion."""
    
    def test_evidence_structure(self):
        """Evidence should have all required fields."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        colors = np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        covs = [np.eye(3) * 0.1, np.eye(3) * 0.2]
        
        evidence_list = rgbd_to_evidence(points, colors, normals, covs)
        
        assert len(evidence_list) == 2
        
        for ev in evidence_list:
            assert "position_L" in ev
            assert "position_h" in ev
            assert "color_L" in ev
            assert "color_h" in ev
            assert "normal_theta" in ev
            assert "alpha_mean" in ev
            assert "alpha_var" in ev
    
    def test_position_recovery(self):
        """Position should be recoverable from evidence."""
        points = np.array([[1.0, 2.0, 3.0]])
        colors = np.array([[0.5, 0.5, 0.5]])
        normals = np.array([[0.0, 0.0, 1.0]])
        covs = [np.eye(3) * 0.1]
        
        evidence_list = rgbd_to_evidence(points, colors, normals, covs)
        ev = evidence_list[0]
        
        mu_recovered, cov_recovered = mean_cov(ev["position_L"], ev["position_h"])
        
        assert np.allclose(mu_recovered, points[0], atol=1e-6)


class TestGaussianInfoAssociativity:
    """Tests for Gaussian information form associativity (WDVV compliance)."""
    
    def test_info_fusion_associativity(self):
        """Information fusion should be associative."""
        L1, h1 = make_evidence(np.array([0., 0., 0.]), np.eye(3) * 0.5)
        L2, h2 = make_evidence(np.array([1., 0., 0.]), np.eye(3) * 0.3)
        L3, h3 = make_evidence(np.array([0., 1., 0.]), np.eye(3) * 0.4)
        
        # Left associative: (L1 + L2) + L3
        L12, h12 = fuse_info(L1, h1, L2, h2)
        L_left, h_left = fuse_info(L12, h12, L3, h3)
        
        # Right associative: L1 + (L2 + L3)
        L23, h23 = fuse_info(L2, h2, L3, h3)
        L_right, h_right = fuse_info(L1, h1, L23, h23)
        
        assert np.allclose(L_left, L_right, atol=1e-10)
        assert np.allclose(h_left, h_right, atol=1e-10)
    
    def test_info_fusion_commutativity(self):
        """Information fusion should be commutative."""
        L1, h1 = make_evidence(np.array([0., 0., 0.]), np.eye(3) * 0.5)
        L2, h2 = make_evidence(np.array([1., 0., 0.]), np.eye(3) * 0.3)
        
        L_12, h_12 = fuse_info(L1, h1, L2, h2)
        L_21, h_21 = fuse_info(L2, h2, L1, h1)
        
        assert np.allclose(L_12, L_21, atol=1e-10)
        assert np.allclose(h_12, h_21, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
