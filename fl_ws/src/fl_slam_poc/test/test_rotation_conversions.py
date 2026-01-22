"""
Unit tests for rotation conversions, especially edge cases near π.

Priority: Fix rotation frame issues causing ~135° ATE rotation error.
Expected: ATE rotation should drop to <20° after fixes.
"""

import numpy as np
import pytest
import math

from fl_slam_poc.common.se3 import (
    rotvec_to_rotmat,
    rotmat_to_rotvec,
    quat_to_rotmat,
    rotmat_to_quat,
    quat_to_rotvec,
    rotvec_to_rotmat,
)


class TestRotationConversions:
    """Test rotation conversions for correctness and edge cases."""
    
    def test_rotvec_rotmat_roundtrip_identity(self):
        """Test rotvec ↔ rotmat roundtrip for identity."""
        rotvec = np.array([0.0, 0.0, 0.0])
        R = rotvec_to_rotmat(rotvec)
        rotvec_recovered = rotmat_to_rotvec(R)
        
        assert np.allclose(R, np.eye(3), atol=1e-10)
        assert np.allclose(rotvec_recovered, rotvec, atol=1e-10)
    
    def test_rotvec_rotmat_roundtrip_small_angle(self):
        """Test rotvec ↔ rotmat roundtrip for small angles."""
        rotvec = np.array([0.01, 0.02, 0.03])
        R = rotvec_to_rotmat(rotvec)
        rotvec_recovered = rotmat_to_rotvec(R)
        
        # Should recover original (within numerical precision)
        assert np.allclose(rotvec_recovered, rotvec, atol=1e-8)
    
    def test_rotvec_rotmat_roundtrip_near_pi(self):
        """Test rotvec ↔ rotmat roundtrip near π (critical edge case)."""
        # Test various angles near π
        angles = [math.pi - 0.1, math.pi - 0.01, math.pi - 0.001, math.pi - 1e-6]
        
        for angle in angles:
            # Create rotation vector with magnitude near π
            axis = np.array([1.0, 0.0, 0.0]) / np.linalg.norm([1.0, 0.0, 0.0])
            rotvec = axis * angle
            
            R = rotvec_to_rotmat(rotvec)
            rotvec_recovered = rotmat_to_rotvec(R)
            
            # Recovered angle should match (axis may flip, but angle should be correct)
            angle_recovered = np.linalg.norm(rotvec_recovered)
            
            # Angle should be close to original (accounting for 2π ambiguity)
            angle_diff = min(angle_recovered, 2*math.pi - angle_recovered)
            expected_diff = min(angle, 2*math.pi - angle)
            
            assert abs(angle_diff - expected_diff) < 1e-6, \
                f"Angle mismatch near π: {angle} -> {angle_recovered}"
            
            # Recovered rotation should produce same matrix
            R_recovered = rotvec_to_rotmat(rotvec_recovered)
            assert np.allclose(R, R_recovered, atol=1e-6), \
                f"Matrix mismatch near π for angle {angle}"
    
    def test_rotvec_rotmat_exactly_pi(self):
        """Test rotvec ↔ rotmat for exactly π rotation."""
        # 180° rotation around x-axis
        rotvec = np.array([math.pi, 0.0, 0.0])
        R = rotvec_to_rotmat(rotvec)
        
        # Should be 180° rotation: R = diag([1, -1, -1]) for x-axis
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        
        assert np.allclose(R, expected, atol=1e-6)
        
        # Roundtrip
        rotvec_recovered = rotmat_to_rotvec(R)
        angle_recovered = np.linalg.norm(rotvec_recovered)
        
        # Should recover π (or equivalent)
        assert abs(angle_recovered - math.pi) < 1e-6 or abs(angle_recovered - math.pi) < 1e-6
    
    def test_quat_rotvec_roundtrip_near_pi(self):
        """Test quat ↔ rotvec roundtrip near π."""
        angles = [math.pi - 0.1, math.pi - 0.01, math.pi - 0.001]
        
        for angle in angles:
            axis = np.array([1.0, 0.0, 0.0])
            rotvec = axis * angle
            
            # rotvec -> rotmat -> quat -> rotvec
            R = rotvec_to_rotmat(rotvec)
            qx, qy, qz, qw = rotmat_to_quat(R)
            rotvec_recovered = quat_to_rotvec(qx, qy, qz, qw)
            
            # Check angle magnitude
            angle_recovered = np.linalg.norm(rotvec_recovered)
            angle_diff = min(angle_recovered, 2*math.pi - angle_recovered)
            expected_diff = min(angle, 2*math.pi - angle)
            
            assert abs(angle_diff - expected_diff) < 1e-6, \
                f"Quat roundtrip failed near π: {angle} -> {angle_recovered}"
    
    def test_quat_rotmat_roundtrip_near_pi(self):
        """Test quat ↔ rotmat roundtrip near π."""
        angles = [math.pi - 0.1, math.pi - 0.01]
        
        for angle in angles:
            axis = np.array([0.0, 1.0, 0.0])
            rotvec = axis * angle
            
            R1 = rotvec_to_rotmat(rotvec)
            qx, qy, qz, qw = rotmat_to_quat(R1)
            R2 = quat_to_rotmat(qx, qy, qz, qw)
            
            assert np.allclose(R1, R2, atol=1e-6), \
                f"Quat-rotmat roundtrip failed near π for angle {angle}"
    
    def test_quat_sign_consistency(self):
        """Test that quaternion sign doesn't affect rotation."""
        # q and -q represent the same rotation
        qx, qy, qz, qw = 0.5, 0.5, 0.5, 0.5
        R1 = quat_to_rotmat(qx, qy, qz, qw)
        R2 = quat_to_rotmat(-qx, -qy, -qz, -qw)
        
        assert np.allclose(R1, R2, atol=1e-10)
    
    def test_rotvec_axis_consistency(self):
        """Test that rotvec axis direction doesn't matter (2π ambiguity)."""
        # rotvec and rotvec - 2π*axis represent the same rotation
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.1
        rotvec1 = axis * angle
        rotvec2 = axis * (angle - 2*math.pi)
        
        R1 = rotvec_to_rotmat(rotvec1)
        R2 = rotvec_to_rotmat(rotvec2)
        
        assert np.allclose(R1, R2, atol=1e-10)
    
    def test_large_rotation_vectors(self):
        """Test handling of rotation vectors with magnitude > π."""
        # Large rotation should be wrapped to [-π, π]
        rotvec_large = np.array([2.0 * math.pi + 0.1, 0.0, 0.0])
        R = rotvec_to_rotmat(rotvec_large)
        
        # Should be equivalent to small rotation
        rotvec_small = np.array([0.1, 0.0, 0.0])
        R_small = rotvec_to_rotmat(rotvec_small)
        
        assert np.allclose(R, R_small, atol=1e-6)
    
    def test_quat_normalization(self):
        """Test that quaternion normalization is handled correctly."""
        # Unnormalized quaternion
        qx, qy, qz, qw = 0.6, 0.8, 0.0, 0.0  # Not normalized
        R = quat_to_rotmat(qx, qy, qz, qw)
        
        # Should still produce valid rotation matrix
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)  # Orthogonal
        assert abs(np.linalg.det(R) - 1.0) < 1e-6  # Determinant = 1


class TestRotationFrameConsistency:
    """Test frame convention consistency across conversions."""
    
    def test_quat_xyzw_convention(self):
        """Verify quaternion convention is (x, y, z, w)."""
        # 90° rotation around z-axis
        qx, qy, qz, qw = 0.0, 0.0, 0.70710678, 0.70710678  # (x, y, z, w)
        R = quat_to_rotmat(qx, qy, qz, qw)
        
        # Should rotate x -> y, y -> -x
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis_expected = np.array([0.0, 1.0, 0.0])
        y_axis_actual = R @ x_axis
        
        assert np.allclose(y_axis_actual, y_axis_expected, atol=1e-6)
    
    def test_rotvec_handedness(self):
        """Verify rotation vector follows right-hand rule."""
        # Positive rotation around z-axis should rotate x -> y
        rotvec = np.array([0.0, 0.0, math.pi / 2])
        R = rotvec_to_rotmat(rotvec)
        
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis_expected = np.array([0.0, 1.0, 0.0])
        y_axis_actual = R @ x_axis
        
        assert np.allclose(y_axis_actual, y_axis_expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
