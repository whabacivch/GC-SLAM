"""
Tests for IMU preintegration operator.

Validates basic functionality of IMUPreintegrator class.
"""

import numpy as np
import pytest

from fl_slam_poc.operators.imu_preintegration import IMUPreintegrator


def test_imu_preintegrator_initialization():
    """Test that preintegrator initializes correctly."""
    preint = IMUPreintegrator(
        gyro_noise_density=1e-3,
        accel_noise_density=1e-2,
        gyro_random_walk=1e-5,
        accel_random_walk=1e-4,
    )
    
    assert preint.gyro_noise_density == 1e-3
    assert preint.accel_noise_density == 1e-2
    assert np.allclose(preint.gravity, [0, 0, -9.81])


def test_imu_preintegration_identity():
    """Test preintegration with no motion (should produce identity)."""
    preint = IMUPreintegrator()
    
    # Create measurements with zero motion
    measurements = [
        (0.0, np.array([0, 0, 9.81]), np.array([0, 0, 0])),  # Only gravity
        (0.1, np.array([0, 0, 9.81]), np.array([0, 0, 0])),
        (0.2, np.array([0, 0, 9.81]), np.array([0, 0, 0])),
    ]
    
    delta_rotvec, delta_v, delta_p, cov, report = preint.integrate(
        0.0, 0.2, measurements
    )
    
    # With only gravity (no motion), rotation should be near identity
    assert np.linalg.norm(delta_rotvec) < 0.1  # Small rotation
    assert np.linalg.norm(delta_v) < 1.0  # Small velocity
    assert cov.shape == (9, 9)
    assert report.name == "IMUPreintegration"


def test_imu_preintegration_insufficient_measurements():
    """Test handling of insufficient measurements."""
    preint = IMUPreintegrator()
    
    # Only one measurement
    measurements = [
        (0.0, np.array([0, 0, 9.81]), np.array([0, 0, 0])),
    ]
    
    delta_rotvec, delta_v, delta_p, cov, report = preint.integrate(
        0.0, 0.1, measurements
    )
    
    # Should return identity with large uncertainty
    assert np.allclose(delta_rotvec, 0, atol=1e-6)
    assert np.allclose(delta_v, 0, atol=1e-6)
    assert np.allclose(delta_p, 0, atol=1e-6)
    assert "InsufficientMeasurements" in report.approximation_triggers


def test_imu_preintegration_with_bias():
    """Test preintegration with non-zero bias."""
    preint = IMUPreintegrator()
    
    bias_gyro = np.array([0.01, 0.01, 0.01])  # rad/s
    bias_accel = np.array([0.1, 0.1, 0.1])  # m/s²
    
    # Create measurements
    measurements = [
        (0.0, np.array([0, 0, 9.81]) + bias_accel, np.array([0, 0, 0]) + bias_gyro),
        (0.1, np.array([0, 0, 9.81]) + bias_accel, np.array([0, 0, 0]) + bias_gyro),
        (0.2, np.array([0, 0, 9.81]) + bias_accel, np.array([0, 0, 0]) + bias_gyro),
    ]
    
    delta_rotvec, delta_v, delta_p, cov, report = preint.integrate(
        0.0, 0.2, measurements, bias_gyro=bias_gyro, bias_accel=bias_accel
    )
    
    # With bias correction, should still be near identity
    assert cov.shape == (9, 9)
    assert "Linearization" in report.approximation_triggers
    assert "BiasConstant" in report.approximation_triggers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
