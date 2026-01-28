#!/usr/bin/env python3
"""
Diagnostic to verify Wahba SVD rotation sign convention.

This creates synthetic data with KNOWN rotations and verifies that Wahba
recovers the correct rotation (not its inverse/transpose).

If Wahba returns R^T instead of R, we have a sign/convention bug.
"""

import sys
sys.path.insert(0, "/home/will/Documents/Coding/Phantom Fellowship MIT/Impact_Project_v1/fl_ws/src/fl_slam_poc")

import numpy as np
from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.backend.operators.wahba import wahba_svd


def create_test_directions(n_dirs=48):
    """Create well-distributed test directions (Fibonacci sphere)."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    indices = np.arange(n_dirs)
    theta = 2 * np.pi * indices / phi
    z = 1 - (2 * indices + 1) / n_dirs
    r = np.sqrt(1 - z**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def test_wahba_with_known_rotation(R_true, description):
    """
    Test Wahba SVD with a known rotation.

    Setup:
    - mu_map = R_true @ mu_body  (map directions in world frame)
    - mu_scan = mu_body          (scan directions in body frame)

    Wahba should find R such that R @ mu_scan ≈ mu_map
    So R_hat should equal R_true.

    If R_hat ≈ R_true^T, we have a sign bug.
    """
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")

    # Create directions in body frame
    mu_body = create_test_directions(48)

    # Transform to world frame using the known rotation
    mu_world = (R_true @ mu_body.T).T  # (N, 3)

    # For Wahba: mu_map is world, mu_scan is body
    # We want R such that R @ mu_scan ≈ mu_map
    # So R should be R_true

    weights = np.ones(len(mu_body))

    result, cert, effect = wahba_svd(
        mu_map=jnp.array(mu_world),
        mu_scan=jnp.array(mu_body),
        weights=jnp.array(weights),
    )

    R_hat = np.array(result.R_hat)

    # Check if R_hat matches R_true or R_true^T
    err_correct = np.linalg.norm(R_hat - R_true, 'fro')
    err_transpose = np.linalg.norm(R_hat - R_true.T, 'fro')

    # Extract yaw for comparison
    def yaw_from_R(R):
        return np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    yaw_true = yaw_from_R(R_true)
    yaw_hat = yaw_from_R(R_hat)
    yaw_hat_T = yaw_from_R(R_hat.T)

    print(f"\nTrue rotation yaw:     {yaw_true:+8.2f}°")
    print(f"Wahba R_hat yaw:       {yaw_hat:+8.2f}°")
    print(f"Wahba R_hat^T yaw:     {yaw_hat_T:+8.2f}°")

    print(f"\nFrobenius error (R_hat vs R_true):   {err_correct:.6f}")
    print(f"Frobenius error (R_hat vs R_true^T): {err_transpose:.6f}")

    if err_correct < err_transpose:
        print(f"\n✓ CORRECT: Wahba returns R_true (not transpose)")
        return True
    else:
        print(f"\n✗ BUG: Wahba returns R_true^T (transpose/inverse)!")
        print(f"  This means the rotation has the WRONG SIGN.")
        print(f"  Fix: Either swap mu_map/mu_scan or transpose R_hat.")
        return False


def test_wahba_incremental():
    """
    Test that simulates actual SLAM usage:
    - First scan: map initialized with identity pose
    - Second scan: robot rotated by +30° yaw

    Expected: Wahba should return R ≈ Rz(+30°)
    """
    print(f"\n{'='*60}")
    print("TEST: Incremental SLAM simulation")
    print(f"{'='*60}")

    # First scan: directions in body frame, robot at identity
    mu_body_scan1 = create_test_directions(48)
    R_world_body_scan1 = np.eye(3)  # Identity pose

    # Map directions = R_world_body @ mu_body (for first scan, this is just mu_body)
    mu_map = (R_world_body_scan1 @ mu_body_scan1.T).T

    # Second scan: robot rotated +30° yaw (CCW when looking down +Z)
    yaw_deg = 30.0
    yaw_rad = np.radians(yaw_deg)
    R_yaw = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    R_world_body_scan2 = R_yaw  # Robot is now at +30° yaw

    # Scan directions in current body frame (same directions, body moved)
    mu_scan = mu_body_scan1  # Same directions in body frame

    # Wahba should find R such that R @ mu_scan ≈ mu_map
    # Since mu_map = I @ mu_body and mu_scan = mu_body
    # We need R @ mu_body ≈ mu_body, so R ≈ I
    #
    # WAIT - this is wrong. Let me think again...
    #
    # Actually, the scan directions are measured in the CURRENT body frame.
    # If the robot rotated, the same world direction appears rotated in the body frame.
    #
    # Let's say there's a world-fixed direction d_world.
    # In body frame at scan 1: d_body1 = R_world_body1^T @ d_world
    # In body frame at scan 2: d_body2 = R_world_body2^T @ d_world
    #
    # If R_world_body1 = I and R_world_body2 = Rz(30), then:
    # d_body1 = d_world
    # d_body2 = Rz(30)^T @ d_world = Rz(-30) @ d_world
    #
    # So the scan directions appear rotated by -30° in the body frame.

    # Actually, let me reconsider the physics:
    # - Robot has a forward direction that points in body +X
    # - At scan 1, robot faces world +X, so body +X = world +X
    # - At scan 2, robot rotated +30° CCW, so body +X points 30° left of world +X
    #
    # A world-fixed object directly ahead at scan 1 (direction [1,0,0] in both frames)
    # At scan 2, the same object is now 30° to the RIGHT in body frame
    # because the robot turned left.
    #
    # So d_body2 = Rz(-30) @ d_body1 (body frame sees world rotated oppositely)

    # Let me set up the test properly:
    # - mu_map = directions in world frame (from scan 1, which used identity pose)
    # - mu_scan = directions in current body frame
    #
    # If robot is at yaw=+30°, then:
    # d_body = R_world_body^T @ d_world = Rz(30)^T @ d_world = Rz(-30) @ d_world

    # Simulating this:
    mu_scan_rotated = (R_yaw.T @ mu_body_scan1.T).T  # Body frame sees world rotated

    print(f"\nSetup:")
    print(f"  Robot rotated: +{yaw_deg}° yaw (CCW)")
    print(f"  Map directions: in world frame (from scan 1 at identity)")
    print(f"  Scan directions: in body frame (robot now at +{yaw_deg}°)")
    print(f"  Expected R_hat: Rz(+{yaw_deg}°) = R_world<-body")

    weights = np.ones(len(mu_map))

    result, cert, effect = wahba_svd(
        mu_map=jnp.array(mu_map),
        mu_scan=jnp.array(mu_scan_rotated),
        weights=jnp.array(weights),
    )

    R_hat = np.array(result.R_hat)

    def yaw_from_R(R):
        return np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    yaw_hat = yaw_from_R(R_hat)
    yaw_expected = yaw_deg

    print(f"\nResults:")
    print(f"  Expected yaw: {yaw_expected:+8.2f}°")
    print(f"  Wahba yaw:    {yaw_hat:+8.2f}°")
    print(f"  Difference:   {yaw_hat - yaw_expected:+8.2f}°")

    # Check
    if abs(yaw_hat - yaw_expected) < 1.0:
        print(f"\n✓ CORRECT: Wahba returns expected rotation")
        return True
    elif abs(yaw_hat - (-yaw_expected)) < 1.0:
        print(f"\n✗ BUG: Wahba returns NEGATIVE of expected rotation!")
        print(f"  R_hat ≈ Rz({yaw_hat:.1f}°) instead of Rz(+{yaw_expected}°)")
        print(f"  The attitude matrix may have mu_map and mu_scan swapped.")
        return False
    else:
        print(f"\n⚠ UNEXPECTED: Wahba result doesn't match either expectation")
        print(f"  Need further investigation")
        return False


def main():
    print("=" * 60)
    print("WAHBA SVD SIGN CONVENTION DIAGNOSTIC")
    print("=" * 60)

    all_passed = True

    # Test 1: Identity rotation
    R_identity = np.eye(3)
    all_passed &= test_wahba_with_known_rotation(R_identity, "Identity rotation")

    # Test 2: +30° yaw rotation
    yaw_rad = np.radians(30)
    R_yaw_pos = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    all_passed &= test_wahba_with_known_rotation(R_yaw_pos, "+30° yaw rotation")

    # Test 3: -30° yaw rotation
    R_yaw_neg = R_yaw_pos.T
    all_passed &= test_wahba_with_known_rotation(R_yaw_neg, "-30° yaw rotation")

    # Test 4: +45° pitch rotation
    pitch_rad = np.radians(45)
    R_pitch = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    all_passed &= test_wahba_with_known_rotation(R_pitch, "+45° pitch rotation")

    # Test 5: Incremental SLAM simulation
    all_passed &= test_wahba_incremental()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if all_passed:
        print("✓ All tests passed - Wahba sign convention is correct")
        print("  The bug must be elsewhere (frame transforms, map accumulation, etc.)")
    else:
        print("✗ Some tests failed - Wahba has a sign convention issue")
        print("  Recommended fix: Transpose R_hat or swap mu_map/mu_scan")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
