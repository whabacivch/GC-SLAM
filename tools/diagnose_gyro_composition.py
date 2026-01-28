#!/usr/bin/env python3
"""
Diagnostic script to detect left/right composition mismatch in gyro evidence.

This tests the key hypothesis: is the gyro rotation increment being applied
on the wrong side of the state rotation?

The test:
1. Create a known rotation state R_start
2. Apply a known gyro delta (e.g., +10 deg yaw)
3. Check if evidence pushes state in the CORRECT direction

If there's a composition-side mismatch:
- Evidence would push state by -delta instead of +delta
- Or equivalently, R^T @ M instead of M @ R^T
"""

import sys
sys.path.insert(0, "/home/will/Documents/Coding/Phantom Fellowship MIT/Impact_Project_v1/fl_ws/src/fl_slam_poc")

import numpy as np
from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.backend.operators.imu_gyro_evidence import imu_gyro_rotation_evidence


def test_gyro_evidence_direction():
    """
    Test: If gyro says "rotate +10 deg about Z", does evidence push state in +Z direction?
    """
    print("=" * 70)
    print("GYRO EVIDENCE COMPOSITION-SIDE DIAGNOSTIC")
    print("=" * 70)

    # Start at identity (yaw = 0)
    rotvec_start = np.array([0.0, 0.0, 0.0])

    # Predict is also at identity (no prediction motion model)
    rotvec_pred = np.array([0.0, 0.0, 0.0])

    # Gyro says: rotate +10 degrees about Z (CCW when looking down +Z)
    delta_yaw_deg = 10.0
    delta_yaw_rad = np.radians(delta_yaw_deg)
    delta_rotvec = np.array([0.0, 0.0, delta_yaw_rad])

    # Gyro covariance (arbitrary small value)
    Sigma_g = 0.001 * np.eye(3)
    dt_int = 0.1  # 100ms integration time

    print(f"\nInputs:")
    print(f"  rotvec_start (deg): {np.degrees(rotvec_start)}")
    print(f"  rotvec_pred (deg):  {np.degrees(rotvec_pred)}")
    print(f"  delta_rotvec (deg): {np.degrees(delta_rotvec)}")
    print(f"  dt_int: {dt_int}")

    # Run the gyro evidence
    result, cert, effect = imu_gyro_rotation_evidence(
        rotvec_start_WB=jnp.array(rotvec_start),
        rotvec_end_pred_WB=jnp.array(rotvec_pred),
        delta_rotvec_meas=jnp.array(delta_rotvec),
        Sigma_g=jnp.array(Sigma_g),
        dt_int=dt_int,
    )

    r_rot = np.array(result.r_rot)
    L_gyro = np.array(result.L_gyro)
    h_gyro = np.array(result.h_gyro)

    print(f"\nOutputs:")
    print(f"  r_rot (deg): {np.degrees(r_rot)}")
    print(f"  h_gyro[3:6]: {h_gyro[3:6]}")

    # The residual should be the delta (since pred ≈ start)
    print(f"\nExpected r_rot ≈ delta_rotvec:")
    print(f"  |r_rot - delta_rotvec| = {np.linalg.norm(r_rot - delta_rotvec):.6e}")

    # Check: if we solve for the posterior with zero prior h,
    # the mode should be approximately r_rot
    L_rot = L_gyro[3:6, 3:6]
    h_rot = h_gyro[3:6]

    # With zero prior: delta_theta_post = L_rot^{-1} @ h_rot
    delta_theta_post = np.linalg.solve(L_rot, h_rot)

    print(f"\nPosterior rotation increment (assuming zero prior):")
    print(f"  delta_theta_post (deg): {np.degrees(delta_theta_post)}")

    # KEY TEST: Does delta_theta_post have the SAME SIGN as delta_rotvec?
    print(f"\n" + "=" * 70)
    print("COMPOSITION-SIDE TEST")
    print("=" * 70)

    # For yaw (Z component):
    gyro_says_positive_yaw = delta_rotvec[2] > 0
    evidence_pushes_positive_yaw = delta_theta_post[2] > 0

    print(f"\nGyro delta yaw: {'POSITIVE' if gyro_says_positive_yaw else 'NEGATIVE'} ({np.degrees(delta_rotvec[2]):.2f} deg)")
    print(f"Evidence push:  {'POSITIVE' if evidence_pushes_positive_yaw else 'NEGATIVE'} ({np.degrees(delta_theta_post[2]):.2f} deg)")

    if gyro_says_positive_yaw == evidence_pushes_positive_yaw:
        print("\n✓ CORRECT: Evidence pushes state in the SAME direction as gyro measurement")
        print("  -> No composition-side mismatch detected in gyro evidence")
    else:
        print("\n✗ BUG DETECTED: Evidence pushes state in OPPOSITE direction from gyro measurement!")
        print("  -> This indicates a LEFT/RIGHT composition mismatch!")
        print("  -> The residual sign or composition order is wrong")

    # Additional test: verify with the full recompose logic
    print(f"\n" + "=" * 70)
    print("FULL COMPOSITION CHAIN TEST")
    print("=" * 70)

    # If we apply delta_theta_post to R_pred via RIGHT composition:
    R_pred = se3_jax.so3_exp(jnp.array(rotvec_pred))
    R_new_right = np.array(R_pred) @ np.array(se3_jax.so3_exp(jnp.array(delta_theta_post)))

    # What we expect (IMU-predicted end orientation):
    R_start_mat = np.array(se3_jax.so3_exp(jnp.array(rotvec_start)))
    R_delta_mat = np.array(se3_jax.so3_exp(jnp.array(delta_rotvec)))
    R_expected = R_start_mat @ R_delta_mat  # Right composition

    # Extract yaw from both
    def yaw_from_R(R):
        return np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    yaw_new_right = yaw_from_R(R_new_right)
    yaw_expected = yaw_from_R(R_expected)

    print(f"\nExpected end yaw (from IMU): {yaw_expected:.2f} deg")
    print(f"Actual end yaw (after applying evidence): {yaw_new_right:.2f} deg")
    print(f"Difference: {yaw_new_right - yaw_expected:.4f} deg")

    if abs(yaw_new_right - yaw_expected) < 0.1:
        print("\n✓ CORRECT: State reaches expected orientation after applying evidence")
    else:
        print("\n✗ MISMATCH: State does NOT reach expected orientation!")
        print("  -> Check composition order in recompose step")

    return result, cert, effect


def test_negative_gyro():
    """Test with negative gyro delta to verify symmetry."""
    print(f"\n" + "=" * 70)
    print("NEGATIVE GYRO DELTA TEST")
    print("=" * 70)

    rotvec_start = np.array([0.0, 0.0, 0.0])
    rotvec_pred = np.array([0.0, 0.0, 0.0])

    # Gyro says: rotate -10 degrees about Z
    delta_rotvec = np.array([0.0, 0.0, np.radians(-10.0)])
    Sigma_g = 0.001 * np.eye(3)
    dt_int = 0.1

    result, cert, effect = imu_gyro_rotation_evidence(
        rotvec_start_WB=jnp.array(rotvec_start),
        rotvec_end_pred_WB=jnp.array(rotvec_pred),
        delta_rotvec_meas=jnp.array(delta_rotvec),
        Sigma_g=jnp.array(Sigma_g),
        dt_int=dt_int,
    )

    L_rot = np.array(result.L_gyro)[3:6, 3:6]
    h_rot = np.array(result.h_gyro)[3:6]
    delta_theta_post = np.linalg.solve(L_rot, h_rot)

    print(f"Gyro delta yaw: {np.degrees(delta_rotvec[2]):.2f} deg")
    print(f"Evidence push:  {np.degrees(delta_theta_post[2]):.2f} deg")

    if (delta_rotvec[2] < 0) == (delta_theta_post[2] < 0):
        print("✓ CORRECT: Negative gyro gives negative evidence push")
    else:
        print("✗ BUG: Sign mismatch with negative gyro!")


if __name__ == "__main__":
    test_gyro_evidence_direction()
    test_negative_gyro()
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
