#!/usr/bin/env python3
"""
Trace Wahba inputs and outputs at runtime.

This script patches the wahba_svd function to dump its inputs and outputs,
then runs a simple test to see what's happening.
"""

import sys
sys.path.insert(0, "/home/will/Documents/Coding/Phantom Fellowship MIT/Impact_Project_v1/fl_ws/src/fl_slam_poc")

import numpy as np
from fl_slam_poc.common.jax_init import jnp
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.backend.operators.wahba import wahba_svd, _wahba_svd_core


def test_wahba_with_realistic_scenario():
    """
    Simulate the exact scenario from scan 2:
    - Map has directions from scan 1 (at identity pose)
    - Scan 2's robot has rotated +30 degrees yaw
    """
    print("=" * 70)
    print("WAHBA RUNTIME TRACE - REALISTIC SCENARIO")
    print("=" * 70)

    # Create test directions (Fibonacci sphere with 48 bins)
    def create_fibonacci_dirs(n_dirs=48):
        phi = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_dirs)
        theta = 2 * np.pi * indices / phi
        z = 1 - (2 * indices + 1) / n_dirs
        r = np.sqrt(1 - z**2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y, z], axis=1)

    n_bins = 48

    # Scan 1 directions in body frame (body1 = world since scan 1 at identity)
    d_body1 = create_fibonacci_dirs(n_bins)  # (48, 3)

    # On scan 1: map is initialized with R_for_map = I (identity pose)
    # So mu_map = d_body1 (same as world directions since R=I)
    R_scan1 = np.eye(3)
    S_dir_map = R_scan1 @ d_body1.T  # Map stores R_scan1 @ d_body1 = d_body1
    mu_map = S_dir_map.T / np.linalg.norm(S_dir_map.T, axis=1, keepdims=True)

    print(f"\nScan 1 setup:")
    print(f"  R_scan1 (yaw): {np.degrees(np.arctan2(R_scan1[1,0], R_scan1[0,0])):.1f}°")
    print(f"  Map initialized with scan 1 directions (in world frame)")

    # On scan 2: robot has rotated +30 degrees yaw
    yaw_rad = np.radians(30)
    R_scan2 = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    print(f"\nScan 2 setup:")
    print(f"  Robot yaw: {np.degrees(np.arctan2(R_scan2[1,0], R_scan2[0,0])):.1f}°")

    # Scan 2 sees the same world features, but in body2 frame
    # A world direction d_world = d_body1 (since scan 1 was at identity)
    # appears in body2 as: d_body2 = R_scan2^T @ d_world = R_scan2^T @ d_body1
    d_body2 = (R_scan2.T @ d_body1.T).T  # (48, 3)

    # mu_scan = normalized scan 2 directions in body frame
    mu_scan = d_body2 / np.linalg.norm(d_body2, axis=1, keepdims=True)

    # Check one bin to verify directions
    print(f"\nBin 0 (check):")
    print(f"  mu_map[0]:  {mu_map[0]}")
    print(f"  mu_scan[0]: {mu_scan[0]}")
    print(f"  Expected mu_scan[0] = R_scan2.T @ mu_map[0]: {R_scan2.T @ mu_map[0]}")

    # Wahba weights (simple uniform)
    weights = np.ones(n_bins)

    # Run Wahba
    result, cert, effect = wahba_svd(
        mu_map=jnp.array(mu_map),
        mu_scan=jnp.array(mu_scan),
        weights=jnp.array(weights),
    )

    R_hat = np.array(result.R_hat)
    yaw_hat = np.degrees(np.arctan2(R_hat[1, 0], R_hat[0, 0]))

    print(f"\nWahba result:")
    print(f"  R_hat yaw: {yaw_hat:.2f}°")
    print(f"  Expected:  {np.degrees(yaw_rad):.2f}° (same as robot yaw)")
    print(f"  R_hat:")
    print(R_hat)

    # Verify R_hat @ mu_scan ≈ mu_map
    transformed = (R_hat @ mu_scan.T).T
    residual = np.linalg.norm(transformed - mu_map, axis=1).mean()
    print(f"\n  Mean ||R_hat @ mu_scan - mu_map||: {residual:.6f}")

    # Check sign
    if abs(yaw_hat - np.degrees(yaw_rad)) < 5:
        print(f"\n✓ CORRECT: Wahba returns expected rotation")
    elif abs(yaw_hat - (-np.degrees(yaw_rad))) < 5:
        print(f"\n✗ BUG: Wahba returns NEGATIVE rotation!")
    else:
        print(f"\n⚠ UNEXPECTED: Wahba result doesn't match either expectation")

    # Additional check: what does the attitude matrix look like?
    print(f"\n" + "=" * 70)
    print("ATTITUDE MATRIX ANALYSIS")
    print("=" * 70)

    B = np.einsum('b,bi,bj->ij', weights, mu_map, mu_scan)
    print(f"\nAttitude matrix B:")
    print(B)

    U, S, Vt = np.linalg.svd(B)
    print(f"\nSVD singular values: {S}")
    print(f"det(U @ Vt) = {np.linalg.det(U @ Vt):.4f}")

    # Check if swapping mu_map and mu_scan gives the transpose
    B_swapped = np.einsum('b,bi,bj->ij', weights, mu_scan, mu_map)
    R_swapped = wahba_svd(
        mu_map=jnp.array(mu_scan),  # Swapped!
        mu_scan=jnp.array(mu_map),  # Swapped!
        weights=jnp.array(weights),
    )[0].R_hat
    yaw_swapped = np.degrees(np.arctan2(R_swapped[1, 0], R_swapped[0, 0]))
    print(f"\nIf mu_map/mu_scan SWAPPED:")
    print(f"  R_swapped yaw: {yaw_swapped:.2f}° (should be -{np.degrees(yaw_rad):.2f}°)")


if __name__ == "__main__":
    test_wahba_with_realistic_scenario()
