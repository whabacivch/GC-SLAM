#!/usr/bin/env python3
"""
Analyze yaw increment invariants from diagnostic data.

This script compares yaw increments from three sources:
1. Gyro-integrated yaw (after IMU→base rotation)
2. Odom yaw increment
3. LiDAR/Wahba yaw increment

This helps identify sign mismatches in:
- Gyro (A/B/C): IMU→base rotation, axis sign flip, or left/right convention
- Global frame (D): LiDAR extrinsic
"""

import argparse
import numpy as np
from pathlib import Path


def analyze_yaw_invariants(diagnostics_path: str):
    """Load diagnostics and analyze yaw increment consistency."""
    print("=" * 80)
    print("YAW INCREMENT INVARIANT ANALYSIS")
    print("=" * 80)
    print()
    
    # Load diagnostics
    data = np.load(diagnostics_path, allow_pickle=True)
    
    if "dyaw_gyro" not in data:
        print("ERROR: Diagnostic file does not contain yaw increment data.")
        print("       Make sure you're using a recent run with invariant test enabled.")
        return
    
    dyaw_gyro = data["dyaw_gyro"]
    dyaw_odom = data["dyaw_odom"]
    dyaw_wahba = data["dyaw_wahba"]
    timestamps = data["timestamps"]
    scan_numbers = data["scan_numbers"]
    
    n_scans = len(dyaw_gyro)
    print(f"Loaded {n_scans} scans from {diagnostics_path}")
    print()
    
    # Analyze sign consistency
    print("=" * 80)
    print("SIGN CONSISTENCY ANALYSIS")
    print("=" * 80)
    print()
    
    # Count sign matches/mismatches
    gyro_wahba_same = np.sign(dyaw_gyro) == np.sign(dyaw_wahba)
    gyro_odom_same = np.sign(dyaw_gyro) == np.sign(dyaw_odom)
    odom_wahba_same = np.sign(dyaw_odom) == np.sign(dyaw_wahba)
    
    print(f"Gyro ↔ Wahba: {np.sum(gyro_wahba_same)}/{n_scans} scans have same sign ({100*np.mean(gyro_wahba_same):.1f}%)")
    print(f"Gyro ↔ Odom:  {np.sum(gyro_odom_same)}/{n_scans} scans have same sign ({100*np.mean(gyro_odom_same):.1f}%)")
    print(f"Odom ↔ Wahba: {np.sum(odom_wahba_same)}/{n_scans} scans have same sign ({100*np.mean(odom_wahba_same):.1f}%)")
    print()
    
    # Show first 20 scans in detail
    print("=" * 80)
    print("DETAILED COMPARISON (First 20 scans)")
    print("=" * 80)
    print()
    print(f"{'Scan':<6} {'Δyaw_gyro':>12} {'Δyaw_odom':>12} {'Δyaw_wahba':>12} {'Gyro↔Wahba':>12} {'Gyro↔Odom':>12} {'Odom↔Wahba':>12}")
    print("-" * 80)
    
    for i in range(min(20, n_scans)):
        scan_num = scan_numbers[i]
        gyro = dyaw_gyro[i]
        odom = dyaw_odom[i]
        wahba = dyaw_wahba[i]
        
        match_gw = "✓" if gyro_wahba_same[i] else "✗"
        match_go = "✓" if gyro_odom_same[i] else "✗"
        match_ow = "✓" if odom_wahba_same[i] else "✗"
        
        print(f"{scan_num:<6} {gyro:+11.2f}° {odom:+11.2f}° {wahba:+11.2f}° {match_gw:>12} {match_go:>12} {match_ow:>12}")
    
    print()
    
    # Statistical analysis
    print("=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    print()
    
    # Compute correlation coefficients
    def correlation(x, y):
        """Compute correlation coefficient, handling zero variance."""
        x = np.array(x)
        y = np.array(y)
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std == 0 or y_std == 0:
            return 0.0
        return np.mean(x_centered * y_centered) / (x_std * y_std)
    
    corr_gyro_wahba = correlation(dyaw_gyro, dyaw_wahba)
    corr_gyro_odom = correlation(dyaw_gyro, dyaw_odom)
    corr_odom_wahba = correlation(dyaw_odom, dyaw_wahba)
    
    print(f"Correlation coefficients:")
    print(f"  Gyro ↔ Wahba: {corr_gyro_wahba:+.3f}")
    print(f"  Gyro ↔ Odom:  {corr_gyro_odom:+.3f}")
    print(f"  Odom ↔ Wahba: {corr_odom_wahba:+.3f}")
    print()
    
    # Mean magnitudes
    print(f"Mean absolute yaw increments:")
    print(f"  Gyro:  {np.mean(np.abs(dyaw_gyro)):.2f}°")
    print(f"  Odom:  {np.mean(np.abs(dyaw_odom)):.2f}°")
    print(f"  Wahba: {np.mean(np.abs(dyaw_wahba)):.2f}°")
    print()
    
    # Identify problematic scans
    print("=" * 80)
    print("PROBLEMATIC SCANS (Sign mismatches)")
    print("=" * 80)
    print()
    
    mismatches = ~gyro_wahba_same
    if np.any(mismatches):
        print(f"Scans where Gyro and Wahba have opposite signs ({np.sum(mismatches)} scans):")
        for i in np.where(mismatches)[0][:10]:  # Show first 10
            print(f"  Scan {scan_numbers[i]}: gyro={dyaw_gyro[i]:+.2f}°, wahba={dyaw_wahba[i]:+.2f}°")
        if np.sum(mismatches) > 10:
            print(f"  ... and {np.sum(mismatches) - 10} more")
    else:
        print("No sign mismatches between Gyro and Wahba!")
    
    print()
    
    # Interpretation guide
    print("=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print()
    print("If Gyro and Wahba have opposite signs consistently:")
    print("  → Sign mismatch in gyro processing (A/B/C):")
    print("     A) IMU→base rotation (T_base_imu) may be wrong")
    print("     B) Gyro axis sign flip (check gyro reading convention)")
    print("     C) Left/right hand convention mismatch")
    print()
    print("If Odom and Wahba have opposite signs consistently:")
    print("  → Sign mismatch in global frame (D):")
    print("     D) LiDAR extrinsic (T_base_lidar) may be wrong")
    print()
    print("If all three disagree:")
    print("  → Multiple issues or frame convention problems")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze yaw increment invariants from diagnostic data"
    )
    parser.add_argument(
        "diagnostics_path",
        type=str,
        help="Path to diagnostics.npz file (e.g., results/gc_YYYYMMDD_HHMMSS/diagnostics.npz)",
    )
    
    args = parser.parse_args()
    
    diagnostics_path = Path(args.diagnostics_path)
    if not diagnostics_path.exists():
        print(f"ERROR: Diagnostics file not found: {diagnostics_path}")
        return 1
    
    analyze_yaw_invariants(str(diagnostics_path))
    return 0


if __name__ == "__main__":
    exit(main())
