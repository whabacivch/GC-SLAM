#!/usr/bin/env python3
"""
Diagnostic to analyze yaw increment consistency between gyro, odom, and Wahba.

This tool analyzes NPZ diagnostic files to check for systematic sign mismatches
between the three yaw sources.

If there's a composition-side bug, we'd expect to see:
- gyro yaw increments consistently opposite to odom/Wahba
- OR gyro correct but state moving in wrong direction

Usage:
    python3 tools/diagnose_yaw_mismatch.py <diagnostics.npz>
"""

import sys
import numpy as np
from pathlib import Path


def load_diagnostics(npz_path: str) -> dict:
    """Load diagnostics NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def analyze_yaw_consistency(diag: dict):
    """Analyze yaw increment consistency between gyro, odom, and Wahba."""

    # Check if yaw diagnostics are present
    if 'dyaw_gyro' not in diag:
        print("ERROR: No yaw diagnostics found in NPZ file")
        print("Available keys:", list(diag.keys()))
        return

    dyaw_gyro = diag['dyaw_gyro']
    dyaw_odom = diag['dyaw_odom']
    dyaw_wahba = diag['dyaw_wahba']

    n_scans = len(dyaw_gyro)
    print(f"Analyzing {n_scans} scans...")
    print()

    # Filter out near-zero values (no meaningful rotation)
    threshold_deg = 0.5  # Minimum yaw change to consider
    mask = (np.abs(dyaw_gyro) > threshold_deg) | (np.abs(dyaw_odom) > threshold_deg)
    n_meaningful = np.sum(mask)
    print(f"Scans with meaningful rotation (|Δyaw| > {threshold_deg}°): {n_meaningful}")

    if n_meaningful == 0:
        print("No meaningful rotations to analyze")
        return

    # Sign agreement analysis
    gyro_odom_same_sign = np.sign(dyaw_gyro[mask]) == np.sign(dyaw_odom[mask])
    gyro_wahba_same_sign = np.sign(dyaw_gyro[mask]) == np.sign(dyaw_wahba[mask])
    odom_wahba_same_sign = np.sign(dyaw_odom[mask]) == np.sign(dyaw_wahba[mask])

    pct_gyro_odom = 100 * np.mean(gyro_odom_same_sign)
    pct_gyro_wahba = 100 * np.mean(gyro_wahba_same_sign)
    pct_odom_wahba = 100 * np.mean(odom_wahba_same_sign)

    print()
    print("=" * 60)
    print("SIGN AGREEMENT ANALYSIS (meaningful rotations only)")
    print("=" * 60)
    print(f"  Gyro ↔ Odom:  {pct_gyro_odom:5.1f}% same sign")
    print(f"  Gyro ↔ Wahba: {pct_gyro_wahba:5.1f}% same sign")
    print(f"  Odom ↔ Wahba: {pct_odom_wahba:5.1f}% same sign")

    # Interpretation
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if pct_gyro_odom < 40:
        print("⚠ WARNING: Gyro and Odom have OPPOSITE signs most of the time!")
        print("  Possible causes:")
        print("    - Wrong IMU extrinsic rotation (R_base_imu)")
        print("    - Gyro driver sign convention mismatch")
        print("    - Odom pose interpretation error (parent/child swap)")
    elif pct_gyro_odom < 60:
        print("⚠ WARNING: Gyro and Odom sign agreement is inconsistent")
        print("  May indicate noisy data or marginal alignment")
    else:
        print("✓ Gyro and Odom generally agree on rotation direction")

    if pct_gyro_wahba < 40:
        print("⚠ WARNING: Gyro and Wahba (LiDAR) have OPPOSITE signs!")
        print("  Possible causes:")
        print("    - Wrong LiDAR extrinsic rotation")
        print("    - Wahba pairing direction error (map/scan swapped)")
    elif pct_gyro_wahba < 60:
        print("⚠ WARNING: Gyro and Wahba sign agreement is inconsistent")
    else:
        print("✓ Gyro and Wahba generally agree on rotation direction")

    # Magnitude comparison
    print()
    print("=" * 60)
    print("MAGNITUDE ANALYSIS")
    print("=" * 60)

    ratio_gyro_odom = np.abs(dyaw_gyro[mask]) / (np.abs(dyaw_odom[mask]) + 1e-6)
    ratio_gyro_wahba = np.abs(dyaw_gyro[mask]) / (np.abs(dyaw_wahba[mask]) + 1e-6)

    print(f"  |Gyro| / |Odom|:  median={np.median(ratio_gyro_odom):.2f}, mean={np.mean(ratio_gyro_odom):.2f}")
    print(f"  |Gyro| / |Wahba|: median={np.median(ratio_gyro_wahba):.2f}, mean={np.mean(ratio_gyro_wahba):.2f}")

    # Sample values
    print()
    print("=" * 60)
    print("SAMPLE VALUES (first 10 meaningful rotations)")
    print("=" * 60)
    print(f"{'Scan':>6} {'Gyro':>10} {'Odom':>10} {'Wahba':>10} {'G↔O':>5} {'G↔W':>5}")
    print("-" * 60)

    indices = np.where(mask)[0][:10]
    for i in indices:
        go_match = "YES" if np.sign(dyaw_gyro[i]) == np.sign(dyaw_odom[i]) else "NO"
        gw_match = "YES" if np.sign(dyaw_gyro[i]) == np.sign(dyaw_wahba[i]) else "NO"
        print(f"{i:6d} {dyaw_gyro[i]:10.2f}° {dyaw_odom[i]:10.2f}° {dyaw_wahba[i]:10.2f}° {go_match:>5} {gw_match:>5}")


def main():
    if len(sys.argv) < 2:
        # Try to find the most recent diagnostics file
        results_dir = Path("/home/will/Documents/Coding/Phantom Fellowship MIT/Impact_Project_v1/results")
        npz_files = list(results_dir.glob("**/diagnostics*.npz"))
        if npz_files:
            npz_path = str(max(npz_files, key=lambda p: p.stat().st_mtime))
            print(f"Using most recent diagnostics: {npz_path}")
        else:
            print("Usage: python3 tools/diagnose_yaw_mismatch.py <diagnostics.npz>")
            print("\nNo diagnostics files found in results/")
            sys.exit(1)
    else:
        npz_path = sys.argv[1]

    if not Path(npz_path).exists():
        print(f"ERROR: File not found: {npz_path}")
        sys.exit(1)

    print(f"Loading: {npz_path}")
    print()

    try:
        diag = load_diagnostics(npz_path)
        analyze_yaw_consistency(diag)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
