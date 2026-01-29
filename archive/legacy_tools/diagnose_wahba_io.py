#!/usr/bin/env python3
"""
Diagnostic to dump actual Wahba inputs (mu_map, mu_scan) and output (R_hat).

This verifies:
1. Whether R_hat @ mu_scan ≈ mu_map (Wahba is solving the right problem)
2. The relationship between mu_map, mu_scan, and the actual robot motion

Usage:
    python3 tools/diagnose_wahba_io.py <diagnostics.npz>
"""

import sys
import numpy as np
from pathlib import Path


def load_diagnostics(npz_path: str) -> dict:
    """Load diagnostics NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def analyze_wahba_io(diag: dict):
    """Analyze Wahba inputs and outputs."""

    # Check what's available
    print("Available keys in diagnostics:")
    for k in sorted(diag.keys()):
        v = diag[k]
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

    # Check if we have the detailed Wahba diagnostic data
    if 'R_hat' not in diag:
        print("\nERROR: R_hat not found in diagnostics")
        print("The pipeline may not be saving Wahba outputs.")
        print("Available rotation-related keys:", [k for k in diag.keys() if 'rot' in k.lower() or 'yaw' in k.lower() or 'R_' in k])
        return

    print("\n" + "=" * 70)
    print("WAHBA I/O ANALYSIS")
    print("=" * 70)

    R_hat = diag.get('R_hat')

    if R_hat is not None and R_hat.ndim == 2:
        # Single scan
        print(f"\nR_hat (single scan):")
        print(R_hat)
        yaw = np.degrees(np.arctan2(R_hat[1, 0], R_hat[0, 0]))
        print(f"Yaw from R_hat: {yaw:.2f}°")
    elif R_hat is not None and R_hat.ndim == 3:
        # Multiple scans
        n_scans = R_hat.shape[0]
        print(f"\nFound {n_scans} R_hat matrices")

        # Show first few
        for i in range(min(5, n_scans)):
            R_i = R_hat[i]
            yaw_i = np.degrees(np.arctan2(R_i[1, 0], R_i[0, 0]))
            print(f"\nScan {i+1}: R_hat yaw = {yaw_i:+.2f}°")

    # Look for yaw increments
    if 'dyaw_wahba' in diag and 'dyaw_gyro' in diag:
        dyaw_wahba = diag['dyaw_wahba']
        dyaw_gyro = diag['dyaw_gyro']
        dyaw_odom = diag.get('dyaw_odom', np.zeros_like(dyaw_wahba))

        print("\n" + "=" * 70)
        print("YAW INCREMENT COMPARISON (first 10 meaningful scans)")
        print("=" * 70)

        # Find meaningful rotations
        threshold = 0.5  # degrees
        meaningful = np.abs(dyaw_gyro) > threshold
        indices = np.where(meaningful)[0][:10]

        print(f"\n{'Scan':>6} {'Gyro':>10} {'Odom':>10} {'Wahba':>10} {'G↔W sign':>10}")
        print("-" * 60)

        for i in indices:
            sign_match = "SAME" if dyaw_gyro[i] * dyaw_wahba[i] > 0 else "OPPOSITE"
            print(f"{i:6d} {dyaw_gyro[i]:+10.2f}° {dyaw_odom[i]:+10.2f}° {dyaw_wahba[i]:+10.2f}° {sign_match:>10}")

        # Overall statistics
        meaningful_mask = np.abs(dyaw_gyro) > threshold
        if np.sum(meaningful_mask) > 0:
            same_sign = np.sign(dyaw_gyro[meaningful_mask]) == np.sign(dyaw_wahba[meaningful_mask])
            pct_same = 100 * np.mean(same_sign)
            print(f"\n{np.sum(meaningful_mask)} meaningful rotations, {pct_same:.1f}% have same sign")


def main():
    if len(sys.argv) < 2:
        # Try to find the most recent diagnostics file
        results_dir = Path("/home/will/Documents/Coding/Phantom Fellowship MIT/Impact_Project_v1/results")
        npz_files = list(results_dir.glob("**/diagnostics*.npz"))
        if npz_files:
            npz_path = str(max(npz_files, key=lambda p: p.stat().st_mtime))
            print(f"Using most recent diagnostics: {npz_path}")
        else:
            print("Usage: python3 tools/diagnose_wahba_io.py <diagnostics.npz>")
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
        analyze_wahba_io(diag)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
