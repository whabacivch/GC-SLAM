#!/usr/bin/env python3
"""
Check the actual mounting angle of the Livox Mid-360 lidar by analyzing point cloud data.

This script:
1. Reads point clouds from the bag
2. Analyzes point distributions when robot is stationary
3. Determines if lidar is mounted horizontally or at an angle
"""

import argparse
import sqlite3
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add project root to path
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rosbag_sqlite_utils import resolve_db3_path, topic_id, topic_type


def analyze_pointcloud_orientation(points: np.ndarray) -> dict:
    """
    Analyze point cloud to determine mounting orientation.
    
    Returns statistics about point distribution.
    """
    if points.shape[0] == 0:
        return None
    
    # Compute principal directions using PCA
    points_centered = points - np.mean(points, axis=0)
    cov = np.cov(points_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Principal directions
    pc1 = eigvecs[:, 0]  # Largest variance direction
    pc2 = eigvecs[:, 1]
    pc3 = eigvecs[:, 2]  # Smallest variance direction
    
    # Check angle of principal components relative to world axes
    # If lidar is horizontal, Z should be close to vertical (0,0,1)
    # If lidar is at 45°, Z will be at ~45° to vertical
    
    z_axis = np.array([0, 0, 1])
    
    # Angle between PC1 and Z-axis
    dot_pc1_z = np.abs(np.dot(pc1, z_axis))
    angle_pc1_z = np.arccos(np.clip(dot_pc1_z, -1, 1)) * 180 / np.pi
    
    # Angle between PC3 and Z-axis (should be small if horizontal)
    dot_pc3_z = np.abs(np.dot(pc3, z_axis))
    angle_pc3_z = np.arccos(np.clip(dot_pc3_z, -1, 1)) * 180 / np.pi
    
    # Point distribution statistics
    z_range = np.max(points[:, 2]) - np.min(points[:, 2])
    z_mean = np.mean(points[:, 2])
    z_std = np.std(points[:, 2])
    
    # Check if points are mostly in a horizontal plane (small Z variance)
    # or distributed vertically (large Z variance)
    horizontal_ratio = eigvals[2] / (eigvals[0] + 1e-12)  # Smallest / largest
    
    return {
        'n_points': points.shape[0],
        'pc1': pc1,
        'pc2': pc2,
        'pc3': pc3,
        'eigvals': eigvals,
        'angle_pc1_z_deg': angle_pc1_z,
        'angle_pc3_z_deg': angle_pc3_z,
        'z_range': z_range,
        'z_mean': z_mean,
        'z_std': z_std,
        'horizontal_ratio': horizontal_ratio,
        'points_mean': np.mean(points, axis=0),
        'points_std': np.std(points, axis=0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check Livox Mid-360 lidar mounting angle from point cloud data.")
    ap.add_argument("bag_path", help="Bag directory containing *.db3")
    ap.add_argument("--lidar-topic", default="/livox/mid360/lidar")
    ap.add_argument("--n-scans", type=int, default=10, help="Number of scans to analyze")
    ap.add_argument("--skip-scans", type=int, default=50, help="Skip first N scans")
    args = ap.parse_args()
    
    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        print(f"ERROR: Could not locate *.db3 under '{args.bag_path}'", file=sys.stderr)
        return 1
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    try:
        tid = topic_id(cur, args.lidar_topic)
        if tid is None:
            print(f"ERROR: Topic not found: {args.lidar_topic}", file=sys.stderr)
            return 2
        
        print("=" * 70)
        print("LIDAR MOUNTING ANGLE ANALYSIS")
        print("=" * 70)
        print(f"Bag: {db_path}")
        print(f"Topic: {args.lidar_topic}")
        print(f"Analyzing {args.n_scans} scans (skipping first {args.skip_scans})")
        print()
        
        import rclpy
        from rclpy.serialization import deserialize_message
        
        rclpy.init()
        
        scan_count = 0
        all_stats = []
        
        # Read messages
        for row in cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (tid,),
        ):
            if scan_count < args.skip_scans:
                scan_count += 1
                continue
            
            ts, data = row
            try:
                # Try to parse as CustomMsg
                from livox_ros_driver2.msg import CustomMsg
                msg = deserialize_message(data, CustomMsg)
                
                # Extract points from CustomMsg
                points_list = []
                for point in msg.points:
                    if point.line < 6:  # Valid point
                        x = point.x / 1000.0  # mm to m
                        y = point.y / 1000.0
                        z = point.z / 1000.0
                        points_list.append([x, y, z])
                
                if len(points_list) == 0:
                    continue
                
                points = np.array(points_list, dtype=np.float64)
                
                # Analyze orientation
                stats = analyze_pointcloud_orientation(points)
                if stats:
                    all_stats.append(stats)
                    scan_count += 1
                    
                    if scan_count >= args.n_scans + args.skip_scans:
                        break
                        
            except Exception as e:
                print(f"Warning: Failed to parse scan at {ts}: {e}", file=sys.stderr)
                continue
        
        rclpy.shutdown()
        
        if len(all_stats) == 0:
            print("ERROR: No valid point clouds found", file=sys.stderr)
            return 3
        
        # Aggregate statistics
        print(f"Successfully analyzed {len(all_stats)} scans")
        print()
        
        # Average angles
        avg_angle_pc1_z = np.mean([s['angle_pc1_z_deg'] for s in all_stats])
        avg_angle_pc3_z = np.mean([s['angle_pc3_z_deg'] for s in all_stats])
        avg_horizontal_ratio = np.mean([s['horizontal_ratio'] for s in all_stats])
        avg_z_range = np.mean([s['z_range'] for s in all_stats])
        avg_z_std = np.mean([s['z_std'] for s in all_stats])
        
        print("POINT CLOUD ORIENTATION ANALYSIS")
        print("-" * 70)
        print(f"Average angle between PC1 (largest variance) and Z-axis: {avg_angle_pc1_z:.2f}°")
        print(f"Average angle between PC3 (smallest variance) and Z-axis: {avg_angle_pc3_z:.2f}°")
        print(f"Average horizontal ratio (smallest/largest eigenvalue): {avg_horizontal_ratio:.4f}")
        print(f"Average Z-range: {avg_z_range:.3f} m")
        print(f"Average Z-std: {avg_z_std:.3f} m")
        print()
        
        # Interpretation
        print("INTERPRETATION")
        print("-" * 70)
        
        # If lidar is horizontal, PC3 should be close to vertical (small angle)
        # If lidar is at 45°, PC1 or PC2 might be at 45° to vertical
        if avg_angle_pc3_z < 15:
            print("✓ Lidar appears to be mounted HORIZONTALLY")
            print(f"  (PC3 is {avg_angle_pc3_z:.1f}° from vertical, indicating horizontal scan plane)")
            print("  → Current T_base_lidar rotation [0,0,0] is likely CORRECT")
        elif avg_angle_pc1_z > 30 and avg_angle_pc1_z < 60:
            print("⚠ Lidar appears to be mounted at an ANGLE")
            print(f"  (PC1 is {avg_angle_pc1_z:.1f}° from vertical)")
            print("  → Current T_base_lidar rotation [0,0,0] may be INCORRECT")
            print("  → Consider estimating lidar extrinsics using estimate_lidar_base_extrinsic.py")
        else:
            print("? Unclear mounting orientation")
            print(f"  PC1-Z angle: {avg_angle_pc1_z:.1f}°, PC3-Z angle: {avg_angle_pc3_z:.1f}°")
        
        print()
        print("Z-DISTRIBUTION ANALYSIS")
        print("-" * 70)
        if avg_z_std < 0.5:
            print("✓ Points have small Z-variance (consistent with horizontal mounting)")
        elif avg_z_std > 2.0:
            print("⚠ Points have large Z-variance (may indicate angled mounting or 3D environment)")
        else:
            print("? Moderate Z-variance (check individual scans)")
        
        # Show sample principal components
        if all_stats:
            sample = all_stats[0]
            print()
            print("SAMPLE SCAN PRINCIPAL COMPONENTS (first scan)")
            print("-" * 70)
            print(f"PC1 (largest variance): {sample['pc1']}")
            print(f"PC2: {sample['pc2']}")
            print(f"PC3 (smallest variance): {sample['pc3']}")
            print(f"Eigenvalues: {sample['eigvals']}")
            print(f"Point mean: {sample['points_mean']}")
            print(f"Point std: {sample['points_std']}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
