#!/usr/bin/env python3
"""
Comprehensive coordinate frame and extrinsics diagnostic.

This script inspects the actual data from the rosbag to determine:
1. LiDAR frame convention (Z-up vs Z-down) by analyzing raw point clouds
2. IMU frame convention by analyzing gravity direction
3. Odom frame conventions and covariance ordering
4. Actual vs expected transformations

This is a FIRST-PRINCIPLES diagnostic - no guessing, only data.
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


def analyze_lidar_z_convention(points: np.ndarray) -> dict:
    """
    Determine if LiDAR is Z-up or Z-down by analyzing point distribution.
    
    Strategy:
    - If Z-up: ground points have negative Z (below sensor)
    - If Z-down: ground points have positive Z (below sensor)
    - Analyze dominant plane (ground) normal direction
    """
    if points.shape[0] < 100:
        return None
    
    # Center points
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # PCA to find dominant plane (ground)
    cov = np.cov(points_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Smallest eigenvalue = normal to dominant plane (ground)
    ground_normal = eigvecs[:, 0]  # Direction of smallest variance
    
    # Check if normal points up or down
    z_axis = np.array([0, 0, 1])
    dot_with_z = np.dot(ground_normal, z_axis)
    
    # Z distribution statistics
    z_mean = np.mean(points[:, 2])
    z_std = np.std(points[:, 2])
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    
    # Count points above/below sensor origin
    n_below = np.sum(points[:, 2] < 0)
    n_above = np.sum(points[:, 2] > 0)
    
    return {
        'ground_normal': ground_normal,
        'ground_normal_dot_z': dot_with_z,
        'z_mean': z_mean,
        'z_std': z_std,
        'z_min': z_min,
        'z_max': z_max,
        'n_below': n_below,
        'n_above': n_above,
        'n_total': points.shape[0],
        'eigvals': eigvals,
    }


def analyze_imu_gravity(accel_samples: np.ndarray) -> dict:
    """
    Analyze IMU accelerometer data to determine the *specific force* direction in sensor frame.

    IMPORTANT (by physics, not convention):
      - IMU accelerometers measure specific force (reaction to gravity), not gravity itself.
      - When stationary and level in a Z-up world, the expected *specific force* is +Z:
            gravity = [0, 0, -g]
            specific_force = -gravity = [0, 0, +g]
    """
    if accel_samples.shape[0] < 10:
        return None
    
    # Average accel direction (should point opposite to gravity)
    accel_mean = np.mean(accel_samples, axis=0)
    accel_norm = np.linalg.norm(accel_mean)
    
    # Normalize to get direction
    if accel_norm > 0.1:  # Should be ~9.81 m/s²
        accel_dir = accel_mean / accel_norm
    else:
        return None
    
    # Expected specific force direction in base_footprint (Z-up): +Z
    expected_specific_force_base = np.array([0, 0, 1])
    
    # Compare with actual
    dot_with_expected = np.dot(accel_dir, expected_specific_force_base)
    angle_to_expected = np.arccos(np.clip(dot_with_expected, -1, 1)) * 180 / np.pi
    
    return {
        'accel_mean': accel_mean,
        'accel_dir': accel_dir,
        'accel_magnitude': accel_norm,
        'expected_specific_force_base': expected_specific_force_base,
        'dot_with_expected': dot_with_expected,
        'angle_to_expected_deg': angle_to_expected,
    }


def analyze_odom_covariance(odom_cov: np.ndarray) -> dict:
    """
    Analyze odometry covariance to determine ordering convention.
    
    ROS convention: [x, y, z, roll, pitch, yaw]
    GC convention:  [x, y, z, roll, pitch, yaw]  (same as ROS)
    Legacy/permuted convention: [rx, ry, rz, tx, ty, tz]
    """
    if odom_cov.shape != (6, 6):
        return None
    
    # Extract diagonal variances
    diag = np.diag(odom_cov)
    
    # For 2D wheeled robot:
    # - x, y, yaw should have LOW variance (well-observed)
    # - z, roll, pitch should have HIGH variance (unobserved)
    
    # Check ROS ordering [x,y,z,roll,pitch,yaw]
    ros_xy_yaw = diag[[0, 1, 5]]  # x, y, yaw
    ros_z_rp = diag[[2, 3, 4]]    # z, roll, pitch
    
    # Check legacy/permuted ordering [rx,ry,rz,tx,ty,tz]
    perm_rot = diag[[0, 1, 2]]      # rx, ry, rz
    perm_trans = diag[[3, 4, 5]]    # tx, ty, tz
    
    ros_xy_yaw_mean = np.mean(ros_xy_yaw)
    ros_z_rp_mean = np.mean(ros_z_rp)
    perm_rot_mean = np.mean(perm_rot)
    perm_trans_mean = np.mean(perm_trans)
    
    # Determine which ordering makes sense
    # For 2D robot: xy_yaw should be small, z_rp should be large
    ros_makes_sense = ros_xy_yaw_mean < ros_z_rp_mean
    
    # For permuted: trans (xy) should be small, rot (rx,ry) should be large
    perm_makes_sense = perm_trans_mean < perm_rot_mean

    if ros_makes_sense:
        likely = 'ROS [x,y,z,roll,pitch,yaw] (GC matches ROS)'
    elif perm_makes_sense:
        likely = 'LEGACY PERMUTED [rx,ry,rz,tx,ty,tz]'
    else:
        likely = 'UNKNOWN'
    
    return {
        'diag': diag,
        'ros_xy_yaw_mean': ros_xy_yaw_mean,
        'ros_z_rp_mean': ros_z_rp_mean,
        'perm_rot_mean': perm_rot_mean,
        'perm_trans_mean': perm_trans_mean,
        'ros_makes_sense': ros_makes_sense,
        'perm_makes_sense': perm_makes_sense,
        'likely_ordering': likely,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnose coordinate frames and extrinsics from rosbag data"
    )
    ap.add_argument("bag_path", help="Bag directory containing *.db3")
    ap.add_argument("--lidar-topic", default="/livox/mid360/lidar")
    ap.add_argument("--imu-topic", default="/livox/mid360/imu")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--n-scans", type=int, default=20, help="Number of scans to analyze")
    args = ap.parse_args()
    
    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        print(f"ERROR: Could not locate *.db3 under '{args.bag_path}'", file=sys.stderr)
        return 1
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    try:
        import rclpy
        from rclpy.serialization import deserialize_message
        rclpy.init()
        
        print("=" * 80)
        print("COORDINATE FRAME DIAGNOSTIC")
        print("=" * 80)
        print(f"Bag: {db_path}")
        print()
        
        # =====================================================================
        # 1. LiDAR Frame Convention (Z-up vs Z-down)
        # =====================================================================
        print("1. LIDAR FRAME CONVENTION")
        print("-" * 80)
        
        lidar_tid = topic_id(cur, args.lidar_topic)
        if lidar_tid is None:
            print(f"  ERROR: Topic not found: {args.lidar_topic}")
        else:
            try:
                from livox_ros_driver2.msg import CustomMsg
                
                lidar_stats = []
                scan_count = 0
                
                for row in cur.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
                    (lidar_tid, args.n_scans * 10),  # Get more to filter valid ones
                ):
                    ts, data = row
                    try:
                        msg = deserialize_message(data, CustomMsg)
                        
                        # Extract raw points (in livox_frame, NO transforms applied)
                        points_list = []
                        for point in msg.points:
                            if point.line < 6:
                                x = point.x / 1000.0  # mm to m
                                y = point.y / 1000.0
                                z = point.z / 1000.0
                                points_list.append([x, y, z])
                        
                        if len(points_list) < 100:
                            continue
                        
                        points = np.array(points_list, dtype=np.float64)
                        stats = analyze_lidar_z_convention(points)
                        
                        if stats:
                            lidar_stats.append(stats)
                            scan_count += 1
                            if scan_count >= args.n_scans:
                                break
                                
                    except Exception as e:
                        continue
                
                if lidar_stats:
                    # Aggregate
                    avg_normal = np.mean([s['ground_normal'] for s in lidar_stats], axis=0)
                    avg_normal_dot_z = np.mean([s['ground_normal_dot_z'] for s in lidar_stats])
                    avg_z_mean = np.mean([s['z_mean'] for s in lidar_stats])
                    avg_n_below = np.mean([s['n_below'] for s in lidar_stats])
                    avg_n_above = np.mean([s['n_above'] for s in lidar_stats])
                    
                    print(f"  Analyzed {len(lidar_stats)} scans")
                    print(f"  Ground plane normal (avg): {avg_normal}")
                    print(f"  Ground normal · Z-axis: {avg_normal_dot_z:.4f}")
                    print(f"  Average Z-mean: {avg_z_mean:.4f} m")
                    print(f"  Points below origin (Z<0): {avg_n_below:.0f} avg")
                    print(f"  Points above origin (Z>0): {avg_n_above:.0f} avg")
                    print()
                    
                    # Interpretation
                    if avg_normal_dot_z > 0.7:  # Normal points mostly up
                        print("  → INTERPRETATION: LiDAR appears to be Z-UP")
                        print("     (Ground normal points up, consistent with Z-up convention)")
                        print("     → T_base_lidar rotation should be [0, 0, 0] (identity)")
                    elif avg_normal_dot_z < -0.7:  # Normal points mostly down
                        print("  → INTERPRETATION: LiDAR appears to be Z-DOWN")
                        print("     (Ground normal points down, requires 180° X-rotation)")
                        print("     → T_base_lidar rotation should be [π, 0, 0] = [3.141593, 0, 0]")
                    else:
                        print("  → INTERPRETATION: Unclear (ground normal not aligned with Z)")
                        print(f"     (Normal · Z = {avg_normal_dot_z:.4f}, angle = {np.arccos(abs(avg_normal_dot_z))*180/np.pi:.1f}°)")
                else:
                    print("  ERROR: No valid point clouds found")
                    
            except ImportError:
                print(f"  ERROR: Cannot import livox_ros_driver2.msg.CustomMsg")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print()
        
        # =====================================================================
        # 2. IMU Gravity Direction
        # =====================================================================
        print("2. IMU GRAVITY DIRECTION (in sensor frame)")
        print("-" * 80)
        
        imu_tid = topic_id(cur, args.imu_topic)
        if imu_tid is None:
            print(f"  ERROR: Topic not found: {args.imu_topic}")
        else:
            try:
                from sensor_msgs.msg import Imu
                
                accel_samples = []
                sample_count = 0
                
                for row in cur.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
                    (imu_tid, 200),  # Get ~1 second at 200Hz
                ):
                    ts, data = row
                    try:
                        msg = deserialize_message(data, Imu)
                        # CRITICAL: Livox MID-360 IMU outputs in g units, not m/s²
                        # Scale by 9.81 to convert to m/s² for analysis
                        accel_g = np.array([
                            msg.linear_acceleration.x,
                            msg.linear_acceleration.y,
                            msg.linear_acceleration.z,
                        ])
                        accel_ms2 = accel_g * 9.81  # Convert g → m/s²
                        accel_samples.append(accel_ms2)
                        sample_count += 1
                    except Exception:
                        continue
                
                if accel_samples:
                    accel_array = np.array(accel_samples)
                    stats = analyze_imu_gravity(accel_array)
                    
                    if stats:
                        print(f"  Analyzed {len(accel_samples)} IMU samples")
                        print(f"  Average accel vector (m/s²): {stats['accel_mean']}")
                        print(f"  Accel magnitude: {stats['accel_magnitude']:.4f} m/s² = {stats['accel_magnitude']/9.81:.4f} g")
                        print(f"  Accel direction (normalized): {stats['accel_dir']}")
                        print(f"  Expected specific force in base_footprint (Z-up): {stats['expected_specific_force_base']}")
                        print(f"  Angle to expected: {stats['angle_to_expected_deg']:.2f}°")
                        print()
                        
                        # Interpretation
                        if stats['angle_to_expected_deg'] < 10:
                            print("  → INTERPRETATION: IMU specific force matches Z-up base_footprint")
                            print("     (No rotation needed, or current T_base_imu is correct)")
                        elif stats['angle_to_expected_deg'] > 170:
                            print("  → INTERPRETATION: IMU specific force is opposite to Z-up")
                            print("     (Requires 180° rotation)")
                        else:
                            print(f"  → INTERPRETATION: IMU has {stats['angle_to_expected_deg']:.1f}° misalignment")
                            print("     (Requires rotation correction)")
                            
                            # Estimate rotation
                            accel_dir = stats['accel_dir']
                            expected = stats['expected_specific_force_base']
                            
                            # Find rotation that aligns accel_dir to expected
                            # Using cross product to find axis
                            axis = np.cross(accel_dir, expected)
                            axis_norm = np.linalg.norm(axis)
                            if axis_norm > 1e-6:
                                axis = axis / axis_norm
                                angle = np.arccos(np.clip(np.dot(accel_dir, expected), -1, 1))
                                rotvec = axis * angle
                                print(f"     Estimated rotation vector: [{rotvec[0]:.6f}, {rotvec[1]:.6f}, {rotvec[2]:.6f}]")
                    else:
                        print("  ERROR: Could not analyze IMU data (insufficient samples or motion)")
                else:
                    print("  ERROR: No valid IMU samples found")
                    
            except ImportError:
                print(f"  ERROR: Cannot import sensor_msgs.msg.Imu")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print()
        
        # =====================================================================
        # 3. Odom Covariance Ordering
        # =====================================================================
        print("3. ODOMETRY COVARIANCE ORDERING")
        print("-" * 80)
        
        odom_tid = topic_id(cur, args.odom_topic)
        if odom_tid is None:
            print(f"  ERROR: Topic not found: {args.odom_topic}")
        else:
            try:
                from nav_msgs.msg import Odometry
                
                odom_samples = []
                
                for row in cur.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
                    (odom_tid, 50),
                ):
                    ts, data = row
                    try:
                        msg = deserialize_message(data, Odometry)
                        cov = np.array(msg.pose.covariance).reshape(6, 6)
                        odom_samples.append(cov)
                    except Exception:
                        continue
                
                if odom_samples:
                    # Use first valid sample
                    cov = odom_samples[0]
                    stats = analyze_odom_covariance(cov)
                    
                    if stats:
                        print(f"  Analyzed odometry covariance (6×6)")
                        print(f"  Diagonal variances: {stats['diag']}")
                        print()
                        print(f"  ROS ordering [x,y,z,roll,pitch,yaw]:")
                        print(f"    xy_yaw mean: {stats['ros_xy_yaw_mean']:.6f}")
                        print(f"    z_rp mean:   {stats['ros_z_rp_mean']:.6f}")
                        print(f"    Makes sense for 2D robot: {stats['ros_makes_sense']}")
                        print()
                        print(f"  Legacy/permuted ordering [rx,ry,rz,tx,ty,tz]:")
                        print(f"    rot mean:    {stats['perm_rot_mean']:.6f}")
                        print(f"    trans mean:  {stats['perm_trans_mean']:.6f}")
                        print(f"    Makes sense: {stats['perm_makes_sense']}")
                        print()
                        print(f"  → INTERPRETATION: {stats['likely_ordering']}")
                        print(f"     (For 2D wheeled robot: xy_yaw should be small, z_rp should be large)")
                else:
                    print("  ERROR: No valid odometry messages found")
                    
            except ImportError:
                print(f"  ERROR: Cannot import nav_msgs.msg.Odometry")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("Review the interpretations above to determine:")
        print("  1. LiDAR frame convention (Z-up vs Z-down)")
        print("  2. IMU rotation correction needed")
        print("  3. Odom covariance ordering (ROS standard vs legacy-permuted)")
        print()
        
        rclpy.shutdown()
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
