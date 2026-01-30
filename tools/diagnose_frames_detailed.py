#!/usr/bin/env python3
"""
Detailed frame convention diagnostic - shows RAW DATA, no interpretation guessing.

This script dumps the actual values so we can see what the data looks like.
"""

import argparse
import sqlite3
import sys
import numpy as np

# Add project root to path
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rosbag_sqlite_utils import resolve_db3_path, topic_id


def main() -> int:
    ap = argparse.ArgumentParser(description="Detailed frame diagnostic - raw data dump")
    ap.add_argument("bag_path", help="Bag directory containing *.db3")
    ap.add_argument("--lidar-topic", default="/livox/mid360/lidar")
    ap.add_argument("--imu-topic", default="/livox/mid360/imu")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--gt-topic", default="/vrpn_client_node/UGV/pose")
    ap.add_argument("--n-scans", type=int, default=10)
    ap.add_argument("--n-imu", type=int, default=500)
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

        print("=" * 100)
        print("DETAILED FRAME DIAGNOSTIC - RAW DATA")
        print("=" * 100)
        print(f"Bag: {db_path}")
        print()

        # =====================================================================
        # 1. RAW LIDAR POINT STATISTICS
        # =====================================================================
        print("=" * 100)
        print("1. RAW LIDAR POINT CLOUD STATISTICS")
        print("=" * 100)

        lidar_tid = topic_id(cur, args.lidar_topic)
        if lidar_tid is None:
            print(f"  ERROR: Topic not found: {args.lidar_topic}")
        else:
            try:
                from livox_ros_driver2.msg import CustomMsg

                all_points = []
                scan_count = 0

                for row in cur.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
                    (lidar_tid, args.n_scans * 5),
                ):
                    ts, data = row
                    try:
                        msg = deserialize_message(data, CustomMsg)

                        # Get ALL points, not just line < 6
                        for point in msg.points:
                            # Dynamic01_ros2 livox_ros_driver2/CustomMsg points are already in meters.
                            # Do NOT apply a mm→m conversion here (it collapses the cloud to ~cm scale).
                            x = float(point.x)
                            y = float(point.y)
                            z = float(point.z)
                            # Filter obvious invalid points
                            if abs(x) < 100 and abs(y) < 100 and abs(z) < 100:
                                all_points.append([x, y, z])

                        scan_count += 1
                        if scan_count >= args.n_scans:
                            break
                    except Exception as e:
                        continue

                if all_points:
                    points = np.array(all_points, dtype=np.float64)
                    print(f"  Total points from {scan_count} scans: {len(points)}")
                    print()

                    # Per-axis statistics
                    for i, axis in enumerate(['X', 'Y', 'Z']):
                        vals = points[:, i]
                        print(f"  {axis}-axis:")
                        print(f"    Min:    {np.min(vals):10.4f} m")
                        print(f"    Max:    {np.max(vals):10.4f} m")
                        print(f"    Mean:   {np.mean(vals):10.4f} m")
                        print(f"    Median: {np.median(vals):10.4f} m")
                        print(f"    Std:    {np.std(vals):10.4f} m")
                        print(f"    P5:     {np.percentile(vals, 5):10.4f} m")
                        print(f"    P25:    {np.percentile(vals, 25):10.4f} m")
                        print(f"    P75:    {np.percentile(vals, 75):10.4f} m")
                        print(f"    P95:    {np.percentile(vals, 95):10.4f} m")
                        print()

                    # Critical question: Where is the "ground"?
                    z_vals = points[:, 2]
                    print("  Z-axis histogram (10 bins):")
                    hist, edges = np.histogram(z_vals, bins=10)
                    for i in range(len(hist)):
                        bar = '#' * int(hist[i] / max(hist) * 50)
                        print(f"    [{edges[i]:7.2f}, {edges[i+1]:7.2f}): {hist[i]:6d} {bar}")
                    print()

                    # Points above/below various Z thresholds
                    print("  Points by Z threshold:")
                    for thresh in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
                        below = np.sum(z_vals < thresh)
                        pct = 100.0 * below / len(z_vals)
                        print(f"    Z < {thresh:5.1f}m: {below:6d} ({pct:5.1f}%)")
                    print()

                    # Find the "floor" - cluster of points at the lowest Z
                    # If Z-up and sensor at 0.78m, floor should be around Z = -0.78m
                    z_sorted = np.sort(z_vals)
                    floor_sample = z_sorted[:int(len(z_sorted)*0.1)]  # lowest 10%
                    print(f"  Lowest 10% of points (potential floor):")
                    print(f"    Range: [{np.min(floor_sample):.4f}, {np.max(floor_sample):.4f}] m")
                    print(f"    Mean:  {np.mean(floor_sample):.4f} m")
                    print()

                    # If sensor is at height h with Z-up, floor is at Z = -h
                    # If sensor is at height h with Z-down, floor is at Z = +h
                    print("  INTERPRETATION:")
                    floor_z = np.mean(floor_sample)
                    if floor_z < -0.3:
                        print(f"    Floor appears to be at Z ≈ {floor_z:.2f}m")
                        print(f"    → Suggests Z-UP convention (floor below sensor)")
                        print(f"    → Sensor height estimate: {-floor_z:.2f}m")
                    elif floor_z > 0.3:
                        print(f"    Floor appears to be at Z ≈ {floor_z:.2f}m")
                        print(f"    → Suggests Z-DOWN convention (floor 'below' = +Z)")
                        print(f"    → Sensor height estimate: {floor_z:.2f}m")
                    else:
                        print(f"    Floor Z is near zero ({floor_z:.2f}m) - unclear convention")
                        print(f"    → May need manual inspection of point cloud")

            except ImportError as e:
                print(f"  ERROR: Cannot import livox_ros_driver2.msg.CustomMsg: {e}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        print()

        # =====================================================================
        # 2. RAW IMU DATA
        # =====================================================================
        print("=" * 100)
        print("2. RAW IMU ACCELEROMETER DATA")
        print("=" * 100)

        imu_tid = topic_id(cur, args.imu_topic)
        if imu_tid is None:
            print(f"  ERROR: Topic not found: {args.imu_topic}")
        else:
            try:
                from sensor_msgs.msg import Imu

                accel_raw = []
                gyro_raw = []

                for row in cur.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
                    (imu_tid, args.n_imu),
                ):
                    ts, data = row
                    try:
                        msg = deserialize_message(data, Imu)
                        accel_raw.append([
                            msg.linear_acceleration.x,
                            msg.linear_acceleration.y,
                            msg.linear_acceleration.z,
                        ])
                        gyro_raw.append([
                            msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z,
                        ])
                    except Exception:
                        continue

                if accel_raw:
                    accel = np.array(accel_raw, dtype=np.float64)
                    gyro = np.array(gyro_raw, dtype=np.float64)

                    print(f"  Analyzed {len(accel)} IMU samples")
                    print()

                    # Raw values (before any scaling)
                    accel_mean = np.mean(accel, axis=0)
                    accel_mag = np.linalg.norm(accel_mean)

                    print("  RAW accelerometer values (as received from sensor):")
                    print(f"    Mean X: {accel_mean[0]:10.6f}")
                    print(f"    Mean Y: {accel_mean[1]:10.6f}")
                    print(f"    Mean Z: {accel_mean[2]:10.6f}")
                    print(f"    Magnitude: {accel_mag:.6f}")
                    print()

                    # Determine units
                    if 0.8 < accel_mag < 1.2:
                        print("  → Magnitude ≈ 1.0 → Units appear to be g's (need to multiply by 9.81)")
                        scale = 9.81
                    elif 8.0 < accel_mag < 12.0:
                        print("  → Magnitude ≈ 9.8 → Units appear to be m/s²")
                        scale = 1.0
                    else:
                        print(f"  → Magnitude = {accel_mag:.2f} → UNUSUAL (not 1g or 9.81 m/s²)")
                        scale = 1.0

                    accel_scaled = accel * scale
                    accel_mean_scaled = accel_mean * scale
                    accel_mag_scaled = accel_mag * scale

                    print()
                    print("  Scaled accelerometer (m/s²):")
                    print(f"    Mean X: {accel_mean_scaled[0]:10.4f} m/s²")
                    print(f"    Mean Y: {accel_mean_scaled[1]:10.4f} m/s²")
                    print(f"    Mean Z: {accel_mean_scaled[2]:10.4f} m/s²")
                    print(f"    Magnitude: {accel_mag_scaled:.4f} m/s² ({accel_mag_scaled/9.81:.4f}g)")
                    print()

                    # The gravity direction analysis
                    accel_dir = accel_mean_scaled / accel_mag_scaled
                    print("  Specific force direction (normalized):")
                    print(f"    [{accel_dir[0]:.4f}, {accel_dir[1]:.4f}, {accel_dir[2]:.4f}]")
                    print()

                    print("  INTERPRETATION:")
                    print("    Specific force = -gravity (what accelerometer measures)")
                    print("    If Z-up world, expect specific force ≈ [0, 0, +g]")
                    print("    If Z-down world, expect specific force ≈ [0, 0, -g]")
                    print()

                    dominant_axis = np.argmax(np.abs(accel_dir))
                    dominant_sign = np.sign(accel_dir[dominant_axis])
                    axis_names = ['X', 'Y', 'Z']

                    print(f"    Dominant axis: {axis_names[dominant_axis]} (value: {accel_dir[dominant_axis]:.4f})")

                    if dominant_axis == 2:  # Z dominant
                        if accel_dir[2] > 0.7:
                            print("    → Specific force points +Z → IMU frame is likely Z-up")
                        elif accel_dir[2] < -0.7:
                            print("    → Specific force points -Z → IMU frame is likely Z-down")
                        else:
                            print(f"    → Z component = {accel_dir[2]:.4f} (tilted)")
                    else:
                        print(f"    → Gravity is NOT aligned with Z! Dominant is {axis_names[dominant_axis]}")
                        print(f"    → This suggests IMU frame has different convention or is severely tilted")

                    # Gyro sanity check
                    gyro_mean = np.mean(gyro, axis=0)
                    print()
                    print("  Gyroscope mean (rad/s) - should be near zero if stationary:")
                    print(f"    [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}]")

            except ImportError as e:
                print(f"  ERROR: Cannot import sensor_msgs.msg.Imu: {e}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        print()

        # =====================================================================
        # 3. GROUND TRUTH FRAME
        # =====================================================================
        print("=" * 100)
        print("3. GROUND TRUTH (VRPN/MOCAP) DATA")
        print("=" * 100)

        gt_tid = topic_id(cur, args.gt_topic)
        if gt_tid is None:
            print(f"  Topic not found: {args.gt_topic}")
            print("  (This is expected if GT is in a separate file)")
        else:
            try:
                from geometry_msgs.msg import PoseStamped

                gt_poses = []

                for row in cur.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
                    (gt_tid, 1000),
                ):
                    ts, data = row
                    try:
                        msg = deserialize_message(data, PoseStamped)
                        pos = msg.pose.position
                        ori = msg.pose.orientation
                        gt_poses.append([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
                    except Exception:
                        continue

                if gt_poses:
                    poses = np.array(gt_poses, dtype=np.float64)
                    print(f"  Analyzed {len(poses)} GT poses")
                    print()

                    # Position statistics
                    pos = poses[:, :3]
                    print("  Position statistics:")
                    for i, axis in enumerate(['X', 'Y', 'Z']):
                        vals = pos[:, i]
                        print(f"    {axis}: range [{np.min(vals):.4f}, {np.max(vals):.4f}], "
                              f"mean {np.mean(vals):.4f}, std {np.std(vals):.4f}")
                    print()

                    # For a planar robot, one axis should have small variance (height)
                    stds = np.std(pos, axis=0)
                    min_std_axis = np.argmin(stds)
                    axis_names = ['X', 'Y', 'Z']
                    print(f"  Axis with smallest variance: {axis_names[min_std_axis]} (std={stds[min_std_axis]:.4f}m)")
                    print(f"  → For planar robot, this is likely the 'up' axis in GT frame")
                    print()

                    # Check if Z is constant (typical for planar robot in Z-up world)
                    z_vals = pos[:, 2]
                    print(f"  GT Z-axis (potential height):")
                    print(f"    Range: [{np.min(z_vals):.4f}, {np.max(z_vals):.4f}]")
                    print(f"    Mean:  {np.mean(z_vals):.4f}")
                    print(f"    Std:   {np.std(z_vals):.4f}")

                    if np.std(z_vals) < 0.1:
                        print(f"  → GT Z is nearly constant (~{np.mean(z_vals):.2f}m) - consistent with planar motion in Z-up world")
                    else:
                        print(f"  → GT Z varies significantly - either non-planar motion or different axis convention")

            except ImportError as e:
                print(f"  ERROR: Cannot import geometry_msgs.msg.PoseStamped: {e}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        print()

        # =====================================================================
        # 4. ODOMETRY MOTION DIRECTION
        # =====================================================================
        print("=" * 100)
        print("4. ODOMETRY MOTION DIRECTION")
        print("=" * 100)

        odom_tid = topic_id(cur, args.odom_topic)
        if odom_tid is None:
            print(f"  ERROR: Topic not found: {args.odom_topic}")
        else:
            try:
                from nav_msgs.msg import Odometry

                odom_poses = []

                for row in cur.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
                    (odom_tid, 500),
                ):
                    ts, data = row
                    try:
                        msg = deserialize_message(data, Odometry)
                        pos = msg.pose.pose.position
                        ori = msg.pose.pose.orientation
                        odom_poses.append([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
                    except Exception:
                        continue

                if odom_poses:
                    poses = np.array(odom_poses, dtype=np.float64)
                    print(f"  Analyzed {len(poses)} odom poses")
                    print()

                    pos = poses[:, :3]
                    print("  Position statistics:")
                    for i, axis in enumerate(['X', 'Y', 'Z']):
                        vals = pos[:, i]
                        print(f"    {axis}: range [{np.min(vals):.4f}, {np.max(vals):.4f}], "
                              f"ptp {np.ptp(vals):.4f}m")
                    print()

                    # Check which axes have motion
                    ptp = np.ptp(pos, axis=0)  # peak-to-peak
                    print(f"  Motion extent (ptp):")
                    print(f"    X: {ptp[0]:.4f}m, Y: {ptp[1]:.4f}m, Z: {ptp[2]:.4f}m")
                    print()

                    if ptp[2] < 0.1 and (ptp[0] > 0.5 or ptp[1] > 0.5):
                        print("  → Odom shows planar motion in XY plane (Z constant)")
                        print("  → Consistent with Z-up convention for odom frame")
                    elif ptp[1] < 0.1 and (ptp[0] > 0.5 or ptp[2] > 0.5):
                        print("  → Odom shows motion in XZ plane (Y constant)")
                        print("  → Suggests Y-up or Y-forward convention")
                    else:
                        print("  → Motion pattern unclear")

            except ImportError as e:
                print(f"  ERROR: Cannot import nav_msgs.msg.Odometry: {e}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        print()
        print("=" * 100)
        print("END DIAGNOSTIC")
        print("=" * 100)

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
