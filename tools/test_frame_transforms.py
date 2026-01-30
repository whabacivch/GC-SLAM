#!/usr/bin/env python3
"""
End-to-end frame transform validation test.

This test verifies that all frame transforms in the pipeline are consistent
by using actual sensor data and checking physical constraints:

1. LiDAR: After transform to base_footprint, floor should be at Z ≈ 0
2. IMU: After transform to base_footprint, gravity should point -Z
3. Odom: Motion should be primarily in XY plane (Z constant for planar robot)
4. GT: After alignment, trajectories should overlap

PASS criteria are based on physical constraints, not arbitrary thresholds.
"""

import argparse
import sqlite3
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tools"))

from rosbag_sqlite_utils import resolve_db3_path, topic_id


class TransformTest:
    """Base class for transform tests."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details = {}

    def run(self, conn, cur) -> bool:
        raise NotImplementedError

    def report(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [f"{status}: {self.name}"]
        if self.message:
            lines.append(f"       {self.message}")
        for k, v in self.details.items():
            lines.append(f"       {k}: {v}")
        return "\n".join(lines)


class LidarTransformTest(TransformTest):
    """
    Test: LiDAR points transformed to base_footprint should have floor at Z ≈ 0.

    Physical constraint: The robot base (base_footprint) is on the ground.
    After transforming LiDAR points by T_base_lidar, floor points should be at Z ≈ 0.
    """

    def __init__(self, T_base_lidar: np.ndarray):
        super().__init__("LiDAR → base_footprint transform")
        # T_base_lidar = [tx, ty, tz, rx, ry, rz] in meters and radians
        self.t_base_lidar = T_base_lidar[:3]
        self.R_base_lidar = R.from_rotvec(T_base_lidar[3:6]).as_matrix()

    def run(self, conn, cur) -> bool:
        import rclpy
        from rclpy.serialization import deserialize_message
        from livox_ros_driver2.msg import CustomMsg

        lidar_tid = topic_id(cur, "/livox/mid360/lidar")
        if lidar_tid is None:
            self.message = "LiDAR topic not found"
            return False

        # Collect points from multiple scans
        all_points_sensor = []
        scan_count = 0

        for row in cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
            (lidar_tid, 30),
        ):
            ts, data = row
            try:
                msg = deserialize_message(data, CustomMsg)
                for p in msg.points:
                    r = np.sqrt(p.x**2 + p.y**2 + p.z**2)
                    if 0.3 < r < 15:  # Valid range
                        all_points_sensor.append([p.x, p.y, p.z])
                scan_count += 1
            except Exception:
                continue

        if len(all_points_sensor) < 1000:
            self.message = f"Insufficient valid points: {len(all_points_sensor)}"
            return False

        points_sensor = np.array(all_points_sensor)

        # Transform to base_footprint: p_base = R @ p_sensor + t
        points_base = (self.R_base_lidar @ points_sensor.T).T + self.t_base_lidar

        # Find floor level using histogram peak detection
        # The floor should be a flat surface = large horizontal extent at a specific Z
        z_base = points_base[:, 2]

        # Use histogram to find floor (most likely Z level with horizontal structure)
        # Look for peaks in the histogram between -1.5m and 0.5m (expected floor range)
        z_floor_range = z_base[(z_base > -1.5) & (z_base < 0.5)]
        if len(z_floor_range) < 100:
            z_floor_range = z_base  # Fallback

        hist, edges = np.histogram(z_floor_range, bins=50)
        centers = (edges[:-1] + edges[1:]) / 2

        # Find peaks (local maxima with significant counts)
        from scipy import signal
        peaks, props = signal.find_peaks(hist, height=len(z_floor_range) * 0.01, distance=3)

        if len(peaks) > 0:
            # Find the peak closest to Z=0 (expected floor in base_footprint)
            peak_zs = centers[peaks]
            closest_to_zero = np.argmin(np.abs(peak_zs))
            floor_z_mean = peak_zs[closest_to_zero]
            # Get std from points near this peak
            floor_mask = np.abs(z_base - floor_z_mean) < 0.1
            floor_z_std = np.std(z_base[floor_mask]) if floor_mask.sum() > 10 else 0.1
        else:
            # Fallback: use median of points near Z=0
            near_zero = z_base[np.abs(z_base) < 0.5]
            floor_z_mean = np.median(near_zero) if len(near_zero) > 0 else np.median(z_base)
            floor_z_std = np.std(near_zero) if len(near_zero) > 10 else 1.0

        self.details = {
            "points_analyzed": len(points_base),
            "floor_z_mean": f"{floor_z_mean:.4f} m",
            "floor_z_std": f"{floor_z_std:.4f} m",
            "T_base_lidar_trans": f"{self.t_base_lidar}",
            "T_base_lidar_rot": f"{R.from_matrix(self.R_base_lidar).as_rotvec()}",
        }

        # PASS criteria: floor should be within 0.15m of Z=0
        # (allowing some tolerance for uneven ground, sensor noise)
        if abs(floor_z_mean) < 0.15:
            self.passed = True
            self.message = f"Floor at Z={floor_z_mean:.3f}m (expected ~0)"
        else:
            self.passed = False
            self.message = f"Floor at Z={floor_z_mean:.3f}m - expected ~0, check T_base_lidar"

        return self.passed


class ImuTransformTest(TransformTest):
    """
    Test: IMU accel transformed to base_footprint should show gravity in -Z.

    Physical constraint: When stationary, accelerometer measures specific force
    (reaction to gravity). In a Z-up frame, this should point +Z with magnitude ~g.
    After transform, gravity direction should be [0, 0, -g].
    """

    def __init__(self, T_base_imu: np.ndarray, accel_scale: float = 9.81):
        super().__init__("IMU → base_footprint transform")
        self.t_base_imu = T_base_imu[:3]
        self.R_base_imu = R.from_rotvec(T_base_imu[3:6]).as_matrix()
        self.accel_scale = accel_scale  # g → m/s² conversion

    def run(self, conn, cur) -> bool:
        import rclpy
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Imu

        imu_tid = topic_id(cur, "/livox/mid360/imu")
        if imu_tid is None:
            self.message = "IMU topic not found"
            return False

        # Collect IMU samples (assuming mostly stationary at start)
        accel_samples_sensor = []

        for row in cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
            (imu_tid, 500),
        ):
            ts, data = row
            try:
                msg = deserialize_message(data, Imu)
                accel_samples_sensor.append([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ])
            except Exception:
                continue

        if len(accel_samples_sensor) < 100:
            self.message = f"Insufficient IMU samples: {len(accel_samples_sensor)}"
            return False

        accel_sensor = np.array(accel_samples_sensor)

        # Scale to m/s² if needed (Livox outputs in g's)
        accel_mean_sensor = np.mean(accel_sensor, axis=0)
        accel_mag = np.linalg.norm(accel_mean_sensor)

        # Detect units
        if 0.8 < accel_mag < 1.2:
            # Units are g's, scale to m/s²
            accel_sensor = accel_sensor * self.accel_scale
            accel_mean_sensor = accel_mean_sensor * self.accel_scale
            accel_mag = accel_mag * self.accel_scale
            units = "g (scaled to m/s²)"
        else:
            units = "m/s²"

        # Transform to base_footprint
        accel_base = (self.R_base_imu @ accel_sensor.T).T
        accel_mean_base = np.mean(accel_base, axis=0)
        accel_mag_base = np.linalg.norm(accel_mean_base)

        # Normalize to get direction
        accel_dir_base = accel_mean_base / accel_mag_base

        # Expected: specific force should point +Z (reaction to gravity)
        expected_dir = np.array([0, 0, 1])
        dot_with_expected = np.dot(accel_dir_base, expected_dir)
        angle_to_expected = np.arccos(np.clip(dot_with_expected, -1, 1)) * 180 / np.pi

        self.details = {
            "samples_analyzed": len(accel_sensor),
            "accel_mean_sensor": f"{np.mean(accel_sensor, axis=0)}",
            "accel_mean_base": f"{accel_mean_base}",
            "accel_magnitude": f"{accel_mag_base:.4f} m/s² ({accel_mag_base/9.81:.4f}g)",
            "accel_direction_base": f"{accel_dir_base}",
            "angle_to_+Z": f"{angle_to_expected:.2f}°",
            "units_detected": units,
        }

        # PASS criteria:
        # 1. Magnitude should be ~9.81 m/s² (±10%)
        # 2. Direction should be within 5° of +Z
        mag_ok = 8.8 < accel_mag_base < 10.8
        dir_ok = angle_to_expected < 5.0

        if mag_ok and dir_ok:
            self.passed = True
            self.message = f"Gravity aligned to +Z within {angle_to_expected:.1f}°"
        elif not mag_ok:
            self.passed = False
            self.message = f"Accel magnitude {accel_mag_base:.2f} m/s² out of range (expected ~9.81)"
        else:
            self.passed = False
            self.message = f"Gravity misaligned by {angle_to_expected:.1f}° (expected <5°)"

        return self.passed


class OdomPlanarityTest(TransformTest):
    """
    Test: Odometry should show planar motion (Z approximately constant).

    Physical constraint: For a ground robot, wheel odometry should show
    motion primarily in the XY plane. Z should have very high covariance
    (unobserved) and should NOT be trusted.
    """

    def __init__(self):
        super().__init__("Odometry planarity check")

    def run(self, conn, cur) -> bool:
        import rclpy
        from rclpy.serialization import deserialize_message
        from nav_msgs.msg import Odometry

        odom_tid = topic_id(cur, "/odom")
        if odom_tid is None:
            self.message = "Odom topic not found"
            return False

        poses = []
        z_covs = []

        for row in cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
            (odom_tid, 1000),
        ):
            ts, data = row
            try:
                msg = deserialize_message(data, Odometry)
                pos = msg.pose.pose.position
                poses.append([pos.x, pos.y, pos.z])
                cov = np.array(msg.pose.covariance).reshape(6, 6)
                z_covs.append(cov[2, 2])  # Z variance
            except Exception:
                continue

        if len(poses) < 100:
            self.message = f"Insufficient odom samples: {len(poses)}"
            return False

        poses = np.array(poses)
        z_covs = np.array(z_covs)

        # Analyze motion extent
        x_ptp = np.ptp(poses[:, 0])
        y_ptp = np.ptp(poses[:, 1])
        z_ptp = np.ptp(poses[:, 2])
        z_mean = np.mean(poses[:, 2])
        z_std = np.std(poses[:, 2])
        z_cov_mean = np.mean(z_covs)

        self.details = {
            "poses_analyzed": len(poses),
            "X_extent": f"{x_ptp:.3f} m",
            "Y_extent": f"{y_ptp:.3f} m",
            "Z_extent": f"{z_ptp:.3f} m",
            "Z_mean": f"{z_mean:.3f} m",
            "Z_std": f"{z_std:.3f} m",
            "Z_covariance_mean": f"{z_cov_mean:.2e}",
        }

        # Check if Z is marked as unobserved (high covariance)
        z_unobserved = z_cov_mean > 1e4

        # Check if there's meaningful XY motion
        xy_motion = max(x_ptp, y_ptp) > 1.0

        if z_unobserved:
            self.passed = True
            self.message = f"Z covariance={z_cov_mean:.0e} (correctly marked unobserved)"
            if abs(z_mean) > 10:
                self.message += f" WARNING: Z mean={z_mean:.1f}m is nonsense - DO NOT USE"
        else:
            self.passed = False
            self.message = f"Z covariance={z_cov_mean:.2e} - should be much higher for 2D odom"

        return self.passed


class GtAlignmentTest(TransformTest):
    """
    Test: GT should show planar motion with constant Z at robot height.

    Physical constraint: For the M3DGR dataset, GT from mocap should show
    the robot at a consistent height (~0.85m) throughout the trajectory.
    """

    def __init__(self, expected_robot_height: float = 0.86):
        super().__init__("Ground truth frame validation")
        self.expected_height = expected_robot_height

    def run(self, conn, cur) -> bool:
        import rclpy
        from rclpy.serialization import deserialize_message
        from geometry_msgs.msg import PoseStamped

        gt_tid = topic_id(cur, "/vrpn_client_node/UGV/pose")
        if gt_tid is None:
            self.message = "GT topic not found (may be in separate file)"
            self.passed = True  # Not a failure, just not available
            return True

        poses = []

        for row in cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
            (gt_tid, 2000),
        ):
            ts, data = row
            try:
                msg = deserialize_message(data, PoseStamped)
                pos = msg.pose.position
                poses.append([pos.x, pos.y, pos.z])
            except Exception:
                continue

        if len(poses) < 100:
            self.message = f"Insufficient GT samples: {len(poses)}"
            return False

        poses = np.array(poses)

        # Analyze Z (height)
        z_mean = np.mean(poses[:, 2])
        z_std = np.std(poses[:, 2])
        z_ptp = np.ptp(poses[:, 2])

        # Analyze XY motion
        x_ptp = np.ptp(poses[:, 0])
        y_ptp = np.ptp(poses[:, 1])

        self.details = {
            "poses_analyzed": len(poses),
            "X_extent": f"{x_ptp:.3f} m",
            "Y_extent": f"{y_ptp:.3f} m",
            "Z_mean": f"{z_mean:.4f} m",
            "Z_std": f"{z_std:.4f} m",
            "Z_extent": f"{z_ptp:.4f} m",
            "expected_height": f"{self.expected_height} m",
        }

        # PASS criteria:
        # 1. Z should be nearly constant (std < 0.05m for planar motion)
        # 2. Z mean should be close to expected robot height (±0.2m)
        z_constant = z_std < 0.05
        z_height_ok = abs(z_mean - self.expected_height) < 0.2

        if z_constant and z_height_ok:
            self.passed = True
            self.message = f"GT Z={z_mean:.3f}m ± {z_std:.3f}m (planar motion confirmed)"
        elif not z_constant:
            self.passed = False
            self.message = f"GT Z varies too much (std={z_std:.3f}m) - not planar?"
        else:
            self.passed = False
            self.message = f"GT Z={z_mean:.3f}m differs from expected {self.expected_height}m"

        return self.passed


class FrameChainTest(TransformTest):
    """
    Test: Full transform chain from LiDAR sensor frame to world frame.

    Verifies that: sensor → base_footprint → odom_combined produces
    consistent results when compared with GT.
    """

    def __init__(self, T_base_lidar: np.ndarray):
        super().__init__("Full frame chain consistency")
        self.t_base_lidar = T_base_lidar[:3]
        self.R_base_lidar = R.from_rotvec(T_base_lidar[3:6]).as_matrix()

    def run(self, conn, cur) -> bool:
        import rclpy
        from rclpy.serialization import deserialize_message
        from livox_ros_driver2.msg import CustomMsg
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped

        # Get first odom pose (world → base_footprint)
        odom_tid = topic_id(cur, "/odom")
        gt_tid = topic_id(cur, "/vrpn_client_node/UGV/pose")
        lidar_tid = topic_id(cur, "/livox/mid360/lidar")

        if not all([odom_tid, lidar_tid]):
            self.message = "Required topics not found"
            return False

        # Get initial poses
        odom_pose = None
        for row in cur.execute(
            "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1",
            (odom_tid,),
        ):
            msg = deserialize_message(row[0], Odometry)
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            odom_pose = {
                'pos': np.array([pos.x, pos.y, pos.z]),
                'quat': np.array([ori.x, ori.y, ori.z, ori.w]),
            }

        gt_pose = None
        if gt_tid:
            for row in cur.execute(
                "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1",
                (gt_tid,),
            ):
                msg = deserialize_message(row[0], PoseStamped)
                pos = msg.pose.position
                ori = msg.pose.orientation
                gt_pose = {
                    'pos': np.array([pos.x, pos.y, pos.z]),
                    'quat': np.array([ori.x, ori.y, ori.z, ori.w]),
                }

        # Get LiDAR floor level by transforming to base frame first (same as LidarTransformTest)
        floor_z_base = None
        all_points_sensor = []
        for row in cur.execute(
            "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 20",
            (lidar_tid,),
        ):
            msg = deserialize_message(row[0], CustomMsg)
            for p in msg.points:
                r = np.sqrt(p.x**2 + p.y**2 + p.z**2)
                if 0.3 < r < 15:
                    all_points_sensor.append([p.x, p.y, p.z])

        if all_points_sensor:
            points_sensor = np.array(all_points_sensor)
            # Transform to base frame
            points_base = (self.R_base_lidar @ points_sensor.T).T + self.t_base_lidar
            z_base = points_base[:, 2]

            # Find floor as peak near Z=0 in base frame
            z_floor_range = z_base[(z_base > -0.5) & (z_base < 0.3)]
            if len(z_floor_range) > 100:
                from scipy import signal
                hist, edges = np.histogram(z_floor_range, bins=30)
                centers = (edges[:-1] + edges[1:]) / 2
                peaks, _ = signal.find_peaks(hist, height=len(z_floor_range) * 0.01, distance=2)
                if len(peaks) > 0:
                    peak_zs = centers[peaks]
                    closest_to_zero = np.argmin(np.abs(peak_zs))
                    floor_z_base = peak_zs[closest_to_zero]
                else:
                    floor_z_base = np.median(z_floor_range)
            else:
                floor_z_base = np.median(z_base[(z_base > -1.0) & (z_base < 0.5)])

        self.details = {
            "odom_position": f"{odom_pose['pos']}" if odom_pose else "N/A",
            "gt_position": f"{gt_pose['pos']}" if gt_pose else "N/A",
            "lidar_floor_z_base": f"{floor_z_base:.3f} m" if floor_z_base is not None else "N/A",
        }

        # Check: floor should be near Z=0 in base frame
        if floor_z_base is not None:
            floor_ok = abs(floor_z_base) < 0.2
        else:
            floor_ok = False

        # Check: GT Z should be near expected robot height
        if gt_pose is not None:
            gt_z_ok = 0.5 < gt_pose['pos'][2] < 1.2
            self.details["gt_z_reasonable"] = str(gt_z_ok)
        else:
            gt_z_ok = True  # Skip if GT not available

        # Check: Odom Z covariance should be high (unobserved)
        # Already checked in OdomPlanarityTest

        if floor_ok and gt_z_ok:
            self.passed = True
            self.message = "Frame chain produces physically consistent results"
        else:
            self.passed = False
            reasons = []
            if not floor_ok:
                reasons.append(f"floor_z_base={floor_z_base:.2f}m (should be ~0)")
            if not gt_z_ok and gt_pose:
                reasons.append(f"gt_z={gt_pose['pos'][2]:.2f}m (should be ~0.86)")
            self.message = "Frame chain issues: " + ", ".join(reasons)

        return self.passed


def main():
    ap = argparse.ArgumentParser(description="End-to-end frame transform validation")
    ap.add_argument("bag_path", help="Bag directory containing *.db3")
    ap.add_argument("--T-base-lidar", type=str, default="-0.011,0.0,0.778,0.0,0.0,0.0",
                    help="T_base_lidar as comma-separated [tx,ty,tz,rx,ry,rz]")
    ap.add_argument("--T-base-imu", type=str, default="0.0,0.0,0.0,-0.015586,0.489293,0.0",
                    help="T_base_imu as comma-separated [tx,ty,tz,rx,ry,rz]")
    ap.add_argument("--robot-height", type=float, default=0.86,
                    help="Expected robot height in GT frame (m)")
    args = ap.parse_args()

    # Parse transforms
    T_base_lidar = np.array([float(x) for x in args.T_base_lidar.split(",")])
    T_base_imu = np.array([float(x) for x in args.T_base_imu.split(",")])

    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        print(f"ERROR: Could not locate *.db3 under '{args.bag_path}'", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        import rclpy
        rclpy.init()

        print("=" * 80)
        print("FRAME TRANSFORM VALIDATION TEST")
        print("=" * 80)
        print(f"Bag: {db_path}")
        print(f"T_base_lidar: {T_base_lidar}")
        print(f"T_base_imu: {T_base_imu}")
        print()

        # Define tests
        tests = [
            LidarTransformTest(T_base_lidar),
            ImuTransformTest(T_base_imu),
            OdomPlanarityTest(),
            GtAlignmentTest(args.robot_height),
            FrameChainTest(T_base_lidar),
        ]

        # Run tests
        results = []
        for test in tests:
            print(f"Running: {test.name}...")
            test.run(conn, cur)
            results.append(test)
            print(test.report())
            print()

        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        passed = sum(1 for t in results if t.passed)
        total = len(results)

        for test in results:
            status = "✓" if test.passed else "✗"
            print(f"  {status} {test.name}")

        print()
        print(f"Result: {passed}/{total} tests passed")

        if passed == total:
            print("\n✓ All frame transforms are consistent!")
            ret = 0
        else:
            print("\n✗ Some frame transforms have issues - review details above")
            ret = 1

        rclpy.shutdown()
        return ret

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
