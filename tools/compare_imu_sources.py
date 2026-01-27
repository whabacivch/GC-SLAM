#!/usr/bin/env python3
"""
Compare IMU sources (camera vs Livox) to determine which is better for SLAM.

Analyzes:
1. Gravity direction (for frame alignment)
2. Noise characteristics
3. Data quality (dropouts, outliers)
4. Frame conventions
"""

import argparse
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add project root to path
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.rosbag_sqlite_utils import resolve_db3_path, topic_id
from sensor_msgs.msg import Imu
import rclpy
from rclpy.serialization import deserialize_message
import sqlite3


def analyze_imu_gravity(imu_data: list, frame_name: str) -> dict:
    """Analyze gravity direction from IMU data."""
    accels = np.array([(d['accel'][0], d['accel'][1], d['accel'][2]) for d in imu_data])
    
    # Filter to stationary periods (low variance in accel magnitude)
    accel_mags = np.linalg.norm(accels, axis=1)
    accel_mag_mean = np.mean(accel_mags)
    accel_mag_std = np.std(accel_mags)
    
    # Use points with stable magnitude (within 1 std)
    stable_mask = np.abs(accel_mags - accel_mag_mean) < accel_mag_std
    stable_accels = accels[stable_mask]
    
    if len(stable_accels) < 100:
        return None
    
    # Average gravity direction (normalized)
    gravity_dir = np.mean(stable_accels, axis=0)
    gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)
    
    # Expected gravity in base frame (Z-up): [0, 0, -1]
    expected_gravity = np.array([0.0, 0.0, -1.0])
    
    # Compute rotation to align IMU gravity to expected
    # R_base_imu such that R_base_imu @ gravity_imu = expected_gravity
    # This is a Wahba problem: find R that minimizes ||R @ g_imu - g_base||
    # Solution: R = V @ U^T where [U, S, V] = SVD(g_base @ g_imu^T)
    g_imu = gravity_dir.reshape(3, 1)
    g_base = expected_gravity.reshape(3, 1)
    H = g_base @ g_imu.T
    U, S, Vt = np.linalg.svd(H)
    R_align = U @ Vt
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R_align) < 0:
        U[:, -1] *= -1
        R_align = U @ Vt
    
    # Convert to rotation vector
    rot = R.from_matrix(R_align)
    rotvec = rot.as_rotvec()
    
    # Angle of misalignment
    angle_deg = np.arccos(np.clip(np.dot(gravity_dir, expected_gravity), -1, 1)) * 180 / np.pi
    
    return {
        'frame': frame_name,
        'gravity_dir': gravity_dir,
        'gravity_magnitude': accel_mag_mean,
        'gravity_magnitude_std': accel_mag_std,
        'rotation_to_base': rotvec,
        'misalignment_angle_deg': angle_deg,
        'stable_samples': len(stable_accels),
        'total_samples': len(accels),
    }


def analyze_imu_noise(imu_data: list) -> dict:
    """Analyze noise characteristics."""
    gyros = np.array([(d['gyro'][0], d['gyro'][1], d['gyro'][2]) for d in imu_data])
    accels = np.array([(d['accel'][0], d['accel'][1], d['accel'][2]) for d in imu_data])
    
    # Compute noise as std of differences (high-pass filter)
    gyro_diff = np.diff(gyros, axis=0)
    accel_diff = np.diff(accels, axis=0)
    
    gyro_noise = np.std(gyro_diff, axis=0)
    accel_noise = np.std(accel_diff, axis=0)
    
    return {
        'gyro_noise_std': gyro_noise,
        'accel_noise_std': accel_noise,
        'gyro_noise_rms': np.sqrt(np.mean(gyro_noise**2)),
        'accel_noise_rms': np.sqrt(np.mean(accel_noise**2)),
    }


def load_imu_data(cur, topic: str, max_samples: int = 10000) -> list:
    """Load IMU data from rosbag."""
    tid = topic_id(cur, topic)
    if tid is None:
        return []
    
    data = []
    for row in cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT ?",
        (tid, max_samples),
    ):
        ts_ns, msg_data = row
        msg = deserialize_message(msg_data, Imu)
        data.append({
            'stamp': ts_ns / 1e9,
            'gyro': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'accel': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
        })
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Compare IMU sources')
    parser.add_argument('bag_path', help='Path to rosbag directory')
    parser.add_argument('--camera-topic', default='/camera/imu', help='Camera IMU topic')
    parser.add_argument('--livox-topic', default='/livox/mid360/imu', help='Livox IMU topic')
    parser.add_argument('--max-samples', type=int, default=10000, help='Max samples per IMU')
    args = parser.parse_args()
    
    db_path = resolve_db3_path(args.bag_path)
    if not db_path:
        print(f"ERROR: Could not find .db3 file in {args.bag_path}")
        return 1
    
    rclpy.init()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    print("=" * 80)
    print("IMU Source Comparison")
    print("=" * 80)
    print()
    
    # Load data
    print(f"Loading camera IMU: {args.camera_topic}")
    camera_data = load_imu_data(cur, args.camera_topic, args.max_samples)
    print(f"  Loaded {len(camera_data)} samples")
    
    print(f"Loading Livox IMU: {args.livox_topic}")
    livox_data = load_imu_data(cur, args.livox_topic, args.max_samples)
    print(f"  Loaded {len(livox_data)} samples")
    print()
    
    if not camera_data or not livox_data:
        print("ERROR: Could not load IMU data from both sources")
        return 1
    
    # Analyze gravity
    print("=" * 80)
    print("GRAVITY ANALYSIS (Frame Alignment)")
    print("=" * 80)
    
    camera_gravity = analyze_imu_gravity(camera_data, 'camera_imu_optical_frame')
    livox_gravity = analyze_imu_gravity(livox_data, 'livox_frame')
    
    if camera_gravity:
        print(f"\nCamera IMU ({camera_gravity['frame']}):")
        print(f"  Gravity direction: [{camera_gravity['gravity_dir'][0]:.4f}, {camera_gravity['gravity_dir'][1]:.4f}, {camera_gravity['gravity_dir'][2]:.4f}]")
        print(f"  Gravity magnitude: {camera_gravity['gravity_magnitude']:.4f} ± {camera_gravity['gravity_magnitude_std']:.4f}")
        print(f"  Misalignment angle: {camera_gravity['misalignment_angle_deg']:.2f}°")
        print(f"  Rotation to base: [{camera_gravity['rotation_to_base'][0]:.6f}, {camera_gravity['rotation_to_base'][1]:.6f}, {camera_gravity['rotation_to_base'][2]:.6f}]")
        print(f"  Stable samples: {camera_gravity['stable_samples']}/{camera_gravity['total_samples']}")
    
    if livox_gravity:
        print(f"\nLivox IMU ({livox_gravity['frame']}):")
        print(f"  Gravity direction: [{livox_gravity['gravity_dir'][0]:.4f}, {livox_gravity['gravity_dir'][1]:.4f}, {livox_gravity['gravity_dir'][2]:.4f}]")
        print(f"  Gravity magnitude: {livox_gravity['gravity_magnitude']:.4f} ± {livox_gravity['gravity_magnitude_std']:.4f}")
        print(f"  Misalignment angle: {livox_gravity['misalignment_angle_deg']:.2f}°")
        print(f"  Rotation to base: [{livox_gravity['rotation_to_base'][0]:.6f}, {livox_gravity['rotation_to_base'][1]:.6f}, {livox_gravity['rotation_to_base'][2]:.6f}]")
        print(f"  Stable samples: {livox_gravity['stable_samples']}/{livox_gravity['total_samples']}")
    
    # Analyze noise
    print("\n" + "=" * 80)
    print("NOISE ANALYSIS")
    print("=" * 80)
    
    camera_noise = analyze_imu_noise(camera_data)
    livox_noise = analyze_imu_noise(livox_data)
    
    print(f"\nCamera IMU:")
    print(f"  Gyro noise (std): [{camera_noise['gyro_noise_std'][0]:.6f}, {camera_noise['gyro_noise_std'][1]:.6f}, {camera_noise['gyro_noise_std'][2]:.6f}] rad/s")
    print(f"  Gyro noise (RMS): {camera_noise['gyro_noise_rms']:.6f} rad/s")
    print(f"  Accel noise (std): [{camera_noise['accel_noise_std'][0]:.6f}, {camera_noise['accel_noise_std'][1]:.6f}, {camera_noise['accel_noise_std'][2]:.6f}] m/s²")
    print(f"  Accel noise (RMS): {camera_noise['accel_noise_rms']:.6f} m/s²")
    
    print(f"\nLivox IMU:")
    print(f"  Gyro noise (std): [{livox_noise['gyro_noise_std'][0]:.6f}, {livox_noise['gyro_noise_std'][1]:.6f}, {livox_noise['gyro_noise_std'][2]:.6f}] rad/s")
    print(f"  Gyro noise (RMS): {livox_noise['gyro_noise_rms']:.6f} rad/s")
    print(f"  Accel noise (std): [{livox_noise['accel_noise_std'][0]:.6f}, {livox_noise['accel_noise_std'][1]:.6f}, {livox_noise['accel_noise_std'][2]:.6f}] m/s²")
    print(f"  Accel noise (RMS): {livox_noise['accel_noise_rms']:.6f} m/s²")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if camera_gravity and livox_gravity:
        camera_angle = camera_gravity['misalignment_angle_deg']
        livox_angle = livox_gravity['misalignment_angle_deg']
        
        print(f"\nMisalignment angles:")
        print(f"  Camera IMU: {camera_angle:.2f}°")
        print(f"  Livox IMU:  {livox_angle:.2f}°")
        
        if camera_angle < livox_angle:
            print(f"\n✓ RECOMMENDATION: Use Camera IMU (better alignment: {camera_angle:.2f}° vs {livox_angle:.2f}°)")
        else:
            print(f"\n✓ RECOMMENDATION: Use Livox IMU (better alignment: {livox_angle:.2f}° vs {camera_angle:.2f}°)")
        
        # Check noise
        if camera_noise['gyro_noise_rms'] < livox_noise['gyro_noise_rms']:
            print(f"  Camera IMU also has lower gyro noise ({camera_noise['gyro_noise_rms']:.6f} vs {livox_noise['gyro_noise_rms']:.6f} rad/s)")
        else:
            print(f"  Livox IMU has lower gyro noise ({livox_noise['gyro_noise_rms']:.6f} vs {camera_noise['gyro_noise_rms']:.6f} rad/s)")
    
    conn.close()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
