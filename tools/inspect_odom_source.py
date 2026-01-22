#!/usr/bin/env python3
"""
Inspect M3DGR /odom messages to determine if wheel-only or IMU-fused.
"""
import rclpy
from rclpy.serialization import serialize_message, deserialize_message
from nav_msgs.msg import Odometry
import sqlite3
import numpy as np
import sys
import os

# Add workspace to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

bag_path = 'rosbags/m3dgr/Dynamic01_ros2'
db_path = f'{bag_path}/Dynamic01_ros2.db3'

if not os.path.exists(db_path):
    print(f"Error: Database not found at {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get first few odom messages
cursor.execute('''
    SELECT timestamp, data 
    FROM messages 
    WHERE topic_id = (SELECT id FROM topics WHERE name = '/odom')
    ORDER BY timestamp 
    LIMIT 5
''')

rows = cursor.fetchall()
if rows:
    rclpy.init()
    print("=== /odom Message Analysis ===\n")
    
    for i, (timestamp, data) in enumerate(rows):
        msg = deserialize_message(data, Odometry)
        print(f"Message {i+1}:")
        print(f"  Frame: {msg.header.frame_id} -> {msg.child_frame_id}")
        print(f"  Position: ({msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f}, {msg.pose.pose.position.z:.3f})")
        print(f"  Orientation: ({msg.pose.pose.orientation.x:.4f}, {msg.pose.pose.orientation.y:.4f}, {msg.pose.pose.orientation.z:.4f}, {msg.pose.pose.orientation.w:.4f})")
        print(f"  Twist linear: ({msg.twist.twist.linear.x:.4f}, {msg.twist.twist.linear.y:.4f}, {msg.twist.twist.linear.z:.4f})")
        print(f"  Twist angular: ({msg.twist.twist.angular.x:.4f}, {msg.twist.twist.angular.y:.4f}, {msg.twist.twist.angular.z:.4f})")
        
        # Check covariance structure
        if len(msg.pose.covariance) >= 36:
            cov = np.array(msg.pose.covariance).reshape(6, 6)
            pos_cov = np.diag(cov[:3, :3])
            rot_cov = np.diag(cov[3:, 3:])
            print(f"  Position cov (diag): X={pos_cov[0]:.6f}, Y={pos_cov[1]:.6f}, Z={pos_cov[2]:.6f}")
            print(f"  Rotation cov (diag): Roll={rot_cov[0]:.6f}, Pitch={rot_cov[1]:.6f}, Yaw={rot_cov[2]:.6f}")
            
            # Wheel-only typically has:
            # - High Z uncertainty (no vertical motion)
            # - High rotation uncertainty around X/Y (no pitch/roll from IMU)
            # - Lower XY uncertainty (good wheel encoders)
            indicators = []
            if pos_cov[2] > 0.1:  # High Z uncertainty
                indicators.append("High Z uncertainty (wheel-only indicator)")
            if rot_cov[0] > 0.01 or rot_cov[1] > 0.01:  # High pitch/roll uncertainty
                indicators.append("High pitch/roll uncertainty (wheel-only indicator)")
            if pos_cov[0] < 0.1 and pos_cov[1] < 0.1:  # Low XY uncertainty
                indicators.append("Low XY uncertainty (good wheel encoders)")
            
            if indicators:
                print(f"  → Indicators: {'; '.join(indicators)}")
        
        print()
    
    # Summary analysis
    print("=== Summary ===")
    if rows:
        first_msg = deserialize_message(rows[0][1], Odometry)
        if len(first_msg.pose.covariance) >= 36:
            cov = np.array(first_msg.pose.covariance).reshape(6, 6)
            pos_cov = np.diag(cov[:3, :3])
            rot_cov = np.diag(cov[3:, 3:])
            
            z_uncertainty_high = pos_cov[2] > 0.1
            pitch_roll_uncertainty_high = rot_cov[0] > 0.01 or rot_cov[1] > 0.01
            xy_uncertainty_low = pos_cov[0] < 0.1 and pos_cov[1] < 0.1
            
            if z_uncertainty_high and pitch_roll_uncertainty_high:
                print("✓ CONFIRMED: Wheel-only odometry")
                print("  - High Z uncertainty (no vertical motion tracking)")
                print("  - High pitch/roll uncertainty (no IMU fusion)")
            elif not z_uncertainty_high and not pitch_roll_uncertainty_high:
                print("✗ Likely IMU-fused odometry")
                print("  - Low Z uncertainty suggests IMU fusion")
                print("  - Low pitch/roll uncertainty suggests IMU fusion")
            else:
                print("? Mixed signals - need more analysis")
    
    rclpy.shutdown()
else:
    print("No /odom messages found")

conn.close()
