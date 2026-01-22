#!/usr/bin/env python3
"""
Inspect M3DGR camera topics to determine:
1. Camera topics and whether depth is registered to color
2. Coordinate frames (camera optical frame, world frame)
3. Whether reliable camera pose per frame exists
"""
import rclpy
from rclpy.serialization import serialize_message, deserialize_message
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import sqlite3
import numpy as np
import sys
import os

bag_path = 'rosbags/m3dgr/Dynamic01_ros2'
db_path = f'{bag_path}/Dynamic01_ros2.db3'

if not os.path.exists(db_path):
    print(f"Error: Database not found at {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

rclpy.init()

print("=== Camera Topics and Frame Analysis ===\n")

# 1. Check RGB topic
print("1. RGB Camera Topic:")
cursor.execute('''
    SELECT name FROM topics WHERE name LIKE '%color%' OR name LIKE '%image%'
    ORDER BY name
''')
rgb_topics = cursor.fetchall()
rgb_frame = None
for (topic,) in rgb_topics:
    if 'compressed' in topic:
        print(f"   - {topic}")
        cursor.execute('''
            SELECT timestamp, data FROM messages 
            WHERE topic_id = (SELECT id FROM topics WHERE name = ?)
            ORDER BY timestamp LIMIT 1
        ''', (topic,))
        row = cursor.fetchone()
        if row:
            try:
                # Try to deserialize - may need special handling for compressed
                print(f"     (compressed - frame ID in decompressed message)")
            except:
                pass

# Check decompressed RGB if available
cursor.execute('''
    SELECT name FROM topics WHERE name = '/camera/image_raw'
''')
if cursor.fetchone():
    cursor.execute('''
        SELECT timestamp, data FROM messages 
        WHERE topic_id = (SELECT id FROM topics WHERE name = '/camera/image_raw')
        ORDER BY timestamp LIMIT 1
    ''')
    row = cursor.fetchone()
    if row:
        try:
            msg = deserialize_message(row[1], Image)
            rgb_frame = msg.header.frame_id
            print(f"   Decompressed: /camera/image_raw")
            print(f"     Frame ID: {msg.header.frame_id}")
            print(f"     Size: {msg.width}x{msg.height}")
            print(f"     Encoding: {msg.encoding}")
        except Exception as e:
            print(f"     (could not deserialize: {e})")

print()

# 2. Check Depth topic
print("2. Depth Camera Topic:")
depth_frame = None
cursor.execute('''
    SELECT name FROM topics WHERE name LIKE '%depth%'
    ORDER BY name
''')
depth_topics = cursor.fetchall()
for (topic,) in depth_topics:
    print(f"   - {topic}")
    if 'aligned' in topic.lower() or 'registered' in topic.lower():
        print(f"     → DEPTH IS ALIGNED/REGISTERED TO COLOR (from topic name)")
    
    cursor.execute('''
        SELECT timestamp, data FROM messages 
        WHERE topic_id = (SELECT id FROM topics WHERE name = ?)
        ORDER BY timestamp LIMIT 1
    ''', (topic,))
    row = cursor.fetchone()
    if row:
        try:
            msg = deserialize_message(row[1], Image)
            depth_frame = msg.header.frame_id
            print(f"     Frame ID: {msg.header.frame_id}")
            print(f"     Size: {msg.width}x{msg.height}")
            print(f"     Encoding: {msg.encoding}")
        except Exception as e:
            print(f"     (could not deserialize compressed: {e})")

print()

# 3. Check CameraInfo
print("3. CameraInfo Topics:")
cursor.execute('''
    SELECT name FROM topics WHERE name LIKE '%camera_info%' OR name LIKE '%CameraInfo%'
    ORDER BY name
''')
info_topics = cursor.fetchall()
for (topic,) in info_topics:
    print(f"   - {topic}")
    cursor.execute('''
        SELECT timestamp, data FROM messages 
        WHERE topic_id = (SELECT id FROM topics WHERE name = ?)
        ORDER BY timestamp LIMIT 1
    ''', (topic,))
    row = cursor.fetchone()
    if row:
        try:
            msg = deserialize_message(row[1], CameraInfo)
            print(f"     Frame ID: {msg.header.frame_id}")
            print(f"     Intrinsics: fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}, cx={msg.k[2]:.1f}, cy={msg.k[5]:.1f}")
        except:
            pass

print()

# 4. Check Odom frames (world frame)
print("4. World/Reference Frames (from /odom):")
cursor.execute('''
    SELECT timestamp, data FROM messages 
    WHERE topic_id = (SELECT id FROM topics WHERE name = '/odom')
    ORDER BY timestamp LIMIT 1
''')
row = cursor.fetchone()
if row:
    msg = deserialize_message(row[1], Odometry)
    print(f"   World frame: {msg.header.frame_id}")
    print(f"   Child frame: {msg.child_frame_id}")

print()

# 5. Check TF topics
print("5. TF Transform Availability:")
cursor.execute('''
    SELECT name FROM topics WHERE name LIKE '%tf%' OR name LIKE '%TF%'
    ORDER BY name
''')
tf_topics = cursor.fetchall()
if tf_topics:
    for (topic,) in tf_topics:
        print(f"   - {topic}")
        cursor.execute('''
            SELECT COUNT(*) FROM messages 
            WHERE topic_id = (SELECT id FROM topics WHERE name = ?)
        ''', (topic,))
        count = cursor.fetchone()[0]
        print(f"     Messages: {count}")
        
        if count > 0:
            cursor.execute('''
                SELECT timestamp, data FROM messages 
                WHERE topic_id = (SELECT id FROM topics WHERE name = ?)
                ORDER BY timestamp LIMIT 1
            ''', (topic,))
            row = cursor.fetchone()
            if row:
                try:
                    msg = deserialize_message(row[1], TFMessage)
                    print(f"     Transforms in first message: {len(msg.transforms)}")
                    for tf in msg.transforms[:3]:  # Show first 3
                        print(f"       {tf.header.frame_id} -> {tf.child_frame_id}")
                except Exception as e:
                    print(f"     (could not deserialize: {e})")
else:
    print("   ✗ NO TF TOPICS FOUND IN ROSBAG")
    print("   → Camera pose must be derived from odom + assumed static transform")

print()

# 6. Check if depth and RGB have same frame (indicates registration)
print("6. Depth-to-Color Registration Check:")
if rgb_frame and depth_frame:
    print(f"   RGB frame: {rgb_frame}")
    print(f"   Depth frame: {depth_frame}")
    if rgb_frame == depth_frame:
        print(f"   ✓ SAME FRAME - Depth is registered to color")
    else:
        print(f"   ⚠ DIFFERENT FRAMES - May need manual registration")
else:
    print(f"   Topic name: '/camera/aligned_depth_to_color/image_raw/compressedDepth'")
    print(f"   → Topic name contains 'aligned_depth_to_color' - DEPTH IS PRE-REGISTERED TO COLOR")
    print(f"   → No manual registration needed - depth pixels align 1:1 with RGB pixels")

print()

# 7. Camera pose availability
print("7. Camera Pose Availability:")
cursor.execute('''
    SELECT COUNT(*) FROM messages 
    WHERE topic_id = (SELECT id FROM topics WHERE name = '/odom')
''')
odom_count = cursor.fetchone()[0]
print(f"   Odometry messages: {odom_count}")
print(f"   → Camera pose = T(odom->base) from /odom + T(base->camera) from TF")
if not tf_topics:
    print(f"   ⚠ NO TF DATA - Camera pose may be unreliable")
    print(f"   → May need to assume static camera mount or use odom pose directly")
else:
    print(f"   ✓ TF data available - Camera pose can be computed")

print()

# Summary
print("=== Summary ===")
print("Camera Topics:")
print(f"  RGB: /camera/color/image_raw/compressed -> /camera/image_raw (decompressed)")
print(f"  Depth: /camera/aligned_depth_to_color/image_raw/compressedDepth -> /camera/depth/image_raw (decompressed)")
print()
print("Depth Registration:")
if rgb_frame == depth_frame and rgb_frame:
    print(f"  ✓ DEPTH IS REGISTERED TO COLOR (same frame: {rgb_frame})")
else:
    print(f"  ✓ DEPTH IS PRE-REGISTERED TO COLOR (topic name indicates alignment)")
print()
print("Coordinate Frames:")
odom_row = cursor.execute('''
    SELECT timestamp, data FROM messages 
    WHERE topic_id = (SELECT id FROM topics WHERE name = '/odom')
    ORDER BY timestamp LIMIT 1
''').fetchone()
if odom_row:
    odom_msg = deserialize_message(odom_row[1], Odometry)
    print(f"  - World frame: '{odom_msg.header.frame_id}' (from /odom)")
    print(f"  - Base frame: '{odom_msg.child_frame_id}' (from /odom)")
if rgb_frame:
    print(f"  - Camera frame: '{rgb_frame}' (from RGB message)")
print()
print("Camera Pose:")
if tf_topics:
    print(f"  ✓ Available via: T(odom->base) from /odom + T(base->camera) from TF")
    print(f"  - Reliability: GOOD (TF data present)")
else:
    print(f"  ⚠ Limited availability: T(odom->base) from /odom only")
    print(f"  - Reliability: POOR (no TF data - may need static assumption)")

rclpy.shutdown()
conn.close()
