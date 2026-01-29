# M3DGR Dataset Status

**Official repo:** [github.com/sjtuyinjie/M3DGR](https://github.com/sjtuyinjie/M3DGR) (IROS 2025; GT in TUM format from RTK/Mocap; evaluation via evo; "TF tree method" for trajectory in TUM). See `docs/TRACE_TRAJECTORY_AND_GROUND_TRUTH.md` for trajectory/GT frame trace.

## ✓ Completed

1. **Downloaded M3DGR Dynamic01 sequence** (2.2 GB, 175 seconds)
   - Indoor sequence with dynamic people
   - Multiple sensor modalities
   
2. **Converted ROS 1 bag to ROS 2 format** using `rosbags` library
   - Input: `Dynamic01.bag` (ROS 1)
   - Output: `Dynamic01_ros2/` (ROS 2)
   - Conversion successful: 175,337 messages

3. **Verified bag contents:**
   ```
   ✓ /odom - nav_msgs/Odometry @ 20Hz (3,504 messages)
   ✓ /camera/color/image_raw/compressed - RGB @ 15Hz (5,251 messages)
   ✓ /camera/aligned_depth_to_color/image_raw/compressedDepth - Depth (5,251 messages)
   ✓ /camera/imu - IMU @ 200Hz (35,024 messages)
   ✓ /livox/mid360/imu - IMU @ 200Hz (35,032 messages)
   ⚠ /livox/mid360/lidar - Livox CustomMsg @ 10Hz (1,752 messages)
   ```

4. **Bag playback works** - Topics are being published

## ⚠️ Remaining Issue

**LiDAR Data Format:** The LiDAR topic uses **Livox's proprietary CustomMsg format**, not standard `sensor_msgs/PointCloud2`.

### The Problem:
- Our FL-SLAM system expects `sensor_msgs/PointCloud2`
- M3DGR uses `livox_ros_driver2/msg/CustomMsg`
- This custom message type is not in our ROS 2 workspace

### CustomMsg Structure (from Livox documentation):
```
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
livox_ros_driver2/CustomPoint[] points

CustomPoint:
  float32 x, y, z
  uint8 reflectivity
  uint8 tag
  uint8 line
```

## Solutions (Pick One)

### Option 1: Install Livox ROS 2 Driver ⭐ (Simplest if available)
```bash
# Check if Livox driver exists for ROS 2 Jazzy
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
cd livox_ros_driver2
# Build and install message definitions
```

### Option 2: Create CustomMsg Definitions in Our Workspace
Add message definitions to `fl_slam_poc`:
1. Create `livox_ros_driver2/msg/CustomMsg.msg`
2. Create `livox_ros_driver2/msg/CustomPoint.msg`
3. Add to `CMakeLists.txt`
4. Rebuild workspace
5. Create converter node

### Option 3: Pre-process Bag Offline
Write Python script using `rosbags` library to:
- Read Livox CustomMsg from bag
- Convert to PointCloud2
- Write new bag with added PointCloud2 topic

### Option 4: Contact M3DGR Authors
Email: `zhangjunjie587@gmail.com`
- Ask if they have ROS 2 bags with PointCloud2
- Or ask for Livox message definitions

### Option 5: Use Different Dataset
- Oxford Spires (needs manual download)
- Newer College (needs registration)
- Record own data with PointCloud2-compatible LiDAR

## Recommended Next Step

**Try Option 1 first** - Check if Livox ROS 2 driver exists and can be built for Jazzy.

If not available → **Option 2** (add message definitions to our workspace)

## Testing Once Conversion is Done

Once we have PointCloud2 data available:

```bash
cd ~/Documents/Coding/Phantom\ Fellowship\ MIT/Impact\ Project_v1

# Run the current MVP pipeline (SLAM + plots/metrics)
bash tools/run_and_evaluate.sh
```

## Current Bag Location

```
rosbags/m3dgr/
├── Dynamic01.bag              # Original ROS 1 bag (2.2 GB)
├── Dynamic01_ros2/            # Converted ROS 2 bag
│   ├── Dynamic01_ros2.db3     # SQLite database (2.2 GB)
│   └── metadata.yaml
├── Dynamic01.txt              # Ground truth trajectory (4.2 MB)
└── convert_config.yaml
```

## Summary

**We're 90% there!** The bag is converted, has odometry, and plays correctly. We just need to handle the Livox message format to get PointCloud2 data for FL-SLAM to process.
