# SLAM Dataset Download Guide

This guide explains how to download the recommended datasets for testing FL-SLAM with both 2D LaserScan and 3D PointCloud2 data.

---

## Already Downloaded ✓

### M3DGR Dynamic01 (Multi-sensor Ground Robot Dataset)
- **Location:** `rosbags/m3dgr/Dynamic01_ros2/`
- **Size:** 2.2 GB
- **Duration:** 175 seconds (~3 minutes)
- **Sensors:** 
  - Livox MID-360 LiDAR @ 10Hz
  - RealSense D435i RGB camera @ 15Hz
  - RealSense D435i depth @ 15Hz
  - **Wheel Odometry** @ 20Hz ✓✓✓
  - Multiple IMUs @ 200Hz
- **Status:** ✓ Converted to ROS 2, ready to test
- **Note:** LiDAR uses Livox CustomMsg format (needs converter to PointCloud2)

### r2b_storage (NVIDIA Isaac ROS Benchmark)
- **Location:** `rosbags/r2b_storage/`
- **Size:** 2.9 GB
- **Sensors:** LiDAR (Hesai XT32), RGB cameras (RealSense D455, HAWK stereo), Depth, IMU
- **Duration:** ~10 seconds
- **Issue:** No odometry topic - can't test SLAM directly
- **Use:** Good for perception benchmarking, but not SLAM

---

## Recommended Downloads

### 1. M3DGR Dataset ⭐⭐⭐ (BEST - HAS EVERYTHING!)

**Why this one:**
- Has **LiDAR + RGB cameras + Depth + Wheel Odometry + IMU + GNSS**
- Ground truth from RTK/Mocap
- Multiple challenging scenarios (dark, dynamic, slippery terrain)
- 40+ SLAM algorithms already tested on it
- **DESIGNED FOR SLAM** (not just benchmarking)

**Sensors:**
- Livox MID-360 LiDAR (360° FOV) @ 10Hz + Livox Avia @ 10Hz
- RealSense D435i RGB camera (640×480) @ 15Hz
- RealSense D435i depth camera @ 15Hz
- Insta360 X4 omnidirectional camera @ 15Hz
- Wheel odometer @ 20Hz ✓✓✓
- Multiple IMUs @ 200Hz
- GNSS @ 10Hz

**ROS Topics:**
```
/livox/mid360/lidar              - PointCloud2 @ 10Hz
/camera/color/image_raw/compressed - RGB @ 15Hz
/camera/aligned_depth_to_color/image_raw/compressedDepth - Depth @ 15Hz
/odom                            - Wheel Odometry @ 20Hz ✓✓✓
/camera/imu                      - IMU @ 200Hz
/livox/mid360/imu                - IMU @ 200Hz
```

**Recommended Sequences to Start:**

| Sequence | Size | Duration | Features | Rosbag Link | GT Link |
|----------|------|----------|----------|-------------|---------|
| **Outdoor01** ⭐ | 6.1GB | 411s | Standard outdoor | [OneDrive](https://1drv.ms/u/c/2b4bfc0edf421186/EYYRQt8O_EsggCstEAAAAAABZ7QHYpH3MyAxb6aOaAbdcQ?e=qNkjBh) / [Alipan](https://www.alipan.com/s/ZpMgYah2qJe) | [OneDrive](https://1drv.ms/t/c/2b4bfc0edf421186/EYYRQt8O_EsggCtLEAAAAAABkYZk3nHvsmV_KQ1o5-6fdw?e=BfWfty) / [Alipan](https://www.alipan.com/s/XaxjWT29UoM) |
| **Outdoor04** | 13.4GB | 782s | Longer outdoor | [OneDrive](https://1drv.ms/u/c/2b4bfc0edf421186/EYYRQt8O_EsggCsuEAAAAAABK6fj3Exz0XSflNv-v2IT8A?e=VDP9jC) / [Alipan](https://www.alipan.com/s/rva3bNuNEmU) | [OneDrive](https://1drv.ms/t/c/2b4bfc0edf421186/EYYRQt8O_EsggCtNEAAAAAABIZ7TtoQPhpkJhlT1t8eVUA?e=egPxew) / [Alipan](https://www.alipan.com/s/uiiCQZNqbnq) |
| **Dynamic01** | 2.1GB | 175s | Indoor, moving people | [OneDrive](https://1drv.ms/u/c/2b4bfc0edf421186/EYYRQt8O_EsggCv0DwAAAAAB-86r95z48cuIi_MTyIoq8A?e=IiMGzk) / [Alipan](https://www.alipan.com/s/RyMLSPqjuer) | [OneDrive](https://1drv.ms/t/c/2b4bfc0edf421186/EYYRQt8O_EsggCsOEAAAAAABoct7u6wv4vWo3w3qZMOmtg?e=Lv6zoE) / [Alipan](https://www.alipan.com/s/Q1d7PuCtRz7) |

**Download Instructions:**
1. Visit the OneDrive or Alipan link for the rosbag
2. Click download and save to: `rosbags/m3dgr/`
3. Download the corresponding GT file to the same directory
4. If using Alipan (Chinese cloud), you may need to merge split files by running the `.bat` script

**GitHub Repo:** https://github.com/sjtuyinjie/M3DGR

---

### 2. Newer College Dataset ⭐ (Alternative)

**Why this one:**
- Has LiDAR + cameras + odometry + IMU (everything we need!)
- Centimeter-accurate ground truth
- Loop closures present
- Well-documented and widely used

**Sensors:**
- Ouster OS-1 64-beam LiDAR @ 10Hz
- Intel RealSense D435i stereo cameras @ 30Hz
- IMU @ 100Hz
- Ground truth trajectory

**Download:**
1. Visit: https://tinyurl.com/newer-college
   - Or direct: https://drive.google.com/drive/folders/1ZJwQNherz-VZmZk1JxOWNP-D_ThXXmKZ

2. Navigate to folder: `2020-ouster-os1-64-realsense/rosbags/`

3. Download one of these sequences:
   - **`01_short_experiment.bag`** (~500MB) - Best to start with!
     - Short loop closure test
     - Quick to test
   - **`02_long_experiment.bag`** (~5GB)
     - Full 2.2km trajectory
     - Multiple loop closures
   - **`03_quad_with_dynamics.bag`** (~2GB)
     - Aggressive motion test

4. Save to: `rosbags/newer_college/`

**ROS Topics:**
```
/os1_cloud_node/points         - PointCloud2 @ 10Hz
/camera/infra1/image_rect_raw  - Left IR image @ 30Hz
/camera/infra2/image_rect_raw  - Right IR image @ 30Hz
/camera/imu                    - IMU @ 250Hz
/os1_cloud_node/imu            - LiDAR IMU @ 100Hz
/tf_static                     - Calibration transforms
```

**After Download:**
```bash
# Extract if compressed
cd rosbags/newer_college/
rosbag decompress 01_short_experiment.bag  # if needed

# Inspect topics
ros2 bag info 01_short_experiment.bag

# Use our inspection script
cd ../..
bash scripts/inspect_rosbag_topics.sh rosbags/newer_college/01_short_experiment.bag
```

---

### 2. M3DGR Dataset (Multi-scenario, for later)

**Why this one:**
- Multi-sensor, multi-scenario
- Degraded/challenging conditions
- 40+ SLAM methods benchmarked
- Good for robustness testing

**Sensors:**
- LiDAR (multiple configurations)
- RGB stereo cameras
- IMU + wheel odometry
- GNSS

**Download:**
1. Visit: https://github.com/sjtuyinjie/M3DGR

2. Check the README for latest download links
   - Some sequences are released
   - For early access, email: zhangjunjie587@gmail.com

3. Download a few diverse sequences to: `rosbags/m3dgr/`

**Status:** Sequences being released gradually (as of Jan 2026)

---

## Alternative: Oxford Spires Dataset (2025, NEW!)

**Mentioned in Newer College site as 10x larger successor**
- May have even better quality
- Check: https://ori-drs.github.io/newer-college-dataset/ for links

---

## Quick Test After Download

Once you have Newer College `01_short_experiment`:

```bash
# 3D Mode (PointCloud2)
POINTCLOUD_TOPIC=/os1_cloud_node/points \
ODOM_TOPIC=/integrated/odom \
## Phase 2 note
Alternative launch files are stored under `phase2/` and are not installed by the MVP package by default.
See: `phase2/fl_ws/src/fl_slam_poc/launch/poc_3d_rosbag.launch.py`

ros2 launch fl_slam_poc poc_3d_rosbag.launch.py play_bag:=true

# Or create a custom launch config
```

---

## Installation Notes

If you want to use `gdown` for automated Google Drive downloads:

```bash
pip3 install gdown

# Then download entire folder
cd rosbags/newer_college/
gdown --folder https://drive.google.com/drive/folders/[FOLDER_ID]
```

---

## Summary Table

| Dataset | Sensors | Odometry | Size | Best For |
|---------|---------|----------|------|----------|
| ✓ TB3 (have) | LaserScan, cameras | ✓ Yes | Small | 2D testing |
| ✓ r2b (have) | LiDAR, RGB, depth | ✗ No | 2.9GB | Perception only |
| → **M3DGR** ⭐ | LiDAR, RGB, depth, IMU | ✓ **Wheel + GNSS** | 2-13GB | **3D SLAM (BEST!)** |
| → Newer College | LiDAR, stereo, IMU | ✓ Yes (GT only) | 500MB-5GB | 3D SLAM alternative |

---

**Next Steps:**
1. **Download M3DGR `Outdoor01` sequence** (6.1GB, easiest to start)
   - Rosbag: https://1drv.ms/u/c/2b4bfc0edf421186/EYYRQt8O_EsggCstEAAAAAABZ7QHYpH3MyAxb6aOaAbdcQ?e=qNkjBh
   - GT: https://1drv.ms/t/c/2b4bfc0edf421186/EYYRQt8O_EsggCtLEAAAAAABkYZk3nHvsmV_KQ1o5-6fdw?e=BfWfty
   - Save both to: `rosbags/m3dgr/`
2. Run `bash scripts/inspect_rosbag_topics.sh rosbags/m3dgr/Outdoor01.bag`
3. Configure `poc_3d_rosbag.launch.py` with correct topic names:
   - `pointcloud_topic=/livox/mid360/lidar`
   - `odom_topic=/odom`
4. Run: `ros2 launch fl_slam_poc poc_3d_rosbag.launch.py play_bag:=true` (Phase 2)
