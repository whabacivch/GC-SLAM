# Velodyne VLP-16 (Puck) Overview

The Velodyne VLP-16, also known as the Puck, is a compact, 16-channel mechanical spinning LiDAR sensor designed for 3D mapping, autonomous navigation, and robotics applications. It was originally developed by Velodyne LiDAR (now part of Ouster, which offers an updated version with enhanced specs like 40° vertical FOV and 200 m range). For datasets like the 10_14 sequences (campus_outdoor_1014_compressed), it refers to the legacy Velodyne model, widely used in research datasets for its affordability and reliability in outdoor environments.

It is a time-of-flight sensor using 16 laser/detector pairs that spin at 5–20 Hz to generate real-time 3D point clouds. Data is output via Ethernet (UDP packets), commonly recorded in .pcap format and processed in tools like ROS, PCL, or VeloView.

---

## Key Specifications

| Property | Value |
|----------|--------|
| **Channels** | 16 lasers |
| **Range** | Up to 100 m |
| **Range Accuracy** | ±3 cm (typical) |
| **Field of View** | Horizontal: 360° (continuous); Vertical: 30° (±15° from horizontal) |
| **Angular Resolution** | Horizontal: 0.1°–0.4° (depends on rotation rate; e.g. ~0.2° at 10 Hz); Vertical: ~2° average (non-uniform spacing) |
| **Rotation Rate** | 5–20 Hz (300–1200 RPM, configurable in 60 RPM increments) |
| **Point Rate** | ~300,000 points/s (single return); ~600,000 (dual return) |
| **Return Modes** | Strongest, Last, or Dual (strongest + last) |
| **Laser** | 905 nm, Class 1 eye-safe |
| **Power** | ~8 W average |
| **Weight** | ~830 g |
| **Dimensions** | ~103 mm diameter × 72 mm height |
| **Data Output** | UDP (port 2368 data, 8308 position/GPS); distance (2 mm resolution), calibrated intensity (0–255), azimuth, timestamps |

---

## Vertical Beam Angles and Calibration

The 16 lasers have fixed, interleaved vertical angles (±15° total FOV) to minimize crosstalk and provide better coverage. Angles are not uniformly spaced—lower beams (even IDs) cover the bottom half, upper (odd IDs) the top half, fired in an alternating sequence.

| Laser ID | Vertical Angle (°) | Vertical Offset Correction (mm) |
|----------|--------------------|---------------------------------|
| 0        | -15                | 11.2                            |
| 1        | 1                  | -0.7                            |
| 2        | -13                | 9.7                             |
| 3        | 3                  | -2.2                            |
| 4        | -11                | 8.1                             |
| 5        | 5                  | -3.7                            |
| 6        | -9                 | 6.6                             |
| 7        | 7                  | -5.1                            |
| 8        | -7                 | 5.1                             |
| 9        | 9                  | -6.6                            |
| 10       | -5                 | 3.7                             |
| 11       | 11                 | -8.1                            |
| 12       | -3                 | 2.2                             |
| 13       | 13                 | -9.7                            |
| 14       | -1                 | 0.7                             |
| 15       | 15                 | -11.2                           |

These offsets correct for the physical position of each laser relative to the sensor origin. In processing (e.g. ROS `velodyne_pointcloud` package), use calibration files (e.g. VLP-16.yaml or .xml) that include these `vert_correction` values, plus rotational/horizontal offsets if needed. Point coordinates are computed as:

- **X** = distance × cos(ω) × sin(α)
- **Y** = distance × cos(ω) × cos(α)
- **Z** = distance × sin(ω)

where **α** = azimuth, **ω** = vertical angle.

---

## Key Considerations for Testing SLAM

The VLP-16 is popular for SLAM research (e.g. LOAM, LeGO-LOAM, A-LOAM, HDL-Graph-SLAM) due to its ring-structured point clouds, but its limitations require careful handling.

### 1. Sparse Vertical Resolution

- Only 16 beams over 30° FOV means lower point density vertically compared to 32/64-channel sensors.
- In campus outdoor environments (buildings, trees, ground), this works well for structured features but can struggle in sparse/open areas or with heavy vegetation (fewer planar/edge features).
- Algorithms like LOAM leverage **ring** labels (laser ID) for feature extraction (edges/planes)—preserve laser IDs in your pipeline.

### 2. Motion Distortion

- Full 360° scan takes ~0.1 s at 10 Hz. Vehicle/platform motion during a sweep distorts the point cloud (rolling shutter effect).
- Mitigate by:
  - Undistorting points using high-rate IMU data (linear interpolation of poses).
  - Assuming constant velocity within a scan.
- Critical for dynamic sequences like driving on campus paths.

### 3. Return Modes and Intensity

- Dual returns help in vegetation/rain (separates foreground/background).
- Calibrated intensity useful for feature matching or ground segmentation.

### 4. Data Handling

- **Raw data**: .pcap files with UDP packets.
- **Timestamps**: GPS-synced (PPS + NMEA for UTC alignment)—ensure synchronization if your dataset includes GPS/IMU.
- Horizontal density increases with higher RPM; 10 Hz is common for balance (avoids excessive blur).

### 5. SLAM Performance Tips

- Pair with IMU/GPS for better odometry (e.g. LIO-SAM, FAST-LIO).
- Expect challenges with pitch/roll (limited vertical overlap between scans).
- Test in structured areas first; campus sequences likely have good ground/building features.
- Common issues: noise in bright sunlight, range drop on low-reflectivity surfaces.
- Validate with ground truth (if available) for loop closure and drift.

Many open-source tools (ROS velodyne driver, PCL) have built-in VLP-16 support with the above calibration. Start with 10 Hz, single/dual return, and undistortion enabled for reliable results in outdoor campus testing. If your dataset is from a specific collection (e.g. Boreas or similar), check for provided calibration files.

---

## Noise prior for GC (lidar_sigma_meas)

- **Range accuracy:** ±3 cm → variance per axis ≈ (0.03)² ≈ 9e-4 m².
- **Angular:** ~0.15° → combined with range gives order ~1e-3 m²/axis for Cartesian noise.
- **GC config:** Use `lidar_sigma_meas: 1e-3` (m² isotropic) in Kimera/VLP-16 profile (e.g. `config/gc_kimera.yaml`). See `docs/KIMERA_FRAME_MAPPING.md` and `docs/POINTCLOUD2_LAYOUTS.md`.

---

## Relation to This Project

- **Kimera-Multi-Data 10_14** (Campus-Outdoor): LiDAR point clouds are from **Velodyne VLP-16** (`/xxx/lidar_points`, `sensor_msgs/PointCloud2`). See `rosbags/Kimera_Data/TOPICS.md` and `PREP_README.md`.
- **Kimera** (current GC eval): Uses **Velodyne VLP-16** (`/acl_jackal/lidar_points`). See `docs/BAG_TOPICS_AND_USAGE.md` and [KIMERA_FRAME_MAPPING.md](KIMERA_FRAME_MAPPING.md).
