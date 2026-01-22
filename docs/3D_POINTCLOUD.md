# 3D Point Cloud Support

This document describes FL-SLAM's 3D point cloud processing capabilities with GPU acceleration.

## Overview

FL-SLAM supports two sensor modalities:
1. **2D LaserScan** (default) - Traditional 2D LIDAR for planar SLAM
2. **3D PointCloud2** - Full 3D point cloud for volumetric SLAM

Both modes use the same information-geometric backend - the Frobenius-Legendre framework is dimension-agnostic.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Node                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  SensorIO   │ -> │  GPU Processor   │ -> │ Loop Processor│  │
│  │ (PC2 input) │    │ (voxel + ICP)    │    │               │  │
│  └─────────────┘    └──────────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend Node                                 │
│              (Information Form - No Changes!)                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SE(3) State: (L, h) Information Form                    │   │
│  │  Fusion: L_fused = L1 + L2 (exact, associative)         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Enable 3D Mode

Set these parameters in your launch file:

```python
{
    "use_3d_pointcloud": True,        # Switch to 3D mode
    "enable_pointcloud": True,        # Subscribe to PointCloud2
    "pointcloud_topic": "/lidar/points",  # PointCloud2 topic (override to /camera/depth/points for RGB-D cameras)
    "use_gpu": True,                  # Enable GPU acceleration
    "voxel_size": 0.05,               # Voxel grid size (meters)
}
```

Optional (recommended when RGB-D is available): include camera appearance + depth histograms in soft association:

```python
{
    "enable_image": True,
    "enable_depth": True,
    "rgbd_sync_max_dt_sec": 0.1,      # Max RGB↔depth timestamp mismatch
    "rgbd_min_depth_m": 0.1,          # Depth descriptor min range (m)
    "rgbd_max_depth_m": 10.0,         # Depth descriptor max range (m)
}
```

### GPU Configuration

```python
{
    "use_gpu": True,                  # Enable GPU acceleration
    "gpu_device_index": 0,            # CUDA device index
    "gpu_fallback_to_cpu": True,      # Fall back if GPU unavailable
}
```

### Point Cloud Processing

```python
{
    "voxel_size": 0.05,               # Downsample resolution (m)
    "max_points_after_filter": 50000, # Max points to process
    "min_points_for_icp": 100,        # Minimum points for ICP
    "icp_max_correspondence_distance": 0.5,  # Max ICP distance (m)
    "pointcloud_rate_limit_hz": 30.0, # Rate limit for processing
}
```

## Launch Files

### 3D Rosbag Playback

```bash
# Phase 2 note: this launch file lives under `phase2/` and is not installed by the MVP package by default.
# See: `phase2/fl_ws/src/fl_slam_poc/launch/poc_3d_rosbag.launch.py`
```

### Enable 3D in Existing Launch

```bash
# Phase 2 note: this launch file lives under `phase2/` and is not installed by the MVP package by default.
# See: `phase2/fl_ws/src/fl_slam_poc/launch/poc_tb3_rosbag.launch.py`
```

## Compatible Datasets

### NVIDIA r2b Dataset

The ROS2 Benchmark dataset contains RealSense D455 data with point clouds:

```bash
# Download dataset
./tools/download_r2b_dataset.sh

# Test with FL-SLAM
# Phase 2 note: see `phase2/fl_ws/src/fl_slam_poc/launch/poc_3d_rosbag.launch.py`
```

### Custom Rosbags

Any rosbag with PointCloud2 messages is compatible:

**Required topics:**
- PointCloud2 (e.g., `/lidar/points` or `/camera/depth/points`)
- Odometry (e.g., `/odom`)

**Optional topics:**
- RGB images for visualization
- TF transforms

## GPU Acceleration

### Requirements

- NVIDIA GPU with CUDA support
- Open3D >= 0.18.0 with CUDA support
- Tested on RTX 4050

### Installation

```bash
pip install open3d>=0.18.0
# Or for CUDA support:
pip install open3d-gpu>=0.18.0  # If available
```

### Performance

| Operation | CPU | GPU (RTX 4050) |
|-----------|-----|----------------|
| Voxel Filter (50K pts) | ~50ms | ~5ms |
| ICP (10K pts) | ~100ms | ~20ms |

### Fallback

If GPU is unavailable, the system automatically falls back to CPU processing:

```python
if is_gpu_available():
    # Use Open3D GPU pipeline
else:
    # Use numpy/scipy CPU pipeline
```

## Design Invariants

The 3D point cloud upgrade maintains all FL-SLAM design invariants:

### P1: Closed-form-first exactness
- ICP uses SVD-based closed-form registration
- Backend fusion is exact Gaussian fusion in information form

### P2: Associative, order-robust fusion
- Information form fusion: L_fused = L1 + L2
- Commutative and associative by construction

### P6: One-shot loop correction
- Loop factors applied directly via information addition
- No iterative re-optimization

### Backend Unchanged
The backend is **dimension-agnostic** - it operates on (L, h) information form regardless of whether evidence came from 2D or 3D sensors.

## API Reference

### GPUPointCloudProcessor

```python
from fl_slam_poc.operators.pointcloud_gpu import GPUPointCloudProcessor

proc = GPUPointCloudProcessor(
    voxel_size=0.05,
    max_correspondence_distance=0.5,
    device_index=0,
    fallback_to_cpu=True
)

# Voxel filter
filtered_points = proc.voxel_filter(points)

# ICP registration
result = proc.icp(source, target, init=None, max_iter=15, tol=1e-4)
```

### Convenience Functions

```python
from fl_slam_poc.operators.pointcloud_gpu import (
    is_gpu_available,
    voxel_filter_gpu,
    icp_gpu,
)

# Check GPU availability
if is_gpu_available():
    print("GPU acceleration available")

# Quick voxel filter
filtered = voxel_filter_gpu(points, voxel_size=0.05)

# Quick ICP
result = icp_gpu(source, target, max_iter=20)
```

### SensorIO 3D Mode

```python
# SensorIO automatically subscribes to PointCloud2 in 3D mode
sensor_io = SensorIO(node, {
    "use_3d_pointcloud": True,
    "pointcloud_topic": "/lidar/points",
    ...
})

# Get latest point cloud
points, timestamp, frame_id = sensor_io.get_latest_pointcloud()

# Check mode
if sensor_io.is_3d_mode():
    print("Running in 3D point cloud mode")
```

## Testing

### Unit Tests

```bash
cd fl_ws
colcon build --symlink-install
source install/setup.bash
pytest src/fl_slam_poc/test/test_pointcloud_3d.py -v
```

### Integration Test

```bash
# Download test data
./tools/download_r2b_dataset.sh

# Phase 2 note: the 3D alternative launch file lives under `phase2/` and is not installed by the MVP package.
# See: `phase2/fl_ws/src/fl_slam_poc/launch/poc_3d_rosbag.launch.py`
```

## Troubleshooting

### GPU Not Detected

```python
from fl_slam_poc.frontend.pointcloud_gpu import is_gpu_available
print(f"GPU available: {is_gpu_available()}")
```

Check CUDA installation:
```bash
nvidia-smi
python -c "import open3d; print(open3d.core.cuda.is_available())"
```

### High Memory Usage

Reduce `max_points_after_filter` or increase `voxel_size`:

```python
{
    "voxel_size": 0.1,  # Larger voxels = fewer points
    "max_points_after_filter": 20000,
}
```

### ICP Not Converging

Increase max iterations or correspondence distance:

```python
{
    "icp_max_iter_prior": 30,
    "icp_max_correspondence_distance": 1.0,
}
```

### Rate Limiting

If point clouds arrive too fast, increase rate limiting:

```python
{
    "pointcloud_rate_limit_hz": 15.0,  # Process fewer frames
}
```
