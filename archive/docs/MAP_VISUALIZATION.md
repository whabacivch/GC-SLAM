# Hybrid Laser + RGB-D Map Visualization

## Overview

The system implements a **dual-layer hybrid map architecture** that combines:
- **Sparse Anchor Modules** (Laser-based): Keyframe poses and point clouds
- **Dense 3D Modules** (RGB-D-based): High-density colored points with surface normals

Both layers use **information geometry** principles with exact closed-form operations.

## Architecture

### Map Layers

#### Layer 1: Sparse Anchors (Laser)
- **Color**: Yellow in `/cdwm/map`
- **Count**: ~20-50 per run
- **Purpose**: Pose estimation, loop closure, structural landmarks
- **Data**: SE(3) pose + covariance + point cloud

#### Layer 2: Dense Modules (RGB-D)
- **Color**: True RGB color from camera
- **Count**: ~1000-5000 per run
- **Purpose**: Dense reconstruction, appearance, surface normals
- **Data**: 3D position + color + vMF normal + opacity

### Sensor Fusion Strategy

At **anchor locations** (overlapping laser + RGB-D):
```
Position fusion (exact): Lambda_total = Lambda_laser_3d + Lambda_rgbd_3d
- Laser provides strong XY constraint (2D → 3D with weak Z prior)
- RGB-D provides Z + normals + color (3D)
```

Between anchors (RGB-D only):
```
- RGB-D creates independent dense modules
- No laser constraint (pure RGB-D evidence)
```

This is **information form addition** (exact, closed-form, associative).

## New Components

### Image Decompression Node
- **File**: `nodes/image_decompress_node.py`
- **Purpose**: Decompress JPEG/PNG images from rosbag compressed topics
- **Input**: `/stereo_camera/.../compressed` topics
- **Output**: `/camera/image_raw`, `/camera/depth/image_raw`

### RGB-D Processor
- **File**: `frontend/rgbd_processor.py`
- **Functions**:
  - `depth_to_pointcloud()`: Backproject depth to 3D with covariances
  - `compute_normals_from_depth()`: Estimate surface normals from gradients
  - `rgbd_to_evidence()`: Convert to exponential family evidence

### vMF Geometry Operators
- **File**: `operators/vmf_geometry.py`
- **Functions**:
  - `vmf_barycenter()`: Closed-form Bregman barycenter (WDVV associative)
  - `vmf_fisher_rao_distance()`: Exact metric via Bessel functions
  - `vmf_make_evidence()`, `vmf_mean_param()`: Parameter conversions

### Multi-Modal Fusion Operators
- **File**: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/gaussian_info.py` (core operators)
- **Functions**:
  - `fuse_info()`: Exact information form fusion (additive)
  - `make_evidence()`: Convert moment to information form
  - `embed_info_form()`: Lift low-dimensional evidence to full state

### Module Classes
- **File**: `nodes/fl_backend_node.py`
- **Classes**:
  - `SparseAnchorModule`: Laser keyframe with optional RGB-D fusion
  - `Dense3DModule`: RGB-D module with position, color, normal, opacity

## Foxglove Visualization

### Connect to System
```
ws://localhost:8765
```

### Topics

#### Dual-Layer Point Cloud Map
- **Topic**: `/cdwm/map`
- **Type**: `sensor_msgs/PointCloud2`
- **Content**: 
  - Sparse anchors (yellow points)
  - Dense modules (true RGB color)
- **Fields**: x, y, z, r, g, b

#### Trajectory
- **Topic**: `/cdwm/trajectory`
- **Type**: `nav_msgs/Path`
- **Color**: Cyan/Blue

#### Anchor Markers
- **Topic**: `/cdwm/loop_markers`
- **Type**: `visualization_msgs/MarkerArray`
- **Yellow spheres**: Anchor keyframes
- **Blue lines**: Loop closure constraints

#### Covariance Ellipse
- **Topic**: `/cdwm/markers`
- **Type**: `visualization_msgs/MarkerArray`
- **Green ellipsoid**: Current pose uncertainty

#### Backend Status
- **Topic**: `/cdwm/backend_status`
- **Type**: JSON string
- **Fields**: 
  - `mode`: SLAM_ACTIVE, DEAD_RECKONING, SLAM_STALE
  - `sparse_anchors`: Count of laser keyframes
  - `dense_modules`: Count of RGB-D modules
  - `rgbd_fused_anchors`: Anchors with RGB-D fusion

### 3D Panel Settings
- **Fixed frame**: `odom`
- **Follow mode**: `base_link`
- **Show TF**: Enabled
- **Grid**: Enabled
- **Point size**: 2-4 (adjust for visibility)

## Running the System

### With RGB-D Enabled (Default)
```bash
cd "/home/will/Documents/Coding/Phantom Fellowship MIT/Impact Project_v1"
source /opt/ros/jazzy/setup.bash
source fl_ws/install/setup.bash

# MVP launch file (M3DGR dataset with Livox conversion)
ros2 launch fl_slam_poc poc_m3dgr_rosbag.launch.py \
  bag:=rosbags/m3dgr/Dynamic01_ros2 \
  play_bag:=true \
  enable_decompress_cpp:=true \
  enable_image:=true \
  enable_depth:=true
```

### Without RGB-D (LiDAR-Only)
```bash
ros2 launch fl_slam_poc poc_m3dgr_rosbag.launch.py \
  bag:=rosbags/m3dgr/Dynamic01_ros2 \
  play_bag:=true \
  enable_decompress_cpp:=false \
  enable_image:=false \
  enable_depth:=false
```

> **Note:** Alternative dataset launch files (TurtleBot3, r2b) are planned for future work.
> Currently, only the M3DGR pipeline is fully supported via `poc_m3dgr_rosbag.launch.py`.

## Information Geometry Compliance

### P1: Closed-Form Operations ✅
- Gaussian fusion: Information form addition (exact)
- vMF fusion: Bregman barycenter (exact via Bessel)
- Sensor fusion: Information addition (exact)

### P2: Associative Fusion ✅
- WDVV associativity verified in tests
- `(A + B) + C = A + (B + C)` for all operations

### P3: Metric Distances ✅
- vMF Fisher-Rao: True metric, triangle inequality
- Position: Mahalanobis via information form

### P4-P7: Other Invariants ✅
- No magic numbers (uses principled priors)
- Full provenance (OpReport logging)
- Explicit approximation triggers (none needed for fusion)
- Order-robust (commutative fusion)

## Architecture Benefits

### Why Hybrid?
1. **Laser + RGB-D synergy**: Best of both sensors
2. **Accurate XY from laser**: LIDAR precision in ground plane
3. **Accurate Z from RGB-D**: Depth camera vertical information
4. **Dense appearance**: True colors, surface normals for rendering
5. **Lightweight**: Only fuse at anchors, not every frame

### Information-Theoretic Justification
- Sensor fusion via **information addition** is mathematically exact
- No approximations in fusion (unlike EKF sensor fusion)
- Laser and RGB-D evidence combine additively in natural parameter space
- Higher precision dominates automatically (no tuning weights)

## Troubleshooting

### No RGB-D Data in Map
```bash
# Check decompress node
ros2 topic echo /camera/image_raw --once
ros2 topic echo /camera/depth/image_raw --once

# Check if cv_bridge is available
python3 -c "from cv_bridge import CvBridge; print('cv_bridge OK')"
```

### Map Has Only Yellow Points (No RGB)
- RGB-D processing not enabled or not receiving images
- Check `enable_decompress_cpp:=true` and `enable_image:=true` in launch

### Backend Status Shows 0 Dense Modules
- RGB-D evidence not being processed
- Check frontend logs for RGB-D subscription status

### Import Errors in Test
```bash
# Ensure workspace is sourced
source fl_ws/install/setup.bash

# Run tests
cd fl_ws
colcon test --packages-select fl_slam_poc
```

## Next Steps

### Gaussian Splatting Renderer (Future)
- Render dense modules as 3D Gaussians
- Use vMF normals for shading
- Publish rendered images on `/cdwm/splat_image`

### Semantic Fusion (Future)
- Add Dirichlet semantics to dense modules
- Fuse semantic labels via information projection

### Nav2 Integration (Future)
- Convert dense map to occupancy grid
- Publish `nav_msgs/OccupancyGrid` for navigation
