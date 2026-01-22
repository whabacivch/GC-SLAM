# Bag Topics and Usage Map (Canonical)

This document is the single source of truth for:

- **What topics/messages exist** in supported datasets/bags
- **Which ones FL-SLAM consumes today** (MVP)
- **Which ones are present but not yet used** (roadmap)
- **Non-negotiable assumptions** (frames, TF presence, intrinsics, extrinsics)

Use `tools/inspect_rosbag_deep.py` to regenerate counts/frames when bags change.

## Conventions

- **Bag truth**: values observed directly from the rosbag database (`*.db3`). No guessing.
- **Frame IDs**:
  - `header.frame_id` is the *data frame* for the message.
  - `child_frame_id` (Odometry) is the *body frame* being tracked.
- **No-TF mode**:
  - If `/tf` and `/tf_static` are absent, extrinsics must be provided as parameters.
  - For LiDAR pointcloud processing, use `lidar_base_extrinsic = T_base_lidar` (6DOF) to map points into the base frame.

## M3DGR Dynamic01 (ROS 2 bag)

### Summary (bag truth)

- **TF**: `/tf` and `/tf_static` are **absent**
- **CameraInfo**: `sensor_msgs/msg/CameraInfo` topics are **absent**
- **Odom frames**:
  - `/odom.header.frame_id = odom_combined`
  - `/odom.child_frame_id = base_footprint`
- **Camera frames**:
  - `/camera/color/image_raw/compressed.header.frame_id = camera_color_optical_frame`
  - `/camera/aligned_depth_to_color/image_raw/compressedDepth.header.frame_id = camera_color_optical_frame`
  - `/camera/imu.header.frame_id = camera_imu_optical_frame`
- **Livox frames**:
  - `/livox/mid360/lidar.header.frame_id = livox_frame`
  - `/livox/mid360/imu.header.frame_id = livox_frame`
  - `/livox/avia/*` topics may be present, but require `livox_ros_driver` message package to decode.

### Topic inventory (observed)

Run:

```bash
source /opt/ros/jazzy/setup.bash
source fl_ws/install/setup.bash
python3 tools/inspect_rosbag_deep.py rosbags/m3dgr/Dynamic01_ros2 --json /tmp/m3dgr_summary.json
```

Key topics:

| Topic | Message type | Used now? | How it is used |
| --- | --- | --- | --- |
| `/odom` | `nav_msgs/msg/Odometry` | **Yes** | Converted to delta odom by `tb3_odom_bridge` and fused in backend (`/sim/odom`). Frame truth: `odom_combined -> base_footprint`. |
| `/livox/mid360/lidar` | `livox_ros_driver2/msg/CustomMsg` | **Yes** | Converted to `PointCloud2` by `livox_converter` and consumed by frontend pointcloud path. Preserves `frame_id=livox_frame`. |
| `/livox/mid360/imu` | `sensor_msgs/msg/Imu` | Not yet (MVP) | Present for future LiDAR-IMU coupling / calibration; currently frontend uses `/camera/imu`. |
| `/camera/imu` | `sensor_msgs/msg/Imu` | **Yes** | Frontend buffers IMU and publishes `/sim/imu_segment` (Contract B); backend re-integrates. Frame: `camera_imu_optical_frame`. |
| `/camera/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Optional | Decompressed by `image_decompress_cpp` to `/camera/image_raw`. Downstream usage depends on `enable_image/publish_rgbd_evidence`. |
| `/camera/aligned_depth_to_color/image_raw/compressedDepth` | `sensor_msgs/msg/CompressedImage` | Optional | Decompressed by `image_decompress_cpp` to `/camera/depth/image_raw`. Depth is aligned to color (same frame). |
| `/vrpn_client_node/UGV/pose` | `geometry_msgs/msg/PoseStamped` | **Evaluation only** | Ground-truth source for offline evaluation. Must never be fused into inference. |
| `/livox/avia/lidar` | `livox_ros_driver/msg/CustomMsg` | Not yet | Present in bag; requires `livox_ros_driver` Python messages to decode and use. |
| `/livox/avia/imu` | `sensor_msgs/msg/Imu` | Not yet | Present for future work; currently unused. |

### Current LiDAR conversion contract (MVP)

- **Converter**: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/livox_converter.py`
- **Output**: `sensor_msgs/msg/PointCloud2` on `/lidar/points`
- **Fields** (current):
  - `x,y,z` (float32)
  - `intensity` (uint8, from reflectivity)
  - `ring` (uint8, from line)
  - `tag` (uint8)
  - `timebase` (uint64, message-level timebase preserved per point)
- **Per-point time offset**: not available in `livox_ros_driver2` MID360 messages; if available for other drivers it will be included as `time_offset`.

### Critical assumptions (must be explicit)

- With no TF in bag, **LiDAR extrinsics are unknown unless declared**.
  - Use launch arg `lidar_base_extrinsic` as `T_base_lidar = [x,y,z,rx,ry,rz]`.
  - If omitted, frontend will warn and pointcloud may remain in sensor frame.
- With no CameraInfo, camera intrinsics must be provided via parameters and logged.

## TurtleBot3 / other bags (placeholder)

Add sections here as datasets are added.

Each section should include:

- bag truth frames
- TF/camera_info presence
- topic inventory and what the pipeline consumes
## NVIDIA r2b / other bags (placeholder)

Add sections here as datasets are added.

Each section should include:

- bag truth frames
- TF/camera_info presence
- topic inventory and what the pipeline consumes

