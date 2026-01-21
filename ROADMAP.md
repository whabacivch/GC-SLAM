# FL-SLAM Roadmap (Impact Project v1)

This roadmap is organized around the current **M3DGR rosbag MVP** pipeline and a clean separation between:
- **MVP operational code**: required to run `scripts/run_and_evaluate.sh`
- **Future/experimental code**: kept for later work, but not required for the MVP

## 1) MVP Status

**Primary entrypoint**
- `scripts/run_and_evaluate.sh`: runs the M3DGR Dynamic01 pipeline end-to-end (SLAM + plots/metrics).

**Launch**
- `fl_ws/src/fl_slam_poc/launch/poc_m3dgr_rosbag.launch.py`

**Nodes in the MVP pipeline**
- Frontend: `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/frontend_node.py`
- Backend: `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
- Utility: `fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/image_decompress.py`
- Utility: `fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/livox_converter.py`
- Utility: `fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/tb3_odom_bridge.py` (generic abs→delta odom bridge; legacy name)

**Evaluation**
- `scripts/align_ground_truth.py`
- `scripts/evaluate_slam.py`

## 2) Near-Term (Priority 1): Algorithm Fixes

### A) SE(3) drift investigation

**Symptom**
- Trajectory blows up (e.g., kilometers instead of meters).

**Primary files**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/common/transforms/se3.py`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/tb3_odom_bridge.py`

**Checklist**
- Verify pose composition conventions and frame semantics (odom/base).
- Confirm odometry is delta (twist-integrated) where expected.
- Validate quaternion/rotvec conversions and covariance transport.

### B) Timestamp monotonicity

**Symptom**
- Remaining non-monotonic gaps / duplicates impacting association and evaluation.

**Primary files**
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/processing/sensor_io.py`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/backend/parameters/timestamp.py`
- `scripts/align_ground_truth.py`

## 3) Medium-Term (Priority 2): Alternative Datasets

### A) TurtleBot3 (2D) validation

**Files**
- `phase2/fl_ws/src/fl_slam_poc/launch/poc_tb3_rosbag.launch.py`
- `scripts/download_tb3_rosbag.sh`

### B) NVIDIA r2b (3D) validation / GPU

**Files**
- `phase2/fl_ws/src/fl_slam_poc/launch/poc_3d_rosbag.launch.py`
- `scripts/download_r2b_dataset.sh`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/loops/pointcloud_gpu.py`

## 4) Medium-Term (Priority 3): Features

### A) Enable GPU acceleration
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/loops/pointcloud_gpu.py`

### B) RGB-D dense reconstruction integration
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/processing/rgbd_processor.py`
- `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/backend/fusion/multimodal_fusion.py`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/frontend/loops/vmf_geometry.py`

### C) Gazebo live testing
- `phase2/fl_ws/src/fl_slam_poc/launch/poc_tb3.launch.py`
- `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/utility_nodes/sim_world.py`

## 5) Long-Term (Priority 4): Research Features

### A) Dirichlet semantic SLAM integration
- `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/nodes/dirichlet_backend_node.py`
- `phase2/fl_ws/src/fl_slam_poc/fl_slam_poc/nodes/sim_semantics_node.py`
- `fl_ws/src/fl_slam_poc/fl_slam_poc/operators/dirichlet_geom.py`

### B) Visualization
- RViz: `config/fl_slam_rviz.rviz` (local/optional)
- Rerun bridge: removed from MVP; revisit later if needed
