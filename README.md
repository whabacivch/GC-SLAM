# Frobenius-Legendre SLAM POC (Impact Project v1)

**Compositional inference for dynamic SLAM using information-geometric methods**

---

## Quick Start

### Build
```bash
cd fl_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select fl_slam_poc
source install/setup.bash
```

### Run (Gazebo)
```bash
ros2 launch fl_slam_poc poc_tb3.launch.py model:=waffle use_sim_time:=true
```

### Run (Rosbag)
```bash
# Download sample bag (first time only)
./scripts/download_tb3_rosbag.sh

# Run integration test
./scripts/test-integration.sh
```

### Run (3D Point Cloud Mode)
```bash
# Download NVIDIA r2b dataset
./scripts/download_r2b_dataset.sh

# Run with GPU acceleration
ros2 launch fl_slam_poc poc_3d_rosbag.launch.py \
    bag:=rosbags/r2b_storage \
    play_bag:=true \
    use_gpu:=true

# See docs/3D_POINTCLOUD.md for full documentation
```

---

## Validation & Evaluation

FL-SLAM includes publication-quality validation against ground truth using standard SLAM metrics.

### Run with Evaluation

```bash
# Full pipeline: SLAM + metrics + plots
bash scripts/run_and_evaluate.sh
```

This computes:
- **ATE (Absolute Trajectory Error)**: Global consistency (translation + rotation)
- **RPE (Relative Pose Error)**: Local drift at multiple scales (1m, 5m, 10m)
- **Trajectory Validation**: Checks for timestamp issues and coordinate ranges
- **Publication-Quality Plots**: 4-view trajectory, error heatmap, pose graph

### Output Files

Results are saved to `results/m3dgr_YYYYMMDD_HHMMSS/` with:

**Trajectory Plots:**
- `trajectory_comparison.png` - 4-view overlay (XY, XZ, YZ, 3D)
- `trajectory_heatmap.png` - Trajectory colored by error magnitude
- `pose_graph.png` - Pose nodes with odometry edges

**Error Analysis:**
- `error_analysis.png` - Error over time + histogram distribution

**Metrics:**
- `metrics.txt` - Human-readable summary (ATE/RPE translation + rotation)
- `metrics.csv` - Spreadsheet-ready with all statistics

**Trajectories:**
- `estimated_trajectory.tum` - SLAM output in TUM format
- `ground_truth_aligned.tum` - Aligned ground truth

### Expected Performance (M3DGR Dynamic01)

Based on system design:
- **ATE Translation RMSE**: Target < 5m (outdoor dynamic sequence)
- **ATE Rotation RMSE**: Target < 5 deg
- **RPE @ 1m**: Target < 0.1 m/m (10% local drift)
- **Loop closures**: Detected via ICP + NIG descriptors
- **Map consistency**: Verified via anchor point clouds

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for detailed guide.

---

## Documentation

### Essential
- **[AGENTS.md](AGENTS.md)** - Design invariants and rules (P1-P7)
- **[CHANGELOG.md](CHANGELOG.md)** - Project history and decisions
- **[docs/POC_Testing_Status.md](docs/POC_Testing_Status.md)** - Current testing state

### Operational
- **[docs/GAZEBO_INTEGRATION.md](docs/GAZEBO_INTEGRATION.md)** - Gazebo setup and troubleshooting
- **[docs/ROSBAG.md](docs/ROSBAG.md)** - Rosbag testing workflow
- **[docs/TESTING.md](docs/TESTING.md)** - Testing framework and workflows
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** - Installation and setup guide
- **[docs/3D_POINTCLOUD.md](docs/3D_POINTCLOUD.md)** - 3D point cloud mode with GPU acceleration

### Reference
- **[docs/Comprehensive Information Geometry.md](docs/Comprehensive Information Geometry.md)** - Mathematical formulas
- **[docs/Project_Implimentation_Guide.sty](docs/Project_Implimentation_Guide.sty)** - Formal specification
- **[docs/MAP_VISUALIZATION.md](docs/MAP_VISUALIZATION.md)** - Visualization guide
- **[docs/ORDER_INVARIANCE.md](docs/ORDER_INVARIANCE.md)** - Order invariance documentation
- **[docs/PROJECT_RESOURCES_SUMMARY.md](docs/PROJECT_RESOURCES_SUMMARY.md)** - Project resources overview

---

## Architecture

### Core Principles (Non-Negotiable)
1. **P1**: Closed-form-first exactness
2. **P2**: Associative, order-robust fusion
3. **P3**: Legendre/Bregman foundations
4. **P4**: Frobenius third-order correction for approximations
5. **P5**: Soft association (no heuristic gating)
6. **P6**: One-shot loop correction by recomposition
7. **P7**: Local modularity

### Code Structure
```
fl_slam_poc/
├── operators/      # Core math (Gaussian, Dirichlet, ICP, distances) - EXACT
├── models/         # Generative models (NIG, birth, adaptive) - EXACT
├── geometry/       # SE(3) operations + documentation - EXACT
├── frontend/       # Helper modules (orchestration, call operators/models)
├── utils/          # Infrastructure (sensor sync, status monitoring)
├── nodes/          # ROS nodes (frontend, backend, simulation)
├── constants.py    # Centralized magic numbers
└── config.py       # Parameter grouping
```

### Data Flow
```
Sensors → Frontend (association + ICP) → LoopFactor → Backend (fusion) → State/Trajectory
Ground Truth Odom → Odom Bridge (abs→delta) ──────────────────────────────┘
```

### Key Nodes
- **frontend_node**: Data association via Fisher-Rao distances, ICP registration, anchor management
- **fl_backend_node**: Bregman barycenters, loop closure fusion, state estimation
- **sim_world_node**: Ground truth + noise simulation (for Gazebo)
- **tb3_odom_bridge_node**: Absolute → delta odometry conversion

---

## Mathematical Foundation

### Information Geometry
- **Exponential families** with Legendre duality
- **Bregman barycenters** for fusion
- **Fisher-Rao metric** for soft association
- **Frobenius structure** for third-order corrections

### Representations
- **SE(3)**: Rotation vectors (axis-angle) in `so(3)` tangent space
- **Covariance**: Tangent space at identity `[δx, δy, δz, δωx, δωy, δωz]`
- **Transport**: Via Adjoint representation (exact)

### Operations
- **NIG descriptors**: Normal-Inverse-Gamma model for probabilistic descriptors
- **Stochastic birth**: Poisson model with intensity λ = λ₀ * r_new
- **ICP**: Point cloud registration for loop factors
- **Frobenius correction**: Third-order retraction for linearization

---

## Dependencies

### System
- ROS 2 Jazzy
- Python 3.10+
- NumPy, SciPy

### Optional
- Gazebo (for simulation)
- TurtleBot3 packages (for Gazebo demo)
- Foxglove Bridge (for visualization)

### Install
```bash
pip install -r requirements.txt
```

---

## Testing

See **[docs/TESTING.md](docs/TESTING.md)** for complete testing documentation.

### Quick Validation (Minimal Tests)
```bash
./scripts/test-minimal.sh
```

Tests core functionality and mathematical invariants (~30 seconds).

### Full System Test (Integration)
```bash
./scripts/test-integration.sh
```

Tests complete SLAM pipeline with rosbag data (~90 seconds).

**Pass Criteria**:
- `/sim/anchor_create` published ≥1 times
- `/sim/loop_factor` published ≥1 times
- `/cdwm/backend_status` reports `mode: "SLAM_ACTIVE"`

---

## Docker

### Build
```bash
./scripts/docker-build.sh
```

### Run (Interactive)
```bash
./scripts/docker-run.sh
```

### Testing
```bash
# Quick validation
./scripts/docker-test.sh

# Full integration test
./scripts/docker-test-integration.sh
```

---

## Key Topics

### Input
- `/scan` - LaserScan
- `/odom` - Odometry (absolute or delta)
- `/camera/image_raw` - Image (optional)
- `/camera/depth/image_raw` - Depth (optional)

### Output
- `/sim/anchor_create` - Anchor creation events
- `/sim/loop_factor` - Loop closure constraints
- `/cdwm/state` - Estimated state
- `/cdwm/trajectory` - Trajectory path
- `/cdwm/op_report` - Operation reports (JSON)
- `/cdwm/frontend_status` - Frontend sensor status (JSON)
- `/cdwm/backend_status` - Backend mode and diagnostics (JSON)

---

## References

### Information Geometry
- Amari & Nagaoka (2000): *Methods of Information Geometry*
- Amari (2016): *Information Geometry and Its Applications*
- Combe (2022-2025): Frobenius statistical manifolds, quantum geometry, hexagonal webs

### Robotics
- Barfoot (2017): *State Estimation for Robotics*
- Miyamoto et al. (2024): Closed-form information distances

---

## License

[To be determined]

---

## Contact

[Project contact information]
