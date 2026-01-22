# Frobenius-Legendre SLAM POC (Impact Project v1)

**Compositional inference for dynamic SLAM using information-geometric methods**

---

## MVP Status

The current **MVP (Minimum Viable Product)** focuses on the **M3DGR Dynamic01 rosbag** evaluation pipeline. This is the smallest reproducible case for validating the FL-SLAM algorithm.

**MVP Components:**
- M3DGR rosbag processing (Livox LiDAR + RGB-D)
- Frontend: sensor association, ICP loop detection, anchor management
- Backend: information-geometric fusion, trajectory estimation
- Evaluation: ATE/RPE metrics and publication-quality plots

**Phase 2 (Future/Experimental):** Alternative datasets, Gazebo simulation, Dirichlet semantic SLAM, and other experimental features are kept in `phase2/` to maintain a minimal MVP surface area. See [`phase2/README.md`](phase2/README.md) for details.

**Current Priorities:** See [`ROADMAP.md`](ROADMAP.md) for algorithm fixes (SE(3) drift, timestamp issues) and planned features.

---

## Quick Start (MVP)

### Build
```bash
cd fl_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select fl_slam_poc
source install/setup.bash
```

### Run MVP Pipeline
```bash
# Full pipeline: SLAM + metrics + plots
bash tools/run_and_evaluate.sh
```

This runs the M3DGR Dynamic01 rosbag through the complete SLAM pipeline and generates evaluation metrics/plots in `results/m3dgr_YYYYMMDD_HHMMSS/`.

### Alternative Datasets (Phase 2)

For TB3 rosbags, Gazebo simulation, or 3D point cloud datasets, see:
- [`phase2/README.md`](phase2/README.md) - How to re-enable Phase 2 features
- [`legacy_docs/GAZEBO_INTEGRATION.md`](legacy_docs/GAZEBO_INTEGRATION.md) - Gazebo setup
- [`docs/3D_POINTCLOUD.md`](docs/3D_POINTCLOUD.md) - 3D point cloud mode

---

## Validation & Evaluation

FL-SLAM includes publication-quality validation against ground truth using standard SLAM metrics.

### Run with Evaluation

```bash
# Full pipeline: SLAM + metrics + plots
bash tools/run_and_evaluate.sh
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

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for detailed evaluation guide.

**Current Status & Next Steps:** See [`ROADMAP.md`](ROADMAP.md) for:
- Priority 1: Algorithm fixes (SE(3) drift, timestamp monotonicity)
- Priority 2-4: Alternative datasets, GPU acceleration, research features

---

## Documentation

### Essential (Start Here)
- **[ROADMAP.md](ROADMAP.md)** - Current priorities and planned work (MVP status, algorithm fixes, future features)
- **[AGENTS.md](AGENTS.md)** - Design invariants and rules (P1-P7)
- **[CHANGELOG.md](CHANGELOG.md)** - Project history and decisions
- **[phase2/README.md](phase2/README.md)** - Phase 2 features and how to re-enable them

### MVP Workflow
- **[docs/EVALUATION.md](docs/EVALUATION.md)** - Evaluation metrics and plots
- **[docs/ROSBAG.md](docs/ROSBAG.md)** - Rosbag testing workflow
- **[docs/TESTING.md](docs/TESTING.md)** - Testing framework and workflows
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** - Installation and setup guide

### Phase 2 / Advanced
- **[legacy_docs/GAZEBO_INTEGRATION.md](legacy_docs/GAZEBO_INTEGRATION.md)** - Gazebo setup (Phase 2)
- **[docs/3D_POINTCLOUD.md](docs/3D_POINTCLOUD.md)** - 3D point cloud mode with GPU acceleration

### Reference
- **[docs/Comprehensive Information Geometry.md](docs/Comprehensive Information Geometry.md)** - Mathematical formulas
- **[docs/Project_Implimentation_Guide.sty](docs/Project_Implimentation_Guide.sty)** - Formal specification
- **[docs/MAP_VISUALIZATION.md](docs/MAP_VISUALIZATION.md)** - Visualization guide
- **[docs/ORDER_INVARIANCE.md](docs/ORDER_INVARIANCE.md)** - Order invariance documentation
- **[docs/POC_Testing_Status.md](docs/POC_Testing_Status.md)** - Current testing state

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

### Code Structure (MVP)
```
fl_slam_poc/
├── frontend/        # Frontend orchestration + sensor I/O + loop/anchor processing
│   ├── frontend_node.py
│   ├── processing/  # Sensor subscriptions, RGB-D processing
│   ├── loops/       # ICP registration, loop detection
│   └── anchors/     # Anchor management, descriptor building
├── backend/         # Backend SLAM inference + fusion
│   ├── backend_node.py
│   ├── fusion/      # Information-geometric fusion operators
│   └── parameters/  # Adaptive parameter models (NIG, birth, etc.)
├── common/          # Shared transforms/constants/op reports
│   └── transforms/  # SE(3) operations
└── utility_nodes/   # MVP utility nodes
    ├── tb3_odom_bridge.py    # Absolute → delta odometry
    ├── image_decompress.py   # Rosbag image decompression
    └── livox_converter.py     # Livox → PointCloud2

# Phase 2 (in phase2/): Gazebo sim, Dirichlet nodes, alternative launches
```

### Data Flow
```
Sensors → Frontend (association + ICP) → LoopFactor → Backend (fusion) → State/Trajectory
Ground Truth Odom → Odom Bridge (abs→delta) ──────────────────────────────┘
```

### Key Nodes (MVP)
- **frontend_node**: Data association via Fisher-Rao distances, ICP registration, anchor management
- **backend_node**: Bregman barycenters, loop closure fusion, state estimation
- **tb3_odom_bridge**: Absolute → delta odometry conversion (generic, not TB3-specific)
- **image_decompress**: Rosbag image decompression for RGB-D processing
- **livox_converter**: Livox LiDAR message conversion

**Phase 2 nodes** (Gazebo simulation, Dirichlet semantic SLAM) are in `phase2/`.

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

### Optional (Phase 2)
- Gazebo (for simulation) - see `phase2/`
- TurtleBot3 packages (for Gazebo demo) - see `phase2/`
- GPU acceleration (CUDA) - for 3D point cloud processing

### Install
```bash
pip install -r requirements.txt
```

---

## Testing

See **[docs/TESTING.md](docs/TESTING.md)** for complete testing documentation.

### MVP Validation (M3DGR)
```bash
bash tools/run_and_evaluate.sh
```

Runs the full M3DGR rosbag pipeline and produces metrics/plots under `results/`.

### Integration Tests
```bash
# MVP integration test
./tools/test-integration.sh

# Alternative datasets (Phase 2)
# See phase2/README.md for re-enabling Phase 2 features
```

See [`docs/TESTING.md`](docs/TESTING.md) for complete testing documentation.

---

## Key Topics

### Input
- `/scan` - LaserScan
- `/lidar/points` - PointCloud2 (optional / 3D mode)
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

## Contact

**William Habacivch**  
Email: whab13@mit.edu
