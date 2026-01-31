# FL-SLAM Evaluation Guide

## Overview

FL-SLAM includes quantitative validation against ground truth using standard SLAM metrics:
- **ATE (Absolute Trajectory Error)**: Global consistency
- **RPE (Relative Pose Error)**: Local drift per meter

## Alignment: Same Coordinate System as the Estimate

SLAM is **relative**: the robot arbitrarily says “start here (0,0,0)” and estimates its path from there. There is no global reference. To evaluate meaningfully, **ground truth must be in the same coordinate system as the estimate**.

We **align GT to the estimate frame** (initial-pose alignment): at the first associated pose, we transform GT so that GT = estimate. Both trajectories then start at (0,0,0). This is like **zeroing a scale** — we normalize away the initial offset so we can measure how good our estimates are vs external truth over the trajectory.

- **Evaluation** (`evaluate_slam.py`, default `--align initial`): Full SE(3) at first pose; GT is transformed into the estimate frame. ATE/RPE are computed in that single frame.
- **Dashboard**: GT is plotted in the estimate frame (translation so both start at 0,0,0); the per-scan offset table is in the same frame.

### Initial offset and frame conventions

Any **pre-alignment** initial offset (how far apart the two trajectories were before we aligned them) is **not** reported as part of ATE/RPE — we align it away. Conceptually:

- **Large pre-alignment offset** could indicate **extrinsics or calibration** issues (e.g. wrong body_T_wheel, wrong LiDAR/IMU frame). If you ever want to investigate that, you could log or report the alignment transform (translation/rotation applied to GT) before computing metrics.
- For now we **document** the approach: we align GT to the estimate frame so both start at 0,0,0; we do not treat initial offset as a “metric” because half of it is often **frame convention** (e.g. world vs body, axis flip, different origin) and not necessarily sensor failure. For M3DGR wheel vs body frame and GT semantics see `archive/docs/TRACE_TRAJECTORY_AND_GROUND_TRUTH.md`. Kimera bags use different topics/frames (see `rosbags/Kimera_Data/`).

## Running Evaluation

### Full Pipeline (SLAM + Evaluation)

```bash
cd "Phantom Fellowship MIT/Impact Project_v1"
bash tools/run_and_evaluate.sh
```

This will:
1. Run SLAM on M3DGR Dynamic01
2. Export estimated trajectory to TUM format
3. Align ground truth timestamps
4. Compute ATE/RPE metrics
5. Generate comparison plots

Results are saved to `results/m3dgr_YYYYMMDD_HHMMSS/`.

### Quick Run (No Evaluation)

For rapid iteration without evaluation:

```bash
ros2 launch fl_slam_poc poc_m3dgr_rosbag.launch.py \
  bag:=rosbags/m3dgr/Dynamic01_ros2 \
  play_bag:=true
```

### Manual Evaluation

If you already have trajectory files:

```bash
# Align timestamps (use venv Python)
.venv/bin/python tools/align_ground_truth.py \
  rosbags/m3dgr/Dynamic01.txt \
  /tmp/fl_slam_trajectory.tum \
  /tmp/ground_truth_aligned.tum

# Compute metrics and plots (use venv Python)
.venv/bin/python tools/evaluate_slam.py \
  /tmp/ground_truth_aligned.tum \
  /tmp/fl_slam_trajectory.tum \
  results/my_evaluation
```

## Output Files

Each evaluation run creates:
- `trajectory_comparison.png` - XY, XZ, YZ, 3D trajectory plots
- `error_analysis.png` - Error over time + histogram
- `metrics.txt` - Quantitative metrics (RMSE, mean, median, etc.)
- `estimated_trajectory.tum` - SLAM output in TUM format
- `ground_truth_aligned.tum` - Aligned ground truth

## Metrics Explanation

### Absolute Trajectory Error (ATE)
- **What it measures**: Global drift from ground truth (in the **estimate frame** after alignment).
- **Good performance**: RMSE < 0.5m for indoor sequences
- **Computation**: GT is aligned to the estimate frame (initial-pose alignment by default); then translation (and rotation) error at each pose. Use `--align umeyama` for full SE(3) best-fit (not valid for relative SLAM).

### Relative Pose Error (RPE)
- **What it measures**: Local consistency (drift per meter) in the same frame.
- **Good performance**: RMSE < 0.05 m/m (5% drift)
- **Computation**: Relative motion errors over 1m (and 5m, 10m) segments after the same alignment.

## Dependencies

- `evo` - SLAM evaluation library (installed automatically)
- `matplotlib` - Plotting (usually pre-installed)
- `numpy`, `scipy` - Math libraries (already required by FL-SLAM)

## Troubleshooting

### "Estimated trajectory not found"

The trajectory file is written to `/tmp/fl_slam_trajectory.tum` during SLAM execution. If missing:
- Check that the backend node ran successfully
- Verify the `trajectory_export_path` parameter in the launch file
- Check ROS2 logs for backend errors

### "Empty trajectory file"

- Ensure the SLAM system ran for enough time to generate poses
- Check that loop closures were detected (look for "Loop factor" messages)
- Verify odometry input is being received

### Alignment Issues

The alignment script expects TUM format:
```
timestamp x y z qx qy qz qw
```

If you see alignment errors:
- Check that both files are in TUM format
- Verify timestamps are numeric (not strings)
- Ensure quaternions are normalized

### "Our XZ looks like GT XY" (axis-convention mismatch)

If the **trajectory comparison plot** shows that the **estimated XZ view** (our X vs our Z) resembles the **ground-truth XY view** (GT X vs GT Y) in shape or extent, that suggests a **Y/Z axis convention mismatch** between the estimate frame and the GT frame (e.g. our Z axis corresponds to GT’s Y, or vice versa).

- **Diagnostic:** Re-run evaluation with **`--gt-swap-yz`** so GT’s Y and Z axes are swapped before alignment and metrics. If the trajectories then overlay much better, the GT (or its frame definition) likely uses a different Y/Z convention.
- **Example:**  
  `tools/evaluate_slam.py .../ground_truth_aligned.tum .../estimated_trajectory.tum results/out op_report.jsonl --gt-swap-yz`
- **Root cause:** M3DGR GT is in body (camera_imu) frame; we compare after transforming our estimate to body via `body_T_wheel`. Residual axis difference (e.g. body vs wheel Y/Z) or GT file column/axis semantics can produce this. See `archive/docs/TRACE_TRAJECTORY_AND_GROUND_TRUTH.md` (M3DGR). For Kimera see `rosbags/Kimera_Data/` and tools (e.g. `run_gc_kimera.sh`, `kimera_gt_to_tum.py`).

**Sanity check and convention fix (Umeyama-derived):** The evaluator prints **ptp(x,y,z)** for GT and EST before plotting. For a planar robot, **GT ptp(z) should be ~0.04 m**; if the plot shows GT "Z" swinging by meters, axes are swapped. Use **`--verbose`** to print the first 5 rows of positions (TUM: t, x, y, z, qx, qy, qz, qw). Umeyama (EST → GT) yields a signed permutation: GT x ≈ −EST z, GT y ≈ EST x, GT z ≈ −EST y. Use **`--apply-est-to-gt-convention`** to transform EST by `R_EST_TO_GT` so both trajectories are in GT axis convention and 2D projections are same-plane.

## Validation Strategy

FL-SLAM validation follows standard SLAM evaluation practices:

1. **Quantitative Metrics**: ATE/RPE for objective comparison
2. **Visual Inspection**: Trajectory plots for qualitative assessment
3. **Error Distribution**: Histogram to identify systematic vs random errors
4. **Temporal Analysis**: Error over time to detect drift patterns

This approach is used by:
- ORB-SLAM2/3
- VINS-Mono/Fusion
- Academic SLAM benchmarks (EuRoC, TUM RGB-D, etc.)
