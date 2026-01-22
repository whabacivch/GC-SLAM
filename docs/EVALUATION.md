# FL-SLAM Evaluation Guide

## Overview

FL-SLAM includes quantitative validation against ground truth using standard SLAM metrics:
- **ATE (Absolute Trajectory Error)**: Global consistency
- **RPE (Relative Pose Error)**: Local drift per meter

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
# Align timestamps
python3 tools/align_ground_truth.py \
  rosbags/m3dgr/Dynamic01.txt \
  /tmp/fl_slam_trajectory.tum \
  /tmp/ground_truth_aligned.tum

# Compute metrics and plots
python3 tools/evaluate_slam.py \
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
- **What it measures**: Global drift from ground truth
- **Good performance**: RMSE < 0.5m for indoor sequences
- **Computation**: SE(3) alignment, then translation error at each pose

### Relative Pose Error (RPE)
- **What it measures**: Local consistency (drift per meter)
- **Good performance**: RMSE < 0.05 m/m (5% drift)
- **Computation**: Relative motion errors over 1m segments

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
