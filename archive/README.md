# Archive Directory

This directory contains archived files that are no longer actively used but are preserved for historical reference.

## Archived Documentation (`archive/docs/`)

The following documentation files have been archived as they are no longer relevant or have been superseded:

- **`Untitled`** - Draft file with temporary notes about trajectory errors and LiDAR tilt correction (content addressed in other docs)
- **`CODE_DIFF_GOOD_RUN_vs_CURRENT.md`** - Outdated code comparison snapshot from a specific "good run" commit (2026-01-26). Use git history for comparisons instead.
- **`CODE_DIFF_SUMMARY.md`** - Summary of the above code diff document (also outdated)
- **`YAW_DRIFT_ROOT_CAUSE_ANALYSIS.md`** - Analysis of yaw drift issues that were resolved (2026-01-27). Preserved for historical context.
- **`EVALUATION_IMPROVEMENTS.md`** - Recommendations document for evaluation enhancements. Most recommendations have been implemented in `tools/evaluate_slam.py`.
- **`Fusion_issues.md`** - Historical notes about sensor fusion status. All sensors (LiDAR + IMU + Odom) are now fused. Preserved for reference.
- **`TRACE_Z_EVIDENCE_AND_TRAJECTORY.md`** - M3DGR-era: where z in pose/trajectory comes from (odom weak z, LiDAR/map feedback). Planar fixes are in code; project switching to Kimera.
- **`RAW_MEASUREMENTS_VS_PIPELINE.md`** - M3DGR-era: what pipeline uses from odom/IMU vs raw message audit. Preserved for reference.
- **`TRACE_TRAJECTORY_AND_GROUND_TRUTH.md`** - M3DGR-era: trajectory export frame, GT (Dynamic01, body_T_wheel, camera_imu). Kimera uses different bags/frames.
- **`M3DGR_DYNAMIC01_ARCHIVE.md`** - M3DGR Dynamic01 is no longer the default dataset; default is Kimera. This doc records paths and `PROFILE=m3dgr` usage for running with Dynamic01 if the bag is present.

## Other Archived Items

- **`build_3d/`** - Obsolete build directory from pre-unified build system
- **`legacy_frontend/`** - Legacy frontend code that has been superseded

## Note

Files in this archive are not imported or referenced by the active codebase. They are kept for historical reference only.
