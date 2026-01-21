# Phase 2 (Non-MVP)

This folder contains code and assets intentionally removed from the **MVP “smallest reproducible failing case”** workflow.

**Goal:** keep the active MVP surface area minimal and prevent accidental dependencies on future/experimental features.

## What belongs here

- Experimental nodes (Dirichlet / semantics)
- Gazebo-only simulation utilities
- Alternative dataset launch files (TB3, generic 3D, etc.)
- Future fusion modules not needed for the M3DGR MVP
- Non-MVP tests that exercise Phase 2 features

## How to re-enable Phase 2

Move files back into `fl_ws/src/fl_slam_poc/` (or selectively cherry-pick modules) and update:
- `fl_ws/src/fl_slam_poc/setup.py` (entry points + installed launch files)
- Documentation / roadmap pointers

