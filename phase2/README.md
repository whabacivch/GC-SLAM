# Phase 2 (Non-MVP)

This folder contains code and assets intentionally removed from the **MVP “smallest reproducible failing case”** workflow.

**Goal:** keep the active MVP surface area minimal and prevent accidental dependencies on future/experimental features.

## What belongs here

- Experimental nodes (Dirichlet / semantics)
- Gazebo-only simulation utilities
- Alternative dataset launch files (TB3, generic 3D, etc.)
- Future fusion modules not needed for the M3DGR MVP
- Non-MVP tests that exercise Phase 2 features
- Wrapper scripts for Phase 2 nodes (`scripts/`)

## How to re-enable Phase 2

Move files back into `fl_ws/src/fl_slam_poc/` (or selectively cherry-pick modules) and update:
- `fl_ws/src/fl_slam_poc/setup.py` (entry points + installed launch files)
- `fl_ws/src/fl_slam_poc/CMakeLists.txt` (wrapper scripts in `install(PROGRAMS ...)`)
- Documentation / roadmap pointers

**Scripts location**: Phase 2 wrapper scripts are in `phase2/fl_ws/src/fl_slam_poc/scripts/`:
- `sim_world`, `sim_world_node` - Gazebo simulation world
- `dirichlet_backend_node` - Dirichlet semantic SLAM backend
- `sim_semantics_node` - Semantic category simulation
