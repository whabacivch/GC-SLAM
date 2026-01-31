# Pipeline Depth Contract

Single contract for depth in the Golden Child visual–LiDAR pipeline.

## Contract

**camera_depth is the authoritative depth sensor (RGB-D).** LiDAR is fused into it via `lidar_depth_evidence` in `splat_prep_fused` (Product-of-Experts: Λf = Λc + Λ_ell, θf = θc + θ_ell).

- **No separate use** of raw camera_depth for a different likelihood.
- **No double-counting**: one fused depth path only (camera Λc, θc from feature meta + LiDAR Λ_ell, θ_ell from `lidar_depth_evidence`).

## Code alignment

- **Frontend**: `splat_prep_fused` (frontend/sensors/splat_prep.py) uses camera depth from feature meta (`depth_Lambda_c`, `depth_theta_c`) and `lidar_depth_evidence(u, v, ...)` from `lidar_camera_depth_fusion.py`; fused depth is the single 3D lift for splats.
- **Backend**: Camera batch is built from RGB + depth; no second path consumes camera_depth for a separate term. Primitives (surfels + camera splats) are the single measurement representation.

## Reference

- Pipeline and data flow: `docs/IMU_BELIEF_MAP_AND_FUSION.md`
- Golden Child spec: `docs/GOLDEN_CHILD_INTERFACE_SPEC.md`
