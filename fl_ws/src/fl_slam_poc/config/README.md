# Geometric Compositional SLAM v2 configuration

**Single source of truth: `gc_unified.yaml`**

Launch loads `gc_backend.ros__parameters` from this file. No overrides in launch file or run scripts.

- **gc_unified.yaml** — Canonical config (hub + backend). Contains extrinsics from Kimera calibration.
- **calibration/** — Reference extrinsics from Kimera_Data/calibration (6D format).
- **gc_backend.yaml** — Deprecated. Prefer gc_unified.yaml.
- **gc_dead_end_audit.yaml** — Audit node config.
