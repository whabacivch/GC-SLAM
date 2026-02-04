# Rerun API Reference (rerun-sdk ≥ 0.15)

This doc summarizes the Rerun APIs used by our post-run builder (`tools/build_rerun_from_splat.py`) and Option 1 (multiple archetypes + blueprint). Our constraint: **`rerun-sdk>=0.15`** (see `requirements.txt`).

---

## 1. Archetypes (logging)

### Points3D

- **Required:** `positions` — (N, 3) float32 or array-like.
- **Recommended:** `radii`, `colors`.
- **Optional:** `labels`, `class_ids`, `keypoint_ids`, etc.

**Colors:** Rerun accepts **Rgba32**: either uint8 0–255 (e.g. `(N, 3)` or `(N, 4)`) or float 0–1. Examples from docs: `colors=rng.uniform(0, 255, size=[10, 3])`, or `colors=[0, 0, 255]` (single RGB), or RGBA float `[[1.0-c, c, 0.5, 0.5] for c in ...]`.

**Example:**
```python
rr.log("gc/map/splats/colored", rr.Points3D(
    positions,           # (N, 3) float32
    colors=colors_uint8, # (N, 3) uint8 0-255
    radii=radii,         # (N,) float32 — world-space radii
))
```

**Radii:** By default interpreted as **world-space units**. For “dense” splat look use e.g. `radii = np.linalg.norm(half_sizes, axis=1) * scale_factor` (or mean/max of half_axes). Optional: `rr.Radius.ui_points([...])` for UI-point radii.

---

### Arrows3D

- **Required:** `vectors` — (N, 3) direction + length (endpoint = origin + vector).
- **Recommended:** `origins` — (N, 3). If omitted, origin is (0,0,0) for all.
- **Optional:** `colors`, `radii`, `labels`, etc.

**Example (from Rerun docs):**
```python
rr.log("arrows", rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))
```

**For normals:** `origins = positions`, `vectors = directions * length[:, None]` with `length` e.g. from kappa: `np.clip(kappas / np.percentile(kappas, 95), 0.1, 2.0)`. Colors: RGBA float 0–1 or uint8; e.g. `colors=[255, 200, 0, 200]` (orange, semi-transparent) for a single color for all arrows.

---

### Ellipsoids3D

- **Required:** `half_sizes` — (N, 3) half-extents per axis.
- **Recommended:** `centers`, `colors`.
- **Optional:** `quaternions` (xyzw), `rotation_axis_angles`, `line_radii`, `fill_mode`, etc.

Our `build_rerun_from_splat.py` already uses: `centers=positions`, `half_sizes=half_sizes`, `quaternions=quats`, `colors=colors_uint8`. No change needed for Option 1 unless we move the entity path (e.g. under `gc/map/splats/ellipsoids`).

---

## 2. Time

- **Set time:** `rr.set_time("time", timestamp=time_sec)` (or `duration=...`, `sequence=...` depending on timeline type).
- Our script uses a single `"time"` timeline; when we have `timestamps` we log per-primitive with `rr.set_time("time", timestamp=float(timestamps[i]))` before each log. For Option 1, logging all four archetypes once at `t=0` is simplest; for time-aligned playback, set the same time before each of the four `rr.log` calls in the loop.

---

## 3. Blueprint (programmatic layout, 0.15+)

- **Module:** `import rerun.blueprint as rrb`
- **Send:** `rr.send_blueprint(blueprint)` — applies to the current app (same `application_id` as `rr.init(...)`). Our script uses `rr.init("fl_slam_poc", ...)`, so the blueprint applies to `"fl_slam_poc"`.

### Containers

- **Tabs:** `rrb.Tabs(*args, contents=None, active_tab=None, name=None)` — each positional arg is a `SpaceView` or `Container`.
- **Horizontal:** `rrb.Horizontal(*args, contents=None, column_shares=None, name=None)`
- **Vertical:** `rrb.Vertical(*args, contents=None, row_shares=None, name=None)`
- **Grid:** `rrb.Grid(*args, contents=None, column_shares=None, row_shares=None, grid_columns=None, name=None)`

### Space views

- **Spatial3DView:** `rrb.Spatial3DView(origin='/', contents='$origin/**', name=None)`
  - `origin`: entity path that defines the 3D view’s origin (e.g. `"gc/map/splats/colored"` to focus on that subtree).
  - `name`: display name in the UI (tab title, etc.).
  - `contents`: optional query expression; default `'$origin/**'` includes the origin and children.

### Panels

- **SelectionPanel:** `rrb.SelectionPanel(expanded=None)`
- **TimePanel:** `rrb.TimePanel(expanded=None)`
- **BlueprintPanel:** `rrb.BlueprintPanel(expanded=None)`

### Top-level Blueprint

- **Blueprint:** `rrb.Blueprint(*parts, auto_layout=None, auto_space_views=None, collapse_panels=False)`
  - `*parts`: mix of containers, space views, and panels (only one of each panel type).
  - If you pass multiple `ContainerLike` (e.g. several `Spatial3DView`s), they are wrapped in a single root **Tab** container.

**Example for Option 1 (tabs per view + panels):**
```python
import rerun.blueprint as rrb

blueprint = rrb.Blueprint(
    rrb.Tabs(
        rrb.Spatial3DView(name="Dense Color", origin="gc/map/splats/colored"),
        rrb.Spatial3DView(name="Uncertainty Ellipsoids", origin="gc/map/splats/ellipsoids"),
        rrb.Spatial3DView(name="Normals (vMF)", origin="gc/map/splats/normals"),
        rrb.Spatial3DView(name="Weights", origin="gc/map/splats/weights"),
        rrb.Spatial3DView(name="Trajectory", origin="gc/trajectory"),
        name="GC Map Views",
    ),
    rrb.SelectionPanel(),
    rrb.TimePanel(),
    collapse_panels=False,
)
rr.send_blueprint(blueprint)
```

**Note:** Entity paths in the blueprint (`origin=...`) must match the paths used in `rr.log(...)`. So we should log Option 1 data under e.g. `gc/map/splats/colored`, `gc/map/splats/ellipsoids`, `gc/map/splats/normals`, `gc/map/splats/weights`; trajectory stays `gc/trajectory`. Our current ellipsoids are logged at `gc/map/ellipsoids` (or per-entity under `gc/map/ellipsoids/<id>`); we can either keep that path and use `origin="gc/map/ellipsoids"` in the blueprint, or add a second log under `gc/map/splats/ellipsoids` if we want a dedicated tab name.

---

## 4. Weight → color (scalar colormap)

Rerun does not expose a built-in “apply colormap to scalar” for Points3D in the Python API; we must compute RGB ourselves. Options:

- **Matplotlib:** `import matplotlib.cm as cm; cm.get_cmap('viridis')(norm(weights))` → (N, 4) RGBA float; then convert to uint8 if needed.
- **Simple LUT:** Precompute a 256-entry viridis-like RGB LUT, then `idx = np.clip((weights - w_min) / (w_max - w_min + 1e-12) * 255, 0, 255).astype(np.int32)` and index the LUT.

Our script already has no matplotlib dependency for the main path; adding a small numpy-only LUT (e.g. 256×3) keeps it that way.

---

## 5. References

- Points3D: https://www.rerun.io/docs/reference/types/archetypes/points3d  
- Arrows3D: https://www.rerun.io/docs/reference/types/archetypes/arrows3d  
- Ellipsoids3D: https://www.rerun.io/docs/reference/types/archetypes/ellipsoids3d  
- Python archetypes: https://ref.rerun.io/docs/python/stable/common/archetypes  
- Blueprints: https://www.rerun.io/docs/concepts/blueprints  
- Blueprint APIs (0.15): https://ref.rerun.io/docs/python/0.15.0/common/blueprint_apis/
