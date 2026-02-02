# PointCloud2 Field Layouts (GC v2)

This document is the **single source of truth** for PointCloud2 field layouts supported by the GC v2 backend. The backend selects the parser by **explicit config** (`pointcloud_layout: vlp16` for Kimera). No silent fallback; fail-fast if layout does not match message fields.

**Reference:** [BAG_TOPICS_AND_USAGE.md](BAG_TOPICS_AND_USAGE.md) for topic and frame conventions.

---

## VLP-16 (Kimera)

**Used for:** Kimera acl_jackal (`/acl_jackal/lidar_points`, `sensor_msgs/PointCloud2`). See [VELODYNE_VLP16.md](VELODYNE_VLP16.md).

---

## Livox (archived)

**Used for:** Archived M3DGR Dynamic01 (Livox MID-360); CustomMsg was converted to PointCloud2 by `livox_converter`.

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `x` | float | X coordinate (m) |
| `y` | float | Y coordinate (m) |
| `z` | float | Z coordinate (m) |
| `ring` | uint8 | Ring/line id |
| `tag` | uint8 | Tag class |
| `timebase_low` | uint32 | Timebase low 32 bits (ns) |
| `timebase_high` | uint32 | Timebase high 32 bits (ns) |
| `time_offset` | uint32 | Per-point time offset (ns) relative to timebase |

**Optional (used when present):** `intensity`.

**Per-point timestamp:** `timebase_sec + time_offset * 1e-9` (timebase from low/high 64-bit ns).

**Parser:** `parse_pointcloud2_vectorized` in `fl_slam_poc/backend/backend_node.py`. Fail-fast if any required field is missing.

---

## VLP-16 (Kimera / Velodyne Puck)

**Used for:** Kimera-Multi-Data 10_14 (Velodyne VLP-16); topic `/acl_jackal/lidar_points`, frame `acl_jackal2/velodyne_link`.

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `x` | float | X coordinate (m) |
| `y` | float | Y coordinate (m) |
| `z` | float | Z coordinate (m) |
| `ring` | uint8 | Ring/laser id (0–15 for VLP-16) |

**Optional fields:**

| Field | Type | Description |
|-------|------|-------------|
| `intensity` | float / uint8 | Reflectivity |
| `t` or `time` | float | Per-point timestamp (units: seconds or nanoseconds; driver-dependent) |

**Per-point timestamp:** If `t` or `time` is present, use it (convert to seconds if in ns). Otherwise use `header.stamp` for entire scan (no per-point deskew).

**Pipeline contract:** VLP-16 parser outputs the same shape as Livox parser: `(points, timestamps, weights, ring, tag)`. For VLP-16: `tag=0` (unused); `timebase` = `header.stamp`; `time_offset` derived from per-point `t`/`time` if present, else 0.

**Parser:** `parse_pointcloud2_vlp16` (select when `pointcloud_layout: vlp16`). Fail-fast if `x`, `y`, `z`, or `ring` is missing.

---

## Config

- **Backend parameter:** `pointcloud_layout` (string): `livox` | `vlp16`.
- **YAML (gc_unified.yaml / gc_kimera.yaml):** `pointcloud_layout: livox` for M3DGR; `pointcloud_layout: vlp16` for Kimera.
- **Selection:** In `on_lidar`, backend reads `pointcloud_layout` and calls the corresponding parser. If message fields do not match the selected layout, raise immediately (no fallback).
