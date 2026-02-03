"""
Data structures for Geometric Compositional SLAM v2.

PrimitiveMap and MeasurementBatch are the canonical map and measurement types.
IW states (process/measurement noise) live in operators/structures as needed.

Phase 2: AtlasMap and PrimitiveMapTile provide tiling infrastructure.
"""

from fl_slam_poc.backend.structures.primitive_map import (
    # Legacy interface (wraps single tile)
    PrimitiveMap,
    PrimitiveMapView,
    RenderablePrimitiveBatch,
    create_empty_primitive_map,
    extract_primitive_map_view,
    renderable_batch_from_view,
    primitive_map_fuse,
    primitive_map_insert,
    primitive_map_cull,
    primitive_map_forget,
    primitive_map_merge_reduce,
    # Phase 2: Tiling infrastructure
    PrimitiveMapTile,
    AtlasMap,
    create_empty_tile,
    create_empty_atlas_map,
    atlas_to_primitive_map,
    primitive_map_to_atlas,
)
from fl_slam_poc.backend.structures.measurement_batch import (
    MeasurementBatch,
    create_empty_measurement_batch,
)
from fl_slam_poc.backend.structures.inverse_wishart_jax import (
    ProcessNoiseIWState,
    create_datasheet_process_noise_state,
)
from fl_slam_poc.backend.structures.measurement_noise_iw_jax import (
    MeasurementNoiseIWState,
    create_datasheet_measurement_noise_state,
)

__all__ = [
    # Legacy interface
    "PrimitiveMap",
    "PrimitiveMapView",
    "RenderablePrimitiveBatch",
    "create_empty_primitive_map",
    "extract_primitive_map_view",
    "renderable_batch_from_view",
    "primitive_map_fuse",
    "primitive_map_insert",
    "primitive_map_cull",
    "primitive_map_forget",
    "primitive_map_merge_reduce",
    # Phase 2: Tiling infrastructure
    "PrimitiveMapTile",
    "AtlasMap",
    "create_empty_tile",
    "create_empty_atlas_map",
    "atlas_to_primitive_map",
    "primitive_map_to_atlas",
    # Other structures
    "MeasurementBatch",
    "create_empty_measurement_batch",
    "ProcessNoiseIWState",
    "create_datasheet_process_noise_state",
    "MeasurementNoiseIWState",
    "create_datasheet_measurement_noise_state",
]
