"""
Data structures for Geometric Compositional SLAM v2.

PrimitiveMap and MeasurementBatch are the canonical map and measurement types.
IW states (process/measurement noise) live in operators/structures as needed.
"""

from fl_slam_poc.backend.structures.primitive_map import (
    PrimitiveMap,
    PrimitiveMapView,
    create_empty_primitive_map,
    extract_primitive_map_view,
    primitive_map_fuse,
    primitive_map_insert,
    primitive_map_cull,
    primitive_map_forget,
    primitive_map_merge_reduce,
)
from fl_slam_poc.backend.structures.measurement_batch import (
    MeasurementBatch,
    create_empty_measurement_batch,
)

__all__ = [
    "PrimitiveMap",
    "PrimitiveMapView",
    "create_empty_primitive_map",
    "extract_primitive_map_view",
    "primitive_map_fuse",
    "primitive_map_insert",
    "primitive_map_cull",
    "primitive_map_forget",
    "primitive_map_merge_reduce",
    "MeasurementBatch",
    "create_empty_measurement_batch",
]
