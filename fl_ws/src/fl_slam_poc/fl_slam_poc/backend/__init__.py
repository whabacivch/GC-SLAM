"""
Golden Child SLAM v2 Backend.

Branch-free compositional inference backend per docs/GOLDEN_CHILD_INTERFACE_SPEC.md.

Structure:
- operators/: Branch-free operators (predict, fuse, recompose, etc.)
- structures/: Data structures (BinAtlas, UTCache)
- pipeline.py: Main pipeline functions
- backend_node.py: ROS2 node entry point

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    "PipelineConfig",
    "RuntimeManifest",
]


def __getattr__(name):
    if name == "PipelineConfig":
        from fl_slam_poc.backend.pipeline import PipelineConfig
        return PipelineConfig
    elif name == "RuntimeManifest":
        from fl_slam_poc.backend.pipeline import RuntimeManifest
        return RuntimeManifest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
