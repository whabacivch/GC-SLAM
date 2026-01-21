"""
Common package for FL-SLAM.

Shared utilities and transforms used by both frontend and backend.

Subpackages:
- transforms/: SE(3) geometry operations
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "OpReport",
    "constants",
]

_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "OpReport": ("fl_slam_poc.common.op_report", "OpReport"),
    # Expose these as submodules, but do not eagerly import them at package import time.
    "constants": ("fl_slam_poc.common.constants", None),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    return module if attr_name is None else getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))
