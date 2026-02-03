"""
Best-effort runtime counters for per-scan DeviceRuntimeCert population.

These counters are intentionally lightweight and are reset once per scan
by the backend node. They are used to estimate host/device transfers and
host syncs without introducing control-flow gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np


@dataclass
class RuntimeCounters:
    host_sync_count_est: int = 0
    device_to_host_bytes_est: int = 0
    host_to_device_bytes_est: int = 0
    jit_recompile_count: int = 0


_lock = threading.Lock()
_counters = RuntimeCounters()


def _estimate_nbytes(obj: Any) -> int:
    if obj is None:
        return 0
    if hasattr(obj, "nbytes"):
        try:
            return int(obj.nbytes)
        except Exception:
            return 0
    if hasattr(obj, "size") and hasattr(obj, "dtype"):
        try:
            return int(obj.size * obj.dtype.itemsize)
        except Exception:
            return 0
    try:
        arr = np.asarray(obj)
        return int(arr.nbytes)
    except Exception:
        return 0


def reset_runtime_counters() -> None:
    """Reset counters at the start of a scan."""
    global _counters
    with _lock:
        _counters = RuntimeCounters()


def record_host_to_device(obj: Any = None, *, nbytes: int | None = None) -> None:
    """Record an estimated host->device transfer (bytes)."""
    bytes_est = int(nbytes) if nbytes is not None else _estimate_nbytes(obj)
    if bytes_est <= 0:
        return
    with _lock:
        _counters.host_to_device_bytes_est += bytes_est


def record_device_to_host(obj: Any = None, *, nbytes: int | None = None, syncs: int = 1) -> None:
    """Record an estimated device->host transfer and host sync."""
    bytes_est = int(nbytes) if nbytes is not None else _estimate_nbytes(obj)
    with _lock:
        _counters.device_to_host_bytes_est += max(0, bytes_est)
        _counters.host_sync_count_est += max(0, int(syncs))


def record_host_sync(syncs: int = 1) -> None:
    """Record a host sync without a transfer size estimate."""
    with _lock:
        _counters.host_sync_count_est += max(0, int(syncs))


def record_jit_recompile(count: int = 1) -> None:
    """Record an estimated JIT recompilation event."""
    with _lock:
        _counters.jit_recompile_count += max(0, int(count))


def snapshot_runtime_counters() -> RuntimeCounters:
    """Return a snapshot of the current counters (no reset)."""
    with _lock:
        return RuntimeCounters(
            host_sync_count_est=_counters.host_sync_count_est,
            device_to_host_bytes_est=_counters.device_to_host_bytes_est,
            host_to_device_bytes_est=_counters.host_to_device_bytes_est,
            jit_recompile_count=_counters.jit_recompile_count,
        )


def consume_runtime_counters() -> RuntimeCounters:
    """Return counters and reset (single-scan consumption)."""
    global _counters
    with _lock:
        snapshot = RuntimeCounters(
            host_sync_count_est=_counters.host_sync_count_est,
            device_to_host_bytes_est=_counters.device_to_host_bytes_est,
            host_to_device_bytes_est=_counters.host_to_device_bytes_est,
            jit_recompile_count=_counters.jit_recompile_count,
        )
        _counters = RuntimeCounters()
    return snapshot
