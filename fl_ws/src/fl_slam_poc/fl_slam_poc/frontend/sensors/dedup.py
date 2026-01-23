"""
Duplicate message suppression helpers.

Used when subscribing with multiple QoS profiles (e.g., RELIABLE + BEST_EFFORT)
to avoid processing the same message twice.
"""

from __future__ import annotations

from typing import Optional


def stamp_key(stamp, frame_id: str = "") -> Optional[tuple[int, int, str]]:
    if stamp is None:
        return None
    return (int(stamp.sec), int(stamp.nanosec), str(frame_id or ""))


def is_duplicate(last_keys: dict, key: str, stamp, frame_id: str = "") -> bool:
    k = stamp_key(stamp, frame_id=frame_id)
    if k is None:
        return False
    if last_keys.get(key) == k:
        return True
    last_keys[key] = k
    return False

