"""
Shared rosbag2 SQLite helpers for tools scripts.

Goal: eliminate copy/paste drift across `tools/*.py` while preserving the
current scripts' behavior (notably: `resolve_db3_path()` returns "" on failure).

This module intentionally lives in `tools/` (not the ROS package) so scripts
executed as `python3 tools/<script>.py` can import it without installation.
"""

from __future__ import annotations

import os
from typing import Optional


def resolve_db3_path(bag_path: str) -> str:
    """
    Resolve a rosbag2 SQLite .db3 file from either:
      - a direct .db3 path
      - a directory containing one or more *.db3 files (returns first in sorted order)

    Returns:
      - resolved .db3 path, or "" if not found (preserves existing tool behavior).
    """
    if os.path.isfile(bag_path) and bag_path.endswith(".db3"):
        return bag_path
    if not os.path.isdir(bag_path):
        return ""
    for name in sorted(os.listdir(bag_path)):
        if name.endswith(".db3"):
            return os.path.join(bag_path, name)
    return ""


def topic_id(cur, name: str) -> Optional[int]:
    cur.execute("SELECT id FROM topics WHERE name = ? LIMIT 1", (name,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def topic_type(cur, name: str) -> Optional[str]:
    cur.execute("SELECT type FROM topics WHERE name = ? LIMIT 1", (name,))
    row = cur.fetchone()
    return row[0] if row else None

