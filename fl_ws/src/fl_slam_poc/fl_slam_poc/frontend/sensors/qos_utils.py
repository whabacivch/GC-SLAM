"""
QoS utilities for frontend sensor subscriptions.

Purpose: keep QoS resolution consistent across SensorIO and utility nodes
without copy/paste drift.
"""

from __future__ import annotations

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def resolve_qos_profiles(reliability: str, depth: int) -> tuple[list[QoSProfile], list[str]]:
    """
    Resolve QoS profiles from a reliability string.

    Supported values:
      - reliable
      - best_effort
      - system_default
      - both (subscribe twice: RELIABLE + BEST_EFFORT)
    """
    reliability = str(reliability).lower()
    rel_map = {
        "reliable": ReliabilityPolicy.RELIABLE,
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
        "system_default": ReliabilityPolicy.SYSTEM_DEFAULT,
    }
    if reliability == "both":
        rels = [ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT]
        names = ["reliable", "best_effort"]
    elif reliability in rel_map:
        rels = [rel_map[reliability]]
        names = [reliability]
    else:
        rels = [ReliabilityPolicy.RELIABLE]
        names = ["reliable"]

    profiles = [
        QoSProfile(
            reliability=rel,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=int(depth),
        )
        for rel in rels
    ]
    return profiles, names

