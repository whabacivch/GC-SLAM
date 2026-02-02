"""
Frontend audit/observability nodes for Geometric Compositional.

These nodes must not perform inference math; they provide accountability over
what data streams exist, which ones are consumed, and which are intentionally
unused (dead ends) for future expansion.

Nodes:
    - dead_end_audit_node: Subscribes to unused topics for accountability
    - wiring_auditor: Collects status and produces end-of-run summary
"""

from fl_slam_poc.frontend.audit.dead_end_audit_node import DeadEndAuditNode
from fl_slam_poc.frontend.audit.wiring_auditor import WiringAuditorNode

__all__ = [
    "DeadEndAuditNode",
    "WiringAuditorNode",
]
