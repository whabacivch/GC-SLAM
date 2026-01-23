"""
Frontend-side OpReport publishing helper.

Goal: keep OpReport validation + JSON publishing consistent across frontend nodes
without copy/paste drift.
"""

from __future__ import annotations

from std_msgs.msg import String

from fl_slam_poc.common.op_report import OpReport


def publish_op_report(node, pub_report, report: OpReport) -> None:
    report.validate()
    msg = String()
    msg.data = report.to_json()
    pub_report.publish(msg)

