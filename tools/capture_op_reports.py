#!/usr/bin/env python3
"""
Capture OpReport messages to a JSONL file reliably.

Why this exists:
- `ros2 topic echo` can emit startup warnings to stdout and/or exit early when the
  topic isn't published yet, leaving `op_report.jsonl` out-of-sync.
- This subscriber waits for real messages and writes *only* `msg.data` (one JSON
  object per line), which `tools/evaluate_slam.py` expects.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class _OpReportRecorder(Node):
    def __init__(self, topic: str, out_path: Path):
        super().__init__("op_report_recorder")
        self._topic = topic
        self._out_path = out_path
        self._fh = out_path.open("w", encoding="utf-8")
        self._count = 0
        self._last_log = 0.0

        self.create_subscription(String, topic, self._on_msg, 10)
        self.get_logger().info(f"Recording OpReports from {topic} -> {out_path}")

    def _on_msg(self, msg: String) -> None:
        # We assume msg.data is already a JSON object string.
        line = (msg.data or "").strip()
        if not line:
            return

        self._fh.write(line + "\n")
        self._fh.flush()
        self._count += 1

        # Periodic progress without spamming.
        now = time.time()
        if now - self._last_log > 5.0:
            self._last_log = now
            self.get_logger().info(f"Captured {self._count} OpReports so far...")

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/cdwm/op_report")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = _OpReportRecorder(args.topic, out_path)

    def _shutdown(*_a):
        try:
            node.get_logger().info("Shutting down OpReport recorder...")
        except Exception:
            pass
        try:
            node.close()
        finally:
            rclpy.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    finally:
        try:
            node.close()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

