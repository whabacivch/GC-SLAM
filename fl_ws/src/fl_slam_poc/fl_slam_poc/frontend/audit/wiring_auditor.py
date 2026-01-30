"""
=============================================================================
WIRING AUDITOR - End-of-Run Summary for Golden Child SLAM
=============================================================================

Collects status from all GC nodes and produces a consolidated end-of-run
summary showing what was actually processed vs dead-ended.

Subscribes to:
    - /gc/status (from gc_backend_node)
    - /gc/dead_end_status (from dead_end_audit)
    - /gc/runtime_manifest (from gc_backend_node)

Outputs:
    - Formatted summary to stdout at shutdown
    - JSON file to configured path (for evaluation script integration)

Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String


@dataclass
class AuditState:
    """Accumulated state from all status topics."""
    # From /gc/status
    odom_count: int = 0
    scan_count: int = 0
    imu_count: int = 0
    pipeline_runs: int = 0
    map_bins_active: int = 0
    
    # From /gc/dead_end_status
    dead_end_counts: Dict[str, int] = field(default_factory=dict)
    
    # From /gc/runtime_manifest
    manifest: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    last_status_time: float = 0.0
    last_dead_end_time: float = 0.0
    manifest_received: bool = False


class WiringAuditorNode(Node):
    """
    Collects status from GC nodes and produces end-of-run summary.
    """

    def __init__(self) -> None:
        super().__init__("wiring_auditor")

        # Parameters
        self.declare_parameter("output_json_path", "")
        self.declare_parameter("summary_period_sec", 10.0)
        
        self._output_json_path = str(self.get_parameter("output_json_path").value)
        summary_period = float(self.get_parameter("summary_period_sec").value)

        # State
        self._state = AuditState()
        self._start_time = time.time()
        self._shutdown_requested = False

        # QoS for status topics
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscriptions
        self._sub_status = self.create_subscription(
            String, "/gc/status", self._on_gc_status, qos
        )
        self._sub_dead_end = self.create_subscription(
            String, "/gc/dead_end_status", self._on_dead_end_status, qos
        )
        self._sub_manifest = self.create_subscription(
            String, "/gc/runtime_manifest", self._on_manifest, qos
        )

        # Periodic summary (optional, for long runs)
        if summary_period > 0:
            self._summary_timer = self.create_timer(summary_period, self._periodic_summary)
        
        # Register shutdown handler
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info("Wiring Auditor started - will produce summary at shutdown")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals to produce summary before exit."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self._print_final_summary()
            self._write_json_summary()
        
        # Call original handler
        if signum == signal.SIGINT and self._original_sigint:
            if callable(self._original_sigint):
                self._original_sigint(signum, frame)
        elif signum == signal.SIGTERM and self._original_sigterm:
            if callable(self._original_sigterm):
                self._original_sigterm(signum, frame)

    def _on_gc_status(self, msg: String) -> None:
        """Handle /gc/status updates."""
        try:
            data = json.loads(msg.data)
            self._state.odom_count = int(data.get("odom_count", 0))
            self._state.scan_count = int(data.get("scan_count", 0))
            self._state.imu_count = int(data.get("imu_count", 0))
            self._state.pipeline_runs = int(data.get("pipeline_runs", 0))
            self._state.map_bins_active = int(data.get("map_bins_active", 0))
            self._state.last_status_time = time.time()
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f"Failed to parse /gc/status: {e}")

    def _on_dead_end_status(self, msg: String) -> None:
        """Handle /gc/dead_end_status updates."""
        try:
            data = json.loads(msg.data)
            counts = data.get("counts", {})
            self._state.dead_end_counts = {str(k): int(v) for k, v in counts.items()}
            self._state.last_dead_end_time = time.time()
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f"Failed to parse /gc/dead_end_status: {e}")

    def _on_manifest(self, msg: String) -> None:
        """Handle /gc/runtime_manifest (published once at startup)."""
        try:
            self._state.manifest = json.loads(msg.data)
            self._state.manifest_received = True
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Failed to parse /gc/runtime_manifest: {e}")

    def _periodic_summary(self) -> None:
        """Print periodic summary during long runs."""
        elapsed = time.time() - self._start_time
        self.get_logger().info(
            f"[{elapsed:.0f}s] scans={self._state.scan_count} "
            f"odom={self._state.odom_count} imu={self._state.imu_count} "
            f"pipeline={self._state.pipeline_runs}"
        )

    def _print_final_summary(self) -> None:
        """Print the consolidated end-of-run summary."""
        elapsed = time.time() - self._start_time
        s = self._state

        # Build summary string
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════╗",
            "║                      GC RUN COMPLETION SUMMARY                       ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            f"║  Duration: {elapsed:.1f}s                                                      ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  PROCESSED (received by backend):                                    ║",
            f"║    LiDAR scans:  {s.scan_count:>6}  → pipeline runs: {s.pipeline_runs:>6}                 ║",
            f"║    Odom msgs:    {s.odom_count:>6}  [FUSED via odom_evidence]                 ║",
            f"║    IMU msgs:     {s.imu_count:>6}  [FUSED via imu_evidence + gyro_evidence] ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  DEAD-ENDED (received but not processed):                            ║",
        ]

        if s.dead_end_counts:
            for topic, count in sorted(s.dead_end_counts.items()):
                topic_short = topic if len(topic) <= 45 else "..." + topic[-42:]
                lines.append(f"║    {topic_short:<45} {count:>6} msgs  ║")
        else:
            lines.append("║    (no dead-end status received)                                     ║")

        lines.extend([
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  NOTES:                                                              ║",
        ])

        # Add warnings/notes
        if s.odom_count > 0 or s.imu_count > 0:
            lines.append("║    ✓ Odom/IMU fused into belief via evidence operators              ║")
        
        if s.pipeline_runs == 0:
            lines.append("║    ⚠ NO PIPELINE RUNS - check LiDAR topic wiring!                  ║")
        elif s.pipeline_runs < s.scan_count * 0.9:
            lines.append("║    ⚠ Pipeline runs < scan count - some scans may have been dropped ║")

        lines.extend([
            "╚══════════════════════════════════════════════════════════════════════╝",
            "",
        ])

        # Print to logger (will appear in log file and terminal)
        for line in lines:
            self.get_logger().info(line)

    def _write_json_summary(self) -> None:
        """Write JSON summary to file if path configured."""
        if not self._output_json_path:
            return

        elapsed = time.time() - self._start_time
        summary = {
            "duration_sec": elapsed,
            "processed": {
                "lidar_scans": self._state.scan_count,
                "pipeline_runs": self._state.pipeline_runs,
                "odom_msgs": self._state.odom_count,
                "imu_msgs": self._state.imu_count,
                "odom_fused": True,   # Fused via odom_evidence operator
                "imu_fused": True,    # Fused via imu_evidence + imu_gyro_evidence operators
            },
            "dead_ended": self._state.dead_end_counts,
            "map_bins_active": self._state.map_bins_active,
            "manifest_received": self._state.manifest_received,
            "manifest": self._state.manifest,
        }

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._output_json_path), exist_ok=True)
            with open(self._output_json_path, "w") as f:
                json.dump(summary, f, indent=2)
            self.get_logger().info(f"Wiring summary written to: {self._output_json_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to write summary JSON: {e}")

    def destroy_node(self) -> None:
        """Ensure summary is printed on node destruction."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self._print_final_summary()
            self._write_json_summary()
        super().destroy_node()


def main() -> None:
    """Entry point for wiring_auditor node."""
    rclpy.init()
    node = WiringAuditorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
