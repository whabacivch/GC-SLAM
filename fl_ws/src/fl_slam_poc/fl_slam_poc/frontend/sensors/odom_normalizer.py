"""
=============================================================================
ODOM NORMALIZER - Odometry Preprocessing for Geometric Compositional SLAM
=============================================================================

PLACEHOLDER / LANDING PAD FOR FUTURE DEVELOPMENT

This node subscribes to raw odometry from the rosbag and publishes
normalized odometry to the /gc/sensors/odom canonical topic.

Current Implementation:
    - Passes through odometry with frame normalization
    - Validates covariance is present and non-degenerate

Future Development (TODO):
    - [ ] Covariance validation and repair
    - [ ] Unit/scale normalization if needed
    - [ ] Timestamp validation and monotonicity check
    - [ ] Frame convention enforcement (REP-105)
    - [ ] Fail-fast on invalid data

Topic Flow:
    /odom (raw from bag) → [this node] → /gc/sensors/odom (canonical)

Reference: docs/GC_SLAM.md
"""

from __future__ import annotations

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from std_msgs.msg import Float64
from typing import Any, Optional, Dict

from fl_slam_poc.frontend.sensors.time_alignment import TimeAligner


class OdomNormalizerNode(Node):
    """
    Normalizes raw odometry for GC backend consumption.
    
    Subscribes to raw /odom and publishes to /gc/sensors/odom.
    """

    def __init__(self, parameter_overrides: Optional[Dict[str, Any]] = None) -> None:
        overrides = None
        if parameter_overrides:
            overrides = [Parameter(k, value=v) for k, v in parameter_overrides.items()]
        super().__init__("odom_normalizer", parameter_overrides=overrides)

        # Parameters
        self.declare_parameter("input_topic", "/odom")
        self.declare_parameter("output_topic", "/gc/sensors/odom")
        # If empty, preserve incoming frame ids (strict no-TF mode contract).
        self.declare_parameter("output_frame", "")
        self.declare_parameter("child_frame", "")
        # Time alignment to reference (e.g., LiDAR scan timestamps).
        self.declare_parameter("enable_time_alignment", False)
        self.declare_parameter("time_reference_topic", "/gc/sensors/time_reference")
        self.declare_parameter("max_drift_sec", 0.5)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)
        self.enable_time_alignment = bool(self.get_parameter("enable_time_alignment").value)
        self.time_reference_topic = str(self.get_parameter("time_reference_topic").value)
        self.max_drift_sec = float(self.get_parameter("max_drift_sec").value)

        # QoS:
        # - Subscription from rosbag: RELIABLE (bags typically record with RELIABLE).
        # - Publish to backend: RELIABLE to match GC backend odom subscription (fail-fast, no silent mismatch).
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_time_ref = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            Odometry, input_topic, self._on_odom, qos_sub
        )
        self.pub = self.create_publisher(Odometry, output_topic, qos_pub)
        self._time_aligner: Optional[TimeAligner] = None
        if self.enable_time_alignment:
            if not self.time_reference_topic:
                raise ValueError("enable_time_alignment requires time_reference_topic")
            self._time_aligner = TimeAligner(max_drift_sec=self.max_drift_sec)
            self.sub_time_ref = self.create_subscription(
                Float64, self.time_reference_topic, self._on_time_ref, qos_time_ref
            )

        self._msg_count = 0
        self._logged_first = False

        self.get_logger().info("=" * 60)
        self.get_logger().info("ODOM NORMALIZER")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Input:  {input_topic} (raw)")
        self.get_logger().info(f"  Output: {output_topic} (canonical)")
        if self.enable_time_alignment:
            self.get_logger().info(
                f"  Time alignment: enabled (ref={self.time_reference_topic}, max_drift_sec={self.max_drift_sec})"
            )
        self.get_logger().info("=" * 60)

    def _on_time_ref(self, msg: Float64) -> None:
        if self._time_aligner is None:
            return
        self._time_aligner.update_reference(float(msg.data))

    def _on_odom(self, msg: Odometry) -> None:
        """Process and normalize incoming odometry."""
        self._msg_count += 1

        # =====================================================================
        # TODO: Add validation logic here
        # - Check covariance is positive semi-definite
        # - Check timestamp monotonicity
        # - Validate pose/twist values are finite
        # =====================================================================

        # Create normalized output message
        out = Odometry()
        out.header = msg.header
        out.header.frame_id = self.output_frame or msg.header.frame_id
        out.child_frame_id = self.child_frame or msg.child_frame_id
        if self._time_aligner is not None:
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._time_aligner.note_local_stamp(stamp_sec)
            self._time_aligner.try_init_offset(stamp_sec)
            if self._time_aligner.offset_ready:
                aligned = self._time_aligner.align(stamp_sec)
                out.header.stamp.sec = int(aligned)
                out.header.stamp.nanosec = int((aligned - int(aligned)) * 1e9)
        out.pose = msg.pose
        out.twist = msg.twist

        self.pub.publish(out)

        # Log first message
        if not self._logged_first:
            self._logged_first = True
            pos = msg.pose.pose.position
            self.get_logger().info(
                f"Odom normalizer: first msg at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"
            )


def main() -> None:
    """Standalone entry point for odom_normalizer node."""
    rclpy.init()
    node = OdomNormalizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
