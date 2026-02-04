"""
=============================================================================
IMU NORMALIZER - IMU Preprocessing for Geometric Compositional SLAM
=============================================================================

PLACEHOLDER / LANDING PAD FOR FUTURE DEVELOPMENT

This node subscribes to raw IMU data from the rosbag and publishes
normalized IMU to the /gc/sensors/imu canonical topic.

Current Implementation:
    - Passes through IMU with frame normalization
    - Basic finite-value validation

Future Development (TODO):
    - [ ] Gyro/accel range validation
    - [ ] Bias estimation and correction (if needed)
    - [ ] Covariance validation
    - [ ] Timestamp validation and monotonicity check
    - [ ] Frame convention enforcement
    - [ ] Unit validation (rad/s for gyro, m/s^2 for accel)
    - [ ] Fail-fast on invalid data

Topic Flow:
    /livox/mid360/imu (raw from bag) → [this node] → /gc/sensors/imu (canonical)

Reference: docs/GC_SLAM.md
"""

from __future__ import annotations

import numpy as np
import rclpy
from sensor_msgs.msg import Imu
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from std_msgs.msg import Float64
from typing import Any, Optional, Dict

from fl_slam_poc.frontend.sensors.time_alignment import TimeAligner


class ImuNormalizerNode(Node):
    """
    Normalizes raw IMU data for GC backend consumption.
    
    Subscribes to raw /livox/mid360/imu and publishes to /gc/sensors/imu.
    """

    def __init__(self, parameter_overrides: Optional[Dict[str, Any]] = None) -> None:
        overrides = None
        if parameter_overrides:
            overrides = [Parameter(k, value=v) for k, v in parameter_overrides.items()]
        super().__init__("imu_normalizer", parameter_overrides=overrides)

        # Parameters
        self.declare_parameter("input_topic", "/livox/mid360/imu")
        self.declare_parameter("output_topic", "/gc/sensors/imu")
        # If empty, preserve incoming frame id (strict no-TF mode contract).
        self.declare_parameter("output_frame", "")
        # Time alignment to reference (e.g., LiDAR scan timestamps).
        self.declare_parameter("enable_time_alignment", False)
        self.declare_parameter("time_reference_topic", "/gc/sensors/time_reference")
        self.declare_parameter("max_drift_sec", 0.5)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.enable_time_alignment = bool(self.get_parameter("enable_time_alignment").value)
        self.time_reference_topic = str(self.get_parameter("time_reference_topic").value)
        self.max_drift_sec = float(self.get_parameter("max_drift_sec").value)

        # QoS: BEST_EFFORT for IMU (high-rate sensor data)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            Imu, input_topic, self._on_imu, qos
        )
        self.pub = self.create_publisher(Imu, output_topic, qos)
        self._time_aligner: Optional[TimeAligner] = None
        if self.enable_time_alignment:
            if not self.time_reference_topic:
                raise ValueError("enable_time_alignment requires time_reference_topic")
            self._time_aligner = TimeAligner(max_drift_sec=self.max_drift_sec)
            self.sub_time_ref = self.create_subscription(
                Float64, self.time_reference_topic, self._on_time_ref, qos
            )

        self._msg_count = 0
        self._logged_first = False
        self._invalid_count = 0

        self.get_logger().info("=" * 60)
        self.get_logger().info("IMU NORMALIZER")
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

    def _on_imu(self, msg: Imu) -> None:
        """Process and normalize incoming IMU data."""
        self._msg_count += 1

        # =====================================================================
        # TODO: Add validation logic here
        # - Check gyro values are finite and within reasonable range
        # - Check accel values are finite and within reasonable range
        # - Validate covariances if present
        # - Check timestamp monotonicity
        # =====================================================================

        # Basic finite check (placeholder validation)
        gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        accel = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        if not all(np.isfinite(gyro)) or not all(np.isfinite(accel)):
            self._invalid_count += 1
            if self._invalid_count == 1:
                self.get_logger().warn(
                    f"IMU normalizer: invalid values in msg #{self._msg_count} (warn-once, then fail-fast)"
                )
                return
            self.get_logger().error(
                f"IMU normalizer: invalid values observed again at msg #{self._msg_count}; failing fast."
            )
            rclpy.shutdown()
            raise RuntimeError("IMU normalizer: repeated invalid IMU values (non-finite).")

        # Create normalized output message
        out = Imu()
        out.header = msg.header
        out.header.frame_id = self.output_frame or msg.header.frame_id
        if self._time_aligner is not None:
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._time_aligner.note_local_stamp(stamp_sec)
            self._time_aligner.try_init_offset(stamp_sec)
            if self._time_aligner.offset_ready:
                aligned = self._time_aligner.align(stamp_sec)
                out.header.stamp.sec = int(aligned)
                out.header.stamp.nanosec = int((aligned - int(aligned)) * 1e9)
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance
        out.angular_velocity = msg.angular_velocity
        out.angular_velocity_covariance = msg.angular_velocity_covariance
        out.linear_acceleration = msg.linear_acceleration
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        self.pub.publish(out)

        # Log first message
        if not self._logged_first:
            self._logged_first = True
            self.get_logger().info(
                f"IMU normalizer: first msg - gyro=({gyro[0]:.3f}, {gyro[1]:.3f}, {gyro[2]:.3f}), "
                f"accel=({accel[0]:.3f}, {accel[1]:.3f}, {accel[2]:.3f})"
            )


def main() -> None:
    """Standalone entry point for imu_normalizer node."""
    rclpy.init()
    node = ImuNormalizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
