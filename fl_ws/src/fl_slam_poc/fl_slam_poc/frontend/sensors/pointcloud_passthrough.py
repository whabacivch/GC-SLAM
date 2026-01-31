"""
PointCloud2 passthrough: subscribe to PointCloud2, republish to canonical topic.

For bags (e.g. Kimera/VLP-16) that publish PointCloud2 directly instead of Livox CustomMsg.
Single path; no conversion. Params from ROS. Used when pointcloud_layout is vlp16.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2


class PointcloudPassthroughNode(Node):
    """
    Republishes PointCloud2 from input_topic to output_topic.
    Used when config uses pointcloud_passthrough (e.g. Kimera) instead of livox_converter.
    """

    def __init__(self, parameter_overrides: Optional[Dict[str, Any]] = None) -> None:
        overrides = None
        if parameter_overrides:
            from rclpy.parameter import Parameter
            overrides = [Parameter(k, value=v) for k, v in parameter_overrides.items()]
        super().__init__("pointcloud_passthrough", parameter_overrides=overrides)

        self.declare_parameter("input_topic", "/acl_jackal/lidar_points")
        self.declare_parameter("output_topic", "/gc/sensors/lidar_points")
        self.declare_parameter("qos_depth", 10)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        qos_depth = int(self.get_parameter("qos_depth").value)

        if not input_topic or not output_topic:
            raise ValueError("pointcloud_passthrough: input_topic and output_topic must be set")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )
        self.pub = self.create_publisher(PointCloud2, output_topic, qos)
        self.sub = self.create_subscription(PointCloud2, input_topic, self._cb, qos)
        self.get_logger().info(
            f"PointCloud2 passthrough: {input_topic} -> {output_topic}"
        )

    def _cb(self, msg: PointCloud2) -> None:
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = PointcloudPassthroughNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
