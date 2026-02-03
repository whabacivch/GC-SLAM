"""
Depth passthrough: subscribe to raw depth (sensor_msgs/Image), republish to canonical topic.

For bags (e.g. Kimera) that publish raw depth (16UC1 mm) instead of compressed depth.
Optional scale_mm_to_m: convert 16UC1 mm to 32FC1 m on output.
Single path; no fallbacks. Params from ROS.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


def main():
    rclpy.init()
    node = Node("depth_passthrough")
    node.declare_parameter("depth_raw_topic", "")
    node.declare_parameter("depth_output_topic", "/gc/sensors/camera_depth")
    node.declare_parameter("scale_mm_to_m", True)
    node.declare_parameter("qos_depth", 10)

    depth_in = node.get_parameter("depth_raw_topic").value
    depth_out = node.get_parameter("depth_output_topic").value
    scale_mm_to_m = node.get_parameter("scale_mm_to_m").value
    qos_depth = int(node.get_parameter("qos_depth").value)

    if not depth_in or not depth_out:
        node.get_logger().error("depth_raw_topic and depth_output_topic must be set")
        rclpy.shutdown()
        return

    # Sensor bags often use BEST_EFFORT; use it for subscription so we receive from bag.
    qos_sensor = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=qos_depth,
        durability=DurabilityPolicy.VOLATILE,
    )
    pub = node.create_publisher(Image, depth_out, qos_depth)
    import numpy as np

    def cb(msg: Image):
        height = int(msg.height)
        width = int(msg.width)
        if height <= 0 or width <= 0:
            node.get_logger().warn(
                "Depth passthrough: invalid image size %sx%s" % (height, width)
            )
            return
        if msg.encoding in ("16UC1", "16SC1"):
            if not scale_mm_to_m:
                node.get_logger().error(
                    "Depth passthrough: received %s but scale_mm_to_m=False; "
                    "depth output must be 32FC1 for backend" % msg.encoding
                )
                return
            # 16-bit mm -> 32FC1 m (respect step; slice to width if padding exists)
            row_elems = max(1, int(msg.step) // 2)
            data = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, row_elems)
            data = data[:, :width]
            out = (data.astype(np.float32) / 1000.0)
            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = height
            out_msg.width = width
            out_msg.encoding = "32FC1"
            out_msg.is_bigendian = 0
            out_msg.step = width * 4
            out_msg.data = out.tobytes()
            pub.publish(out_msg)
            return
        if msg.encoding == "32FC1":
            # Ensure packed rows (no padding) for backend reshape.
            row_elems = max(1, int(msg.step) // 4)
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(height, row_elems)
            data = data[:, :width]
            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = height
            out_msg.width = width
            out_msg.encoding = "32FC1"
            out_msg.is_bigendian = 0
            out_msg.step = width * 4
            out_msg.data = data.astype(np.float32).tobytes()
            pub.publish(out_msg)
            return
        node.get_logger().error(
            "Depth passthrough: unsupported encoding %r (expected 16UC1 or 32FC1)"
            % msg.encoding
        )

    sub = node.create_subscription(Image, depth_in, cb, qos_sensor)
    node.get_logger().info(
        "Depth passthrough: %s -> %s (scale_mm_to_m=%s)"
        % (depth_in, depth_out, scale_mm_to_m)
    )
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
