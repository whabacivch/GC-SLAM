"""
Depth passthrough: subscribe to raw depth (sensor_msgs/Image), republish to canonical topic.

For bags (e.g. Kimera) that publish raw depth (16UC1 mm) instead of compressed depth.
Optional scale_mm_to_m: convert 16UC1 mm to 32FC1 m on output.
Single path; no fallbacks. Params from ROS.
"""

import rclpy
from rclpy.node import Node
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

    pub = node.create_publisher(Image, depth_out, qos_depth)
    import numpy as np

    def cb(msg: Image):
        if scale_mm_to_m and msg.encoding in ("16UC1", "16SC1"):
            # 16-bit mm -> 32FC1 m (step is bytes per row; 16UC1 => step = width*2)
            data = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
            out = (data.astype(np.float32) / 1000.0)
            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = msg.height
            out_msg.width = msg.width
            out_msg.encoding = "32FC1"
            out_msg.is_bigendian = 0
            out_msg.step = out_msg.width * 4
            out_msg.data = out.astype(np.float32).tobytes()
            pub.publish(out_msg)
        else:
            pub.publish(msg)

    sub = node.create_subscription(Image, depth_in, cb, qos_depth)
    node.get_logger().info(
        "Depth passthrough: %s -> %s (scale_mm_to_m=%s)",
        depth_in, depth_out, scale_mm_to_m,
    )
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
