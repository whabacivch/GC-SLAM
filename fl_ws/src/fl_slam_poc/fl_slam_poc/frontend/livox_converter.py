#!/usr/bin/env python3
"""
Livox CustomMsg to PointCloud2 Converter Node.

Converts Livox proprietary CustomMsg format to standard sensor_msgs/PointCloud2
so FL-SLAM can process Livox LiDAR data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from livox_ros_driver2.msg import CustomMsg


class LivoxConverterNode(Node):
    """Converts Livox CustomMsg to PointCloud2."""

    def __init__(self):
        super().__init__('livox_converter')
        
        # Parameters
        self.declare_parameter('input_topic', '/livox/mid360/lidar')
        self.declare_parameter('output_topic', '/lidar/points')
        self.declare_parameter('frame_id', 'mid360_frame')
        
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        
        # Publisher for PointCloud2
        self.publisher = self.create_publisher(
            PointCloud2,
            output_topic,
            10
        )

        self.subscription = self.create_subscription(
            CustomMsg,
            input_topic,
            self._on_custom_msg,
            10
        )
        
        self.get_logger().info(f'Livox converter node started')
        self.get_logger().info(f'  Input:  {input_topic}')
        self.get_logger().info(f'  Output: {output_topic}')
        self.get_logger().info(f'  Frame:  {self.frame_id}')

    def _on_custom_msg(self, msg: CustomMsg):
        if not msg.points:
            return

        points = np.array([(p.x, p.y, p.z) for p in msg.points], dtype=np.float32)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]

        cloud_msg = PointCloud2()
        cloud_msg.header = Header()
        cloud_msg.header.stamp = msg.header.stamp
        cloud_msg.header.frame_id = self.frame_id or msg.header.frame_id
        cloud_msg.height = 1
        cloud_msg.width = points.shape[0]
        cloud_msg.is_dense = bool(points.shape[0] == len(msg.points))
        cloud_msg.is_bigendian = False
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 12
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.data = points.tobytes()

        self.publisher.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LivoxConverterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
