"""
Image decompression node for compressed RGB-D bag playback.

Decompresses compressed image topics from rosbags to raw image format
for downstream processing by the frontend.

Subscribes to:
- /stereo_camera/left/image_rect_color/compressed/throttled (CompressedImage)
- /stereo_camera/depth/depth_registered/compressedDepth/throttled (CompressedImage)

Publishes:
- /camera/image_raw (sensor_msgs/Image)
- /camera/depth/image_raw (sensor_msgs/Image)

Reference: Hybrid Laser + RGB-D Sensor Fusion Architecture
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
import numpy as np

try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


class ImageDecompressNode(Node):
    """
    Decompresses compressed RGB and depth images for RGB-D processing.
    
    Handles two compressed image formats:
    - JPEG/PNG for RGB images (standard compressed format)
    - compressedDepth format for depth images (12-byte header + PNG)
    """
    
    def __init__(self):
        super().__init__("image_decompress")
        
        if not CV_AVAILABLE:
            self.get_logger().error(
                "cv_bridge or OpenCV not available. "
                "Install with: pip install opencv-python && sudo apt install ros-jazzy-cv-bridge"
            )
            return
        
        self.bridge = CvBridge()
        
        # Declare parameters for topic remapping
        self.declare_parameter("rgb_compressed_topic", 
                               "/stereo_camera/left/image_rect_color/compressed/throttled")
        self.declare_parameter("depth_compressed_topic",
                               "/stereo_camera/depth/depth_registered/compressedDepth/throttled")
        self.declare_parameter("rgb_output_topic", "/camera/image_raw")
        self.declare_parameter("depth_output_topic", "/camera/depth/image_raw")
        self.declare_parameter("depth_scale_mm_to_m", True)  # Convert mm to meters
        self.declare_parameter("qos_reliability", "reliable")
        
        rgb_in = self.get_parameter("rgb_compressed_topic").value
        depth_in = self.get_parameter("depth_compressed_topic").value
        rgb_out = self.get_parameter("rgb_output_topic").value
        depth_out = self.get_parameter("depth_output_topic").value
        self.depth_scale = self.get_parameter("depth_scale_mm_to_m").value
        qos_reliability = str(self.get_parameter("qos_reliability").value).lower()
        
        self._last_msg_keys = {}
        
        qos_profiles, qos_names = self._resolve_qos_profiles(qos_reliability)
        
        # Subscribers (optionally dual QoS)
        self.sub_rgb = []
        self.sub_depth = []
        for qos in qos_profiles:
            self.sub_rgb.append(self.create_subscription(
                CompressedImage,
                rgb_in,
                self.on_rgb_compressed,
                qos
            ))
            self.sub_depth.append(self.create_subscription(
                CompressedImage,
                depth_in,
                self.on_depth_compressed,
                qos
            ))
        
        # Publishers
        self.pub_rgb = self.create_publisher(Image, rgb_out, 10)
        self.pub_depth = self.create_publisher(Image, depth_out, 10)
        
        # Statistics
        self.rgb_count = 0
        self.depth_count = 0
        self.rgb_errors = 0
        self.depth_errors = 0
        
        self.get_logger().info(
            f"Image decompression node started:\n"
            f"  RGB:   {rgb_in} -> {rgb_out}\n"
            f"  Depth: {depth_in} -> {depth_out}\n"
            f"  QoS reliability: {', '.join(qos_names)}"
        )

    def _resolve_qos_profiles(self, reliability: str):
        """
        Resolve QoS profiles from reliability setting.

        Supported values:
          - reliable
          - best_effort
          - system_default
          - both (subscribe twice: RELIABLE + BEST_EFFORT)
        """
        rel_map = {
            "reliable": ReliabilityPolicy.RELIABLE,
            "best_effort": ReliabilityPolicy.BEST_EFFORT,
            "system_default": ReliabilityPolicy.SYSTEM_DEFAULT,
        }
        if reliability == "both":
            rels = [ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT]
            names = ["reliable", "best_effort"]
        elif reliability in rel_map:
            rels = [rel_map[reliability]]
            names = [reliability]
        else:
            rels = [ReliabilityPolicy.RELIABLE]
            names = ["reliable"]
        
        profiles = [
            QoSProfile(
                reliability=rel,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            for rel in rels
        ]
        return profiles, names

    def _is_duplicate(self, key: str, stamp) -> bool:
        """Prevent double-processing when subscribing with multiple QoS profiles."""
        if stamp is None:
            return False
        stamp_key = (stamp.sec, stamp.nanosec)
        if self._last_msg_keys.get(key) == stamp_key:
            return True
        self._last_msg_keys[key] = stamp_key
        return False
    
    def on_rgb_compressed(self, msg: CompressedImage):
        """Decompress RGB JPEG/PNG image."""
        if self._is_duplicate("rgb", msg.header.stamp):
            return
        try:
            # Decode compressed image (JPEG or PNG)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.rgb_errors += 1
                if self.rgb_errors <= 5:
                    self.get_logger().warning(
                        f"RGB decode returned None (format: {msg.format})"
                    )
                return
            
            # Convert BGR to RGB for ROS convention
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Convert to ROS Image
            img_msg = self.bridge.cv2_to_imgmsg(cv_image_rgb, encoding="rgb8")
            img_msg.header = msg.header
            self.pub_rgb.publish(img_msg)
            
            self.rgb_count += 1
            if self.rgb_count == 1:
                self.get_logger().info(
                    f"First RGB image decompressed: {cv_image.shape[1]}x{cv_image.shape[0]}"
                )
                
        except Exception as e:
            self.rgb_errors += 1
            if self.rgb_errors <= 5:
                self.get_logger().error(f"RGB decompression failed: {e}")
    
    def on_depth_compressed(self, msg: CompressedImage):
        """
        Decompress depth image (compressedDepth format).
        
        compressedDepth format:
        - First 12 bytes: header with depth quantization info
        - Remaining bytes: PNG-encoded depth image
        
        The depth values are typically in millimeters (uint16).
        """
        try:
            if self._is_duplicate("depth", msg.header.stamp):
                return
            # Check format
            if "compressedDepth" in msg.format or "png" in msg.format.lower():
                # compressedDepth format: skip 12-byte header
                # The header contains: depth_quantization (float32) and 
                # max_depth (float32), plus 4 bytes padding
                header_size = 12
                
                if len(msg.data) <= header_size:
                    self.depth_errors += 1
                    if self.depth_errors <= 5:
                        self.get_logger().warning("Depth message too short")
                    return
                
                # Try with header first
                np_arr = np.frombuffer(msg.data[header_size:], np.uint8)
                cv_depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                
                # If that fails, try without header (some bags use raw PNG)
                if cv_depth is None:
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            else:
                # Standard compressed (no header)
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            
            if cv_depth is None:
                self.depth_errors += 1
                if self.depth_errors <= 5:
                    self.get_logger().warning(
                        f"Depth decode returned None (format: {msg.format})"
                    )
                return
            
            # Convert to float32 meters
            if self.depth_scale and cv_depth.dtype in [np.uint16, np.int16]:
                # Typical: depth in millimeters -> convert to meters
                depth_m = cv_depth.astype(np.float32) / 1000.0
            elif cv_depth.dtype == np.float32:
                # Already in meters
                depth_m = cv_depth
            else:
                # Unknown format, try direct conversion
                depth_m = cv_depth.astype(np.float32)
            
            # Convert to ROS Image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
            depth_msg.header = msg.header
            self.pub_depth.publish(depth_msg)
            
            self.depth_count += 1
            if self.depth_count == 1:
                valid_depth = depth_m[depth_m > 0]
                if len(valid_depth) > 0:
                    self.get_logger().info(
                        f"First depth image decompressed: {cv_depth.shape[1]}x{cv_depth.shape[0]}, "
                        f"range: {valid_depth.min():.2f}m - {valid_depth.max():.2f}m"
                    )
                else:
                    self.get_logger().info(
                        f"First depth image decompressed: {cv_depth.shape[1]}x{cv_depth.shape[0]} (no valid depth)"
                    )
                    
        except Exception as e:
            self.depth_errors += 1
            if self.depth_errors <= 5:
                self.get_logger().error(f"Depth decompression failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    if not CV_AVAILABLE:
        print("ERROR: cv_bridge or OpenCV not available")
        return
    
    node = ImageDecompressNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Shutting down. Processed {node.rgb_count} RGB, {node.depth_count} depth images. "
            f"Errors: RGB={node.rgb_errors}, Depth={node.depth_errors}"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
