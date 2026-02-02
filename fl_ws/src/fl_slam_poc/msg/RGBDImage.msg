# Paired RGB-D frame for single-path camera pipeline.
# Publisher pairs RGB and depth by timestamp; consumer receives coherent frames.

std_msgs/Header header    # Timestamp of the pair (typically from RGB)

sensor_msgs/Image rgb     # RGB image (encoding: rgb8)
sensor_msgs/Image depth   # Depth image (encoding: 32FC1, meters)
