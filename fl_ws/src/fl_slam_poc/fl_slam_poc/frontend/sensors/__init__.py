"""
Sensor utility nodes for Geometric Compositional SLAM v2.

LiDAR: pointcloud_passthrough (PointCloud2 bags, e.g. Kimera/VLP-16).
Normalizers: odom_normalizer, imu_normalizer.

ROS-dependent node classes are not imported here so that backend/pipeline can be
imported without a ROS environment. Use submodules directly, e.g.:
  from fl_slam_poc.frontend.sensors.pointcloud_passthrough import PointcloudPassthroughNode
"""

__all__ = []
