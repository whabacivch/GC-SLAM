"""
GC Sensor Hub - Single-process frontend for Geometric Compositional SLAM.

This module provides a unified sensor hub that runs all frontend
preprocessing nodes in a single process with MultiThreadedExecutor.
"""

from fl_slam_poc.frontend.hub.gc_sensor_hub import main as sensor_hub_main

__all__ = ["sensor_hub_main"]
