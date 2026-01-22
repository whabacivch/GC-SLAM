"""
Status Monitoring Utility.

Centralizes sensor connectivity monitoring and status publishing.
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from std_msgs.msg import String

from fl_slam_poc.common import constants


@dataclass
class SensorStatus:
    """
    Tracks sensor connection status with timestamps.
    
    Replaces the inline SensorStatus from frontend_node.py
    """
    name: str
    topic: str
    last_received: Optional[float] = None
    message_count: int = 0
    warned_missing: bool = False
    warned_stale: bool = False
    
    def mark_received(self):
        """Mark that a message was received."""
        self.last_received = time.time()
        self.message_count += 1
        self.warned_stale = False  # Reset stale warning on new data
    
    def is_connected(self, timeout_sec: float = constants.SENSOR_TIMEOUT_DEFAULT) -> bool:
        """Check if sensor is connected (received data recently)."""
        if self.last_received is None:
            return False
        return (time.time() - self.last_received) < timeout_sec
    
    def age_sec(self) -> Optional[float]:
        """Get age of last message in seconds."""
        if self.last_received is None:
            return None
        return time.time() - self.last_received
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "topic": self.topic,
            "connected": self.is_connected(),
            "message_count": self.message_count,
            "age_sec": self.age_sec(),
            "ever_received": self.last_received is not None,
        }


class StatusMonitor:
    """
    Monitors sensor connectivity and publishes status.
    
    This extracts and consolidates the status monitoring logic that was
    scattered throughout the frontend and backend nodes.
    
    Features:
    - Tracks multiple sensors with individual timeouts
    - Warns once per sensor (no log spam)
    - Grace period after startup
    - JSON status publishing
    - Operational state determination
    """
    
    def __init__(
        self,
        node,
        status_topic: str = "/cdwm/status",
        grace_period: float = constants.SENSOR_STARTUP_GRACE_PERIOD,
        check_period: float = constants.STATUS_CHECK_PERIOD,
        sensor_timeout: float = constants.SENSOR_TIMEOUT_DEFAULT,
    ):
        """
        Initialize status monitor.
        
        Args:
            node: ROS node (for logging and publishing)
            status_topic: Topic to publish JSON status
            grace_period: Grace period after startup before warnings
            check_period: How often to check status (seconds)
            sensor_timeout: Timeout for considering sensor stale
        """
        self.node = node
        self.sensors: Dict[str, SensorStatus] = {}
        self.grace_period = grace_period
        self.sensor_timeout = sensor_timeout
        self.start_time = time.time()
        
        # Publisher
        self.pub_status = node.create_publisher(String, status_topic, 10)
        
        # Timer for periodic status checks
        self.timer = node.create_timer(check_period, self._check_status_callback)
        
        # Additional counters (can be updated by node)
        self.extra_metrics: Dict[str, any] = {}
    
    def add_sensor(self, name: str, topic: str):
        """Register a sensor to monitor."""
        self.sensors[name] = SensorStatus(name=name, topic=topic)
    
    def mark_received(self, sensor_name: str):
        """Mark that a sensor received data."""
        if sensor_name in self.sensors:
            self.sensors[sensor_name].mark_received()
    
    def is_connected(self, sensor_name: str) -> bool:
        """Check if a sensor is connected."""
        if sensor_name not in self.sensors:
            return False
        return self.sensors[sensor_name].is_connected(self.sensor_timeout)
    
    def get_sensor(self, sensor_name: str) -> Optional[SensorStatus]:
        """Get sensor status object."""
        return self.sensors.get(sensor_name)
    
    def set_extra_metric(self, key: str, value: any):
        """Set an extra metric to include in status."""
        self.extra_metrics[key] = value
    
    def _check_status_callback(self):
        """Periodic status check callback."""
        elapsed = time.time() - self.start_time
        in_grace_period = elapsed < self.grace_period
        
        status_dict = {
            "timestamp": time.time(),
            "elapsed_sec": elapsed,
            "in_grace_period": in_grace_period,
            "sensors": {},
            "warnings": [],
            **self.extra_metrics,  # Include any extra metrics
        }
        
        # Check each sensor
        for name, sensor in self.sensors.items():
            status_dict["sensors"][name] = sensor.to_dict()
            
            if not in_grace_period:
                # Warn if sensor was never received
                if not sensor.last_received and not sensor.warned_missing:
                    sensor.warned_missing = True
                    msg = f"SENSOR MISSING: {sensor.name} on {sensor.topic} - never received!"
                    self.node.get_logger().warn(msg)
                    status_dict["warnings"].append(msg)
                
                # Warn if sensor went stale
                elif sensor.last_received and not sensor.is_connected(self.sensor_timeout):
                    if not sensor.warned_stale:
                        sensor.warned_stale = True
                        age = sensor.age_sec()
                        msg = f"SENSOR STALE: {sensor.name} on {sensor.topic} - last received {age:.1f}s ago"
                        self.node.get_logger().warn(msg)
                        status_dict["warnings"].append(msg)
        
        # Publish status
        msg = String()
        msg.data = json.dumps(status_dict)
        self.pub_status.publish(msg)
    
    def check_operational(
        self,
        required_sensors: List[str],
        log_status: bool = True,
    ) -> bool:
        """
        Check if system is operational (all required sensors connected).
        
        Args:
            required_sensors: List of sensor names that must be connected
            log_status: Whether to log operational status
            
        Returns:
            True if all required sensors are connected
        """
        elapsed = time.time() - self.start_time
        in_grace_period = elapsed < self.grace_period
        
        # Check required sensors
        all_connected = all(
            self.is_connected(name) for name in required_sensors
        )
        
        # Log if not operational
        if not all_connected and not in_grace_period and log_status:
            missing = [name for name in required_sensors if not self.is_connected(name)]
            self.node.get_logger().warn(
                f"SYSTEM NOT OPERATIONAL - missing sensors: {missing}. "
                "Connect real sensors or a simulator."
            )
        
        return all_connected
    
    def get_status_dict(self) -> dict:
        """Get current status as dictionary (for manual queries)."""
        elapsed = time.time() - self.start_time
        return {
            "timestamp": time.time(),
            "elapsed_sec": elapsed,
            "in_grace_period": elapsed < self.grace_period,
            "sensors": {
                name: sensor.to_dict()
                for name, sensor in self.sensors.items()
            },
            **self.extra_metrics,
        }
