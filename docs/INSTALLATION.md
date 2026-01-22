# Installation Requirements

## System Dependencies

### Required
```bash
# ROS 2 Jazzy (already installed)
# Python 3.10+ (already installed)
# NumPy, SciPy (see requirements.txt)
```

### Optional - Visualization
```bash
# Foxglove Bridge (for real-time visualization)
sudo apt install ros-jazzy-foxglove-bridge

# Foxglove Studio (already installed)
# Desktop app available at: foxglove-studio
```

## Python Dependencies
```bash
pip install -r requirements.txt
```

## Building the Project
```bash
cd fl_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select fl_slam_poc
source install/setup.bash
```

## Running Tests

See **[TESTING.md](TESTING.md)** for complete testing documentation.

### Quick Validation
```bash
cd /path/to/Impact Project_v1/fl_ws/src/fl_slam_poc
pytest -q
```

### Full Integration Test (requires test data)
```bash
# Download test data first
./tools/download_tb3_rosbag.sh

# Run with Foxglove visualization
./tools/test-integration.sh

# Run without Foxglove
ENABLE_FOXGLOVE=0 ./tools/test-integration.sh
```

## Visualization Setup

If foxglove-bridge is not installed, the integration test will fail at launch (unless disabled).

**To install manually:**
```bash
sudo apt update
sudo apt install ros-jazzy-foxglove-bridge
```

**Then verify:**
```bash
source /opt/ros/jazzy/setup.bash
ros2 pkg list | grep foxglove
# Should show: foxglove_bridge
```
