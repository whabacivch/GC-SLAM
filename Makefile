# Impact Project v1 — Geometric Compositional SLAM v2
# Run from project root.

.PHONY: eval gc-eval build clean

# Primary eval: run GC pipeline + evaluation (artifacts under results/gc_*)
eval: gc-eval

gc-eval:
	bash tools/run_and_evaluate_gc.sh

# Build only (no rosbag run)
build:
	cd fl_ws && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fl_slam_poc && source install/setup.bash

# Remove evaluation results and ROS/log build artifacts to reduce bulk
clean:
	rm -rf results/gc_*
	rm -f results/gc_slam_diagnostics.npz
	rm -rf fl_ws/build fl_ws/install fl_ws/log
