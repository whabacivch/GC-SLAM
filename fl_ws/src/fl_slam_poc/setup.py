from setuptools import find_packages, setup

package_name = "fl_slam_poc"

setup(
    name=package_name,
    version="0.1.0",  # Geometric Compositional v2
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/gc_rosbag.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/gc_dead_end_audit.yaml",
                "config/gc_unified.yaml",
            ],
        ),
    ],
    install_requires=["setuptools", "numpy", "scipy", "jax", "pyyaml", "rerun-sdk"],
    zip_safe=True,
    maintainer="Will Haber",
    maintainer_email="whab13@mit.edu",
    description="Geometric Compositional SLAM v2 - Branch-free compositional inference SLAM (ROS 2 Jazzy)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Geometric Compositional backend (branch-free implementation)
            "gc_backend_node = fl_slam_poc.backend.backend_node:main",
            # Sensor Hub (single process with all frontend nodes)
            "gc_sensor_hub = fl_slam_poc.frontend.hub.gc_sensor_hub:main",
            # Individual sensor nodes (can also run standalone)
            "odom_normalizer = fl_slam_poc.frontend.sensors.odom_normalizer:main",
            "imu_normalizer = fl_slam_poc.frontend.sensors.imu_normalizer:main",
            # Audit / accountability
            "gc_dead_end_audit_node = fl_slam_poc.frontend.audit.dead_end_audit_node:main",
            "wiring_auditor = fl_slam_poc.frontend.audit.wiring_auditor:main",
        ],
    },
)
