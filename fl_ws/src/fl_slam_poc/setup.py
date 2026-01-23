from setuptools import find_packages, setup

package_name = "fl_slam_poc"

setup(
    name=package_name,
    version="0.1.0",  # Golden Child v2
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
                "config/gc_backend.yaml",
            ],
        ),
    ],
    install_requires=["setuptools", "numpy", "scipy", "jax"],
    zip_safe=True,
    maintainer="Will Haber",
    maintainer_email="whab13@mit.edu",
    description="Golden Child SLAM v2 - Branch-free compositional inference SLAM (ROS 2 Jazzy)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Golden Child backend (branch-free implementation)
            "gc_backend_node = fl_slam_poc.backend.backend_node:main",
            "backend_node = fl_slam_poc.backend.backend_node:main",
            # Utility nodes
            "odom_bridge = fl_slam_poc.frontend.sensors.odom_bridge:main",
            "livox_converter = fl_slam_poc.frontend.sensors.livox_converter:main",
        ],
    },
)
