from setuptools import find_packages, setup

package_name = "fl_slam_poc"

setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/poc_m3dgr_rosbag.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/qos_override.yaml",
                "config/fl_slam_poc_base.yaml",
            ],
        ),
        (
            "share/" + package_name + "/config/presets",
            [
                "config/presets/m3dgr.yaml",
            ],
        ),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="Will Haber",
    maintainer_email="whab13@mit.edu",
    description="Frobenius-Legendre compositional inference SLAM POC (ROS 2 Jazzy)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Main nodes
            "frontend_node = fl_slam_poc.frontend.frontend_node:main",
            "backend_node = fl_slam_poc.backend.backend_node:main",
            # Utility nodes
            "odom_bridge = fl_slam_poc.frontend.sensors.odom_bridge:main",
            "livox_converter = fl_slam_poc.frontend.sensors.livox_converter:main",
        ],
    },
)
