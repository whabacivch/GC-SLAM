cmake_minimum_required(VERSION 3.10...3.29)
project(fl_slam_poc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AnchorCreate.msg"
  "msg/IMUSegment.msg"
  "msg/LoopFactor.msg"
  "msg/RGBDImage.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

install(DIRECTORY fl_slam_poc/
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

add_executable(camera_rgbd_node src/camera_rgbd_node.cpp)
ament_target_dependencies(camera_rgbd_node rclcpp sensor_msgs)
target_link_libraries(camera_rgbd_node cv_bridge::cv_bridge ${OpenCV_LIBRARIES})
# Link with generated messages from this package
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(camera_rgbd_node "${cpp_typesupport_target}")

install(TARGETS camera_rgbd_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python node entrypoints into the ROS 2 "libexec" location so that
# `ros2 launch` can resolve Node(executable=...) for this package.
install(PROGRAMS
  scripts/gc_backend_node
  scripts/gc_sensor_hub
  scripts/odom_normalizer
  scripts/imu_normalizer
  scripts/gc_dead_end_audit_node
  scripts/wiring_auditor
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch resource
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
