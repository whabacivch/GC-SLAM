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
  DEPENDENCIES std_msgs geometry_msgs
)

install(DIRECTORY fl_slam_poc/
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

add_executable(image_decompress_cpp src/image_decompress_node.cpp)
ament_target_dependencies(image_decompress_cpp rclcpp sensor_msgs)
target_link_libraries(image_decompress_cpp cv_bridge::cv_bridge ${OpenCV_LIBRARIES})

install(TARGETS image_decompress_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python node entrypoints into the ROS 2 "libexec" location so that
# `ros2 launch` can resolve Node(executable=...) for this package.
install(PROGRAMS
  scripts/frontend_node
  scripts/backend_node
  scripts/odom_bridge
  scripts/livox_converter
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch resource config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
