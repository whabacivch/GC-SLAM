cmake_minimum_required(VERSION 3.10...3.29)
project(fl_slam_poc)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AnchorCreate.msg"
  "msg/IMUFactor.msg"
  "msg/LoopFactor.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

install(DIRECTORY fl_slam_poc/
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(PROGRAMS
  scripts/backend_node
  scripts/frontend_node
  scripts/image_decompress
  scripts/livox_converter
  scripts/tb3_odom_bridge
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch resource config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
