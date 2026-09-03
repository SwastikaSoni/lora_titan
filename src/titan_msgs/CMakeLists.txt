cmake_minimum_required(VERSION 3.8)

# Project name — MUST match <name> in package.xml
project(titan_msgs)

# Use C++17 and enable all warnings (ROS 2 convention)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Find dependencies ──
# ament_cmake: the ROS 2 build system
# rosidl_default_generators: the tool that reads .msg files and generates
#   Python classes + C++ structs from them
# std_msgs, geometry_msgs, sensor_msgs: standard message types we reference
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# ── Declare message files ──
# Every .msg file you add to the msg/ folder must be listed here.
# rosidl_generate_interfaces reads these files, and generates:
#   - Python: from titan_msgs.msg import Heartbeat
#   - C++:    #include "titan_msgs/msg/heartbeat.hpp"
rosidl_generate_interfaces(${PROJECT_NAME}
  # Placeholder message — we'll add real ones as we build each subsystem
  "msg/Heartbeat.msg"

  # If our messages reference standard types (they will), list them here:
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
)

# ── Testing (optional, skip for now) ──
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ── Finalize ──
ament_package()