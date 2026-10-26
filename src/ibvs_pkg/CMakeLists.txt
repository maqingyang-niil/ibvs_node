cmake_minimum_required(VERSION 3.18)
project(ibvs_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
  core
  imgproc
  dnn
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    /usr/include/aarch64-linux-gnu
    /usr/include
)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS
    /usr/lib/aarch64-linux-gnu
    /usr/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "找不到TensorRT开发文件，请确认已安装NvInfer.h和libnvinfer.so")
endif()

add_executable(button_position_capture_node
  src/button_position_capture_node.cpp
  src/yolo_red_button_detector.cpp
)

target_include_directories(button_position_capture_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(button_position_capture_node
  Eigen3::Eigen
  ${OpenCV_LIBRARIES}
  ${TENSORRT_LIBRARY}
  CUDA::cudart
)

ament_target_dependencies(button_position_capture_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  message_filters
  cv_bridge
)

install(TARGETS
  button_position_capture_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
