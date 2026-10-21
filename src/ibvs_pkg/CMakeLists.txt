cmake_minimum_required(VERSION 3.8)
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
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
  core
  imgproc
  dnn
)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime解压目录")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS
    ${ONNXRUNTIME_ROOT}/include
    $ENV{ONNXRUNTIME_ROOT}/include
)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS
    ${ONNXRUNTIME_ROOT}/lib
    $ENV{ONNXRUNTIME_ROOT}/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "找不到ONNX Runtime。请设置ONNXRUNTIME_ROOT为ONNX Runtime解压目录")
endif()

get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)

add_executable(ibvs_node
  src/ibvs_node.cpp
  src/ibvs_controller.cpp
  src/red_button_detector.cpp
  src/yolo_obj_detector.cpp
)

target_include_directories(ibvs_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(ibvs_node
  Eigen3::Eigen
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIBRARY}
)

set_target_properties(ibvs_node PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
  INSTALL_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
)

ament_target_dependencies(ibvs_node
  rclcpp
  sensor_msgs
  geometry_msgs
  message_filters
  cv_bridge
)

install(TARGETS ibvs_node
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
