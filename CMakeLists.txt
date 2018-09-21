cmake_minimum_required(VERSION 2.8.3)
project(uav_detect)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  dynamic_reconfigure
  cv_bridge
  image_transport
  image_geometry
  message_generation
  mrs_msgs
  tf2
  tf2_ros
  tf2_msgs
  tf2_sensor_msgs
  tf2_eigen
  tf2_geometry_msgs
  mrs_lib
)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -g -O0")

find_package(OpenCV REQUIRED)

# include Eigen3
find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

add_message_files(DIRECTORY msgs FILES
  Detection.msg
  Detections.msg
)

generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

generate_dynamic_reconfigure_options(
  config/dm_params.cfg
  )

catkin_package(
  CATKIN_DEPENDS message_runtime image_transport mrs_lib # roscpp rospy std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  # ~/git/darknet_ocl/src
)

add_executable(depth_detect_node src/depth_detect_node.cpp src/DepthBlobDetector.cpp src/utils.cpp)
add_dependencies(depth_detect_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(depth_detect_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
