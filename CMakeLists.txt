cmake_minimum_required(VERSION 2.8.3)
project(uav_detect)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  mavros_msgs
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
  mrs_estimation
)

set(CMAKE_CXX_FLAGS "-std=c++11 -DGPU -DOPENCV -DOPENCL -Wall -O2")

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

catkin_package(
  CATKIN_DEPENDS message_runtime image_transport mrs_estimation # roscpp rospy std_msgs
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
add_definitions(-g)
add_executable(display_detections src/display_detections.cpp)
add_dependencies(display_detections
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(display_detections
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# add_executable(uav_detect src/uav_detect.cpp src/detector.cpp)
# add_dependencies(uav_detect
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS}
# )
# target_link_libraries(uav_detect
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   darknet-cpp-shared
# )

# add_executable(filter_single src/filter_single.cpp src/rlcnn_util.cpp src/ocam_functions.cpp)
add_executable(filter_single src/filter_single.cpp src/rlcnn_util.cpp)
add_dependencies(filter_single
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(filter_single
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
