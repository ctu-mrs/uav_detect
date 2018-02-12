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
  dynamic_reconfigure
  cv_bridge
  message_generation
  mrs_msgs
)

set(CMAKE_CXX_FLAGS "-std=c++11 -DGPU -DOPENCV -DOPENCL -Wall -O2")

find_package(OpenCV REQUIRED)

add_message_files(DIRECTORY msgs FILES
  Detection.msg
  Detections.msg
)

generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime # image_transport roscpp rospy std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ~/git/darknet_ocl/src
)


add_executable(display_detections src/display_detections.cpp)
add_dependencies(display_detections
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(display_detections
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# add_definitions(-DDEBUG)
add_executable(uav_detect src/uav_detect.cpp src/detector.cpp)
add_dependencies(uav_detect
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(uav_detect
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  darknet-cpp-shared
)
