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

add_message_files(DIRECTORY msgs FILES
  uav_detect.msg
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
)

add_executable(uav_detect src/uav_detect.cpp)
add_dependencies(uav_detect ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(collision_flier
  ${catkin_LIBRARIES}
  darknet-cpp-shared
)
