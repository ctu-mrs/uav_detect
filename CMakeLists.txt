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
  nodelet
)

find_file(DISTCC_FULL_PATH distcc)
find_file(DISTCPP_FULL_PATH distg++)
IF (EXISTS ${DISTCC_FULL_PATH})
  IF (EXISTS ${DISTCPP_FULL_PATH})
    message(STATUS "Found distcc and distg++, using distributed compilation")
    SET(CMAKE_C_COMPILER ${DISTCC_FULL_PATH})
    SET(CMAKE_CXX_COMPILER ${DISTCPP_FULL_PATH})
  ELSE()
    message(WARNING "Found only distcc, but not distg++, not using distributed compilation")
  ENDIF()
ELSE()
  message(STATUS "Not found distcc, not using distributed compilation")
ENDIF()

set(CMAKE_CXX_FLAGS "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")
set(CMAKE_CXX_FLAGS_RELEASE "-Werror -O3")

add_compile_options("-std=c++17" "-Wall")

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

find_package(OpenCV REQUIRED)

# include Eigen3
find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

add_message_files(DIRECTORY msgs FILES
  Detection.msg
  Detections.msg
  Contour.msg
  ImagePixel.msg
  BlobDetection.msg
  BlobDetections.msg
)

generate_messages(DEPENDENCIES
  std_msgs
  sensor_msgs
)

generate_dynamic_reconfigure_options(
  config/DetectionParams.cfg
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

add_library(DepthDetector src/depth_detect_nodelet.cpp src/DepthBlobDetector.cpp)
add_dependencies(DepthDetector
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(DepthDetector
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(dbg_display_node src/dbg_display_node.cpp)
add_dependencies(dbg_display_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(dbg_display_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
