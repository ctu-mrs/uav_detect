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
  pcl_ros
)

# set(CMAKE_CXX_FLAGS "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN -DDISABLE_PARALLELIZATION -DDEBUG_PARAMS")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN -DDISABLE_PARALLELIZATION -DDEBUG_PARAMS")
# set(CMAKE_CXX_FLAGS_RELEASE "-Werror -O3 -g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN -DDISABLE_PARALLELIZATION -DDEBUG_PARAMS")
set(CMAKE_CXX_FLAGS "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN -DDISABLE_PARALLELIZATION -DDEBUG_PARAMS")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN -DDISABLE_PARALLELIZATION -DDEBUG_PARAMS")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN -DDISABLE_PARALLELIZATION -DDEBUG_PARAMS")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options("-Wall")

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED COMPONENTS)

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
  CATKIN_DEPENDS message_runtime pcl_ros image_transport mrs_lib # roscpp rospy std_msgs
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
  ${PCL_INCLUDE_DIRS}
  # ~/git/darknet_ocl/src
)

## Add link directories
link_directories(
  ${PCL_LIBRARY_DIRS}
)

# This causes adds the -march=native gcc flag that prevents ditributed compilation
add_definitions(
  ${PCL_DEFINITIONS}
  )

add_library(PCLDetector src/pcl_detect_nodelet.cpp)
add_dependencies(PCLDetector
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(PCLDetector
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)
target_include_directories(PCLDetector SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})

add_library(DepthmapPreprocessor src/depthmap_preprocessing.cpp)
add_dependencies(DepthmapPreprocessor
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(DepthmapPreprocessor
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
