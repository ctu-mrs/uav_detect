#ifndef MAIN_H
#define MAIN_H

#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>

#include <std_msgs/Float64.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float64.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/geometries.hpp>

#include <list>

#include <uav_detect/Detection.h>
#include <uav_detect/Detections.h>
#include <uav_detect/ImagePixel.h>
#include <mrs_lib/ParamLoader.h>
#include <mrs_lib/DynamicReconfigureMgr.h>
#include <mrs_lib/subscribe_handler.h>
#include <mrs_lib/transformer.h>
#include <mrs_lib/geometry_utils.h>

#endif //  MAIN_H
