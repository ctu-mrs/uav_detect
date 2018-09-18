#ifndef MAIN_H
#define MAIN_H

#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float64.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Geometry>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <list>

#include <uav_detect/DepthMapParamsConfig.h>
#include <mrs_lib/ParamLoader.h>

  /* tf2_to_eigen - helper function to convert tf2::Transform to Eigen::Affine3d *//*//{*/
  Eigen::Affine3d tf2_to_eigen(const tf2::Transform& tf2_t)
  {
    Eigen::Affine3d eig_t;
    for (int r_it = 0; r_it < 3; r_it++)
      for (int c_it = 0; c_it < 3; c_it++)
        eig_t(r_it, c_it) = tf2_t.getBasis()[r_it][c_it];
    eig_t(0, 3) = tf2_t.getOrigin().getX();
    eig_t(1, 3) = tf2_t.getOrigin().getY();
    eig_t(2, 3) = tf2_t.getOrigin().getZ();
    return eig_t;
  }/*//}*/

#endif //  MAIN_H
