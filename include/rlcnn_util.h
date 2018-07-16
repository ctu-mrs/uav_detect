#ifndef RLCNN_UTIL_H
#define RLCNN_UTIL_H

#include <Eigen/Dense>

#include <uav_detect/Detection.h>
#include <uav_detect/Detections.h>

#include <image_geometry/pinhole_camera_model.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include "ocam_functions.h"

namespace rlcnn
{
  /* Globals used for calculations */
  extern Eigen::Affine3d c2w_tf;
  extern image_geometry::PinholeCameraModel camera_model;
  extern ocam_model oc_model;  // OCamCalib camera model
  extern int w_camera, h_camera, w_used, h_used;

  void update_camera_info(const uav_detect::Detections& dets_msg);

  /* calculate_direction_pinhole - Calculates a direction vector of the detection in camera coordinates using pinhole camera model */
  Eigen::Vector3d calculate_direction_pinhole(double px_x, double px_y);

  /* calculate_direction_ocam - Calculates a direction vector of the detection in camera coordinates using OCamCalib camera model */
  Eigen::Vector3d calculate_direction_ocam(double px_x, double px_y);

  /* tf2_to_eigen - helper function to convert tf2::Transform to Eigen::Affine3d */
  Eigen::Affine3d tf2_to_eigen(const tf2::Transform& tf2_t);
}

#endif // RLCNN_UTIL_H
