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

namespace rlcnn
{
  /* Globals used for calculations */
  extern Eigen::Affine3d c2w_tf;
  extern image_geometry::PinholeCameraModel camera_model;
  extern int w_camera, h_camera, w_used, h_used;

  void update_camera_info(const uav_detect::Detections& dets_msg);

  /* Calculates a projection half-line of the detection */
  void calculate_3D_projection(
      const uav_detect::Detection& det,
      Eigen::Vector3d& out_point,
      Eigen::Vector3d& out_vector
      );

  /* tf2_to_eigen - helper function to convert tf2::Transform to Eigen::Affine3d */
  Eigen::Affine3d tf2_to_eigen(const tf2::Transform& tf2_t);
}

#endif // RLCNN_UTIL_H
