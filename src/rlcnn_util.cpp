#include "rlcnn_util.h"

using namespace Eigen;
using namespace std;

namespace rlcnn
{
  /* Globals used for calculations */
  Eigen::Affine3d c2w_tf;
  image_geometry::PinholeCameraModel camera_model;
  ocam_model oc_model;// OCamCalib camera model
  int w_camera, h_camera, w_used, h_used;

  void update_camera_info(const uav_detect::Detections& dets_msg)
  {
    camera_model.fromCameraInfo(dets_msg.camera_info);
    w_camera = dets_msg.camera_info.width;
    h_camera = dets_msg.camera_info.height;
    w_used = dets_msg.w_used;
    h_used = dets_msg.h_used;
  }


  /* calculate_direction_pinhole - Calculates a direction vector of the detection in camera coordinates using pinhole camera model //{ */
  Eigen::Vector3d calculate_direction_pinhole(double px_x, double px_y)
  {
    // Calculate pixel position of the detection
    cv::Point2d det_pt(px_x, px_y);
    det_pt = camera_model.rectifyPoint(det_pt);  // do not forget to rectify the points!
    cv::Point3d cv_vec = camera_model.projectPixelTo3dRay(det_pt);
    return Vector3d(cv_vec.x, cv_vec.y, cv_vec.z).normalized(); 
  }
  //}

  /* calculate_direction_ocam - Calculates a direction vector of the detection in camera coordinates using OCamCalib camera model //{ */
  Eigen::Vector3d calculate_direction_ocam(double px_x, double px_y)
  {
    double pt2d[2] = {px_y, px_x};
    double pt3d[3] = {0.0, 0.0, 0.0};
    // project the left point of the UAV using OCamCalib camera calibration to 3D
    cam2world(pt3d, pt2d, &oc_model);
    return Vector3d(pt3d[1], pt3d[0], -pt3d[2]).normalized();
  }
  //}

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
}
