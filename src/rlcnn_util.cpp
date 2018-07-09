#include "rlcnn_util.h"

using namespace Eigen;
using namespace std;

namespace rlcnn
{
  /* Globals used for calculations */
  Eigen::Affine3d c2w_tf;
  image_geometry::PinholeCameraModel camera_model;
  int w_camera, h_camera, w_used, h_used;

  void update_camera_info(const uav_detect::Detections& dets_msg)
  {
    camera_model.fromCameraInfo(dets_msg.camera_info);
    w_camera = dets_msg.camera_info.width;
    h_camera = dets_msg.camera_info.height;
    w_used = dets_msg.w_used;
    h_used = dets_msg.h_used;
  }


  /* Calculates a projection half-line of the detection *//*//{*/
  void calculate_3D_projection(
      const uav_detect::Detection& det,
      Eigen::Vector3d& out_point,
      Eigen::Vector3d& out_vector
      )
  {
    // Calculate pixel position of the detection
    int px_x = (int)round(
                      (w_camera-w_used)/2.0 +  // offset between the detection rectangle and camera image
                      (det.x_relative)*w_used);
    int px_y = (int)round(
                      (h_camera-h_used)/2.0 +  // offset between the detection rectangle and camera image
                      (det.y_relative)*h_used);
    cv::Point2d center_pt = camera_model.rectifyPoint(cv::Point2d(px_x, px_y));

    // Calculate projections of the center, left and right points of the detected bounding box
    cv::Point3d ray_vec = camera_model.projectPixelTo3dRay(center_pt);

    double ray_vec_norm = sqrt(ray_vec.x*ray_vec.x + ray_vec.y*ray_vec.y + ray_vec.z*ray_vec.z);
    out_point << 0.0, 0.0, 0.0;
    out_vector << ray_vec.x/ray_vec_norm, ray_vec.y/ray_vec_norm, ray_vec.z/ray_vec_norm;
    out_point = c2w_tf*out_point;
    out_vector = c2w_tf*out_vector - out_point;
  }/*//}*/

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
