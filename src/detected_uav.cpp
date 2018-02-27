#include "detected_uav.h"

using namespace uav_detect;
using namespace std;

static float gauss(float x, float mu, float sigma)
{
  return 1.0/(sigma*sqrt(2.0*M_PI))*exp(-(x-mu)*(x-mu)/(sigma*sigma*2.0));
}

Detected_UAV::Detected_UAV(uav_detect::Detection det, tf2::Transform camera2world_tf, float IoU_threshold) : _UAV_width(0.5)
{
  _ref_det = det;
  _c2w_tf = camera2world_tf;
  _prob = 0.5;
  _IoU_threshold = IoU_threshold;
  _cur_x = std::numeric_limits<double>::infinity();
  _cur_y = std::numeric_limits<double>::infinity();
  _cur_z = std::numeric_limits<double>::infinity();
}

// Intersection over Union
float Detected_UAV::IoU(const uav_detect::Detection &det1, const uav_detect::Detection &det2)
{
  float det1_l = det1.x_relative-det1.w_relative/2.0;
  float det1_r = det1.x_relative+det1.w_relative/2.0;
  float det1_b = det1.y_relative-det1.h_relative/2.0;
  float det1_t = det1.y_relative+det1.h_relative/2.0;

  float det2_l = det2.x_relative-det2.w_relative/2.0;
  float det2_r = det2.x_relative+det2.w_relative/2.0;
  float det2_b = det2.y_relative-det2.h_relative/2.0;
  float det2_t = det2.y_relative+det2.h_relative/2.0;

  float width  = min(det1_r, det2_r) - max(det1_l, det2_l);
  float height = min(det1_t, det2_t) - max(det1_b, det2_b);

  if (width < 0.0 || height < 0.0)
    return 0.0;

  // Area of Overlap
  float AoO = width*height;
  // Area of Union
  float AoU = det1.w_relative*det1.h_relative + det2.w_relative*det2.h_relative - AoO;

  return AoO/AoU;
}

int Detected_UAV::update(const uav_detect::Detections& new_detections, tf2::Transform camera2world_tf)
{
  _c2w_tf = camera2world_tf;
  // Find matching detection
  double best_IoU = 0.0;
  int best_match_it = -1; // -1 indicates no match found
  int it = 0;
  double cur_IoU = 0.0;

  //cout << "Checking IoUs" << std::endl;
  for (const uav_detect::Detection det : new_detections.detections)
  {
    cur_IoU = IoU(_ref_det, det);
    //cout << cur_IoU << std::endl;
    if (cur_IoU > _IoU_threshold && cur_IoU > best_IoU)
    {
      best_IoU = cur_IoU;
      best_match_it = it;
    }
    it++;
  }

  if (best_match_it >= 0)
  {
    Detection best_match = new_detections.detections.at(best_match_it);
    _prob = 0.95*_prob + 0.04*gauss(best_IoU, 0.8, 0.4) + 0.01*best_match.probability;
//    float total_area = (best_match.w_relative*best_match.h_relative)
//                     + (_ref_det.w_relative*_ref_det.h_relative);
    //float N_dets = new_detections.detections.size();

//    float p_e_h = gauss(cur_IoU, 0.8, 0.4);
//    //p_h   = 1.0/N_dets;
//    float p_h   = _prob;
//    float p_e   = total_area;
//    _prob       = (p_e_h * p_h)/p_e;

    _ref_det.x_relative = 0.9*_ref_det.x_relative + 0.1*best_match.x_relative;
    _ref_det.y_relative = 0.9*_ref_det.y_relative + 0.1*best_match.y_relative;
    _ref_det.w_relative = 0.9*_ref_det.w_relative + 0.1*best_match.w_relative;
    _ref_det.h_relative = 0.9*_ref_det.h_relative + 0.1*best_match.h_relative;

    image_geometry::PinholeCameraModel cam_model;
    cam_model.fromCameraInfo(new_detections.camera_info);

    int w_used = new_detections.w_used;
    int h_used = new_detections.h_used;
    int cam_image_w = new_detections.camera_info.width;
    int cam_image_h = new_detections.camera_info.height;

    int px_x = (int)round(
                      (cam_image_w-w_used)/2.0 +  // offset between the detection rectangle and camera image
                      (_ref_det.x_relative)*w_used);
    int px_y = (int)round(
                      (cam_image_h-h_used)/2.0 +  // offset between the detection rectangle and camera image
                      (_ref_det.y_relative)*h_used);
    int px_w = (int)round(_ref_det.w_relative*w_used);
    cv::Point2d center_pt = cam_model.rectifyPoint(cv::Point2d(px_x, px_y));
    cv::Point2d left_pt = cam_model.rectifyPoint(cv::Point2d(px_x-px_w/2, px_y));
    cv::Point2d right_pt = cam_model.rectifyPoint(cv::Point2d(px_x+px_w/2, px_y));
//    cv::Point2d center_pt(px_x, px_y);
//    cv::Point2d left_pt(px_x-px_w, px_y);
//    cv::Point2d right_pt(px_x+px_w, px_y);

    cv::Point3d ray_vec = cam_model.projectPixelTo3dRay(center_pt);
    cv::Point3d left_vec = cam_model.projectPixelTo3dRay(left_pt);
    cv::Point3d right_vec = cam_model.projectPixelTo3dRay(right_pt);

    float proj_width = sqrt((left_vec.x-right_vec.x)*(left_vec.x-right_vec.x)
                           +(left_vec.y-right_vec.y)*(left_vec.y-right_vec.y));
                           //+(left_vec.z-right_vec.z)*(left_vec.z-right_vec.z)); // z coordinate should be 1.0
//    float proj_width = _ref_det.w_relative;
    // cout << "Projection width: " << proj_width << ", x-focal length: " << cam_model.fx() << std::endl;
    cout << "Max. camera width: " << cam_image_w << std::endl;
    cout << "Used camera width: " << w_used << std::endl;
    cout << "Projection width: " << proj_width << std::endl;
    float est_dist = _UAV_width/proj_width;
    tf2::Vector3 cur_pt(ray_vec.x*est_dist, ray_vec.y*est_dist, est_dist);
    cur_pt = _c2w_tf*cur_pt; // transform to world coordinate system

    _cur_x = cur_pt.getX();
    _cur_y = cur_pt.getY();
    _cur_z = cur_pt.getZ();
  } else
  {
//    float area  = _ref_det.w_relative*_ref_det.h_relative;
//    float p_e_h = 0.2;
//    //p_h   = 1.0/N_dets;
//    float p_h   = _prob;
//    float p_e   = area;
//    _prob       = (p_e_h * p_h)/p_e;
  }

  return best_match_it;
}
