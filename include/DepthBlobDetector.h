#ifndef DEPTHBLOBDETECTOR_H
#define DEPTHBLOBDETECTOR_H

#include "main.h"
#include <uav_detect/DetectionParamsConfig.h>

namespace dbd
{

/* struct Blob //{*/
struct Blob
{
  int32_t id = 0;
  double confidence = 0;
  cv::Point2d location;
  double radius = 0;
  uint32_t area = 0;
  uint32_t max_area_diff = 0;
  double circularity = 0;
  double angle = 0;
  double inertia = 0;
  double convexity = 0;
  double avg_depth = 0;
  double known_pixels_ratio = 0;
  std::vector<std::vector<cv::Point> > contours;
};
/*//}*/

/* struct Params //{*/
struct Params
{
  // Filter by area
  bool filter_by_area;
  int min_area;
  int max_area;
  int max_area_diff;
  // Filter by circularity
  bool filter_by_circularity;
  double min_circularity;
  double max_circularity;
  // Filter by orientation
  bool filter_by_orientation;
  double min_angle;
  double max_angle;
  // Filter by inertia
  bool filter_by_inertia;
  double min_inertia_ratio;
  double max_inertia_ratio;
  // Filter by convexity
  bool filter_by_convexity;
  double min_convexity;
  double max_convexity;
  // Filter by color
  bool filter_by_color;
  int min_depth;
  int max_depth;
  // Filter by known pixels
  bool filter_by_known_pixels;
  double min_known_pixels_ratio;
  // thresholding
  int threshold_step;
  bool use_threshold_width;
  int threshold_width;
  int min_repeatability;
  // Other filtering criterions
  double min_dist_between;

  Params(){};
  Params(const uav_detect::DetectionParamsConfig& cfg);
  void set_from_cfg(const uav_detect::DetectionParamsConfig& cfg);
};
/*//}*/

class DepthBlobDetector
{
  public:
    DepthBlobDetector(){};
    DepthBlobDetector(const uav_detect::DetectionParamsConfig& cfg, uint16_t unknown_pixel_value);
    void detect(cv::Mat image, cv::Mat mask_image, std::vector<Blob>& ret_blobs);
    void update_params(const uav_detect::DetectionParamsConfig& cfg);

  private:
    double median(cv::Mat image, cv::Mat mask, uint32_t& n_known_pixels) const;
    double median(cv::Mat image, std::vector<cv::Point> points, uint32_t& n_known_pixels) const;
    void findBlobs(cv::Mat binary_image, cv::Mat orig_image, cv::Mat mask_image, std::vector<Blob>& ret_blobs) const;

  private:
    Params params;
    uint16_t m_unknown_pixel_value;
};

}

#endif // DEPTHBLOBDETECTOR_H
