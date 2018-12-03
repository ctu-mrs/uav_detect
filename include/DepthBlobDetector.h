#ifndef DEPTHBLOBDETECTOR_H
#define DEPTHBLOBDETECTOR_H

#include "main.h"
#include <uav_detect/DetectionParamsConfig.h>

namespace dbd
{

/* struct Blob //{*/
struct Blob
{
  int32_t id;
  double confidence;
  cv::Point2d location;
  double radius;
  uint32_t area;
  double circularity;
  double angle;
  double inertia;
  double convexity;
  double avg_depth;
  uint32_t known_pixels;
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
  int min_known_pixels;
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
    DepthBlobDetector(const uav_detect::DetectionParamsConfig& cfg);
    void detect(cv::Mat image, cv::Mat mask_image, std::vector<Blob>& ret_blobs);
    void update_params(const uav_detect::DetectionParamsConfig& cfg);

  private:
    void findBlobs(cv::Mat binary_image, cv::Mat orig_image, cv::Mat mask_image, std::vector<Blob>& ret_blobs) const;

  private:
    Params params;
};

}

#endif // DEPTHBLOBDETECTOR_H
