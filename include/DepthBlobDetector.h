#ifndef DEPTHBLOBDETECTOR_H
#define DEPTHBLOBDETECTOR_H

#include "main.h"
#include <uav_detect/DetectionParamsConfig.h>

namespace dbd
{

struct Blob
{
  double confidence;
  cv::Point2d location;
  double radius;
  double avg_depth;
  double circularity;
  double convexity;
  double angle;
  uint32_t area;
  double inertia;
  std::vector<std::vector<cv::Point> > contours;
};

struct Params
{
  // Filter by color
  bool filter_by_color;
  int min_depth;
  int max_depth;
  // Filter by area
  bool filter_by_area;
  int min_area;
  int max_area;
  // Filter by circularity
  bool filter_by_circularity;
  double min_circularity;
  double max_circularity;
  // Filter by convexity
  bool filter_by_convexity;
  double min_convexity;
  double max_convexity;
  // Filter by orientation
  bool filter_by_orientation;
  double min_angle;
  double max_angle;
  // Filter by inertia
  bool filter_by_inertia;
  double min_inertia_ratio;
  double max_inertia_ratio;
  // thresholding
  int threshold_step;
  bool use_threshold_width;
  int threshold_width;
  int min_repeatability;
  // Other filtering criterions
  double min_dist_between;

  Params(uav_detect::DetectionParamsConfig cfg);
};

class DepthBlobDetector
{
  public:
    DepthBlobDetector(const Params& parameters);
    void detect(cv::Mat image, std::vector<Blob>& ret_blobs);

  private:
    void findBlobs(cv::Mat binary_image, cv::Mat orig_image, std::vector<Blob>& ret_blobs) const;

  private:
    Params params;
};

}

#endif // DEPTHBLOBDETECTOR_H
