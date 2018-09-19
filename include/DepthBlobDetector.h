#ifndef DEPTHBLOBDETECTOR_H
#define DEPTHBLOBDETECTOR_H

#include "main.h"

namespace dbd
{

struct Center
{
  double confidence;
  cv::Point2d location;
  double radius;
};

struct Params
{
  // Filter by color
  bool filter_by_color;
  double color;
  // Filter by area
  bool filter_by_area;
  double min_area;
  double max_area;
  // Filter by circularity
  bool filter_by_circularity;
  double min_circularity;
  double max_circularity;
  // Filter by convexity
  bool filter_by_convexity;
  double min_convexity;
  double max_convexity;
  // Filter by inertia
  bool filter_by_inertia;
  double min_inertia_ratio;
  double max_inertia_ratio;
  // thresholding
  double min_threshold;
  double max_threshold;
  double threshold_step;
  unsigned min_repeatability;
  // Other filtering criterions
  double min_dist_between;
};

class DepthBlobDetector
{
  public:
    DepthBlobDetector(const Params& parameters);
    void detect(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat mask = cv::Mat());

  private:
    void findBlobs(cv::Mat image, cv::Mat binaryImage, std::vector<Center>& centers) const;

  private:
    Params params;
};

}

#endif // DEPTHBLOBDETECTOR_H
