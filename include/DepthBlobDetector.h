#ifndef DEPTHBLOBDETECTOR_H
#define DEPTHBLOBDETECTOR_H

#include "main.h"

namespace dbd
{

struct Blob
{
  double confidence;
  cv::Point2d location;
  double radius;
  double avg_depth;
};

struct Params
{
  // Filter by color
  bool filter_by_color;
  uint16_t min_depth;
  uint16_t max_depth;
  // Filter by area
  bool filter_by_area;
  uint32_t min_area;
  uint32_t max_area;
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
  uint16_t min_threshold;
  uint16_t max_threshold;
  uint16_t threshold_step;
  uint16_t threshold_width;
  uint16_t min_repeatability;
  // Other filtering criterions
  double min_dist_between;
};

class DepthBlobDetector
{
  public:
    DepthBlobDetector(const Params& parameters);
    void detect(cv::Mat image, std::vector<Blob>& blobs, cv::Mat mask = cv::Mat());

  private:
    void findBlobs(cv::Mat image, cv::Mat binaryImage, std::vector<Blob>& blobs) const;

  private:
    Params params;
};

}

#endif // DEPTHBLOBDETECTOR_H
