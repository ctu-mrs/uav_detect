#include "DepthBlobDetector.h"
/* #define DEBUG_BLOB_DETECTOR */

using namespace cv;
using namespace std;
using namespace dbd;

DepthBlobDetector::DepthBlobDetector(const uav_detect::DetectionParamsConfig& cfg)
  : params(cfg)
{}

void DepthBlobDetector::update_params(const uav_detect::DetectionParamsConfig& cfg)
{
  params.set_from_cfg(cfg);
}

/* median function //{ */
double median(cv::Mat image, cv::Mat mask, uint32_t& n_known_pixels)
{
  vector<uint16_t> vals;
  vals.reserve(image.rows*image.cols);
  n_known_pixels = 0;

  for (int row_it = 0; row_it < image.rows; row_it++)
  {
    for (int col_it = 0; col_it < image.cols; col_it++)
    {
      if (mask.at<uint8_t>(row_it, col_it))
      {
        uint16_t cur_val = image.at<uint16_t>(row_it, col_it);
        if (cur_val != 0)
          n_known_pixels++;
        vals.push_back(cur_val);
      }
    }
  }

  if (vals.empty())
    return std::numeric_limits<double>::quiet_NaN();
  nth_element(vals.begin(), vals.begin()+vals.size()/2, vals.end());
  return vals.at(vals.size()/2);
}

double median(cv::Mat image, std::vector<cv::Point> points, uint32_t& n_known_pixels)
{
  vector<uint16_t> vals;
  vals.reserve(points.size());
  n_known_pixels = 0;

  for (const auto& pt : points)
  {
    uint16_t cur_val = image.at<uint16_t>(pt);
    if (cur_val != 0)
      n_known_pixels++;
    vals.push_back(cur_val);
  }

  if (vals.empty())
    return std::numeric_limits<double>::quiet_NaN();
  nth_element(vals.begin(), vals.begin()+vals.size()/2, vals.end());
  return vals.at(vals.size()/2);
}
//}

#ifdef DEBUG_BLOB_DETECTOR //{
double cur_depth;
#endif //}

/* Params methods //{ */
Params::Params(const uav_detect::DetectionParamsConfig& cfg)
{
  set_from_cfg(cfg);
}
void Params::set_from_cfg(const uav_detect::DetectionParamsConfig& cfg)
{
  use_threshold_width = cfg.use_threshold_width;
  threshold_step = cfg.threshold_step;
  threshold_width = cfg.threshold_width;
  // Filter by area
  filter_by_area = cfg.filter_by_area;
  min_area = cfg.min_area;
  max_area = cfg.max_area;
  // Filter by circularity
  filter_by_circularity = cfg.filter_by_circularity;
  min_circularity = cfg.min_circularity;
  max_circularity = cfg.max_circularity;
  // Filter by orientation
  filter_by_orientation = cfg.filter_by_orientation;
  min_angle = cfg.min_angle;
  max_angle = cfg.max_angle;
  // Filter by inertia
  filter_by_inertia = cfg.filter_by_inertia;
  min_inertia_ratio = cfg.min_inertia_ratio;
  max_inertia_ratio = cfg.max_inertia_ratio;
  // Filter by convexity
  filter_by_convexity = cfg.filter_by_convexity;
  min_convexity = cfg.min_convexity;
  max_convexity = cfg.max_convexity;
  // Filter by color
  filter_by_color = cfg.filter_by_color;
  min_depth = cfg.min_depth;
  max_depth = cfg.max_depth;
  // Filter by known points
  filter_by_known_pixels = cfg.filter_by_known_pixels;
  min_known_pixels = cfg.min_known_pixels;
  // Other filtering criterions
  min_dist_between = cfg.min_dist_between;
  min_repeatability = cfg.min_repeatability;
}
//}

/* method void DepthBlobDetector::findBlobs(cv::Mat image, cv::Mat binaryImage, std::vector<Blob>& blobs) const //{ */
/* inspired by https://github.com/opencv/opencv/blob/3.4/modules/features2d/src/blobdetector.cpp */
void DepthBlobDetector::findBlobs(cv::Mat binary_image, cv::Mat orig_image, cv::Mat mask_image, std::vector<Blob>& ret_blobs) const
{
  ret_blobs.clear();

  std::vector<std::vector<Point>> contours;
  findContours(binary_image, contours, RETR_LIST, CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR //{
  Mat keypointsImage;
  cvtColor(binaryImage, keypointsImage, CV_GRAY2RGB);

  drawContours(keypointsImage, contours, -1, Scalar(0, 255, 0));
  imshow("opencv_debug", keypointsImage);
#endif //}

  for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
  {
    Blob blob;
    blob.confidence = 1;
    Moments moms = moments(Mat(contours[contourIdx]));
    blob.area = moms.m00;
    if (moms.m00 == 0.0)
      continue;

    blob.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);
    if (!mask_image.empty() && !mask_image.at<uint8_t>(blob.location))
      continue;

    /* Filter by area //{ */
    if (params.filter_by_area)
    {
      const double area = moms.m00;
      if (area < params.min_area || area >= params.max_area)
        continue;
    }
    //}

    /* Filter by circularity //{ */
    {
      const double area = moms.m00;
      const double perimeter = arcLength(Mat(contours[contourIdx]), true);
      const double ratio = 4 * CV_PI * area / (perimeter * perimeter);
      blob.circularity = ratio;
      if (params.filter_by_circularity && (ratio < params.min_circularity || ratio >= params.max_circularity))
        continue;
    }
    //}

    /* Filter by orientation //{ */
    {
      constexpr double eps = 1e-3;
      double angle = 0;
      if (abs(moms.mu20 - moms.mu02) > eps)
        angle = abs(0.5 * atan2((2 * moms.mu11), (moms.mu20 - moms.mu02)));
      blob.angle = angle;
      if (params.filter_by_orientation && (angle < params.min_angle || angle > params.max_angle))
        continue;
    }
    //}
    
    /* Filter by intertia //{ */
    {
      const double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
      constexpr double eps = 1e-2;
      double ratio;
      if (denominator > eps)
      {
        const double cosmin = (moms.mu20 - moms.mu02) / denominator;
        const double sinmin = 2 * moms.mu11 / denominator;
        const double cosmax = -cosmin;
        const double sinmax = -sinmin;

        const double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
        const double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
        ratio = imin / imax;
      } else
      {
        ratio = 1;
      }

      blob.inertia = ratio;
      if (params.filter_by_inertia && (ratio < params.min_inertia_ratio || ratio >= params.max_inertia_ratio))
        continue;

      blob.confidence = ratio * ratio;
    }
    //}

    /* Filter by convexity //{ */
    {
      std::vector<Point> hull;
      convexHull(Mat(contours[contourIdx]), hull);
      const double area = contourArea(Mat(contours[contourIdx]));
      const double hullArea = contourArea(Mat(hull));
      if (fabs(hullArea) < DBL_EPSILON)
        continue;
      const double ratio = area / hullArea;
      blob.convexity = ratio;
      if (params.filter_by_convexity && (ratio < params.min_convexity || ratio >= params.max_convexity))
        continue;
    }
    //}

    uint32_t n_known_pixels; // filled out in the median function when calculating blob color (depth)
    /* Filter by color (depth) //{ */
    {
      /* const Rect roi = boundingRect(contours[contourIdx]); */
      /* const Mat mask(roi.size(), CV_8UC1); */
      /* drawContours(mask, contours, contourIdx, Scalar(1), CV_FILLED, LINE_8, noArray(), INT_MAX, -roi.tl()); */
      /* const double avg_color = median(orig_image(roi), mask, n_known_pixels); */
      const double avg_color = median(orig_image, contours[contourIdx], n_known_pixels);

      blob.avg_depth = avg_color / 1000.0;

      if (params.filter_by_color && (std::isnan(avg_color) || avg_color < params.min_depth || avg_color > params.max_depth))
        continue;
    }
    //}

    /* Filter by number of known pixels //{ */
    {
      blob.known_pixels = n_known_pixels;
      if (params.filter_by_known_pixels && n_known_pixels < (uint32_t)params.min_known_pixels)
        continue;
    }
    //}

    /* Calculate blob radius //{ */
    {
      std::vector<double> dists;
      for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
      {
        Point2d pt = contours[contourIdx][pointIdx];
        dists.push_back(norm(blob.location - pt));
      }
      std::nth_element(dists.begin(), dists.begin() + dists.size()/2, dists.end());
      double post_median = dists.at(dists.size()/2);
      std::nth_element(dists.begin(), dists.begin() + (dists.size()-1)/2, dists.end());
      double pre_median = dists.at((dists.size()-1)/2);
      blob.radius = (pre_median + post_median) / 2.;
    }
    //}

    blob.contours.push_back(contours[contourIdx]);

    ret_blobs.push_back(blob);

#ifdef DEBUG_BLOB_DETECTOR //{
    drawContours(keypointsImage, contours, contourIdx, Scalar(0, blob.avg_depth * 255.0 / params.max_depth, 0), CV_FILLED, LINE_4);
    circle(keypointsImage, blob.location, 1, Scalar(0, 0, 255), 1);
#endif //}
  }
#ifdef DEBUG_BLOB_DETECTOR //{
  cv::putText(keypointsImage, string("cur_depth: ") + to_string(cur_depth), Point(0, 120), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
  imshow("opencv_debug", keypointsImage);
  waitKey(50);
#endif //}
}
//}

/* method void DepthBlobDetector::detect(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat mask) //{ */
/* inspired by https://github.com/opencv/opencv/blob/3.4/modules/features2d/src/blobdetector.cpp */
void DepthBlobDetector::detect(cv::Mat image, cv::Mat mask_image, std::vector<Blob>& ret_blobs)
{
  ret_blobs.clear();
  assert(params.min_repeatability != 0);

  std::vector<std::vector<Blob>> blobs;
  int thresh_start = params.min_depth;
  if (params.use_threshold_width)
    thresh_start = params.min_depth + params.threshold_width;
  for (int thresh = thresh_start; thresh < params.max_depth; thresh += params.threshold_step)
  {
    Mat binary_image;

    if (params.use_threshold_width)
      inRange(image, thresh - params.threshold_width, thresh, binary_image);
    else
      inRange(image, params.min_depth, thresh, binary_image);

#ifdef DEBUG_BLOB_DETECTOR //{
    ROS_INFO("[%s]: using threshold %u", ros::this_node::getName().c_str(), thresh);
    if (params.use_threshold_width)
      cur_depth = (thresh + params.threshold_width) / 1000.0;
    else
      cur_depth = thresh / 1000.0;
#endif //}

    std::vector<Blob> curBlobs;
    findBlobs(binary_image, image, mask_image, curBlobs);
    std::vector<std::vector<Blob>> newBlobs;
    for (size_t i = 0; i < curBlobs.size(); i++)
    {
      bool isNew = true;
      for (size_t j = 0; j < blobs.size(); j++)
      {
        double dist = norm(blobs[j][blobs[j].size() / 2].location - curBlobs[i].location);
        isNew = dist >= params.min_dist_between && dist >= blobs[j][blobs[j].size() / 2].radius && dist >= curBlobs[i].radius;
        if (!isNew)
        {
          blobs[j].push_back(curBlobs[i]);

          size_t k = blobs[j].size() - 1;
          while (k > 0 && blobs[j][k].radius < blobs[j][k - 1].radius)
          {
            blobs[j][k] = blobs[j][k - 1];
            k--;
          }
          blobs[j][k] = curBlobs[i];

          break;
        }
      }
      if (isNew)
        newBlobs.push_back(std::vector<Blob>(1, curBlobs[i]));
    }
    std::copy(newBlobs.begin(), newBlobs.end(), std::back_inserter(blobs));
  }

  for (size_t i = 0; i < blobs.size(); i++)
  {
    vector<Blob> cur_blobs = blobs[i];
    if (cur_blobs.size() < (size_t)params.min_repeatability)
      continue;
    Point2d sumPoint(0, 0);
    double normalizer = 0;
    vector<vector<Point> > contours;
    contours.reserve(cur_blobs.size());
    for (size_t j = 0; j < cur_blobs.size(); j++)
    {
      sumPoint += cur_blobs[j].confidence * cur_blobs[j].location;
      normalizer += cur_blobs[j].confidence;
      contours.push_back(cur_blobs[j].contours[0]);
    }
    sumPoint *= (1. / normalizer);
    Blob result_blob;
    result_blob.confidence = normalizer / cur_blobs.size();
    result_blob.location = sumPoint;
    result_blob.radius = cur_blobs[cur_blobs.size() / 2].radius;
    result_blob.area = cur_blobs[cur_blobs.size() / 2].area;
    result_blob.circularity = cur_blobs[cur_blobs.size() / 2].circularity;
    result_blob.convexity = cur_blobs[cur_blobs.size() / 2].convexity;
    result_blob.avg_depth = cur_blobs[cur_blobs.size() / 2].avg_depth;
    result_blob.known_pixels = cur_blobs[cur_blobs.size() / 2].known_pixels;
    result_blob.angle = cur_blobs[cur_blobs.size() / 2].angle;
    result_blob.inertia = cur_blobs[cur_blobs.size() / 2].inertia;
    result_blob.contours = contours;
    ret_blobs.push_back(result_blob);
  }

}
//}
