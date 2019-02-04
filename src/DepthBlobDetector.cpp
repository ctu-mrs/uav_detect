#include "DepthBlobDetector.h"
/* #define DEBUG_BLOB_DETECTOR */
#define DEBUG_PARAMS

using namespace cv;
using namespace std;
using namespace dbd;

DepthBlobDetector::DepthBlobDetector(const uav_detect::DetectionParamsConfig& cfg, uint16_t unknown_pixel_value)
  : params(cfg), m_unknown_pixel_value(unknown_pixel_value)
{}

void DepthBlobDetector::update_params(const uav_detect::DetectionParamsConfig& cfg)
{
  params.set_from_cfg(cfg);
}

/* median() method //{ */
double DepthBlobDetector::median(cv::Mat image, cv::Mat mask, uint32_t& n_known_pixels) const
{
  vector<uint16_t> vals;
  vals.reserve(image.rows*image.cols);

  for (int row_it = 0; row_it < image.rows; row_it++)
  {
    for (int col_it = 0; col_it < image.cols; col_it++)
    {
      if (mask.at<uint8_t>(row_it, col_it))
      {
        uint16_t cur_val = image.at<uint16_t>(row_it, col_it);
        if (cur_val != m_unknown_pixel_value)
          vals.push_back(cur_val);
      }
    }
  }

  n_known_pixels = vals.size();
  if (vals.empty())
    return std::numeric_limits<double>::quiet_NaN();
  nth_element(vals.begin(), vals.begin()+vals.size()/2, vals.end());
  return vals.at(vals.size()/2);
}

/* double DepthBlobDetector::median(cv::Mat image, std::vector<cv::Point> points, uint32_t& n_known_pixels) const */
/* { */
/*   vector<uint16_t> vals; */
/*   vals.reserve(points.size()); */

/*   for (const auto& pt : points) */
/*   { */
/*     uint16_t cur_val = image.at<uint16_t>(pt); */
/*     if (cur_val != m_unknown_pixel_value) */
/*       vals.push_back(cur_val); */
/*   } */

/*   n_known_pixels = vals.size(); */
/*   if (vals.empty()) */
/*     return std::numeric_limits<double>::quiet_NaN(); */
/*   nth_element(vals.begin(), vals.begin()+vals.size()/2, vals.end()); */
/*   return vals.at(vals.size()/2); */
/* } */
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
  max_area_diff = cfg.max_area_diff;
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
  min_known_pixels_ratio = cfg.min_known_pixels_ratio;
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
#ifndef DEBUG_PARAMS
    if (params.filter_by_circularity)
#endif
    {
      const double area = moms.m00;
      const double perimeter = arcLength(Mat(contours[contourIdx]), true);
      const double ratio = 4 * CV_PI * area / (perimeter * perimeter);
      blob.circularity = ratio;
      if (
#ifdef DEBUG_PARAMS
          params.filter_by_circularity &&
#endif
          (ratio < params.min_circularity || ratio >= params.max_circularity))
        continue;
    }
    //}

    /* Filter by orientation //{ */
#ifndef DEBUG_PARAMS
    if (params.filter_by_orientation)
#endif
    {
      constexpr double eps = 1e-3;
      double angle = 0;
      if (abs(moms.mu20 - moms.mu02) > eps)
        angle = abs(0.5 * atan2((2 * moms.mu11), (moms.mu20 - moms.mu02)));
      blob.angle = angle;
      if (
#ifdef DEBUG_PARAMS
          params.filter_by_orientation &&
#endif
          (angle < params.min_angle || angle > params.max_angle))
        continue;
    }
    //}
    
    /* Filter by intertia //{ */
#ifndef DEBUG_PARAMS
#endif
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
#ifndef DEBUG_PARAMS
    if (params.filter_by_convexity)
#endif
    {
      std::vector<Point> hull;
      convexHull(Mat(contours[contourIdx]), hull);
      const double area = contourArea(Mat(contours[contourIdx]));
      const double hullArea = contourArea(Mat(hull));
      if (fabs(hullArea) < DBL_EPSILON)
        continue;
      const double ratio = area / hullArea;
      blob.convexity = ratio;
      if (
#ifdef DEBUG_PARAMS
          params.filter_by_convexity &&
#endif
          (ratio < params.min_convexity || ratio >= params.max_convexity))
        continue;
    }
    //}

    uint32_t n_known_pixels; // filled out in the median function when calculating blob color (depth)
    /* Filter by color (depth) //{ */
#ifndef DEBUG_PARAMS
    if (params.filter_by_color)
#endif
    {
      const Rect roi = boundingRect(contours[contourIdx]);
      const Mat mask(roi.size(), CV_8UC1);
      drawContours(mask, contours, contourIdx, Scalar(1), CV_FILLED, LINE_8, noArray(), INT_MAX, -roi.tl());
      erode(mask, mask, Mat()); // remove the contour itself from the mask (leave only inner area)
      const double avg_color = median(orig_image(roi), mask, n_known_pixels);
      /* const double avg_color = median(orig_image, contours[contourIdx], n_known_pixels); */

      blob.avg_depth = avg_color / 1000.0;

      if (
#ifdef DEBUG_PARAMS
          params.filter_by_color &&
#endif
          (std::isnan(avg_color) || avg_color < params.min_depth || avg_color > params.max_depth))
        continue;
    }
    //}

    /* Filter by number of known pixels //{ */
#ifndef DEBUG_PARAMS
    if (params.filter_by_known_pixels)
#endif
    {
#ifndef DEBUG_PARAMS
      if (!params.filter_by_color)
      {
        const Rect roi = boundingRect(contours[contourIdx]);
        Mat known_mask;
        cv::compare(orig_image(roi), m_unknown_pixel_value, known_mask, cv::CMP_NE);
        Mat blob_mask(roi.size(), CV_8UC1);
        drawContours(blob_mask, contours, contourIdx, Scalar(1), CV_FILLED, LINE_8, noArray(), INT_MAX, -roi.tl());
        erode(blob_mask, blob_mask, Mat()); // remove the contour itself from the mask (leave only inner area)
        
        n_known_pixels = cv::sum(blob_mask & known_mask)[0];
      }
#endif
      blob.known_pixels_ratio = n_known_pixels/blob.area;
      if (
#ifdef DEBUG_PARAMS
          params.filter_by_known_pixels &&
#endif
          blob.known_pixels_ratio < params.min_known_pixels_ratio)
        continue;
    }
    //}

    /* Calculate blob radius //{ */
#ifndef DEBUG_PARAMS
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
#endif
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
  assert(params.min_repeatability != 0);

  // detect blobs in thresholded images
  size_t n_steps = floor((params.max_depth - params.min_depth)/params.threshold_step);
  // no need for mutex since each iteration only writes to its own preallocated mem
  std::vector<std::vector<Blob>> thresh_blobs(n_steps);
#pragma omp parallel for
  for (size_t thresh_it = 0; thresh_it < n_steps; thresh_it++)
  {
    int thresh = params.min_depth + thresh_it*params.threshold_step;
    cv::Mat binary_image;

    inRange(image, thresh, std::numeric_limits<uint16_t>::max(), binary_image);

/* #ifdef DEBUG_BLOB_DETECTOR //{ */
/*     ROS_INFO("[%s]: using threshold %u", ros::this_node::getName().c_str(), thresh); */
/*     if (params.use_threshold_width) */
/*       cur_depth = (thresh + params.threshold_width) / 1000.0; */
/*     else */
/*       cur_depth = thresh / 1000.0; */
/* #endif //} */

    std::vector<Blob> cur_blobs;
    findBlobs(binary_image, image, mask_image, cur_blobs);

    thresh_blobs.at(thresh_it) = cur_blobs;
  }

  // group blobs to groups
  // this has to be done sequentially (add only new blobs to the vector, associate
  // similar blobs to existing ones)
  std::vector<std::vector<Blob>> blob_groups;
  for (const auto& cur_blobs : thresh_blobs)
  {
    std::vector<std::vector<Blob>> new_blobs;
    new_blobs.reserve(cur_blobs.size());
    for (size_t i = 0; i < cur_blobs.size(); i++)
    {
      bool isNew = true;
      for (size_t j = 0; j < blob_groups.size(); j++)
      {
        double dist = norm(blob_groups[j][blob_groups[j].size() / 2].location - cur_blobs[i].location);
        isNew = dist >= params.min_dist_between && dist >= blob_groups[j][blob_groups[j].size() / 2].radius && dist >= cur_blobs[i].radius;
        if (!isNew)
        {
          blob_groups[j].push_back(cur_blobs[i]);

          size_t k = blob_groups[j].size() - 1;
          while (k > 0 && blob_groups[j][k].radius < blob_groups[j][k - 1].radius)
          {
            blob_groups[j][k] = blob_groups[j][k - 1];
            k--;
          }
          blob_groups[j][k] = cur_blobs[i];

          break;
        }
      }
      if (isNew)
        new_blobs.push_back(std::vector<Blob>(1, cur_blobs[i]));
    }
    std::copy(new_blobs.begin(), new_blobs.end(), std::back_inserter(blob_groups));
  }

  // calculate common blob group characteristics
  ret_blobs.resize(blob_groups.size());
#pragma omp parallel for
  for (size_t i = 0; i < blob_groups.size(); i++)
  {
    vector<Blob> cur_blobs = blob_groups[i];
    if (cur_blobs.size() < (size_t)params.min_repeatability)
      continue;

    Point2d sumPoint(0, 0);
    double normalizer = 0;
    vector<vector<Point> > contours;
    contours.reserve(cur_blobs.size());
    uint32_t prev_area = cur_blobs.at(0).area;
    uint32_t max_area_diff = 0;
    for (const auto& cur_blob : cur_blobs)
    {
      uint32_t cur_area_diff = abs(cur_blob.area - prev_area);
      if (cur_area_diff > uint32_t(params.max_area_diff))
        break; // area of the following blobs can only increase - they can safely be skipped

      if (cur_area_diff > max_area_diff)
        max_area_diff = cur_area_diff;
      sumPoint += cur_blob.confidence * cur_blob.location;
      normalizer += cur_blob.confidence;
      contours.push_back(cur_blob.contours[0]);
      prev_area = cur_blob.area;
    }
    sumPoint *= (1. / normalizer);
    Blob result_blob;
    result_blob.confidence = normalizer / cur_blobs.size();
    result_blob.location = sumPoint;
    result_blob.radius = cur_blobs[cur_blobs.size() / 2].radius;
    result_blob.max_area_diff = max_area_diff;
    result_blob.area = cur_blobs[cur_blobs.size() / 2].area;
    result_blob.circularity = cur_blobs[cur_blobs.size() / 2].circularity;
    result_blob.convexity = cur_blobs[cur_blobs.size() / 2].convexity;
    result_blob.avg_depth = cur_blobs[cur_blobs.size() / 2].avg_depth;
    result_blob.known_pixels_ratio = cur_blobs[cur_blobs.size() / 2].known_pixels_ratio;
    result_blob.angle = cur_blobs[cur_blobs.size() / 2].angle;
    result_blob.inertia = cur_blobs[cur_blobs.size() / 2].inertia;
    result_blob.contours = contours;
    ret_blobs[i] = result_blob;
  }

  // erase empty blob groups
  for (auto it = ret_blobs.begin(); it != ret_blobs.end(); it++)
  {
    if (it->contours.size() < (size_t)params.min_repeatability)
    {
      it = ret_blobs.erase(it);
      it--;
    }
  }

}
//}
