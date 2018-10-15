#include "DepthBlobDetector.h"
/* #define DEBUG_BLOB_DETECTOR */

using namespace cv;
using namespace std;
using namespace dbd;

DepthBlobDetector::DepthBlobDetector(const Params& parameters)
{
  params = parameters;
}

double median(cv::Mat image, cv::Mat mask)
{
  vector<uint16_t> vals;
  vals.reserve(image.rows*image.cols);
  for (int row_it = 0; row_it < image.rows; row_it++)
  {
    for (int col_it = 0; col_it < image.cols; col_it++)
    {
      if (mask.at<uint8_t>(row_it, col_it))
      {
        vals.push_back(image.at<uint16_t>(row_it, col_it));
      }
    }
  }

  if (vals.empty())
    return std::numeric_limits<double>::quiet_NaN();
  nth_element(vals.begin(), vals.begin()+vals.size()/2, vals.end());
  return vals.at(vals.size()/2);
}


#ifdef DEBUG_BLOB_DETECTOR
double cur_depth;
#endif

/* method void DepthBlobDetector::findBlobs(cv::Mat image, cv::Mat binaryImage, std::vector<Blob>& blobs) const //{ */
/* inspired by https://github.com/opencv/opencv/blob/3.4/modules/features2d/src/blobdetector.cpp */
void DepthBlobDetector::findBlobs(cv::Mat image, cv::Mat known_pixels, cv::Mat binaryImage, std::vector<Blob>& blobs) const
{
  blobs.clear();

  std::vector<std::vector<Point>> contours;
  /* findContours(tmpBinaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE); */
  findContours(binaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR
  Mat keypointsImage;
  cvtColor(binaryImage, keypointsImage, CV_GRAY2RGB);

  drawContours(keypointsImage, contours, -1, Scalar(0, 255, 0));
  imshow("opencv_debug", keypointsImage);
#endif

  for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
  {
    Blob blob;
    blob.confidence = 1;
    Moments moms = moments(Mat(contours[contourIdx]));
    blob.area = moms.m00;
    if (moms.m00 == 0.0)
      continue;

    if (params.filter_by_area)
    {
      double area = moms.m00;
      if (area < params.min_area || area >= params.max_area)
        continue;
    }

    {
      double area = moms.m00;
      double perimeter = arcLength(Mat(contours[contourIdx]), true);
      double ratio = 4 * CV_PI * area / (perimeter * perimeter);
      blob.circularity = ratio;
      if (params.filter_by_circularity && (ratio < params.min_circularity || ratio >= params.max_circularity))
        continue;
    }

    {
      constexpr double eps = 1e-3;
      double angle = 0;
      if (abs(moms.mu20 - moms.mu02) > eps)
        angle = abs(0.5 * atan2((2 * moms.mu11), (moms.mu20 - moms.mu02)));
      blob.angle = angle;
      if (params.filter_by_orientation && (angle < params.min_angle || angle > params.max_angle))
        continue;
    }

    {
      double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
      constexpr double eps = 1e-2;
      double ratio;
      if (denominator > eps)
      {
        double cosmin = (moms.mu20 - moms.mu02) / denominator;
        double sinmin = 2 * moms.mu11 / denominator;
        double cosmax = -cosmin;
        double sinmax = -sinmin;

        double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
        double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
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

    {
      std::vector<Point> hull;
      convexHull(Mat(contours[contourIdx]), hull);
      double area = contourArea(Mat(contours[contourIdx]));
      double hullArea = contourArea(Mat(hull));
      if (fabs(hullArea) < DBL_EPSILON)
        continue;
      double ratio = area / hullArea;
      blob.convexity = ratio;
      if (params.filter_by_convexity && (ratio < params.min_convexity || ratio >= params.max_convexity))
        continue;
    }

    // compute blob average depth
    {
      Rect roi = boundingRect(contours[contourIdx]);
      Mat mask(roi.size(), CV_8UC1);
      drawContours(mask, contours, contourIdx, Scalar(1), CV_FILLED, LINE_8, noArray(), INT_MAX, -roi.tl());
      cv::bitwise_and(mask, known_pixels(roi), mask);
      double avg_color = median(image(roi), mask);
      /* Scalar tmp = mean(image(roi), mask(roi)); */
      /* avg_color = tmp[0]; */
      /* for (const Point& pt : contours[contourIdx]) */
      /* { */
      /*   avg_color += image.at<uint16_t>(pt); */
      /* } */
      /* avg_color = avg_color/contours[contourIdx].size(); */

      blob.avg_depth = avg_color / 1000.0;

      if (params.filter_by_color && (avg_color < params.min_depth || avg_color > params.max_depth))
        continue;
    }

    blob.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

    // compute blob radius
    {
      std::vector<double> dists;
      for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
      {
        Point2d pt = contours[contourIdx][pointIdx];
        dists.push_back(norm(blob.location - pt));
      }
      std::sort(dists.begin(), dists.end());
      blob.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
    }

    blob.contours.push_back(contours[contourIdx]);

    blobs.push_back(blob);

#ifdef DEBUG_BLOB_DETECTOR
    drawContours(keypointsImage, contours, contourIdx, Scalar(0, avg_color * 255.0 / params.max_depth, 0), CV_FILLED, LINE_4);
#endif

#ifdef DEBUG_BLOB_DETECTOR
    circle(keypointsImage, blob.location, 1, Scalar(0, 0, 255), 1);
#endif
  }
#ifdef DEBUG_BLOB_DETECTOR
  cv::putText(keypointsImage, string("cur_depth: ") + to_string(cur_depth), Point(0, 120), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
  imshow("opencv_debug", keypointsImage);
  waitKey(50);
#endif
}
//}

/* method void DepthBlobDetector::detect(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat mask) //{ */
/* inspired by https://github.com/opencv/opencv/blob/3.4/modules/features2d/src/blobdetector.cpp */
void DepthBlobDetector::detect(cv::Mat image, cv::Mat known_pixels, cv::Mat unknown_pixels, cv::Mat image_raw, std::vector<Blob>& ret_blobs)
{
  ret_blobs.clear();
  assert(params.min_repeatability != 0);
  Mat grayscaleImage;
  if (image.channels() == 3 || image.channels() == 4)
    cvtColor(image, grayscaleImage, COLOR_BGR2GRAY);
  else
    grayscaleImage = image;

  if (grayscaleImage.type() != CV_8UC1 && grayscaleImage.type() != CV_16UC1)
  {
    ROS_ERROR("Blob detector only supports 8-bit and 16-bit images!");
    return;
  }

  std::vector<std::vector<Blob>> blobs;
  uint16_t thresh_start = params.min_threshold;
  if (params.use_threshold_width)
    thresh_start = params.min_threshold + params.threshold_width;
  for (uint16_t thresh = thresh_start; thresh < params.max_threshold; thresh += params.threshold_step)
  {
    Mat binarizedImage;
#ifdef DEBUG_BLOB_DETECTOR
    ROS_INFO("[%s]: using threshold %u", ros::this_node::getName().c_str(), thresh);
#endif
    if (params.use_threshold_width)
      inRange(grayscaleImage, thresh - params.threshold_width, thresh, binarizedImage);
    else
      inRange(grayscaleImage, params.min_depth, thresh, binarizedImage);
    /* cv::bitwise_or(binarizedImage, unknown_pixels, binarizedImage); // maybe not such a good idea for detecting the UAV against the sky */

#ifdef DEBUG_BLOB_DETECTOR
    cur_depth = (thresh + params.threshold_width) / 1000.0;
#endif
    std::vector<Blob> curBlobs;
    findBlobs(image_raw, known_pixels, binarizedImage, curBlobs);
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
    if (cur_blobs.size() < params.min_repeatability)
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
    result_blob.avg_depth = cur_blobs[cur_blobs.size() / 2].avg_depth;
    result_blob.convexity = cur_blobs[cur_blobs.size() / 2].convexity;
    result_blob.angle = cur_blobs[cur_blobs.size() / 2].angle;
    result_blob.area = cur_blobs[cur_blobs.size() / 2].area;
    result_blob.circularity = cur_blobs[cur_blobs.size() / 2].circularity;
    result_blob.inertia = cur_blobs[cur_blobs.size() / 2].inertia;
    result_blob.contours = contours;
    ret_blobs.push_back(result_blob);
  }

  /* if (!mask.empty()) */
  /* { */
  /*     KeyPointsFilter::runByPixelsMask(keypoints, mask); */
  /* } */
}
//}
