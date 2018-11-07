#include "main.h"
#include "display_utils.h"

#include <uav_detect/BlobDetection.h>
#include <uav_detect/BlobDetections.h>
#include <uav_detect/Contour.h>

#define OPENCV_VISUALISE

using namespace cv;
using namespace std;
using namespace uav_detect;
using namespace Eigen;

#ifdef OPENCV_VISUALISE //{
Point cursor_pos;
void mouse_callback([[maybe_unused]]int event, int x, int y, [[maybe_unused]]int flags, [[maybe_unused]]void* userdata)
{
  cursor_pos = Point(x, y);
}
#endif
//}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "dbg_display_node");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  mrs_lib::ParamLoader pl(nh);
  std::string path_to_mask = pl.load_param2<std::string>("path_to_mask", std::string());

  cv::Mat mask_im_inv;
  if (path_to_mask.empty())
  {
    ROS_INFO("[%s]: Not using image mask", ros::this_node::getName().c_str());
  } else
  {
    mask_im_inv = ~cv::imread(path_to_mask, cv::IMREAD_GRAYSCALE);
    if (mask_im_inv.empty())
    {
      ROS_ERROR("[%s]: Error loading image mask from file '%s'! Ending node.", ros::this_node::getName().c_str(), path_to_mask.c_str());
      ros::shutdown();
    } else if (mask_im_inv.type() != CV_8UC1)
    {
      ROS_ERROR("[%s]: Loaded image mask has unexpected type: '%u' (expected %u)! Ending node.", ros::this_node::getName().c_str(), mask_im_inv.type(), CV_8UC1);
      ros::shutdown();
    }
  }

  /** Create publishers and subscribers //{**/
  // Initialize other subs and pubs
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> sh_dm;
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> sh_dmp;
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> sh_img;
  mrs_lib::SubscribeHandlerPtr<uav_detect::BlobDetections> sh_blobs;

  mrs_lib::SubscribeMgr smgr(nh);
  sh_dm = smgr.create_handler_threadsafe<sensor_msgs::ImageConstPtr>("depthmap", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_dmp = smgr.create_handler_threadsafe<sensor_msgs::ImageConstPtr>("processed_depthmap", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_img = smgr.create_handler_threadsafe<sensor_msgs::ImageConstPtr>("rgb_img", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_blobs = smgr.create_handler_threadsafe<uav_detect::BlobDetections>("blob_detections", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));

  if (!smgr.loaded_successfully())
  {
    ROS_ERROR("[%s]: Failed to subscribe some nodes", ros::this_node::getName().c_str());
    ros::shutdown();
  }

  //}

#ifdef OPENCV_VISUALISE //{
  /* Open OpenCV windows to display the debug info //{ */
  int window_flags = WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL;
  string rgb_winname = "RGB_image";
  string dm_winname = "depth_image";
  string det_winname = "depth_detections";
  cv::namedWindow(rgb_winname, window_flags);
  cv::namedWindow(dm_winname, window_flags);
  cv::namedWindow(det_winname, window_flags);
  setMouseCallback(det_winname, mouse_callback, NULL);
  //}
#endif  //}
  
  ros::Rate r(30);
  bool paused = false;
  bool fill_blobs = true;
  cv::Mat source_img, processed_img;
  uav_detect::BlobDetections cur_detections;
  bool cur_detections_initialized;

  std::list<sensor_msgs::ImageConstPtr> dm_buffer;
  std::list<sensor_msgs::ImageConstPtr> dmp_buffer;
  std::list<sensor_msgs::ImageConstPtr> img_buffer;

  while (ros::ok())
  {
    ros::spinOnce();
  
    if (sh_dm->new_data())
      add_to_buffer(sh_dm->get_data(), dm_buffer);
    if (sh_dmp->new_data())
      add_to_buffer(sh_dmp->get_data(), dmp_buffer);
    if (sh_img->new_data())
      add_to_buffer(sh_img->get_data(), img_buffer);

    bool has_data = sh_dm->has_data() && sh_dmp->has_data() && sh_img->has_data() && sh_blobs->has_data();

    if (has_data)
    {
      if (!paused || !cur_detections_initialized)
      {
        cur_detections = sh_blobs->get_data();
        cur_detections_initialized = true;
      }

      if (!paused || source_img.empty())
      {
        sensor_msgs::ImageConstPtr img_ros = find_closest(cur_detections.header.stamp, dm_buffer);
        source_img = (cv_bridge::toCvCopy(img_ros, string("16UC1")))->image;
      }

      if (!paused || processed_img.empty())
      {
        sensor_msgs::ImageConstPtr img_ros = find_closest(cur_detections.header.stamp, dmp_buffer);
        cv::cvtColor((cv_bridge::toCvCopy(img_ros, string("16UC1")))->image, processed_img, COLOR_GRAY2BGR);
      }

      cv::Mat rgb_im;
      if ((!paused && sh_img->has_data()) || (rgb_im.empty() && sh_img->has_data()))
      {
        sensor_msgs::ImageConstPtr img_ros = find_closest(cur_detections.header.stamp, img_buffer);
        rgb_im = (cv_bridge::toCvCopy(img_ros, sensor_msgs::image_encodings::BGR8))->image;
      }

      cv::Mat dm_im_colormapped;
      {
        double min;
        double max;
        cv::Mat unknown_pixels;
        cv::compare(source_img, 0, unknown_pixels, cv::CMP_EQ);
#ifndef SIMULATION
        cv::minMaxIdx(source_img, &min, &max, nullptr, nullptr, ~unknown_pixels);
#else
        min = 0.0;
        max = 12000.0;
#endif
        cv::Mat im_8UC1;
        source_img.convertTo(im_8UC1, CV_8UC1, 255 / (max-min), -min); 
        applyColorMap(im_8UC1, dm_im_colormapped, cv::COLORMAP_JET);
        cv::Mat blackness = cv::Mat::zeros(dm_im_colormapped.size(), dm_im_colormapped.type());
        blackness.copyTo(dm_im_colormapped, unknown_pixels);
      }

      cv::Mat processed_im_copy;
      processed_img.copyTo(processed_im_copy);
      int sure = 0;
      bool displaying_info = false;
      for (const auto& blob : cur_detections.blobs)
      {

        auto max = blob.contours.size();
        if (!blob.contours.empty())
        {
          sure++;
          if (fill_blobs)
          {
            for (size_t it = blob.contours.size()-1; it; it--)
            /* size_t it = blob.contours.size()/2; */
            {
              const auto& pxs = blob.contours.at(it);
              vector<cv::Point> cnt;
              cnt.reserve(pxs.pixels.size());
              for (const uav_detect::ImagePixel px : pxs.pixels)
                cnt.push_back(cv::Point(px.x, px.y));

              vector<vector<cv::Point>> cnts;
              cnts.push_back(cnt);
              cv::drawContours(processed_im_copy, cnts, 0, Scalar(0, 65535, 65535/max*it), CV_FILLED);
              if (!displaying_info && pointPolygonTest(cnt, cursor_pos, false) > 0)
              {
                // display information about this contour
                displaying_info = true;
                cv::putText(processed_im_copy, string("avg_depth: ") + to_string(blob.avg_depth), Point(0, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_im_copy, string("confidence: ") + to_string(blob.confidence), Point(0, 65), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_im_copy, string("repeatability: ") + to_string(blob.contours.size()), Point(0, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_im_copy, string("convexity: ") + to_string(blob.convexity), Point(0, 95), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_im_copy, string("angle: ") + to_string(blob.angle), Point(0, 110), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_im_copy, string("area: ") + to_string(blob.area), Point(0, 125), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_im_copy, string("circularity: ") + to_string(blob.circularity), Point(0, 140), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_im_copy, string("radius: ") + to_string(blob.radius), Point(0, 155), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_im_copy, string("inertia: ") + to_string(blob.inertia), Point(0, 170), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
              }
            }
          } else
          {
            cv::circle(processed_im_copy, Point(blob.x, blob.y), blob.radius, Scalar(0, 0, 65535), 2);
          }
          if (!rgb_im.empty())
            cv::circle(rgb_im, Point(blob.x, blob.y), blob.radius, Scalar(0, 0, 255), 2);
          cv::circle(dm_im_colormapped, Point(blob.x, blob.y), blob.radius, Scalar(0, 0, 255), 2);
        }
      }
      cv::putText(processed_im_copy, string("found: ") + to_string(sure), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(0, 0, 65535), 2);

      // highlight masked-out areas
      if (!mask_im_inv.empty())
      {
        cv::Mat red(processed_im_copy.size(), processed_im_copy.type());
        red.setTo(cv::Scalar(0, 0, 65535), mask_im_inv);
        cv::Mat tmp;
        cv::addWeighted(processed_im_copy, 0.7, red, 0.3, 0.0, tmp);
        tmp.copyTo(processed_im_copy, mask_im_inv);
      }

#ifdef OPENCV_VISUALISE //{
      if (!rgb_im.empty())
        imshow(rgb_winname, rgb_im);
      imshow(dm_winname, dm_im_colormapped);
      imshow(det_winname, processed_im_copy);
      int key = waitKey(3);
      switch (key)
      {
        case ' ':
          paused = !paused;
          break;
        case 'f':
          fill_blobs = !fill_blobs;
          break;
      }
#endif //}

    /* sensor_msgs::ImagePtr out_msg = dbg_img.toImageMsg(); */
    /* thresholded_pub.publish(out_msg); */
    }

    r.sleep();
  }
}
