#include "main.h"
#define OPENCV_VISUALISE

using namespace cv;
using namespace std;
using namespace uav_detect;
using namespace Eigen;

/* Callbacks //{ */
// Callback for the rgb image
bool new_rgb = false;
sensor_msgs::Image last_rgb_msg;
void rgb_callback(const sensor_msgs::Image& rgb_msg)
{
  ROS_INFO_THROTTLE(1.0, "Getting new rgb images");
  last_rgb_msg = rgb_msg;
  new_rgb = true;
}

// Callback for the depth map
bool new_dm = false;
sensor_msgs::Image last_dm_msg;
void depthmap_callback(const sensor_msgs::Image& dm_msg)
{
  ROS_INFO_THROTTLE(1.0, "Getting new depth images");
  last_dm_msg = dm_msg;
  new_dm = true;
}

// Callback for the depth map
bool new_processed_dm = false;
sensor_msgs::Image last_processed_dm_msg;
void processed_depthmap_callback(const sensor_msgs::Image& processed_dm_msg)
{
  ROS_INFO_THROTTLE(1.0, "Getting new processed depth images");
  last_processed_dm_msg = processed_dm_msg;
  new_processed_dm = true;
}

// Callback for the depth map
bool new_blobs = false;
uav_detect::BlobDetections last_blobs_msg;
void blobs_callback(const uav_detect::BlobDetections& blobs_msg)
{
  ROS_INFO_THROTTLE(1.0, "Getting new blobs");
  last_blobs_msg = blobs_msg;
  new_blobs = true;
}
//}

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

  /** Create publishers and subscribers //{**/
  // Initialize other subs and pubs
  ros::Subscriber rgb_sub = nh.subscribe("rgb_img", 1, rgb_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber depthmap_sub = nh.subscribe("depthmap", 1, depthmap_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber processed_depthmap_sub = nh.subscribe("processed_depthmap", 1, processed_depthmap_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber detected_blobs_pub = nh.subscribe("blob_detections", 1, blobs_callback, ros::TransportHints().tcpNoDelay());
  /* ros::Publisher detected_UAV_sub = nh.subscribe("detected_uav", 10); */
  /* ros::Publisher dbg_img_sub = nh.subscribe("debug_im", 1); */
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
  
  ros::Rate r(50);
  bool paused = false;
  bool fill_blobs = true;
  cv::Mat source_img, processed_img;
  uav_detect::BlobDetections cur_detections;
  bool cur_detections_initialized;

  while (ros::ok())
  {
    ros::spinOnce();
  
    if (new_dm && new_blobs && new_processed_dm)
    {
      if (!paused || source_img.empty())
      {
        source_img = (cv_bridge::toCvCopy(last_dm_msg, string("16UC1")))->image;
        new_dm = false;
      }

      if (!paused || processed_img.empty())
      {
        cv::cvtColor((cv_bridge::toCvCopy(last_processed_dm_msg, string("16UC1")))->image, processed_img, COLOR_GRAY2BGR);
        new_processed_dm = false;
      }

      if (!paused || cur_detections_initialized)
      {
        cur_detections = last_blobs_msg;
        new_blobs = false;
        cur_detections_initialized = true;
      }

      cv::Mat rgb_im;
      if (new_rgb)
      {
        rgb_im = (cv_bridge::toCvCopy(last_rgb_msg, sensor_msgs::image_encodings::BGR8))->image;
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
              cv::drawContours(processed_img, cnts, 0, Scalar(0, 65535, 65535/max*it), CV_FILLED);
              if (!displaying_info && pointPolygonTest(cnt, cursor_pos, false) > 0)
              {
                // display information about this contour
                displaying_info = true;
                cv::putText(processed_img, string("avg_depth: ") + to_string(blob.avg_depth), Point(0, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_img, string("confidence: ") + to_string(blob.confidence), Point(0, 65), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_img, string("repeatability: ") + to_string(blob.contours.size()), Point(0, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_img, string("convexity: ") + to_string(blob.convexity), Point(0, 95), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_img, string("angle: ") + to_string(blob.angle), Point(0, 110), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_img, string("area: ") + to_string(blob.area), Point(0, 125), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_img, string("circularity: ") + to_string(blob.circularity), Point(0, 140), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_img, string("radius: ") + to_string(blob.radius), Point(0, 155), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(processed_img, string("inertia: ") + to_string(blob.inertia), Point(0, 170), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
              }
            }
          } else
          {
            cv::circle(processed_img, Point(blob.x, blob.y), blob.radius, Scalar(0, 0, 65535), 2);
          }
          if (new_rgb)
            cv::circle(rgb_im, Point(blob.x, blob.y), blob.radius, Scalar(0, 0, 255), 2);
          cv::circle(dm_im_colormapped, Point(blob.x, blob.y), blob.radius, Scalar(0, 0, 255), 2);
        }
      }
      cv::putText(processed_img, string("found: ") + to_string(sure), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(0, 0, 65535), 2);

#ifdef OPENCV_VISUALISE //{
      if (new_rgb)
        imshow(rgb_winname, rgb_im);
      imshow(dm_winname, dm_im_colormapped);
      imshow(det_winname, processed_img);
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

      cout << "Image processed" << endl;
    }
    r.sleep();
  }
}
