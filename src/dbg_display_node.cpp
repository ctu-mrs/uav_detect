#include "main.h"

using namespace cv;
using namespace std;
using namespace uav_detect;
using namespace Eigen;

// Callback for the rgb image
bool new_rgb = false;
sensor_msgs::Image last_rgb_msg;
void rgb_callback(const sensor_msgs::Image& rgb_msg)
{
  ROS_INFO_THROTTLE(1.0, "Got new rgb image");
  last_rgb_msg = rgb_msg;
  new_rgb = true;
}

// Callback for the depth map
bool new_dm = false;
sensor_msgs::Image last_dm_msg;
void depthmap_callback(const sensor_msgs::Image& dm_msg)
{
  ROS_INFO_THROTTLE(1.0, "Got new depth image");
  last_dm_msg = dm_msg;
  new_dm = true;
}

Point cursor_pos;
void mouse_callback([[maybe_unused]]int event, int x, int y, [[maybe_unused]]int flags, [[maybe_unused]]void* userdata)
{
  cursor_pos = Point(x, y);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "dbg_display_node");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Create publishers and subscribers //{**/
  tf2_ros::Buffer tf_buffer;
  // Initialize transform listener
  tf2_ros::TransformListener* tf_listener = new tf2_ros::TransformListener(tf_buffer);
  // Initialize other subs and pubs
  ros::Subscriber rgb_sub = nh.subscribe("rgb_image", 1, rgb_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber depthmap_sub = nh.subscribe("depth_map", 1, depthmap_callback, ros::TransportHints().tcpNoDelay());
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
  
  bool paused = false;
  bool fill_blobs = true;

  while (ros::ok())
  {
  
    if (!paused || source_img.image.empty())
    {
      source_img = *cv_bridge::toCvCopy(last_dm_msg, string("16UC1"));
      new_dm = false;
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
        cv::minMaxIdx(raw_im, &min, &max);
        cv::Mat im_8UC1;
        raw_im.convertTo(im_8UC1, CV_8UC1, 255 / (max-min), -min); 
        applyColorMap(im_8UC1, dm_im_colormapped, cv::COLORMAP_JET);
        cv::Mat blackness = cv::Mat::zeros(dm_im_colormapped.size(), dm_im_colormapped.type());
        blackness.copyTo(dm_im_colormapped, unknown_pixels);
      }

      int sure = 0;
      bool displaying_info = false;
      for (const auto& blob : blobs)
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
              cv::drawContours(dbg_img.image, blob.contours, it, Scalar(0, 65535, 65535/max*it), CV_FILLED);
              auto cur_blob = blob.contours.at(it);
              if (!displaying_info && pointPolygonTest(cur_blob, cursor_pos, false) > 0)
              {
                // display information about this contour
                displaying_info = true;
                cv::putText(dbg_img.image, string("avg_depth: ") + to_string(blob.avg_depth), Point(0, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("confidence: ") + to_string(blob.confidence), Point(0, 65), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("repeatability: ") + to_string(blob.contours.size()), Point(0, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("convexity: ") + to_string(blob.convexity), Point(0, 95), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("angle: ") + to_string(blob.angle), Point(0, 110), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("area: ") + to_string(blob.area), Point(0, 125), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("circularity: ") + to_string(blob.circularity), Point(0, 140), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("radius: ") + to_string(blob.radius), Point(0, 155), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("inertia: ") + to_string(blob.inertia), Point(0, 170), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
              }
            }
          } else
          {
            cv::circle(dbg_img.image, blob.location, blob.radius, Scalar(0, 0, 65535), 2);
          }
          if (new_rgb)
            cv::circle(rgb_im, blob.location, blob.radius, Scalar(0, 0, 255), 2);
          cv::circle(dm_im_colormapped, blob.location, blob.radius, Scalar(0, 0, 255), 2);
        }
      }
      cv::putText(dbg_img.image, string("found: ") + to_string(sure), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(0, 0, 65535), 2);

#ifdef OPENCV_VISUALISE //{
      if (new_rgb)
        imshow(rgb_winname, rgb_im);
      if (new_thermal)
        imshow(thermal_winname, thermal_im_colormapped);
      imshow(dm_winname, dm_im_colormapped);
      imshow(det_winname, dbg_img.image);
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

    sensor_msgs::ImagePtr out_msg = dbg_img.toImageMsg();
    thresholded_pub.publish(out_msg);

  }
}
