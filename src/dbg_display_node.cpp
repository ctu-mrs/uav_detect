#include "main.h"
#include "display_utils.h"

#include <uav_detect/BlobDetection.h>
#include <uav_detect/BlobDetections.h>
#include <uav_detect/Contour.h>

using namespace cv;
using namespace std;
using namespace uav_detect;
using namespace Eigen;

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

  mrs_lib::ParamLoader pl(nh);
  std::string path_to_mask = pl.loadParam2<std::string>("path_to_mask", std::string());
  int unknown_pixel_value = pl.loadParam2<int>("unknown_pixel_value", 0);

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
  mrs_lib::SubscribeHandlerOptions shopts;
  shopts.no_message_timeout = ros::Duration(5.0);
  auto sh_dm = mrs_lib::SubscribeHandler<sensor_msgs::Image>(shopts, "depthmap");
  auto sh_dmp = mrs_lib::SubscribeHandler<sensor_msgs::Image>(shopts, "processed_depthmap");
  auto sh_img = mrs_lib::SubscribeHandler<sensor_msgs::Image>(shopts, "rgb_img");
  auto sh_blobs = mrs_lib::SubscribeHandler<uav_detect::BlobDetections>(shopts, "blob_detections");

  //}

  /* Open OpenCV windows to display the debug info //{ */
  int window_flags = WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL;
  std::string rgb_winname = "RGB_image";
  std::string dm_winname = "depth_image";
  std::string det_winname = "depth_detections";
  //}
  
  ros::Rate r(30);
  bool show_rgb = false;
  bool rgb_window_exists = false;
  bool show_raw = false;
  bool raw_window_exists = false;
  bool show_proc = true;
  bool proc_window_exists = false;
  bool paused = false;
  bool fill_blobs = true;
  bool draw_mask = false;
  size_t max_draw_contours = 20;
  cv::Mat source_img, processed_img, processed_img_raw;
  uav_detect::BlobDetections cur_detections;
  bool cur_detections_initialized;

  std::list<sensor_msgs::ImageConstPtr> dm_buffer;
  std::list<sensor_msgs::ImageConstPtr> dmp_buffer;
  std::list<sensor_msgs::ImageConstPtr> img_buffer;

  while (ros::ok())
  {
    ros::spinOnce();
  
    if (sh_dm.newMsg())
      add_to_buffer(sh_dm.getMsg(), dm_buffer);
    if (sh_dmp.newMsg())
      add_to_buffer(sh_dmp.getMsg(), dmp_buffer);
    if (sh_img.newMsg())
      add_to_buffer(sh_img.getMsg(), img_buffer);

    bool has_data = (sh_dm.hasMsg() || sh_dmp.hasMsg() || sh_img.hasMsg()) && sh_blobs.hasMsg();

    if (has_data)
    {
      if (!paused || !cur_detections_initialized)
      {
        cur_detections = *(sh_blobs.getMsg());
        cur_detections_initialized = true;
      }

      if (show_raw && sh_dm.hasMsg() && (!paused || source_img.empty()))
      {
        sensor_msgs::ImageConstPtr img_ros = find_closest(cur_detections.header.stamp, dm_buffer);
        source_img = (cv_bridge::toCvCopy(img_ros, string("16UC1")))->image;
        
        cv::namedWindow(dm_winname, window_flags);
        raw_window_exists = true;
      } else if (!show_raw && raw_window_exists)
      {
        try
        {
          cv::destroyWindow(dm_winname);
        } catch (cv::Exception)
        {}
        raw_window_exists = false;
      }

      if (show_proc && sh_dmp.hasMsg() && (!paused || processed_img.empty()))
      {
        sensor_msgs::ImageConstPtr img_ros = find_closest(cur_detections.header.stamp, dmp_buffer);
        processed_img_raw = cv_bridge::toCvCopy(img_ros, string("16UC1"))->image;
        cv::cvtColor(processed_img_raw, processed_img, COLOR_GRAY2BGR);

        cv::namedWindow(det_winname, window_flags);
        setMouseCallback(det_winname, mouse_callback, NULL);
        proc_window_exists = true;
      } else if (!show_proc && proc_window_exists)
      {
        try
        {
          cv::destroyWindow(det_winname);
        } catch (cv::Exception)
        {}
        proc_window_exists = false;
      }

      cv::Mat rgb_im;
      if (show_rgb && sh_img.hasMsg() && (!paused || rgb_im.empty()))
      {
        sensor_msgs::ImageConstPtr img_ros = find_closest(cur_detections.header.stamp, img_buffer);
        rgb_im = cv_bridge::toCvCopy(img_ros, sensor_msgs::image_encodings::BGR8)->image;

        cv::namedWindow(rgb_winname, window_flags);
        rgb_window_exists = true;
      } else if (!show_rgb && rgb_window_exists)
      {
        try
        {
          cv::destroyWindow(rgb_winname);
        } catch (cv::Exception)
        {}
        rgb_window_exists = false;
      }

      cv::Mat dm_im_colormapped;
      if (show_raw && !source_img.empty())
      {
        double min;
        double max;
        cv::Mat unknown_pixels;
        cv::compare(source_img, unknown_pixel_value, unknown_pixels, cv::CMP_EQ);
        cv::minMaxIdx(source_img, &min, &max, nullptr, nullptr, ~unknown_pixels);
        cv::Mat im_8UC1;
        source_img.convertTo(im_8UC1, CV_8UC1, 255.0 / (max-min), -min * 255.0 / (max-min)); 
        applyColorMap(im_8UC1, dm_im_colormapped, cv::COLORMAP_JET);
        cv::Mat blackness = cv::Mat::zeros(dm_im_colormapped.size(), dm_im_colormapped.type());
        blackness.copyTo(dm_im_colormapped, unknown_pixels);
      }

      cv::Mat processed_im_copy;
      if (show_proc && !processed_img.empty())
      {
        processed_img.copyTo(processed_im_copy);
      }
      int sure = 0;
      bool displaying_info = false;
      for (const auto& blob : cur_detections.blobs)
      {

        auto n_contours = blob.contours.size();
        if (!blob.contours.empty())
        {
          sure++;
          const std::string id_txt = "id: " + to_string(blob.id);
          bool blob_displaying_info = false;

          /* Draw blobs to the processed depthmap //{ */
          if (show_proc && !processed_img.empty())
          {
            if (fill_blobs)
            {
              std::vector<cv::Point> cnt_detail_info(0);

              size_t max_it = min(blob.contours.size(), max_draw_contours);
              for (size_t it = 0; it < max_it; it++)
              {
                const auto& pxs = blob.contours.at(it);
                vector<cv::Point> cnt;
                cnt.reserve(pxs.pixels.size());
                for (const uav_detect::ImagePixel px : pxs.pixels)
                  cnt.push_back(cv::Point(px.x, px.y));
          
                vector<vector<cv::Point>> cnts;
                cnts.push_back(cnt);
                cv::drawContours(processed_im_copy, cnts, 0, Scalar(0, 65535, 65535/n_contours*it), CV_FILLED);
                if (!displaying_info && pointPolygonTest(cnt, cursor_pos, false) > 0)
                {
                  if (it == max_it-1)
                  {
                    cnt_detail_info = cnt;
                  }
                  // display information about this contour
                  displaying_info = true;
                  blob_displaying_info = true;
                  const int line_offset = 50;
                  const int line_space = 15;
                  int line_it = 0;

                  cv::putText(processed_im_copy, string("area: ") + to_string(blob.area), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("max. area diff: ") + to_string(blob.max_area_diff), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("circularity: ") + to_string(blob.circularity), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("convexity: ") + to_string(blob.convexity), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("avg. depth: ") + to_string(blob.avg_depth), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("known pixels ratio: ") + to_string(blob.known_pixels_ratio), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("known pixels: ") + to_string(blob.known_pixels), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("angle: ") + to_string(blob.angle), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("inertia: ") + to_string(blob.inertia), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("repeatability: ") + to_string(blob.contours.size()), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
          
                  cv::putText(processed_im_copy, string("confidence: ") + to_string(blob.confidence), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                  cv::putText(processed_im_copy, string("radius: ") + to_string(blob.radius), Point(0, line_offset+line_it++*line_space), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                } else if (blob_displaying_info && it == max_it-1)
                {
                  cnt_detail_info = cnt;
                }

              }

              /* if (!cnt_detail_info.empty()) */
              /* { */
              /*   double area = cv::contourArea(cnt_detail_info); */
              /*   cv::putText(processed_im_copy, string("|   ") + to_string(area), Point(300, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 1); */
              /* } */
            } else
            {
              cv::circle(processed_im_copy, Point(blob.x, blob.y), blob.radius, Scalar(0, 0, 65535), 3);
            }
            /* cv::putText(processed_im_copy, id_txt, Point(blob.x + blob.radius + 5, blob.y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 65535), 1); */
          }
          //}

          /* Draw detections to the rgb image //{ */
          if (show_rgb && !rgb_im.empty())
          {
            if (fill_blobs)
            {
              for (size_t it = blob.contours.size()-1; it; it--)
              {
                const auto& pxs = blob.contours.at(it);
                vector<cv::Point> cnt;
                cnt.reserve(pxs.pixels.size());
                for (const uav_detect::ImagePixel px : pxs.pixels)
                  cnt.push_back(cv::Point(px.x, px.y));
          
                vector<vector<cv::Point>> cnts;
                cnts.push_back(cnt);
                cv::drawContours(rgb_im, cnts, 0, Scalar(0, 255, 255/n_contours*it), CV_FILLED);
              }
            } else
            {
              cv::circle(rgb_im, Point(blob.x, blob.y), blob.radius, Scalar(0, 0, 255), 3);
            }
            /* cv::putText(rgb_im, id_txt, Point(blob.x + blob.radius + 5, blob.y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2); */
          }
          //}

          /* Draw detections to the raw depthmap //{ */
          if (show_raw && !dm_im_colormapped.empty())
          {
            if (fill_blobs)
            {
              for (size_t it = blob.contours.size()-1; it; it--)
              {
                const auto& pxs = blob.contours.at(it);
                vector<cv::Point> cnt;
                cnt.reserve(pxs.pixels.size());
                for (const uav_detect::ImagePixel px : pxs.pixels)
                  cnt.push_back(cv::Point(px.x, px.y));
          
                vector<vector<cv::Point>> cnts;
                cnts.push_back(cnt);
                cv::drawContours(dm_im_colormapped, cnts, 0, Scalar(0, 255, 255/n_contours*it), CV_FILLED);
              }
            } else
            {
              cv::circle(dm_im_colormapped, Point(blob.x, blob.y), blob.radius, Scalar(0, 0, 255), 3);
            }
            /* cv::putText(dm_im_colormapped, id_txt, Point(blob.x + blob.radius + 5, blob.y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2); */
          }
          //}

        }
      }

      /* if (show_proc && !processed_img.empty()) */
      /* { */
      /*   cv::putText(processed_im_copy, string("found: ") + to_string(sure), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(0, 0, 65535), 2); */
      /*   uint16_t depth = processed_img_raw.at<uint16_t>(cursor_pos); */
      /*   cv::putText(processed_im_copy, string("depth: ") + to_string(depth), Point(200, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 65535), 2); */
      /* } */

      /* highlight masked-out area //{ */
      if (draw_mask && !mask_im_inv.empty())
      {
        if (show_proc && !processed_im_copy.empty())
        {
          cv::Mat red, tmp;
          red = cv::Mat(processed_im_copy.size(), processed_im_copy.type());
          red.setTo(cv::Scalar(0, 0, 65535), mask_im_inv);
          cv::addWeighted(processed_im_copy, 0.7, red, 0.3, 0.0, tmp);
          tmp.copyTo(processed_im_copy, mask_im_inv);
        }
      
        {
          cv::Mat red;
          if (show_raw && !dm_im_colormapped.empty())
          {
            cv::Mat tmp;
            red = cv::Mat(dm_im_colormapped.size(), dm_im_colormapped.type());
            red.setTo(cv::Scalar(0, 0, 255), mask_im_inv);
            cv::addWeighted(dm_im_colormapped, 0.7, red, 0.3, 0.0, tmp);
            tmp.copyTo(dm_im_colormapped, mask_im_inv);
          }
      
          if (show_rgb && !rgb_im.empty())
          {
            cv::Mat tmp;
            if (red.empty())
            {
              red = cv::Mat(rgb_im.size(), rgb_im.type());
              red.setTo(cv::Scalar(0, 0, 255), mask_im_inv);
            }
            cv::addWeighted(rgb_im, 0.7, red, 0.3, 0.0, tmp);
            tmp.copyTo(rgb_im, mask_im_inv);
          }
        }
      }
      //}

      if (show_raw && !dm_im_colormapped.empty())
        imshow(dm_winname, dm_im_colormapped);
      if (show_proc && !processed_im_copy.empty())
        imshow(det_winname, processed_im_copy);
      if (show_rgb && !rgb_im.empty())
        imshow(rgb_winname, rgb_im);

      int key = waitKey(1);
      switch (key)
      {
        case ' ':
          ROS_INFO("[%s]: %spausing", ros::this_node::getName().c_str(), paused?"un":"");
          paused = !paused;
          break;
        case 'f':
          ROS_INFO("[%s]: %sfilling blobs", ros::this_node::getName().c_str(), fill_blobs?"not ":"");
          fill_blobs = !fill_blobs;
          break;
        case 'm':
          ROS_INFO("[%s]: %sdrawing mask", ros::this_node::getName().c_str(), draw_mask?"not ":"");
          draw_mask = !draw_mask;
          break;
        case 'd':
          ROS_INFO("[%s]: %sshowing raw depthmap", ros::this_node::getName().c_str(), show_raw?"not ":"");
          show_raw = !show_raw;
          break;
        case 'p':
          ROS_INFO("[%s]: %sshowing processed depthmap", ros::this_node::getName().c_str(), show_proc?"not ":"");
          show_proc = !show_proc;
          break;
        case 'r':
          ROS_INFO("[%s]: %sshowing rgb image", ros::this_node::getName().c_str(), show_rgb?"not ":"");
          show_rgb = !show_rgb;
          break;
        case '+':
          max_draw_contours++;
          ROS_INFO("[%s]: displaying max. %lu contours", ros::this_node::getName().c_str(), max_draw_contours);
          break;
        case '-':
          max_draw_contours--;
          ROS_INFO("[%s]: displaying max. %lu contours", ros::this_node::getName().c_str(), max_draw_contours);
          break;
 
      }

    }

    r.sleep();
  }
}
