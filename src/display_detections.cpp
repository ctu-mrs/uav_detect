/**
 * This program simply draws detection bounding boxes into the last camera frame
 * and publishes that for debugging purposes.
 * **/
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <image_geometry/pinhole_camera_model.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cnn_detect/Detection.h>
#include <cnn_detect/Detections.h>

#include <list>

using namespace cv;
using namespace std;

bool new_cam_image = false;
bool new_detections = false;
int image_buffer_max_size;
std::list<sensor_msgs::ImageConstPtr> image_ptr_buffer;
cnn_detect::Detections last_detections;

void camera_callback(const sensor_msgs::ImageConstPtr& image_msg)
{
  cout << "Got camera image" << std::endl;
  image_ptr_buffer.push_back(image_msg);
  if ((int)image_ptr_buffer.size() > image_buffer_max_size)
    image_ptr_buffer.pop_front();
  new_cam_image = true;
}

bool got_cinfo = false;
image_geometry::PinholeCameraModel camera_model;
void cinfo_callback(const sensor_msgs::CameraInfo& msg)
{
  if (got_cinfo)
    return;
  cout << "Got camera info" << std::endl;
  camera_model.fromCameraInfo(msg);
  got_cinfo = true;
}


sensor_msgs::ImageConstPtr find_closest(const std::list<sensor_msgs::ImageConstPtr>& image_ptr_buffer, ros::Time stamp)
{
  size_t it = 0;
  size_t closest_it;
  bool closest_set = false;
  double closest_diff;
  ros::Time closest_stamp;
  sensor_msgs::ImageConstPtr closest_image = nullptr;
  for (const auto& image_ptr : image_ptr_buffer)
  {
    it++;
    ros::Time cur_stamp = image_ptr->header.stamp;
    double cur_diff = abs(double((stamp - cur_stamp).toSec()));
    if (!closest_set || cur_diff < closest_diff)
    {
      closest_diff = cur_diff;
      closest_stamp = cur_stamp;
      closest_image = image_ptr;
      closest_it = it;
      closest_set = true;
    }
  }
  
  if (closest_set)
    ROS_INFO("Closest image stamp %.3f (looking for %.3f) found at index %lu/%lu",
            closest_stamp.toSec(),
            stamp.toSec(),
            closest_it,
            image_ptr_buffer.size());
            /* (*begin(image_ptr_buffer))->header.stamp.toSec(), */
            /* (*end(image_ptr_buffer))->header.stamp.toSec()); */
  else
    ROS_INFO("No image found");
  return closest_image;
}

void detections_callback(const cnn_detect::Detections& dets_msg)
{
  cout << "Got detection" << std::endl;
  last_detections = dets_msg;
  new_detections = true;
}

int main(int argc, char **argv)
{
  string uav_name, ovname;
  double prob_threshold;

  ros::init(argc, argv, "display_detections");
  ROS_INFO ("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Load parameters from ROS **/
  // UAV name
  nh.param("uav_name", uav_name, string());
  if (uav_name.empty())
  {
    ROS_ERROR("UAV_NAME is empty");
    ros::shutdown();
  }
  nh.param("image_buffer_max_size", image_buffer_max_size, 60);
  nh.param("threshold", prob_threshold, 0.2);

  cout << "Using parameters:" << std::endl;
  cout << "\tuav name:\t" << uav_name << std::endl;
  cout << "\tmax. size of image buffer:\t" << image_buffer_max_size << std::endl;
  cout << "\tprob. threshold:\t" << prob_threshold << std::endl;

  /** Create publishers and subscribers **/
  ros::Subscriber camera_sub = nh.subscribe("camera_input", 1, camera_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber cinfo_sub = nh.subscribe("camera_info", 1, cinfo_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber detections_sub = nh.subscribe("detections", 1, detections_callback, ros::TransportHints().tcpNoDelay());
  ros::Publisher det_imgs_pub = nh.advertise<sensor_msgs::Image>("det_imgs", 1);

  cout << "----------------------------------------------------------" << std::endl;

  int window_flags = WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL;
  string det_winname = "CNN_detections";
  cv::namedWindow(det_winname, window_flags);

  cv::Mat det_image;
  /* bool det_image_usable = false; */

  while (ros::ok())
  {
    ros::spinOnce();

    ros::Rate r(25);
    if (new_cam_image && new_detections)
    {
      new_cam_image = false;
      new_detections = false;
      cout << "Processing new detections" << std::endl;

      sensor_msgs::ImageConstPtr image_msg = find_closest(image_ptr_buffer, last_detections.header.stamp);
      if (image_msg == nullptr)
        continue;

      cv_bridge::CvImagePtr last_cam_image_ptr;
      last_cam_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8); 

      /* camera_model.rectifyImage(last_cam_image_ptr->image, det_image); */
      det_image = last_cam_image_ptr->image;  // a SHALLOW copy! sub_image shares pixels

      for (const auto &det : last_detections.detections)
      {
        if (det.confidence < prob_threshold)
          continue;
        cout << "Drawing one detection" << std::endl;

        int w_used = det.roi.width;
        int h_used = det.roi.height;
        int x_lt = (int)round((det.x-det.width/2.0)*w_used) + det.roi.x_offset;
        int y_lt = (int)round((det.y-det.height/2.0)*h_used) + det.roi.y_offset;
        int w = (int)round(det.width*w_used);
        int h = (int)round(det.height*h_used);
        cv::Rect det_rect(
                  x_lt,
                  y_lt,
                  w,
                  h);
        cv::rectangle(det_image, det_rect, Scalar(0, 0, 255), round(det.confidence*10));
        char buffer[255];
        sprintf(buffer, "mrs_mav, %.2f", det.confidence);
        cv::putText(det_image, buffer, Point(x_lt, y_lt-16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255), 2);
      }

      /* cv_bridge::CvImage img_bridge; */
      /* sensor_msgs::Image img_msg; // >> message to be sent */
      /* std_msgs::Header header; // empty header */
      /* header.stamp = ros::Time::now(); // time */
      /* img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, det_image); */
      /* img_bridge.toImageMsg(img_msg); // from cv_bridge to sensor_msgs::Image */
      /* det_imgs_pub.publish(img_msg); // ros::Publisher pub_img = node.advertise<senso */

      imshow(det_winname, det_image);
      waitKey(3);
      /* int key = waitKey(3); */

      cout << "Detection processed" << std::endl;
    } else
    {
      ROS_INFO_STREAM_THROTTLE(1.0, "[" << ros::this_node::getName().c_str() << "]: new_cam_image=" << new_cam_image << ", new_detections=" << new_detections);
      r.sleep();
    }
    /* if (save_vid && det_image_usable) */
    /* { */
    /*   outputVideo.write(det_image); */
    /* } */
  }
}
