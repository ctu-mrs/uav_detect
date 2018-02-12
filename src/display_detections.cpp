/**
 * This program simply draws detection bounding boxes into the last camera frame
 * and publishes that for debugging purposes.
 * **/
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <uav_detect/Detection.h>
#include <uav_detect/Detections.h>

using namespace cv;
using namespace std;

bool new_cam_image = false;
bool new_detections = false;
cv_bridge::CvImagePtr last_cam_image_ptr;
uav_detect::Detections last_detections;

void camera_callback(const sensor_msgs::ImagePtr& image_msg)
{
  last_cam_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  new_cam_image = true;
}

void detections_callback(const uav_detect::Detections& dets_msg)
{
  last_detections = dets_msg;
  new_detections = true;
}

int main(int argc, char **argv)
{
  string uav_name;

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
  cout << "Using parameters:" << std::endl;
  cout << "\tuav name:\t" << uav_name << std::endl;

  /** Create publishers and subscribers **/
  ros::Subscriber camera_sub = nh.subscribe("camera_input", 1, camera_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber detections_sub = nh.subscribe("detections", 1, detections_callback, ros::TransportHints().tcpNoDelay());
  ros::Publisher det_imgs_pub = nh.advertise<sensor_msgs::Image>("det_imgs", 1);

  cout << "----------------------------------------------------------" << std::endl;

  while (ros::ok())
  {
    ros::spinOnce();

    ros::Rate r(10);
    if (new_cam_image && new_detections)
    {
      new_cam_image = false;
      new_detections = false;
      cout << "Processing new detections" << std::endl;

      for (auto det : last_detections.detections)
      {
        cout << "Drawing one detection" << std::endl;
        cv::Rect det_rect(
                  (int)round((det.x_relative-det.w_relative/2)*last_detections.w_camera),
                  (int)round((det.y_relative-det.h_relative/2)*last_detections.h_camera),
                  (int)round(det.w_relative*last_detections.w_camera),
                  (int)round(det.h_relative*last_detections.h_camera));
        cv::rectangle(last_cam_image_ptr->image, det_rect, Scalar(0, 0, 255), 5);
      }

      det_imgs_pub.publish(last_cam_image_ptr->toImageMsg());

      cout << "Detection processed" << std::endl;
    } else
    {
      r.sleep();
    }
  }
}
