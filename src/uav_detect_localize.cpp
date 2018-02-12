/**
 * This program tries to localize other MAVs using the detected bounding boxes.
 * Detected bounding boxes spawn Kalman Filters.
 * Each Kalman Filter predicts its corresponding bounding box.
 * For each detection the Kalman Filtres from previous iteration are matched
 * to new bounding boxes (according to center distance?).
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

bool new_detections = false;
uav_detect::Detections last_detections;

void detections_callback(const uav_detect::Detections& dets_msg)
{
  last_detections = dets_msg;
  new_detections = true;
}

int main(int argc, char **argv)
{
  string uav_name;

  ros::init(argc, argv, "uav_detect_localize");
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
  ros::Subscriber detections_sub = nh.subscribe("detections", 1, detections_callback, ros::TransportHints().tcpNoDelay());

  cout << "----------------------------------------------------------" << std::endl;

  while (ros::ok())
  {
    ros::spinOnce();

    ros::Rate r(10);
    if (new_detections)
    {
      new_detections = false;
      cout << "Processing new detections" << std::endl;

      for (auto det : last_detections.detections)
      {
        cout << "Processing one detection" << std::endl;
        // TODO: Actual processing...
      }

      cout << "Detection processed" << std::endl;
    } else
    {
      r.sleep();
    }
  }
}
