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
uav_detect::Detections latest_detections;

void detections_callback(const uav_detect::Detections& dets_msg)
{
  latest_detections = dets_msg;
  new_detections = true;
}

// TODO: implement IoU(det1, det2)
vector<int> find_matching_detection(
                  const uav_detect::Detection& ref_det,
                  const uav_detect::Detections& dets,
                  double IoU_threshold = 0.5)
{
  double IoUmax = 0.0;
  int best_match = -1;  //indicates no match found
  int it = 0;
  for (const uav_detect::Detection det : dets)
  {
    double cur_IoU = IoU(ref_det, det);
    if (cur_IoU > IoU_threshold && cur_IoU > IoUmax)
    {
      best_match = it;
    }
    it++;
  }

  return best_match;
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

  uav_detect::Detections prev_detections;
  bool first_run = true;
  while (ros::ok())
  {
    ros::spinOnce();

    ros::Rate r(10);
    if (new_detections)
    {
      if (first_run)
      {
        prev_detections = latest_detections;
        first_run = false;
      }
      new_detections = false;
      cout << "Processing new detections" << std::endl;

      for (auto det : latest_detections.detections)
      {
        cout << "Processing one detection" << std::endl;
        int match = find_matching_detection(det, prev_detections);
        if (match < 0)
          cout << "No previous matching detection found" << std::endl;

        // TODO: Actual processing...
      }

      prev_detections = new_detections;
      cout << "Detection processed" << std::endl;
    } else
    {
      r.sleep();
    }
  }
}
