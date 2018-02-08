#include "uav_detect.h"

using namespace cv;
using namespace std;

cv_bridge::CvImagePtr last_cam_image;

void camera_callback(const sensor_msgs::ImageConstPtr& image_msg)
{
  cv_bridge::CvImagePtr image_in = cv_bridge::toCvCopy(image_msg, enc::BGR8);
  if (last_cam_image->header.seq != image_in->header.seq)
  {
    last_cam_image = CvImageConstPtr toCvShare(const sensor_msgs::ImageConstPtr& image_msg,
                          enc::BGR8);
  }
}

int main(int argc, char **argv)
{
  string uav_name, data_file, names_file, cfg_file, weights_file;

  ros::init(argc, argv, "uav_detect");
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
  // Data file of the neural network
  nh.param("data_file", data_file, string());
  if (data_file.empty())
  {
    ROS_ERROR("No *.data file specified!");
    ros::shutdown();
  }
  // Names file of the neural network
  nh.param("names_file", names_file, string());
  if (names_file.empty())
  {
    ROS_ERROR("No *.names file specified!");
    ros::shutdown();
  }
  // Configuration file of the neural network
  nh.param("cfg_file", cfg_file, string());
  if (cfg_file.empty())
  {
    ROS_ERROR("No *.cfg file specified!");
    ros::shutdown();
  }
  // Weights file of the neural network
  nh.param("weights_file", weights_file, string());
  if (weights_file.empty())
  {
    ROS_ERROR("No *.weights file specified!");
    ros::shutdown();
  }

  /** Create publishers and subscribers **/
  ros::Publisher detections_pub = nh.advertise<uav_detect::Detections>("detections", 10);
  ros::Subscriber camera_sub = nh.subscribe("camera_input", 1, camera_callback, ros::TransportHints().tcpNoDelay());
  //ros::Publisher dbg_pub = nh.advertise<collision_avoidance_tw::FutureCollisions>("detections_DBG", 1);

  printf("Creating detector object\n");
  MRS_Detector detector(data_file.c_str(), names_file.c_str(), cfg_file.c_str(), weights_file.c_str(), 0.2, 0.1, 1);
  printf("Initializing detector object\n");
  detector.initialize();

  VideoCapture cap(0); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
    return -1;

  while (ros::ok())
  {
    ros::spinOnce();

    Mat camera_frame;
    cap >> camera_frame; // get a new frame from camera
    auto detections = detector.detect(
            camera_frame,
            0.1,
            0.1);
    for (auto det : detections)
    {
      cout << "Object detected:" << std::endl;
      /*Point pt1((det.bounding_box.x - det.bounding_box.w/2.0)*camera_frame.cols,
                (det.bounding_box.y - det.bounding_box.h/2.0)*camera_frame.rows);
      Point pt2((det.bounding_box.x + det.bounding_box.w/2.0)*camera_frame.cols,
                (det.bounding_box.y + det.bounding_box.h/2.0)*camera_frame.rows);
      rectangle(camera_frame, pt1, pt2, Scalar(0, 0, 255));*/
    }
    cout << "End of frame." << std::endl;
    //imshow("edges", camera_frame);
    //if(waitKey(30) >= 0) break;
    // /uav1/mobius_front/image_raw
  }
}