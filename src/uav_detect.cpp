#include "uav_detect.h"

using namespace cv;
using namespace std;

bool new_cam_image = false;
cv_bridge::CvImageConstPtr last_cam_image_ptr;

void camera_callback(const sensor_msgs::ImageConstPtr& image_msg)
{
  cout << "Camera callback OK" << std::endl;
  last_cam_image_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
  new_cam_image = true;
}

int main(int argc, char **argv)
{
  string uav_name, data_file, names_file, cfg_file, weights_file;
  float threshold, hier_threshold;

  ros::init(argc, argv, "uav_detect");
  ROS_INFO ("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  // Initialize cout to print with precision to two dec. places
  cout << std::fixed << std::setprecision(2);


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
  // Detection threshold
  nh.param("threshold", threshold, 0.1f);
  // Detection hier threshold
  nh.param("hier_threshold", hier_threshold, 0.1f);

  /** Create publishers and subscribers **/
  ros::Publisher detections_pub = nh.advertise<uav_detect::Detections>("detections", 20);
  ros::Subscriber camera_sub = nh.subscribe("camera_input", 1, camera_callback, ros::TransportHints().tcpNoDelay());

  printf("Creating detector object\n");
  MRS_Detector detector(data_file.c_str(), names_file.c_str(), cfg_file.c_str(), weights_file.c_str(), 0.2, 0.1, 1);
  printf("Initializing detector object\n");
  detector.initialize();

  ros::Time last_frame = ros::Time::now();
  ros::Time new_frame = ros::Time::now();
  while (ros::ok())
  {
    ros::spinOnce();

    if (new_cam_image)
    {
      new_cam_image = false;
      cout << "Got new camera image." << std::endl;

      vector<uav_detect::Detection> detections = detector.detect(
              last_cam_image_ptr->image,
              threshold,
              hier_threshold);
      for (auto det : detections)
      {
        cout << "Object detected!" << std::endl;
        cout << "\t" << detector.get_class_name(det.class_ID) << ", p=" << det.probability << std::endl;
        cout << "\t[" << det.x_relative << "; " << det.y_relative << "]" << std::endl;
        cout << "\tw=" << det.w_relative << ", h=" << det.h_relative << std::endl;
      }

      uav_detect::Detections msg;
      msg.detections = detections;
      msg.w_camera = last_cam_image_ptr->image.cols;
      msg.h_camera = last_cam_image_ptr->image.rows;
      detections_pub.publish(msg);

      // Calculate FPS
      new_frame = ros::Time::now();
      ros::Duration frame_duration = new_frame-last_frame;
      last_frame = new_frame;
      cout << "Image processed (" << (1/frame_duration.toSec()) << "FPS)" << std::endl;
    } else
    {
      // wait for a new camera image (this branch executing is quite unlikely)
      ros::Rate r(20);
      r.sleep();
    }
  }
}
