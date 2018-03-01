#include "uav_detect.h"

using namespace cv;
using namespace std;

bool new_cam_image = false;
cv_bridge::CvImageConstPtr last_cam_image_ptr;
bool got_cam_info = false;
sensor_msgs::CameraInfoConstPtr last_cam_info_ptr;

void camera_image_callback(const sensor_msgs::ImageConstPtr& image_msg)
{
  cout << "Camera image callback OK" << std::endl;
  last_cam_image_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
  new_cam_image = true;
}

void camera_info_callback(const sensor_msgs::CameraInfoConstPtr& info_msg)
{
  cout << "Camera info callback OK" << std::endl;
  last_cam_info_ptr = info_msg;
  got_cam_info = true;
}

int main(int argc, char **argv)
{
  string uav_name, data_file, names_file, cfg_file, weights_file;
  float threshold, hier_threshold;
  bool only_subsquare;

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
  // Whether to use only subsquare from the image
  nh.param("only_subsquare", only_subsquare, true);

  cout << "Using parameters:" << std::endl;
  cout << "\tuav name:\t" << uav_name << std::endl;
  cout << "\tdata file:\t" << data_file << std::endl;
  cout << "\tnames file:\t" << names_file << std::endl;
  cout << "\tcfg file:\t" << cfg_file << std::endl;
  cout << "\tweights file:\t" << weights_file << std::endl;
  cout << "\tthreshold:\t" << threshold << std::endl;
  cout << "\thier threshold:\t" << hier_threshold << std::endl;
  cout << "\tonly subsquare:\t" << only_subsquare << std::endl;

  /** Create publishers and subscribers **/
  ros::Publisher detections_pub = nh.advertise<uav_detect::Detections>("detections", 20);
  #ifdef DEBUG
  // Debug only
  #warning "Building with -DDEBUG (turn off in CMakeLists.txt)"
  ros::Publisher det_imgs_pub = nh.advertise<sensor_msgs::Image>("det_imgs", 1);
  #endif //DEBUG
  ros::Subscriber camera_image_sub = nh.subscribe("camera_input", 1, camera_image_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber camera_info_sub = nh.subscribe("camera_info", 1, camera_info_callback, ros::TransportHints().tcpNoDelay());

  cout << "Creating detector object\n";
  MRS_Detector detector(data_file.c_str(), names_file.c_str(), cfg_file.c_str(), weights_file.c_str(), 0.2, 0.1, 1);
  cout << "Initializing detector object\n";
  detector.initialize();

  cout << "----------------------------------------------------------" << std::endl;
  ros::Time last_frame = ros::Time::now();
  ros::Time new_frame = ros::Time::now();
  while (ros::ok())
  {
    ros::spinOnce();

    if (new_cam_image && got_cam_info)
    {
      new_cam_image = false;
      cout << "Got new camera image." << std::endl;

      cv::Mat det_image = last_cam_image_ptr->image;
      int h_used = det_image.rows;
      int w_used = det_image.cols;

      if (only_subsquare)
      {
        w_used = h_used;
        cv::Rect sub_rect((det_image.cols - w_used)/2, 0, h_used, w_used);
        det_image = det_image(sub_rect);  // a SHALLOW copy! sub_image shares pixels
      }

      vector<uav_detect::Detection> detections = detector.detect(
              det_image,
              threshold,
              hier_threshold);
      for (const auto &det : detections)
      {
        cout << "Object detected!" << std::endl;
        cout << "\t" << detector.get_class_name(det.class_ID) << ", p=" << det.probability << std::endl;
        cout << "\t[" << det.x_relative << "; " << det.y_relative << "]" << std::endl;
        cout << "\tw=" << det.w_relative << ", h=" << det.h_relative << std::endl;
      }

      uav_detect::Detections msg;
      msg.detections = detections;
      msg.w_used = w_used;
      msg.h_used = h_used;
      msg.camera_info = sensor_msgs::CameraInfo(*last_cam_info_ptr);
      msg.stamp = last_cam_image_ptr->header.stamp;
      detections_pub.publish(msg);

      // Calculate FPS
      new_frame = ros::Time::now();
      ros::Duration frame_duration = new_frame-last_frame;
      last_frame = new_frame;
      float freq = -1.0;
      if (frame_duration.toSec() != 0.0)
        freq = 1/frame_duration.toSec();
      cout << "Image processed (" << freq << "FPS)" << std::endl;
    } else
    {
      // wait for a new camera image (this branch executing is quite unlikely)
      ros::Rate r(20);
      r.sleep();
    }
  }
}
