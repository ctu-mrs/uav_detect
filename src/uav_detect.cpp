#include "uav_detect.h"

using namespace cv;
using namespace std;

bool new_cam_image = false;
cv_bridge::CvImageConstPtr last_cam_image_ptr;
bool got_cam_info = false;
sensor_msgs::CameraInfo last_cam_info;

void camera_image_callback(const sensor_msgs::ImageConstPtr& image_msg)
{
  cout << "Camera image callback OK" << std::endl;
  last_cam_image_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
  new_cam_image = true;
}

string preset_calib_name;
string preset_distortion_model;
int preset_width, preset_height;
std::vector<double> preset_camera_matrix;
std::vector<double> preset_distortion_coefficients;
std::vector<double> preset_rectification_matrix;
std::vector<double> preset_projection_matrix;
 
void camera_info_callback(sensor_msgs::CameraInfo info_msg)
{
  if (info_msg.distortion_model.empty())
  {
    cout << "Camera not calibrated. Using preset '" << preset_calib_name << "' calibration instead" << std::endl;
    info_msg.distortion_model = preset_distortion_model;
    info_msg.D = preset_distortion_coefficients;
    std::copy(std::begin(preset_camera_matrix), std::end(preset_camera_matrix), std::begin(info_msg.K));
    std::copy(std::begin(preset_rectification_matrix), std::end(preset_rectification_matrix), std::begin(info_msg.R));
    std::copy(std::begin(preset_projection_matrix), std::end(preset_projection_matrix), std::begin(info_msg.P));
    if ((int)info_msg.width != preset_width || (int)info_msg.height != preset_height)
    {
      cout << "Warning: the camera size does not match preset values!" << std::endl;
    }
  } else
  {
    cout << "Camera info callback OK" << std::endl;
  }
  last_cam_info = info_msg;
  got_cam_info = true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "uav_detect");
  ROS_INFO ("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  // Initialize cout to print with precision to two dec. places
  cout << std::fixed << std::setprecision(2);


  /**Load parameters from ROS* //{*/
  string uav_name, data_file, names_file, cfg_file, weights_file;
  float threshold, hier_threshold;
  bool only_subsquare, double_detection;

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
  nh.param("double_detection", double_detection, true);

  // Load preset camera calibration parameters
  nh.param("image_width", preset_width, -1);
  nh.param("image_height", preset_height, -1);
  nh.param("camera_name", preset_calib_name, string("NONE"));
  nh.param("distortion_model", preset_distortion_model, string("NONE"));
  nh.getParam("camera_matrix/data", preset_camera_matrix);
  nh.getParam("distortion_coefficients/data", preset_distortion_coefficients);
  nh.getParam("rectification_matrix/data", preset_rectification_matrix);
  nh.getParam("projection_matrix/data", preset_projection_matrix);

  cout << "Using parameters:" << std::endl;
  cout << "\tuav name:\t" << uav_name << std::endl;
  cout << "\tdata file:\t" << data_file << std::endl;
  cout << "\tnames file:\t" << names_file << std::endl;
  cout << "\tcfg file:\t" << cfg_file << std::endl;
  cout << "\tweights file:\t" << weights_file << std::endl;
  cout << "\tthreshold:\t" << threshold << std::endl;
  cout << "\thier threshold:\t" << hier_threshold << std::endl;
  cout << "\tonly subsquare:\t" << only_subsquare << std::endl;
  cout << "\tdouble detection:\t" << double_detection << std::endl;
  cout << "\tpreset camera calibration:\t" << preset_calib_name << std::endl;
  //}

  /**Create publishers and subscribers* //{*/
  ros::Publisher detections_pub = nh.advertise<uav_detect::Detections>("detections", 20);
  #ifdef DEBUG
  // Debug only
  #warning "Building with -DDEBUG (turn off in CMakeLists.txt)"
  ros::Publisher det_imgs_pub = nh.advertise<sensor_msgs::Image>("det_imgs", 1);
  #endif //DEBUG
  ros::Subscriber camera_image_sub = nh.subscribe("camera_input", 1, camera_image_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber camera_info_sub = nh.subscribe("camera_info", 1, camera_info_callback, ros::TransportHints().tcpNoDelay());
  //}

  cout << "Creating detector object\n";
  MRS_Detector detector(data_file.c_str(), names_file.c_str(), cfg_file.c_str(), weights_file.c_str(), 0.2, 0.1, 1);
  int w_cnn = 416;
  int h_cnn = 416;
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
      for (auto &det : detections)
      {
        cout << "Object detected!" << std::endl;

        if (double_detection)
        {
          // find top-left corner of an area around the detection with width and height in pixels
          // equal to width and height of the CNN so that it is in bounds of the image
          int l = det.x*w_used-w_cnn/2;
          if (l < 0)
            l = 0;
          int t = det.y*h_used-h_cnn/2;
          if (t < 0)
            t = 0;
          int r = l+w_cnn;
          if (r >= w_used)
          {
            r = w_used-1;
            l = r-w_cnn;
          }
          int b = t+h_cnn;
          if (b >= h_used)
          {
            b = h_used-1;
            t = b-h_cnn;
          }

          cv::Rect around_det(l, t, w_cnn, h_cnn);
          cv::Mat sub_image = det_image(around_det);

          vector<uav_detect::Detection> subdetections = detector.detect(
                sub_image,
                threshold,
                hier_threshold);

          double max_prob = 0.0;
          bool redetected = false;
          uav_detect::Detection best_det;
          for (const auto& subdet : subdetections)
          {
            if (subdet.confidence > max_prob)
            {
              max_prob = subdet.confidence;
              best_det = subdet;
              redetected = true;
            }
          }
          if (redetected)
          {
            best_det.x = (best_det.x*double(w_cnn) + l)/double(w_used);
            best_det.y = (best_det.y*double(h_cnn) + t)/double(h_used);
            best_det.width = best_det.width*double(w_cnn)/double(w_used);
            best_det.height = best_det.height*double(h_cnn)/double(h_used);
            det = best_det;
          }
        }
        sensor_msgs::RegionOfInterest roi;
        roi.x_offset = (det_image.cols - w_used)/2;
        roi.y_offset = (det_image.rows - h_used)/2;
        roi.width = w_used;
        roi.height = h_used;
        det.roi = roi;
        cout << "\t" << detector.get_class_name(det.class_ID) << ", p=" << det.confidence << std::endl;
        cout << "\t[" << det.x << "; " << det.y << "]" << std::endl;
        cout << "\tw=" << det.width << ", h=" << det.height << std::endl;
      }

      uav_detect::Detections msg;
      msg.detections = detections;
      msg.header = last_cam_image_ptr->header;
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
