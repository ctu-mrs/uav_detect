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
  ros::init(argc, argv, "cnn_detect");
  ROS_INFO ("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  // Initialize cout to print with precision to two dec. places
  cout << std::fixed << std::setprecision(2);

  /**Load parameters from ROS* //{*/
  mrs_lib::ParamLoader pl(nh);
  string uav_name, data_file, names_file, cfg_file, weights_file;
  float threshold, hier_threshold;
  bool double_detection;

  // UAV name
  uav_name = pl.load_param2<string>("uav_name");
  // Data file of the neural network
  data_file = pl.load_param2<string>("data_file");
  // Names file of the neural network
  names_file = pl.load_param2<string>("names_file");
  // Configuration file of the neural network
  cfg_file = pl.load_param2<string>("cfg_file");
  // Weights file of the neural network
  weights_file = pl.load_param2<string>("weights_file");
  // Detection threshold
  threshold = pl.load_param2<double>("threshold", 0.1f);
  // Detection hier threshold
  hier_threshold = pl.load_param2<double>("hier_threshold", 0.1f);
  // Whether to use only subsquare from the image
  /* only_subsquare = pl.load_param2<bool>("only_subsquare", true); */
  double_detection = pl.load_param2<bool>("double_detection", true);
  int ocl_platform_id = pl.load_param2<int>("ocl_platform_id", 0);
  if (ocl_platform_id < 0)
    ocl_platform_id = 0;
  int ocl_device_id = pl.load_param2<int>("ocl_device_id", 0);
  if (ocl_device_id < 0)
    ocl_device_id = 0;

  sensor_msgs::RegionOfInterest roi;
  roi.x_offset = pl.load_param2<int>("roi_x_offset", 0);
  roi.y_offset = pl.load_param2<int>("roi_y_offset", 0);
  roi.width = pl.load_param2<int>("roi_width", -1);
  roi.height = pl.load_param2<int>("roi_height", -1);
  /* bool use_roi = roi.width > 0 && roi.height > 0; */
  /* cout << (use_roi ? "" : "not ") << "using ROI" << endl; */

  /* // Load preset camera calibration parameters */
  /* preset_width = pl.load_param2<int>("image_width", -1); */
  /* preset_height = pl.load_param2<int>("image_height", -1); */
  /* preset_calib_name = pl.load_param2<string>("camera_name", string("NONE")); */
  /* preset_distortion_model = pl.load_param2<string>("distortion_model", string("NONE")); */
  /* preset_camera_matrix = pl.load_param2<double>("camera_matrix/data", */ 
  /* preset_distortion_coefficients = pl.load_param2<double>("distortion_coefficients/data", */ 
  /* preset_rectification_matrix = pl.load_param2<double>("rectification_matrix/data", */ 
  /* preset_projection_matrix = pl.load_param2<double>("projection_matrix/data", */ 
  //}

  /**Create publishers and subscribers* //{*/
  ros::Publisher detections_pub = nh.advertise<cnn_detect::Detections>("detections", 20);
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
  cout << "Initializing detector object using platform id " << ocl_platform_id << " and device id " << ocl_device_id << "\n";
  if (!detector.initialize(ocl_platform_id, ocl_device_id))
  {
    cerr << "Failed to initialize detector, ending node" << std::endl;
    return 1;
  }

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

      if (long(roi.y_offset) + roi.height > h_used)
         roi.height = std::min(h_used, int(h_used - roi.y_offset));
      if (long(roi.x_offset) + roi.width > w_used)
         roi.width = std::min(w_used, int(w_used - roi.x_offset));

      h_used = roi.height;
      w_used = roi.width;
      cv::Rect roi_rect(roi.x_offset, roi.y_offset, roi.width, roi.height);
      det_image = det_image(roi_rect);  // a SHALLOW copy! sub_image shares pixels
      cout << roi << endl;

      vector<cnn_detect::Detection> detections = detector.detect(
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

          vector<cnn_detect::Detection> subdetections = detector.detect(
                sub_image,
                threshold,
                hier_threshold);

          double max_prob = 0.0;
          bool redetected = false;
          cnn_detect::Detection best_det;
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
        det.roi = roi;
        cout << "\t" << detector.get_class_name(det.class_ID) << ", p=" << det.confidence << std::endl;
        cout << "\t[" << det.x << "; " << det.y << "]" << std::endl;
        cout << "\tw=" << det.width << ", h=" << det.height << std::endl;
      }

      cnn_detect::Detections msg;
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
