#include "main.h"
#include "DepthBlobDetector.h"

using namespace cv;
using namespace std;
using namespace uav_detect;
using namespace Eigen;

// Callback for the depth map
bool new_dm = false;
sensor_msgs::Image last_dm_msg;
void depthmap_callback(const sensor_msgs::Image& dm_msg)
{
  ROS_INFO("Got new image");
  last_dm_msg = dm_msg;
  new_dm = true;
}

// Callback for the depthmap camera info
bool got_dm_cinfo = false;
image_geometry::PinholeCameraModel camera_model;
double d_fx, d_fy, d_cx, d_cy;
void dm_cinfo_callback(const sensor_msgs::CameraInfo& dm_cinfo_msg)
{
  if (!got_dm_cinfo)
  {
    ROS_INFO("Got camera image");
    camera_model.fromCameraInfo(dm_cinfo_msg);
    d_fx = camera_model.fx();
    d_fy = camera_model.fy();
    d_cx = camera_model.cx();
    d_cy = camera_model.cy();
    got_dm_cinfo = true;
  }
}

double get_height(const Eigen::Vector3d& c2w_z, uint16_t px_x, uint16_t px_y, double depth_m)
{
  Eigen::Vector3d pos((px_x-d_cx)/d_fx*depth_m, (px_y-d_cy)/d_fy*depth_m, depth_m);
  return c2w_z.dot(pos);
}

// shortcut type to the dynamic reconfigure manager template instance
typedef mrs_lib::DynamicReconfigureMgr<uav_detect::DepthMapParamsConfig> drmgr_t;

/** Utility functions //{**/
double min_x, max_x;
double min_y, max_y;
double min_z, max_z;
bool position_valid(Eigen::Vector3d pos_vec)
{
  return (pos_vec(0) > min_x && pos_vec(0) < max_x) && (pos_vec(1) > min_y && pos_vec(2) < max_y) && (pos_vec(2) > min_z && pos_vec(2) < max_z);
}
//}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "uav_detect_localize");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Load parameters from ROS * //{*/
  mrs_lib::ParamLoader pl(nh);
  // LOAD STATIC PARAMETERS
  ROS_INFO("Loading static parameters:");
  string uav_name = pl.load_param2<string>(string("uav_name"));
  string uav_frame = pl.load_param2("uav_frame", std::string("fcu_") + uav_name);
  string world_frame = pl.load_param2("world_frame", std::string("local_origin"));
  // Load the camera transformation parameters
  double camera_offset_x = pl.load_param2<double>("camera_offset_x");
  double camera_offset_y = pl.load_param2<double>("camera_offset_y");
  double camera_offset_z = pl.load_param2<double>("camera_offset_z");
  double camera_offset_roll = pl.load_param2<double>("camera_offset_roll");
  double camera_offset_pitch = pl.load_param2<double>("camera_offset_pitch");
  double camera_offset_yaw = pl.load_param2<double>("camera_offset_yaw");
  /* double camera_delay = load_param<double>(nh, "camera_delay");; */
  // LOAD DYNAMIC PARAMETERS
  drmgr_t drmgr;
  // Load the detection parameters
  // Filter by color
  bool& blob_filter_by_color = drmgr.config.blob_filter_by_color;
  double& min_dist = drmgr.config.min_dist;
  double& max_dist = drmgr.config.max_dist;
  // Filter by area
  bool& blob_filter_by_area = drmgr.config.blob_filter_by_area;
  double& blob_min_area = drmgr.config.blob_min_area;
  double& blob_max_area = drmgr.config.blob_max_area;
  // Filter by circularity
  bool& blob_filter_by_circularity = drmgr.config.blob_filter_by_circularity;
  double& blob_min_circularity = drmgr.config.blob_min_circularity;
  double& blob_max_circularity = drmgr.config.blob_max_circularity;
  // Filter by convexity
  bool& blob_filter_by_convexity = drmgr.config.blob_filter_by_convexity;
  double& blob_min_convexity = drmgr.config.blob_min_convexity;
  double& blob_max_convexity = drmgr.config.blob_max_convexity;
  // Filter by inertia
  bool& blob_filter_by_inertia = drmgr.config.blob_filter_by_inertia;
  double& blob_min_inertia_ratio = drmgr.config.blob_min_inertia_ratio;
  double& blob_max_inertia_ratio = drmgr.config.blob_max_inertia_ratio;
  // Other filtering criterions
  double& blob_min_dist_between = drmgr.config.blob_min_dist_between;
  double& blob_threshold_step = drmgr.config.blob_threshold_step;
  int& blob_min_repeatability = drmgr.config.blob_min_repeatability;

  if (!pl.loaded_successfully())
  {
    ROS_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
    ros::shutdown();
  }
  //}

  /** Build the UAV to camera transformation * //{*/
  tf2::Transform uav2camera_transform;
  {
    tf2::Quaternion q;
    tf2::Vector3 origin;
    // camera transformation
    origin.setValue(camera_offset_x, camera_offset_y, camera_offset_z);
    // camera rotation
    q.setRPY(camera_offset_roll / 180.0 * M_PI, camera_offset_pitch / 180.0 * M_PI, camera_offset_yaw / 180.0 * M_PI);

    uav2camera_transform.setOrigin(origin);
    uav2camera_transform.setRotation(q);
  }
  //}

  /** Create publishers and subscribers //{**/
  tf2_ros::Buffer tf_buffer;
  // Initialize transform listener
  tf2_ros::TransformListener* tf_listener = new tf2_ros::TransformListener(tf_buffer);
  // Initialize other subs and pubs
  ros::Subscriber depthmap_sub = nh.subscribe("depth_map", 1, depthmap_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber dm_cinfo_sub = nh.subscribe("camera_info", 1, dm_cinfo_callback, ros::TransportHints().tcpNoDelay());
  ros::Publisher detected_UAV_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("detected_uav", 10);
  ros::Publisher thresholded_pub = nh.advertise<sensor_msgs::Image&>("thresholded_dm", 1);
  ros::Publisher distance_pub = nh.advertise<std_msgs::Float64>("detected_uav_distance", 10);
  //}

  cout << "----------------------------------------------------------" << std::endl;

  ros::Rate r(10);
  while (ros::ok())
  {
    ros::spinOnce();

    // Check if we got a new message containing a depthmap
    /* if (new_dm) */
    if (new_dm && got_dm_cinfo)
    {
      cout << "Processsing image" << std::endl;
      new_dm = false;

      // Construct a new world to camera transform //{
      Eigen::Affine3d c2w_tf;
      geometry_msgs::TransformStamped transform;
      tf2::Transform world2uav_transform;
      tf2::Vector3 origin;
      tf2::Quaternion orientation;
      try
      {
        const ros::Duration timeout(1.0 / 6.0);
        // Obtain transform from world into uav frame
        transform = tf_buffer.lookupTransform(uav_frame, world_frame, last_dm_msg.header.stamp, timeout);
        /* tf2::convert(transform, world2uav_transform); */
        origin.setValue(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);

        orientation.setX(transform.transform.rotation.x);
        orientation.setY(transform.transform.rotation.y);
        orientation.setZ(transform.transform.rotation.z);
        orientation.setW(transform.transform.rotation.w);

        world2uav_transform.setOrigin(origin);
        world2uav_transform.setRotation(orientation);

        // Obtain transform from camera frame into world
        c2w_tf = tf2_to_eigen(uav2camera_transform * world2uav_transform).inverse();
      }
      catch (tf2::TransformException& ex)
      {
        ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", world_frame.c_str(), uav_frame.c_str(), ex.what());
        continue;
      }
      //}

      /* Prepare the image for detection //{ */
      /* cv::Mat out_img(im_h, im_w, CV_16UC1); */
      cv_bridge::CvImage out_img;
      /* out_img.image = cv::Mat(im_h, im_w, CV_16UC1); */
      out_img = *cv_bridge::toCvCopy(last_dm_msg, string("16UC1"));
      cv::Mat tmp;
      cv::inRange(out_img.image, min_dist*1000, max_dist*1000, tmp);
      cv::cvtColor(tmp, out_img.image, COLOR_GRAY2BGR);
      out_img.encoding = string("bgr8");
      cv::Mat detect_im(out_img.image.size(), CV_8UC1);
      //}

      /* Use OpenCV SimpleBlobDetector to find blobs //{ */
      vector<KeyPoint> keypoints;
      dbd::Params params;
      params.filter_by_color = false;
      /* params.color = min_dist * 255.0 / max_dist; */
      params.filter_by_area = blob_filter_by_area;
      params.min_area = blob_min_area;
      params.max_area = blob_max_area;
      params.filter_by_circularity = blob_filter_by_circularity;
      params.min_circularity = blob_min_circularity;
      params.max_circularity = blob_max_circularity;
      params.filter_by_convexity = blob_filter_by_convexity;
      params.min_convexity = blob_min_convexity;
      params.max_convexity = blob_max_convexity;
      params.filter_by_inertia = blob_filter_by_inertia;
      params.min_inertia_ratio = blob_min_inertia_ratio;
      params.max_inertia_ratio = blob_max_inertia_ratio;
      params.min_threshold = 1;
      params.max_threshold = min_dist * 255.0 / max_dist;
      params.threshold_step = blob_threshold_step;
      params.min_repeatability = blob_min_repeatability;
      params.min_dist_between = blob_min_dist_between;

      dbd::DepthBlobDetector detector(params);
      detector.detect(detect_im, keypoints);

      /* cv::drawKeypoints(out_img.image, keypoints, out_img.image, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG); */
      /* int potential = 0; */
      /* int unsure = 0; */
      /* int sure = 0; */
      /* for (const auto& kpt : keypoints) */
      /* { */
      /*   uint8_t dist = detect_im.at<uint8_t>(kpt.pt); */
      /*   if (dist > min_dist * 255.0 / max_dist && dist < 253) */
      /*   { */
             /* double height = get_height(c2w_z, kpt.pt.x, kpt.pt.y, dist); */
      /*     if (height > 0.5) */
      /*     { */
      /*       sure++; */
      /*       cv::circle(out_img.image, kpt.pt, kpt.size, cv::Scalar(0, 0, 255), 3, 8, 0); */
      /*     } else */
      /*     { */
      /*       unsure++; */
      /*       cv::circle(out_img.image, kpt.pt, kpt.size, cv::Scalar(0, 255, 0), 3, 8, 0); */
      /*     } */
      /*   } else */
      /*   { */
      /*     potential++; */
      /*     cv::circle(out_img.image, kpt.pt, kpt.size, cv::Scalar(255, 0, 0), 3, 8, 0); */
      /*   } */
      /* } */
      /* cv::putText(out_img.image, string("potential: ") + to_string(potential), Point(0, 40), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 0, 0), 3); */
      /* cv::putText(out_img.image, string("unsure: ") + to_string(unsure), Point(0, 80), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 3); */
      /* cv::putText(out_img.image, string("sure: ") + to_string(sure), Point(0, 120), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3); */
      /* /1* cv::circle(out_img.image, Point(im_h/2, 100), 50, Scalar(255), 10); *1/ */
      /* cout << "Number of keypoints: " << keypoints.size() << std::endl; */

      //}

      // Finally publish the message
      sensor_msgs::ImagePtr out_msg = out_img.toImageMsg();
      out_msg->header = last_dm_msg.header;
      /* cout << "New encoding: " << out_msg->encoding << std::endl; */
      thresholded_pub.publish(out_msg);

      cout << "Image processed" << std::endl;
    } else
    {
      r.sleep();
    }
  }
  delete tf_listener;
}
