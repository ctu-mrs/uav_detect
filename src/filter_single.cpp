#include <ros/package.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float64.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Geometry>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <list>

#include <dynamic_reconfigure/server.h>
#include <uav_detect/DepthMapParamsConfig.h>

#include "param_loader.h"
#include "rlcnn_util.h"

#include "ocam_functions.h"

#define cot(x) tan(M_PI_2 - x)

using namespace cv;
using namespace std;
using namespace rlcnn;
using namespace uav_detect;
using namespace Eigen;

extern bool load_successful;

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
extern image_geometry::PinholeCameraModel rlcnn::camera_model;
void dm_cinfo_callback(const sensor_msgs::CameraInfo& dm_cinfo_msg)
{
  if (!got_dm_cinfo)
  {
    ROS_INFO("Got camera image");
    camera_model.fromCameraInfo(dm_cinfo_msg);
    got_dm_cinfo = true;
  }
}

// shortcut type to the dynamic reconfigure manager template instance
typedef DynamicReconfigureMgr<uav_detect::DepthMapParamsConfig> drmgr_t;

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
  // LOAD STATIC PARAMETERS
  ROS_INFO("Loading static parameters:");
  string uav_name = load_param_compulsory<string>(nh, string("uav_name"));
  string uav_frame = load_param(nh, "uav_frame", std::string("fcu_") + uav_name);
  string world_frame = load_param(nh, "world_frame", std::string("local_origin"));
  // Load the camera transformation parameters
  double camera_offset_x = load_param_compulsory<double>(nh, "camera_offset_x");
  double camera_offset_y = load_param_compulsory<double>(nh, "camera_offset_y");
  double camera_offset_z = load_param_compulsory<double>(nh, "camera_offset_z");
  double camera_offset_roll = load_param_compulsory<double>(nh, "camera_offset_roll");
  double camera_offset_pitch = load_param_compulsory<double>(nh, "camera_offset_pitch");
  double camera_offset_yaw = load_param_compulsory<double>(nh, "camera_offset_yaw");
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

  if (!load_successful)
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
      new_dm = false;

      // Construct a new world to camera transform
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

      /* ros::Time cur_t = ros::Time::now(); */

      // TODO: create the actual pointcloud?
      /* cout << last_dm_msg.encoding << std::endl; */
      /* Vector3d ground_pt1(0, 0, 1); */
      /* Vector3d ground_pt2(1, 0, 1); */
      /* Vector3d ground_pt3(0, 1, 1); */
      /* Vector3d ground_normal(0, 0, 1); */
      /* double ground_height = 0.0; */
      /* Hyperplane<double, 3> ground_plane(ground_normal, ground_height); */
      /* ground_plane.transform(w2c_tf.rotation()); */
      /* double mean_dist = 0.0; */
      /* unsigned n_used_px = 0; */
      /* unsigned n_ground_px = 0; */
      /* unsigned n_away_px = 0; */
      /* unsigned n_invalid_px = 0; */

      /* uint32_t im_h = last_dm_msg.height; */
      /* uint32_t im_w = last_dm_msg.width; */

      /* /1* Perform masking of the image //{ *1/ */
      /* for (uint32_t px_it = 0; px_it < im_h*im_w; px_it++) */
      /* { */
      /*   uint16_t cur_dist; */
      /*   uint16_t cur_x = px_it%im_w; */
      /*   uint16_t cur_y = px_it/im_w; */
      /*   if (last_dm_msg.is_bigendian) */
      /*   { */
      /*     cur_dist = last_dm_msg.data[px_it*2 + 0] << 8 */
      /*              | last_dm_msg.data[px_it*2 + 1]; */
      /*   } else */
      /*   { */
      /*     cur_dist = last_dm_msg.data[px_it*2 + 0] */
      /*              | last_dm_msg.data[px_it*2 + 1] << 8; */
      /*   } */
      /*   if (cur_dist == 0) */
      /*   { */
      /*     last_dm_msg.data[px_it*2 + 0] = last_dm_msg.data[px_it*2 + 1] = 255; */
      /*     n_invalid_px++; */
      /*   } else if (cur_dist > max_dist) */
      /*   { */
      /*     last_dm_msg.data[px_it*2 + 0] = last_dm_msg.data[px_it*2 + 1] = 255; */
      /*     n_away_px++; */
      /*   } else */
      /*   { */
      /*     Vector3d cur_pt = calculate_direction_pinhole(cur_x, cur_y)*cur_dist; */
      /*     double dist_above_ground = ground_plane.signedDistance(cur_pt); */
      /*     if (dist_above_ground < height_threshold) */
      /*     { */
      /*       last_dm_msg.data[px_it*2 + 0] = last_dm_msg.data[px_it*2 + 1] = 255; */
      /*       n_ground_px++; */
      /*     } else */
      /*     { */
      /*       mean_dist += cur_dist; */
      /*       n_used_px++; */
      /*     } */
      /*   } */
      /* } */
      /* mean_dist /= n_used_px; */
      /* cout << "Mean distance of background: " << mean_dist << std::endl; */
      /* cout << "number of invalid pixels" << n_invalid_px << std::endl; */
      /* cout << "number of faraway pixels" << n_away_px << std::endl; */
      /* cout << "number of ground pixels" << n_ground_px << std::endl; */
      /* cout << "number of used pixels" << n_used_px << std::endl; */
      /* //} */

      /* /1* Visualize the result //{ *1/ */
      /* for (uint32_t px_it = 0; px_it < im_h*im_w; px_it++) */
      /* { */
      /*   uint16_t cur_dist; */
      /*   if (last_dm_msg.is_bigendian) */
      /*   { */
      /*     cur_dist = last_dm_msg.data[px_it*2 + 0] << 8 */
      /*              | last_dm_msg.data[px_it*2 + 1]; */
      /*   } else */
      /*   { */
      /*     cur_dist = last_dm_msg.data[px_it*2 + 0] */
      /*              | last_dm_msg.data[px_it*2 + 1] << 8; */
      /*   } */
      /*   /1* if (px_it == 0) *1/ */
      /*   /1*   cout << "topleft pixel: " << cur_px << std::endl; *1/ */
      /*   if (cur_dist < mean_dist - bg_dist) */
      /*     last_dm_msg.data[px_it*2 + 0] = last_dm_msg.data[px_it*2 + 1] = 0; */
      /*   else */
      /*     last_dm_msg.data[px_it*2 + 0] = last_dm_msg.data[px_it*2 + 1] = 255; */
      /* } */
      /* //} */

      /* Use OpenCV SimpleBlobDetector to find blobs //{ */
      vector<KeyPoint> keypoints;
      /* cv::Mat out_img(im_h, im_w, CV_16UC1); */
      cv_bridge::CvImage out_img;
      /* out_img.image = cv::Mat(im_h, im_w, CV_16UC1); */
      out_img = *cv_bridge::toCvCopy(last_dm_msg, string("16UC1"));
      cv::Mat detect_im(out_img.image.size(), CV_8UC1);
      out_img.image.convertTo(detect_im, CV_8UC1, 255.0 / max_dist / 1000);
      // output debug image
      cv::Mat tmp;
      out_img.image.convertTo(tmp, CV_8UC1, 255.0 / max_dist / 1000);
      cv::cvtColor(tmp, out_img.image, COLOR_GRAY2BGR);
      out_img.encoding = string("bgr8");
      SimpleBlobDetector::Params params;

      // Filter by color thresholds
      params.filterByColor = blob_filter_by_color;
      params.minThreshold = min_dist * 255.0 / max_dist;
      params.maxThreshold = 254.0;
      // Filter by area.
      params.filterByArea = blob_filter_by_area;
      params.minArea = blob_min_area;
      params.maxArea = blob_max_area;
      // Filter by circularity
      params.filterByCircularity = blob_filter_by_circularity;
      params.minCircularity = blob_min_circularity;
      params.maxCircularity = blob_max_circularity;
      // Filter by convexity
      params.filterByConvexity = blob_filter_by_convexity;
      params.minConvexity = blob_min_convexity;
      params.maxConvexity = blob_max_convexity;
      // Filter by inertia
      params.filterByInertia = blob_filter_by_inertia;
      params.minInertiaRatio = blob_min_inertia_ratio;
      params.maxInertiaRatio = blob_max_inertia_ratio;
      // Other filtering criterions
      params.minDistBetweenBlobs = blob_min_dist_between;
      params.thresholdStep = blob_threshold_step;
      params.minRepeatability = blob_min_repeatability;

      auto detector = SimpleBlobDetector::create(params);
      detector->detect(detect_im, keypoints);

      /* cv::drawKeypoints(out_img.image, keypoints, out_img.image, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG); */
      int potential = 0;
      int unsure = 0;
      int sure = 0;
      for (const auto& kpt : keypoints)
      {
        uint8_t dist = detect_im.at<uint8_t>(kpt.pt);
        if (dist > min_dist * 255.0 / max_dist && dist < 253)
        {
          Eigen::Vector3d pt3d = calculate_direction_pinhole(kpt.pt.x, kpt.pt.y);
          pt3d *= dist;
          pt3d = c2w_tf*pt3d;
          if (pt3d(2) > 0.5)
          {
            sure++;
            cv::circle(out_img.image, kpt.pt, kpt.size, cv::Scalar(0, 0, 255), 3, 8, 0);
          } else
          {
            unsure++;
            cv::circle(out_img.image, kpt.pt, kpt.size, cv::Scalar(0, 255, 0), 3, 8, 0);
          }
        } else
        {
          potential++;
          cv::circle(out_img.image, kpt.pt, kpt.size, cv::Scalar(255, 0, 0), 3, 8, 0);
        }
      }
      cv::putText(out_img.image, string("potential: ") + to_string(potential), Point(0, 40), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 0, 0), 3);
      cv::putText(out_img.image, string("unsure: ") + to_string(unsure), Point(0, 80), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 3);
      cv::putText(out_img.image, string("sure: ") + to_string(sure), Point(0, 120), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
      /* cv::circle(out_img.image, Point(im_h/2, 100), 50, Scalar(255), 10); */
      cout << "Number of keypoints: " << keypoints.size() << std::endl;

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
