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

/* extern Eigen::Affine3d rlcnn::c2w_tf; */
Eigen::Affine3d w2c_tf;

bool new_dm = false;
sensor_msgs::Image last_dm_msg;
void depthmap_callback(const sensor_msgs::Image& dm_msg)
{
  ROS_INFO("Got new image");
  last_dm_msg = dm_msg;
  new_dm = true;
}

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

typedef DynamicReconfigureMgr<uav_detect::DepthMapParamsConfig> drmgr_t;
drmgr_t drmgr;

/** Utility functions //{**/
double min_x, max_x;
double min_y, max_y;
double min_z, max_z;
bool position_valid(Eigen::Vector3d pos_vec)
{
  return  (pos_vec(0) > min_x && pos_vec(0) < max_x) &&
          (pos_vec(1) > min_y && pos_vec(2) < max_y) &&
          (pos_vec(2) > min_z && pos_vec(2) < max_z);
}
//}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "uav_detect_localize");
  ROS_INFO ("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Load parameters from ROS * //{*/
  string uav_name = load_param_compulsory<string>(nh, string("uav_name"));
  string uav_frame = load_param(nh, "uav_frame", std::string("fcu_") + uav_name);
  string world_frame = load_param(nh, "world_frame", std::string("local_origin"));
  /* double UAV_width = load_param<double>(nh, "UAV_width");; */
  // Load the camera transformation parameters
  double camera_offset_x = load_param_compulsory<double>(nh, "camera_offset_x");
  double camera_offset_y = load_param_compulsory<double>(nh, "camera_offset_y");
  double camera_offset_z = load_param_compulsory<double>(nh, "camera_offset_z");
  double camera_offset_roll = load_param_compulsory<double>(nh, "camera_offset_roll");
  double camera_offset_pitch = load_param_compulsory<double>(nh, "camera_offset_pitch");
  double camera_offset_yaw = load_param_compulsory<double>(nh, "camera_offset_yaw");
  /* double camera_delay = load_param<double>(nh, "camera_delay");; */
  // Load the detection parameters
  /* double height_threshold = load_param<double>(nh, "height_threshold", 1.0);; */
  // Filter by color
  bool &blob_filter_by_color = drmgr.config_latest.blob_filter_by_color;
  double min_dist = load_param<double>(nh, "min_dist", 300.0);
  double max_dist = load_param<double>(nh, "max_dist", 18000.0);
  // Filter by area
  bool blob_filter_by_area = load_param<bool>(nh, "blob_filter_by_area", false);
  double blob_min_area = load_param<double>(nh, "blob_min_area", 200.0);
  double blob_max_area = load_param<double>(nh, "blob_max_area", 921600.0);
  // Filter by circularity
  bool blob_filter_by_circularity = load_param<bool>(nh, "blob_filter_by_circularity", false);
  double blob_min_circularity = load_param<double>(nh, "blob_min_circularity", 0.0);
  double blob_max_circularity = load_param<double>(nh, "blob_max_circularity", 1.0);
  // Filter by convexity
  bool blob_filter_by_convexity = load_param<bool>(nh, "blob_filter_by_convexity", false);
  double blob_min_convexity = load_param<double>(nh, "blob_min_convexity", 0.0);
  double blob_max_convexity = load_param<double>(nh, "blob_max_convexity", 1.0);
  // Filter by inertia
  bool blob_filter_by_inertia = load_param<bool>(nh, "blob_filter_by_inertia", false);
  double blob_min_inertia_ratio = load_param<double>(nh, "blob_min_inertia_ratio", 0.0);
  double blob_max_inertia_ratio = load_param<double>(nh, "blob_max_inertia_ratio", 1.0);
  // Other filtering criterions
  double blob_min_dist_between = load_param<double>(nh, "blob_min_dist_between", 10.0);
  double blob_threshold_step = load_param<double>(nh, "blob_threshold_step", 10.0);
  size_t blob_min_repeatability = load_param<int>(nh, "blob_min_repeatability", 2);
  //}

  /** Build the UAV to camera transformation * //{*/
  tf2::Transform uav2camera_transform;
  {
    tf2::Quaternion q;
    tf2::Vector3    origin;
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
  tf2_ros::TransformListener *tf_listener = new tf2_ros::TransformListener(tf_buffer);
  // Initialize other subs and pubs
  ros::Subscriber depthmap_sub = nh.subscribe("depth_map", 1, depthmap_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber dm_cinfo_sub = nh.subscribe("camera_info", 1, dm_cinfo_callback, ros::TransportHints().tcpNoDelay());
  ros::Publisher detected_UAV_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("detected_uav", 10);
  ros::Publisher thresholded_pub = nh.advertise<sensor_msgs::Image&>("thresholded_dm", 1);
  ros::Publisher distance_pub = nh.advertise<std_msgs::Float64>("detected_uav_distance", 10);

  // dynamic_reconfigure server
  dynamic_reconfigure::Server<uav_detect::DepthMapParamsConfig> server;
  dynamic_reconfigure::Server<uav_detect::DepthMapParamsConfig>::CallbackType f;

  f = boost::bind(&drmgr_t::dynamic_reconfigure_callback, &drmgr, _1, _2);
  server.setCallback(f);
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
      geometry_msgs::TransformStamped transform;
      tf2::Transform  world2uav_transform;
      tf2::Vector3    origin;
      tf2::Quaternion orientation;
      try
      {
        const ros::Duration timeout(1.0/6.0);
        // Obtain transform from world into uav frame
        transform = tf_buffer.lookupTransform(
            uav_frame,
            world_frame,
            last_dm_msg.header.stamp,
            timeout
            );
        /* tf2::convert(transform, world2uav_transform); */
        origin.setValue(
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
            );

       orientation.setX(transform.transform.rotation.x);
       orientation.setY(transform.transform.rotation.y);
       orientation.setZ(transform.transform.rotation.z);
       orientation.setW(transform.transform.rotation.w);

       world2uav_transform.setOrigin(origin);
       world2uav_transform.setRotation(orientation);

       // Obtain transform from camera frame into world
       w2c_tf = tf2_to_eigen(uav2camera_transform * world2uav_transform);
     } catch (tf2::TransformException& ex)
     {
       ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", world_frame.c_str(), uav_frame.c_str(), ex.what());
       continue;
     }
     //}

     ros::Time cur_t = ros::Time::now();

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
     cv::Mat tmp(out_img.image.size(), CV_8UC1);
     out_img.image.convertTo(tmp, CV_8UC1, 255.0/max_dist);
     out_img.image = tmp;
     out_img.encoding = string("8UC1");
      SimpleBlobDetector::Params params;

      // Filter by color thresholds
      params.filterByColor = blob_filter_by_color;
      params.minThreshold = min_dist*255.0/max_dist;
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
      detector->detect(out_img.image, keypoints);

     cv::drawKeypoints(out_img.image, keypoints, out_img.image, Scalar(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
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
