#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <image_geometry/pinhole_camera_model.h>
#include <dynamic_reconfigure/server.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/transform_listener.h>
#include <mrs_msgs/TrackerPoint.h>
#include <mrs_msgs/TrackerPointStamped.h> // ros message to set desired goal
#include <mrs_msgs/TrackerTrajectory.h> // ros message to set desired trajectory
#include <std_srvs/Empty.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <memory>

#include <uav_detect/Detection.h>
#include <uav_detect/Detections.h>
#include "uav_detect/pid_2DConfig.h"

#include <rlcnn_util.h>

using namespace cv;
using namespace std;
using namespace rlcnn;
using namespace Eigen;
using namespace uav_detect;

/**Callbacks* //{*/
bool new_detections = false;
bool go = false;
uav_detect::Detections dets_msg;

void detections_callback(const uav_detect::Detections& msg)
{
  ROS_INFO("Got new detections");
  dets_msg = msg;
  new_detections = true;
}

bool new_odom = false;
nav_msgs::Odometry odom_msg;
void odom_callback(const nav_msgs::Odometry& msg)
{
  ROS_INFO("Got new odom");
  odom_msg = msg;
  new_odom = true;
}

double Kpx, Kix, Kdx, max_windupx;
double Kpz, Kiz, Kdz, max_windupz;
double z_setpoint;
void reconf_callback(uav_detect::pid_2DConfig &config, uint32_t level)
{
  ROS_INFO("Reconfigure Request: %f %f %f %f", config.Kpx, config.Kix, config.Kdx, config.max_windupx);
  ROS_INFO("                     %f %f %f %f", config.Kpz, config.Kiz, config.Kdz, config.max_windupz);
  ROS_INFO("                     %f", config.z_setpoint);
  Kpx = config.Kpx;
  Kix = config.Kix;
  Kdx = config.Kdx;
  max_windupx = config.max_windupx;
  Kpz = config.Kpz;
  Kiz = config.Kiz;
  Kdz = config.Kdz;
  max_windupz = config.max_windupz;
  z_setpoint = config.z_setpoint;
}

bool service_start_callback(std_srvs::Empty::Request  &req, std_srvs::Empty::Response &res )
{
  ROS_INFO("Starting following 2D");
  go = true;
  return true;
}


bool service_stop_callback(std_srvs::Empty::Request  &req, std_srvs::Empty::Response &res )
{
  ROS_INFO("Stopping following 2D");
  go = false;
  return true;
}
//}

/**Utility functions for leader-follower* //{*/
Detection find_closest(const vector<Detection>& dets, const Detection& ref_det)
{
  double smallest_dist = 0.0;
  bool dist_initialized = false;
  Detection closest_det;

  for (const auto& det : dets)
  {
    double cur_dist = (det.x_relative-ref_det.x_relative)*(det.x_relative-ref_det.x_relative)
                    + (det.y_relative-ref_det.y_relative)*(det.y_relative-ref_det.y_relative);
    if (cur_dist < smallest_dist || !dist_initialized)
    {
      smallest_dist = cur_dist;
      dist_initialized = true;
      closest_det = det;
    }
  }
  return closest_det;
}

Detection weighted_average(const Detection& new_det, const Detection& ref_det)
{
  Detection ret;
  ret.probability = new_det.probability + ref_det.probability;
  ret.x_relative = (new_det.probability*new_det.x_relative + ref_det.probability*ref_det.x_relative)/ret.probability;
  ret.y_relative = (new_det.probability*new_det.y_relative + ref_det.probability*ref_det.y_relative)/ret.probability;
  ret.w_relative = (new_det.probability*new_det.w_relative + ref_det.probability*ref_det.w_relative)/ret.probability;
  ret.h_relative = (new_det.probability*new_det.h_relative + ref_det.probability*ref_det.h_relative)/ret.probability;
  return ret;
}

double *dist_filter;
int dist_filter_window;
bool filter_initialized;
double do_filter(double new_val)
{
  if (!filter_initialized)
    for (int it = 0; it < dist_filter_window; it++)
      dist_filter[it] = new_val;
  filter_initialized = true;
  double ret = 0.0;
  for (int it = 0; it < dist_filter_window; it++)
    ret += dist_filter[it];
  for (int it = 0; it < dist_filter_window-1; it++)
    dist_filter[it] = dist_filter[it+1];
  dist_filter[dist_filter_window-1] = new_val;
  return ret/dist_filter_window;
}
//}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "leader_follower_2D");
  ROS_INFO ("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  dynamic_reconfigure::Server<uav_detect::pid_2DConfig> reconf_server;
  dynamic_reconfigure::Server<uav_detect::pid_2DConfig>::CallbackType reconf;
  reconf = boost::bind(&reconf_callback, _1, _2);
  reconf_server.setCallback(reconf);

  ros::ServiceServer service_start, service_stop;

  /**Load parameters from ROS* //{*/
  string uav_name, uav_frame, world_frame;
  double dist_corr_p0, dist_corr_p1, max_dist;
  double camera_offset_x, camera_offset_y, camera_offset_z;
  double camera_offset_roll, camera_offset_pitch, camera_offset_yaw;
  double camera_desired_angle;
  double camera_offset_angle;
  double UAV_width;
  double z;
  bool x_is_const;

  // UAV name
  nh.param("uav_name", uav_name, string());
  if (uav_name.empty())
  {
    ROS_ERROR("UAV_NAME is empty");
    ros::shutdown();
  }
  nh.param("world_frame", world_frame, std::string("local_origin"));
  nh.param("uav_frame", uav_frame, std::string("fcu_uav1"));

  //
  nh.param("dist_corr_p0", dist_corr_p0, 2.0);
  nh.param("dist_corr_p1", dist_corr_p1, 0.33);
  nh.param("dist_filter_window", dist_filter_window, 3);
  nh.param("max_dist", max_dist, 15.0);
  
  // camera x offset
  nh.param("camera_offset_x", camera_offset_x, numeric_limits<double>::infinity());
  if (isinf(camera_offset_x))
  {
    ROS_ERROR("Camera X offset not specified");
    ros::shutdown();
  }
  // camera y offset
  nh.param("camera_offset_y", camera_offset_y, numeric_limits<double>::infinity());
  if (isinf(camera_offset_y))
  {
    ROS_ERROR("Camera Y offset not specified");
    ros::shutdown();
  }
  // camera z offset
  nh.param("camera_offset_z", camera_offset_z, numeric_limits<double>::infinity());
  if (isinf(camera_offset_z))
  {
    ROS_ERROR("Camera Z offset not specified");
    ros::shutdown();
  }
  // camera roll rotation
  nh.param("camera_offset_roll", camera_offset_roll, numeric_limits<double>::infinity());
  if (isinf(camera_offset_roll))
  {
    ROS_ERROR("Camera roll not specified");
    ros::shutdown();
  }
  // camera pitch rotation
  nh.param("camera_offset_pitch", camera_offset_pitch, numeric_limits<double>::infinity());
  if (isinf(camera_offset_pitch))
  {
    ROS_ERROR("Camera pitch not specified");
    ros::shutdown();
  }
  // camera yaw rotation
  nh.param("camera_offset_yaw", camera_offset_yaw, numeric_limits<double>::infinity());
  if (isinf(camera_offset_yaw))
  {
    ROS_ERROR("Camera yaw not specified");
    ros::shutdown();
  }

  // desired camera yaw angle
  nh.param("camera_desired_angle", camera_desired_angle, numeric_limits<double>::infinity());
  if (isinf(camera_desired_angle))
  {
    ROS_ERROR("Desired camera yaw angle not specified");
    ros::shutdown();
  }
  // offset camera yaw angle
  nh.param("camera_offset_angle", camera_offset_angle, numeric_limits<double>::infinity());
  if (isinf(camera_offset_angle))
  {
    ROS_ERROR("Offset camera yaw angle not specified");
    ros::shutdown();
  }
  nh.param("UAV_width", UAV_width, 0.28*2.0);
  
  // PID parameters for x following
  nh.param("Kpx", Kpx, 26.0);
  nh.param("Kix", Kix, 10.0);
  nh.param("Kdx", Kdx, 0.0);
  nh.param("max_windupx", max_windupx, 0.1);
  // PID parameters for z following (in camera coordinates)
  nh.param("Kpz", Kpz, 1.0);
  nh.param("Kiz", Kiz, 0.1);
  nh.param("Kdz", Kdz, 0.0);
  nh.param("max_windupz", max_windupz, 1.0);
  // leader-follower parameters
  nh.param("z", z, 10.0);
  nh.param("x_is_const", x_is_const, false);
  nh.param("z_setpoint", z_setpoint, 7.0);

  cout << "Using parameters:" << std::endl;
  cout << "\tuav name:\t" << uav_name << std::endl;
  cout << "\tuav frame:\t" << uav_frame << std::endl;
  cout << "\tworld frame:\t" << world_frame << std::endl;
  cout << "\tcamera X offset:\t" << camera_offset_x << "m" << std::endl;
  cout << "\tcamera Y offset:\t" << camera_offset_y << "m" << std::endl;
  cout << "\tcamera Z offset:\t" << camera_offset_z << "m" << std::endl;
  cout << "\tcamera roll:\t" << camera_offset_roll << "°" << std::endl;
  cout << "\tcamera pitch:\t" << camera_offset_pitch << "°" << std::endl;
  cout << "\tcamera yaw:\t" << camera_offset_yaw << "°"  << std::endl;
  cout << "\tcamera desired yaw angle:\t" << camera_desired_angle << "°" << std::endl;
  cout << "\tcamera offset yaw angle:\t" << camera_offset_angle << "°" << std::endl;
  cout << "\tUAV width:\t" << UAV_width << "ms" << std::endl;
  cout << "\tKpx:\t" << Kpx << std::endl;
  cout << "\tKix:\t" << Kix << std::endl;
  cout << "\tKdx:\t" << Kdx << std::endl;
  cout << "\tmax_windupx:\t" << max_windupx << std::endl;
  cout << "\tKpz:\t" << Kpz << std::endl;
  cout << "\tKiz:\t" << Kiz << std::endl;
  cout << "\tKdz:\t" << Kdz << std::endl;
  cout << "\tmax_windupz:\t" << max_windupz << std::endl;
  cout << "\tz:\t" << z << "m" << std::endl;
  cout << "\tsetpoint for z distance:\t" << z_setpoint << std::endl;

  filter_initialized = false;
  dist_filter = new double[dist_filter_window];
  //}

  /**Build the UAV to camera transformation* //{*/
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
  double uav_yaw = (camera_desired_angle-camera_offset_angle)/180.0*M_PI;

  /**Create publishers and subscribers* //{*/
  bool new_tf = false;
  tf2_ros::Buffer tf_buffer;
  // Initialize transform listener
  tf2_ros::TransformListener *tf_listener = new tf2_ros::TransformListener(tf_buffer);
  // Initialize other subs and pubs
  ros::Subscriber detections_sub = nh.subscribe("detections", 1, detections_callback, ros::TransportHints().tcpNoDelay());
  string odom_topic = string("/")+uav_name+string("/mrs_odometry/new_odom");
  ros::Subscriber odom_sub = nh.subscribe(odom_topic, 1, odom_callback, ros::TransportHints().tcpNoDelay());
  /* string setp_topic = string("/")+uav_name+string("/trackers_manager/mpc_tracker/desired_trajectory"); */
  /* ros::Publisher setp_pub = nh.advertise<mrs_msgs::TrackerTrajectory>(setp_topic.c_str(), 1); */
  string setp_topic = string("/")+uav_name+string("/trackers_manager/mpc_tracker/desired_position");
  ros::Publisher setp_pub = nh.advertise<mrs_msgs::TrackerPointStamped>(setp_topic.c_str(), 1);
  
  // create Services
  service_start = nh.advertiseService("start_lf2D", &service_start_callback);
  service_stop = nh.advertiseService("stop_lf2D", &service_stop_callback);
  //}

  cout << "----------------------------------------------------------" << std::endl;

  Detection ref_det;
  bool det_initialized = false;

  while (ros::ok())
  {
    ros::spinOnce();

    ros::Rate r(10);

    if (!new_detections)
    {
      r.sleep();
      continue;
    }

    /**Update the transforms* //{*/
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
          dets_msg.stamp,
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
      c2w_tf = tf2_to_eigen((uav2camera_transform * world2uav_transform).inverse());
      new_tf = true;
    } catch (tf2::TransformException& ex)
    {
      ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", world_frame.c_str(), "usb_cam", ex.what());
      new_tf = false;
      continue;
    }
    //}

    // Check if we have all new required messages
    if (new_detections && new_odom && new_tf)
    {
      new_detections = false;
      cout << "Processing "
           << dets_msg.detections.size()
           << " new detections ---------------------------------"
           << std::endl;

      // updates the following variables:
      // camera_model
      // camera_model_absolute (unused)
      // w_camera
      // h_camera
      // w_used
      // h_used
      update_camera_info(dets_msg);

      if (!det_initialized)
      {
        if (dets_msg.detections.size() == 1)
        {
          ref_det = dets_msg.detections.at(0);
          det_initialized = true;
        }
      } else
      {
        if (dets_msg.detections.size() > 0)
        {
          Detection new_det = find_closest(dets_msg.detections, ref_det);

          ref_det = new_det;
        }

        ros::Time cur_t = ros::Time::now();
        bool dist_valid = true;

        /**Calculate the estimated distance using pinhole camera model projection* //{*/
        /* // left point on the other UAV */
        /* cv::Point2d det_l_pt(ref_det.x_relative-ref_det.w_relative/2.0, */
        /*                      ref_det.y_relative); */
        /* if (camera_model_absolute) */
        /*   det_l_pt = cv::Point2e((ref_det.x_relative-ref_det.w_relative/2.0)*dets_msg.w_used, */
        /*                          (ref_det.y_relative)*dets_msg.h_used); */
        /* /1* det_l_pt = camera_model.rectifyPoint(det_l_pt);  // do not forget to rectify the points! // rectification doesnt work :(( *1/ */
        /* cv::Point3d cvl_ray = camera_model.projectPixelTo3dRay(det_l_pt); */
        /* Eigen::Vector3d l_ray(cvl_ray.x, cvl_ray.y, cvl_ray.z); */
        /* // right point on the other UAV */
        /* cv::Point2d det_r_pt(ref_det.x_relative+ref_det.w_relative/2.0, */
        /*                      ref_det.y_relative); */
        /* if (camera_model_absolute) */
        /*   det_r_pt = cv::Point2e((ref_det.x_relative+ref_det.w_relative/2.0)*dets_msg.w_used, */
        /*                          (ref_det.y_relative)*dets_msg.h_used); */
        /* /1* det_r_pt = camera_model.rectifyPoint(det_r_pt); *1/ */
        /* cv::Point3d cvr_ray = camera_model.projectPixelTo3dRay(det_r_pt); */
        /* Eigen::Vector3d r_ray(cvr_ray.x, cvr_ray.y, cvr_ray.z); */
        /* // now calculate the estimated distance */
        /* double ray_angle = acos(l_ray.dot(r_ray)/(l_ray.norm()*r_ray.norm())); */
        /* double est_dist = dist_corr_p0 + dist_corr_p1*UAV_width/(2.0*tan(ray_angle/2.0)); */
        /* if (isnan(est_dist) || est_dist < 0.0 || est_dist > max_dist) */
        /*   dist_valid = false; */
        /* if (dist_valid) */
        /*   est_dist = do_filter(est_dist); */
        /* cout << "Estimated distance: " << est_dist << std::endl; */
        /* //} */

        /**Calculate the estimated distance using pinhole camera model projection and using rectification* //{*/
        // left point on the other UAV
        cv::Point2d det_l_pt((ref_det.x_relative-ref_det.w_relative/2.0)*w_used + (w_camera-w_used)/2.0,
                             (ref_det.y_relative)*h_used + (h_camera-h_used)/2.0);
        det_l_pt = camera_model.rectifyPoint(det_l_pt);  // do not forget to rectify the points!
        cv::Point3d cvl_ray = camera_model.projectPixelTo3dRay(det_l_pt);
        Eigen::Vector3d l_ray(cvl_ray.x, cvl_ray.y, cvl_ray.z);
        // right point on the other UAV
        cv::Point2d det_r_pt((ref_det.x_relative+ref_det.w_relative/2.0)*w_used + (w_camera-w_used)/2.0,
                             (ref_det.y_relative)*h_used + (h_camera-h_used)/2.0);
        det_r_pt = camera_model.rectifyPoint(det_r_pt);
        cv::Point3d cvr_ray = camera_model.projectPixelTo3dRay(det_r_pt);
        Eigen::Vector3d r_ray(cvr_ray.x, cvr_ray.y, cvr_ray.z);
        // now calculate the estimated distance
        double ray_angle = acos(l_ray.dot(r_ray)/(l_ray.norm()*r_ray.norm()));
        double est_dist = dist_corr_p0 + dist_corr_p1*UAV_width/(2.0*tan(ray_angle/2.0));
        if (isnan(est_dist) || est_dist < 0.0 || est_dist > max_dist)
          dist_valid = false;
        if (dist_valid)
          est_dist = do_filter(est_dist);
        cout << "Estimated distance: " << est_dist << std::endl;
        //}
        
        /*Calculate the estimated position of the other UAV* //{*/
        // left point on the other UAV
        cv::Point2d det_pt((ref_det.x_relative)*w_used + (w_camera-w_used)/2.0,
                           (ref_det.y_relative)*h_used + (h_camera-h_used)/2.0);
        det_pt = camera_model.rectifyPoint(det_pt);  // do not forget to rectify the points!
        cv::Point3d cv_ray = camera_model.projectPixelTo3dRay(det_pt);
        Eigen::Vector3d ray(cv_ray.x, cv_ray.y, cv_ray.z);
        ray = est_dist*ray;
        // now calculate the estimated location
        cout << "Estimated location (camera CS): [" << ray(0) << ", " << ray(1) << ", " << ray(2) << "]" << std::endl;
        ray = c2w_tf*ray;
        cout << "Estimated location (world  CS): [" << ray(0) << ", " << ray(1) << ", " << ray(2) << "]" << std::endl;
        //}
        
        /**Calculate PID for (camera) x coordinate* //{*/
        /* double ex = ref_det.x_relative - 0.5; */
        /* cout << "Relative position error:\t" << ex << std::endl; */
        /* iex += ex*dt; */
        /* if (iex > max_windupx) */
        /*   iex = max_windupx; */
        /* else if (iex < -max_windupx) */
        /*   iex = -max_windupx; */
        /* cout << "\t\tintegrated:\t" << iex << std::endl; */
        /* double dex = (ex - prev_ex)/dt; */
        /* cout << "\t\tderivated:\t" << dex << std::endl; */
        /* double ux = (Kpx*ex + Kix*iex + Kdx*dex); */
        /* prev_ex = ex; */
        /* //} */

        /**Calculate PID for (camera) z coordinate* //{*/
        /* double ez = 0.0, uz = 0.0; */
        /* if (dist_valid) */
        /* { */
        /*   ez = est_dist - z_setpoint; */
        /*   cout << "Relative distance error:\t" << ez << std::endl; */
        /* } else */
        /*   cout << "Invalid distance estimation:\t" << est_dist << std::endl; */
        /* iez += ez*dt; */
        /* if (iez > max_windupz) */
        /*   iez = max_windupz; */
        /* else if (iez < -max_windupz) */
        /*   iez = -max_windupz; */
        /* cout << "\t\tintegrated:\t" << iez << std::endl; */
        /* double dez = (ez - prev_ez)/dt; */
        /* cout << "\t\tderivated:\t" << dez << std::endl; */
        /* uz = (Kpz*ez + Kiz*iez + Kdz*dez); */
        /* prev_ez = ez; */
        /* //} */

        /**Calculate the new speed setpoint* //{*/
        /* mrs_msgs::TrackerTrajectory new_traj; */
        /* new_traj.points.reserve(traj_n_pts); */
        /* double last_pos_x = odom_msg.pose.pose.position.x; */
        /* double last_pos_y = odom_msg.pose.pose.position.y; */
        /* /1* // Calculate estimated position of the other UAV *1/ */
        /* /1* cv::Point2d det_pt(ref_det.x_relative, ref_det.y_relative); *1/ */
        /* /1* det_pt = camera_model.rectifyPoint(det_pt);  // do not forget to rectify the points! *1/ */
        /* /1* cv::Point3d cv_ray = camera_model.projectPixelTo3dRay(det_pt); *1/ */
        /* /1* Eigen::Vector3d other_pos(cv_ray.x, cv_ray.y, cv_ray.z); *1/ */
        /* /1* other_pos = c2w_tf*(other_pos.normalized()*est_dist); *1/ */
        /* /1* // Calculate angle between the two in the XY plane *1/ */
        /* /1* double angle = atan2(other_pos(1)-last_pos_y, other_pos(0)-last_pos_x); *1/ */
        /* // Find orientation of the camera (most importantly the yaw) */
        /* /1* Eigen::Vector3d cam_z_vec(0.0, 0.0, 1.0); *1/ */
        /* /1* cam_z_vec = c2w_tf.rotation()*cam_z_vec; *1/ */
        /* /1* double yaw = atan2(cam_z_vec(1), cam_z_vec(0)); *1/ */
        /* tf2::Vector3 x_vec(1.0, 0.0, 0.0); */
        /* x_vec = world2uav_transform.getBasis().inverse()*x_vec; */
        /* double cur_yaw = atan2(x_vec.y(), x_vec.x()); */
        /* // Calculate the speed vector */
        /* double new_speed_x, new_speed_y; */ 
        /* bool setp_valid = true; */
        /* if (dist_valid) */
        /* { */
        /*   new_speed_x = uz*cos(cur_yaw) + ux*sin(cur_yaw); */
        /*   new_speed_y = uz*sin(cur_yaw) + ux*cos(cur_yaw); */
        /* } else */
        /* { */
        /*   new_speed_x = ux*sin(cur_yaw); */
        /*   new_speed_y = ux*cos(cur_yaw); */
        /* } */

        /* for (int i = 0; i < traj_n_pts; i++) */
        /* { */
        /*   mrs_msgs::TrackerPoint new_target; */
        /*   if (i == 0) */
        /*   { */
        /*     new_target.x = last_pos_x + new_speed_x*traj_dt1; */
        /*     new_target.y = last_pos_y + new_speed_y*traj_dt1; */
        /*   } */
        /*   else */
        /*   { */
        /*     new_target.x = last_pos_x + new_speed_x*traj_dt2; */
        /*     new_target.y = last_pos_y + new_speed_y*traj_dt2; */
        /*   } */
        /*   last_pos_x = new_target.x; */
        /*   last_pos_y = new_target.y; */
        /*   new_target.yaw = uav_yaw; */
        /*   new_target.z = 5.0; */
        /*   if (isnan(new_target.x) || isinf(new_target.x) || */
        /*       isnan(new_target.y) || isinf(new_target.y) || */
        /*       isnan(new_target.z) || isinf(new_target.z) || */
        /*       isnan(new_target.yaw) || isinf(new_target.yaw)) */
        /*   { */
        /*     setp_valid = false; */
        /*   } */
        /*   new_traj.points.push_back(new_target); */
        /* } */
        /* //} */
        
        if (go)
        {
          mrs_msgs::TrackerPointStamped new_setp;
          new_setp.header.frame_id = world_frame;
          new_setp.header.stamp = ros::Time::now();
          new_setp.use_yaw = true;
          new_setp.position.x = ray(0);
          new_setp.position.y = ray(1) - z_setpoint;
          new_setp.position.z = z;
          new_setp.position.yaw = uav_yaw;
          setp_pub.publish(new_setp);
        }
      }



      cout << "Detections processed" << std::endl;
    }

    r.sleep();
  }
  delete tf_listener;
  delete[] dist_filter;
}
