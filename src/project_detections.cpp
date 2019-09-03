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

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <list>

#include <cnn_detect/Detection.h>
#include <cnn_detect/Detections.h>
#include <mrs_lib/ParamLoader.h>

#define cot(x) tan(M_PI_2 - x)

using namespace cv;
using namespace std;
using namespace cnn_detect;

/** Utility functions //{**/
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
  string uav_name, uav_frame, world_frame, calib_path;
  bool use_ocam;
  double UAV_width;
  double max_dist;
  double dist_corr_p0, dist_corr_p1;
  double camera_offset_x, camera_offset_y, camera_offset_z;
  double camera_offset_roll, camera_offset_pitch, camera_offset_yaw;
  double camera_delay;
  double xy_covariance_coeff;
  double z_covariance_coeff;

  ros::init(argc, argv, "cnn_project_detections");
  ROS_INFO ("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Load parameters from ROS * //{*/
  // UAV name
  nh.param("uav_name", uav_name, string());
  if (uav_name.empty())
  {
    ROS_ERROR("UAV_NAME is empty");
    ros::shutdown();
  }
  nh.param("world_frame", world_frame, std::string("local_origin"));
  nh.param("uav_frame", uav_frame, std::string("fcu_uav3"));
  nh.param("calib_path", calib_path, std::string(""));
  nh.param("use_ocam", use_ocam, false);
  if (use_ocam && calib_path.empty())
  {
    ROS_ERROR("Path to OCamCalib calibration file not specified");
    ros::shutdown();
  }

  nh.param("min_x", min_x, -numeric_limits<double>::infinity());
  nh.param("max_x", max_x, numeric_limits<double>::infinity());
  nh.param("min_y", min_y, -numeric_limits<double>::infinity());
  nh.param("max_y", max_y, numeric_limits<double>::infinity());
  nh.param("min_z", min_z, 0.0);
  nh.param("max_z", max_z, numeric_limits<double>::infinity());
  nh.param("UAV_width", UAV_width, 0.28*2.0);
  nh.param("max_dist", max_dist, 15.0);
  nh.param("dist_corr_p0", dist_corr_p0, -2.0);
  nh.param("dist_corr_p1", dist_corr_p1, 0.8);

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
  // camera delay
  nh.param("camera_delay", camera_delay, numeric_limits<double>::infinity());
  if (isinf(camera_delay))
  {
    ROS_ERROR("Camera delay not specified");
    ros::shutdown();
  }
  nh.param("dist_filter_window", dist_filter_window, 3);
  nh.param("xy_covariance_coeff", xy_covariance_coeff, 0.5);
  nh.param("z_covariance_coeff", z_covariance_coeff, 0.1);

  cout << "Using parameters:" << std::endl;
  cout << "\tuav name:\t" << uav_name << std::endl;
  cout << "\tuav frame:\t" << uav_frame << std::endl;
  cout << "\tworld frame:\t" << world_frame << std::endl;
  cout << "\tusing OCamCalib:\t" << use_ocam << std::endl;
  if (use_ocam)
    cout << "\t\tcalibration path:\t" << calib_path << std::endl;

  cout << "\tBounding area:" << std::endl
       << "\t\tx in <" << min_x << ", " << max_x << ">" << std::endl
       << "\t\ty in <" << min_y << ", " << max_y << ">" << std::endl
       << "\t\tz in <" << min_z << ", " << max_z << ">" << std::endl;

  cout << "\tUAV width:\t" << UAV_width << "m" << std::endl;
  cout << "\tdist. correction p0:\t" << dist_corr_p0 << "m" << std::endl;
  cout << "\tdist. correction p1:\t" << dist_corr_p1 << std::endl;
  cout << "\tmax. detection dist.:\t" << max_dist << "m" << std::endl;
  cout << "\tcamera X offset:\t" << camera_offset_x << "m" << std::endl;
  cout << "\tcamera Y offset:\t" << camera_offset_y << "m" << std::endl;
  cout << "\tcamera Z offset:\t" << camera_offset_z << "m" << std::endl;
  cout << "\tcamera roll:\t" << camera_offset_roll << "°" << std::endl;
  cout << "\tcamera pitch:\t" << camera_offset_pitch << "°" << std::endl;
  cout << "\tcamera yaw:\t" << camera_offset_yaw << "°"  << std::endl;
  cout << "\tcamera delay:\t" << camera_delay << "ms" << std::endl;
  cout << "\tdist. filter size:\t" << dist_filter_window << std::endl;
  cout << "\txy covar. coeff.:\t" << xy_covariance_coeff << std::endl;
  cout << "\tz covar. coeff.:\t" << z_covariance_coeff << std::endl;

  filter_initialized = false;
  dist_filter = new double[dist_filter_window];
  
  if (use_ocam)
  {
    // Load OCamCalib calibration
    get_ocam_model(&oc_model, (char*)calib_path.c_str());
  }
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
  tf2_ros::TransformListener tf_listener(tf_buffer);
  // Initialize other subs and pubs
  ros::Subscriber detections_sub = nh.subscribe("detections", 1, detections_callback, ros::TransportHints().tcpNoDelay());
  ros::Publisher detected_UAV_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("detected_uav", 10);
  ros::Publisher distance_pub = nh.advertise<std_msgs::Float64>("detected_uav_distance", 10);
  //}

  cout << "----------------------------------------------------------" << std::endl;

  ros::Rate r(10);
  while (ros::ok())
  {
    ros::spinOnce();

    // Check if we got a new message containing detections
    if (new_detections)
    {
      new_detections = false;
      cout << "Processing "
           << latest_detections.detections.size()
           << " new detections ---------------------------------"
           << std::endl;

      /** Update the transforms //{**/
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
            latest_detections.stamp,
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
      } catch (tf2::TransformException& ex)
      {
        ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", world_frame.c_str(), uav_frame.c_str(), ex.what());
        continue;
      }
      //}

      // updates the following variables:
      // camera_model
      // camera_model_absolute (unused)
      // w_camera
      // h_camera
      // w_used
      // h_used
      update_camera_info(latest_detections);

      ros::Time cur_t = ros::Time::now();

      /** Pick the detection to be used (according to highest probability) //{**/
      Detection ref_det;
      double max_prob = 0;
      bool max_prob_set = false;
      for (const auto& det : latest_detections.detections)
      {
        if (!max_prob_set || det.probability > max_prob)
        {
          max_prob = det.probability;
          max_prob_set = true;
          ref_det = Detection(det);
        }
      }
      //}
      
      // If there was no detection, don't publish anything
      if (!max_prob_set)
      {
        cout << "No detection" << std::endl;
        continue;
      }

      /** Calculate the estimated distance using pinhole camera model projection and using camera rectification //{**/
      bool dist_valid = true;
      double l_px = (ref_det.x_relative-ref_det.w_relative/2.0)*w_used + (w_camera-w_used)/2.0;
      double r_px = (ref_det.x_relative+ref_det.w_relative/2.0)*w_used + (w_camera-w_used)/2.0;
      double y_px = (ref_det.y_relative)*h_used + (h_camera-h_used)/2.0;
      Eigen::Vector3d l_vec;
      Eigen::Vector3d r_vec;
      if (use_ocam)
      {
        l_vec = calculate_direction_ocam(l_px, y_px);
        r_vec = calculate_direction_ocam(r_px, y_px);
      } else
      {
        l_vec = calculate_direction_pinhole(l_px, y_px);
        r_vec = calculate_direction_pinhole(r_px, y_px);
      }

      // now calculate the estimated distance
      double alpha = acos(l_vec.dot(r_vec))/2.0;
      /* double est_dist = dist_corr_p0 + dist_corr_p1*UAV_width/(2.0*tan(ray_angle/2.0)); */
      double est_dist = UAV_width*sin(M_PI_2 - alpha)*(tan(alpha) + cot(alpha));
      est_dist = dist_corr_p0 + dist_corr_p1*est_dist;
      cout << "Estimated distance: " << est_dist << std::endl;
      std_msgs::Float64 dist_msg;
      dist_msg.data = est_dist;
      distance_pub.publish(dist_msg);
      if (isnan(est_dist) || est_dist < 0.0 || est_dist > max_dist)
      {
        dist_valid = false;
        est_dist = est_dist < 0.0 ? 0.0 : max_dist;
        cout << "Invalid estimated distance, cropping to " << est_dist << "m" << std::endl;
      }
      /* est_dist = do_filter(est_dist); */
      //}
      
      /** Calculate the estimated position of the other UAV //{**/
      Eigen::Vector3d c_vec = (l_vec + r_vec)/2.0;
      c_vec = est_dist*c_vec;
      // now calculate the estimated location
      cout << "Estimated location (camera CS): [" << c_vec(0) << ", " << c_vec(1) << ", " << c_vec(2) << "]" << std::endl;
      Eigen::Vector3d pos_vec = c2w_tf*c_vec;
      cout << "Estimated location (world  CS): [" << pos_vec(0) << ", " << pos_vec(1) << ", " << pos_vec(2) << "]" << std::endl;

      // Check whether the estimated position makes sense
      bool pos_valid = true;
      if (!position_valid(pos_vec))
      {
        pos_valid = false;
        cout << "Invalid estimated position, throwing it away" << std::endl;
      }
      //}
      
      /** Calculate the corresponding covariance matrix of the estimated position //{**/
      Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Identity();  // prepare the covariance matrix
      double tol = 1e-9;
      pos_cov(0, 0) = pos_cov(1, 1) = xy_covariance_coeff;
      // if the distance estimation is fishy, give it a very large covariance (the information is probably wrong)
      if (!dist_valid)
      {
        pos_cov(2, 2) = 66.6*z_covariance_coeff;
      } else // otherwise give further detections smaller weights than nearer
      {
        pos_cov(2, 2) = est_dist*sqrt(est_dist)*z_covariance_coeff;
        if (pos_cov(2, 2) < 0.33*z_covariance_coeff)
          pos_cov(2, 2) = 0.33*z_covariance_coeff;
        pos_cov *= 1.0/max_prob;
      }
      // Rotation matrix from the camera CS to the world CS
      Eigen::Matrix3d c2w_rot = c2w_tf.rotation();
      // Find the rotation matrix to rotate the covariance to point in the direction of the estimated position
      Eigen::Vector3d a(0.0, 0.0, 1.0);
      Eigen::Vector3d b = c_vec.normalized();
      Eigen::Vector3d v = a.cross(b);
      double sin_ab = v.norm();
      double cos_ab = a.dot(b);
      Eigen::Matrix3d vec_rot = Eigen::Matrix3d::Identity();
      if (sin_ab < tol)  // unprobable, but possible - then it is identity or 180deg
      {
        if (cos_ab + 1.0 < tol)  // that would be 180deg
        {
          vec_rot << -1.0, 0.0, 0.0,
                     0.0, -1.0, 0.0,
                     0.0, 0.0, 1.0;
        } // otherwise its identity
      } else  // otherwise just construct the matrix
      {
        Eigen::Matrix3d v_x; v_x << 0.0, -v(2), v(1),
                                    v(2), 0.0, -v(0),
                                    -v(1), v(0), 0.0;
        vec_rot = Eigen::Matrix3d::Identity() + v_x + (1-cos_ab)/(sin_ab*sin_ab)*(v_x*v_x);
      }
      pos_cov = vec_rot*pos_cov*vec_rot.transpose();  // rotate the covariance to point in direction of est. position
      pos_cov = c2w_rot*pos_cov*c2w_rot.transpose();  // rotate the covariance into local_origin tf
      //}

      /** Fill the message with the calculated values (and some placeholder values for the rotation) //{**/
      Eigen::Matrix3d rot_mat = Eigen::Matrix3d::Identity();
      Eigen::Matrix3d rot_cov = 666*Eigen::Matrix3d::Identity();
      // Fill the covariance array
      geometry_msgs::PoseWithCovarianceStamped est_pos;
      for (int r_it = 0; r_it < 6; r_it++)
        for (int c_it = 0; c_it < 6; c_it++)
          if (r_it < 3 && c_it < 3)
            est_pos.pose.covariance[r_it + c_it*6] = pos_cov(r_it, c_it);
          else if (r_it >= 3 && c_it >= 3)
            est_pos.pose.covariance[r_it + c_it*6] = rot_cov(r_it-3, c_it-3);
          else
            est_pos.pose.covariance[r_it + c_it*6] = 0.0;
      est_pos.header.stamp = latest_detections.stamp;
      est_pos.header.frame_id = "local_origin";
      est_pos.pose.pose.position.x = pos_vec(0);
      est_pos.pose.pose.position.y = pos_vec(1);
      est_pos.pose.pose.position.z = pos_vec(2);
      Eigen::Quaterniond tmp(rot_mat);
      est_pos.pose.pose.orientation.x = tmp.x();
      est_pos.pose.pose.orientation.y = tmp.y();
      est_pos.pose.pose.orientation.z = tmp.z();
      est_pos.pose.pose.orientation.w = tmp.w();
      //}
      
      // Finally publish the message
      if (pos_valid)
      {
        detected_UAV_pub.publish(est_pos);
      }

      cout << "Detections processed" << std::endl;
    } else
    {
      r.sleep();
    }

  }
  delete tf_listener;
  delete[] dist_filter;
}
