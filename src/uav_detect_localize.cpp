/**
 * This program tries to localize other MAVs using the detected bounding boxes.
 * Detected bounding boxes spawn Kalman Filters.
 * Each Kalman Filter predicts its corresponding bounding box.
 * For each detection the Kalman Filtres from previous iteration are matched
 * to new bounding boxes (according to center distance?).
 * **/
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>

#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <list>

#include <uav_detect/Detection.h>
#include <uav_detect/Detections.h>

#include "detected_uav.h"

using namespace cv;
using namespace std;

bool new_detections = false;
uav_detect::Detections latest_detections;

void detections_callback(const uav_detect::Detections& dets_msg)
{
  latest_detections = dets_msg;
  new_detections = true;
}

int main(int argc, char **argv)
{
  string uav_name, uav_frame, world_frame;
  double camera_offset_x, camera_offset_y, camera_offset_z;
  double camera_offset_roll, camera_offset_pitch, camera_offset_yaw;
  double camera_delay;
  double ass_thresh, unr_thresh, sim_thresh;

  ros::init(argc, argv, "uav_detect_localize");
  ROS_INFO ("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Load parameters from ROS *//*//{*/
  // UAV name
  nh.param("uav_name", uav_name, string());
  if (uav_name.empty())
  {
    ROS_ERROR("UAV_NAME is empty");
    ros::shutdown();
  }
  nh.param("world_frame", world_frame, std::string("local_origin"));
  nh.param("uav_frame", uav_frame, std::string("fcu_uav1"));
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

  // detection association threshold
  nh.param("ass_thresh", ass_thresh, numeric_limits<double>::infinity());
  if (isinf(ass_thresh))
  {
    ROS_ERROR("Detection association threshold not specified");
    ros::shutdown();
  }
  // unreliability of detected UAVs threshold
  nh.param("unr_thresh", unr_thresh, numeric_limits<double>::infinity());
  if (isinf(unr_thresh))
  {
    ROS_ERROR("Detection unreliability threshold not specified");
    ros::shutdown();
  }
  // similarity of detected UAVs threshold
  nh.param("sim_thresh", sim_thresh, numeric_limits<double>::infinity());
  if (isinf(sim_thresh))
  {
    ROS_ERROR("Detection association threshold not specified");
    ros::shutdown();
  }

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
  cout << "\tcamera delay:\t" << camera_delay << "ms" << std::endl;
  cout << "\tassociation threshold:\t" << ass_thresh << "ms" << std::endl;
  cout << "\tunreliability threshold:\t" << unr_thresh << "ms" << std::endl;
  cout << "\tsimilarity threshold:\t" << sim_thresh << "ms" << std::endl;
  /*//}*/

  // build the UAV to camera transformation//{
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

    camera_delay = camera_delay/1000.0; // recalculate to seconds
  }/*//}*/

  /** Create publishers and subscribers **//*//{*/
  tf2_ros::Buffer tf_buffer;
  ros::Subscriber detections_sub = nh.subscribe("detections", 1, detections_callback, ros::TransportHints().tcpNoDelay());
  // Initialize transform listener
  tf2_ros::TransformListener* tf_listener = new tf2_ros::TransformListener(tf_buffer);//}

  cout << "----------------------------------------------------------" << std::endl;

  list<Detected_UAV> detUAVs;
  while (ros::ok())
  {
    ros::spinOnce();

    ros::Rate r(10);
    // Check if we got a new message containing detections
    if (new_detections)
    {
      new_detections = false;
      cout << "Processing new detections ---------------------------------" << std::endl;

      // First, update the transforms
      geometry_msgs::TransformStamped transform;
      tf2::Transform  world2uav_transform, camera2world_transform;
      tf2::Vector3    origin;
      tf2::Quaternion orientation;
      try
      {
        const ros::Duration timeout(1.0/6.0);
        // Obtain transform from world into uav frame
        transform = tf_buffer.lookupTransform(uav_frame, world_frame, latest_detections.stamp - ros::Duration(camera_delay), timeout);
        origin.setValue(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);

        orientation.setX(transform.transform.rotation.x);
        orientation.setY(transform.transform.rotation.y);
        orientation.setZ(transform.transform.rotation.z);
        orientation.setW(transform.transform.rotation.w);

        world2uav_transform.setOrigin(origin);
        world2uav_transform.setRotation(orientation);

        // Obtain transform from world into camera frame
        camera2world_transform = (uav2camera_transform * world2uav_transform).inverse();
      } catch (tf2::TransformException& ex)
      {
        ROS_ERROR("Error during transform from \"%s\" frame to \"%s\" frame. MSG: %s", world_frame.c_str(), "usb_cam", ex.what());
        continue;
      }

      // Process the detections
      vector<bool> used_detections; // bools correspond to new detections
      used_detections.resize(latest_detections.detections.size(), false);
      for (auto &detUAV : detUAVs)
      {
        int used = detUAV.update(latest_detections, camera2world_transform);
        if (used >= 0)
        {
          cout << "Redetected " << used << ". detection" << std::endl;
          used_detections.at(used) = true;
        }
      }

      for (size_t it = 0; it < latest_detections.detections.size(); it++)
      {
        if (!used_detections.at(it))
        {
          Detected_UAV n_det(ass_thresh, unr_thresh, sim_thresh, 0.58, &nh);
          n_det.initialize(
                  latest_detections.detections.at(it),
                  latest_detections.w_used,
                  latest_detections.h_used,
                  latest_detections.camera_info,
                  camera2world_transform);
          detUAVs.push_back(std::move(n_det));
          cout << "Adding new detected UAV!" << std::endl;
        }
      }

      for (auto det_it = std::begin(detUAVs); det_it != std::end(detUAVs); det_it++)
      {
        //cout << "An UAV is detected with " << det_it->get_prob() << " probability" << std::endl;
        cout << "UAV detected with estimated relative position: ["
             << det_it->get_x() << ", "
             << det_it->get_y() << ", "
             << det_it->get_z() << "]" << std::endl;
        if (det_it->unreliable())
        {
          det_it = detUAVs.erase(det_it);
          cout << "\tkicking out uncertain UAV detection" << std::endl;
          // check if it wasn't the last element
          if (det_it == std::end(detUAVs))
              break;
        }

        // Erase similar elements to avoid duplicates
        for (auto det2_it = std::next(det_it); det2_it != std::end(detUAVs); det2_it++)
        {
          if (det2_it->similar_to(*det_it))
          {
            cout << "\terasing similar UAV detection" << std::endl;
            if (det2_it->more_uncertain_than(*det_it))
            {
              det2_it = detUAVs.erase(det2_it);
              if (det2_it == std::end(detUAVs))
                break;
            } else
            {
              det_it = detUAVs.erase(det_it);
              if (det_it == std::end(detUAVs))
                break;
            }
          }
        }
      }

      cout << "Detection processed" << std::endl;
    } else
    {
      r.sleep();
    }
  }
  delete tf_listener;
}
