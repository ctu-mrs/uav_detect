/**
 * This program simply draws detection bounding boxes into the last camera frame
 * and publishes that for debugging purposes.
 * **/
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <uav_detect/Detection.h>
#include <uav_detect/Detections.h>

using namespace cv;
using namespace std;

bool new_cam_image = false;
bool new_detections = false;
cv_bridge::CvImagePtr last_cam_image_ptr;
uav_detect::Detections last_detections;

void camera_callback(const sensor_msgs::ImageConstPtr& image_msg)
{
  last_cam_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  new_cam_image = true;
}

void detections_callback(const uav_detect::Detections& dets_msg)
{
  last_detections = dets_msg;
  new_detections = true;
}

int main(int argc, char **argv)
{
  string uav_name, uav_frame, world_frame;
//  double camera_offset_x, camera_offset_y, camera_offset_z;
//  double camera_offset_roll, camera_offset_pitch, camera_offset_yaw;
//  double camera_delay;

  ros::init(argc, argv, "display_detections");
  ROS_INFO ("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Load parameters from ROS **/
  // UAV name
  nh.param("uav_name", uav_name, string());
  if (uav_name.empty())
  {
    ROS_ERROR("UAV_NAME is empty");
    ros::shutdown();
  }

//  nh.param("world_frame", world_frame, std::string("local_origin"));
//  nh.param("uav_frame", uav_frame, std::string("fcu_uav1"));
//  nh.param("camera_offset_x", camera_offset_x, numeric_limits<double>::infinity());
//  nh.param("camera_offset_y", camera_offset_y, numeric_limits<double>::infinity());
//  nh.param("camera_offset_z", camera_offset_z, numeric_limits<double>::infinity());
//  nh.param("camera_offset_roll", camera_offset_roll, numeric_limits<double>::infinity());
//  nh.param("camera_offset_pitch", camera_offset_pitch, numeric_limits<double>::infinity());
//  nh.param("camera_offset_yaw", camera_offset_yaw, numeric_limits<double>::infinity());
//  nh.param("camera_delay", camera_delay, numeric_limits<double>::infinity());

  cout << "Using parameters:" << std::endl;
  cout << "\tuav name:\t" << uav_name << std::endl;

//  // build the UAV to camera transformation
//  tf2::Transform uav2camera_transform;
//  {
//    tf2::Quaternion q;
//    tf2::Vector3    origin;
//    // camera transformation
//    origin.setValue(camera_offset_x, camera_offset_y, camera_offset_z);
//    // camera rotation
//    q.setRPY(camera_offset_roll / 180.0 * M_PI, camera_offset_pitch / 180.0 * M_PI, camera_offset_yaw / 180.0 * M_PI);
//
//    uav2camera_transform.setOrigin(origin);
//    uav2camera_transform.setRotation(q);
//
//    camera_delay = camera_delay/1000.0; // recalculate to seconds
//  }

//  tf2_ros::Buffer tf_buffer;
  /** Create publishers and subscribers **/
  ros::Subscriber camera_sub = nh.subscribe("camera_input", 1, camera_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber detections_sub = nh.subscribe("detections", 1, detections_callback, ros::TransportHints().tcpNoDelay());
  ros::Publisher det_imgs_pub = nh.advertise<sensor_msgs::Image>("det_imgs", 1);
//  tf2_ros::TransformListener* tf_listener = new tf2_ros::TransformListener(tf_buffer);

  cout << "----------------------------------------------------------" << std::endl;

  while (ros::ok())
  {
    ros::spinOnce();

    ros::Rate r(10);
    if (new_cam_image && new_detections)
    {
      new_cam_image = false;
      new_detections = false;
      cout << "Processing new detections" << std::endl;

//      // First, update the transforms
//      geometry_msgs::TransformStamped transform;
//      tf2::Transform  world2uav_transform, world2cam_transform;
//      tf2::Vector3    origin;
//      tf2::Quaternion orientation;
//      try
//      {
//        const ros::Duration timeout(1.0/6.0);
//        // Obtain transform from world into uav frame
//        transform = tf_buffer.lookupTransform(uav_frame, world_frame, last_detections.stamp - ros::Duration(camera_delay), timeout);
//        origin.setValue(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);
//
//        orientation.setX(transform.transform.rotation.x);
//        orientation.setY(transform.transform.rotation.y);
//        orientation.setZ(transform.transform.rotation.z);
//        orientation.setW(transform.transform.rotation.w);
//
//        world2uav_transform.setOrigin(origin);
//        world2uav_transform.setRotation(orientation);
//
//        // Obtain transform from world into camera frame
//        world2cam_transform = uav2camera_transform * world2uav_transform;
//      }
//      catch (tf2::TransformException& ex)
//      {
//        ROS_ERROR("Error during transform from \"%s\" frame to \"%s\" frame. MSG: %s", world_frame.c_str(), "usb_cam", ex.what());
//        continue;
//      }

      int cam_image_w = last_cam_image_ptr->image.cols;
      int cam_image_h = last_cam_image_ptr->image.rows;
      for (const auto &det : last_detections.detections)
      {
        cout << "Drawing one detection" << std::endl;
        //det.x_relative = (det.x_relative*w_camera + (last_cam_image_ptr->image.cols - w_camera)/2)/last_cam_image_ptr->image.cols;
        //det.w_relative = det.w_relative/w_camera*last_cam_image_ptr->image.cols;

        int w_used = last_detections.w_used;
        int h_used = last_detections.h_used;

        int x_lt = (int)round(
                          (cam_image_w-w_used)/2.0 +  // offset between the detection rectangle and camera image
                          (det.x_relative-det.w_relative/2.0)*w_used);
        int y_lt = (int)round(
                          (cam_image_h-h_used)/2.0 +  // offset between the detection rectangle and camera image
                          (det.y_relative-det.h_relative/2.0)*h_used);
        int w = (int)round(det.w_relative*w_used);
        int h = (int)round(det.h_relative*h_used);
        cv::Rect det_rect(
                  x_lt,
                  y_lt,
                  w,
                  h);
        cv::rectangle(last_cam_image_ptr->image, det_rect, Scalar(0, 0, 255), 5);

//        tf2::Vector3 tst(-5.0, 0.0, 5.0);
//        tst = world2cam_transform*tst;
//        cv::circle(last_cam_image_ptr->image, tst, 5, Scalar(0, 0, 255), 5);
      }

      det_imgs_pub.publish(last_cam_image_ptr->toImageMsg());

      cout << "Detection processed" << std::endl;
    } else
    {
      r.sleep();
    }
  }
}
