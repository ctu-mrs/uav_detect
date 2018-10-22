#include "main.h"

using namespace cv;
using namespace std;
using namespace uav_detect;

template <typename T>
void add_to_buffer(T img, std::list<T>& bfr)
{
  bfr.push_back(img);
  if (bfr.size() > 100)
    bfr.pop_front();
}

template <class T>
T find_closest(ros::Time stamp, std::list<T>& bfr)
{
  T closest;
  double closest_diff;
  bool closest_set = false;

  for (auto& imptr : bfr)
  {
    double cur_diff = abs((imptr->header.stamp - stamp).toSec());

    if (!closest_set || cur_diff < closest_diff)
    {
      closest = imptr;
      closest_diff = cur_diff;
      closest_set = true;
    }
  }
  return closest;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "backproject_display_node");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  mrs_lib::SubscribeHandlerPtr<geometry_msgs::PoseWithCovarianceStamped> sh_pose;
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> sh_img;
  mrs_lib::SubscribeHandlerPtr<uav_detect::DetectionsConstPtr> sh_det;
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::CameraInfo> sh_cinfo;

  mrs_lib::SubscribeMgr smgr;
  sh_pose = smgr.create_handler_threadsafe<geometry_msgs::PoseWithCovarianceStamped>(nh, "localized_uav", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_img = smgr.create_handler_threadsafe<sensor_msgs::ImageConstPtr>(nh, "image_rect", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_det = smgr.create_handler_threadsafe<uav_detect::DetectionsConstPtr>(nh, "detections", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_cinfo = smgr.create_handler_threadsafe<sensor_msgs::CameraInfo>(nh, "camera_info", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener = tf2_ros::TransformListener(tf_buffer);

  if (!smgr.loaded_successfully())
  {
    ROS_ERROR("[%s]: Failed to subscribe some nodes", ros::this_node::getName().c_str());
    ros::shutdown();
  }

  int window_flags = WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL;
  std::string window_name = "backprojected_detection";
  cv::namedWindow(window_name, window_flags);
  image_geometry::PinholeCameraModel camera_model;
  ros::Rate r(100);

  std::list<sensor_msgs::ImageConstPtr> img_buffer;
  std::list<uav_detect::DetectionsConstPtr> det_buffer;

  while (ros::ok())
  {
    ros::spinOnce();

    if (sh_cinfo->has_data() && !sh_cinfo->used_data())
    {
      camera_model.fromCameraInfo(sh_cinfo->get_data());
    }

    if (sh_img->new_data())
      add_to_buffer(sh_img->get_data(), img_buffer);

    if (sh_det->new_data())
      add_to_buffer(sh_det->get_data(), det_buffer);

    bool has_data = sh_pose->has_data() && sh_img->has_data();
    bool new_data = sh_pose->new_data();

    if (has_data && new_data && sh_cinfo->used_data() && sh_det->used_data())
    {
      geometry_msgs::PoseWithCovarianceStamped pose_in = sh_pose->get_data();
      sensor_msgs::ImageConstPtr img_ros = find_closest(pose_in.header.stamp, img_buffer);
      uav_detect::DetectionsConstPtr dets = find_closest(pose_in.header.stamp, det_buffer);

      geometry_msgs::TransformStamped transform = tf_buffer.lookupTransform(img_ros->header.frame_id, pose_in.header.frame_id, pose_in.header.stamp, ros::Duration(1.0));
      geometry_msgs::Point point_transformed;
      tf2::doTransform(pose_in.pose.pose.position, point_transformed, transform);

      cv::Point3d pt3d;
      pt3d.x = point_transformed.x;
      pt3d.y = point_transformed.y;
      pt3d.z = point_transformed.z;
      double dist = sqrt(pt3d.x*pt3d.x + pt3d.y*pt3d.y + pt3d.z*pt3d.z);
      
      cv::Point pt2d = camera_model.project3dToPixel(pt3d);
    
      cv::Mat img;
      cv_bridge::CvImagePtr img_ros2 = cv_bridge::toCvCopy(img_ros, "bgr8");
      img = img_ros2->image;

      cv::circle(img, pt2d, 40, Scalar(0, 0, 255), 2);
      cv::line(img, cv::Point(pt2d.x - 15, pt2d.y), cv::Point(pt2d.x + 15, pt2d.y), Scalar(0, 0, 220));
      cv::line(img, cv::Point(pt2d.x, pt2d.y - 15), cv::Point(pt2d.x, pt2d.y + 15), Scalar(0, 0, 220));
      cv::putText(img, "distance: " + std::to_string(dist), cv::Point(pt2d.x + 30, pt2d.y + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
      /* cout << "Backprojected: " << pt2d << endl; */
      /* for (const uav_detect::Detection& det : dets->detections) */
      /* { */
      /*   cv::Point det_pt; */
      /*   det_pt.x = det.x * det.roi.width + det.roi.x_offset; */
      /*   det_pt.y = det.y * det.roi.height + det.roi.y_offset; */
      /*   cv::circle(img, det_pt, 5, Scalar(0, 255, 0), 2); */
      /*   cout << "Detection: " << det_pt << endl; */
      /* } */

      cv::imshow(window_name, img);
      cv::waitKey(1);
    }

    r.sleep();
  }
}