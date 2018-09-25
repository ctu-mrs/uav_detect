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
  // Load the image preprocessing parameters
  int& dilate_iterations = drmgr.config.dilate_iterations;
  int& erode_iterations = drmgr.config.erode_iterations;
  int& gaussianblur_size = drmgr.config.gaussianblur_size;
  int& medianblur_size = drmgr.config.medianblur_size;
  // Load the detection parameters
  // Filter by color
  bool& blob_filter_by_color = drmgr.config.blob_filter_by_color;
  int& min_depth = drmgr.config.min_depth;
  int& max_depth = drmgr.config.max_depth;
  bool& use_threshold_width = drmgr.config.use_threshold_width;
  int& blob_threshold_step = drmgr.config.blob_threshold_step;
  int& blob_threshold_width = drmgr.config.blob_threshold_width;
  // Filter by area
  bool& blob_filter_by_area = drmgr.config.blob_filter_by_area;
  int& blob_min_area = drmgr.config.blob_min_area;
  int& blob_max_area = drmgr.config.blob_max_area;
  // Filter by circularity
  bool& blob_filter_by_circularity = drmgr.config.blob_filter_by_circularity;
  double& blob_min_circularity = drmgr.config.blob_min_circularity;
  double& blob_max_circularity = drmgr.config.blob_max_circularity;
  // Filter by orientation
  bool& blob_filter_by_orientation = drmgr.config.blob_filter_by_orientation;
  double& blob_min_angle = drmgr.config.blob_min_angle;
  double& blob_max_angle = drmgr.config.blob_max_angle;
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

  bool paused = false;
  cv_bridge::CvImage source_img;
  ros::Rate r(10);
  while (ros::ok())
  {
    ros::spinOnce();

    // Check if we got a new message containing a depthmap
    /* if (new_dm) */
    if (new_dm && got_dm_cinfo)
    {
      cout << "Processsing image" << std::endl;

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

      if (!paused || source_img.image.empty())
      {
        source_img = *cv_bridge::toCvCopy(last_dm_msg, string("16UC1"));
        new_dm = false;
      }

      /* Prepare the image for detection //{ */
      // create the detection image
      cv::Mat detect_im = source_img.image.clone();
      cv::Mat raw_im = source_img.image;
      // prepare the debug image
      cv_bridge::CvImage dbg_img = source_img;

      if (drmgr.config.blur_empty_areas)
      {
        cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(20, 5), Point(-1, -1));
        cv::Mat mask, tmp;
        cv::inRange(detect_im, 0, 0, mask);
        cv::dilate(detect_im, tmp, element, Point(-1, -1), 3);
        /* cv::GaussianBlur(tmp, tmp, Size(19, 75), 40); */
        cv::blur(detect_im, tmp, Size(115, 215));
        tmp.copyTo(detect_im, mask);
      }

      // dilate and erode the image if requested
      {
        cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
        cv::dilate(detect_im, detect_im, element, Point(-1, -1), dilate_iterations);
        cv::erode(detect_im, detect_im, element, Point(-1, -1), erode_iterations);
      }

      // blur it if requested
      if (gaussianblur_size % 2 == 1)
      {
        cv::GaussianBlur(detect_im, detect_im, cv::Size(gaussianblur_size, gaussianblur_size), 0);
      }
      // blur it if requested
      if (medianblur_size % 2 == 1)
      {
        cv::medianBlur(detect_im, detect_im, medianblur_size);
      }

      // convert the output image from grayscale to color to enable colorful drawing
      cv::cvtColor(detect_im, dbg_img.image, COLOR_GRAY2BGR);
      dbg_img.encoding = string("bgr16");

      //}

      /* Use OpenCV SimpleBlobDetector to find blobs //{ */
      vector<dbd::Blob> blobs;
      dbd::Params params;
      params.filter_by_color = blob_filter_by_color;
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
      params.filter_by_orientation = blob_filter_by_orientation;
      params.min_angle = blob_min_angle;
      params.max_angle = blob_max_angle;
      params.filter_by_inertia = blob_filter_by_inertia;
      params.min_inertia_ratio = blob_min_inertia_ratio;
      params.max_inertia_ratio = blob_max_inertia_ratio;
      params.min_threshold = min_depth;
      params.max_threshold = max_depth;
      params.min_depth = min_depth;
      params.max_depth = max_depth;
      params.use_threshold_width = use_threshold_width;
      params.threshold_step = blob_threshold_step;
      params.threshold_width = blob_threshold_width;
      params.min_repeatability = blob_min_repeatability;
      params.min_dist_between = blob_min_dist_between;

      dbd::DepthBlobDetector detector(params);
      ROS_INFO("[%s]: Starting Blob detector", ros::this_node::getName().c_str());
      detector.detect(detect_im, raw_im, blobs);
      ROS_INFO("[%s]: Blob detector finished", ros::this_node::getName().c_str());

      /* cv::drawKeypoints(dbg_img.image, blobs, dbg_img.image, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG); */
      /* int potential = 0; */
      /* int unsure = 0; */
      int sure = 0;
      for (const auto& blob : blobs)
      {
        sure++;

        auto max = blob.contours.size();
        for (size_t it = blob.contours.size()-1; it; it--)
        {
          cv::drawContours(dbg_img.image, blob.contours, it, Scalar(0, 65535, 65535/max*it), CV_FILLED);
        }
      }
      /* cv::putText(dbg_img.image, string("potential: ") + to_string(potential), Point(0, 40), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 0, 0), 3); */
      /* cv::putText(dbg_img.image, string("unsure: ") + to_string(unsure), Point(0, 80), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 3); */
      cv::putText(dbg_img.image, string("found: ") + to_string(sure), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(0, 0, 65535), 2);
      /* cv::circle(dbg_img.image, Point(im_h/2, 100), 50, Scalar(255), 10); */
      cout << "Number of blobs: " << blobs.size() << std::endl;

      //}

      imshow("image_raw", raw_im);
      imshow("found_blobs", dbg_img.image);
      if (waitKey(25) == ' ')
      {
        paused = !paused;
      }
      sensor_msgs::ImagePtr out_msg = dbg_img.toImageMsg();
      thresholded_pub.publish(out_msg);

      cout << "Image processed" << std::endl;
    } else
    {
      r.sleep();
    }
  }
  delete tf_listener;
}
