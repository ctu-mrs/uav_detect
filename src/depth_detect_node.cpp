#include "main.h"
#include "DepthBlobDetector.h"

using namespace cv;
using namespace std;
using namespace uav_detect;
using namespace Eigen;

// Callback for the thermal image
bool new_thermal = false;
sensor_msgs::Image last_thermal_msg;
void thermal_callback(const sensor_msgs::Image& thermal_msg)
{
  ROS_INFO_THROTTLE(1.0, "Got new thermal image");
  last_thermal_msg = thermal_msg;
  new_thermal = true;
}

// Callback for the rgb image
bool new_rgb = false;
sensor_msgs::Image last_rgb_msg;
void rgb_callback(const sensor_msgs::Image& rgb_msg)
{
  ROS_INFO_THROTTLE(1.0, "Got new rgb image");
  last_rgb_msg = rgb_msg;
  new_rgb = true;
}

// Callback for the depth map
bool new_dm = false;
sensor_msgs::Image last_dm_msg;
void depthmap_callback(const sensor_msgs::Image& dm_msg)
{
  ROS_INFO_THROTTLE(1.0, "Got new depth image");
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
    ROS_INFO_THROTTLE(1.0, "Got depth camera info");
    camera_model.fromCameraInfo(dm_cinfo_msg);
    d_fx = camera_model.fx();
    d_fy = camera_model.fy();
    d_cx = camera_model.cx();
    d_cy = camera_model.cy();
    got_dm_cinfo = true;
  }
}

Point cursor_pos;
void mouse_callback([[maybe_unused]]int event, int x, int y, [[maybe_unused]]int flags, [[maybe_unused]]void* userdata)
{
  cursor_pos = Point(x, y);
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
  int& erode_ignore_empty_iterations = drmgr.config.erode_ignore_empty_iterations;
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
  ros::Subscriber thermal_sub = nh.subscribe("thermal_image", 1, thermal_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber rgb_sub = nh.subscribe("rgb_image", 1, rgb_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber depthmap_sub = nh.subscribe("depth_map", 1, depthmap_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber dm_cinfo_sub = nh.subscribe("camera_info", 1, dm_cinfo_callback, ros::TransportHints().tcpNoDelay());
  ros::Publisher detected_UAV_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("detected_uav", 10);
  ros::Publisher thresholded_pub = nh.advertise<sensor_msgs::Image&>("thresholded_dm", 1);
  ros::Publisher distance_pub = nh.advertise<std_msgs::Float64>("detected_uav_distance", 10);
  //}

  cout << "----------------------------------------------------------" << std::endl;

  int window_flags = WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL;
  string rgb_winname = "RGB_image";
  string thermal_winname = "thermal_image";
  string dm_winname = "depth_image";
  string det_winname = "depth_detections";
  cv::namedWindow(rgb_winname, window_flags);
  cv::namedWindow(thermal_winname, window_flags);
  cv::namedWindow(dm_winname, window_flags);
  cv::namedWindow(det_winname, window_flags);
  setMouseCallback(det_winname, mouse_callback, NULL);
  bool paused = false;
  bool fill_blobs = true;
  cv_bridge::CvImage source_img;
  ros::Rate r(50);
  while (ros::ok())
  {
    ros::spinOnce();

    ros::Time start_t = ros::Time::now();

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
      cv::Mat unknown_pixels, known_pixels;
      inRange(raw_im, 0, 0, unknown_pixels);
      known_pixels = ~unknown_pixels;
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
        cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(6, 3), Point(-1, -1));
        cv::dilate(detect_im, detect_im, element, Point(-1, -1), dilate_iterations);
        cv::erode(detect_im, detect_im, element, Point(-1, -1), erode_iterations);

        // erode without using zero (unknown) pixels
        if (erode_ignore_empty_iterations > 0)
        {
          cv::Mat unknown_as_max = cv::Mat(raw_im.size(), CV_16UC1, std::numeric_limits<uint16_t>::max());
          raw_im.copyTo(unknown_as_max, known_pixels);
          cv::erode(unknown_as_max, detect_im, element, Point(-1, -1), erode_ignore_empty_iterations);
        }
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
      detector.detect(detect_im, known_pixels, unknown_pixels, raw_im, blobs);
      ROS_INFO("[%s]: Blob detector finished", ros::this_node::getName().c_str());

      cv::Mat rgb_im;
      if (new_rgb)
      {
        rgb_im = (cv_bridge::toCvCopy(last_rgb_msg, sensor_msgs::image_encodings::BGR8))->image;
      }
      cv::Mat thermal_im_colormapped;
      if (new_thermal)
      {
        cv::Mat thermal_im = (cv_bridge::toCvCopy(last_thermal_msg, string("16UC1")))->image;
        double min;
        double max;
        cv::minMaxIdx(thermal_im, &min, &max);
        cv::Mat im_8UC1;
        thermal_im.convertTo(im_8UC1, CV_8UC1, 255 / (max-min), -min); 
        applyColorMap(im_8UC1, thermal_im_colormapped, cv::COLORMAP_JET);
      }
      cv::Mat dm_im_colormapped;
      {
        double min;
        double max;
        cv::minMaxIdx(raw_im, &min, &max);
        cv::Mat im_8UC1;
        raw_im.convertTo(im_8UC1, CV_8UC1, 255 / (max-min), -min); 
        applyColorMap(im_8UC1, dm_im_colormapped, cv::COLORMAP_JET);
        cv::Mat blackness = cv::Mat::zeros(dm_im_colormapped.size(), dm_im_colormapped.type());
        blackness.copyTo(dm_im_colormapped, unknown_pixels);
      }

      int sure = 0;
      bool displaying_info = false;
      for (const auto& blob : blobs)
      {

        auto max = blob.contours.size();
        if (!blob.contours.empty())
        {
          sure++;
          if (fill_blobs)
          {
            for (size_t it = blob.contours.size()-1; it; it--)
            /* size_t it = blob.contours.size()/2; */
            {
              cv::drawContours(dbg_img.image, blob.contours, it, Scalar(0, 65535, 65535/max*it), CV_FILLED);
              auto cur_blob = blob.contours.at(it);
              if (!displaying_info && pointPolygonTest(cur_blob, cursor_pos, false) > 0)
              {
                // display information about this contour
                displaying_info = true;
                cv::putText(dbg_img.image, string("avg_depth: ") + to_string(blob.avg_depth), Point(0, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("confidence: ") + to_string(blob.confidence), Point(0, 65), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("repeatability: ") + to_string(blob.contours.size()), Point(0, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("convexity: ") + to_string(blob.convexity), Point(0, 95), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("angle: ") + to_string(blob.angle), Point(0, 110), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("area: ") + to_string(blob.area), Point(0, 125), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("circularity: ") + to_string(blob.circularity), Point(0, 140), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("radius: ") + to_string(blob.radius), Point(0, 155), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
                cv::putText(dbg_img.image, string("inertia: ") + to_string(blob.inertia), Point(0, 170), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 65535), 2);
              }
            }
          } else
          {
            cv::circle(dbg_img.image, blob.location, blob.radius, Scalar(0, 0, 65535), 2);
          }
          if (new_rgb)
            cv::circle(rgb_im, blob.location, blob.radius, Scalar(0, 0, 255), 2);
          cv::circle(dm_im_colormapped, blob.location, blob.radius, Scalar(0, 0, 255), 2);
        }
      }
      /* cv::putText(dbg_img.image, string("potential: ") + to_string(potential), Point(0, 40), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 0, 0), 3); */
      /* cv::putText(dbg_img.image, string("unsure: ") + to_string(unsure), Point(0, 80), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 3); */
      cv::putText(dbg_img.image, string("found: ") + to_string(sure), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(0, 0, 65535), 2);
      /* cv::circle(dbg_img.image, Point(im_h/2, 100), 50, Scalar(255), 10); */
      cout << "Number of blobs: " << blobs.size() << std::endl;

      //}

      if (new_rgb)
        imshow(rgb_winname, rgb_im);
      if (new_thermal)
        imshow(thermal_winname, thermal_im_colormapped);
      imshow(dm_winname, dm_im_colormapped);
      imshow(det_winname, dbg_img.image);
      int key = waitKey(3);
      switch (key)
      {
        case ' ':
          paused = !paused;
          break;
        case 'f':
          fill_blobs = !fill_blobs;
          break;
      }
      sensor_msgs::ImagePtr out_msg = dbg_img.toImageMsg();
      thresholded_pub.publish(out_msg);

      cout << "Image processed" << std::endl;
      ros::Time end_t = ros::Time::now();
      double dt = (end_t - start_t).toSec();
      cout << "processing FPS: " << 1/dt << "Hz" << std::endl;
    } else
    {
      r.sleep();
    }
  }
  delete tf_listener;
}
