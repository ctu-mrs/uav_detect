#include "main.h"
#include "DepthBlobDetector.h"

using namespace cv;
using namespace std;
using namespace uav_detect;
using namespace Eigen;

/* Callbacks //{ */
// Callback for the depth map
bool new_dm = false;
sensor_msgs::Image last_dm_msg;
void depthmap_callback(const sensor_msgs::Image& dm_msg)
{
  ROS_INFO_THROTTLE(1.0, "Getting new depth images");
  last_dm_msg = dm_msg;
  new_dm = true;
}

/* // Callback for the depthmap camera info */
/* bool got_dm_cinfo = false; */
/* image_geometry::PinholeCameraModel camera_model; */
/* double d_fx, d_fy, d_cx, d_cy; */
/* void dm_cinfo_callback(const sensor_msgs::CameraInfo& dm_cinfo_msg) */
/* { */
/*   if (!got_dm_cinfo) */
/*   { */
/*     ROS_INFO_THROTTLE(1.0, "Got depth camera info"); */
/*     camera_model.fromCameraInfo(dm_cinfo_msg); */
/*     d_fx = camera_model.fx(); */
/*     d_fy = camera_model.fy(); */
/*     d_cx = camera_model.cx(); */
/*     d_cy = camera_model.cy(); */
/*     got_dm_cinfo = true; */
/*   } */
/* } */

/* double get_height(const Eigen::Vector3d& c2w_z, uint16_t px_x, uint16_t px_y, double depth_m) */
/* { */
/*   Eigen::Vector3d pos((px_x-d_cx)/d_fx*depth_m, (px_y-d_cy)/d_fy*depth_m, depth_m); */
/*   return c2w_z.dot(pos); */
/* } */
//}

// shortcut type to the dynamic reconfigure manager template instance
typedef mrs_lib::DynamicReconfigureMgr<uav_detect::DepthMapParamsConfig> drmgr_t;

/* Utility functions //{**/
/*double min_x, max_x;*/
/*double min_y, max_y;*/
/*double min_z, max_z;*/
/*bool position_valid(Eigen::Vector3d pos_vec)*/
/*{*/
/*  return (pos_vec(0) > min_x && pos_vec(0) < max_x) && (pos_vec(1) > min_y && pos_vec(2) < max_y) && (pos_vec(2) > min_z && pos_vec(2) < max_z);*/
/*}*/
/*//}*/

int main(int argc, char** argv)
{
  ros::init(argc, argv, "uav_detect_localize");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Load parameters from ROS * //{*/
  mrs_lib::ParamLoader pl(nh);
  // LOAD STATIC PARAMETERS
  ROS_INFO("Loading static parameters:");
  /* string uav_name = pl.load_param2<string>(string("uav_name")); */
  /* string uav_frame = pl.load_param2("uav_frame", std::string("fcu_") + uav_name); */
  /* string world_frame = pl.load_param2("world_frame", std::string("local_origin")); */
  /* // Load the camera transformation parameters */
  /* double camera_offset_x = pl.load_param2<double>("camera_offset_x"); */
  /* double camera_offset_y = pl.load_param2<double>("camera_offset_y"); */
  /* double camera_offset_z = pl.load_param2<double>("camera_offset_z"); */
  /* double camera_offset_roll = pl.load_param2<double>("camera_offset_roll"); */
  /* double camera_offset_pitch = pl.load_param2<double>("camera_offset_pitch"); */
  /* double camera_offset_yaw = pl.load_param2<double>("camera_offset_yaw"); */
  /* double camera_delay = load_param<double>(nh, "camera_delay");; */
  // LOAD DYNAMIC PARAMETERS
  drmgr_t drmgr;
  if (!drmgr.loaded_successfully())
  {
    ROS_ERROR("Some dynamic parameter default values were not loaded successfully, ending the node");
    ros::shutdown();
  }
  // Load the image preprocessing parameters
  int& dilate_iterations = drmgr.config.dilate_iterations;
  int& erode_iterations = drmgr.config.erode_iterations;
  int& erode_ignore_empty_iterations = drmgr.config.erode_ignore_empty_iterations;
  int& gaussianblur_size = drmgr.config.gaussianblur_size;
  int& medianblur_size = drmgr.config.medianblur_size;

  /* ROS_INFO("[%s]: Loading default values of dynamically reconfigurable variables", ros::this_node::getName().c_str()); */
  /* // Load default values of dynamically reconfigurable parameters */
  /* pl.load_param("dilate_iterations", dilate_iterations); */
  /* pl.load_param("erode_iterations", erode_iterations); */
  /* pl.load_param("erode_ignore_empty_iterations", erode_ignore_empty_iterations); */
  /* pl.load_param("gaussianblur_size", gaussianblur_size); */
  /* pl.load_param("medianblur_size", medianblur_size); */
  /* pl.load_param("filter_by_color", params.filter_by_color); */
  /* pl.load_param("min_depth", params.min_depth); */
  /* pl.load_param("max_depth", params.max_depth); */
  /* pl.load_param("use_threshold_width", params.use_threshold_width); */
  /* pl.load_param("threshold_step", params.threshold_step); */
  /* pl.load_param("threshold_width", params.threshold_width); */
  /* pl.load_param("filter_by_area", params.filter_by_area); */
  /* pl.load_param("min_area", params.min_area); */
  /* pl.load_param("max_area", params.max_area); */
  /* pl.load_param("filter_by_circularity", params.filter_by_circularity); */
  /* pl.load_param("min_circularity", params.min_circularity); */
  /* pl.load_param("max_circularity", params.max_circularity); */
  /* pl.load_param("filter_by_orientation", params.filter_by_orientation); */
  /* pl.load_param("min_angle", params.min_angle); */
  /* pl.load_param("max_angle", params.max_angle); */
  /* pl.load_param("filter_by_convexity", params.filter_by_convexity); */
  /* pl.load_param("min_convexity", params.min_convexity); */
  /* pl.load_param("max_convexity", params.max_convexity); */
  /* pl.load_param("filter_by_inertia", params.filter_by_inertia); */
  /* pl.load_param("min_inertia_ratio", params.min_inertia_ratio); */
  /* pl.load_param("max_inertia_ratio", params.max_inertia_ratio); */
  /* pl.load_param("min_dist_between", params.min_dist_between); */
  /* pl.load_param("min_repeatability", params.min_repeatability); */

  if (!pl.loaded_successfully())
  {
    ROS_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
    ros::shutdown();
  }
  //}

  /** Build the UAV to camera transformation * //{*/
  /*tf2::Transform uav2camera_transform;*/
  /*{*/
  /*  tf2::Quaternion q;*/
  /*  tf2::Vector3 origin;*/
  /*  // camera transformation*/
  /*  origin.setValue(camera_offset_x, camera_offset_y, camera_offset_z);*/
  /*  // camera rotation*/
  /*  q.setRPY(camera_offset_roll / 180.0 * M_PI, camera_offset_pitch / 180.0 * M_PI, camera_offset_yaw / 180.0 * M_PI);*/

  /*  uav2camera_transform.setOrigin(origin);*/
  /*  uav2camera_transform.setRotation(q);*/
  /*}*/
  /*//}*/

  /* Create publishers and subscribers //{ */
  /* tf2_ros::Buffer tf_buffer; */
  /* // Initialize transform listener */
  /* tf2_ros::TransformListener* tf_listener = new tf2_ros::TransformListener(tf_buffer); */
  // Initialize other subs and pubs
  ros::Subscriber depthmap_sub = nh.subscribe("depthmap", 1, depthmap_callback, ros::TransportHints().tcpNoDelay());
   /* ros::Subscriber dm_cinfo_sub = nh.subscribe("camera_info", 1, dm_cinfo_callback, ros::TransportHints().tcpNoDelay()); */ 
  ros::Publisher detections_pub = nh.advertise<uav_detect::Detections>("detections", 10); 
  ros::Publisher detected_blobs_pub = nh.advertise<uav_detect::BlobDetections>("blob_detections", 1);
  ros::Publisher processed_deptmap_pub = nh.advertise<sensor_msgs::Image&>("processed_depthmap", 1);
  //}

  cout << "----------------------------------------------------------" << std::endl;

  cv_bridge::CvImage source_msg;
  ros::Rate r(50);
  while (ros::ok())
  {
    ros::spinOnce();

    ros::Time start_t = ros::Time::now();

    if (new_dm)
    {
      cout << "Processsing image" << std::endl;

      /* // Construct a new world to camera transform //{ */
      /* Eigen::Affine3d c2w_tf; */
      /* geometry_msgs::TransformStamped transform; */
      /* tf2::Transform world2uav_transform; */
      /* tf2::Vector3 origin; */
      /* tf2::Quaternion orientation; */
      /* try */
      /* { */
      /*   const ros::Duration timeout(1.0 / 6.0); */
      /*   // Obtain transform from world into uav frame */
      /*   transform = tf_buffer.lookupTransform(uav_frame, world_frame, last_dm_msg.header.stamp, timeout); */
      /*   /1* tf2::convert(transform, world2uav_transform); *1/ */
      /*   origin.setValue(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z); */

      /*   orientation.setX(transform.transform.rotation.x); */
      /*   orientation.setY(transform.transform.rotation.y); */
      /*   orientation.setZ(transform.transform.rotation.z); */
      /*   orientation.setW(transform.transform.rotation.w); */

      /*   world2uav_transform.setOrigin(origin); */
      /*   world2uav_transform.setRotation(orientation); */

      /*   // Obtain transform from camera frame into world */
      /*   c2w_tf = tf2_to_eigen(uav2camera_transform * world2uav_transform).inverse(); */
      /* } */
      /* catch (tf2::TransformException& ex) */
      /* { */
      /*   ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", world_frame.c_str(), uav_frame.c_str(), ex.what()); */
      /*   continue; */
      /* } */
      /* //} */

      source_msg = *cv_bridge::toCvCopy(last_dm_msg, string("16UC1"));
      new_dm = false;

      /* Prepare the image for detection //{ */
      // create the detection image
      cv::Mat detect_im = source_msg.image.clone();
      cv::Mat raw_im = source_msg.image;
      cv::Mat known_pixels;
      inRange(raw_im, 1, std::numeric_limits<uint16_t>::max(), known_pixels);

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

      //}

      /* Use OpenCV SimpleBlobDetector to find blobs //{ */
      vector<dbd::Blob> blobs;
      dbd::DepthBlobDetector detector(dbd::Params(drmgr.config));
      ROS_INFO("[%s]: Starting Blob detector", ros::this_node::getName().c_str());
      detector.detect(detect_im, known_pixels, raw_im, blobs);
      ROS_INFO("[%s]: Blob detector finished", ros::this_node::getName().c_str());

      cout << "Number of blobs: " << blobs.size() << std::endl;

      //}
      
      /* Create and publish the message with detections //{ */
      uav_detect::Detections dets;
      dets.header.frame_id = source_msg.header.frame_id;
      dets.header.stamp = source_msg.header.stamp;
      dets.detections.reserve(blobs.size());
      for (const dbd::Blob& blob : blobs)
      {
        uav_detect::Detection det;
        det.class_ID = -1;

        det.roi.x_offset = 0;
        det.roi.y_offset = 0;
        det.roi.width = detect_im.cols;
        det.roi.height = detect_im.rows;

        cv::Rect brect = cv::boundingRect(blob.contours.at(blob.contours.size()/2));
        det.x = brect.x/double(detect_im.cols);
        det.y = brect.y/double(detect_im.rows);
        det.depth = blob.avg_depth;
        det.width = brect.width/double(detect_im.cols);
        det.height = brect.height/double(detect_im.cols);

        det.confidence = -1;

        dets.detections.push_back(det);
      }
      detections_pub.publish(dets);
      //}

      if (processed_deptmap_pub.getNumSubscribers() > 0)
      {
        /* Create and publish the debug image //{ */
        cv_bridge::CvImage processed_depthmap_cvb = source_msg;
        processed_depthmap_cvb.image = detect_im;
        sensor_msgs::ImagePtr out_msg = processed_depthmap_cvb.toImageMsg();
        processed_deptmap_pub.publish(out_msg);
        //}
      }

      if (detected_blobs_pub.getNumSubscribers() > 0)
      {
        /* Create and publish the message with raw blob data //{ */
        uav_detect::BlobDetections dets;
        dets.header.frame_id = source_msg.header.frame_id;
        dets.header.stamp = source_msg.header.stamp;
        dets.blobs.reserve(blobs.size());
        for (const dbd::Blob& blob : blobs)
        {
          uav_detect::BlobDetection det;

          det.x = blob.location.x;
          det.y = blob.location.y;
          det.avg_depth = blob.avg_depth;
          det.confidence = blob.confidence;
          det.convexity = blob.convexity;
          det.angle = blob.angle;
          det.area = blob.area;
          det.circularity = blob.circularity;
          det.radius = blob.radius;
          det.inertia = blob.inertia;
          det.contours.reserve(blob.contours.size());
          for (const auto& cont : blob.contours)
          {
            Contour cnt;
            cnt.pixels.reserve(cont.size());
            for (const auto& pt : cont)
            {
              uav_detect::ImagePixel px;
              px.x = pt.x;
              px.y = pt.y;
              cnt.pixels.push_back(px);
            }
            det.contours.push_back(cnt);
          }

          dets.blobs.push_back(det);
        }
        detected_blobs_pub.publish(dets);
        //}
      }

      cout << "Image processed" << std::endl;
      ros::Time end_t = ros::Time::now();
      static double dt = (end_t - start_t).toSec();
      dt = 0.9*dt + 0.1*(end_t - start_t).toSec();
      cout << "processing FPS: " << 1/dt << "Hz" << std::endl;
    } else
    {
      r.sleep();
    }
  }
  /* delete tf_listener; */
}
