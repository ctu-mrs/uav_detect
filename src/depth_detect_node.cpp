#include "main.h"

#include <uav_detect/BlobDetection.h>
#include <uav_detect/BlobDetections.h>
#include <uav_detect/Contour.h>
#include <uav_detect/DetectionParamsConfig.h>

#include "DepthBlobDetector.h"

using namespace cv;
using namespace std;
using namespace uav_detect;

// shortcut type to the dynamic reconfigure manager template instance
typedef mrs_lib::DynamicReconfigureMgr<uav_detect::DetectionParamsConfig> drmgr_t;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "uav_detect_localize");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Load parameters from ROS * //{*/
  mrs_lib::ParamLoader pl(nh);
  // LOAD STATIC PARAMETERS
  ROS_INFO("Loading static parameters:");
  int unknown_pixel_value = pl.load_param2<int>("unknown_pixel_value", 0);
  std::string path_to_mask = pl.load_param2<std::string>("path_to_mask", std::string());

  // LOAD DYNAMIC PARAMETERS
  drmgr_t drmgr;

  // CHECK LOADING STATUS
  if (!pl.loaded_successfully())
  {
    ROS_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
    ros::shutdown();
  }

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
  //}

  /* Create publishers and subscribers //{ */
  // Initialize subscribers
  mrs_lib::SubscribeMgr smgr(nh);
  
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::Image> depthmap_sh = smgr.create_handler<sensor_msgs::Image>("depthmap", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  // Initialize publishers
  ros::Publisher detections_pub = nh.advertise<uav_detect::Detections>("detections", 10); 
  ros::Publisher detected_blobs_pub = nh.advertise<uav_detect::BlobDetections>("blob_detections", 1);
  ros::Publisher processed_deptmap_pub = nh.advertise<sensor_msgs::Image&>("processed_depthmap", 1);
  //}

  cv::Mat mask_im;
  if (path_to_mask.empty())
  {
    ROS_INFO("[%s]: Not using image mask", ros::this_node::getName().c_str());
  } else
  {
    mask_im = cv::imread(path_to_mask, cv::IMREAD_GRAYSCALE);
    if (mask_im.empty())
    {
      ROS_ERROR("[%s]: Error loading image mask from file '%s'! Ending node.", ros::this_node::getName().c_str(), path_to_mask.c_str());
      ros::shutdown();
    } else if (mask_im.type() != CV_8UC1)
    {
      ROS_ERROR("[%s]: Loaded image mask has unexpected type: '%u' (expected %u)! Ending node.", ros::this_node::getName().c_str(), mask_im.type(), CV_8UC1);
      ros::shutdown();
    }
  }

  cout << "----------------------------------------------------------" << std::endl;

  ros::Rate r(50);
  while (ros::ok())
  {
    ros::spinOnce();

    ros::Time start_t = ros::Time::now();

    if (depthmap_sh->new_data())
    {
      cout << "Processsing image" << std::endl;

      cv_bridge::CvImage source_msg = *cv_bridge::toCvCopy(depthmap_sh->get_data(), string("16UC1"));

      /* Prepare the image for detection //{ */
      // create the detection image
      cv::Mat detect_im = source_msg.image.clone();
      cv::Mat raw_im = source_msg.image;
      cv::Mat known_pixels;
      cv::compare(raw_im, unknown_pixel_value, known_pixels, cv::CMP_NE);

      if (drmgr.config.blur_empty_areas)
      {
        cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(20, 5), Point(-1, -1));
        cv::Mat mask, tmp;
        mask = ~known_pixels;
        cv::dilate(detect_im, tmp, element, Point(-1, -1), 3);
        /* cv::GaussianBlur(tmp, tmp, Size(19, 75), 40); */
        cv::blur(detect_im, tmp, Size(115, 215));
        tmp.copyTo(detect_im, mask);
      }

      // dilate and erode the image if requested
      {
        cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(9, 9), Point(-1, -1));
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
      detector.detect(detect_im, mask_im, blobs);
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
        det.x = (brect.x + brect.width/2.0)/double(detect_im.cols);
        det.y = (brect.y + brect.height/2.0)/double(detect_im.rows);
        det.depth = blob.avg_depth;
        det.width = brect.width/double(detect_im.cols);
        det.height = brect.height/double(detect_im.rows);

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
}
