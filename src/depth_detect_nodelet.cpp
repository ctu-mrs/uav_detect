#include "main.h"

#include <nodelet/nodelet.h>

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

namespace uav_detect
{

  class DepthDetector : public nodelet::Nodelet
  {
  public:

    /* onInit() method //{ */
    void onInit()
    {
      ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

      m_node_name = "DepthDetector";

      /* Load parameters from ROS //{*/
      mrs_lib::ParamLoader pl(nh, m_node_name);
      // LOAD STATIC PARAMETERS
      ROS_INFO("Loading static parameters:");
      pl.load_param("unknown_pixel_value", m_unknown_pixel_value, 0);
      std::string path_to_mask = pl.load_param2<std::string>("path_to_mask", std::string());

      // LOAD DYNAMIC PARAMETERS
      // CHECK LOADING STATUS
      if (!pl.loaded_successfully())
      {
        ROS_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
        ros::shutdown();
      }

      m_drmgr_ptr = make_unique<drmgr_t>(nh, m_node_name);
      if (!m_drmgr_ptr->loaded_successfully())
      {
        ROS_ERROR("Some dynamic parameter default values were not loaded successfully, ending the node");
        ros::shutdown();
      }
      //}

      /* Create publishers and subscribers //{ */
      // Initialize subscribers
      mrs_lib::SubscribeMgr smgr(nh, m_node_name);
      
      m_depthmap_sh = smgr.create_handler<sensor_msgs::Image>("depthmap", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
      // Initialize publishers
      m_detections_pub = nh.advertise<uav_detect::Detections>("detections", 10); 
      m_detected_blobs_pub = nh.advertise<uav_detect::BlobDetections>("blob_detections", 1);
      m_processed_deptmap_pub = nh.advertise<sensor_msgs::Image>("processed_depthmap", 1);
      //}

      /* Initialize other varibles //{ */
      if (path_to_mask.empty())
      {
        ROS_INFO("[%s]: Not using image mask", ros::this_node::getName().c_str());
      } else
      {
        m_mask_im = cv::imread(path_to_mask, cv::IMREAD_GRAYSCALE);
        if (m_mask_im.empty())
        {
          ROS_ERROR("[%s]: Error loading image mask from file '%s'! Ending node.", ros::this_node::getName().c_str(), path_to_mask.c_str());
          ros::shutdown();
        } else if (m_mask_im.type() != CV_8UC1)
        {
          ROS_ERROR("[%s]: Loaded image mask has unexpected type: '%u' (expected %u)! Ending node.", ros::this_node::getName().c_str(), m_mask_im.type(), CV_8UC1);
          ros::shutdown();
        }
      }

      m_last_detection_id = 0;

      m_det_blobs = 0;
      m_images_processed = 0;
      m_avg_fps = 0.0f;
      m_avg_delay = 0.0f;
      //}

      m_main_loop_timer = nh.createTimer(ros::Rate(1000), &DepthDetector::main_loop, this);
      m_info_loop_timer = nh.createTimer(ros::Rate(1), &DepthDetector::info_loop, this);

      cout << "----------------------------------------------------------" << std::endl;

    }
    //}

    /* main_loop() method //{ */
    void main_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      if (m_depthmap_sh->new_data())
      {
        ros::Time start_t = ros::Time::now();

        cv_bridge::CvImage source_msg = *cv_bridge::toCvCopy(m_depthmap_sh->get_data(), string("16UC1"));

        /* Prepare the image for detection //{ */
        // create the detection image
        cv::Mat detect_im = source_msg.image.clone();
        cv::Mat raw_im = source_msg.image;
        cv::Mat known_pixels;
        cv::compare(raw_im, m_unknown_pixel_value, known_pixels, cv::CMP_NE);

        if (m_drmgr_ptr->config.blur_empty_areas)
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
          cv::dilate(detect_im, detect_im, element, Point(-1, -1), m_drmgr_ptr->config.dilate_iterations);
          cv::erode(detect_im, detect_im, element, Point(-1, -1), m_drmgr_ptr->config.erode_iterations);

          // erode without using zero (unknown) pixels
          if (m_drmgr_ptr->config.erode_ignore_empty_iterations > 0)
          {
            cv::Mat unknown_as_max = cv::Mat(raw_im.size(), CV_16UC1, std::numeric_limits<uint16_t>::max());
            raw_im.copyTo(unknown_as_max, known_pixels);
            cv::erode(unknown_as_max, detect_im, element, Point(-1, -1), m_drmgr_ptr->config.erode_ignore_empty_iterations);
          }
        }

        // blur it if requested
        if (m_drmgr_ptr->config.gaussianblur_size % 2 == 1)
        {
          cv::GaussianBlur(detect_im, detect_im, cv::Size(m_drmgr_ptr->config.gaussianblur_size, m_drmgr_ptr->config.gaussianblur_size), 0);
        }
        // blur it if requested
        if (m_drmgr_ptr->config.medianblur_size % 2 == 1)
        {
          cv::medianBlur(detect_im, detect_im, m_drmgr_ptr->config.medianblur_size);
        }

        //}

        /* Use DepthBlobDetector to find blobs //{ */
        vector<dbd::Blob> blobs;
        dbd::DepthBlobDetector detector(dbd::Params(m_drmgr_ptr->config));
        detector.detect(detect_im, m_mask_im, blobs);

        /* cout << "Number of blobs: " << blobs.size() << std::endl; */

        //}
        
        /* Create and publish the message with detections //{ */
        uav_detect::Detections dets;
        dets.header.frame_id = source_msg.header.frame_id;
        dets.header.stamp = source_msg.header.stamp;
        dets.detections.reserve(blobs.size());
        for (const dbd::Blob& blob : blobs)
        {
          uav_detect::Detection det;
          det.id = m_last_detection_id++;
          det.class_id = -1;

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
        m_detections_pub.publish(dets);
        //}

        if (m_processed_deptmap_pub.getNumSubscribers() > 0)
        {
          /* Create and publish the debug image //{ */
          cv_bridge::CvImage processed_depthmap_cvb = source_msg;
          processed_depthmap_cvb.image = detect_im;
          sensor_msgs::ImagePtr out_msg = processed_depthmap_cvb.toImageMsg();
          m_processed_deptmap_pub.publish(out_msg);
          //}
        }

        if (m_detected_blobs_pub.getNumSubscribers() > 0)
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
            det.area = blob.area;
            det.circularity = blob.circularity;
            det.convexity = blob.convexity;
            det.avg_depth = blob.avg_depth;
            det.known_pixels = blob.known_pixels;
            det.angle = blob.angle;
            det.inertia = blob.inertia;
            det.confidence = blob.confidence;
            det.radius = blob.radius;
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
          m_detected_blobs_pub.publish(dets);
          //}
        }

        /* Update statistics for info_loop //{ */
        {
          std::lock_guard<std::mutex> lck(m_stat_mtx);
          const ros::Time end_t = ros::Time::now();
          const float delay = (end_t - source_msg.header.stamp).toSec();
          m_avg_delay = 0.9*m_avg_delay + 0.1*delay;
          const float fps = 1/(end_t - start_t).toSec();
          m_avg_fps = 0.9*m_avg_fps + 0.1*fps;
          m_images_processed++;
          m_det_blobs += blobs.size();
        }
        //}
      }
    }
    //}

    /* info_loop() method //{ */
    void info_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      const float dt = (evt.current_real - evt.last_real).toSec();
      std::lock_guard<std::mutex> lck(m_stat_mtx);
      const float blobs_per_image = m_det_blobs/float(m_images_processed);
      const float input_fps = m_images_processed/dt;
      ROS_INFO_STREAM("[" << m_node_name << "]: det. blobs/image: " << blobs_per_image << " | inp. FPS: " << round(input_fps) << " | proc. FPS: " << round(m_avg_fps) << " | delay: " << round(1000.0f*m_avg_delay) << "ms");
      m_det_blobs = 0;
      m_images_processed = 0;
    }
    //}

  private:

    // --------------------------------------------------------------
    // |                ROS-related member variables                |
    // --------------------------------------------------------------

    /* Parameters, loaded from ROS //{ */
    int m_unknown_pixel_value;
    //}

    /* ROS related variables (subscribers, timers etc.) //{ */
    std::unique_ptr<drmgr_t> m_drmgr_ptr;
    mrs_lib::SubscribeHandlerPtr<sensor_msgs::Image> m_depthmap_sh;
    ros::Publisher m_detections_pub;
    ros::Publisher m_detected_blobs_pub;
    ros::Publisher m_processed_deptmap_pub;
    ros::Timer m_main_loop_timer;
    ros::Timer m_info_loop_timer;
    std::string m_node_name;
    //}

  private:

    // --------------------------------------------------------------
    // |                   Other member variables                   |
    // --------------------------------------------------------------

    /* Image mask //{ */
    cv::Mat m_mask_im;
    //}

    uint32_t m_last_detection_id;
    
    /* Statistics variables //{ */
    std::mutex m_stat_mtx;
    unsigned   m_det_blobs;
    unsigned   m_images_processed;
    float      m_avg_fps;
    float      m_avg_delay;
    //}

  }; // class DepthDetector
}; // namespace uav_detect

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(uav_detect::DepthDetector, nodelet::Nodelet)
