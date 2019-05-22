#include "main.h"

#include <nodelet/nodelet.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <uav_detect/DetectionParamsConfig.h>

// shortcut type to the dynamic reconfigure manager template instance
typedef mrs_lib::DynamicReconfigureMgr<uav_detect::DetectionParamsConfig> drmgr_t;

namespace uav_detect
{

  class DepthmapPreprocessor : public nodelet::Nodelet
  {
  public:

    /* onInit() method //{ */
    void onInit()
    {
      ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

      m_node_name = "DepthmapPreprocessor";

      /* Load parameters from ROS //{*/
      mrs_lib::ParamLoader pl(nh, m_node_name);
      // LOAD STATIC PARAMETERS
      ROS_INFO("Loading static parameters:");
      pl.load_param("unknown_pixel_value", m_unknown_pixel_value, 0);
      m_roi.x_offset = pl.load_param2<int>("roi/x_offset", 0);
      m_roi.y_offset = pl.load_param2<int>("roi/y_offset", 0);
      m_roi.width = pl.load_param2<int>("roi/width", 0);
      m_roi.height = pl.load_param2<int>("roi/height", 0);
      std::string path_to_mask = pl.load_param2<std::string>("path_to_mask", std::string());

      // LOAD DYNAMIC PARAMETERS
      // CHECK LOADING STATUS
      if (!pl.loaded_successfully())
      {
        ROS_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
        ros::shutdown();
      }

      m_drmgr_ptr = std::make_unique<drmgr_t>(nh, m_node_name);
      //}

      m_depthmap_sub = nh.subscribe("depthmap", 1, &DepthmapPreprocessor::dephmap_callback, this, ros::TransportHints().tcpNoDelay());
      m_processed_depthmap_pub = nh.advertise<sensor_msgs::Image>("processed_depthmap", 1);
    }
    //}

  private:
    void dephmap_callback(sensor_msgs::Image::ConstPtr msg)
    {
      ROS_INFO_THROTTLE(1.0, "[DepthmapPreprocessor]: Receiving dephmap");
      cv_bridge::CvImage source_msg = *cv_bridge::toCvCopy(msg, std::string("16UC1"));

      /* Apply ROI //{ */
      if (m_roi.y_offset + m_roi.height > unsigned(source_msg.image.rows) || m_roi.height == 0)
        m_roi.height = std::clamp(int(source_msg.image.rows - m_roi.y_offset), 0, source_msg.image.rows);
      if (m_roi.x_offset + m_roi.width > unsigned(source_msg.image.cols) || m_roi.width == 0)
        m_roi.width = std::clamp(int(source_msg.image.cols - m_roi.x_offset), 0, source_msg.image.cols);
      
      cv::Rect roi(m_roi.x_offset, m_roi.y_offset, m_roi.width, m_roi.height);
      source_msg.image = source_msg.image(roi);
      //}

      /* Prepare the image for detection //{ */
      // create the detection image
      cv::Mat detect_im = source_msg.image.clone();
      cv::Mat raw_im = source_msg.image;
      cv::Mat known_pixels;
      if (m_unknown_pixel_value != std::numeric_limits<uint16_t>::max() || m_drmgr_ptr->config.blur_empty_areas)
      {
        cv::compare(raw_im, m_unknown_pixel_value, known_pixels, cv::CMP_NE);
      }

      if (m_drmgr_ptr->config.blur_empty_areas)
      {
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 3), cv::Point(-1, -1));
        cv::Mat mask, tmp;
        mask = ~known_pixels;
        cv::dilate(detect_im, tmp, element, cv::Point(-1, -1), 3);
        /* cv::GaussianBlur(tmp, tmp, cv::Size(19, 75), 40); */
        cv::blur(detect_im, tmp, cv::Size(115, 215));
        tmp.copyTo(detect_im, mask);
      }

      // dilate and erode the image if requested
      {
        const int elem_a = m_drmgr_ptr->config.structuring_element_a;
        const int elem_b = m_drmgr_ptr->config.structuring_element_b;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(elem_a, elem_b), cv::Point(-1, -1));
        cv::dilate(detect_im, detect_im, element, cv::Point(-1, -1), m_drmgr_ptr->config.dilate_iterations);
        cv::erode(detect_im, detect_im, element, cv::Point(-1, -1), m_drmgr_ptr->config.erode_iterations);

        // erode without using zero (unknown) pixels
        if (m_drmgr_ptr->config.erode_ignore_empty_iterations > 0)
        {
          cv::Mat unknown_as_max = detect_im;
          if (m_unknown_pixel_value != std::numeric_limits<uint16_t>::max())
          {
            unknown_as_max = cv::Mat(raw_im.size(), CV_16UC1, std::numeric_limits<uint16_t>::max());
            detect_im.copyTo(unknown_as_max, known_pixels);
          }
          cv::erode(unknown_as_max, detect_im, element, cv::Point(-1, -1), m_drmgr_ptr->config.erode_ignore_empty_iterations);
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

      cv_bridge::CvImage processed_depthmap_cvb = source_msg;
      processed_depthmap_cvb.image = detect_im;
      sensor_msgs::ImageConstPtr out_msg = processed_depthmap_cvb.toImageMsg();
      m_processed_depthmap_pub.publish(out_msg);
    }

  private:
    /* Parameters, loaded from ROS //{ */
    int m_unknown_pixel_value;
    sensor_msgs::RegionOfInterest m_roi;
    //}

    ros::Subscriber m_depthmap_sub;
    ros::Publisher m_processed_depthmap_pub;
    std::unique_ptr<drmgr_t> m_drmgr_ptr;
    std::string m_node_name;

  }; // class PCLDetector
}; // namespace uav_detect

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(uav_detect::DepthmapPreprocessor, nodelet::Nodelet)
