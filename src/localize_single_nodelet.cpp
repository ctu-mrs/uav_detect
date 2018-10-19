#include "main.h"
#include "utils.h"
#include "mrs_lib/Lkf.h"

#include <nodelet/nodelet.h>

using namespace cv;
using namespace std;
using namespace uav_detect;
using namespace Eigen;

namespace uav_detect
{
  class LocalizeSingle : public nodelet::Nodelet
  {
  private:
    double m_lkf_dt;
    string m_world_frame;

  private:
    tf2_ros::Buffer m_tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> m_tf_listener_ptr;
    ros::Subscriber m_sub_detections;
    ros::Subscriber m_sub_camera_info;
    ros::Timer m_lkf_update_timer;
    ros::Timer m_main_loop_timer;

  public:

    /* LocalizeSingle() constructor //{ */
    LocalizeSingle()
      : m_new_detections(false)
    {
    }
    //}
    
    /* onInit() method //{ */
    void onInit()
    {
      ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

      /* Load parameters from ROS //{*/
      mrs_lib::ParamLoader pl(nh, "LocalizeSingle");
      // LOAD STATIC PARAMETERS
      ROS_INFO("Loading static parameters:");
      pl.load_param("world_frame", m_world_frame, std::string("local_origin"));
      pl.load_param("lkf_dt", m_lkf_dt);

      if (!pl.loaded_successfully())
      {
        ROS_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
        ros::shutdown();
      }
      //}

      /* Create publishers and subscribers //{ */
      // Initialize transform listener
      m_tf_listener_ptr = std::make_unique<tf2_ros::TransformListener>(m_tf_buffer);
      // Initialize other subs and pubs
      m_sub_detections = nh.subscribe("detections", 1, &LocalizeSingle::detections_callback, this, ros::TransportHints().tcpNoDelay());
      m_sub_camera_info = nh.subscribe("camera_info", 1, &LocalizeSingle::camera_info_callback, this, ros::TransportHints().tcpNoDelay());
      //}

      m_lkf_update_timer = nh.createTimer(ros::Duration(m_lkf_dt), &LocalizeSingle::lkf_update, this);
      m_main_loop_timer = nh.createTimer(ros::Duration(m_lkf_dt), &LocalizeSingle::main_loop, this);

      cout << "----------------------------------------------------------" << std::endl;
    }
    //}

  private:

    /* detection_to_3dpoint() method //{ */
    Eigen::Vector3d detection_to_3dpoint(uav_detect::Detection det)
    {
      Eigen::Vector3d ret;
      double u = det.x * det.roi.width + det.roi.x_offset;
      double v = det.y * det.roi.height + det.roi.y_offset;
      double x = (u - m_camera_model.cx())/m_camera_model.fx();
      double y = (v - m_camera_model.cy())/m_camera_model.fy();
      ret << x, y, 1.0;
      ret *= det.depth;
      return ret;
    }
    //}

    /* get_transform_to_world() method //{ */
    bool get_transform_to_world(string frame_name, Eigen::Affine3d tf)
    {
      try
      {
        const ros::Duration timeout(1.0 / 6.0);
        geometry_msgs::TransformStamped transform;
        // Obtain transform from world into uav frame
        transform = m_tf_buffer.lookupTransform(frame_name, m_world_frame, m_last_detections_msg.header.stamp, timeout);
        /* tf2::convert(transform, world2uav_transform); */

        // Obtain transform from camera frame into world
        tf = tf2_to_eigen(transform.transform).inverse();
      }
      catch (tf2::TransformException& ex)
      {
        ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", m_world_frame.c_str(), frame_name.c_str(), ex.what());
        return false;
      }
      return true;
    }
    //}

    /* main_loop() method //{ */
    void main_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      ros::Time start_t = ros::Time::now();

      if (m_new_detections)
      {
        cout << "Processsing new detections" << std::endl;

        string sensor_frame = m_last_detections_msg.header.frame_id;
        // Construct a new world to camera transform
        Eigen::Affine3d s2w_tf;
        bool tf_ok = get_transform_to_world(sensor_frame, s2w_tf);

        if (!tf_ok)
          return;

        // TODO: process the detections, create new lkfs etc.
        for (const uav_detect::Detection& det : m_last_detections_msg.detections)
        {
          Eigen::Vector3d det_pos = detection_to_3dpoint(det);
          Eigen::Matrix3d det_cov; // TODO!

          {
            std::lock_guard<std::mutex> lck(m_lkfs_mtx);
            for (const mrs_lib::Lkf& lkf : m_lkfs)
            {
              // TODO:
              Eigen::Vector3d lkf_pos = lkf.getStates().block<3, 1>(0, 0);
              Eigen::Matrix3d lkf_cov = lkf.getCovariance().block<3, 3>(0, 0);
              double divergence = kullback_leibler_divergence(det_pos, det_cov, lkf_pos, lkf_cov);
            
            }
          }
        
        }

        cout << "Detections processed" << std::endl;
        ros::Time end_t = ros::Time::now();
        static double dt = (end_t - start_t).toSec();
        dt = 0.9 * dt + 0.1 * (end_t - start_t).toSec();
        cout << "processing FPS: " << 1 / dt << "Hz" << std::endl;
      }
    }
    //}

  private:
    bool m_new_detections;
    uav_detect::Detections m_last_detections_msg;
    bool m_got_camera_info;
    image_geometry::PinholeCameraModel m_camera_model;


    /* Callbacks //{ */

    // Callback for the detections
    void detections_callback(const uav_detect::Detections& detections_msg)
    {
      ROS_INFO_THROTTLE(1.0, "Getting detections");
      m_last_detections_msg = detections_msg;
      m_new_detections = true;
    }
    
    // Callback for the camera info
    void camera_info_callback(const sensor_msgs::CameraInfo& cinfo_msg)
    {
      if (!m_got_camera_info)
      {
        ROS_INFO_THROTTLE(1.0, "Got camera info");
        m_camera_model.fromCameraInfo(cinfo_msg);
        m_got_camera_info = true;
      }
    }
    //}

  private:
    std::mutex m_lkfs_mtx;
    std::vector<mrs_lib::Lkf> m_lkfs;

    /* Timers //{ */
    void lkf_update(const ros::TimerEvent& evt)
    {
      double dt = (evt.current_real - evt.last_real).toSec();
      Eigen::Matrix<double, 6, 6> A;
      A << 1, 0, 0, dt, 0, 0,
           0, 1, 0, 0, dt, 0,
           0, 0, 1, 0, 0, dt,
           0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 1;

      {
        std::lock_guard<std::mutex> lck(m_lkfs_mtx);
        for (auto& lkf : m_lkfs)
        {
          lkf.setA(A);
          lkf.iterateWithoutCorrection();
        }
      }
    }
    //}

  private:
    double kullback_leibler_divergence(Eigen::Vector3d mu0, Eigen::Matrix3d sigma0, Eigen::Vector3d mu1, Eigen::Matrix3d sigma1)
    {
      const unsigned k = 2; // number of dimensions
      double div = 0.5*( (sigma1.inverse()*sigma0).trace() + (mu1-mu0).transpose()*(sigma1.inverse())*(mu1-mu0) - k + log((sigma1.determinant())/sigma0.determinant()));
      return div;
    }
  };
};

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(uav_detect::LocalizeSingle, nodelet::Nodelet)
