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
      //}

      m_lkf_update_timer = nh.createTimer(ros::Duration(m_lkf_dt), &LocalizeSingle::lkf_update, this);
      m_main_loop_timer = nh.createTimer(ros::Duration(m_lkf_dt), &LocalizeSingle::main_loop, this);

      cout << "----------------------------------------------------------" << std::endl;
    }
    //}

  private:

    /* main_loop() method //{ */
    void main_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      ros::Time start_t = ros::Time::now();

      if (m_new_detections)
      {
        cout << "Processsing new detections" << std::endl;

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

        // TODO: process the detections, create new lkfs etc.

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

    /* Callbacks //{ */

    // Callback for the detections
    void detections_callback(const uav_detect::Detections& detections_msg)
    {
      ROS_INFO_THROTTLE(1.0, "Getting detections");
      m_last_detections_msg = detections_msg;
      m_new_detections = true;
    }
    //}

  private:
    std::mutex m_lkfs_mtx;
    std::vector<mrs_lib::Lkf> m_lkfs;

    /* Timers //{ */
    void lkf_update(const ros::TimerEvent& evt)
    {
      double dt = (evt.current_real - evt.last_real).toSec();
      Eigen::Matrix<double, 4, 4> A;
      A << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1;

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

  };
};

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(uav_detect::LocalizeSingle, nodelet::Nodelet)
