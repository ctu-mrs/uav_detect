#include "main.h"
#include "utils.h"
#include "mrs_lib/Lkf.h"

#include <nodelet/nodelet.h>

using namespace cv;
using namespace std;
using namespace uav_detect;
/* using namespace Eigen; */

namespace uav_detect
{
  class LocalizeSingle : public nodelet::Nodelet
  {
  private:

    /* Parameters, loaded from ROS //{ */
    double m_lkf_dt;
    string m_world_frame;
    double m_xy_covariance_coeff;
    double m_z_covariance_coeff;
    double m_max_update_divergence;
    double m_max_lkf_uncertainty;
    double m_lkf_process_noise;
    //}

  private:

    /* ROS related variables //{ */
    tf2_ros::Buffer m_tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> m_tf_listener_ptr;
    ros::Subscriber m_sub_detections;
    ros::Subscriber m_sub_camera_info;
    ros::Timer m_lkf_update_timer;
    ros::Timer m_main_loop_timer;
    //}

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
      pl.load_param("xy_covariance_coeff", m_xy_covariance_coeff);
      pl.load_param("z_covariance_coeff", m_z_covariance_coeff);
      pl.load_param("max_update_divergence", m_max_update_divergence);
      pl.load_param("max_lkf_uncertainty", m_max_lkf_uncertainty);
      pl.load_param("lkf_process_noise", m_lkf_process_noise);

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
    Eigen::Vector3d detection_to_3dpoint(const uav_detect::Detection& det)
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
    bool get_transform_to_world(const string& frame_name, Eigen::Affine3d& tf_out)
    {
      try
      {
        const ros::Duration timeout(1.0 / 6.0);
        geometry_msgs::TransformStamped transform;
        // Obtain transform from world into uav frame
        transform = m_tf_buffer.lookupTransform(frame_name, m_world_frame, m_last_detections_msg.header.stamp, timeout);
        /* tf2::convert(transform, world2uav_transform); */

        // Obtain transform from camera frame into world
        tf_out = tf2_to_eigen(transform.transform).inverse();
      }
      catch (tf2::TransformException& ex)
      {
        ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", m_world_frame.c_str(), frame_name.c_str(), ex.what());
        return false;
      }
      return true;
    }
    //}

    /* calc_position_covariance() method //{ */
    /* position_sf is position of the detection in 3D in the frame of the sensor (camera) */
    Eigen::Matrix3d calc_position_covariance(const Eigen::Vector3d& position_sf)
    {
      /* Calculates the corresponding covariance matrix of the estimated 3D position */
      Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Identity();  // prepare the covariance matrix
      const double tol = 1e-9;
      pos_cov(0, 0) = pos_cov(1, 1) = m_xy_covariance_coeff;

      pos_cov(2, 2) = position_sf(2)*sqrt(position_sf(2))*m_z_covariance_coeff;
      if (pos_cov(2, 2) < 0.33*m_z_covariance_coeff)
        pos_cov(2, 2) = 0.33*m_z_covariance_coeff;

      // Find the rotation matrix to rotate the covariance to point in the direction of the estimated position
      const Eigen::Vector3d a(0.0, 0.0, 1.0);
      const Eigen::Vector3d b = position_sf.normalized();
      const Eigen::Vector3d v = a.cross(b);
      const double sin_ab = v.norm();
      const double cos_ab = a.dot(b);
      Eigen::Matrix3d vec_rot = Eigen::Matrix3d::Identity();
      if (sin_ab < tol)  // unprobable, but possible - then it is identity or 180deg
      {
        if (cos_ab + 1.0 < tol)  // that would be 180deg
        {
          vec_rot << -1.0, 0.0, 0.0,
                     0.0, -1.0, 0.0,
                     0.0, 0.0, 1.0;
        } // otherwise its identity
      } else  // otherwise just construct the matrix
      {
        Eigen::Matrix3d v_x; v_x << 0.0, -v(2), v(1),
                                    v(2), 0.0, -v(0),
                                    -v(1), v(0), 0.0;
        vec_rot = Eigen::Matrix3d::Identity() + v_x + (1-cos_ab)/(sin_ab*sin_ab)*(v_x*v_x);
      }
      rotate_covariance(pos_cov, vec_rot);  // rotate the covariance to point in direction of est. position
      return pos_cov;
    }
    //}

    Eigen::Matrix3d rotate_covariance(const Eigen::Matrix3d& covariance, const Eigen::Matrix3d& rotation)
    {
      return rotation*covariance*rotation.transpose();  // rotate the covariance to point in direction of est. position
    }

    /* calc_LKF_uncertainty() method //{ */
    double calc_LKF_uncertainty(const mrs_lib::Lkf& lkf)
    {
      Eigen::Matrix3d position_covariance = lkf.getCovariance().block<3, 3>(0, 0);
      double determinant = position_covariance.determinant();
      return sqrt(determinant);
    }
    //}

    struct pos_cov_t
    {
      Eigen::Vector3d position;
      Eigen::Matrix3d covariance;
    };

    /* find_closest_measurement() method //{ */
    /* returns position of the closest measurement in the pos_covs vector */
    size_t find_closest_measurement(const mrs_lib::Lkf& lkf, const std::vector<pos_cov_t>& pos_covs, double& min_divergence_out)
    {
      const Eigen::Vector3d& lkf_pos = lkf.getStates().block<3, 1>(0, 0);
      const Eigen::Matrix3d& lkf_cov = lkf.getCovariance().block<3, 3>(0, 0);
      double min_divergence = std::numeric_limits<double>::max();
      size_t min_div_it = 0;

      // Find measurement with smallest divergence from this LKF and assign the measurement to it
      for (size_t it = 0; it < pos_covs.size(); it++)
      {
        const auto& pos_cov = pos_covs.at(it);
        const Eigen::Vector3d& det_pos = pos_cov.position;
        const Eigen::Matrix3d& det_cov = pos_cov.covariance;
        const double divergence = kullback_leibler_divergence(det_pos, det_cov, lkf_pos, lkf_cov);
      
        if (divergence < min_divergence)
        {
          min_divergence = divergence;
          min_div_it = it;
        }
      }
      min_divergence_out = min_divergence;
      return min_div_it;
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
        // TODO: assignment problem?? (https://en.wikipedia.org/wiki/Hungarian_algorithm)
        {
          size_t n_meas = m_last_detections_msg.detections.size();
          vector<int> meas_used(n_meas, 0);
          vector<pos_cov_t> pos_covs;
          pos_covs.reserve(n_meas);

          /* Calculate 3D positions and covariances of the detections //{ */
          for (const uav_detect::Detection& det : m_last_detections_msg.detections)
          {
            const Eigen::Vector3d det_pos = s2w_tf*detection_to_3dpoint(det);
            const Eigen::Matrix3d det_cov = rotate_covariance(calc_position_covariance(det_pos), s2w_tf.rotation());

            pos_cov_t pos_cov;
            pos_cov.position = det_pos;
            pos_cov.covariance = det_cov;
            pos_covs.push_back(pos_cov);
          }
          //}

          /* Process the LKFs - assign measurements and kick out too uncertain ones //{ */
          if (!m_lkfs.empty())
          {
            std::lock_guard<std::mutex> lck(m_lkfs_mtx);
            for (list<mrs_lib::Lkf>::iterator it = std::begin(m_lkfs); it != std::end(m_lkfs); it++)
            {
              auto& lkf = *it;

              /* Assign a measurement to the LKF based on the smallest divergence //{ */
              {
                double divergence;
                size_t closest_it = find_closest_measurement(lkf, pos_covs, divergence);

                /* Evaluate whether the divergence is small enough to justify the update //{ */
                if (divergence < m_max_update_divergence)
                {
                  Eigen::Vector3d closest_pos = pos_covs.at(closest_it).position;
                  Eigen::Matrix3d closest_cov = pos_covs.at(closest_it).covariance;
                  lkf.setMeasurement(closest_pos, closest_cov);
                  lkf.doCorrection();
                  meas_used.at(closest_it)++;
                }
                //}

              }
              //}

              /* Check if the uncertainty of this LKF is too high and if so, delete it //{ */
              {
                double uncertainty = calc_LKF_uncertainty(lkf);
                if (uncertainty > m_max_lkf_uncertainty)
                {
                  it = m_lkfs.erase(it);
                  it--;
                }
              }
              //}

            }
          }
          //}

          /* Instantiate new LKFs for unused measurements //{ */
          {
            std::lock_guard<std::mutex> lck(m_lkfs_mtx);
            for (size_t it = 0; it < n_meas; it++)
            {
              if (meas_used.at(it) < 1)
                create_new_lkf(m_lkfs, pos_covs.at(it));
            }
          }
          //}

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
    std::list<mrs_lib::Lkf> m_lkfs;
    static const int c_n_states = 6;
    static const int c_n_inputs = 0;
    static const int c_n_measurements = 3;
    typedef Eigen::Matrix<double, c_n_states, c_n_states> lkf_A_t;
    typedef Eigen::Matrix<double, c_n_states, c_n_inputs> lkf_B_t;
    typedef Eigen::Matrix<double, c_n_measurements, c_n_states> lkf_P_t;
    typedef Eigen::Matrix<double, c_n_states, c_n_states> lkf_R_t;
    typedef Eigen::Matrix<double, c_n_measurements, c_n_measurements> lkf_Q_t;

    lkf_A_t create_A(double dt)
    {
      lkf_A_t A;
      A << 1, 0, 0, dt, 0, 0,
           0, 1, 0, 0, dt, 0,
           0, 0, 1, 0, 0, dt,
           0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 1;
      return A;
    }

    lkf_P_t create_P()
    {
      lkf_P_t P;
      P << 1, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0;
      return P;
    }

    lkf_R_t create_R()
    {
      lkf_R_t R = m_lkf_process_noise*lkf_R_t::Identity();
      return R;
    }

    /* create_new_lkf() method //{ */
    void create_new_lkf(std::list<mrs_lib::Lkf>& lkfs, pos_cov_t& initialization)
    {
      const int n_states = 6;
      const int n_inputs = 0;
      const int n_measurements = 3;
      const lkf_A_t A; // changes in dependence on the measured dt, so leave blank for now
      const lkf_B_t B; // zero rows zero cols matrix
      const lkf_P_t P = create_P();
      const lkf_R_t R = create_R();
      const lkf_Q_t Q = initialization.covariance;
    
      lkfs.emplace_back(n_states, n_inputs, n_measurements, A, B, R, Q, P);
      lkfs.back().setMeasurement(initialization.position, initialization.covariance);
      // TODO: initialize the LKF properly
    }
    //}

    /* lkf_update() method //{ */
    void lkf_update(const ros::TimerEvent& evt)
    {
      double dt = (evt.current_real - evt.last_real).toSec();
      lkf_A_t A = create_A(dt);

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
