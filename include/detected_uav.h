
#include <cmath>
#include <limits>
#include <memory>

#include <ros/ros.h>
#include <uav_detect/Detection.h>
#include <uav_detect/Detections.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/CameraInfo.h>

#include <mrs_estimation/lkf.h>

class Detected_UAV
{
  public:
    Detected_UAV(
                  float IoU_threshold = 0.0,
                  double UAV_width = 0.55,
                  ros::NodeHandle *nh = nullptr
                  );

    void initialize(
                  const uav_detect::Detection& det,
                  int w_used,
                  int h_used,
                  const sensor_msgs::CameraInfo& camera_info,
                  const tf2::Transform& camera2world_tf);
    // returns index of the matching detection or -1 if no matching was found
    int update(const uav_detect::Detections& new_detections, const tf2::Transform& camera2world_tf);
    //float get_prob() {return _prob;};
    double get_x() {return _KF->getState(0);};
    double get_y() {return _KF->getState(1);};
    double get_z() {return _KF->getState(2);};
  private:
    // Parameters
    float _IoU_threshold;
    const double _UAV_width;  // in meters
    const double _tol;
    const double _max_det_dist; // meters - maximal distance for the drone detection
    const double _max_dist_est; // meters - maximal distance for "kind of precise" distance estimation from b.b. width

    // Internal variables
    std::unique_ptr<LinearKF> _KF;

    int _w_used, _h_used;
    int _w_camera, _h_camera;
    image_geometry::PinholeCameraModel _camera_model;
    Eigen::Affine3d _c2w_tf;

    bool _dbg_on;
    ros::Publisher _dbg_pub;

    uav_detect::Detection get_reference_detection();
    void detection_to_position(
                        const uav_detect::Detection& det,
                        Eigen::Vector3d& out_meas_position,
                        Eigen::Matrix3d& out_meas_covariance);
    float IoU(const uav_detect::Detection& det1, const uav_detect::Detection& det2);
};
