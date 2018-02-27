
#include <cmath>
#include <limits>

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

class Detected_UAV
{
  public:
    Detected_UAV(uav_detect::Detection det, tf2::Transform camera2world_tf, float IoU_threshold = 0.0);

    // returns index of the matching detection or -1 if no matching was found
    int update(const uav_detect::Detections& new_detections, tf2::Transform camera2world_tf);
    float get_prob() {return _prob;};
    float est_x() {return _cur_x;};
    float est_y() {return _cur_y;};
    float est_z() {return _cur_z;};
  private:
    // Parameters
    float _IoU_threshold;
    const float _UAV_width;  // in meters

    // Internal variables
    float _prob;
    float _cur_x;
    float _cur_y;
    float _cur_z;
    uav_detect::Detection _ref_det;
    tf2::Transform _c2w_tf;

    float IoU(const uav_detect::Detection &det1, const uav_detect::Detection &det2);
};
