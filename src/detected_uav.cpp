#include "detected_uav.h"

using namespace uav_detect;
using namespace std;
using namespace Eigen;
using namespace ros;

/* mgauss - implementation of multivariate gaussian distribution *//*//{*/
// from https://stackoverflow.com/questions/41538095/evaluate-multivariate-normal-gaussian-density-in-c
static double mgauss(VectorXd x, VectorXd mu, MatrixXd sigma)
{
  // avoid magic numbers in your code. Compilers will be able to compute this at compile time:
  const double logSqrt2Pi = 0.5*std::log(2*M_PI);
  typedef Eigen::LLT<Eigen::MatrixXd> Chol;
  Chol chol(sigma);
  // Handle non positive definite covariance somehow:
  if(chol.info()!=Eigen::Success) throw "decomposition failed!";
  const Chol::Traits::MatrixL& L = chol.matrixL();
  double quadform = (L.solve(x - mu)).squaredNorm();
  return std::exp(-x.rows()*logSqrt2Pi - 0.5*quadform) / L.determinant();
}/*//}*/

/* tf2_to_eigen - helper function to convert tf2::Transform to Eigen::Affine3d *//*//{*/
static Eigen::Affine3d tf2_to_eigen(const tf2::Transform& tf2_t)
{
  Eigen::Affine3d eig_t;
  for (int r_it = 0; r_it < 3; r_it++)
    for (int c_it = 0; c_it < 3; c_it++)
      eig_t(r_it, c_it) = tf2_t.getBasis()[r_it][c_it];
  eig_t(0, 3) = tf2_t.getOrigin().getX();
  eig_t(1, 3) = tf2_t.getOrigin().getY();
  eig_t(2, 3) = tf2_t.getOrigin().getZ();
  return eig_t;
}/*//}*/

/* Detected_UAV constructor *//*//{*/
Detected_UAV::Detected_UAV(
                            double association_threshold,
                            double unreliable_threshold,
                            double similarity_threshold,
                            double UAV_width,
                            NodeHandle *nh
                            ) : _association_threshold(association_threshold),
                                _unreliable_threshold(unreliable_threshold),
                                _similarity_threshold(similarity_threshold),
                                _UAV_width(UAV_width),
                                _tol(1e-9),
                                _max_det_dist(15.0),
                                _max_dist_est(2.0)
{
  double est_dt = 0.2;
  const int n = 6; // number of states
  const int m = 0; // number of inputs
  const int p = 3; // number of measurements
  Matrix<double, n, n> A; // state transition matrix
  A << 1.0, 0.0, 0.0, est_dt, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0, est_dt, 0.0,
       0.0, 0.0, 1.0, 0.0, 0.0, est_dt,
       0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  Matrix<double, n, m> B; // input matrix (empty)
  Matrix<double, p, n> H; // measurement matrix
  H << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
//  H << 1.0, 0.0, 0.0,
//       0.0, 1.0, 0.0,
//       0.0, 0.0, 1.0,
//       0.0, 0.0, 0.0,
//       0.0, 0.0, 0.0,
//       0.0, 0.0, 0.0;
  Matrix<double, n, n> R; // process covariance matrix
  R = Matrix<double, n, n>::Zero();
  R(0, 0) = R(1, 1) = R(2, 2) = 0.2;
  R(3, 3) = R(4, 4) = R(5, 5) = 0.3;
  Matrix<double, n, n> Q; // measurement covariance matrix
  Q = Matrix<double, n, n>::Identity(); // this will change depending on the measurement!

  if (nh == nullptr)
  {
    _dbg_on = false;
  } else
  {
    _dbg_pub = nh->advertise<nav_msgs::Odometry>("dbg", 1);
    _dbg_on = true;
  }
  _KF = unique_ptr<LinearKF>(new LinearKF(n, m, p, A, B, R, Q, H));
  //_KF->setInput(Matrix<double, m, 1>());
}/*//}*/

/* Detected_UAV::detection_to_position - calculates a 3D position from a detected bounding box *//*//{*/
void Detected_UAV::detection_to_position(
                        const uav_detect::Detection &det,
                        Eigen::Vector3d &out_meas_position,
                        Eigen::Matrix3d &out_meas_covariance)
{
  int w_used = _w_used;
  int h_used = _h_used;
  int cam_image_w = _w_camera;
  int cam_image_h = _h_camera;

  // Calculate pixel position of the detection
  int px_x = (int)round(
                    (cam_image_w-w_used)/2.0 +  // offset between the detection rectangle and camera image
                    (det.x_relative)*w_used);
  int px_y = (int)round(
                    (cam_image_h-h_used)/2.0 +  // offset between the detection rectangle and camera image
                    (det.y_relative)*h_used);
  int px_w = (int)round(det.w_relative*w_used);
  cv::Point2d center_pt = _camera_model.rectifyPoint(cv::Point2d(px_x, px_y));
  cv::Point2d left_pt = _camera_model.rectifyPoint(cv::Point2d(px_x-px_w/2, px_y));
  cv::Point2d right_pt = _camera_model.rectifyPoint(cv::Point2d(px_x+px_w/2, px_y));

  // Calculate projections of the center, left and right points of the detected bounding box
  cv::Point3d ray_vec = _camera_model.projectPixelTo3dRay(center_pt);
  cv::Point3d left_vec = _camera_model.projectPixelTo3dRay(left_pt);
  cv::Point3d right_vec = _camera_model.projectPixelTo3dRay(right_pt);

  // Width of the projection (on a plane 1.0m distant)
  double proj_width = sqrt((left_vec.x-right_vec.x)*(left_vec.x-right_vec.x)
                         +(left_vec.y-right_vec.y)*(left_vec.y-right_vec.y));
                         //+(left_vec.z-right_vec.z)*(left_vec.z-right_vec.z)); // z coordinate should be 1.0
  // Estimate distance from the BB width
  double est_dist = _UAV_width/proj_width;
  double ray_vec_norm = sqrt(ray_vec.x*ray_vec.x + ray_vec.y*ray_vec.y + ray_vec.z*ray_vec.z);
  Eigen::Vector3d cur_position_estimate(ray_vec.x/ray_vec_norm,
                                        ray_vec.y/ray_vec_norm,
                                        ray_vec.z/ray_vec_norm);
  Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Identity();  // prepare the covariance matrix
  if (est_dist > _max_dist_est) // further than very close distances the distance estimation is very unreliable
  {
    cur_position_estimate *= _max_det_dist-_max_dist_est;
    pos_cov(2, 2) = (_max_det_dist-_max_dist_est)*2.0;
  } else
  {
    cur_position_estimate *= est_dist;
    pos_cov(2, 2) = 1.0;
  }

  // Rotation matrix to rotate the covariance matrix and position vector
  Eigen::Matrix3d c2w_rot = _c2w_tf.rotation();
  // Find the rotation matrix to rotate the covariance to point in the direction of the estimated position
  Eigen::Vector3d a(0.0, 0.0, 1.0);
  Eigen::Vector3d b = cur_position_estimate.normalized();
  Eigen::Vector3d v = a.cross(b);
  double sin_ab = v.norm();
  double cos_ab = a.dot(b);
  Eigen::Matrix3d vec_rot = Eigen::Matrix3d::Identity();
  if (sin_ab < _tol)  // unprobable, but possible - then it is identity or 180deg
  {
    if (cos_ab + 1.0 < _tol)  // that would be 180deg
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
  pos_cov = vec_rot*pos_cov*vec_rot.transpose();  // rotate the covariance to point in direction of est. position
  pos_cov = c2w_rot*pos_cov*c2w_rot.transpose();  // rotate the covariance into local_origin tf
  cur_position_estimate = _c2w_tf*cur_position_estimate; // transform the position estimate to world coordinate system

  /** for debug only **//*//{*/
  if (_dbg_on)
  {
    Eigen::Matrix3d rot_mat = Eigen::Matrix3d::Identity();
    rot_mat = c2w_rot*rot_mat;
    Eigen::Matrix3d rot_cov = 0.1*Eigen::Matrix3d::Identity();
    rot_cov = c2w_rot*rot_cov*c2w_rot.transpose();
    // Fill the covariance array
    nav_msgs::Odometry dbg_msg;
    for (int r_it = 0; r_it < 6; r_it++)
      for (int c_it = 0; c_it < 6; c_it++)
        if (r_it < 3 && c_it < 3)
          dbg_msg.pose.covariance[r_it + c_it*6] = pos_cov(r_it, c_it);
        else if (r_it >= 3 && c_it >= 3)
          dbg_msg.pose.covariance[r_it + c_it*6] = rot_cov(r_it-3, c_it-3);
        else
          dbg_msg.pose.covariance[r_it + c_it*6] = 0.0;
    dbg_msg.header.stamp = ros::Time::now();
    dbg_msg.header.frame_id = "local_origin";
    dbg_msg.pose.pose.position.x = cur_position_estimate(0);
    dbg_msg.pose.pose.position.y = cur_position_estimate(1);
    dbg_msg.pose.pose.position.z = cur_position_estimate(2);
    Eigen::Quaterniond tmp(rot_mat);
    dbg_msg.pose.pose.orientation.x = tmp.x();
    dbg_msg.pose.pose.orientation.y = tmp.y();
    dbg_msg.pose.pose.orientation.z = tmp.z();
    dbg_msg.pose.pose.orientation.w = tmp.w();
    _dbg_pub.publish(dbg_msg);
    cout << "Debug published!" << std::endl;
  } else
  {
    cout << "Debug disabled" << std::endl;
  }
  /** end of debug **//*//}*/

  // assign output variables
  out_meas_covariance = pos_cov;
  out_meas_position = cur_position_estimate;
}/*//}*/

/* Detected_UAV::initialize - initializes this detected UAV with a first detection and camera information *//*//{*/
void Detected_UAV::initialize(
                  const uav_detect::Detection& det,
                  int w_used,
                  int h_used,
                  const sensor_msgs::CameraInfo& camera_info,
                  const tf2::Transform& camera2world_tf)
{
  _camera_model.fromCameraInfo(camera_info);
  _w_used = w_used;
  _h_used = h_used;
  _w_camera = camera_info.width;
  _h_camera = camera_info.height;
  _c2w_tf = tf2_to_eigen(camera2world_tf);

  Eigen::Vector3d meas_position;
  Eigen::Matrix3d meas_covariance;
  detection_to_position(det, meas_position, meas_covariance);
  // The initial KF state is set directly from the estimated position
  for (int st_it = 0; st_it < 3; st_it++)
    _KF->setState(st_it, meas_position(st_it));
  // The covariance of the speed is unknown - position covariance is calculated standardly as for measurement
  Eigen::Matrix<double, 6, 6> tot_covariance = 2.0*Eigen::Matrix<double, 6, 6>::Identity();
  tot_covariance.block<3, 3>(0, 0) = meas_covariance;
  _KF->setCovariance(tot_covariance);

  cout << "Initial state: " << get_x() << ", " << get_y() << ", " << get_z() << std::endl;
}/*//}*/

/* Detected_UAV::similar_to - returns true if the candidate is similar to this detected UAV *//*//{*/
bool Detected_UAV::similar_to(const Detected_UAV &candidate)
{
  // fuck the IoU method, does not make much sense if the MAV moves
  /* return IoU(get_reference_detection(), candidate.get_reference_detection()) > _similarity_threshold; */
  Vector3d mean_1 = _KF->getStates().block<3, 1>(0, 0);
  Matrix3d cov_1 = _KF->getCovariance().block<3, 3>(0, 0);
  Vector3d mean_2 = candidate._KF->getStates().block<3, 1>(0, 0);
  Matrix3d cov_2 = candidate._KF->getCovariance().block<3, 3>(0, 0);

  double likelihood_1 = mgauss(mean_1, mean_2, cov_2);  // likelihood that mean_1 is from N_2
  double likelihood_2 = mgauss(mean_2, mean_1, cov_1);  // likelihood that mean_2 is from N_1
  double likelihood = likelihood_1*likelihood_2;        // total likelihood

  /* double MSE = (x_1 - x_2).norm(); */
  /* cout << "MSE: " << MSE << std::endl; */
  cout << "likelihood: " << likelihood << std::endl;
  return likelihood > _similarity_threshold;
}/*//}*/

/* Detected_UAV::more_uncertain_than - returns true if this detected UAV is more uncertain that the candidate *//*//{*/
bool Detected_UAV::more_uncertain_than(const Detected_UAV &candidate)
{
  Matrix<double, 6, 6> cov_1 = _KF->getCovariance();
  Matrix<double, 6, 6> cov_2 = candidate._KF->getCovariance();
  Matrix<double, 6, 6> diff = cov_1 - cov_2;
  // now we have to find out if diff is positive definite
  Eigen::LLT<Matrix<double, 6, 6>> lltOfA(diff);
  if (lltOfA.info() == Eigen::NumericalIssue)
    return true;
  else return false;
}/*//}*/

/* Detected_UAV::unreliable - returns true if this UAV detection is unreliable according to specified parameters *//*//{*/
bool Detected_UAV::unreliable()
{
  // only the position covariance is important... we don't care about the speed as much
  Matrix3d pos_cov = _KF->getCovariance().block<3, 3>(0, 0);
  double pos_cov_norm = pos_cov.norm();
  return pos_cov_norm > _unreliable_threshold;
}/*//}*/

/* Detected_UAV::IoU - calculates Intersection over Union of two detections *//*//{*/
double Detected_UAV::IoU(const uav_detect::Detection &det1, const uav_detect::Detection &det2)
{
  float det1_l = det1.x_relative-det1.w_relative/2.0;
  float det1_r = det1.x_relative+det1.w_relative/2.0;
  float det1_b = det1.y_relative-det1.h_relative/2.0;
  float det1_t = det1.y_relative+det1.h_relative/2.0;

  float det2_l = det2.x_relative-det2.w_relative/2.0;
  float det2_r = det2.x_relative+det2.w_relative/2.0;
  float det2_b = det2.y_relative-det2.h_relative/2.0;
  float det2_t = det2.y_relative+det2.h_relative/2.0;

  float width  = min(det1_r, det2_r) - max(det1_l, det2_l);
  float height = min(det1_t, det2_t) - max(det1_b, det2_b);

  if (width < 0.0 || height < 0.0)
    return 0.0;

  // Area of Overlap
  float AoO = width*height;
  // Area of Union
  float AoU = det1.w_relative*det1.h_relative + det2.w_relative*det2.h_relative - AoO;

  return AoO/AoU;
}/*//}*/

/* Detected_UAV::get_reference_detection - generates a bounding box from the current state estimate *//*//{*/
Detection const Detected_UAV::get_reference_detection() const
{
  Detection ret;

  // This will be center point of the detected UAV
  /* Eigen::Vector3d pt3d_center = _est_state.block<3, 1>(0, 0); */
  Eigen::Vector3d pt3d_center = _KF->getStates().block<3, 1>(0, 0);
  pt3d_center = _c2w_tf.inverse()*pt3d_center;
  cv::Point3d cvpt3d_center(pt3d_center(0, 0), pt3d_center(1, 0), pt3d_center(2, 0));
  cv::Point2d cvpt2d_center = _camera_model.project3dToPixel(cvpt3d_center);
  double dist = pt3d_center.norm();
  double width = _UAV_width/dist;

  ret.x_relative = cvpt2d_center.x/double(_w_used);
  ret.y_relative = cvpt2d_center.y/double(_h_used);
  ret.w_relative = width/double(_w_used);
  ret.h_relative = width/2.0/double(_h_used);

  return ret;
}/*//}*/

void Detected_UAV::update()
{
  _KF->iterateWithoutCorrection();
}

/* Detected_UAV::update - processes detections into a KF update *//*//{*/
int Detected_UAV::update(const uav_detect::Detections& new_detections, const tf2::Transform& camera2world_tf)
{
  // update camera-related stuff
  _c2w_tf = tf2_to_eigen(camera2world_tf);
  _camera_model.fromCameraInfo(new_detections.camera_info);
  _w_used = new_detections.w_used;
  _h_used = new_detections.h_used;
  _w_camera = new_detections.camera_info.width;
  _h_camera = new_detections.camera_info.height;

  // update the Kalman Filter (prediction step) TODO: this should be done periodically (another thread?)
  _KF->iterateWithoutCorrection();

  /* Find the best matching detection */
  // prepare variables
  double best_likelihood = 0.0;
  int best_match_it = -1; // -1 indicates no match found
  int it = 0;
  Eigen::Vector3d best_position;
  Eigen::Matrix3d best_covariance;
  // search for a detection with highest likelihood
  for (const uav_detect::Detection det : new_detections.detections)
  {
    Vector3d mean = _KF->getStates().block<3, 1>(0, 0);
    Matrix3d cov = _KF->getCovariance().block<3, 3>(0, 0);

    Eigen::Vector3d meas_position;
    Eigen::Matrix3d meas_covariance;
    detection_to_position(det, meas_position, meas_covariance);

    double cur_likelihood = mgauss(meas_position, mean, cov);
    /* cout << "Cur_likelihood: " << cur_likelihood << std::endl; */
    if (cur_likelihood > _association_threshold && cur_likelihood > best_likelihood)
    {
      best_likelihood = cur_likelihood;
      best_match_it = it;
      best_position = meas_position;
      best_covariance = meas_covariance;
    }
    it++;
  }

  // If a matching detection was found, update the KF with it
  if (best_match_it >= 0)
  {
    // update the Kalman Filter (data step)
    _KF->setMeasurement(best_position, best_covariance);
    _KF->doCorrection();
  }

  return best_match_it;
}/*//}*/
