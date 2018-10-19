#include "main.h"

/* tf2_to_eigen - helper function to convert tf2::Transform to Eigen::Affine3d *//*//{*/
Eigen::Affine3d tf2_to_eigen(const tf2::Transform& tf2_t)
{
  Eigen::Affine3d eig_t;
  for (int r_it = 0; r_it < 3; r_it++)
    for (int c_it = 0; c_it < 3; c_it++)
      eig_t(r_it, c_it) = tf2_t.getBasis()[r_it][c_it];
  eig_t(0, 3) = tf2_t.getOrigin().getX();
  eig_t(1, 3) = tf2_t.getOrigin().getY();
  eig_t(2, 3) = tf2_t.getOrigin().getZ();
  return eig_t;
}

Eigen::Affine3d tf2_to_eigen(const geometry_msgs::Transform& tf2_t)
{
  Eigen::Affine3d eig_t;
  Eigen::Quaterniond eig_q;
  eig_q.x() = tf2_t.rotation.x;
  eig_q.y() = tf2_t.rotation.y;
  eig_q.z() = tf2_t.rotation.z;
  eig_q.w() = tf2_t.rotation.w;
  eig_t.linearExt() = eig_q.toRotationMatrix();

  eig_t(0, 3) = tf2_t.translation.x;
  eig_t(1, 3) = tf2_t.translation.y;
  eig_t(2, 3) = tf2_t.translation.z;
  return eig_t;
}

/*//}*/

