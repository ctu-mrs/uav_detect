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
}/*//}*/

