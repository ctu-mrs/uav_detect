#ifndef UTILS_H
#define UTILS_H

Eigen::Affine3d tf2_to_eigen(const tf2::Transform& tf2_t);
Eigen::Affine3d tf2_to_eigen(const geometry_msgs::Transform& tf2_t);

#endif // UTILS_H
