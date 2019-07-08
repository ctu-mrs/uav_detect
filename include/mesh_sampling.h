#ifndef MESH_SAMPLING_H
#define MESH_SAMPLING_H

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

#include <random>
#include <vector>

void uniform_mesh_sampling(const pcl::PolygonMesh& mesh, const size_t n_pts, pcl::PointCloud<pcl::PointNormal>& cloud_out);

#endif // MESH_SAMPLING_H
