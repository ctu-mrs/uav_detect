#include <mesh_sampling.h>

struct triangle_t
{
  Eigen::Vector3f A;
  Eigen::Vector3f B;
  Eigen::Vector3f C;
  triangle_t(
      const Eigen::Vector3f& A,
      const Eigen::Vector3f& B,
      const Eigen::Vector3f& C
      ) :
    A(A), B(B), C(C) {};
};

double triangle_area(const triangle_t& tri)
{
  const float a = (tri.C - tri.B).norm();
  const float b = (tri.A - tri.C).norm();
  const float c = (tri.A - tri.B).norm();
  const float s = (a + b + c)/2;
  const float area = std::sqrt(s*(s-a)*(s-b)*(s-c));
  return area;
}

triangle_t get_triangle(const pcl::Vertices& poly, const pcl::PointCloud<pcl::PointXYZ>& mesh_cloud)
{
  assert(poly.vertices.size() == 3);
  const auto A = mesh_cloud.at(poly.vertices.at(0));
  const auto B = mesh_cloud.at(poly.vertices.at(1));
  const auto C = mesh_cloud.at(poly.vertices.at(2));
  triangle_t ret(A, B, C);
  return ret;
}

std::vector<float> cumulative_mesh_areas(const std::vector<pcl::Vertices>& mesh_polygons, const pcl::PointCloud<pcl::PointXYZ>& mesh_cloud)
{
  float total_area = 0.0;
  std::vector<float> cumulative_areas;
  cumulative_areas.reserve(mesh_polygons.size());
  for (const auto& poly : mesh_polygons)
  {
    const double area = triangle_area(get_triangle(poly, mesh_cloud));
    total_area += area;
    cumulative_areas.push_back(total_area);
  }
  return cumulative_areas;
}

float random(const float min, const float max)
{
  static std::random_device rd;
  static std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dist(min, max);
  return dist(gen);
}

pcl::Point random_point_triangle(const triangle_t& tri)
{
  const float r1 = random(0.0, 1.0);
  const float r2 = random(0.0, 1.0);
  const float r1sqr = std::sqrt(random_coord1);
  const float omr1sqr = (1 - r1sqr);
  const float omr2 = (1 - r2);
  const float a1 = tri.A.x() * omr1sqr;
  const float a2 = tri.A.y() * omr1sqr;
  const float a3 = tri.A.z() * omr1sqr;
  const float b1 = tri.B.x() * omr2;
  const float b2 = tri.B.y() * omr2;
  const float b3 = tri.B.z() * omr2;
  const float c1  tri.C.x() * r1sqr * (r2 * c1 + b1) + a1;
  const float c2  tri.C.y() * r1sqr * (r2 * c2 + b2) + a2;
  const float c3  tri.C.z() * r1sqr * (r2 * c3 + b3) + a3;
  pcl::Point ret(c1, c2, c3);
  return ret;
}

pcl::PointNormal random_point_surface(const std::vector<pcl::Vertices>& mesh_polygons, const pcl::PointCloud<pcl::PointXYZ>& mesh_cloud, const std::vector<double>& cumulative_areas)
{
  const float total_area = cumulative_areas.back();
  const float random_area = random(0.0, total_area);
  const auto& random_triangle_idx = std::lower_bound(std::begin(cumulative_areas), std::end(cumulative_areas), random_area) - std::begin(cumulative_areas);
  const pcl::Vertices random_triangle_poly = mesh_polygons.at(random_triangle_idx);
  const triangle_t& random_triangle = get_triangle(random_triangle_poly, mesh_cloud);
}

void uniform_mesh_sampling(const pcl::PolygonMesh& mesh, const size_t n_pts, pcl::PointCloud<pcl::PointNormal>& cloud_out)
{
  cloud_out.clear();
  cloud_out.reserve(n_pts);
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr mesh_cloud;
  pcl::fromPCLPointCloud2(mesh.cloud, mesh_cloud);
  std::vector<float> cumulative_areas = cumulative_mesh_areas(mesh.polygons, mesh_cloud);
  for (size_t it = 0; it < n_pts; it++)
  {
    pcl::PointNormal pt = random_point_surface(mesh, cumulative_areas);
    cloud_out.push_back(pt);
  }
}

