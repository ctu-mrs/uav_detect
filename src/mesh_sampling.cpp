#include <mesh_sampling.h>

namespace uav_detect
{

  using point_t = pcl::PointXYZ;
  using normal_t = pcl::Normal;
  using point_normal_t = pcl::PointNormal;
  using cloud_t = pcl::PointCloud<point_t>;

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
    triangle_t(
        const point_t& A,
        const point_t& B,
        const point_t& C
        ) :
      A(A.x, A.y, A.z),
      B(B.x, B.y, B.z),
      C(C.x, C.y, C.z) {};
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

  triangle_t get_triangle(const pcl::Vertices& poly, const cloud_t& mesh_cloud)
  {
    assert(poly.vertices.size() == 3);
    const auto A = mesh_cloud.at(poly.vertices.at(0));
    const auto B = mesh_cloud.at(poly.vertices.at(1));
    const auto C = mesh_cloud.at(poly.vertices.at(2));
    triangle_t ret(A, B, C);
    return ret;
  }

  std::vector<float> cumulative_mesh_areas(const std::vector<pcl::Vertices>& mesh_polygons, const cloud_t& mesh_cloud)
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

  point_t random_point_triangle(const triangle_t& tri)
  {
    const float r1 = random(0.0f, 1.0f);
    const float r2 = random(0.0f, 1.0f);
    const float r1sqrt = std::sqrt(r1);
    const float omr1sqrt = 1.0f - r1sqrt;
    const float omr2 = 1.0f - r2;
    const float a1 = tri.A.x() * omr1sqrt;
    const float a2 = tri.A.y() * omr1sqrt;
    const float a3 = tri.A.z() * omr1sqrt;
    const float b1 = tri.B.x() * omr2;
    const float b2 = tri.B.y() * omr2;
    const float b3 = tri.B.z() * omr2;
    const float c1 = r1sqrt * (r2 * tri.C.x() + b1) + a1;
    const float c2 = r1sqrt * (r2 * tri.C.y() + b2) + a2;
    const float c3 = r1sqrt * (r2 * tri.C.z() + b3) + a3;
    point_t ret(c1, c2, c3);
    return ret;
  }

  normal_t triangle_normal(const triangle_t& tri)
  {
    normal_t ret;
    Eigen::Vector3f v1 = tri.A - tri.C;
    Eigen::Vector3f v2 = tri.B - tri.C;
    Eigen::Vector3f n = v1.cross(v2).normalized();
    ret.normal_x = n.x();
    ret.normal_y = n.y();
    ret.normal_z = n.z();
    return ret;
  }

  point_normal_t merge_point_normal(const point_t& pt, const normal_t& normal)
  {
    point_normal_t ret;
    ret.x = pt.x;
    ret.y = pt.y;
    ret.z = pt.z;
    ret.normal_x = normal.normal_x;
    ret.normal_y = normal.normal_y;
    ret.normal_z = normal.normal_z;
    return ret;
  }

  point_normal_t random_point_surface(const std::vector<pcl::Vertices>& mesh_polygons, const cloud_t& mesh_cloud, const std::vector<float>& cumulative_areas)
  {
    const float total_area = cumulative_areas.back();
    const float random_area = random(0.0, total_area);
    const auto& random_triangle_idx = std::lower_bound(std::begin(cumulative_areas), std::end(cumulative_areas), random_area) - std::begin(cumulative_areas);
    const pcl::Vertices random_triangle_poly = mesh_polygons.at(random_triangle_idx);
    const triangle_t& random_triangle = get_triangle(random_triangle_poly, mesh_cloud);
    const point_t random_point = random_point_triangle(random_triangle);
    const normal_t normal = triangle_normal(random_triangle);
    const point_normal_t ret = merge_point_normal(random_point, normal);
    return ret;
  }

  pcl::PointCloud<pcl::PointNormal> uniform_mesh_sampling(const pcl::PolygonMesh& mesh, const size_t n_pts)
  {
    pcl::PointCloud<pcl::PointNormal> ret;
    if (mesh.polygons.empty())
      return ret;
    ret.reserve(n_pts);
    cloud_t mesh_cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, mesh_cloud);
    std::vector<float> cumulative_areas = cumulative_mesh_areas(mesh.polygons, mesh_cloud);
    for (size_t it = 0; it < n_pts; it++)
    {
      pcl::PointNormal pt = random_point_surface(mesh.polygons, mesh_cloud, cumulative_areas);
      ret.push_back(pt);
    }
    ret.header = mesh.header;
    return ret;
  }

}
