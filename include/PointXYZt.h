#include <pcl/impl/point_types.hpp>

// Define all PCL point types
#define TMP PCL_POINT_TYPES
#undef PCL_POINT_TYPES
#define PCL_POINT_TYPES         \
  TMP                           \
  (uav_detect::PointXYZNormalt  \
#undef TMP

// Define all point types that include XYZ data
#define TMP PCL_XYZ_POINT_TYPES
#undef PCL_XYZ_POINT_TYPES
#define PCL_XYZ_POINT_TYPES     \
  TMP                           \
  (uav_detect::PointXYZNormalt  \
#undef TMP

// Define all point types with XYZ and label
#define PCL_XYZL_POINT_TYPES  \
  (pcl::PointXYZL)            \
  (pcl::PointXYZRGBL)         \
  (pcl::PointXYZLNormal)

// Define all point types that include normal[3] data
#define PCL_NORMAL_POINT_TYPES  \
  (pcl::Normal)                 \
  (pcl::PointNormal)            \
  (pcl::PointXYZRGBNormal)      \
  (pcl::PointXYZINormal)        \
  (pcl::PointXYZLNormal)        \
  (pcl::PointSurfel)

// Define all point types that represent features
#define PCL_FEATURE_POINT_TYPES \
  (pcl::PFHSignature125)        \
  (pcl::PFHRGBSignature250)     \
  (pcl::PPFSignature)           \
  (pcl::CPPFSignature)          \
  (pcl::PPFRGBSignature)        \
  (pcl::NormalBasedSignature12) \
  (pcl::FPFHSignature33)        \
  (pcl::VFHSignature308)        \
  (pcl::GASDSignature512)       \
  (pcl::GASDSignature984)       \
  (pcl::GASDSignature7992)      \
  (pcl::GRSDSignature21)        \
  (pcl::ESFSignature640)        \
  (pcl::BRISKSignature512)      \
  (pcl::Narf36)
namespace uav_detect
{
  struct EIGEN_ALIGN16 _PointXYZNormalt
  {
    PCL_ADD_POINT4D;

    PCL_ADD_NORMAL4D;

    union
    {
      struct
      {
        float curvature;
        uint32_t time;
      };
      float data_c[4];
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  /** \brief A point structure representing Euclidean xyz coordinates and time t.
  * \ingroup common
  */
  struct PointXYZNormalt : public _PointXYZNormalt
  {
    inline PointXYZNormalt(const _PointXYZNormalt &p)
    {
      x = p.x; y = p.y; z = p.z;
      data[3] = 0.0f;
      normal_x = p.normal_x; normal_y = p.normal_y; normal_z = p.normal_z;
      data_n[3] = 0.0f;
      curvature = p.curvature;
      time = p.time;
    }

    inline PointXYZNormalt()
    {
      x = y = z = data[3] = 1.0f;
      normal_x = normal_y = normal_z = data_n[3] = 0.0f;
      curvature = 0.0f;
      time = 0u;
    }

    inline PointXYZNormalt(float _x, float _y, float _z, float n_x, float n_y, float n_z, uint32_t _time)
    {
      x = _x; y = _y; z = _z;
      normal_x = n_x; normal_y = n_y; normal_z = n_z;
      data_n[3] = 0.0f;
      curvature = 0.0f;
      time = _time;
    }

    friend std::ostream& operator << (std::ostream& os, const PointXYZNormalt& p);
  };
}
