#include "main.h"

#include <nodelet/nodelet.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/transforms.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/surface/organized_fast_mesh.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/surface/poisson.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <visualization_msgs/Marker.h>
#include <uav_detect/DetectionParamsConfig.h>
#include <mesh_sampling.h>

using namespace cv;
using namespace std;
using namespace uav_detect;

// shortcut type to the dynamic reconfigure manager template instance
typedef mrs_lib::DynamicReconfigureMgr<uav_detect::DetectionParamsConfig> drmgr_t;
typedef pcl::PointCloud<pcl::PointXYZ> PC;

namespace uav_detect
{

  /* helper functions //{ */
  
  float distsq_from_origin(const pcl::PointXYZ& point)
  {
    return point.x*point.x + point.y*point.y + point.z*point.z;
  }
  
  bool scaled_dist_thresholding(const pcl::PointXYZ& point_a, const pcl::PointXYZ& point_b, float squared_distance)
  {
    static const float thresh = 0.25*0.25; // metres squared
    const float d_a = distsq_from_origin(point_a);
    const float d_b = distsq_from_origin(point_b);
    const float d_max = std::max(d_a, d_b);
    const float scaled_thresh = thresh*sqrt(d_max);
    return squared_distance < scaled_thresh;
  }
  
  //}

  class PCLDetector : public nodelet::Nodelet
  {
  public:

    /* onInit() method //{ */
    void onInit()
    {
      ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

      m_node_name = "PCLDetector";

      /* Load parameters from ROS //{*/
      NODELET_INFO("Loading default dynamic parameters:");
      m_drmgr_ptr = make_unique<drmgr_t>(nh, m_node_name);

      mrs_lib::ParamLoader pl(nh, m_node_name);
      // LOAD STATIC PARAMETERS
      NODELET_INFO("Loading static parameters:");
      pl.load_param("world_frame", m_world_frame);
      pl.load_param("filtering_leaf_size", m_drmgr_ptr->config.filtering_leaf_size);
      pl.load_param("active_box_size", m_drmgr_ptr->config.active_box_size);
      pl.load_param("exclude_box/offset/x", m_exclude_box_offset_x);
      pl.load_param("exclude_box/offset/y", m_exclude_box_offset_y);
      pl.load_param("exclude_box/offset/z", m_exclude_box_offset_z);
      pl.load_param("exclude_box/size/x", m_exclude_box_size_x);
      pl.load_param("exclude_box/size/y", m_exclude_box_size_y);
      pl.load_param("exclude_box/size/z", m_exclude_box_size_z);
      pl.load_param("keep_pc_organized", m_keep_pc_organized, false);

      // LOAD DYNAMIC PARAMETERS
      // CHECK LOADING STATUS
      if (!pl.loaded_successfully())
      {
        NODELET_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
        ros::shutdown();
      }
      //}

      /* Create publishers and subscribers //{ */
      // Initialize transform listener
      m_tf_listener_ptr = std::make_unique<tf2_ros::TransformListener>(m_tf_buffer);
      // Initialize subscribers
      mrs_lib::SubscribeMgr smgr(nh, m_node_name);
      const bool subs_time_consistent = false;
      m_pc_sh = smgr.create_handler_threadsafe<PC::ConstPtr, subs_time_consistent>("pc", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
      // Initialize publishers
      /* m_detections_pub = nh.advertise<uav_detect::Detections>("detections", 10); */ 
      /* m_detected_blobs_pub = nh.advertise<uav_detect::BlobDetections>("blob_detections", 1); */
      m_global_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("global_pc", 1);
      m_filtered_input_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("filterd_input_pc", 1);
      m_mesh_pub = nh.advertise<visualization_msgs::Marker>("mesh", 1);
      m_global_mesh_pub = nh.advertise<visualization_msgs::Marker>("global_mesh", 1);
      //}

      m_last_detection_id = 0;

      m_det_blobs = 0;
      m_images_processed = 0;
      m_avg_fps = 0.0f;
      m_avg_delay = 0.0f;

      /* m_detector = dbd::PCLBlobDetector(m_drmgr_ptr->config, m_unknown_pixel_value); */

      m_main_loop_timer = nh.createTimer(ros::Rate(1000), &PCLDetector::main_loop, this);
      /* m_info_loop_timer = nh.createTimer(ros::Rate(1), &PCLDetector::info_loop, this); */

      m_cloud_global = boost::make_shared<pcl::PointCloud<pcl::PointNormal>>();
      m_cloud_global->header.frame_id = m_world_frame;

      cout << "----------------------------------------------------------" << std::endl;

    }
    //}

  private:
    pcl::PointCloud<pcl::PointNormal>::Ptr m_cloud_global;

    /* main_loop() method //{ */
    void main_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      if (m_pc_sh->new_data())
      {
        /* ros::Time start_t = ros::Time::now(); */

        NODELET_INFO_STREAM("[PCLDetector]: Processing new data --------------------------------------------------------- ");

        PC::ConstPtr cloud = m_pc_sh->get_data();
        ros::Time msg_stamp;
        pcl_conversions::fromPCL(cloud->header.stamp, msg_stamp);
        NODELET_INFO_STREAM("[PCLDetector]: Input PC has " << cloud->size() << " points");
        const auto leaf_size = m_drmgr_ptr->config.filtering_leaf_size;

        /* filter input cloud and transform it to world //{ */
        
        PC::Ptr cloud_filtered = boost::make_shared<PC>(*cloud);
        Eigen::Vector3d tf_trans;
        {
          /* filter by cropping points outside a box, relative to the sensor //{ */
          {
            const auto box_size = m_drmgr_ptr->config.active_box_size;
            const Eigen::Vector4f box_point1(box_size/2, box_size/2, box_size/2, 1);
            const Eigen::Vector4f box_point2(-box_size/2, -box_size/2, -box_size/2, 1);
            pcl::CropBox<pcl::PointXYZ> cb;
            cb.setMax(box_point1);
            cb.setMin(box_point2);
            cb.setInputCloud(cloud_filtered);
            cb.setNegative(false);
            cb.setKeepOrganized(m_keep_pc_organized);
            cb.filter(*cloud_filtered);
            /* cb.setNegative(false); */
            /* cb.filter(indices_filtered->indices); */
          }
          //}
          NODELET_INFO_STREAM("[PCLDetector]: Input PC after CropBox 1: " << cloud_filtered->size() << " points");
          
          /* filter by cropping points inside a box, relative to the sensor //{ */
          {
            const Eigen::Vector4f box_point1(
                m_exclude_box_offset_x + m_exclude_box_size_x/2,
                m_exclude_box_offset_y + m_exclude_box_size_y/2,
                m_exclude_box_offset_z + m_exclude_box_size_z/2,
                1);
            const Eigen::Vector4f box_point2(
                m_exclude_box_offset_x - m_exclude_box_size_x/2,
                m_exclude_box_offset_y - m_exclude_box_size_y/2,
                m_exclude_box_offset_z - m_exclude_box_size_z/2,
                1);
            pcl::CropBox<pcl::PointXYZ> cb;
            cb.setMax(box_point1);
            cb.setMin(box_point2);
            cb.setInputCloud(cloud_filtered);
            cb.setNegative(true);
            cb.setKeepOrganized(m_keep_pc_organized);
            cb.filter(*cloud_filtered);
            /* cb.setInputCloud(cloud); */
            /* cb.setIndices(indices_filtered); */
            /* cb.setNegative(true); */
            /* cb.filter(indices_filtered->indices); */
          }
          //}
          NODELET_INFO_STREAM("[PCLDetector]: Input PC after CropBox 2: " << cloud_filtered->size() << " points");

          Eigen::Affine3d s2w_tf;
          bool tf_ok = get_transform_to_world(cloud->header.frame_id, msg_stamp, s2w_tf);
          if (!tf_ok)
          {
            NODELET_ERROR("[PCLDetector]: Could not transform pointcloud to global, skipping.");
            return;
          }
          tf_trans = s2w_tf.translation();
          pcl::transformPointCloud(*cloud_filtered, *cloud_filtered, s2w_tf.cast<float>());
          cloud_filtered->header.frame_id = m_world_frame;

          NODELET_INFO_STREAM("[PCLDetector]: Filtered input PC has " << cloud_filtered->size() << " points");
        }
        
        //}

        pcl::PolygonMesh mesh;
        /* fit a surface to the filtered (still organized) cloud //{ */
        {
          mesh = reconstruct_mesh_organized(cloud_filtered);
          if (mesh.polygons.empty())
            ROS_ERROR("[PCLDetector]: Failed to reconstruct mesh using input pointcloud - is it organized?");
        }
        //}

        pcl::PointCloud<pcl::PointNormal> cloud_with_normals = uniform_mesh_sampling(mesh, m_drmgr_ptr->config.mesh_resample_points);

        /* add filtered input cloud to global cloud and filter it //{ */
        
        {
          *m_cloud_global += cloud_with_normals;

          /* filter by mutual point distance (voxel grid) //{ */
          pcl::VoxelGrid<pcl::PointNormal> vg;
          vg.setLeafSize(leaf_size, leaf_size, leaf_size);
          vg.setInputCloud(m_cloud_global);
          vg.filter(*m_cloud_global);
          //}

          /* filter by cropping points outside a box, relative to the sensor //{ */
          const auto box_size = m_drmgr_ptr->config.active_box_size;
          const Eigen::Vector4f sensor_origin(tf_trans.x(), tf_trans.y(), tf_trans.z(), 1.0f);
          const Eigen::Vector4f box_point1 = sensor_origin - Eigen::Vector4f(box_size/2, box_size/2, box_size/2, 0);
          const Eigen::Vector4f box_point2 = sensor_origin + Eigen::Vector4f(box_size/2, box_size/2, box_size/2, 0);
          pcl::CropBox<pcl::PointNormal> cb;
          cb.setMin(box_point1);
          cb.setMax(box_point2);
          cb.setInputCloud(m_cloud_global);
          cb.filter(*m_cloud_global);
          //}

          NODELET_INFO_STREAM("[PCLDetector]: Global pointcloud has " << m_cloud_global->size() << " points");
        }
        
        //}

        pcl::PolygonMesh global_mesh;
        /* fit a surface to the filtered cloud //{ */
        {
          pcl::Poisson<pcl::PointNormal> poisson;
          poisson.setInputCloud(m_cloud_global);
          poisson.reconstruct(global_mesh);
          NODELET_INFO_STREAM("[PCLDetector]: Global mesh has " << global_mesh.polygons.size() << " polygons");
        }
        //}

        /* filter the fitted mesh surface by raytracing the input pointcloud //{ */
        {
          filter_mesh_raytrace(global_mesh, *cloud);
          NODELET_INFO_STREAM("[PCLDetector]: Filtered global mesh has " << global_mesh.polygons.size() << " polygons");
        }
        //}

        *m_cloud_global = uniform_mesh_sampling(global_mesh, m_drmgr_ptr->config.global_mesh_resample_points);
        NODELET_INFO_STREAM("[PCLDetector]: Resampled global pointcloud has " << m_cloud_global->size() << " points");

        m_cloud_global->header.stamp = cloud->header.stamp;
        m_cloud_global->header.frame_id = m_world_frame;

        if (m_filtered_input_pc_pub.getNumSubscribers() > 0)
          m_filtered_input_pc_pub.publish(cloud_filtered);

        if (m_global_pc_pub.getNumSubscribers() > 0)
          m_global_pc_pub.publish(m_cloud_global);

        if (m_mesh_pub.getNumSubscribers() > 0)
          m_mesh_pub.publish(to_marker_list_msg(mesh));

        if (m_global_mesh_pub.getNumSubscribers() > 0)
          m_global_mesh_pub.publish(to_marker_list_msg(global_mesh));

        NODELET_INFO_STREAM("[PCLDetector]: Done processing data --------------------------------------------------------- ");
  }
  }
    //}

    /* /1* info_loop() method //{ *1/ */
    /* void info_loop([[maybe_unused]] const ros::TimerEvent& evt) */
    /* { */
    /*   const float dt = (evt.current_real - evt.last_real).toSec(); */
    /*   std::lock_guard<std::mutex> lck(m_stat_mtx); */
    /*   const float blobs_per_image = m_det_blobs/float(m_images_processed); */
    /*   const float input_fps = m_images_processed/dt; */
    /*   NODELET_INFO_STREAM("[" << m_node_name << "]: det. blobs/image: " << blobs_per_image << " | inp. FPS: " << round(input_fps) << " | proc. FPS: " << round(m_avg_fps) << " | delay: " << round(1000.0f*m_avg_delay) << "ms"); */
    /*   m_det_blobs = 0; */
    /*   m_images_processed = 0; */
    /* } */
    /* //} */

  private:

    /* ray_triangle_intersect() method //{ */
    // implemented according to https://www.scratchapixel.com/code.php?id=11&origin=/lessons/3d-basic-rendering/ray-tracing-polygon-mesh
    bool ray_triangle_intersect(const Eigen::Vector3f& vec, const pcl::Vertices& poly, const PC& mesh_cloud, const float tol = 0.1)
    {
      const static float eps = 1e-9;
      assert(poly.size() == 3);
      const auto pclA = mesh_cloud.at(poly.vertices.at(0));
      const auto pclB = mesh_cloud.at(poly.vertices.at(1));
      const auto pclC = mesh_cloud.at(poly.vertices.at(2));
      const Eigen::Vector3f A(pclA.x, pclA.y, pclA.z);
      const Eigen::Vector3f B(pclB.x, pclB.y, pclB.z);
      const Eigen::Vector3f C(pclC.x, pclC.y, pclC.z);
      const Eigen::Vector3f v0 = A - B;
      const Eigen::Vector3f v1 = B - C;
      const Eigen::Vector3f v2 = C - A;
      const Eigen::Vector3f v0v1 = v1 - v0;
      const Eigen::Vector3f v0v2 = v2 - v0;
      const float dist = vec.norm();
      const Eigen::Vector3f dir = vec/dist;
      const Eigen::Vector3f pvec = dir.cross(v0v2);
      const float det = v0v1.dot(pvec);
    
      // ray and triangle are parallel if det is close to 0
      if (std::abs(det) < eps)
        return false;
    
      const float inv_det = 1.0f / det;
    
      const Eigen::Vector3f tvec = v0;
      const float u = tvec.dot(pvec) * inv_det;
      if (u < 0 || u > 1)
        return false;
    
      const Eigen::Vector3f qvec = tvec.cross(v0v1);
      const float v = dir.dot(qvec) * inv_det;
      if (v < 0 || u + v > 1)
        return false;
    
      const float int_dist = v0v2.dot(qvec) * inv_det; 
      if (int_dist + tol < dist)
        return false;
    
      return true;
    }
    //}

    /* filter_mesh_raytrace() method //{ */
    void filter_mesh_raytrace(pcl::PolygonMesh& mesh, const PC& cloud)
    {
      PC mesh_cloud;
      pcl::fromPCLPointCloud2(mesh.cloud, mesh_cloud);
      const float intersection_tolerance = m_drmgr_ptr->config.intersection_tolerance;
      for (const auto& point : cloud)
      {
        Eigen::Vector3f vec(point.x, point.y, point.z);
        if (!vec.array().isFinite().all())
          continue;
        for (auto it = std::cbegin(mesh.polygons); it != std::cend(mesh.polygons); ++it)
        {
          const auto& poly = *it;
          if (ray_triangle_intersect(vec, poly, mesh_cloud, intersection_tolerance))
            it = mesh.polygons.erase(it);
        }
      }
    }
    //}

    /* to_marker_list_msg() method and helpers//{ */
    geometry_msgs::Point pcl2gmpt(const pcl::PointXYZ& pt0)
    {
      geometry_msgs::Point ret;
      ret.x = pt0.x;
      ret.y = pt0.y;
      ret.z = pt0.z;
      return ret;
    }

    using marker_pts_t = visualization_msgs::Marker::_points_type;
    void fill_marker_pts_lines(const pcl::Vertices& mesh_verts, const PC& mesh_cloud, marker_pts_t& marker_pts)
    {
      geometry_msgs::Point prev_pt;
      bool prev_pt_set = false;
      for (const auto vert : mesh_verts.vertices)
      {
        const auto idx = vert;
        const auto pt = mesh_cloud.at(idx);
        const geometry_msgs::Point gmpt = pcl2gmpt(pt);
        if (prev_pt_set)
        {
          marker_pts.push_back(prev_pt);
          marker_pts.push_back(gmpt);
        }
        prev_pt = gmpt;
        prev_pt_set = true;
      }
      if (prev_pt_set)
      {
        marker_pts.push_back(prev_pt);
        const auto idx = mesh_verts.vertices.at(0);
        const auto pt = mesh_cloud.at(idx);
        const geometry_msgs::Point gmpt = pcl2gmpt(pt);
        marker_pts.push_back(gmpt);
      }
    }

    void fill_marker_pts_triangles(const pcl::Vertices& mesh_verts, const PC& mesh_cloud, marker_pts_t& marker_pts)
    {
      if (mesh_verts.vertices.size() != 3)
        return;
      for (const auto vert : mesh_verts.vertices)
      {
        const auto idx = vert;
        const auto pt = mesh_cloud.at(idx);
        const geometry_msgs::Point gmpt = pcl2gmpt(pt);
        marker_pts.push_back(gmpt);
      }
    }

    visualization_msgs::Marker to_marker_list_msg(const pcl::PolygonMesh& mesh)
    {
      visualization_msgs::Marker ret;
      ret.header.frame_id = mesh.header.frame_id;
      pcl_conversions::fromPCL(mesh.header.stamp, ret.header.stamp);
      ret.ns = "uav_detect/mesh";
      ret.id = 666;
      ret.action = visualization_msgs::Marker::ADD;
      ret.lifetime = ros::Duration(0.0);
      ret.color.a = ret.color.r = ret.color.g = ret.color.b = 1.0;
      if (mesh.polygons.empty())
        return ret;
    
      const auto n_verts = mesh.polygons.at(0).vertices.size();
      if (n_verts == 3)
      {
        ret.scale.x = ret.scale.y = ret.scale.z = 1.0;
        ret.type = visualization_msgs::Marker::TRIANGLE_LIST;
      }
      else
      {
        ret.scale.x = ret.scale.y = ret.scale.z = 0.1;
        ret.type = visualization_msgs::Marker::LINE_LIST;
      }
      ret.points.reserve(mesh.polygons.size()*n_verts);
      PC mesh_cloud;
      pcl::fromPCLPointCloud2(mesh.cloud, mesh_cloud);
      for (const auto& vert : mesh.polygons)
      {
        if (n_verts == 3)
        {
          if (vert.vertices.size() != n_verts)
            ROS_WARN_THROTTLE(0.1, "[PCLDetector]: Number of vertices in mesh is incosistent (expected: %lu, got %lu)!", n_verts, vert.vertices.size());
          fill_marker_pts_triangles(vert, mesh_cloud, ret.points);
        }
        else
          fill_marker_pts_lines(vert, mesh_cloud, ret.points);
      }
      /* ret.colors; */
      return ret;
    }
    //}

    /* reconstruct_mesh_organized() method //{ */

    pcl::PolygonMesh reconstruct_mesh_organized(PC::Ptr& pc)
    {
      using ofm_t = pcl::OrganizedFastMesh<PC::PointType>;
      pcl::PolygonMesh ret;
      ofm_t ofm;
      ofm.setInputCloud(pc);
      ofm.setTriangulationType(ofm_t::TriangulationType::TRIANGLE_ADAPTIVE_CUT);
      const bool use_shadowed_faces = m_drmgr_ptr->config.orgmesh_use_shadowed;
      const float shadow_angle_tolerance = use_shadowed_faces ? -1.0f : (m_drmgr_ptr->config.orgmesh_shadow_ang_tol/180.0f*M_PI);
      ofm.storeShadowedFaces(use_shadowed_faces);
      ofm.setAngleTolerance(shadow_angle_tolerance);
      ofm.reconstruct(ret);
      return ret;
    }

    //}

    /* estimate_normals_organized() method //{ */
    // THESE VALUES HAVE TO CORRESPOND TO THE DYNAMIC RECONFIGURE ENUM
    enum plane_fit_method_t
    {
      RANSAC = 0,
      SVD = 1,
    };

    pcl::PointCloud<pcl::Normal> estimate_normals_organized(PC& pc, const PC& unfiltered_pc, const bool debugging = false)
    {
      pcl::PointCloud<pcl::Normal> normals;
      const auto neighborhood_rows = m_drmgr_ptr->config.normal_neighborhood_rows;
      const auto neighborhood_cols = m_drmgr_ptr->config.normal_neighborhood_cols;
      const plane_fit_method_t fitting_method = (plane_fit_method_t)m_drmgr_ptr->config.normal_method;
    
      if (debugging)
      {
        const auto i = std::clamp(m_drmgr_ptr->config.normal_debug_col, 0, int(pc.width)-1);
        const auto j = std::clamp(m_drmgr_ptr->config.normal_debug_row, 0, int(pc.height)-1);
        const pcl::Normal n = estimate_normal(i, j, pc, unfiltered_pc, neighborhood_rows, neighborhood_cols, fitting_method, debugging);
        for (unsigned it = 0; it < pc.size(); it++)
          normals.push_back(n);
        normals.width = pc.width;
        normals.height = pc.height;
      } else
      {
        normals.resize(pc.size());
        normals.width = pc.width;
        normals.height = pc.height;
        for (unsigned i = 0; i < pc.width; i++)
          for (unsigned j = 0; j < pc.height; j++)
          {
            const pcl::Normal n = estimate_normal(i, j, pc, unfiltered_pc, neighborhood_rows, neighborhood_cols, fitting_method, debugging);
            normals.at(i, j) = n;
          }
      }
    
      normals.is_dense = pc.is_dense;
      normals.header = pc.header;
      return normals;
    }

    template <class Point_T>
    bool valid_pt(Point_T pt)
    {
      return (std::isfinite(pt.x) &&
              std::isfinite(pt.y) &&
              std::isfinite(pt.z));
    }

    using plane_params_t = Eigen::Vector4f;
    constexpr static float nan = std::numeric_limits<float>::quiet_NaN();
    plane_params_t fit_plane(PC::ConstPtr pcl)
    {
      const static plane_params_t invalid_plane_params = plane_params_t(nan, nan, nan, nan);
      if (pcl->size() < 3)
        return invalid_plane_params;

      Eigen::Matrix<float, 3, -1> points = pcl->getMatrixXfMap(3, 4, 0);
      /* cout << "Fitting plane to points:" << std::endl << points << std::endl; */
      const Eigen::Vector3f centroid = points.rowwise().mean();
      points.colwise() -= centroid;
      const auto svd = points.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
      const Eigen::Vector3f normal = svd.matrixU().rightCols<1>().normalized();
      const double d = normal.dot(centroid);
      const plane_params_t ret(normal.x(), normal.y(), normal.z(), d);
      return ret;
    }

    plane_params_t fit_plane_RANSAC(PC::ConstPtr pcl)
    {
      const static plane_params_t invalid_plane_params = plane_params_t(nan, nan, nan, nan);
      pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p
        = boost::make_shared<pcl::SampleConsensusModelPlane<pcl::PointXYZ>>(pcl, true);
      if (pcl->size() < model_p->getSampleSize())
        return invalid_plane_params;

      pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
      ransac.setDistanceThreshold(m_drmgr_ptr->config.normal_threshold);
      ransac.setMaxIterations(m_drmgr_ptr->config.normal_iterations);
      ransac.setProbability(m_drmgr_ptr->config.normal_probability);
      if (ransac.computeModel())
      {
        Eigen::VectorXf coeffs;
        ransac.getModelCoefficients(coeffs);
        // normalize the normal (lel)
        const double c = coeffs.block<3, 1>(0, 0).norm();
        coeffs = coeffs/c;
        return coeffs;
      } else
      {
        return invalid_plane_params;
      }
    }

    pcl::Normal estimate_normal(const int col, const int row, PC& pc, const PC& unfiltered_pc, const int neighborhood_rows, const int neighborhood_cols, const plane_fit_method_t method, const bool debugging = false)
    {
      /* cout << "Using neighborhood: " << neighborhood << std::endl; */
      const static pcl::Normal invalid_normal(nan, nan, nan);
      const auto pt = pc.at(col, row);
      if (!valid_pt(pt) && !debugging)
        return invalid_normal;

      const int col_bot = std::max(col - neighborhood_cols, 0);
      const int col_top = std::min(col + neighborhood_cols, (int)pc.width-1);
      const int row_bot = std::max(row - neighborhood_rows, 0);
      const int row_top = std::min(row + neighborhood_rows, (int)pc.height-1);
      /* cout << "bounds: [" << col_bot << ", " << col_top << "]; [" << row_bot << ", " << row_top << "]" << std::endl; */
      PC::Ptr neig_pc = boost::make_shared<PC>();
      neig_pc->reserve((2*neighborhood_cols+1)*(2*neighborhood_rows+1));
      for (int i = col_bot; i <= col_top; i++)
        for (int j = row_bot; j <= row_top; j++)
        {
          const auto pt = unfiltered_pc.at(i, j);
          if (valid_pt(pt))
            neig_pc->push_back(pt);
        }

      if (debugging)
        pc = *neig_pc;
      /* cout << "Neighborhood points size: " << neig_pc->size() << std::endl; */
      plane_params_t plane_params;
      switch (method)
      {
        case plane_fit_method_t::RANSAC:
          plane_params = fit_plane_RANSAC(neig_pc);
          break;
        case plane_fit_method_t::SVD:
          plane_params = fit_plane(neig_pc);
          break;
        default:
          ROS_ERROR("[PCLDetector]: Unknown plane fitting method: %d! Skipping.", method);
          plane_params = plane_params_t(nan, nan, nan, nan);
          break;
      }
      Eigen::Vector3f normal_vec = plane_params.block<3, 1>(0, 0);
      if (!normal_vec.array().isFinite().all())
        return invalid_normal;

      const Eigen::Vector3f camera_vec = -Eigen::Vector3f(pt.x, pt.y, pt.z).normalized();
      /* const Eigen::Vector3f camera_vec = -Eigen::Vector3f(0, 0, 1).normalized(); */
      const auto ancos = normal_vec.dot(camera_vec);
      if (ancos < 0.0)
      {
        /* cout << "do flipping normal " << coeffs << " to correspond to " << camera_vec << " (dot product is " << dprod << ")" << std::endl; */
        normal_vec = -normal_vec;
      }
      /* else */
      /* { */
      /*   cout << "not flipping normal " << coeffs << " to correspond to " << camera_vec << " (dot product is " << dprod << ")" << std::endl; */
      /* } */
      const pcl::Normal ret(normal_vec(0), normal_vec(1), normal_vec(2));
      /* cout << "Camera: [" << std::endl << camera_vec.transpose() << std::endl << "]" << std::endl; */
      /* cout << "Normal: [" << std::endl << normal_vec.transpose() << std::endl << "], acos: " << ancos << std::endl; */
      return ret;
    }

    //}

    /* get_transform_to_world() method //{ */
    bool get_transform_to_world(const string& frame_id, ros::Time stamp, Eigen::Affine3d& tf_out) const
    {
      try
      {
        const ros::Duration timeout(1.0 / 100.0);
        // Obtain transform from sensor into world frame
        geometry_msgs::TransformStamped transform;
        transform = m_tf_buffer.lookupTransform(m_world_frame, frame_id, stamp, timeout);
        tf_out = tf2::transformToEigen(transform.transform);
      }
      catch (tf2::TransformException& ex)
      {
        NODELET_WARN_THROTTLE(1.0, "[%s]: Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", m_node_name.c_str(), frame_id.c_str(),
                          m_world_frame.c_str(), ex.what());
        return false;
      }
      return true;
    }
    //}

  private:

    // --------------------------------------------------------------
    // |                ROS-related member variables                |
    // --------------------------------------------------------------

    /* ROS related variables (subscribers, timers etc.) //{ */
    std::unique_ptr<drmgr_t> m_drmgr_ptr;
    tf2_ros::Buffer m_tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> m_tf_listener_ptr;
    mrs_lib::SubscribeHandlerPtr<PC::ConstPtr> m_pc_sh;
    /* ros::Publisher m_detections_pub; */
    ros::Publisher m_global_pc_pub;
    ros::Publisher m_mesh_pub;
    ros::Publisher m_global_mesh_pub;
    ros::Publisher m_filtered_input_pc_pub;
    ros::Timer m_main_loop_timer;
    ros::Timer m_info_loop_timer;
    std::string m_node_name;
    //}

  private:

    // --------------------------------------------------------------
    // |                 Parameters, loaded from ROS                |
    // --------------------------------------------------------------

    /* Parameters, loaded from ROS //{ */
    
    std::string m_world_frame;
    double m_exclude_box_offset_x;
    double m_exclude_box_offset_y;
    double m_exclude_box_offset_z;
    double m_exclude_box_size_x;
    double m_exclude_box_size_y;
    double m_exclude_box_size_z;
    bool m_keep_pc_organized;
    
    //}

  private:

    // --------------------------------------------------------------
    // |                   Other member variables                   |
    // --------------------------------------------------------------

    uint32_t m_last_detection_id;
    
    /* Statistics variables //{ */
    std::mutex m_stat_mtx;
    unsigned   m_det_blobs;
    unsigned   m_images_processed;
    float      m_avg_fps;
    float      m_avg_delay;
    //}

  }; // class PCLDetector
}; // namespace uav_detect

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(uav_detect::PCLDetector, nodelet::Nodelet)
