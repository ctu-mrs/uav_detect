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
#include <pcl/surface/gp3.h>
#include <pcl/surface/concave_hull.h>

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
/* #include <PointXYZt.h> */

using namespace cv;
using namespace std;
using namespace uav_detect;

namespace uav_detect
{
  // shortcut type to the dynamic reconfigure manager template instance
  using drmgr_t = mrs_lib::DynamicReconfigureMgr<uav_detect::DetectionParamsConfig>;
  using pt_XYZNormal_t = pcl::PointNormal;
  using pc_XYZNormal_t = pcl::PointCloud<pt_XYZNormal_t>;
  using pt_XYZNormalt_t = pcl::PointXYZLNormal;
  using pc_XYZNormalt_t = pcl::PointCloud<pt_XYZNormalt_t>;
  using pt_XYZ_t = pcl::PointXYZ;
  using pc_XYZ_t = pcl::PointCloud<pt_XYZ_t>;

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
      m_pc_sh = smgr.create_handler_threadsafe<pc_XYZ_t::ConstPtr, subs_time_consistent>("pc", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
      // Initialize publishers
      /* m_detections_pub = nh.advertise<uav_detect::Detections>("detections", 10); */ 
      /* m_detected_blobs_pub = nh.advertise<uav_detect::BlobDetections>("blob_detections", 1); */
      m_global_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("global_pc", 1);
      m_resampled_global_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("resampled_global_pc", 1);
      m_filtered_input_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_input_pc", 1);
      m_resampled_input_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("resampled_input_pc", 1);
      m_mesh_pub = nh.advertise<visualization_msgs::Marker>("mesh", 1);
      m_global_mesh_pub = nh.advertise<visualization_msgs::Marker>("global_mesh", 1);
      m_clusters_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("clusters_pc", 1);
      //}

      m_last_detection_id = 0;

      m_det_blobs = 0;
      m_images_processed = 0;
      m_avg_fps = 0.0f;
      m_avg_delay = 0.0f;

      /* m_detector = dbd::PCLBlobDetector(m_drmgr_ptr->config, m_unknown_pixel_value); */

      m_main_loop_timer = nh.createTimer(ros::Rate(1000), &PCLDetector::main_loop, this);
      /* m_info_loop_timer = nh.createTimer(ros::Rate(1), &PCLDetector::info_loop, this); */

      m_cloud_global = boost::make_shared<pc_XYZNormal_t>();
      m_cloud_global->header.frame_id = m_world_frame;

      cout << "----------------------------------------------------------" << std::endl;

    }
    //}

  private:
    pc_XYZNormal_t::Ptr m_cloud_global;

    /* main_loop() method //{ */
    void main_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      if (m_pc_sh->new_data())
      {
        /* ros::Time start_t = ros::Time::now(); */

        NODELET_INFO_STREAM("[PCLDetector]: Processing new data --------------------------------------------------------- ");

        pc_XYZ_t::ConstPtr cloud = m_pc_sh->get_data();
        ros::Time msg_stamp;
        pcl_conversions::fromPCL(cloud->header.stamp, msg_stamp);
        NODELET_INFO_STREAM("[PCLDetector]: Input PC has " << cloud->size() << " points");
        const auto leaf_size = m_drmgr_ptr->config.filtering_leaf_size;

        /* filter input cloud and transform it to world //{ */
        
        pc_XYZ_t::Ptr cloud_filtered = boost::make_shared<pc_XYZ_t>(*cloud);
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

        //TODO: filter old points in global pointcloud (change point type to some custom one with stamp)

        /* add filtered input cloud to global cloud and filter it //{ */
        
        {
          /* *m_cloud_global += cloud_add_time(cloud_with_normals, msg_stamp); */
          *m_cloud_global += cloud_with_normals;

          /* filter by mutual point distance (voxel grid) //{ */
          pcl::VoxelGrid<pt_XYZNormal_t> vg;
          vg.setLeafSize(leaf_size, leaf_size, leaf_size);
          vg.setInputCloud(m_cloud_global);
          vg.filter(*m_cloud_global);
          //}

          /* filter by cropping points outside a box, relative to the sensor //{ */
          const auto box_size = m_drmgr_ptr->config.active_box_size;
          const Eigen::Vector4f sensor_origin(tf_trans.x(), tf_trans.y(), tf_trans.z(), 1.0f);
          const Eigen::Vector4f box_point1 = sensor_origin - Eigen::Vector4f(box_size/2, box_size/2, box_size/2, 0);
          const Eigen::Vector4f box_point2 = sensor_origin + Eigen::Vector4f(box_size/2, box_size/2, box_size/2, 0);
          pcl::CropBox<pt_XYZNormal_t> cb;
          cb.setMin(box_point1);
          cb.setMax(box_point2);
          cb.setInputCloud(m_cloud_global);
          cb.filter(*m_cloud_global);
          //}

          /* filter_by_age(*m_cloud_global, uint32_t(m_drmgr_ptr->config.point_decay_time)); */

          NODELET_INFO_STREAM("[PCLDetector]: Global pointcloud has " << m_cloud_global->size() << " points");
        }
        
        //}

        pcl::PolygonMesh global_mesh;
        /* fit a surface to the filtered cloud //{ */
        {
          /* pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>); */
          /* pcl::ConcaveHull<pcl::PointNormal> chull; */
          /* chull.setInputCloud(m_cloud_global); */
          /* chull.setAlpha(1.0); */
          /* chull.reconstruct(global_mesh); */

          pcl::Poisson<pt_XYZNormal_t> mesher;
          /* pcl::GreedyProjectionTriangulation<pt_XYZNormal_t> mesher; */
          mesher.setInputCloud(m_cloud_global);
          /* mesher.setMaximumNearestNeighbors(m_drmgr_ptr->config.meshing_MaximumNearestNeighbors); */
          /* mesher.setMu(m_drmgr_ptr->config.meshing_Mu); */
          /* mesher.setSearchRadius(m_drmgr_ptr->config.meshing_SearchRadius); */
          /* mesher.setMaximumSurfaceAngle(m_drmgr_ptr->config.meshing_MaximumSurfaceAngle); */
          mesher.reconstruct(global_mesh);
          NODELET_INFO_STREAM("[PCLDetector]: Global mesh has " << global_mesh.polygons.size() << " polygons");
        }
        //}

        /* filter the fitted mesh surface by raytracing the input pointcloud //{ */
        {
          filter_mesh_raytrace(global_mesh, *cloud);
          NODELET_INFO_STREAM("[PCLDetector]: Filtered global mesh has " << global_mesh.polygons.size() << " polygons");
        }
        //}

        if (m_global_pc_pub.getNumSubscribers() > 0)
          m_global_pc_pub.publish(m_cloud_global);

        /* pcl::fromPCLPointCloud2(global_mesh.cloud, *m_cloud_global); */
        *m_cloud_global = uniform_mesh_sampling(global_mesh, m_drmgr_ptr->config.global_mesh_resample_points);
        NODELET_INFO_STREAM("[PCLDetector]: Resampled global pointcloud has " << m_cloud_global->size() << " points");

        pcl::PointCloud<pcl::PointXYZL> cloud_clusters = segment_meshes(global_mesh);
        /* /1* extract euclidean clusters //{ *1/ */
        /* { */
        /*   std::vector<pcl::PointIndices> cluster_indices; */
        /*   pcl::EuclideanClusterExtraction<pcl::PointNormal> ec; */
        /*   ec.setClusterTolerance(0.5); */
        /*   ec.setMinClusterSize(1); */
        /*   ec.setMaxClusterSize(25000); */
        /*   ec.setInputCloud(m_cloud_global); */
        /*   ec.extract(cluster_indices); */
        /*   cloud_clusters.reserve(m_cloud_global->size()); */
        /*   int label = 0; */
        /*   for (const auto& idxs : cluster_indices) */
        /*   { */
        /*     for (const auto idx : idxs.indices) */
        /*     { */
        /*       const auto pt_orig = m_cloud_global->at(idx); */
        /*       pcl::PointXYZL pt; */
        /*       pt.x = pt_orig.x; */
        /*       pt.y = pt_orig.y; */
        /*       pt.z = pt_orig.z; */
        /*       pt.label = label; */
        /*       cloud_clusters.push_back(pt); */
        /*     } */
        /*     label++; */
        /*   } */
        /* } */
        /* //} */

        m_cloud_global->header.stamp = cloud->header.stamp;
        m_cloud_global->header.frame_id = m_world_frame;
        cloud_clusters.header = m_cloud_global->header;

        if (m_filtered_input_pc_pub.getNumSubscribers() > 0)
          m_filtered_input_pc_pub.publish(cloud_filtered);

        if (m_resampled_input_pc_pub.getNumSubscribers() > 0)
          m_resampled_input_pc_pub.publish(cloud_with_normals);

        if (m_resampled_global_pc_pub.getNumSubscribers() > 0)
          m_resampled_global_pc_pub.publish(m_cloud_global);

        if (m_mesh_pub.getNumSubscribers() > 0)
          m_mesh_pub.publish(to_marker_list_msg(mesh));

        if (m_global_mesh_pub.getNumSubscribers() > 0)
          m_global_mesh_pub.publish(to_marker_list_msg(global_mesh));

        if (m_clusters_pc_pub.getNumSubscribers() > 0)
          m_clusters_pc_pub.publish(cloud_clusters);

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
    bool ray_triangle_intersect(const Eigen::Vector3f& vec, const pcl::Vertices& poly, const pc_XYZ_t& mesh_cloud, const float tol = 0.1)
    {
      const static float eps = 1e-9;
      assert(poly.vertices.size() == 3);
      const auto pclA = mesh_cloud.at(poly.vertices.at(0));
      const auto pclB = mesh_cloud.at(poly.vertices.at(1));
      const auto pclC = mesh_cloud.at(poly.vertices.at(2));
      const Eigen::Vector3f v0(pclA.x, pclA.y, pclA.z);
      const Eigen::Vector3f v1(pclB.x, pclB.y, pclB.z);
      const Eigen::Vector3f v2(pclC.x, pclC.y, pclC.z);
      /* const Eigen::Vector3f v0 = A - B; */
      /* const Eigen::Vector3f v1 = B - C; */
      /* const Eigen::Vector3f v2 = C - A; */
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
    void filter_mesh_raytrace(pcl::PolygonMesh& mesh, const pc_XYZ_t& cloud)
    {
      pc_XYZ_t mesh_cloud;
      pcl::fromPCLPointCloud2(mesh.cloud, mesh_cloud);
      const float intersection_tolerance = m_drmgr_ptr->config.intersection_tolerance;
      for (const auto& point : cloud)
      {
        const Eigen::Vector3f vec(point.x, point.y, point.z);
        if (!vec.array().isFinite().all())
          continue;
        for (auto it = std::cbegin(mesh.polygons); it != std::cend(mesh.polygons); ++it)
        {
          const auto& poly = *it;
          if (ray_triangle_intersect(vec, poly, mesh_cloud, intersection_tolerance))
          {
            it = mesh.polygons.erase(it);
            it--;
          }
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
    void fill_marker_pts_lines(const pcl::Vertices& mesh_verts, const pc_XYZ_t& mesh_cloud, marker_pts_t& marker_pts)
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

    void fill_marker_pts_triangles(const pcl::Vertices& mesh_verts, const pc_XYZ_t& mesh_cloud, marker_pts_t& marker_pts)
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
      ret.scale.x = ret.scale.y = ret.scale.z = 1.0;
      if (mesh.polygons.empty())
        return ret;
    
      const auto n_verts = mesh.polygons.at(0).vertices.size();
      if (n_verts == 3)
      {
        ret.type = visualization_msgs::Marker::TRIANGLE_LIST;
      }
      else
      {
        ret.scale.x = ret.scale.y = ret.scale.z = 0.1;
        ret.type = visualization_msgs::Marker::LINE_LIST;
      }
      ret.points.reserve(mesh.polygons.size()*n_verts);
      pc_XYZ_t mesh_cloud;
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

    void add_triangle(const unsigned idx0, const unsigned idx1, const unsigned idx2, std::vector<pcl::Vertices>& mesh_polygons)
    {
      pcl::Vertices poly;
      poly.vertices = {idx0, idx1, idx2};
      mesh_polygons.push_back(poly);
    }

    pcl::PolygonMesh reconstruct_mesh_organized(const pc_XYZ_t::Ptr& pc)
    {
      using ofm_t = pcl::OrganizedFastMesh<pc_XYZ_t::PointType>;
      pcl::PolygonMesh mesh;
      ofm_t ofm;
      ofm.setInputCloud(pc);
      ofm.setTriangulationType(ofm_t::TriangulationType::TRIANGLE_ADAPTIVE_CUT);
      const bool use_shadowed_faces = m_drmgr_ptr->config.orgmesh_use_shadowed;
      const float shadow_angle_tolerance = use_shadowed_faces ? -1.0f : (m_drmgr_ptr->config.orgmesh_shadow_ang_tol/180.0f*M_PI);
      ofm.storeShadowedFaces(use_shadowed_faces);
      ofm.setAngleTolerance(shadow_angle_tolerance);
      ofm.reconstruct(mesh);
      
      // | ------------ Add the border cases to the mesh ------------ |
      const auto pc_width = pc->width;
      const auto pc_height = pc->height;
      pc_XYZ_t mesh_cloud;
      auto& mesh_polygons = mesh.polygons;
      pcl::fromPCLPointCloud2(mesh.cloud, mesh_cloud);

      /* stitch the bottom row //{ */
      {
        const int r_it = 0;
        int acc = 0;
        pt_XYZ_t centroid(0.0f, 0.0f, 0.0f);
        for (unsigned c_it = 0; c_it < pc_width; c_it++)
        {
          const auto pt = pc->at(c_it, r_it);
          if (!valid_pt(pt))
            continue;
          centroid.x += pt.x;
          centroid.y += pt.y;
          centroid.z += pt.z;
          acc++;
        }
        if (acc > 0)
        {
          centroid.x /= acc;
          centroid.y /= acc;
          centroid.z /= acc;
        }

        auto idx0 = mesh_cloud.size();
        const auto idxc = idx0;
        mesh_cloud.push_back(centroid);
        idx0++;
        for (unsigned c_it1 = 0; c_it1 < pc_width-1; c_it1++)
        {
          const int c_it2 = c_it1+1;
          const auto pt0 = pc->at(c_it1, r_it);
          const auto pt1 = pc->at(c_it2, r_it);
          if (!valid_pt(pt0)
           || !valid_pt(pt1))
            continue;
          mesh_cloud.push_back(pt0);
          mesh_cloud.push_back(pt1);
          add_triangle(idx0, idx0+1, idxc, mesh_polygons);
          idx0 += 2;
        }
      }
      //}
      
      /* stitch the last and first columns //{ */
      {
        const int c_it1 = pc_width-1;
        const int c_it2 = 0;
        auto idx0 = mesh_cloud.size();
        for (unsigned r_it1 = 0; r_it1 < pc_height-1; r_it1++)
        {
          const int r_it2 = r_it1+1;
          const auto pt0 = pc->at(c_it1, r_it1);
          const auto pt1 = pc->at(c_it2, r_it1);
          const auto pt2 = pc->at(c_it1, r_it2);
          const auto pt3 = pc->at(c_it2, r_it2);
          if (!valid_pt(pt0)
           || !valid_pt(pt1)
           || !valid_pt(pt2)
           || !valid_pt(pt3))
            continue;
          mesh_cloud.push_back(pt0);
          mesh_cloud.push_back(pt1);
          mesh_cloud.push_back(pt2);
          mesh_cloud.push_back(pt3);
          add_triangle(idx0+2, idx0+1, idx0, mesh_polygons);
          add_triangle(idx0+2, idx0+3, idx0+1, mesh_polygons);
          idx0 += 4;
        }
      }
      //}
      pcl::toPCLPointCloud2(mesh_cloud, mesh.cloud);
      return mesh;
    }

    //}

    /* segment_meshes() method //{ */

    /* template <class Point_T> */
    /* std::vector<Point_T> get_poly_pts(const pcl::Vertices& poly, const pcl::PointCloud<Point_T>& mesh_cloud) */
    /* { */
    /*   std::vector<Point_T> ret; */
    /*   ret.reserve(poly.vertices.size()); */
    /*   for (const auto idx : poly.vertices) */
    /*     ret.push_back(mesh_cloud.at(idx)); */
    /*   return ret; */
    /* } */
    using label_t = int;
    label_t get_master_label(const label_t slave, const std::unordered_map<label_t, label_t>& equal_labels)
    {
      label_t master = slave;
      auto master_it = equal_labels.find(slave);
      while (master_it != std::end(equal_labels))
      {
        master = master_it->second;
        master_it = equal_labels.find(master);
      }
      return master;
    }

    pcl::PointCloud<pcl::PointXYZL> segment_meshes(const pcl::PolygonMesh& mesh)
    {
      pcl::PointCloud<pcl::PointXYZL> ret;
      pc_XYZ_t mesh_cloud;
      pcl::fromPCLPointCloud2(mesh.cloud, mesh_cloud);
      const auto& mesh_polygons = mesh.polygons;
      const auto n_pts = mesh_cloud.size();

      ret.reserve(n_pts);
      std::vector<int> labels(n_pts, -1);
      std::unordered_map<int, int> equal_labels;
      int n_labels = 0;
      for (const auto& poly : mesh_polygons)
      {
        int connected_label = -1;
        for (const auto idx : poly.vertices)
        {
          const auto cur_label = labels.at(idx);
          if (cur_label > -1 && connected_label != cur_label) // a label is assigned
          {
            if (connected_label > -1) // a label is already assigned to the polygon
            {
              const label_t master1 = get_master_label(cur_label, equal_labels);
              const label_t master2 = get_master_label(connected_label, equal_labels);
              if (master1 != master2)
                equal_labels.insert({master1, master2});
            } // if (connected_label > -1)
            else
            {
              connected_label = cur_label;
            } // else (connected_label > -1)
          } // if (cur_label > -1) // a label is assigned
        }
        if (connected_label <= -1)
        {
          connected_label = n_labels++;
        }
        for (const auto idx : poly.vertices)
          labels.at(idx) = connected_label;
      }

      std::unordered_map<int, int> equal_labels_filtered;
      for (const auto& keyval : equal_labels)
      {
        const auto slave = keyval.first;
        const auto master = get_master_label(slave, equal_labels);
        equal_labels_filtered.insert({slave, master});
      }

      for (unsigned it = 0; it < labels.size(); it++)
      {
        auto label = labels.at(it);
        if (equal_labels.find(label) != std::end(equal_labels))
          label = equal_labels.at(label);
        const auto pt = mesh_cloud.at(it);
        pcl::PointXYZL ptl;
        ptl.x = pt.x;
        ptl.y = pt.y;
        ptl.z = pt.z;
        ptl.label = label;
        ret.push_back(ptl);
      }

      return ret;
    }

    //}

    /* valid_pt() method //{ */
    template <class Point_T>
    bool valid_pt(Point_T pt)
    {
      return (std::isfinite(pt.x) &&
              std::isfinite(pt.y) &&
              std::isfinite(pt.z));
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
    mrs_lib::SubscribeHandlerPtr<pc_XYZ_t::ConstPtr> m_pc_sh;
    /* ros::Publisher m_detections_pub; */
    ros::Publisher m_global_pc_pub;
    ros::Publisher m_resampled_global_pc_pub;
    ros::Publisher m_mesh_pub;
    ros::Publisher m_global_mesh_pub;
    ros::Publisher m_clusters_pc_pub;
    ros::Publisher m_filtered_input_pc_pub;
    ros::Publisher m_resampled_input_pc_pub;
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
