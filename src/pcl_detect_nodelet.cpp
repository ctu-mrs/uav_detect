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
#include <pcl/filters/extract_indices.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/surface/poisson.h>

#include <pcl/sample_consensus/lmeds.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>

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
  using pt_XYZ_t = pcl::PointXYZ;
  using pc_XYZ_t = pcl::PointCloud<pt_XYZ_t>;
  using pt_XYZt_t = pcl::PointXYZI;
  using pc_XYZt_t = pcl::PointCloud<pt_XYZt_t>;

  using point = boost::geometry::model::d2::point_xy<double>;
  using ring = boost::geometry::model::ring<point>;
  using polygon = boost::geometry::model::polygon<point>;
  using mpolygon = boost::geometry::model::multi_polygon<polygon>;
  using box = boost::geometry::model::box<point>;

  using vec3_t = Eigen::Vector3f;

  /* helper functions //{ */

  float distsq_from_origin(const pcl::PointXYZ& point)
  {
    return point.x * point.x + point.y * point.y + point.z * point.z;
  }

  bool scaled_dist_thresholding(const pcl::PointXYZ& point_a, const pcl::PointXYZ& point_b, float squared_distance)
  {
    static const float thresh = 0.25 * 0.25;  // metres squared
    const float d_a = distsq_from_origin(point_a);
    const float d_b = distsq_from_origin(point_b);
    const float d_max = std::max(d_a, d_b);
    const float scaled_thresh = thresh * sqrt(d_max);
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
      const auto uav_name = pl.load_param2<std::string>("uav_name");
      pl.load_param("world_frame_id", m_world_frame_id);
      pl.load_param("filtering_leaf_size", m_drmgr_ptr->config.filtering_leaf_size);
      pl.load_param("active_box_size", m_drmgr_ptr->config.active_box_size);

      pl.load_param("exclude_box/offset/x", m_exclude_box_offset_x);
      pl.load_param("exclude_box/offset/y", m_exclude_box_offset_y);
      pl.load_param("exclude_box/offset/z", m_exclude_box_offset_z);
      pl.load_param("exclude_box/size/x", m_exclude_box_size_x);
      pl.load_param("exclude_box/size/y", m_exclude_box_size_y);
      pl.load_param("exclude_box/size/z", m_exclude_box_size_z);

      /* load safety area //{ */

      pl.load_param("safety_area/deflation", m_safety_area_deflation);
      pl.load_param("safety_area/height/min", m_safety_area_min_z);
      pl.load_param("safety_area/height/max", m_safety_area_max_z);
      m_safety_area_frame = pl.load_param2<std::string>("safety_area/frame_name");
      m_safety_area_border_points = pl.load_matrix_dynamic2("safety_area/safety_area", -1, 2);
      m_safety_area_init_timer = nh.createTimer(ros::Duration(1.0), &PCLDetector::init_safety_area, this);

      //}

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
      m_pc_sh = smgr.create_handler<pc_XYZ_t, subs_time_consistent>("pc", ros::Duration(5.0));
      // Initialize publishers
      /* m_detections_pub = nh.advertise<uav_detect::Detections>("detections", 10); */
      /* m_detected_blobs_pub = nh.advertise<uav_detect::BlobDetections>("blob_detections", 1); */
      m_pub_filtered_input_pc = nh.advertise<sensor_msgs::PointCloud2>("filtered_input_pc", 1);
      m_pub_map3d = nh.advertise<visualization_msgs::Marker>("map3d", 1);
      m_pub_map3d_bounds = nh.advertise<visualization_msgs::Marker>("map3d_bounds", 1, true);
      m_pub_oparea = nh.advertise<visualization_msgs::Marker>("operation_area", 1, true);
      m_pub_chosen_position = nh.advertise<geometry_msgs::PoseStamped>("passthrough_position", 1, true);
      m_pub_chosen_neighborhood = nh.advertise<sensor_msgs::PointCloud2>("passthrough_neighborhood", 1, true);
      //}

      /* initialize transformer //{ */

      m_transformer = mrs_lib::Transformer(m_node_name, uav_name);

      //}

      m_last_detection_id = 0;

      m_det_blobs = 0;
      m_images_processed = 0;
      m_avg_fps = 0.0f;
      m_avg_delay = 0.0f;

      m_main_loop_timer = nh.createTimer(ros::Rate(1000), &PCLDetector::main_loop, this);

      cout << "----------------------------------------------------------" << std::endl;
    }
    //}

  private:
    /* main_loop() method //{ */
    void main_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      if (!m_safety_area_initialized)
      {
        NODELET_WARN_STREAM_THROTTLE(1.0, "[PCLDetector]: Safety area not initialized, skipping. ");
        return;
      }

      if (m_pc_sh->new_data())
      {
        /* ros::Time start_t = ros::Time::now(); */

        NODELET_INFO_STREAM("[PCLDetector]: Processing new data --------------------------------------------------------- ");

        pc_XYZ_t::ConstPtr cloud = m_pc_sh->get_data();
        ros::Time msg_stamp;
        pcl_conversions::fromPCL(cloud->header.stamp, msg_stamp);
        std::string cloud_frame_id = cloud->header.frame_id;  // cut off the first forward slash
        if (cloud_frame_id.at(0) == '/')
          cloud_frame_id = cloud_frame_id.substr(1);  // cut off the first forward slash
        NODELET_INFO_STREAM("[PCLDetector]: Input PC has " << cloud->size() << " points");

        /* filter input cloud and transform it to world //{ */

        pc_XYZ_t::Ptr cloud_filtered = boost::make_shared<pc_XYZ_t>(*cloud);
        Eigen::Vector3d tf_trans;
        {
          /* filter by cropping points inside a box, relative to the sensor //{ */
          {
            const Eigen::Vector4f box_point1(m_exclude_box_offset_x + m_exclude_box_size_x / 2, m_exclude_box_offset_y + m_exclude_box_size_y / 2,
                                             m_exclude_box_offset_z + m_exclude_box_size_z / 2, 1);
            const Eigen::Vector4f box_point2(m_exclude_box_offset_x - m_exclude_box_size_x / 2, m_exclude_box_offset_y - m_exclude_box_size_y / 2,
                                             m_exclude_box_offset_z - m_exclude_box_size_z / 2, 1);
            pcl::CropBox<pcl::PointXYZ> cb;
            cb.setMax(box_point1);
            cb.setMin(box_point2);
            cb.setInputCloud(cloud_filtered);
            cb.setNegative(true);
            /* cb.setKeepOrganized(m_keep_pc_organized); */
            cb.filter(*cloud_filtered);
            /* cb.setInputCloud(cloud); */
            /* cb.setIndices(indices_filtered); */
            /* cb.setNegative(true); */
            /* cb.filter(indices_filtered->indices); */
          }
          //}
          NODELET_INFO_STREAM("[PCLDetector]: Input PC after CropBox 1: " << cloud_filtered->size() << " points");

          Eigen::Affine3d s2w_tf;
          bool tf_ok = get_transform_to_world(cloud_frame_id, msg_stamp, s2w_tf);
          if (!tf_ok)
          {
            NODELET_ERROR("[PCLDetector]: Could not transform pointcloud to global, skipping.");
            return;
          }
          tf_trans = s2w_tf.translation();
          pcl::transformPointCloud(*cloud_filtered, *cloud_filtered, s2w_tf.cast<float>());
          cloud_filtered->header.frame_id = m_world_frame_id;

          /* filter by cropping points outside a bounding box of the arena //{ */
          {
            const Eigen::Vector4f box_point1(m_arena_bbox_offset_x + m_arena_bbox_size_x / 2.0, m_arena_bbox_offset_y + m_arena_bbox_size_y / 2.0,
                                             m_arena_bbox_offset_z + m_arena_bbox_size_z, 1);
            const Eigen::Vector4f box_point2(m_arena_bbox_offset_x - m_arena_bbox_size_x / 2.0, m_arena_bbox_offset_y - m_arena_bbox_size_y / 2.0,
                                             m_arena_bbox_offset_z, 1);
            pcl::CropBox<pcl::PointXYZ> cb;
            cb.setMax(box_point1);
            cb.setMin(box_point2);
            cb.setInputCloud(cloud_filtered);
            cb.setNegative(false);
            /* cb.setKeepOrganized(m_keep_pc_organized); */
            cb.filter(*cloud_filtered);
            /* cb.setNegative(false); */
            /* cb.filter(indices_filtered->indices); */
          }
          //}
          NODELET_INFO_STREAM("[PCLDetector]: Input PC after arena CropBox 2: " << cloud_filtered->size() << " points");

          // Filter by cropping points outside the safety area
          filter_points(cloud_filtered);
          NODELET_INFO_STREAM("[PCLDetector]: Input PC after arena filtering: " << cloud_filtered->size() << " points");

          NODELET_INFO_STREAM("[PCLDetector]: Filtered input PC has " << cloud_filtered->size() << " points");
        }

        //}

        /* pcl::PointCloud<pcl::PointXYZL> cloud_clusters; */
        /* /1* extract euclidean clusters //{ *1/ */
        /* { */
        /*   std::vector<pcl::PointIndices> cluster_indices; */
        /*   pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec; */
        /*   ec.setClusterTolerance(0.5); */
        /*   ec.setMinClusterSize(1); */
        /*   ec.setMaxClusterSize(25000); */
        /*   ec.setInputCloud(cloud_filtered); */
        /*   ec.extract(cluster_indices); */
        /*   cloud_clusters.reserve(cloud_filtered->size()); */
        /*   int label = 0; */
        /*   for (const auto& idxs : cluster_indices) */
        /*   { */
        /*     for (const auto idx : idxs.indices) */
        /*     { */
        /*       const auto pt_orig = cloud_filtered->at(idx); */
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

        for (const auto& pt : *cloud_filtered)
        {
          const int pt_arena_x = std::floor(pt.x - m_arena_bbox_offset_x + m_arena_bbox_size_x / 2.0);
          const int pt_arena_y = std::floor(pt.y - m_arena_bbox_offset_y + m_arena_bbox_size_y / 2.0);
          const int pt_arena_z = std::floor(pt.z - m_arena_bbox_offset_z);
          map_at_coords(m_map3d, pt_arena_x, pt_arena_y, pt_arena_z) += 1.0;
          map_at_coords(m_map3d_last_update, pt_arena_x, pt_arena_y, pt_arena_z) = msg_stamp;
        }

        std_msgs::Header header;
        header.frame_id = m_world_frame_id;
        header.stamp = msg_stamp;

        const auto most_probable_passthrough_opt = find_most_probable_passthrough(header);
        if (most_probable_passthrough_opt.has_value())
          m_pub_chosen_position.publish(most_probable_passthrough_opt.value());

        if (m_pub_filtered_input_pc.getNumSubscribers() > 0)
          m_pub_filtered_input_pc.publish(cloud_filtered);
        if (m_pub_map3d.getNumSubscribers() > 0)
          m_pub_map3d.publish(map3d_visualization(header));

        const double delay = (ros::Time::now() - msg_stamp).toSec();
        NODELET_INFO_STREAM("[PCLDetector]: Done processing data with delay " << delay << "s ---------------------------------------------- ");
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
    /*   NODELET_INFO_STREAM("[" << m_node_name << "]: det. blobs/image: " << blobs_per_image << " | inp. FPS: " << round(input_fps) << " | proc. FPS: " <<
     * round(m_avg_fps) << " | delay: " << round(1000.0f*m_avg_delay) << "ms"); */
    /*   m_det_blobs = 0; */
    /*   m_images_processed = 0; */
    /* } */
    /* //} */

    /* init_safety_area() method //{ */
    void init_safety_area([[maybe_unused]] const ros::TimerEvent& evt)
    {
      assert(m_safety_area_border_points.cols() == 2);
      const auto tf_opt = m_transformer.getTransform(m_safety_area_frame, m_world_frame_id);
      if (!tf_opt.has_value())
      {
        ROS_ERROR("Safety area could not be transformed!");
        return;
      }

      const auto tf = tf_opt.value();
      /* transform border_points //{ */

      {
        for (int it = 0; it < m_safety_area_border_points.rows(); it++)
        {
          Eigen::Vector2d vec = m_safety_area_border_points.row(it);
          geometry_msgs::Point pt;
          pt.x = vec.x();
          pt.y = vec.y();
          pt.z = 0.0;
          auto tfd = m_transformer.transformHeaderless(tf, pt);
          if (!tfd.has_value())
          {
            ROS_ERROR("Safety area could not be transformed!");
            return;
          } else
          {
            m_safety_area_border_points.row(it).x() = tfd.value().x;
            m_safety_area_border_points.row(it).y() = tfd.value().y;
          }
        }
      }

      //}

      std::vector<point> boost_area_pts;
      boost_area_pts.reserve(m_safety_area_border_points.rows());
      for (int it = 0; it < m_safety_area_border_points.rows(); it++)
      {
        const Eigen::RowVector2d row = m_safety_area_border_points.row(it);
        boost_area_pts.push_back({row.x(), row.y()});
      }
      if (!boost_area_pts.empty())
        boost_area_pts.push_back({m_safety_area_border_points.row(0).x(), m_safety_area_border_points.row(0).y()});
      m_safety_area_ring = ring(std::begin(boost_area_pts), std::end(boost_area_pts));

      // Declare strategies
      const int points_per_circle = 36;
      boost::geometry::strategy::buffer::distance_symmetric distance_strategy(-m_safety_area_deflation);
      boost::geometry::strategy::buffer::join_round join_strategy(points_per_circle);
      boost::geometry::strategy::buffer::end_round end_strategy(points_per_circle);
      boost::geometry::strategy::buffer::point_circle circle_strategy(points_per_circle);
      boost::geometry::strategy::buffer::side_straight side_strategy;

      mpolygon result;
      // Create the buffer of a multi polygon
      boost::geometry::buffer(m_safety_area_ring, result, distance_strategy, side_strategy, join_strategy, end_strategy, circle_strategy);
      if (result.empty())
      {
        m_safety_area_ring = ring();
        ROS_ERROR("[PCLDetector]: Deflated safety area is empty! This probably shouldn't happen!");
        return;
      }

      if (result.size() > 1)
        ROS_WARN("[PCLDetector]: Deflated safety area breaks into multiple pieces! This probably shouldn't happen! Using the first piece...");

      polygon poly = result.at(0);
      m_safety_area_ring = poly.outer();

      box bbox;
      boost::geometry::envelope(m_safety_area_ring, bbox);
      m_arena_bbox_size_x = std::ceil(std::abs(bbox.max_corner().x() - bbox.min_corner().x()));
      m_arena_bbox_size_y = std::ceil(std::abs(bbox.max_corner().y() - bbox.min_corner().y()));
      m_arena_bbox_size_z = std::ceil(std::abs(m_safety_area_max_z - m_safety_area_min_z));
      m_arena_bbox_offset_x = (bbox.max_corner().x() + bbox.min_corner().x()) / 2.0;
      m_arena_bbox_offset_y = (bbox.max_corner().y() + bbox.min_corner().y()) / 2.0;
      m_arena_bbox_offset_z = std::min(m_safety_area_max_z, m_safety_area_min_z);

      m_map_size = m_arena_bbox_size_x * m_arena_bbox_size_y * m_arena_bbox_size_z;
      ROS_INFO("[PCLDetector]: Arena initialized with bounding box size [%d, %d, %d] (%d voxels).", m_arena_bbox_size_x, m_arena_bbox_size_y,
               m_arena_bbox_size_z, m_map_size);

      // initialize the frequency map
      m_map3d.resize(m_map_size);
      for (int it = 0; it < m_map_size; it++)
        m_map3d.at(it) = 0;

      // initialize the stamp map
      m_map3d_last_update.resize(m_map_size);
      for (int it = 0; it < m_map_size; it++)
        m_map3d_last_update.at(it) = ros::Time(0);

      m_safety_area_init_timer.stop();
      m_safety_area_initialized = true;

      std_msgs::Header header;
      header.frame_id = m_world_frame_id;
      header.stamp = ros::Time::now();

      {
        auto msg = oparea_visualization(header);
        m_pub_oparea.publish(msg);
      }

      {
        auto msg = map3d_bounds_visualization(header);
        m_pub_map3d_bounds.publish(msg);
      }
    }
    //}

  private:

    /* arena_to_global() method //{ */
    template <class T>
    T arena_to_global(int x, int y, int z)
    {
      T ret;
      ret.x = x + m_arena_bbox_offset_x - m_arena_bbox_size_x / 2.0 + 0.5;
      ret.y = y + m_arena_bbox_offset_y - m_arena_bbox_size_y / 2.0 + 0.5;
      ret.z = z + m_arena_bbox_offset_z + 0.5;
      return ret;
    }
    //}

    /* to_eigen() method //{ */
    vec3_t to_eigen(const pt_XYZ_t& pt)
    {
      return {pt.x, pt.y, pt.z};
    }
    //}

    /* find_most_probable_passthrough() method //{ */
    std::optional<geometry_msgs::PoseStamped> find_most_probable_passthrough(const std_msgs::Header& header)
    {
      geometry_msgs::PoseStamped ret;
      ret.header = header;

      float maxval = 0.0;
      int max_x = 0, max_y = 0, max_z = 0;
      ros::Time max_stamp = ros::Time(0);
      /* find the maximal index and its value //{ */
      
      for (int x_it = 0; x_it < m_arena_bbox_size_x; x_it++)
      {
        for (int y_it = 0; y_it < m_arena_bbox_size_y; y_it++)
        {
          for (int z_it = 0; z_it < m_arena_bbox_size_z; z_it++)
          {
            const float mapval = map_at_coords(m_map3d, x_it, y_it, z_it);
            if (mapval > maxval)
            {
              maxval = mapval;
              max_x = x_it;
              max_y = y_it;
              max_z = z_it;
              max_stamp = map_at_coords(m_map3d_last_update, x_it, y_it, z_it);
            }
          }
        }
      }
      
      //}

      if (maxval == 0.0)
        return std::nullopt;

      // find neighborhood points and their stamps
      // note that per one voxel, N points are added, where N is the weight of the voxel (its value in the map)
      // TODO: parametrize this shit
      const int m_neighborhood = 3;
      int n_unique_poits = 0;
      pc_XYZ_t::Ptr line_pts = boost::make_shared<pc_XYZ_t>();
      line_pts->reserve(maxval*m_neighborhood*m_neighborhood*m_neighborhood);
      std::vector<float> line_dts;
      line_dts.reserve(maxval*m_neighborhood*m_neighborhood*m_neighborhood);
      for (int x_it = std::max(max_x-m_neighborhood, 0); x_it < std::min(max_x+m_neighborhood, m_arena_bbox_size_x-1); x_it++)
      {
        for (int y_it = std::max(max_y-m_neighborhood, 0); y_it < std::min(max_y+m_neighborhood, m_arena_bbox_size_y-1); y_it++)
        {
          for (int z_it = std::max(max_z-m_neighborhood, 0); z_it < std::min(max_z+m_neighborhood, m_arena_bbox_size_z-1); z_it++)
          {
            const int mapval = std::ceil(map_at_coords(m_map3d, x_it, y_it, z_it));
            const float dt = (map_at_coords(m_map3d_last_update, x_it, y_it, z_it) - max_stamp).toSec();
            const pt_XYZ_t pt {float(x_it), float(y_it), float(z_it)};
            for (int it = 0; it < mapval; it++)
            {
              line_pts->push_back(pt);
              line_dts.push_back(dt);
            }
            if (mapval > 0)
              n_unique_poits++;
          }
        }
      }

      // TODO: parametrize this shit
      const int m_min_linefit_points = 3;
      if (n_unique_poits < m_min_linefit_points)
      {
        ROS_WARN("[PCLDetector]: Not enough points to fit line, skipping (got %d/%d)", n_unique_poits, m_min_linefit_points);
        return std::nullopt;
      }

      // fit a line to the neighborhood points
      auto model_l = boost::make_shared<pcl::SampleConsensusModelLine<pt_XYZ_t>>(line_pts);
      pcl::LeastMedianSquares<pt_XYZ_t> fitter(model_l);
      fitter.setDistanceThreshold(1);
      fitter.computeModel();
      std::vector<int> inliers;
      fitter.getInliers(inliers);

      // get the inlier dts and points
      using dt_pt_t = std::pair<float, vec3_t>;
      // TODO: parametrize this shit
      const float m_linefit_max_point_age_coeff = 2.0;
      const float max_point_age = m_neighborhood*m_linefit_max_point_age_coeff; // seconds
      std::vector<dt_pt_t> in_dt_pts;
      for (const auto idx : inliers)
      {
        const float dt = line_dts.at(idx);
        // ignore too old points
        if (dt < -max_point_age)
          continue;
        in_dt_pts.push_back({dt, to_eigen(line_pts->at(idx))});
      }
      // sort the inliers by relative time to the main point
      std::sort(std::begin(in_dt_pts), std::end(in_dt_pts),
          // comparison lambda function
            [](const dt_pt_t& a, 
               const dt_pt_t& b)
            {
              return a.first < b.first;
            }
          );

      // find the mean velocity between the points
      vec3_t mean_vel(0, 0, 0);
      size_t mean_vel_weight = 0;
      dt_pt_t prev_dt_pt;
      size_t prev_dt_pt_weight = 0;
      bool prev_dt_pt_initialized = false;
      for (const auto& dt_pt : in_dt_pts)
      {
        if (!prev_dt_pt_initialized)
        {
          prev_dt_pt = dt_pt;
          prev_dt_pt_initialized = true;
          prev_dt_pt_weight = 1;
          continue;
        }

        const float cur_dt = dt_pt.first - prev_dt_pt.first;
        const auto cur_pt = dt_pt.second;
        // points from the same time don't give us any information about speed - average them
        if (cur_dt == 0.0)
        {
          prev_dt_pt.second += cur_pt;
          prev_dt_pt_weight += 1;
          continue;
        }
        // in case of weighting, calculate the average
        prev_dt_pt.second /= prev_dt_pt_weight;

        const vec3_t cur_vel = (dt_pt.second - prev_dt_pt.second)/cur_dt;
        mean_vel += cur_vel;
        mean_vel_weight += 1;
        prev_dt_pt.second = dt_pt.second;
        prev_dt_pt_weight = 1;
      }

      // if no velocity estimate could be obtained from the inliers, get at least some velocity estimation even if it might be wrong sign
      if (mean_vel.isZero() || mean_vel_weight == 0.0f)
      {
        Eigen::VectorXf model;
        fitter.getModelCoefficients(model);
        if (model.rows() < 6)
        {
          ROS_ERROR("[PCLDetector]: Line fit failed!");
          return std::nullopt;
        }
        mean_vel = model.block<3, 1>(3, 0);
        ROS_WARN("[PCLDetector]: Unable to estimate speed direction from inliers! Using line fit estimate, which may have a wrong sign (opposite direction).");
        /* ROS_WARN_THROTTLE(0.5, "[PCLDetector]: Unable to estimate speed direction from inliers! Using line fit estimate, which may have a wrong sign (opposite direction)."); */
      }
      else
      {
        // otherwise use the inlier-estimated speed
        mean_vel /= mean_vel_weight;
      }

      // if requested, publish the neighborhood for debugging
      /*  //{ */
      
      if (m_pub_chosen_neighborhood.getNumSubscribers() > 0)
      {
        pc_XYZt_t cloud_out;
        const auto stamp = ros::Time::now();
        pcl_conversions::toPCL(stamp, cloud_out.header.stamp);
        cloud_out.header.frame_id = m_world_frame_id;
        cloud_out.reserve(in_dt_pts.size());
        for (const auto& dt_pt : in_dt_pts)
        {
          const auto dt = dt_pt.first;
          const auto pt = dt_pt.second;
          pt_XYZt_t pcl_pt = arena_to_global<pt_XYZt_t>(pt.x(), pt.y(), pt.z());
          pcl_pt.intensity = dt;
          cloud_out.push_back(pcl_pt);
        }
        m_pub_chosen_neighborhood.publish(cloud_out);
      }
      
      //}

      // use the estimated velocity to set the luring pose orientation
      const float yaw = std::atan2(mean_vel.y(), mean_vel.x());
      const Eigen::AngleAxisf anax(yaw, vec3_t::UnitZ());
      const Eigen::Quaternionf quat(anax);
      ret.pose.orientation.w = quat.w();
      ret.pose.orientation.x = quat.x();
      ret.pose.orientation.y = quat.y();
      ret.pose.orientation.z = quat.z();

      // the passthrouhgh pose position is just the most frequent detetion position (recalculate it from the map to global coordinates)
      ret.pose.position = arena_to_global<geometry_msgs::Point>(max_x, max_y, max_z);

      return ret;
    }
    //}

    /* in_safety_area() method //{ */
    inline bool in_safety_area(const pcl::PointXYZ& pt)
    {
      /* const bool in_poly = boost::geometry::covered_by(point(pt.x, pt.y), m_safety_area_ring); */
      const bool in_poly = boost::geometry::within(point(pt.x, pt.y), m_safety_area_ring);
      const bool height_ok = pt.z > m_safety_area_min_z && pt.z < m_safety_area_max_z;
      return in_poly && height_ok;
    }
    //}

    /* filter_points() method //{ */
    void filter_points(pc_XYZ_t::Ptr cloud)
    {
      pc_XYZ_t::Ptr cloud_out = boost::make_shared<pc_XYZ_t>();
      cloud_out->reserve(cloud->size() / 100);
      for (size_t it = 0; it < cloud->size(); it++)
      {
        if (in_safety_area(cloud->points[it]))
        {
          cloud_out->push_back(cloud->points[it]);
        }
      }
      cloud_out->swap(*cloud);
    }
    //}

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
      const Eigen::Vector3f dir = vec / dist;
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
      } else
      {
        ret.scale.x = ret.scale.y = ret.scale.z = 0.1;
        ret.type = visualization_msgs::Marker::LINE_LIST;
      }
      ret.points.reserve(mesh.polygons.size() * n_verts);
      pc_XYZ_t mesh_cloud;
      pcl::fromPCLPointCloud2(mesh.cloud, mesh_cloud);
      for (const auto& vert : mesh.polygons)
      {
        if (n_verts == 3)
        {
          if (vert.vertices.size() != n_verts)
            ROS_WARN_THROTTLE(0.1, "[PCLDetector]: Number of vertices in mesh is incosistent (expected: %lu, got %lu)!", n_verts, vert.vertices.size());
          fill_marker_pts_triangles(vert, mesh_cloud, ret.points);
        } else
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
      const float shadow_angle_tolerance = use_shadowed_faces ? -1.0f : (m_drmgr_ptr->config.orgmesh_shadow_ang_tol / 180.0f * M_PI);
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
        for (unsigned c_it1 = 0; c_it1 < pc_width - 1; c_it1++)
        {
          const int c_it2 = c_it1 + 1;
          const auto pt0 = pc->at(c_it1, r_it);
          const auto pt1 = pc->at(c_it2, r_it);
          if (!valid_pt(pt0) || !valid_pt(pt1))
            continue;
          mesh_cloud.push_back(pt0);
          mesh_cloud.push_back(pt1);
          add_triangle(idx0, idx0 + 1, idxc, mesh_polygons);
          idx0 += 2;
        }
      }
      //}

      /* stitch the last and first columns //{ */
      {
        const int c_it1 = pc_width - 1;
        const int c_it2 = 0;
        auto idx0 = mesh_cloud.size();
        for (unsigned r_it1 = 0; r_it1 < pc_height - 1; r_it1++)
        {
          const int r_it2 = r_it1 + 1;
          const auto pt0 = pc->at(c_it1, r_it1);
          const auto pt1 = pc->at(c_it2, r_it1);
          const auto pt2 = pc->at(c_it1, r_it2);
          const auto pt3 = pc->at(c_it2, r_it2);
          if (!valid_pt(pt0) || !valid_pt(pt1) || !valid_pt(pt2) || !valid_pt(pt3))
            continue;
          mesh_cloud.push_back(pt0);
          mesh_cloud.push_back(pt1);
          mesh_cloud.push_back(pt2);
          mesh_cloud.push_back(pt3);
          add_triangle(idx0 + 2, idx0 + 1, idx0, mesh_polygons);
          add_triangle(idx0 + 2, idx0 + 3, idx0 + 1, mesh_polygons);
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
          if (cur_label > -1 && connected_label != cur_label)  // a label is assigned
          {
            if (connected_label > -1)  // a label is already assigned to the polygon
            {
              const label_t master1 = get_master_label(cur_label, equal_labels);
              const label_t master2 = get_master_label(connected_label, equal_labels);
              if (master1 != master2)
                equal_labels.insert({master1, master2});
            }  // if (connected_label > -1)
            else
            {
              connected_label = cur_label;
            }  // else (connected_label > -1)
          }    // if (cur_label > -1) // a label is assigned
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
      return (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z));
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
        transform = m_tf_buffer.lookupTransform(m_world_frame_id, frame_id, stamp, timeout);
        tf_out = tf2::transformToEigen(transform.transform);
      }
      catch (tf2::TransformException& ex)
      {
        NODELET_WARN_THROTTLE(1.0, "[%s]: Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", m_node_name.c_str(), frame_id.c_str(),
                              m_world_frame_id.c_str(), ex.what());
        return false;
      }
      return true;
    }
    //}

    /* /1* coords_from_idx() method //{ *1/ */

    /* vec3_t coords_from_idx(const int coord) */
    /* { */
    /*   const float x = (coord % int(std::round(m_arena_bbox_size_y * m_arena_bbox_size_z))) + m_arena_bbox_offset_x/2.0; */
    /*   const float y = ((coord -  % int(std::round(m_arena_bbox_size_z))) + m_arena_bbox_offset_y/2.0; */
    /*   const float z = (coord % int(std::round(m_arena_bbox_size_y * m_arena_bbox_size_z))) + m_arena_bbox_offset_z; */
    /*   return {x, y, z}; */
    /* } */

    /* //} */

    /* map_at_coords() method //{ */

    template <class T>
    T& map_at_coords(std::vector<T>& map, int coord_x, int coord_y, int coord_z)
    {
      return map.at(coord_x * m_arena_bbox_size_y * m_arena_bbox_size_z + coord_y * m_arena_bbox_size_z + coord_z);
    }

    //}

    /* map3d_visualization() method //{ */
    visualization_msgs::Marker map3d_visualization(const std_msgs::Header header)
    {
      visualization_msgs::Marker ret;
      ret.header = header;
      ret.points.reserve(m_map3d.size() / 1000);
      ret.pose.position = arena_to_global<geometry_msgs::Point>(0, 0, 0);
      ret.pose.orientation.w = 1.0;
      ret.scale.x = ret.scale.y = ret.scale.z = 1.0;
      ret.color.a = 1.0;
      ret.color.r = 0.0;
      ret.type = visualization_msgs::Marker::CUBE_LIST;

      float maxval = 0.0;
      for (const auto val : m_map3d)
        if (val > maxval)
          maxval = val;

      for (int x_it = 0; x_it < m_arena_bbox_size_x; x_it++)
      {
        for (int y_it = 0; y_it < m_arena_bbox_size_y; y_it++)
        {
          for (int z_it = 0; z_it < m_arena_bbox_size_z; z_it++)
          {
            const float mapval = map_at_coords(m_map3d, x_it, y_it, z_it);
            if (mapval <= 0.0)
              continue;

            geometry_msgs::Point pt;
            pt.x = x_it;
            pt.y = y_it;
            pt.z = z_it;
            ret.points.push_back(pt);

            std_msgs::ColorRGBA color;
            color.a = mapval / maxval;
            color.r = 1.0;
            ret.colors.push_back(color);
          }
        }
      }

      if (ret.points.empty())
      {
        ret.points.push_back({});
        ret.color.a = 0.0;
      }
      return ret;
    }
    //}

    /* oparea_visualization() method //{ */

    geometry_msgs::Point boost2gmsgs(const point& bpt, const double height)
    {
      geometry_msgs::Point pt;
      pt.x = bpt.x();
      pt.y = bpt.y();
      pt.z = height;
      return pt;
    }

    visualization_msgs::Marker oparea_visualization(const std_msgs::Header header)
    {
      visualization_msgs::Marker safety_area_marker;
      safety_area_marker.header = header;

      safety_area_marker.id = 333;
      safety_area_marker.ns = "oparea";
      safety_area_marker.type = visualization_msgs::Marker::LINE_LIST;
      safety_area_marker.color.a = 0.15;
      safety_area_marker.scale.x = 0.2;
      safety_area_marker.color.r = 0;
      safety_area_marker.color.g = 0;
      safety_area_marker.color.b = 1;

      safety_area_marker.pose.orientation.x = 0;
      safety_area_marker.pose.orientation.y = 0;
      safety_area_marker.pose.orientation.z = 0;
      safety_area_marker.pose.orientation.w = 1;

      /* adding safety area points //{ */

      // bottom border
      for (size_t i = 0; i < m_safety_area_ring.size() - 1; i++)
      {
        const auto pt1 = boost2gmsgs(m_safety_area_ring.at(i), m_safety_area_min_z);
        const auto pt2 = boost2gmsgs(m_safety_area_ring.at(i + 1), m_safety_area_min_z);
        safety_area_marker.points.push_back(pt1);
        safety_area_marker.points.push_back(pt2);
      }

      // top border
      for (size_t i = 0; i < m_safety_area_ring.size() - 1; i++)
      {
        const auto pt1 = boost2gmsgs(m_safety_area_ring.at(i), m_safety_area_max_z);
        const auto pt2 = boost2gmsgs(m_safety_area_ring.at(i + 1), m_safety_area_max_z);
        safety_area_marker.points.push_back(pt1);
        safety_area_marker.points.push_back(pt2);
      }

      // top/bot edges
      for (size_t i = 0; i < m_safety_area_ring.size() - 1; i++)
      {
        const auto pt1 = boost2gmsgs(m_safety_area_ring.at(i), m_safety_area_min_z);
        const auto pt2 = boost2gmsgs(m_safety_area_ring.at(i), m_safety_area_max_z);
        safety_area_marker.points.push_back(pt1);
        safety_area_marker.points.push_back(pt2);
      }

      //}

      return safety_area_marker;
    }

    //}

    /* map3d_bounds_visualization() method //{ */

    visualization_msgs::Marker map3d_bounds_visualization(const std_msgs::Header header)
    {
      visualization_msgs::Marker safety_area_marker;
      safety_area_marker.header = header;

      safety_area_marker.id = 666;
      safety_area_marker.ns = "map3d_bounds";
      safety_area_marker.type = visualization_msgs::Marker::CUBE;
      safety_area_marker.color.a = 0.15;
      safety_area_marker.color.r = 0;
      safety_area_marker.color.g = 0;
      safety_area_marker.color.b = 1;

      safety_area_marker.pose.orientation.x = 0;
      safety_area_marker.pose.orientation.y = 0;
      safety_area_marker.pose.orientation.z = 0;
      safety_area_marker.pose.orientation.w = 1;

      geometry_msgs::Point pt;
      pt.x = m_arena_bbox_offset_x;
      pt.y = m_arena_bbox_offset_y;
      pt.z = m_arena_bbox_offset_z + m_arena_bbox_size_z / 2.0;
      safety_area_marker.pose.position = pt;

      safety_area_marker.scale.x = m_arena_bbox_size_x;
      safety_area_marker.scale.y = m_arena_bbox_size_y;
      safety_area_marker.scale.z = m_arena_bbox_size_z;

      return safety_area_marker;
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
    mrs_lib::SubscribeHandlerPtr<pc_XYZ_t> m_pc_sh;
    /* ros::Publisher m_detections_pub; */
    ros::Publisher m_pub_filtered_input_pc;
    ros::Publisher m_pub_map3d;
    ros::Publisher m_pub_map3d_bounds;
    ros::Publisher m_pub_oparea;
    ros::Publisher m_pub_chosen_neighborhood;
    ros::Publisher m_pub_chosen_position;
    ros::Timer m_main_loop_timer;
    ros::Timer m_info_loop_timer;
    ros::Timer m_safety_area_init_timer;
    std::string m_node_name;

    mrs_lib::Transformer m_transformer;
    //}

  private:
    // --------------------------------------------------------------
    // |                 Parameters, loaded from ROS                |
    // --------------------------------------------------------------

    /* Parameters, loaded from ROS //{ */

    std::string m_world_frame_id;
    double m_exclude_box_offset_x;
    double m_exclude_box_offset_y;
    double m_exclude_box_offset_z;
    double m_exclude_box_size_x;
    double m_exclude_box_size_y;
    double m_exclude_box_size_z;

    std::string m_safety_area_frame;
    Eigen::MatrixXd m_safety_area_border_points;
    double m_safety_area_deflation;
    ring m_safety_area_ring;
    double m_safety_area_min_z;
    double m_safety_area_max_z;

    bool m_keep_pc_organized;

    //}

  private:
    // --------------------------------------------------------------
    // |                   Other member variables                   |
    // --------------------------------------------------------------

    bool m_safety_area_initialized;
    uint32_t m_last_detection_id;
    std::vector<float> m_map3d;
    std::vector<ros::Time> m_map3d_last_update;
    double m_arena_bbox_offset_x;  // x-center of the 3D bounding box
    double m_arena_bbox_offset_y;  // y-center of the 3D bounding box
    double m_arena_bbox_offset_z;  // z-bottom of the 3D bounding box
    int m_arena_bbox_size_x;
    int m_arena_bbox_size_y;
    int m_arena_bbox_size_z;

    int m_map_size;

    /* Statistics variables //{ */
    std::mutex m_stat_mtx;
    unsigned m_det_blobs;
    unsigned m_images_processed;
    float m_avg_fps;
    float m_avg_delay;
    //}

  };  // class PCLDetector
};    // namespace uav_detect

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(uav_detect::PCLDetector, nodelet::Nodelet)
