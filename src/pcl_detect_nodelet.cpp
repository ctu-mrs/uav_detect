/* includes etc. //{ */

#include "main.h"

#include <nodelet/nodelet.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/octree/octree_search.h>
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
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/surface/poisson.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/lmeds.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <uav_detect/DetectionParamsConfig.h>
#include <mesh_sampling.h>
/* #include <PointXYZt.h> */

using namespace cv;
using namespace std;
using namespace uav_detect;
using namespace pcl;

//}

namespace uav_detect
{
  // shortcut type to the dynamic reconfigure manager template instance
  using drmgr_t = mrs_lib::DynamicReconfigureMgr<uav_detect::DetectionParamsConfig>;
  using pt_XYZ_t = pcl::PointXYZ;
  using pc_XYZ_t = pcl::PointCloud<pt_XYZ_t>;
  using pt_XYZt_t = pcl::PointXYZI;
  using pc_XYZt_t = pcl::PointCloud<pt_XYZt_t>;
  using octree = pcl::octree::OctreePointCloudSearch<pt_XYZt_t>;

  using point = boost::geometry::model::d2::point_xy<double>;
  using ring = boost::geometry::model::ring<point>;
  using polygon = boost::geometry::model::polygon<point>;
  using mpolygon = boost::geometry::model::multi_polygon<polygon>;
  using box = boost::geometry::model::box<point>;

  using vec3_t = Eigen::Vector3f;
  using vec4_t = Eigen::Vector4f;
  using quat_t = Eigen::Quaternionf;
  using anax_t = Eigen::AngleAxisf;

  class PCLDetector : public nodelet::Nodelet
  {
  private:
    /* struct linefit_t //{ */

    struct linefit_t
    {
      using params_t = Eigen::Matrix<float, 6, 1>;
      using dt_pt_t = std::pair<float, vec3_t>;

      std::vector<dt_pt_t> dt_pts;
      params_t parameters;
    };

    //}

    /* struct planefit_t //{ */

    struct planefit_t
    {
      vec3_t normal;
      float offset;
    };

    //}

    /* /1* struct line_segment_t //{ *1/ */

    /* struct line_segment_t */
    /* { */
    /*   vec3_t start_pt; */
    /*   vec3_t end_pt; */
    /* }; */

    /* //} */

  public:
    /* onInit() method //{ */
    void onInit()
    {
      ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();
      ROS_INFO("[PCLDetector]: Waiting for valid time...");
      ros::Time::waitForValid();

      m_node_name = "PCLDetector";

      /* Load parameters from ROS //{*/
      NODELET_INFO("Loading default dynamic parameters:");
      m_drmgr_ptr = make_unique<drmgr_t>(nh, m_node_name);

      mrs_lib::ParamLoader pl(nh, m_node_name);
      // LOAD STATIC PARAMETERS
      NODELET_INFO("Loading static parameters:");
      const auto uav_name = pl.load_param2<std::string>("uav_name");
      pl.load_param("world_frame_id", m_world_frame_id);

      pl.load_param("ball_speed", m_ball_speed);
      pl.load_param("max_speed_error", m_max_speed_error);

      pl.load_param("linefit/neighborhood", m_linefit_neighborhood);
      pl.load_param("linefit/point_max_age_coefficient", m_linefit_point_max_age_coeff);

      pl.load_param("classification/max_detection_height", m_classif_max_detection_height);
      pl.load_param("classification/max_detection_width", m_classif_max_detection_width);
      pl.load_param("classification/max_mav_height", m_classif_max_mav_height);
      pl.load_param("classification/ball_wire_length", m_classif_ball_wire_length);
      pl.load_param("classification/close_distance", m_classif_close_dist);
      pl.load_param("classification/mav_width", m_classif_mav_width);
      pl.load_param("classification/ball_width", m_classif_ball_width);

      pl.load_param("plane_fit/min_points", m_plane_fit_min_points);
      pl.load_param("plane_fit/ransac_threshold", m_plane_fit_ransac_threshold);

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
      m_pub_detections_pc = nh.advertise<sensor_msgs::PointCloud2>("detections_pc", 1);
      m_pub_detection = nh.advertise<geometry_msgs::PoseStamped>("detection", 1);
      m_pub_detection_classified = nh.advertise<visualization_msgs::MarkerArray>("detection_classified", 1);
      m_pub_detection_neighborhood = nh.advertise<sensor_msgs::PointCloud2>("detection_neighborhood", 1, true);
      m_pub_plane = nh.advertise<visualization_msgs::MarkerArray>("plane", 1);
      //}

      /* initialize transformer //{ */

      m_transformer = mrs_lib::Transformer(m_node_name, uav_name);

      //}

      m_global_cloud = boost::make_shared<pc_XYZt_t>();
      m_global_cloud->header.frame_id = m_world_frame_id;
      m_global_octree = boost::make_shared<octree>(1.0); // 1 metre resolution
      m_global_octree->setInputCloud(m_global_cloud);
      m_last_detection_id = 0;

      m_det_blobs = 0;
      m_images_processed = 0;
      m_avg_fps = 0.0f;
      m_avg_delay = 0.0f;

      m_start_time = ros::Time::now();

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
        NODELET_WARN_STREAM_THROTTLE(1.0, "[MainLoop]: Safety area not initialized, skipping. ");
        return;
      }

      if (m_pc_sh->new_data())
      {
        const ros::WallTime start_t = ros::WallTime::now();

        NODELET_INFO_STREAM("[MainLoop]: Processing new data --------------------------------------------------------- ");

        pc_XYZ_t::ConstPtr cloud = m_pc_sh->get_data();
        ros::Time msg_stamp;
        pcl_conversions::fromPCL(cloud->header.stamp, msg_stamp);
        std::string cloud_frame_id = cloud->header.frame_id;  // cut off the first forward slash
        if (cloud_frame_id.at(0) == '/')
          cloud_frame_id = cloud_frame_id.substr(1);  // cut off the first forward slash
        NODELET_INFO_STREAM("[MainLoop]: Input PC has " << cloud->size() << " points");

        /* filter input cloud and transform it to world //{ */

        pc_XYZ_t::Ptr cloud_filtered = boost::make_shared<pc_XYZ_t>(*cloud);
        vec3_t tf_trans;
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
          NODELET_INFO_STREAM("[MainLoop]: Input PC after CropBox 1: " << cloud_filtered->size() << " points");

          Eigen::Affine3d s2w_tf;
          bool tf_ok = get_transform_to_world(cloud_frame_id, msg_stamp, s2w_tf);
          if (!tf_ok)
          {
            NODELET_ERROR("[MainLoop]: Could not transform pointcloud to global, skipping.");
            return;
          }
          tf_trans = s2w_tf.translation().cast<float>();
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
          NODELET_INFO_STREAM("[MainLoop]: Input PC after arena CropBox 2: " << cloud_filtered->size() << " points");

          // Filter by cropping points outside the safety area
          filter_points(cloud_filtered);
          NODELET_INFO_STREAM("[MainLoop]: Input PC after arena filtering: " << cloud_filtered->size() << " points");

          NODELET_INFO_STREAM("[MainLoop]: Filtered input PC has " << cloud_filtered->size() << " points");
        }

        //}

        // TODO: parametrize this shit
        const int m_min_cluster_size = 3;
        std::vector<pc_XYZ_t::Ptr> cloud_clusters;
        pcl::PointIndices::Ptr keep_points =
            boost::make_shared<pcl::PointIndices>();  // which points to keep (must be in a cluster with size > m_min_cluster_size)
        keep_points->indices.reserve(cloud_filtered->size());
        /* extract euclidean clusters //{ */
        {
          std::vector<pcl::PointIndices> cluster_indices;
          pcl::EuclideanClusterExtraction<pt_XYZ_t> ec;
          ec.setClusterTolerance(2.0);
          ec.setMinClusterSize(1);
          ec.setMaxClusterSize(25000);
          ec.setInputCloud(cloud_filtered);
          ec.extract(cluster_indices);

          int label = 0;
          cloud_clusters.reserve(cluster_indices.size());
          for (const auto& idxs : cluster_indices)
          {
            // skip too small clusters (to filter out singular points -> noise)
            if (idxs.indices.size() < m_min_cluster_size)
              continue;

            pc_XYZ_t::Ptr cloud_cluster = boost::make_shared<pc_XYZ_t>();
            cloud_cluster->reserve(idxs.indices.size());
            for (const auto idx : idxs.indices)
            {
              const auto pt_orig = cloud_filtered->at(idx);
              pt_XYZ_t pt;
              pt.x = pt_orig.x;
              pt.y = pt_orig.y;
              pt.z = pt_orig.z;
              cloud_cluster->push_back(pt);
              keep_points->indices.push_back(idx);
            }
            cloud_clusters.push_back(cloud_cluster);
            label++;
          }
        }
        //}
        ROS_INFO("[MainLoop]: Found %lu detection candidates", cloud_clusters.size());

        // filter out too small clusters to remove singular points (-> noise)
        /*  //{ */
        
        {
          const size_t n_pts_prev = cloud_filtered->size();
          pcl::ExtractIndices<pt_XYZ_t> ei;
          ei.setIndices(keep_points);
          ei.setInputCloud(cloud_filtered);
          ei.filter(*cloud_filtered);
          ROS_INFO("[MainLoop]: Filtered %lu singular points.", n_pts_prev - cloud_filtered->size());
        }
        
        //}

        std_msgs::Header header;
        header.frame_id = m_world_frame_id;
        header.stamp = msg_stamp;

        std::optional<std::pair<pt_XYZ_t, pc_XYZ_t::Ptr>> ballpos_result_opt = find_ball_position(cloud_clusters, tf_trans, header);

        if (ballpos_result_opt.has_value())
        {
          const auto ballpos = ballpos_result_opt.value().first;
          pt_XYZt_t ballpos_stamped;
          ballpos_stamped.x = ballpos.x;
          ballpos_stamped.y = ballpos.y;
          ballpos_stamped.z = ballpos.z;
          ballpos_stamped.intensity = (msg_stamp - m_start_time).toSec();
          m_global_octree->addPointToCloud(ballpos_stamped, m_global_cloud);

          // update voxels in the frequency map and stamp map
          {
            const auto cloud_used = ballpos_result_opt.value().second;
            const auto [pt_arena_x, pt_arena_y, pt_arena_z] = global_to_arena(ballpos.x, ballpos.y, ballpos.z);
            if (valid_arena_coordinates(pt_arena_x, pt_arena_y, pt_arena_z))
            {
              map_at_coords(m_map3d, pt_arena_x, pt_arena_y, pt_arena_z) += cloud_used->size();
              map_at_coords(m_map3d_last_update, pt_arena_x, pt_arena_y, pt_arena_z) = msg_stamp;
            }
            else
            {
              NODELET_WARN("[MainLoop]: Ball seems to be out of arena bounds. Skipping.");
            }
          }

          // publish the current detection pose, if applicable
          const auto detection_pose_opt = find_detection_pose(ballpos, header);
          if (detection_pose_opt.has_value())
            m_pub_detection.publish(detection_pose_opt.value());
        }

        // find and publish the most probable pose the ball will pass through again from the frequency map
        const auto most_probable_passthrough_opt = find_most_probable_passthrough(header);
        if (most_probable_passthrough_opt.has_value())
          m_pub_chosen_position.publish(most_probable_passthrough_opt.value());

        /* const auto plane_opt = fit_plane(m_global_cloud); */
        /* if (plane_opt.has_value()) */
        /*   m_pub_plane.publish(plane_visualization(plane_opt.value(), header, tf_trans)); */

        // publish some debug shit
        if (m_pub_filtered_input_pc.getNumSubscribers() > 0)
          m_pub_filtered_input_pc.publish(cloud_filtered);
        if (m_pub_map3d.getNumSubscribers() > 0)
          m_pub_map3d.publish(map3d_visualization(header));
        if (m_pub_detections_pc.getNumSubscribers() > 0)
        {
          m_global_cloud->header.stamp = cloud->header.stamp;
          m_pub_detections_pc.publish(m_global_cloud);
        }

        const double delay = (ros::Time::now() - msg_stamp).toSec();
        NODELET_INFO_STREAM("[MainLoop]: Done processing data with delay " << delay << "s ---------------------------------------------- ");
        const ros::WallTime end_t = ros::WallTime::now();
        std::cout << "Processing time: " << end_t - start_t << "s" << std::endl;
      }
    }
    //}

    /* segmentation_loop() method //{ */
    void segmentation_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      if (!m_safety_area_initialized)
      {
        NODELET_WARN_STREAM_THROTTLE(1.0, "[SegmentationLoop]: Safety area not initialized, skipping.");
        return;
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
        ROS_ERROR("[InitSafetyArea]: Deflated safety area is empty! This probably shouldn't happen!");
        return;
      }

      if (result.size() > 1)
        ROS_WARN("[InitSafetyArea]: Deflated safety area breaks into multiple pieces! This probably shouldn't happen! Using the first piece...");

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
      ROS_INFO("[InitSafetyArea]: Arena initialized with bounding box size [%d, %d, %d] (%d voxels).", m_arena_bbox_size_x, m_arena_bbox_size_y,
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
    /* find_ball_position() method //{ */
    enum cluster_class_t
    {
      mav,
      ball,
      mav_with_ball,
      unknown
    };

    // finds the largest cluster, classified as a detection, and the corresponding ball position
    std::optional<std::pair<pt_XYZ_t, pc_XYZ_t::Ptr>> find_ball_position(
        const std::vector<pc_XYZ_t::Ptr>& cloud_clusters,
        const vec3_t& cur_pos,
        const std_msgs::Header& header
        )
    {
      std::optional<std::pair<pt_XYZ_t, pc_XYZ_t::Ptr>> ballpos_result_opt;
      size_t ballpos_n_pts = 0;  // how many points did the cluster used for picking the ball position have
      for (const auto& cluster : cloud_clusters)
      {
        // only the detection with the maximum of points in the cluster is considered
        if (cluster->size() < ballpos_n_pts)
          continue;

        pcl::MomentOfInertiaEstimation<pt_XYZ_t> moie;
        moie.setInputCloud(cluster);
        moie.compute();

        pt_XYZ_t min_pt;
        pt_XYZ_t max_pt;
        pt_XYZ_t center_pt;
        Eigen::Matrix3f rotation;
        moie.getOBB(min_pt, max_pt, center_pt, rotation);
        const vec3_t center = to_eigen(center_pt);
        float height = std::abs(max_pt.z - min_pt.z);
        float width = std::max(std::abs(max_pt.x - min_pt.x), std::abs(max_pt.y - min_pt.y));
        const float dist = (center - cur_pos).norm();

        // make sure that the rotated z-axis is the closest to the original z-axis
        /*  //{ */
        
        const vec3_t base_x = rotation * vec3_t::UnitX();
        const vec3_t base_y = rotation * vec3_t::UnitY();
        const vec3_t base_z = rotation * vec3_t::UnitZ();
        double x_angle = anax_t(quat_t::FromTwoVectors(vec3_t::UnitZ(), base_x)).angle();
        double y_angle = anax_t(quat_t::FromTwoVectors(vec3_t::UnitZ(), base_y)).angle();
        double z_angle = anax_t(quat_t::FromTwoVectors(vec3_t::UnitZ(), base_z)).angle();
        x_angle = std::min(x_angle, M_PI-x_angle);
        y_angle = std::min(y_angle, M_PI-y_angle);
        z_angle = std::min(z_angle, M_PI-z_angle);
        // X is closest to the Z axis
        if (x_angle < y_angle && x_angle < z_angle)
        {
          rotation = anax_t(M_PI_2, base_y) * rotation;
          height = std::abs(max_pt.x - min_pt.x);
          width = std::max(std::abs(max_pt.z - min_pt.z), std::abs(max_pt.y - min_pt.y));
        }
        // Y is closest to the Z axis
        else if (y_angle < x_angle && y_angle < z_angle)
        {
          rotation = anax_t(-M_PI_2, base_x) * rotation;
          height = std::abs(max_pt.y - min_pt.y);
          width = std::max(std::abs(max_pt.z - min_pt.z), std::abs(max_pt.x - min_pt.x));
        }
        // otherwise Z is closest, which is fine
        
        //}
        quat_t quat(rotation);

        // if the cluster is too large, just ignore it
        if (height > m_classif_max_detection_height || width > m_classif_max_detection_width)
        {
          ROS_INFO("[MainLoop]: Skipping too large cluster with height %.2f > %.2f or width %.2f > %.2f.", height, m_classif_max_detection_height, width,
                   m_classif_max_detection_width);
          continue;
        }

        cluster_class_t cclass = unknown;
        // if the cluster is larger than it could be jut for the MAV, it probably includes the ball as well
        if (height > m_classif_max_mav_height)
        {
          cclass = mav_with_ball;
        }
        // if the cluster is smaller in height than MAV + wire + ball, but closer then the threshold distance,
        // decide the class based on the size and shape of the cluster
        else if (dist < m_classif_close_dist)
        {
          const float width_thresh = m_classif_ball_width + std::abs(m_classif_mav_width - m_classif_ball_width) / 2.0f;
          if (width < width_thresh)
            cclass = ball;
          else
            cclass = mav;
        }
        // otherwise it's probably just a detection of the MAV - assume the ball is below it
        else
        {
          cclass = mav;
        }

        if (m_pub_detection_classified.getNumSubscribers() > 0)
          m_pub_detection_classified.publish(classified_detection_visualization(center, width, height, quat, cclass, dist, header));

        switch (cclass)
        {
          case cluster_class_t::ball:
          case cluster_class_t::mav_with_ball:
          {
            // find the lowest point in z - that will probably be the ball
            pt_XYZ_t min_z_pt(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max());
            for (const auto& pt : cluster->points)
              if (pt.z < min_z_pt.z)
                min_z_pt = pt;
            ballpos_result_opt = {min_z_pt, cluster};
            ballpos_n_pts = cluster->size();
          }
          break;
          case cluster_class_t::mav:
          {
            // find the highest point in z - that will probably be the MAV
            pt_XYZ_t max_z_pt(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::lowest());
            for (const auto& pt : cluster->points)
              if (pt.z > max_z_pt.z)
                max_z_pt = pt;
            // subtract length of the wire from the MAV position
            ballpos_result_opt = {{max_z_pt.x, max_z_pt.y, max_z_pt.z - m_classif_ball_wire_length}, cluster};
            ballpos_n_pts = cluster->size();
          }
          break;
          default:
            continue;
        }
      }
      return ballpos_result_opt;
    }
    //}

    /* global_to_arena() method //{ */
    std::tuple<int, int, int> global_to_arena(float glx, float gly, float glz)
    {
      const int x = std::floor(glx - m_arena_bbox_offset_x + m_arena_bbox_size_x / 2.0);
      const int y = std::floor(gly - m_arena_bbox_offset_y + m_arena_bbox_size_y / 2.0);
      const int z = std::floor(glz - m_arena_bbox_offset_z);
      return {x, y, z};
    }
    //}

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

    /* valid_arena_coordinates() method //{ */
    bool valid_arena_coordinates(const int x, const int y, const int z)
    {
      const bool x_ok = x >= 0 && x < m_arena_bbox_size_x;
      const bool y_ok = y >= 0 && y < m_arena_bbox_size_y;
      const bool z_ok = z >= 0 && z < m_arena_bbox_size_z;
      return x_ok && y_ok && z_ok;
    }
    //}

    /* to_eigen() method //{ */
    vec3_t to_eigen(const pt_XYZ_t& pt)
    {
      return {pt.x, pt.y, pt.z};
    }
    //}

    /* to_pcl() method //{ */
    pt_XYZ_t to_pcl(const vec3_t& pt)
    {
      return {pt.x(), pt.y(), pt.z()};
    }
    //}

    /* fit_plane() method //{ */

    std::optional<planefit_t> fit_plane(const pc_XYZt_t::ConstPtr cloud)
    {
      if (cloud->size() < (size_t)m_plane_fit_min_points)
      {
        ROS_WARN("[FitLocalplane]: Not enough points to fit plane, skipping (got %lu/%d)", cloud->size(), m_plane_fit_min_points);
        return std::nullopt;
      }

      // fit a plane to the global cloud points
      /*  //{ */

      auto model_l = boost::make_shared<pcl::SampleConsensusModelPlane<pt_XYZt_t>>(cloud);
      pcl::RandomSampleConsensus<pt_XYZt_t> fitter(model_l);
      /* fitter.setNumberOfThreads(4); */
      fitter.setDistanceThreshold(m_plane_fit_ransac_threshold);
      fitter.computeModel();
      Eigen::VectorXf params;
      fitter.getModelCoefficients(params);

      // check if the fit failed
      if (params.rows() == 0)
        return std::nullopt;
      assert(params.rows() == 4);

      std::vector<int> inliers;
      fitter.getInliers(inliers);

      //}

      const planefit_t ret = {params.block<3, 1>(0, 0), params(3)};
      return ret;
    }
    //}

    /* fit_local_line() method //{ */
    std::optional<linefit_t> fit_local_line(const int neighborhood, const int x_idx, const int y_idx, const int z_idx, const ros::Time& cur_stamp)
    {
      // find neighborhood points and their stamps
      // note that per one voxel, N points are added, where N is the weight of the voxel (its value in the map)
      /*  //{ */

      pc_XYZt_t::Ptr pc = boost::make_unique
      /* int n_unique_pts = 0; */
      /* pc_XYZ_t::Ptr line_pts = boost::make_shared<pc_XYZ_t>(); */
      /* line_pts->reserve(neighborhood * neighborhood * neighborhood); */
      /* std::vector<float> line_dts; */
      /* line_dts.reserve(neighborhood * neighborhood * neighborhood); */
      /* for (int x_it = std::max(x_idx - neighborhood, 0); x_it < std::min(x_idx + neighborhood, m_arena_bbox_size_x - 1); x_it++) */
      /* { */
      /*   for (int y_it = std::max(y_idx - neighborhood, 0); y_it < std::min(y_idx + neighborhood, m_arena_bbox_size_y - 1); y_it++) */
      /*   { */
      /*     for (int z_it = std::max(z_idx - neighborhood, 0); z_it < std::min(z_idx + neighborhood, m_arena_bbox_size_z - 1); z_it++) */
      /*     { */
      /*       const int mapval = std::ceil(map_at_coords(m_map3d, x_it, y_it, z_it)); */
      /*       const float dt = (map_at_coords(m_map3d_last_update, x_it, y_it, z_it) - cur_stamp).toSec(); */
      /*       const pt_XYZ_t pt{float(x_it), float(y_it), float(z_it)}; */
      /*       for (int it = 0; it < mapval; it++) */
      /*       { */
      /*         line_pts->push_back(pt); */
      /*         line_dts.push_back(dt); */
      /*       } */
      /*       if (mapval > 0) */
      /*         n_unique_pts++; */
      /*     } */
      /*   } */
      /* } */

      //}

      // TODO: parametrize this shit
      const int m_min_linefit_points = 3;
      if (n_unique_pts < m_min_linefit_points)
      {
        ROS_WARN("[FitLocalLine]: Not enough points to fit line, skipping (got %d/%d)", n_unique_pts, m_min_linefit_points);
        return std::nullopt;
      }

      // fit a line to the neighborhood points
      /*  //{ */

      auto model_l = boost::make_shared<pcl::SampleConsensusModelLine<pt_XYZ_t>>(line_pts);
      pcl::LeastMedianSquares<pt_XYZ_t> fitter(model_l);
      fitter.setDistanceThreshold(1);
      fitter.computeModel();
      Eigen::VectorXf params;
      fitter.getModelCoefficients(params);

      // check if the fit failed
      if (params.rows() == 0)
        return std::nullopt;
      assert(params.rows() == 6);

      std::vector<int> inliers;
      fitter.getInliers(inliers);

      //}

      // get the inlier dts and points
      /*  //{ */

      using dt_pt_t = linefit_t::dt_pt_t;
      const float max_point_age = neighborhood * m_linefit_point_max_age_coeff;  // seconds
      std::vector<dt_pt_t> in_dt_pts;
      for (const auto idx : inliers)
      {
        const float dt = line_dts.at(idx);
        // ignore too old points
        if (dt >= -max_point_age)
          in_dt_pts.push_back({dt, to_eigen(line_pts->at(idx))});
      }

      //}

      // sort the inliers by relative time to the main point
      /*  //{ */

      std::sort(std::begin(in_dt_pts), std::end(in_dt_pts),
                // comparison lambda function
                [](const dt_pt_t& a, const dt_pt_t& b) { return a.first < b.first; });

      //}

      // average points from the same time
      /*  //{ */

      {
        std::vector<dt_pt_t> tmp;
        tmp.reserve(in_dt_pts.size());
        dt_pt_t dt_pt_sum;
        size_t dt_pt_weight;
        bool dt_pt_initialized = false;
        for (const auto& dt_pt : in_dt_pts)
        {
          if (!dt_pt_initialized)
          {
            dt_pt_sum = dt_pt;
            dt_pt_weight = 1;
            dt_pt_initialized = true;
            continue;
          }

          const float dt_diff = dt_pt.first - dt_pt_sum.first;
          if (dt_diff == 0.0f)
          {
            dt_pt_sum.second += dt_pt.second;
            dt_pt_weight++;
            continue;
          }

          dt_pt_sum.second /= float(dt_pt_weight);
          tmp.push_back(dt_pt_sum);
          dt_pt_sum = dt_pt;
          dt_pt_weight = 1;
        }
        ROS_INFO("[YawEstimation]: Averaged points: %lu/%lu.", tmp.size(), in_dt_pts.size());
        in_dt_pts = tmp;
      }

      //}

      const linefit_t ret = {in_dt_pts, params};
      return ret;
    }
    //}

    /* estimate_yaw_from_map3d() method //{ */
    std::optional<float> estimate_yaw_from_map3d(const int x_idx, const int y_idx, const int z_idx, const ros::Time& cur_stamp, ros::Publisher& dbg_pub)
    {
      const auto linefit_opt = fit_local_line(m_linefit_neighborhood, x_idx, y_idx, z_idx, cur_stamp);
      if (!linefit_opt.has_value())
      {
        ROS_ERROR("[EstimateYaw]: Line fit failed!");
        return std::nullopt;
      }

      const auto& in_dt_pts = linefit_opt->dt_pts;

      // find the mean velocity between the points
      using dt_pt_t = linefit_t::dt_pt_t;
      vec3_t mean_vel(0, 0, 0);
      size_t mean_vel_weight = 0;
      dt_pt_t prev_dt_pt;
      bool prev_dt_pt_initialized = false;
      for (const auto& dt_pt : in_dt_pts)
      {
        if (!prev_dt_pt_initialized)
        {
          prev_dt_pt = dt_pt;
          prev_dt_pt_initialized = true;
          continue;
        }

        const float cur_dt = dt_pt.first - prev_dt_pt.first;
        const auto cur_pt = dt_pt.second;
        assert(cur_dt != 0.0f);

        const vec3_t cur_vel = (cur_pt - prev_dt_pt.second) / cur_dt;
        mean_vel += cur_vel;
        mean_vel_weight += 1;
        prev_dt_pt.second = dt_pt.second;
      }

      // if no velocity estimate could be obtained from the inliers, return
      if (mean_vel.isZero() || mean_vel_weight == 0)
      {
        ROS_WARN("[EstimateYaw]: Unable to estimate speed from inliers!");
        return std::nullopt;
      }
      mean_vel /= mean_vel_weight;

      // check if the estimated speed makes sense
      const double est_speed = mean_vel.norm();
      if (std::abs(est_speed - m_ball_speed) > m_max_speed_error)
      {
        ROS_ERROR("[EstimateYaw]: Object speed is too different from the expected speed (%.2fm/s vs %.2fm/s)!", est_speed, m_ball_speed);
        return std::nullopt;
      }

      // if requested, publish the neighborhood for debugging
      /*  //{ */

      if (dbg_pub.getNumSubscribers() > 0)
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
        dbg_pub.publish(cloud_out);
      }

      //}

      // use the estimated velocity to set the luring pose orientation
      const float yaw = std::atan2(mean_vel.y(), mean_vel.x());
      return yaw;
    }
    //}

    /* estimate_yaw() method //{ */
    std::optional<float> estimate_yaw(const int x_idx, const int y_idx, const int z_idx, const ros::Time& cur_stamp, ros::Publisher& dbg_pub)
    {
      const auto linefit_opt = fit_local_line(m_linefit_neighborhood, x_idx, y_idx, z_idx, cur_stamp);
      if (!linefit_opt.has_value())
      {
        ROS_ERROR("[EstimateYaw]: Line fit failed!");
        return std::nullopt;
      }

      const auto& in_dt_pts = linefit_opt->dt_pts;

      // find the mean velocity between the points
      using dt_pt_t = linefit_t::dt_pt_t;
      vec3_t mean_vel(0, 0, 0);
      size_t mean_vel_weight = 0;
      dt_pt_t prev_dt_pt;
      bool prev_dt_pt_initialized = false;
      for (const auto& dt_pt : in_dt_pts)
      {
        if (!prev_dt_pt_initialized)
        {
          prev_dt_pt = dt_pt;
          prev_dt_pt_initialized = true;
          continue;
        }

        const float cur_dt = dt_pt.first - prev_dt_pt.first;
        const auto cur_pt = dt_pt.second;
        assert(cur_dt != 0.0f);

        const vec3_t cur_vel = (cur_pt - prev_dt_pt.second) / cur_dt;
        mean_vel += cur_vel;
        mean_vel_weight += 1;
        prev_dt_pt.second = dt_pt.second;
      }

      // if no velocity estimate could be obtained from the inliers, return
      if (mean_vel.isZero() || mean_vel_weight == 0)
      {
        ROS_WARN("[EstimateYaw]: Unable to estimate speed from inliers!");
        return std::nullopt;
      }
      mean_vel /= mean_vel_weight;

      // check if the estimated speed makes sense
      const double est_speed = mean_vel.norm();
      if (std::abs(est_speed - m_ball_speed) > m_max_speed_error)
      {
        ROS_ERROR("[EstimateYaw]: Object speed is too different from the expected speed (%.2fm/s vs %.2fm/s)!", est_speed, m_ball_speed);
        return std::nullopt;
      }

      // if requested, publish the neighborhood for debugging
      /*  //{ */

      if (dbg_pub.getNumSubscribers() > 0)
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
        dbg_pub.publish(cloud_out);
      }

      //}

      // use the estimated velocity to set the luring pose orientation
      const float yaw = std::atan2(mean_vel.y(), mean_vel.x());
      return yaw;
    }
    //}

    /* find_detection_pose() method //{ */
    std::optional<geometry_msgs::PoseStamped> find_detection_pose(const pt_XYZ_t& pos, const std_msgs::Header& header)
    {
      geometry_msgs::PoseStamped ret;
      ret.header = header;

      const auto [pt_arena_x, pt_arena_y, pt_arena_z] = global_to_arena(pos.x, pos.y, pos.z);

      const auto yaw_opt = estimate_yaw(pt_arena_x, pt_arena_y, pt_arena_z, header.stamp, m_pub_detection_neighborhood);
      if (!yaw_opt.has_value())
        return std::nullopt;

      const float yaw = yaw_opt.value();
      const Eigen::AngleAxisf anax(yaw, vec3_t::UnitZ());
      const Eigen::Quaternionf quat(anax);
      ret.pose.orientation.w = quat.w();
      ret.pose.orientation.x = quat.x();
      ret.pose.orientation.y = quat.y();
      ret.pose.orientation.z = quat.z();

      ret.pose.position.x = pos.x;
      ret.pose.position.y = pos.y;
      ret.pose.position.z = pos.z;

      return ret;
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

      const auto yaw_opt = estimate_yaw_from_map3d(max_x, max_y, max_z, max_stamp, m_pub_chosen_neighborhood);
      if (!yaw_opt.has_value())
        return std::nullopt;

      // use the estimated velocity to set the luring pose orientation
      const float yaw = yaw_opt.value();
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

  /* plane_origin() method //{ */
  vec3_t plane_origin(const planefit_t& plane, const vec3_t& origin)
  {
    const static double eps = 1e-9;
    const double a = plane.normal(0);
    const double b = plane.normal(1);
    const double c = plane.normal(2);
    const double d = plane.offset;
    const double x = origin.x();
    const double y = origin.y();
    const double z = origin.z();
    vec3_t ret(x, y, z);
    if (abs(a) > eps)
      ret(0) = -(y * b + z * c + d) / a;
    else if (abs(b) > eps)
      ret(1) = -(x * a + z * c + d) / b;
    else if (abs(c) > eps)
      ret(2) = -(x * a + y * b + d) / c;
    return ret;
  }
  //}

  /* plane_visualization() method //{ */
  visualization_msgs::MarkerArray plane_visualization(const planefit_t& plane, const std_msgs::Header& header, const vec3_t& origin)
  {
    visualization_msgs::MarkerArray ret;

    const auto pos = plane_origin(plane, origin);
    const auto quat = quat_t::FromTwoVectors(vec3_t::UnitZ(), plane.normal);

    //TODO: parametrize this shit
    const double m_plane_visualization_size = 100.0;
    const double size = m_plane_visualization_size;
    geometry_msgs::Point ptA;
    ptA.x = size;
    ptA.y = size;
    ptA.z = 0;
    geometry_msgs::Point ptB;
    ptB.x = -size;
    ptB.y = size;
    ptB.z = 0;
    geometry_msgs::Point ptC;
    ptC.x = -size;
    ptC.y = -size;
    ptC.z = 0;
    geometry_msgs::Point ptD;
    ptD.x = size;
    ptD.y = -size;
    ptD.z = 0;

    /* borders marker //{ */
    {
      visualization_msgs::Marker borders_marker;
      borders_marker.header = header;

      borders_marker.ns = "borders";
      borders_marker.id = 0;
      borders_marker.type = visualization_msgs::Marker::LINE_LIST;
      borders_marker.action = visualization_msgs::Marker::ADD;

      borders_marker.pose.position.x = pos.x();
      borders_marker.pose.position.y = pos.y();
      borders_marker.pose.position.z = pos.z();

      borders_marker.pose.orientation.x = quat.x();
      borders_marker.pose.orientation.y = quat.y();
      borders_marker.pose.orientation.z = quat.z();
      borders_marker.pose.orientation.w = quat.w();

      borders_marker.scale.x = 0.1;
      borders_marker.scale.y = 0.1;
      borders_marker.scale.z = 0.1;

      borders_marker.color.a = 0.5;  // Don't forget to set the alpha!
      borders_marker.color.r = 0.0;
      borders_marker.color.g = 0.0;
      borders_marker.color.b = 1.0;

      borders_marker.points.push_back(ptA);
      borders_marker.points.push_back(ptB);

      borders_marker.points.push_back(ptB);
      borders_marker.points.push_back(ptC);

      borders_marker.points.push_back(ptC);
      borders_marker.points.push_back(ptD);

      borders_marker.points.push_back(ptD);
      borders_marker.points.push_back(ptA);

      ret.markers.push_back(borders_marker);
    }
    //}

    /* plane marker //{ */
    {
      visualization_msgs::Marker plane_marker;
      plane_marker.header = header;

      plane_marker.ns = "plane";
      plane_marker.id = 1;
      plane_marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
      plane_marker.action = visualization_msgs::Marker::ADD;

      plane_marker.pose.position.x = pos.x();
      plane_marker.pose.position.y = pos.y();
      plane_marker.pose.position.z = pos.z();

      plane_marker.pose.orientation.x = quat.x();
      plane_marker.pose.orientation.y = quat.y();
      plane_marker.pose.orientation.z = quat.z();
      plane_marker.pose.orientation.w = quat.w();

      plane_marker.scale.x = 1;

      plane_marker.color.a = 0.2;  // Don't forget to set the alpha!
      plane_marker.color.r = 0.0;
      plane_marker.color.g = 0.0;
      plane_marker.color.b = 1.0;

      // triangle ABC
      plane_marker.points.push_back(ptA);
      plane_marker.points.push_back(ptB);
      plane_marker.points.push_back(ptC);

      // triangle ACD
      plane_marker.points.push_back(ptA);
      plane_marker.points.push_back(ptC);
      plane_marker.points.push_back(ptD);
      ret.markers.push_back(plane_marker);
    }
    //}

    return ret;
  }
  //}

    /* classified_detection_visualization() method //{ */
    visualization_msgs::MarkerArray classified_detection_visualization(const vec3_t& center, const float width, const float height, const quat_t& quat,
                                                                       const cluster_class_t cclass, const float distance, const std_msgs::Header& header)
    {
      visualization_msgs::MarkerArray ret;

      {
        visualization_msgs::Marker bcyl;
        /* fill the bounding cylinder marker //{ */

        bcyl.header = header;
        bcyl.color.a = 0.2;
        switch (cclass)
        {
          case cluster_class_t::ball:
            bcyl.color.b = 1.0;
            break;
          case cluster_class_t::mav:
            bcyl.color.r = 1.0;
            break;
          case cluster_class_t::mav_with_ball:
            bcyl.color.g = 1.0;
            break;
          default:
            bcyl.color.b = 1.0;
            bcyl.color.g = 1.0;
            break;
        }
        bcyl.scale.x = width;
        bcyl.scale.y = width;
        bcyl.scale.z = height;
        bcyl.ns = "bounding cylinder";
        bcyl.type = visualization_msgs::Marker::CYLINDER;
        bcyl.pose.orientation.w = quat.w();
        bcyl.pose.orientation.x = quat.x();
        bcyl.pose.orientation.y = quat.y();
        bcyl.pose.orientation.z = quat.z();
        bcyl.pose.position.x = center.x();
        bcyl.pose.position.y = center.y();
        bcyl.pose.position.z = center.z();

        //}
        ret.markers.push_back(bcyl);
      }

      {
        visualization_msgs::Marker arr;
        /* fill the axis arrow marker //{ */

        arr.header = header;
        arr.color.a = 1.0;
        arr.color.r = 1.0;
        arr.scale.x = 0.1;
        arr.scale.y = 0.3;
        arr.ns = "axis x";
        arr.type = visualization_msgs::Marker::ARROW;
        arr.pose.orientation.w = quat.w();
        arr.pose.orientation.x = quat.x();
        arr.pose.orientation.y = quat.y();
        arr.pose.orientation.z = quat.z();
        arr.pose.position.x = center.x();
        arr.pose.position.y = center.y();
        arr.pose.position.z = center.z();

        geometry_msgs::Point pt;
        arr.points.push_back(pt);
        pt.x = 1.0;
        arr.points.push_back(pt);
        //}
        ret.markers.push_back(arr);
      }

      {
        visualization_msgs::Marker arr;
        /* fill the bounding cylinder marker //{ */

        arr.header = header;
        arr.color.a = 1.0;
        arr.color.g = 1.0;
        arr.scale.x = 0.1;
        arr.scale.y = 0.3;
        arr.ns = "axis y";
        arr.type = visualization_msgs::Marker::ARROW;
        arr.pose.orientation.w = quat.w();
        arr.pose.orientation.x = quat.x();
        arr.pose.orientation.y = quat.y();
        arr.pose.orientation.z = quat.z();
        arr.pose.position.x = center.x();
        arr.pose.position.y = center.y();
        arr.pose.position.z = center.z();

        geometry_msgs::Point pt;
        arr.points.push_back(pt);
        pt.y = 1.0;
        arr.points.push_back(pt);
        //}
        ret.markers.push_back(arr);
      }

      {
        visualization_msgs::Marker arr;
        /* fill the bounding cylinder marker //{ */

        arr.header = header;
        arr.color.a = 1.0;
        arr.color.b = 1.0;
        arr.scale.x = 0.1;
        arr.scale.y = 0.3;
        arr.ns = "axis z";
        arr.type = visualization_msgs::Marker::ARROW;
        arr.pose.orientation.w = quat.w();
        arr.pose.orientation.x = quat.x();
        arr.pose.orientation.y = quat.y();
        arr.pose.orientation.z = quat.z();
        arr.pose.position.x = center.x();
        arr.pose.position.y = center.y();
        arr.pose.position.z = center.z();

        geometry_msgs::Point pt;
        arr.points.push_back(pt);
        pt.z = 1.0;
        arr.points.push_back(pt);
        //}
        ret.markers.push_back(arr);
      }

      {
        visualization_msgs::Marker disttxt;
        /* fill the distance text marker //{ */

        disttxt.header = header;
        disttxt.color.a = 1.0;
        if (distance > m_classif_close_dist)
          disttxt.color.b = 1.0;
        else
          disttxt.color.r = 1.0;
        disttxt.scale.z = 1.0;
        disttxt.ns = "distance text";
        disttxt.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        disttxt.pose.orientation.w = quat.w();
        disttxt.pose.orientation.x = quat.x();
        disttxt.pose.orientation.y = quat.y();
        disttxt.pose.orientation.z = quat.z();
        disttxt.pose.position.x = center.x();
        disttxt.pose.position.y = center.y();
        disttxt.pose.position.z = center.z();
        disttxt.text = std::to_string(distance);

        //}
        ret.markers.push_back(disttxt);
      }

      return ret;
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

    ros::Publisher m_pub_map3d;
    ros::Publisher m_pub_map3d_bounds;
    ros::Publisher m_pub_oparea;

    ros::Publisher m_pub_filtered_input_pc;
    ros::Publisher m_pub_detections_pc;

    ros::Publisher m_pub_detection_classified;
    ros::Publisher m_pub_detection;
    ros::Publisher m_pub_detection_neighborhood;

    ros::Publisher m_pub_chosen_neighborhood;
    ros::Publisher m_pub_chosen_position;

    ros::Publisher m_pub_plane;

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

    double m_ball_speed;       // metres per second
    double m_max_speed_error;  // metres per second
    int m_linefit_neighborhood;
    float m_linefit_point_max_age_coeff;

    float m_classif_max_detection_height;
    float m_classif_max_detection_width;
    float m_classif_max_mav_height;
    float m_classif_ball_wire_length;

    float m_classif_close_dist;
    float m_classif_mav_width;
    float m_classif_ball_width;

    int m_plane_fit_min_points;
    float m_plane_fit_ransac_threshold;

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
    octree::Ptr m_global_octree;
    pc_XYZt_t::Ptr m_global_cloud;  // contains position estimates of the ball
    std::vector<ros::Time> m_map3d_last_update;

    ros::Time m_start_time;

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
