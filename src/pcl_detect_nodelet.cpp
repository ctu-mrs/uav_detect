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
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <uav_detect/DetectionParamsConfig.h>

#include <eigen_conversions/eigen_msg.h>
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

  using vec2_t = Eigen::Vector2d;
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

      pc_XYZt_t::Ptr inlier_pts;
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

    /* ball_det_t //{ */
    
    enum cluster_class_t
    {
      mav,
      ball,
      mav_with_ball,
      unknown
    };

    struct ball_det_t
    {
      using cov_t = Eigen::Matrix3f;
      pt_XYZ_t pos_stamped;
      pc_XYZ_t::Ptr cloud;
      cov_t cov;
      cluster_class_t cclass;
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
      
      // CHECK LOADING STATUS
      if (!m_drmgr_ptr->loaded_successfully())
      {
        NODELET_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
        ros::shutdown();
      }

      mrs_lib::ParamLoader pl(nh, m_node_name);
      // LOAD STATIC PARAMETERS
      NODELET_INFO("Loading static parameters:");
      const auto uav_name = pl.loadParam2<std::string>("uav_name");
      pl.loadParam("world_frame_id", m_world_frame_id);

      pl.loadParam("max_speed_error", m_max_speed_error);

      pl.loadParam("linefit/neighborhood", m_linefit_neighborhood);
      pl.loadParam("linefit/point_max_age", m_linefit_point_max_age);

      pl.loadParam("classification/max_detection_height", m_classif_max_detection_height);
      pl.loadParam("classification/max_detection_width", m_classif_max_detection_width);
      pl.loadParam("classification/max_mav_height", m_classif_max_mav_height);
      pl.loadParam("classification/ball_wire_length", m_classif_ball_wire_length);
      pl.loadParam("classification/close_distance", m_classif_close_dist);
      pl.loadParam("classification/mav_width", m_classif_mav_width);
      pl.loadParam("classification/ball_width", m_classif_ball_width);

      pl.loadParam("exclude_box/offset/x", m_exclude_box_offset_x);
      pl.loadParam("exclude_box/offset/y", m_exclude_box_offset_y);
      pl.loadParam("exclude_box/offset/z", m_exclude_box_offset_z);
      pl.loadParam("exclude_box/size/x", m_exclude_box_size_x);
      pl.loadParam("exclude_box/size/y", m_exclude_box_size_y);
      pl.loadParam("exclude_box/size/z", m_exclude_box_size_z);

      /* load safety area //{ */

      pl.loadParam("safety_area/deflation", m_safety_area_deflation);
      pl.loadParam("safety_area/height/min", m_operation_area_min_z);
      pl.loadParam("safety_area/height/max", m_operation_area_max_z);
      m_safety_area_frame = pl.loadParam2<std::string>("safety_area/frame_name");
      m_safety_area_border_points = pl.loadMatrixDynamic2("safety_area/safety_area", -1, 2);
      pl.loadMatrixStatic<2, 1>("safety_area/left_cut_plane/normal", m_safety_area_left_cut_plane_normal);
      pl.loadParam("safety_area/left_cut_plane/offset", m_safety_area_left_cut_plane_offset);
      pl.loadMatrixStatic<2, 1>("safety_area/right_cut_plane/normal", m_safety_area_right_cut_plane_normal);
      pl.loadParam("safety_area/right_cut_plane/offset", m_safety_area_right_cut_plane_offset);
      pl.loadMatrixStatic<2, 1>("safety_area/offset", m_safety_area_offset);

      m_safety_area_init_timer = nh.createTimer(ros::Duration(1.0), &PCLDetector::init_safety_area, this);

      //}
      
      // CHECK LOADING STATUS
      if (!pl.loadedSuccessfully())
      {
        NODELET_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
        ros::shutdown();
      }
      //}

      /* Create publishers and subscribers //{ */
      // Initialize transform listener
      m_tf_listener_ptr = std::make_unique<tf2_ros::TransformListener>(m_tf_buffer);
      // Initialize subscribers
      mrs_lib::SubscribeHandlerOptions shopts;
      shopts.nh = nh;
      shopts.node_name = m_node_name;
      shopts.no_message_timeout = ros::Duration(5.0);

      mrs_lib::construct_object(m_sh_pc, shopts, "pc");
      mrs_lib::construct_object(m_sh_ball_speed, shopts, "gt_ball_speed");
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
      m_pub_detection = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("detection", 1);
      m_pub_detection_classified = nh.advertise<visualization_msgs::MarkerArray>("detection_classified", 1);
      m_pub_detection_neighborhood = nh.advertise<sensor_msgs::PointCloud2>("detection_neighborhood", 1, true);
      m_pub_lidar_fov = nh.advertise<visualization_msgs::Marker>("lidar_fov", 1, true);

      m_reset_server = nh.advertiseService("reset", &PCLDetector::reset_callback, this);
      //}

      /* initialize transformer //{ */

      m_transformer = mrs_lib::Transformer(m_node_name, uav_name);

      //}

      m_global_cloud = boost::make_shared<pc_XYZt_t>();
      reset();
      m_last_detection_id = 0;

      m_det_blobs = 0;
      m_images_processed = 0;
      m_avg_fps = 0.0f;
      m_avg_delay = 0.0f;

      m_main_loop_timer = nh.createTimer(ros::Rate(1000), &PCLDetector::main_loop, this);

      cout << "----------------------------------------------------------" << std::endl;
    }
    //}

  /* reset_callback() method //{ */

  bool reset_callback([[maybe_unused]] std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& resp)
  {
    reset();
    resp.message = "Detector reset.";
    resp.success = true;
    return true;
  }

  //}

  private:

    /* main_loop() method //{ */
    void main_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      // publish the lidar FOV marker (only once)
      /*  //{ */
      
      if (m_sh_pc.hasMsg() && !m_sh_pc.usedMsg())
      {
        const auto msg_ptr = m_sh_pc.peekMsg();
        std_msgs::Header header;
        header.frame_id = msg_ptr->header.frame_id;
        header.stamp = ros::Time::now();
        // TODO: parametrize this shit
        constexpr double hfov = 33.2;
        constexpr double range = 40.0;
        const auto msg = lidar_visualization(hfov, range, header);
        m_pub_lidar_fov.publish(msg);
      }
      
      //}

      if (!m_safety_area_initialized)
      {
        NODELET_WARN_STREAM_THROTTLE(1.0, "[MainLoop]: Safety area not initialized, skipping. ");
        return;
      }

      std::scoped_lock lck(m_reset_mtx);

      /* load covariance coefficients from dynparam //{ */
      
      {
        const std::pair<double&, double&> cov_coeffs_mav =
          {m_drmgr_ptr->config.cov_coeffs__xy__mav, m_drmgr_ptr->config.cov_coeffs__z__mav};
        const std::pair<double&, double&> cov_coeffs_ball =
          {m_drmgr_ptr->config.cov_coeffs__xy__ball, m_drmgr_ptr->config.cov_coeffs__z__ball};
        const std::pair<double&, double&> cov_coeffs_mav_with_ball =
          {m_drmgr_ptr->config.cov_coeffs__xy__mav_with_ball, m_drmgr_ptr->config.cov_coeffs__z__mav_with_ball};
        m_cov_coeffs.insert_or_assign(cluster_class_t::mav, cov_coeffs_mav);
        m_cov_coeffs.insert_or_assign(cluster_class_t::ball, cov_coeffs_ball );
        m_cov_coeffs.insert_or_assign(cluster_class_t::mav_with_ball, cov_coeffs_mav_with_ball);
        m_cov_coeff_vel = m_drmgr_ptr->config.cov_coeffs__xyz__velocity;
      }
      
      //}

      if (m_sh_pc.newMsg())
      {
        const ros::WallTime start_t = ros::WallTime::now();

        NODELET_INFO_STREAM("[MainLoop]: Processing new data --------------------------------------------------------- ");

        pc_XYZ_t::ConstPtr cloud = m_sh_pc.getMsg();
        ros::Time msg_stamp;
        pcl_conversions::fromPCL(cloud->header.stamp, msg_stamp);
        std::string cloud_frame_id = cloud->header.frame_id;  // cut off the first forward slash
        if (cloud_frame_id.at(0) == '/')
          cloud_frame_id = cloud_frame_id.substr(1);  // cut off the first forward slash
        NODELET_INFO_STREAM("[MainLoop]: Input PC has " << cloud->size() << " points");

        /* filter input cloud and transform it to world //{ */

        pc_XYZ_t::Ptr cloud_filtered = boost::make_shared<pc_XYZ_t>(*cloud);
        cloud_filtered->header = cloud->header;
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
          ec.setClusterTolerance(m_classif_max_detection_height);
          ec.setMinClusterSize(m_min_cluster_size);
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

        std::optional<ball_det_t> ball_det_result_opt = find_ball_position(cloud_clusters, tf_trans, header);

        if (ball_det_result_opt.has_value())
        {
          const ball_det_t detection = ball_det_result_opt.value();
          const auto ballpos = detection.pos_stamped;
          const float cur_time = (msg_stamp - m_start_time).toSec();
          pt_XYZt_t ballpos_stamped;
          ballpos_stamped.x = ballpos.x;
          ballpos_stamped.y = ballpos.y;
          ballpos_stamped.z = ballpos.z;
          ballpos_stamped.intensity = cur_time;
          m_global_octree->addPointToCloud(ballpos_stamped, m_global_cloud);

          // update voxels in the frequency map and stamp map
          {
            const auto cloud_used = detection.cloud;
            const auto [pt_arena_x, pt_arena_y, pt_arena_z] = global_to_arena(ballpos.x, ballpos.y, ballpos.z);
            if (valid_arena_coordinates(pt_arena_x, pt_arena_y, pt_arena_z))
            {
              map_at_coords(m_map3d, pt_arena_x, pt_arena_y, pt_arena_z) += cloud_used->size();
              map_at_coords(m_map3d_last_update, pt_arena_x, pt_arena_y, pt_arena_z) = cur_time;
            }
            else
            {
              NODELET_WARN("[MainLoop]: Ball seems to be out of arena bounds. Skipping.");
            }
          }

          // find the velocity direction as well, if possible
          const auto detection_pose_opt = find_detection_pose(ballpos_stamped, m_pub_detection_neighborhood);

          // prepare the output message
          geometry_msgs::PoseWithCovarianceStamped msg;
          msg.header = header;
          for (auto& el : msg.pose.covariance)
            el = 0.0;
          set_pos_cov(detection.cov, msg.pose.covariance);
          if (detection_pose_opt.has_value())
          {
            msg.pose.pose = detection_pose_opt.value();
            Eigen::Matrix3f rot_cov = m_cov_coeff_vel*Eigen::Matrix3f::Identity();
            set_rot_cov(rot_cov, msg.pose.covariance);
          }
          else
          {
            msg.pose.pose.position.x = ballpos.x;
            msg.pose.pose.position.y = ballpos.y;
            msg.pose.pose.position.z = ballpos.z;
            Eigen::Matrix3f rot_cov = M_PI*Eigen::Matrix3f::Identity();
            set_rot_cov(rot_cov, msg.pose.covariance);
          }
          m_pub_detection.publish(msg);

          {
            sensor_msgs::PointCloud2 pc;
            sensor_msgs::PointCloud2Modifier pc2mod(pc);
            pc2mod.setPointCloud2Fields(4,
                "x", 1, sensor_msgs::PointField::FLOAT32,
                "y", 1, sensor_msgs::PointField::FLOAT32,
                "z", 1, sensor_msgs::PointField::FLOAT32,
                "class", 1, sensor_msgs::PointField::INT32
                );
            pc2mod.resize(1);
            sensor_msgs::PointCloud2Iterator<float> iter_x(pc, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(pc, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(pc, "z");
            sensor_msgs::PointCloud2Iterator<int32_t> iter_class(pc, "class");
            *iter_x = ballpos.x;
            *iter_y = ballpos.y;
            *iter_z = ballpos.z;
            *iter_class = detection.cclass;
            pc.header = header;
            m_pub_detections_pc.publish(pc);
          }
        }
        else // no detection - publish an empty message
        {
          sensor_msgs::PointCloud2 pc;
          sensor_msgs::PointCloud2Modifier pc2mod(pc);
          pc2mod.setPointCloud2Fields(4,
              "x", 1, sensor_msgs::PointField::FLOAT32,
              "y", 1, sensor_msgs::PointField::FLOAT32,
              "z", 1, sensor_msgs::PointField::FLOAT32,
              "class", 1, sensor_msgs::PointField::INT32
              );
          pc2mod.resize(0);
          pc.header = header;
          m_pub_detections_pc.publish(pc);
        }

        // find and publish the most probable pose the ball will pass through again from the frequency map
        const auto most_probable_passthrough_opt = find_most_probable_passthrough();
        if (most_probable_passthrough_opt.has_value())
        {
          geometry_msgs::PoseStamped msg;
          msg.header = header;
          msg.pose = most_probable_passthrough_opt.value();
          m_pub_chosen_position.publish(msg);
        }

        // publish some debug shit
        if (m_pub_filtered_input_pc.getNumSubscribers() > 0)
          m_pub_filtered_input_pc.publish(cloud_filtered);
        if (m_pub_map3d.getNumSubscribers() > 0)
          m_pub_map3d.publish(map3d_visualization(header));

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
      {
        std::scoped_lock lck(m_reset_mtx);
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
        m_operation_area_ring = ring(std::begin(boost_area_pts), std::end(boost_area_pts));
        boost::geometry::correct(m_operation_area_ring);
        
        // Declare strategies
        const int points_per_circle = 36;
        boost::geometry::strategy::buffer::distance_symmetric<double> distance_strategy(-m_safety_area_deflation);
        boost::geometry::strategy::buffer::join_round join_strategy(points_per_circle);
        boost::geometry::strategy::buffer::end_round end_strategy(points_per_circle);
        boost::geometry::strategy::buffer::point_circle circle_strategy(points_per_circle);
        boost::geometry::strategy::buffer::side_straight side_strategy;
        
        // some helper variables
        polygon tmp;
        tmp.outer() = (m_operation_area_ring);
        mpolygon input;
        input.push_back(tmp);
        mpolygon result;
        polygon poly;
        box bbox;

        // Deflate the safety area plygon
        boost::geometry::buffer(input, result, distance_strategy, side_strategy, join_strategy, end_strategy, circle_strategy);
        if (result.empty())
        {
          ROS_ERROR("[InitSafetyArea]: Deflated safety area is empty! This probably shouldn't happen!");
          return;
        }
        if (result.size() > 1)
          ROS_WARN("[InitSafetyArea]: Deflated safety area breaks into multiple pieces! This probably shouldn't happen! Using the first piece...");
        poly = result.at(0);
        m_operation_area_ring = poly.outer();

        // cut the polygon from left and right
        // WARNING: prasarny nasleduji
        const vec2_t left_normal = m_safety_area_left_cut_plane_normal.normalized();
        const vec2_t left_dir(left_normal.y(), -left_normal.x());
        const vec2_t left_pos = m_safety_area_left_cut_plane_offset*left_normal;
        const vec2_t left_pt1 = left_pos + 1000.0*left_dir;
        const vec2_t left_pt2 = left_pos - 1000.0*left_dir;
        const point left_pt1b(left_pt1.x(), left_pt1.y());
        const point left_pt2b(left_pt2.x(), left_pt2.y());

        const vec2_t right_normal = m_safety_area_right_cut_plane_normal.normalized();
        const vec2_t right_dir(right_normal.y(), -right_normal.x());
        const vec2_t right_pos = m_safety_area_right_cut_plane_offset*right_normal;
        const vec2_t right_pt1 = right_pos + 1000.0*right_dir;
        const vec2_t right_pt2 = right_pos - 1000.0*right_dir;
        const point right_pt1b(right_pt1.x(), right_pt1.y());
        const point right_pt2b(right_pt2.x(), right_pt2.y());

        ring mask_ring;
        mask_ring.push_back(left_pt1b);
        mask_ring.push_back(right_pt1b);
        mask_ring.push_back(right_pt2b);
        mask_ring.push_back(left_pt2b);
        boost::geometry::correct(mask_ring);

        result.clear();
        boost::geometry::intersection(m_operation_area_ring, mask_ring, result);
        if (result.empty())
        {
          ROS_ERROR("[InitSafetyArea]: Cutted safety area is empty! This probably shouldn't happen!");
          return;
        }
        if (result.size() > 1)
          ROS_WARN("[InitSafetyArea]: Cutted safety area breaks into multiple pieces! This probably shouldn't happen! Using the first piece...");
        poly = result.at(0);
        m_operation_area_ring = poly.outer();

        // apply the safety area offset
        point offset(m_safety_area_offset.x(), m_safety_area_offset.y());
        for (auto& pt : m_operation_area_ring)
          boost::geometry::add_point(pt, offset);
        
        boost::geometry::envelope(m_operation_area_ring, bbox);
        m_arena_bbox_size_x = std::ceil(std::abs(bbox.max_corner().x() - bbox.min_corner().x()));
        m_arena_bbox_size_y = std::ceil(std::abs(bbox.max_corner().y() - bbox.min_corner().y()));
        m_arena_bbox_size_z = std::ceil(std::abs(m_operation_area_max_z - m_operation_area_min_z));
        m_arena_bbox_offset_x = (bbox.max_corner().x() + bbox.min_corner().x()) / 2.0;
        m_arena_bbox_offset_y = (bbox.max_corner().y() + bbox.min_corner().y()) / 2.0;
        m_arena_bbox_offset_z = std::min(m_operation_area_max_z, m_operation_area_min_z);
        
        m_map_size = m_arena_bbox_size_x * m_arena_bbox_size_y * m_arena_bbox_size_z;
        ROS_INFO("[InitSafetyArea]: Arena initialized with bounding box size [%d, %d, %d] (%d voxels).", m_arena_bbox_size_x, m_arena_bbox_size_y,
                 m_arena_bbox_size_z, m_map_size);
        
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

      // reset/initialize arena map, detection pcls etc.
      reset();
    }
    //}

  private:
  /* reset() method //{ */

  void reset()
  {
    std::scoped_lock lck(m_reset_mtx);
    m_global_cloud->clear();
    m_global_cloud->header.frame_id = m_world_frame_id;
    m_global_octree = boost::make_shared<octree>(1.0); // 1 metre resolution
    m_global_octree->setInputCloud(m_global_cloud);
    m_start_time = ros::Time::now();
    ROS_WARN("[PCLDetector]: Global pointcloud, octree, and time reset!");

    if (m_safety_area_initialized)
    {
      // initialize the frequency map
      m_map3d.resize(m_map_size);
      for (int it = 0; it < m_map_size; it++)
        m_map3d.at(it) = 0;

      // initialize the stamp map
      m_map3d_last_update.resize(m_map_size);
      for (int it = 0; it < m_map_size; it++)
        m_map3d_last_update.at(it) = std::numeric_limits<float>::lowest();

      ROS_WARN("[PCLDetector]: Arena map reset!");
    }
  }

  //}

    /* find_ball_position() method //{ */
    ball_det_t::cov_t create_covariance(const cluster_class_t cclass)
    {
      if (m_cov_coeffs.find(cclass) == std::end(m_cov_coeffs))
      {
        ROS_ERROR("[PCLDetector]: Invalid cluster class %d! Covariance will be invalid.", cclass);
        return std::numeric_limits<float>::quiet_NaN()*ball_det_t::cov_t::Ones();
      }
      auto [xy_covariance_coeff, z_covariance_coeff] = m_cov_coeffs.at(cclass);
      ball_det_t::cov_t cov = ball_det_t::cov_t::Zero();
      cov(0, 0) = cov(1, 1) = xy_covariance_coeff;
      cov(2, 2) = z_covariance_coeff;
      return cov;
    }

    // finds the largest cluster, classified as a detection, and the corresponding ball position
    std::optional<ball_det_t> find_ball_position(
        const std::vector<pc_XYZ_t::Ptr>& cloud_clusters,
        const vec3_t& cur_pos,
        const std_msgs::Header& header
        )
    {
      std::optional<ball_det_t> ballpos_result_opt;
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
        const vec3_t center = center_pt.getVector3fMap();
        float height = std::abs(max_pt.z - min_pt.z);
        float width = std::max(std::abs(max_pt.x - min_pt.x), std::abs(max_pt.y - min_pt.y));
        const float dist = (center - cur_pos).norm();

        // make sure that the rotated z-axis is the closest to the original z-axis
        /*  //{ */
        
        {
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
        }
        
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

        ball_det_t::cov_t covariance = rotate_covariance(create_covariance(cclass), rotation);
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
            ballpos_result_opt = {min_z_pt, cluster, covariance, cclass};
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
            ballpos_result_opt = {{max_z_pt.x, max_z_pt.y, max_z_pt.z - m_classif_ball_wire_length}, cluster, covariance, cclass};
            ballpos_n_pts = cluster->size();
          }
          break;
          default:
            continue;
        }
      }
      // only return detections in the operation area to avoid overflows of the voxelmap
      if (ballpos_result_opt.has_value())
      {
        const auto ptt = ballpos_result_opt.value().pos_stamped;
        pt_XYZ_t pt;
        pt.x = ptt.x;
        pt.y = ptt.y;
        pt.z = ptt.z;
        if (in_operation_area(pt))
          return ballpos_result_opt;
        else
          return std::nullopt;
      }
      else
        return std::nullopt;
    }
    //}

    /* set_cov() methods //{ */
    using msg_cov_t = geometry_msgs::PoseWithCovarianceStamped::_pose_type::_covariance_type;
    void set_cov(const Eigen::Matrix3f& e_cov, msg_cov_t& cov, int start_idx)
    {
      for (unsigned r = start_idx; r < (unsigned)start_idx+3; r++)
      {
        for (unsigned c = start_idx; c < (unsigned)start_idx+3; c++)
        {
          cov[r * 6 + c] = e_cov(r-start_idx, c-start_idx);
        }
      }
    }
    void set_pos_cov(const Eigen::Matrix3f& e_cov, msg_cov_t& cov)
    {
      set_cov(e_cov, cov, 0);
    }
    void set_rot_cov(const Eigen::Matrix3f& e_cov, msg_cov_t& cov)
    {
      set_cov(e_cov, cov, 3);
    }
    //}

  /* rotate_covariance() method //{ */
  Eigen::Matrix3f rotate_covariance(const Eigen::Matrix3f& covariance, const Eigen::Matrix3f& rotation)
  {
    return rotation * covariance * rotation.transpose();  // rotate the covariance to point in direction of est. position
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

    /* to_pcl() method //{ */
    pt_XYZ_t to_pcl(const vec3_t& pt)
    {
      return {pt.x(), pt.y(), pt.z()};
    }
    //}

    /* fit_local_line() method //{ */
    std::optional<linefit_t> fit_local_line(const float neighborhood, const pt_XYZt_t& pt)
    {
      // find neighborhood points and their stamps
      pc_XYZt_t::Ptr line_pts = boost::make_shared<pc_XYZt_t>();
      /*  //{ */

      {
        pcl::PointIndices::Ptr neigh_idxs = boost::make_shared<pcl::PointIndices>();
        std::vector<float> neigh_dists;
        m_global_octree->radiusSearch(pt, neighborhood, neigh_idxs->indices, neigh_dists);
        pcl::ExtractIndices<pt_XYZt_t> ei;
        ei.setInputCloud(m_global_octree->getInputCloud());
        ei.setIndices(neigh_idxs);
        ei.filter(*line_pts);
      }

      //}

      // TODO: parametrize this shit
      const int m_min_linefit_points = 3;
      if (line_pts->size() < m_min_linefit_points)
      {
        ROS_WARN("[FitLocalLine]: Not enough points to fit line, skipping (got %lu/%d)", line_pts->size(), m_min_linefit_points);
        return std::nullopt;
      }

      // fit a line to the neighborhood points
      /*  //{ */

      auto model_l = boost::make_shared<pcl::SampleConsensusModelLine<pt_XYZt_t>>(line_pts);
      pcl::LeastMedianSquares<pt_XYZt_t> fitter(model_l);
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

      const float max_point_age = m_linefit_point_max_age;  // seconds
      pc_XYZt_t::Ptr in_pts = boost::make_shared<pc_XYZt_t>();
      in_pts->reserve(inliers.size());
      for (const auto idx : inliers)
      {
        const float dt = line_pts->at(idx).intensity - pt.intensity;
        // ignore too old points
        if (dt >= -max_point_age)
          in_pts->push_back(line_pts->at(idx));
      }

      //}

      // sort the inliers by relative time to the main point
      /*  //{ */

      std::sort(std::begin(in_pts->points), std::end(in_pts->points),
                // comparison lambda function
                [](const pt_XYZt_t& a, const pt_XYZt_t& b) { return a.intensity < b.intensity; });

      //}

      const linefit_t ret = {in_pts, params};
      return ret;
    }
    //}

    /* estimate_velocity() method //{ */
    std::optional<vec3_t> estimate_velocity(const pt_XYZt_t& pt_stamped, ros::Publisher& dbg_pub)
    {
      const auto linefit_opt = fit_local_line(m_linefit_neighborhood, pt_stamped);
      if (!linefit_opt.has_value())
      {
        ROS_ERROR("[EstimateVelocity]: Line fit failed!");
        return std::nullopt;
      }

      const auto& inlier_pts = linefit_opt->inlier_pts;

      // find the mean velocity between the points
      vec3_t mean_vel(0, 0, 0);
      size_t mean_vel_weight = 0;
      pt_XYZt_t prev_pt;
      bool prev_pt_initialized = false;
      for (const auto& pt : inlier_pts->points)
      {
        if (!prev_pt_initialized)
        {
          prev_pt = pt;
          prev_pt_initialized = true;
          continue;
        }

        const float cur_dt = pt.intensity - prev_pt.intensity;
        const auto cur_pt = pt.getVector3fMap();
        /* assert(cur_dt != 0.0f); */
        if (cur_dt == 0.0f)
        {
          NODELET_ERROR("[EstimateVelocity]: dt between points is zero - this shouldn't happen! Prev. time: %.2f, cur. time: %.2f.", prev_pt.intensity, pt.intensity);
          continue;
        }

        const vec3_t cur_vel = (cur_pt - prev_pt.getVector3fMap()) / cur_dt;
        mean_vel += cur_vel;
        mean_vel_weight++;
        prev_pt = pt;
      }

      // if no velocity estimate could be obtained from the inliers, return
      if (mean_vel.isZero() || mean_vel_weight == 0)
      {
        ROS_WARN("[EstimateVelocity]: Unable to estimate speed from inliers (is zero)!");
        return std::nullopt;
      }
      mean_vel /= mean_vel_weight;

      if (!m_sh_ball_speed.hasMsg())
      {
        ROS_WARN("[EstimateVelocity]: Haven't received expected ball speed information - cannot confirm detection! Skipping.");
        return std::nullopt;
      }
      const auto ball_speed = m_sh_ball_speed.getMsg()->data;
      // check if the estimated speed makes sense
      const double est_speed = mean_vel.norm();
      if (std::abs(est_speed - ball_speed) > m_max_speed_error)
      {
        ROS_ERROR("[EstimateVelocity]: Object speed is too different from the expected speed (%.2fm/s vs %.2fm/s)!", est_speed, ball_speed);
        return std::nullopt;
      }

      // if requested, publish the neighborhood for debugging
      if (dbg_pub.getNumSubscribers() > 0)
      {
        inlier_pts->header.frame_id = m_world_frame_id;
        pcl_conversions::toPCL(ros::Time::now(), inlier_pts->header.stamp);
        dbg_pub.publish(inlier_pts);
      }

      return mean_vel;
    }
    //}

    /* find_detection_pose() method //{ */
    std::optional<geometry_msgs::Pose> find_detection_pose(const pt_XYZt_t& pos_stamped, ros::Publisher& pub)
    {
      geometry_msgs::Pose ret;
      const auto vel_opt = estimate_velocity(pos_stamped, pub);
      if (!vel_opt.has_value())
        return std::nullopt;

      const auto vel = vel_opt.value();
      /* const Eigen::AngleAxisf anax(yaw, vec3_t::UnitZ()); */
      const quat_t quat = quat_t::FromTwoVectors(vec3_t::UnitX(), vel);
      ret.orientation.w = quat.w();
      ret.orientation.x = quat.x();
      ret.orientation.y = quat.y();
      ret.orientation.z = quat.z();

      ret.position.x = pos_stamped.x;
      ret.position.y = pos_stamped.y;
      ret.position.z = pos_stamped.z;

      return ret;
    }
    //}

    /* find_most_probable_passthrough() method //{ */
    std::optional<geometry_msgs::Pose> find_most_probable_passthrough()
    {
      float maxval = 0.0;
      int max_x = 0, max_y = 0, max_z = 0;
      float max_time = 0.0f;
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
              max_time = map_at_coords(m_map3d_last_update, x_it, y_it, z_it);
            }
          }
        }
      }

      //}

      if (maxval == 0.0)
        return std::nullopt;

      pt_XYZt_t pos_stamped = arena_to_global<pt_XYZt_t>(max_x, max_y, max_z);;
      pos_stamped.intensity = max_time;

      const auto ret = find_detection_pose(pos_stamped, m_pub_chosen_neighborhood);
      return ret;
    }
    //}

    /* in_operation_area() method //{ */
    inline bool in_operation_area(const pcl::PointXYZ& pt)
    {
      /* const bool in_poly = boost::geometry::covered_by(point(pt.x, pt.y), m_operation_area_ring); */
      const bool in_poly = boost::geometry::within(point(pt.x, pt.y), m_operation_area_ring);
      const bool height_ok = pt.z > m_operation_area_min_z && pt.z < m_operation_area_max_z;
      return in_poly && height_ok;
    }
    //}

    /* filter_points() method //{ */
    void filter_points(pc_XYZ_t::Ptr cloud)
    {
      pc_XYZ_t::Ptr cloud_out = boost::make_shared<pc_XYZ_t>();
      cloud_out->header = cloud->header;
      cloud_out->reserve(cloud->size() / 100);
      for (size_t it = 0; it < cloud->size(); it++)
      {
        if (in_operation_area(cloud->points[it]))
        {
          cloud_out->push_back(cloud->points[it]);
        }
      }
      cloud_out->swap(*cloud);
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
      for (size_t i = 0; i < m_operation_area_ring.size() - 1; i++)
      {
        const auto pt1 = boost2gmsgs(m_operation_area_ring.at(i), m_operation_area_min_z);
        const auto pt2 = boost2gmsgs(m_operation_area_ring.at(i + 1), m_operation_area_min_z);
        safety_area_marker.points.push_back(pt1);
        safety_area_marker.points.push_back(pt2);
      }

      // top border
      for (size_t i = 0; i < m_operation_area_ring.size() - 1; i++)
      {
        const auto pt1 = boost2gmsgs(m_operation_area_ring.at(i), m_operation_area_max_z);
        const auto pt2 = boost2gmsgs(m_operation_area_ring.at(i + 1), m_operation_area_max_z);
        safety_area_marker.points.push_back(pt1);
        safety_area_marker.points.push_back(pt2);
      }

      // top/bot edges
      for (size_t i = 0; i < m_operation_area_ring.size() - 1; i++)
      {
        const auto pt1 = boost2gmsgs(m_operation_area_ring.at(i), m_operation_area_min_z);
        const auto pt2 = boost2gmsgs(m_operation_area_ring.at(i), m_operation_area_max_z);
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

    /* lidar_visualization() method //{ */
    visualization_msgs::Marker lidar_visualization(const double hfov, const double range, const std_msgs::Header& header)
    {
      visualization_msgs::Marker ret;

      ret.header = header;
      ret.color.a = 0.2;
      ret.color.r = 1.0;
      ret.scale.x = 1.0;
      ret.scale.y = 1.0;
      ret.scale.z = 1.0;
      ret.ns = "lidar FOV";
      ret.pose.orientation.w = 1;
      ret.pose.orientation.x = 0;
      ret.pose.orientation.y = 0;
      ret.pose.orientation.z = 0;
      ret.pose.position.x = 0;
      ret.pose.position.y = 0;
      ret.pose.position.z = 0;

      ret.type = visualization_msgs::Marker::TRIANGLE_LIST;
      constexpr int circ_pts_per_meter_radius = 10;
      int circ_pts = std::round(range * circ_pts_per_meter_radius);
      if (circ_pts % 2)
        circ_pts++;
      geometry_msgs::Point center_pt; // just zeros - center of the sensor
      geometry_msgs::Point prev_pt_top;
      geometry_msgs::Point prev_pt_bot;
      for (int it = 0; it < circ_pts; it++)
      {
        const float angle = M_PI / (circ_pts / 2.0f) * it;
        geometry_msgs::Point pt_top;
        pt_top.x = range * cos(angle);
        pt_top.y = range * sin(angle);
        pt_top.z = range * tan(hfov/2.0);

        geometry_msgs::Point pt_bot;
        pt_bot.x = range * cos(angle);
        pt_bot.y = range * sin(angle);
        pt_bot.z = -range * tan(hfov/2.0);
        if (it != 0)
        {
          ret.points.push_back(prev_pt_top);
          ret.points.push_back(pt_top);
          ret.points.push_back(center_pt);

          ret.points.push_back(prev_pt_bot);
          ret.points.push_back(pt_bot);
          ret.points.push_back(center_pt);

          ret.points.push_back(prev_pt_bot);
          ret.points.push_back(pt_bot);
          ret.points.push_back(pt_top);

          ret.points.push_back(prev_pt_top);
          ret.points.push_back(pt_top);
          ret.points.push_back(prev_pt_bot);
        }
        prev_pt_top = pt_top;
        prev_pt_bot = pt_bot;
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
    mrs_lib::SubscribeHandler<pc_XYZ_t> m_sh_pc;
    mrs_lib::SubscribeHandler<std_msgs::Float64> m_sh_ball_speed;

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

    ros::Publisher m_pub_lidar_fov;

    ros::ServiceServer m_reset_server;

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

    double m_max_speed_error;  // metres per second
    float m_linefit_neighborhood;
    float m_linefit_point_max_age;

    float m_classif_max_detection_height;
    float m_classif_max_detection_width;
    float m_classif_max_mav_height;
    float m_classif_ball_wire_length;

    float m_classif_close_dist;
    float m_classif_mav_width;
    float m_classif_ball_width;

    double m_exclude_box_offset_x;
    double m_exclude_box_offset_y;
    double m_exclude_box_offset_z;
    double m_exclude_box_size_x;
    double m_exclude_box_size_y;
    double m_exclude_box_size_z;

    std::string m_safety_area_frame;
    Eigen::MatrixXd m_safety_area_border_points;
    double m_safety_area_deflation;
    ring m_operation_area_ring;
    double m_operation_area_min_z;
    double m_operation_area_max_z;
    Eigen::Vector2d m_safety_area_left_cut_plane_normal;
    double m_safety_area_left_cut_plane_offset;
    Eigen::Vector2d m_safety_area_right_cut_plane_normal;
    double m_safety_area_right_cut_plane_offset;
    Eigen::Vector2d m_safety_area_offset;

    std::map<cluster_class_t, std::pair<double&, double&>> m_cov_coeffs;
    double m_cov_coeff_vel;

    //}

  private:
    // --------------------------------------------------------------
    // |                   Other member variables                   |
    // --------------------------------------------------------------

    std::mutex m_reset_mtx;

    bool m_safety_area_initialized;
    uint32_t m_last_detection_id;
    std::vector<float> m_map3d;
    std::vector<float> m_map3d_last_update;
    octree::Ptr m_global_octree;
    pc_XYZt_t::Ptr m_global_cloud;  // contains position estimates of the ball

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
