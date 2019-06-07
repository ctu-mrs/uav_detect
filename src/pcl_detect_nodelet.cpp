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

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

#include <pcl/surface/poisson.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <uav_detect/DetectionParamsConfig.h>

using namespace cv;
using namespace std;
using namespace uav_detect;

// shortcut type to the dynamic reconfigure manager template instance
typedef mrs_lib::DynamicReconfigureMgr<uav_detect::DetectionParamsConfig> drmgr_t;
typedef pcl::PointCloud<pcl::PointXYZ> PC;

namespace uav_detect
{

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
      pl.load_param("active_box_size", m_drmgr_ptr->config.active_box_size);

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
      m_pcl_sh = smgr.create_handler_threadsafe<PC::ConstPtr, subs_time_consistent>("pcl", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
      // Initialize publishers
      m_detections_pub = nh.advertise<uav_detect::Detections>("detections", 10); 
      /* m_detected_blobs_pub = nh.advertise<uav_detect::BlobDetections>("blob_detections", 1); */
      m_processed_pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("processed_pcl", 1);
      //}

      m_last_detection_id = 0;

      m_det_blobs = 0;
      m_images_processed = 0;
      m_avg_fps = 0.0f;
      m_avg_delay = 0.0f;

      /* m_detector = dbd::PCLBlobDetector(m_drmgr_ptr->config, m_unknown_pixel_value); */

      m_main_loop_timer = nh.createTimer(ros::Rate(1000), &PCLDetector::main_loop, this);
      /* m_info_loop_timer = nh.createTimer(ros::Rate(1), &PCLDetector::info_loop, this); */

      m_cloud_global = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

      cout << "----------------------------------------------------------" << std::endl;

    }
    //}

    PC::Ptr m_cloud_global;

    /* main_loop() method //{ */
    void main_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      if (m_pcl_sh->new_data())
      {
        /* ros::Time start_t = ros::Time::now(); */

        NODELET_INFO_STREAM("[PCLDetector]: Processing new data --------------------------------------------------------- ");

        PC::ConstPtr cloud = m_pcl_sh->get_data();
        ros::Time msg_stamp;
        pcl_conversions::fromPCL(cloud->header.stamp, msg_stamp);
        NODELET_INFO_STREAM("[PCLDetector]: Input PC has " << cloud->size() << " points");

        /* filter input cloud and transform it to world //{ */
        
        Eigen::Vector3d tf_trans;
        {
          /* filter by voxel grid //{ */
          pcl::VoxelGrid<pcl::PointXYZ> vg;
          PC::Ptr cloud_filtered = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
          vg.setLeafSize(0.25f, 0.25f, 0.25f);
          vg.setInputCloud(cloud);
          vg.filter(*cloud_filtered);
          //}

          /* filter by cropping points outside a box, relative to the sensor //{ */
          const auto box_size = m_drmgr_ptr->config.active_box_size;
          const Eigen::Vector4f box_point1(box_size/2, box_size/2, box_size/2, 1);
          const Eigen::Vector4f box_point2(-box_size/2, -box_size/2, -box_size/2, 1);
          pcl::CropBox<pcl::PointXYZ> cb;
          cb.setMax(box_point1);
          cb.setMin(box_point2);
          cb.setInputCloud(cloud_filtered);
          cb.filter(*cloud_filtered);
          //}

          Eigen::Affine3d s2w_tf;
          bool tf_ok = get_transform_to_world(cloud->header.frame_id, msg_stamp, s2w_tf);
          if (!tf_ok)
          {
            NODELET_ERROR("[PCLDetector]: Could not transform pointcloud to global, skipping.");
            return;
          }
          tf_trans = s2w_tf.translation();
          pcl::transformPointCloud(*cloud_filtered, *cloud_filtered, s2w_tf.cast<float>());
          /* tf2::doTransform(cloud, cloud, s2w_tf); */

          cloud = cloud_filtered;
          NODELET_INFO_STREAM("[PCLDetector]: Filtered input PC has " << cloud->size() << " points");
        }
        
        //}

        /* add filtered input cloud to global cloud and filter it //{ */
        
        {
          /* filter by mutual point distance (voxel grid) //{ */
          *m_cloud_global += *cloud;
          pcl::VoxelGrid<pcl::PointXYZ> vg;
          PC::Ptr cloud_filtered = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
          vg.setLeafSize(0.25f, 0.25f, 0.25f);
          vg.setInputCloud(m_cloud_global);
          vg.filter(*cloud_filtered);
          //}

          /* filter by cropping points outside a box, relative to the sensor //{ */
          const auto box_size = m_drmgr_ptr->config.active_box_size;
          const Eigen::Vector4f sensor_origin(tf_trans.x(), tf_trans.y(), tf_trans.z(), 1.0f);
          const Eigen::Vector4f box_point1 = sensor_origin - Eigen::Vector4f(box_size/2, box_size/2, box_size/2, 0);
          const Eigen::Vector4f box_point2 = sensor_origin + Eigen::Vector4f(box_size/2, box_size/2, box_size/2, 0);
          pcl::CropBox<pcl::PointXYZ> cb;
          cb.setMin(box_point1);
          cb.setMax(box_point2);
          cb.setInputCloud(cloud_filtered);
          cb.filter(*cloud_filtered);
          //}

          m_cloud_global = cloud_filtered;
          NODELET_INFO_STREAM("[PCLDetector]: Global pointcloud has " << m_cloud_global->size() << " points");
        }
        
        //}

        /* pcl::PointIndices::Ptr valid_idx = boost::make_shared<pcl::PointIndices>(); */
        /* const auto& vecpts = m_cloud_global->points; */
        /* for (unsigned it = 0; it < vecpts.size(); it++) */
        /* { */
        /*   const auto& pt = vecpts[it]; */
        /*   if (distsq_from_origin(pt) > 20.0f*20.0f) */
        /*     continue; */
        /*   else */
        /*     valid_idx->indices.push_back(it); */
        /* } */
        /* NODELET_INFO_STREAM("[PCLDetector]: Pointcloud after removing far points has " << valid_idx->indices.size() << " points"); */

        // Creating the KdTree object for the search method of the extraction
        /* pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree = boost::make_shared<pcl::search::KdTree<pcl::PointXYZ>>(); */
        /* kdtree->setInputCloud(m_cloud_global); */

        /* std::vector<pcl::PointIndices> cluster_indices; */
        /* pcl::ConditionalEuclideanClustering<pcl::PointXYZ> ec; */
        /* ec.setClusterTolerance(2.5);  // metres */
        /* ec.setConditionFunction(&scaled_dist_thresholding); */
        /* ec.setMinClusterSize(10); */
        /* ec.setMaxClusterSize(500); */
        /* /1* ec.setSearchMethod(kdtree); *1/ */
        /* ec.setInputCloud(m_cloud_global); */
        /* /1* ec.setIndices(valid_idx); *1/ */
        /* ec.segment(cluster_indices); */

        /* pcl::PointCloud<pcl::PointXYZL> cloud_labeled; */
        /* cloud_labeled.reserve(m_cloud_global->size()); */
        /* /1* cloud_labeled.width = n_pts; *1/ */
        /* /1* cloud_labeled.height = 1; *1/ */
        /* /1* cloud_labeled.is_dense = true; *1/ */
        /* int label = 0; */
        /* for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) */
        /* { */
        /*   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(); */
        /*   for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) */
        /*   { */
        /*     const auto& pt = m_cloud_global->points[*pit]; */
        /*     pcl::PointXYZL ptl; */
        /*     ptl.x = pt.x; */
        /*     ptl.y = pt.y; */
        /*     ptl.z = pt.z; */
        /*     ptl.label = label; */
        /*     cloud_labeled.points.push_back(ptl); */
        /*   } */
        /*   label++; */
        /* } */
        /* NODELET_INFO_STREAM("[PCLDetector]: Extracted " << label << " clusters"); */
        /* NODELET_INFO_STREAM("[PCLDetector]: Processed pointcloud has " << cloud_labeled.size() << " points"); */

        sensor_msgs::PointCloud2 dbg_cloud;
        pcl::toROSMsg(*m_cloud_global, dbg_cloud);
        dbg_cloud.header.frame_id = m_world_frame;
        dbg_cloud.header.stamp = msg_stamp;
        m_processed_pcl_pub.publish(dbg_cloud);
        /* /1* Update statistics for info_loop //{ *1/ */
        /* { */
        /*   std::lock_guard<std::mutex> lck(m_stat_mtx); */
        /*   const ros::Time end_t = ros::Time::now(); */
        /*   const float delay = (end_t - source_msg.header.stamp).toSec(); */
        /*   m_avg_delay = 0.9*m_avg_delay + 0.1*delay; */
        /*   const float fps = 1/(end_t - start_t).toSec(); */
        /*   m_avg_fps = 0.9*m_avg_fps + 0.1*fps; */
        /*   m_images_processed++; */
        /*   m_det_blobs += blobs.size(); */
        /* } */
        /* //} */

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
    mrs_lib::SubscribeHandlerPtr<PC::ConstPtr> m_pcl_sh;
    ros::Publisher m_detections_pub;
    /* ros::Publisher m_detected_blobs_pub; */
    ros::Publisher m_processed_pcl_pub;
    ros::Timer m_main_loop_timer;
    ros::Timer m_info_loop_timer;
    std::string m_node_name;
    //}

  private:

    // --------------------------------------------------------------
    // |                 Parameters, loaded from ROS                |
    // --------------------------------------------------------------

    std::string m_world_frame;

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
