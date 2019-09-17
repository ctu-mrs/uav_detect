#!/usr/bin/env python  
import rospy

import math
import tf2_ros
from tf2_msgs.msg import TFMessage
import geometry_msgs.msg
from geometry_msgs.msg import PoseWithCovarianceStamped
import nav_msgs.msg
import cnn_detect.msg as cnn_msg
import uav_detect.msg as uav_msg
import sensor_msgs.msg
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_geometry_msgs
# from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import os
import pickle
import sys
import rosbag

import tf2_ros
# from tf2_sensor_msgs import PointCloud
# from tf2_geometry_msgs import PoseWithCovarianceStamped
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud

pub = None
tf_buffer = None

def callback(msg):
    global pub, tf_buffer
    tgt_frame = "rs_d435_color_optical_frame"

    pose_lo = geometry_msgs.msg.PointStamped()
    pose_lo.header = msg.header
    pose_lo.point = msg.pose.pose.position
    pose_tfd = tf_buffer.transform(pose_lo, tgt_frame, rospy.Duration(0.1))
    trans = tf_buffer.lookup_transform(msg.header.frame_id, tgt_frame, msg.header.stamp, rospy.Duration(0.1))
    quat = trans.transform.rotation
    rot = R.from_quat((quat.x, quat.y, quat.z, quat.w)).inv()
    rot = np.matrix(rot.as_dcm())
    # print("rot: {}".format(rot))

    cov_all = np.matrix(msg.pose.covariance)
    cov_all.shape = (6, 6)
    # print("cov_all: {}".format(cov_all))
    cov = cov_all[:3, :3]
    # print("cov: {}".format(cov))
    cov = rot*cov*rot.transpose()
    cov_all[:3, :3] = cov

    msg.header.frame_id = tgt_frame
    msg.pose.pose.position = pose_tfd.point
    msg.pose.covariance = cov_all.ravel().tolist()[0]

    pub.publish(msg)
    rospy.loginfo("Transformed PoseWithCovarianceStamped")

pcl_pub = None

def pcl_callback(msg):
    global pcl_pub, tf_buffer
    tgt_frame = "rs_d435_color_optical_frame"

    trans = tf_buffer.lookup_transform(msg.header.frame_id, tgt_frame, msg.header.stamp, rospy.Duration(0.1))
    lin = trans.transform.translation
    lin = np.matrix((lin.x, lin.y, lin.z)).transpose()
    quat = trans.transform.rotation
    rot = R.from_quat((quat.x, quat.y, quat.z, quat.w)).inv()
    rot = np.matrix(rot.as_dcm())

    pts = list()
    for pt in msg.points:
        nppt = np.matrix((pt.x, pt.y, pt.z)).transpose()
        nppt = rot*(nppt-lin)
        pt.x = nppt[0]
        pt.y = nppt[1]
        pt.z = nppt[2]
        pts.append(pt)
    msg.header.frame_id = tgt_frame
    msg.points = pts
    # print(msg)

    pcl_pub.publish(msg)
    rospy.loginfo("Transformed PointCloud")

if __name__ == '__main__':
    rospy.init_node('transformer')

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    topic_name = "/uav42/uav_localization/localized_uav"
    rospy.Subscriber(topic_name, PoseWithCovarianceStamped, callback)
    pub = rospy.Publisher(topic_name + "/tfd", PoseWithCovarianceStamped, queue_size=10)

    pcl_topic_name = "/uav42/uav_localization/dbg_measurements_pcl"
    rospy.Subscriber(pcl_topic_name, PointCloud, pcl_callback)
    pcl_pub = rospy.Publisher(pcl_topic_name + "/tfd", PointCloud, queue_size=10)

    rospy.loginfo("Ready")
    r = rospy.Rate(100, reset=True)
    while not rospy.is_shutdown():
        r.sleep()
