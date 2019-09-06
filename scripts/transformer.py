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
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point

pub = None
tf_buffer = None

def callback(msg):
    global pub, tf_buffer
    pose_lo = geometry_msgs.msg.PointStamped()
    pose_lo.header = msg.header
    pose_lo.point = msg.pose.pose.position
    pose_tfd = tf_buffer.transform(pose_lo, "rs_d435_color_optical_frame", rospy.Duration(0.01))
    pub.publish(pose_tfd)
    rospy.loginfo("Transformed")


if __name__ == '__main__':
    rospy.init_node('transformer')

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    topic_name = "/uav42/uav_localization/localized_uav"
    rospy.Subscriber(topic_name, PoseWithCovarianceStamped, callback)
    pub = rospy.Publisher(topic_name + "/tfd", PointStamped, queue_size=10)

    rospy.loginfo("Ready")
    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        r.sleep()
