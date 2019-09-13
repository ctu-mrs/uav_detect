#!/usr/bin/env python  
import rospy

import math
import tf2_ros
import geometry_msgs.msg
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
from tf2_msgs.msg import TFMessage
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point

tfBuffer = None
last_odom1 = None
last_rtk1 = None
last_odom2 = None
last_rtk2 = None

def odom1_callback(data):
    global last_odom1
    rospy.loginfo_throttle(1.0, "getting odom1 data")
    last_odom1 = data

def rtk1_callback(data):
    global last_rtk1
    rospy.loginfo_throttle(1.0, "getting rtk1 data")
    last_rtk1 = data

def odom2_callback(data):
    global last_odom2
    rospy.loginfo_throttle(1.0, "getting odom2 data")
    last_odom2 = data

def rtk2_callback(data):
    global last_rtk2
    rospy.loginfo_throttle(1.0, "getting rtk2 data")
    last_rtk2 = data

def get_tf(rtk, odom, frame):
    ret = TFMessage()
    tf = geometry_msgs.msg.TransformStamped()

    # rot = R.from_quat((odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w)).inv()
    tf.transform.translation.x = rtk.pose.pose.position.x
    tf.transform.translation.y = rtk.pose.pose.position.y
    tf.transform.translation.z = odom.pose.pose.position.z
    tf.transform.rotation.w = odom.pose.pose.orientation.w
    tf.transform.rotation.x = odom.pose.pose.orientation.x
    tf.transform.rotation.y = odom.pose.pose.orientation.y
    tf.transform.rotation.z = odom.pose.pose.orientation.z
    tf.header = odom.header
    tf.child_frame_id = frame

    ret.transforms.append(tf)
    return ret

if __name__ == '__main__':
    rospy.init_node('cnn_detect_evaluator')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.Subscriber("/uav42/odometry/odom_main", nav_msgs.msg.Odometry, odom1_callback)
    rospy.Subscriber("/uav42/odometry/rtk_local_odom", nav_msgs.msg.Odometry, rtk1_callback)
    rospy.Subscriber("/uav4/odometry/odom_main", nav_msgs.msg.Odometry, odom2_callback)
    rospy.Subscriber("/uav4/odometry/rtk_local_odom", nav_msgs.msg.Odometry, rtk2_callback)

    bag = rosbag.Bag("tfs_out.bag", mode='w')

    rospy.loginfo("Spinning")
    r = rospy.Rate(1000)
    while not rospy.is_shutdown():
        if last_rtk1 is not None and last_odom1 is not None:
            msg_t = last_odom1.header.stamp
            tf = get_tf(last_rtk1, last_odom1, "fcu_uav42")
            bag.write("/tf", tf, t=msg_t)
            rospy.loginfo_throttle(1.0, "writing tf messages for UAV42")

        if last_rtk2 is not None and last_odom2 is not None:
            msg_t = last_odom2.header.stamp
            tf = get_tf(last_rtk2, last_odom2, "fcu_uav4")
            bag.write("/tf", tf, t=msg_t)
            rospy.loginfo_throttle(1.0, "writing tf messages for UAV4")

        r.sleep()
    bag.close()
