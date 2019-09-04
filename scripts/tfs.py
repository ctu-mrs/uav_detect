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

tfBuffer = None
cmodel = PinholeCameraModel()
last_img = None
bridge = None
last_odom = None
last_rtk1 = None
last_rtk2 = None

def odom_callback(data):
    global last_odom
    last_odom = data

def rtk1_callback(data):
    global last_rtk1
    last_rtk1 = data

def rtk2_callback(data):
    global last_rtk2
    last_rtk2 = data

def get_tf():
    global last_rtk1
    global last_rtk2
    pt1 = np.array([last_rtk1.pose.pose.position.x, last_rtk1.pose.pose.position.y, last_rtk1.pose.pose.position.z])
    pt2 = np.array([last_rtk2.pose.pose.position.x, last_rtk2.pose.pose.position.y, last_rtk2.pose.pose.position.z])
    trans = pt2 - pt1
    # trans = pt1 - pt2
    rot = R.from_quat((last_odom.pose.pose.orientation.x, last_odom.pose.pose.orientation.y, last_odom.pose.pose.orientation.z, last_odom.pose.pose.orientation.w)).inv()
    # rot = Quaternion(w=last_odom.pose.pose.orientation.w, x=last_odom.pose.pose.orientation.x, y=last_odom.pose.pose.orientation.y, z=last_odom.pose.pose.orientation.z)
    ret = rot.apply(trans)
    print(trans)
    print(ret)
    rot = R.from_euler('ZYX', (1.57, 3.14, 1.57)).inv()
    ret = rot.apply(ret)
    # tmpy = Quaternion(axis=[0, 0, 1], angle=1.57)
    # tmpp = Quaternion(axis=[0, 1, 0], angle=3.14)
    # tmpr = Quaternion(axis=[1, 0, 0], angle=1.57)
    # tmpf = Quaternion(axis=[0, 1, 0], angle=3.14)
    # rot2 = to_quaternion(1.57, 3.14, 1.57)
    # rot2 = tmpr.rotate(tmpp.rotate(tmpy))
    # rot2 = rot2.inverse
    # ret = rot2.rotate(ret)
    # ret = tmpf.rotate(tmpr.rotate(tmpp.rotate(tmpy.rotate(ret))))
    # ret = tmpy.rotate(tmpp.rotate(tmpr.rotate(ret)))
    # ret = rot2.rotate(ret)
    print(ret)
    return ret
    # trans = None
    # try:
    #     trans = tfBuffer.lookup_transform("fcu_uav4", "rs_d435_color_optical_frame", rospy.Time())
    # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #     rospy.logwarn("could not find TF!")
    # return trans

def img_callback(data):
    global last_img
    rospy.loginfo("I heard IMAGE")
    try:
      last_img = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

def cnn_callback(data):
    global last_img
    rospy.loginfo("I heard CNN: %s", data)

    if last_img is None:
        rospy.logwarn("Waiting for images")
        return

    # pt3 = geometry_msgs.msg.PointStamped()
    # pt3.header.stamp = data.header.stamp
    # pt3.header.frame_id = "fcu_uav4"
    # pt3.point.x = 0
    # pt3.point.y = 0
    # pt3.point.z = 0
    # pt3 = tfBuffer.transform(pt3, data.header.frame_id)
    # pt3 = [pt3.point.x, pt3.point.y, pt3.point.z]
    pt3 = get_tf()
    print("pt3: ", pt3)
    pt = cmodel.project3dToPixel(pt3)
    cur_img = np.copy(last_img)
    cv2.circle(cur_img, (int(pt[0]), int(pt[1])), 20, (255, 0, 0))
    cv2.imshow("window", cur_img)
    cv2.waitKey(1)

def depth_callback(data):
    rospy.loginfo("I heard DEPTH: %s", data)
    trans = get_tf()
    print(trans)

def cinfo_callback(data):
    rospy.loginfo("I heard CINFO: %s", data)
    cmodel.fromCameraInfo(data)

if __name__ == '__main__':
    rospy.init_node('tf2_listener')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    bridge = CvBridge()

    rospy.Subscriber("/uav42/odometry/odom_main", nav_msgs.msg.Odometry, odom_callback)
    rospy.Subscriber("/uav42/odometry/rtk_local_odom", nav_msgs.msg.Odometry, rtk1_callback)
    rospy.Subscriber("/uav4/odometry/rtk_local_odom", nav_msgs.msg.Odometry, rtk2_callback)

    rospy.Subscriber("/uav42/cnn_detect/detections", cnn_msg.Detections, cnn_callback)
    rospy.Subscriber("/uav42/uav_detection/detections", uav_msg.Detections, depth_callback)
    rospy.Subscriber("/uav42/rs_d435/color/camera_info", sensor_msgs.msg.CameraInfo, cinfo_callback)
    rospy.Subscriber("/uav42/rs_d435/color/image_rect_color", sensor_msgs.msg.Image, img_callback)

    rospy.loginfo("Spinning")
    rospy.spin()
