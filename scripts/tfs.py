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
last_img_ready_cnn = False
last_img_ready_depth = False
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
    offset = np.array([0, 0, 0.5])
    pt1 = np.array([last_rtk1.pose.pose.position.x, last_rtk1.pose.pose.position.y, last_rtk1.pose.pose.position.z])
    pt2 = np.array([last_rtk2.pose.pose.position.x, last_rtk2.pose.pose.position.y, last_rtk2.pose.pose.position.z])
    trans = pt2 - pt1 + offset
    rot = R.from_quat((last_odom.pose.pose.orientation.x, last_odom.pose.pose.orientation.y, last_odom.pose.pose.orientation.z, last_odom.pose.pose.orientation.w)).inv()
    ret = rot.apply(trans)
    # print(trans)
    # print(ret)
    rot = R.from_euler('ZYX', (1.57, 3.14, 1.57)).inv()
    ret = rot.apply(ret)
    # print(ret)
    return ret

def img_callback(data):
    global last_img
    # rospy.loginfo("I heard IMAGE")
    try:
      last_img = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

TPs = 0
FPs = 0
FNs = 0
FP_dist = 150
def cnn_callback(data):
    global last_img, TPs, FPs, FNs, last_img_ready_cnn
    # rospy.loginfo("I heard CNN: %s", data)

    if last_img is None:
        rospy.logwarn("Waiting for images")
        return

    detected = False
    pt3 = get_tf()
    # print("pt3: ", pt3)
    pt = cmodel.project3dToPixel(pt3)

    # cur_img = np.copy(last_img)
    cur_img = last_img
    cv2.circle(cur_img, (int(pt[0]), int(pt[1])), FP_dist, (255, 0, 0))
    for det in data.detections:
        w = det.roi.width
        h = det.roi.height
        pxx = w*det.x
        pxy = h*det.y
        pxw = w*det.width
        pxh = h*det.height
        cv2.rectangle(cur_img, (int(pxx - pxw/2.0), int(pxy - pxh/2.0)), (int(pxx + pxw/2.0), int(pxy + pxh/2.0)), (0, 0, 255))
        ptdet = np.array([pxx, pxy])
        if np.linalg.norm(pt - ptdet) < FP_dist:
            detected = True
            TPs += 1
        else:
            FPs += 1
    last_img_ready_cnn = True

    if not detected:
        FNs += 1
    rospy.loginfo("TPs: {:d}, FPs: {:d}, FNs: {:d}".format(TPs, FPs, FNs))
    # cv2.imshow("window", cur_img)
    # cv2.waitKey(1)

def depth_callback(data):
    global last_img, TPs, FPs, FNs, last_img_ready_depth
    # rospy.loginfo("I heard DEPTH: %s", data)

    if last_img is None:
        rospy.logwarn("Waiting for images")
        return

    detected = False
    pt3 = get_tf()
    # print("pt3: ", pt3)
    pt = cmodel.project3dToPixel(pt3)

    # cur_img = np.copy(last_img)
    cur_img = last_img
    cv2.circle(cur_img, (int(pt[0]), int(pt[1])), FP_dist, (255, 0, 0))
    for det in data.detections:
        w = det.roi.width
        h = det.roi.height
        pxx = w*det.x
        pxy = h*det.y
        pxr = 200/det.depth
        cv2.circle(cur_img, (int(pxx), int(pxy)), int(pxr), (0, 255, 0))
        ptdet = np.array([pxx, pxy])
        if np.linalg.norm(pt - ptdet) < FP_dist:
            detected = True
            TPs += 1
        else:
            FPs += 1
    last_img_ready_depth = True

    if not detected:
        FNs += 1
    rospy.loginfo("TPs: {:d}, FPs: {:d}, FNs: {:d}".format(TPs, FPs, FNs))
    # cv2.imshow("window", cur_img)
    # cv2.waitKey(1)

def cinfo_callback(data):
    # rospy.loginfo("I heard CINFO: %s", data)
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
    while not rospy.is_shutdown():
        if last_img is not None and last_img_ready_cnn and last_img_ready_depth:
            cv2.imshow("window", last_img)
            cv2.waitKey(1)
            last_img = None
            last_img_ready_cnn = False
            last_img_ready_depth = False
