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

tfBuffer = None
cmodel = PinholeCameraModel()
last_img = None
last_img_ready_cnn = False
last_img_ready_depth = False
bridge = None
last_odom = None
last_rtk1 = None
last_rtk2 = None
detected_cnn = False
detected_depth = False
gts_loaded = False
gt = np.array([0, 0])
gts = [None]

TPs = [0, 0, 0]
FPs = [0, 0, 0]
FNs = [0, 0, 0]
FP_dist = 100

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

gt_it = 0
def img_callback(data):
    global last_img, gt, gts, gts_loaded, gt_it, detected_cnn, detected_depth
    # rospy.loginfo("I heard IMAGE")
    try:
      last_img = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    if gts_loaded:
        gt = gts[gt_it]
        gt_it += 1
    else:
        gts.append(gt)
    detected_cnn = False
    detected_depth = False



def cnn_callback(data):
    global gt
    global last_img, TPs, FPs, FNs, last_img_ready_cnn, detected_cnn
    # rospy.loginfo("I heard CNN: %s", data)

    if last_img is None and not last_img_ready_cnn and gt is not None:
        rospy.logwarn("Waiting for images")
        return

    # pt3 = get_tf()
    # print("pt3: ", pt3)
    # pt = cmodel.project3dToPixel(pt3)
    pt = gt

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
            detected_cnn = True
        else:
            FPs[0] += 1
    last_img_ready_cnn = True
    # cv2.imshow("window", cur_img)
    # cv2.waitKey(1)

def depth_callback(data):
    global gt
    global last_img, TPs, FPs, FNs, last_img_ready_depth, detected_depth
    # rospy.loginfo("I heard DEPTH: %s", data)

    if last_img is None and not last_img_ready_depth and gt is not None:
        rospy.logwarn("Waiting for images")
        return

    # pt3 = get_tf()
    # print("pt3: ", pt3)
    # pt = cmodel.project3dToPixel(pt3)
    pt = gt

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
            detected_depth = True
        else:
            FPs[1] += 1
    last_img_ready_depth = True
    # cv2.imshow("window", cur_img)
    # cv2.waitKey(1)

def cinfo_callback(data):
    # rospy.loginfo("I heard CINFO: %s", data)
    cmodel.fromCameraInfo(data)

def mouse_callback(event, x, y, flags, param):
    global gt
    gt = np.array((x, y))

if __name__ == '__main__':
    rospy.init_node('tf2_listener')

    if os.path.isfile("gts.pkl"):
        with open("gts.pkl", "rb") as ifile:
            gts = pickle.load(ifile)
            gts_loaded = True

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

    cv2.namedWindow("MAV detection")
    if not gts_loaded:
        cv2.setMouseCallback("MAV detection", mouse_callback)

    rospy.loginfo("Spinning")
    while not rospy.is_shutdown():
        if last_img is not None and last_img_ready_cnn and last_img_ready_depth:

            if detected_cnn:
                TPs[0] += 1
            else:
                FNs[0] += 1

            if detected_depth:
                TPs[1] += 1
            else:
                FNs[1] += 1

            if detected_depth or detected_cnn:
                TPs[2] += 1
            else:
                FNs[2] += 1
            FPs[2] = FPs[0] + FPs[1]

            rospy.loginfo("type\tTPs\tFPs\tFNs")
            rospy.loginfo("cnn\t{:d}\t{:d}\t{:d}".format(TPs[0], FPs[0], FNs[0]))
            rospy.loginfo("depth\t{:d}\t{:d}\t{:d}".format(TPs[1], FPs[1], FNs[1]))
            rospy.loginfo("both\t{:d}\t{:d}\t{:d}".format(TPs[2], FPs[2], FNs[2]))
            cv2.imshow("MAV detection", last_img)
            cv2.waitKey(1)
            last_img = None
            last_img_ready_cnn = False
            last_img_ready_depth = False
    if not gts_loaded:
        with open("gts.pkl", "wb") as ofile:
            pickle.dump(gts, ofile)
        rospy.loginfo("Dumped ground-truth positions to gts.pkl")
