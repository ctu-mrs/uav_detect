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

import tf2_ros
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point

show_imgs = True
tfBuffer = None
cmodel = PinholeCameraModel()
last_img = None
last_img_ready_cnn = False
last_img_ready_depth = False
last_img_ready_loc = False
bridge = None
last_odom = None
last_rtk1 = None
last_rtk2 = None
detected_cnn = False
detected_depth = False
detected_loc = False
gts_loaded = False
gt = np.array([0, 0])
gts = [None]

TPs = [0, 0, 0, 0]
FPs = [0, 0, 0, 0]
FNs = [0, 0, 0, 0]
FP_dist = 100

#        left view,          returned to view
times = [(1567515829.515217, 1567515832.846460),
         (1567515970.712066, 1567515972.843928),
         (1567516025.318885, 9999999999.999999)]

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
    if last_rtk1 is None or last_rtk2 is None or last_odom is None:
        return np.array([0, 0, 0])
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
    global gt, gts, gts_loaded, gt_it, detected_cnn, detected_depth, last_img
    # rospy.loginfo("I heard IMAGE")
    if show_imgs:
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
    global TPs, FPs, FNs, last_img_ready_cnn, detected_cnn
    # rospy.loginfo("I heard CNN: %s", data)

    if last_img_ready_cnn or gt is None or (show_imgs and last_img is None):
        rospy.logwarn("Waiting for images")
        return

    # pt3 = get_tf()
    # print("pt3: ", pt3)
    # pt = cmodel.project3dToPixel(pt3)

    # cv2.circle(cur_img, (int(pt[0]), int(pt[1])), FP_dist, (255, 0, 0))
    for det in data.detections:
        w = det.roi.width
        h = det.roi.height
        pxx = w*det.x
        pxy = h*det.y
        pxw = w*det.width
        pxh = h*det.height
        if show_imgs:
            cv2.rectangle(last_img, (int(pxx - pxw/2.0), int(pxy - pxh/2.0)), (int(pxx + pxw/2.0), int(pxy + pxh/2.0)), (0, 0, 255), 2)
        ptdet = np.array([pxx, pxy])
        if np.linalg.norm(gt - ptdet) < FP_dist:
            detected_cnn = True
        else:
            FPs[0] += 1
    last_img_ready_cnn = True
    # cv2.imshow("window", cur_img)
    # cv2.waitKey(1)

def depth_callback(data):
    global gt
    global TPs, FPs, FNs, last_img_ready_depth, detected_depth
    # rospy.loginfo("I heard DEPTH: %s", data)

    if last_img_ready_depth or gt is None or (show_imgs and last_img is None):
        rospy.logwarn("Waiting for images")
        return

    # pt3 = get_tf()
    # print("pt3: ", pt3)
    # pt = cmodel.project3dToPixel(pt3)

    # cv2.circle(cur_img, (int(pt[0]), int(pt[1])), FP_dist, (255, 0, 0))
    for det in data.detections:
        w = det.roi.width
        h = det.roi.height
        pxx = w*det.x
        pxy = h*det.y
        pxr = 200/det.depth
        pxw = pxr
        pxh = pxr
        if show_imgs:
            cv2.rectangle(last_img, (int(pxx - pxw/2.0), int(pxy - pxh/2.0)), (int(pxx + pxw/2.0), int(pxy + pxh/2.0)), (0, 255, 0), 2)
        ptdet = np.array([pxx, pxy])
        if np.linalg.norm(gt - ptdet) < FP_dist:
            detected_depth = True
        else:
            FPs[1] += 1
    last_img_ready_depth = True
    # cv2.imshow("window", cur_img)
    # cv2.waitKey(1)

def loc_callback(data):
    global gt
    global TPs, FPs, FNs, last_img_ready_loc, detected_loc
    # rospy.loginfo("I heard LOC: %s", data)

    if last_img_ready_loc or gt is None or (show_imgs and last_img is None):
        rospy.logwarn("Waiting for images")
        return

    pose_lo = geometry_msgs.msg.PointStamped()
    pose_lo.header = data.header
    pose_lo.point = data.pose.pose.position
    pose_tfd = tfBuffer.transform(pose_lo, "rs_d435_color_optical_frame", rospy.Duration(0.001))
    pt3 = np.array([pose_tfd.point.x, pose_tfd.point.y, pose_tfd.point.z])
    ptdet = cmodel.project3dToPixel(pt3)

    # cv2.circle(cur_img, (int(pt[0]), int(pt[1])), FP_dist, (255, 0, 0))
    pxx = ptdet[0]
    pxy = ptdet[1]
    pxr = 200/np.linalg.norm(pt3)
    if show_imgs:
        cv2.circle(last_img, (int(pxx), int(pxy)), int(pxr), (255, 0, 0), 2)
    if np.linalg.norm(gt - ptdet) < FP_dist:
        detected_loc = True
    else:
        FPs[1] += 1
    last_img_ready_loc = True
    # cv2.imshow("window", cur_img)
    # cv2.waitKey(1)

def cinfo_callback(data):
    # rospy.loginfo("I heard CINFO: %s", data)
    cmodel.fromCameraInfo(data)

def mouse_callback(event, x, y, flags, param):
    global gt
    gt = np.array((x, y))

def calc_precision(TPs, FPs):
    if TPs + FPs == 0:
        return 1
    else:
        return TPs/float(TPs + FPs)

def calc_recall(TPs, FNs):
    if TPs + FNs == 0:
        return 1
    else:
        return TPs/float(TPs + FNs)

if __name__ == '__main__':
    rospy.init_node('cnn_detect_evaluator')

    if os.path.isfile("gts.pkl"):
        with open("gts.pkl", "rb") as ifile:
            gts = pickle.load(ifile)
            gts_loaded = True
            rospy.loginfo("gts loaded")

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    bridge = CvBridge()

    rospy.Subscriber("/uav42/odometry/odom_main", nav_msgs.msg.Odometry, odom_callback)
    rospy.Subscriber("/uav42/odometry/rtk_local_odom", nav_msgs.msg.Odometry, rtk1_callback)
    rospy.Subscriber("/uav4/odometry/rtk_local_odom", nav_msgs.msg.Odometry, rtk2_callback)

    rospy.Subscriber("/uav42/cnn_detect/detections", cnn_msg.Detections, cnn_callback)
    rospy.Subscriber("/uav42/uav_detection/detections", uav_msg.Detections, depth_callback)
    rospy.Subscriber("/uav42/uav_localization/localized_uav", geometry_msgs.msg.PoseWithCovarianceStamped, loc_callback)

    rospy.Subscriber("/uav42/rs_d435/color/camera_info", sensor_msgs.msg.CameraInfo, cinfo_callback)
    rospy.Subscriber("/uav42/rs_d435/color/image_rect_color", sensor_msgs.msg.Image, img_callback)

    if show_imgs:
        cv2.namedWindow("MAV detection")
        if not gts_loaded:
            cv2.setMouseCallback("MAV detection", mouse_callback)

    max_range_cnn = 0.0
    max_range_depth = 0.0
    max_range_loc = 0.0

    rospy.loginfo("Spinning")
    while not rospy.is_shutdown():
        if last_img_ready_cnn and last_img_ready_depth:

            pt3 = get_tf()
            dist = np.linalg.norm(pt3)

            should_detect = True
            cur_t = rospy.Time.now().to_sec()
            for ts in times:
                if cur_t > ts[0] and cur_t < ts[1]:
                    should_detect = False

            if detected_cnn:
                if should_detect:
                    TPs[0] += 1
                    if dist > max_range_cnn:
                        max_range_cnn = dist
                else:
                    FPs[0] += 1
            elif should_detect:
                FNs[0] += 1

            if detected_depth:
                if should_detect:
                    TPs[1] += 1
                    if dist > max_range_depth:
                        max_range_depth = dist
                else:
                    FPs[1] += 1
            elif should_detect:
                FNs[1] += 1

            if detected_depth or detected_cnn:
                TPs[2] += 1
            else:
                FNs[2] += 1
            FPs[2] = FPs[0] + FPs[1]

            if detected_loc:
                if should_detect:
                    TPs[3] += 1
                    if dist > max_range_loc:
                        max_range_loc = dist
                else:
                    FPs[3] += 1
            elif should_detect:
                FNs[3] += 1

            precision = [0, 0, 0, 0]
            recall = [0, 0, 0, 0]
            precision[0] = calc_precision(TPs[0], FPs[0])
            recall[0] = calc_recall(TPs[0], FNs[0])
            precision[1] = calc_precision(TPs[1], FPs[1])
            recall[1] = calc_recall(TPs[1], FNs[1])
            precision[2] = calc_precision(TPs[2], FPs[2])
            recall[2] = calc_recall(TPs[2], FNs[2])
            precision[3] = calc_precision(TPs[3], FPs[3])
            recall[3] = calc_recall(TPs[3], FNs[3])

            rospy.loginfo("type \tTPs\tFPs\tFNs\tprec\t\trec\t\trange")
            rospy.loginfo("cnn  \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}\t{:f}".format(TPs[0], FPs[0], FNs[0], precision[0], recall[0], max_range_cnn))
            rospy.loginfo("depth\t{:d}\t{:d}\t{:d}\t{:f}\t{:f}\t{:f}".format(TPs[1], FPs[1], FNs[1], precision[1], recall[1], max_range_depth))
            rospy.loginfo("both \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}\t{:f}".format(TPs[2], FPs[2], FNs[2], precision[2], recall[2], np.max((max_range_cnn, max_range_depth))))
            rospy.loginfo("loc  \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}\t{:f}".format(TPs[3], FPs[3], FNs[3], precision[3], recall[3], max_range_loc))

            if show_imgs:
                cv2.circle(last_img, (int(gt[0]), int(gt[1])), FP_dist, (255, 0, 0))
                cv2.imshow("MAV detection", last_img)
                cv2.waitKey(1)
            last_img = None
            last_img_ready_cnn = False
            last_img_ready_depth = False
            last_img_ready_loc = False
    if not gts_loaded:
        with open("gts.pkl", "wb") as ofile:
            pickle.dump(gts, ofile)
        rospy.loginfo("Dumped ground-truth positions to gts.pkl")
