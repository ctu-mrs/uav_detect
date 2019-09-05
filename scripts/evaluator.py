#!/usr/bin/env python  
import rospy

import math
import tf2_ros
from tf2_msgs.msg import TFMessage
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
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point

# #{ 

show_imgs = False

def loc_to_pxpt(loc_msg, tf_buffer, cmodel):
    pose_lo = geometry_msgs.msg.PointStamped()
    pose_lo.header = loc_msg.header
    pose_lo.point = loc_msg.pose.pose.position
    pose_tfd = tf_buffer.transform(pose_lo, "rs_d435_color_optical_frame", rospy.Duration(0.1))
    pt3 = np.array([pose_tfd.point.x, pose_tfd.point.y, pose_tfd.point.z])
    ptdet = cmodel.project3dToPixel(pt3)
    return ptdet

# #} end of 

# #{ 

def load_rosbag_msgs(bag_fname, topic, skip_time=0, skip_time_end=0):
    rospy.loginfo("Using rosbag {:s}".format(bag_fname))
    bag = rosbag.Bag(bag_fname)
    n_msgs = bag.get_message_count(topic_filters=topic)
    if n_msgs == 0:
        rospy.logerr("No messages from topic {:s} in bag".format(topic))
    else:
        rospy.loginfo("Loading {:d} messages".format(n_msgs))
    msgs = n_msgs*[None]

    skip = rospy.Duration.from_sec(skip_time)
    start_time = rospy.Time.from_sec(bag.get_start_time()) + skip
    skip_end = rospy.Duration.from_sec(skip_time_end)
    end_time = rospy.Time.from_sec(bag.get_end_time()) - skip_end
    it = 0
    for topic, msg, cur_stamp in bag.read_messages(topics=topic, start_time=start_time, end_time=end_time):
        if rospy.is_shutdown():
            break
        msgs[it] = msg
        it += 1
    return msgs[0:it]

def load_rosbag_msg(bag_fname, topic, skip_time=0, skip_time_end=0):
    rospy.loginfo("Using rosbag {:s}".format(bag_fname))
    bag = rosbag.Bag(bag_fname)

    skip = rospy.Duration.from_sec(skip_time)
    start_time = rospy.Time.from_sec(bag.get_start_time()) + skip
    skip_end = rospy.Duration.from_sec(skip_time_end)
    end_time = rospy.Time.from_sec(bag.get_end_time()) - skip_end
    for topic, msg, cur_stamp in bag.read_messages(topics=topic, start_time=start_time, end_time=end_time):
        if rospy.is_shutdown():
            break
        return msg
    rospy.logerr("No messages from topic {:s} in bag".format(topic))
    return None

def find_msg(msgs, stamp, prev_it):
    it = prev_it
    if it >= len(msgs):
        rospy.logerr("no more messages!")
        return (None, None)
    msg = msgs[it]
    while msg.header.stamp < stamp:
        it += 1
        if it >= len(msgs):
            rospy.logerr("no more messages!")
            return (None, None)
        msg = msgs[it]
    return (msg, it)

def find_msg_tf(msgs, stamp, prev_it):
    it = prev_it
    if it >= len(msgs):
        rospy.logerr("no more messages!")
        return (None, None)
    msg = msgs[it]
    while msg.transforms[0].header.stamp < stamp:
        it += 1
        if it >= len(msgs):
            rospy.logerr("no more messages!")
            return (None, None)
        msg = msgs[it]
    return (msg, it)

def find_msg_t(msgs, time, prev_it):
    it = prev_it
    if it >= len(msgs)-1:
        rospy.logerr("no more messages!")
        return (None, None)
    msg = msgs[it]
    while msg[0] < time:
        it += 1
        if it >= len(msgs)-1:
            rospy.logerr("no more messages!")
            return (None, None)
        msg = msgs[it]
    return (msg[1:], it)

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

# #} end of 

if __name__ == '__main__':
    rospy.init_node('cnn_detect_evaluator')

    if os.path.isfile("gts.pkl"):
        with open("gts.pkl", "rb") as ifile:
            gts = pickle.load(ifile)
            gts_loaded = True
            rospy.loginfo("gts loaded")

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    bridge = CvBridge()

    skip_time = 50
    skip_time_end = 40

    detector_bag = "/home/matous/bag_files/uav_detect/icra2019/_2019-09-03-15-03-02_detector.bag"
    cinfo_bag = "/home/matous/bag_files/uav_detect/icra2019/out/camera_info.bag"
    tf_bag = "/home/matous/bag_files/uav_detect/icra2019/out/tf.bag"
    depth_bag = "/home/matous/bag_files/uav_detect/icra2019/out/depth_detections.bag"
    dloc_bag = "/home/matous/bag_files/uav_detect/icra2019/out/localizations_depth.bag"
    cnn_bag = "/home/matous/bag_files/uav_detect/icra2019/out/cnn_detections.bag"
    cloc_bag = "/home/matous/bag_files/uav_detect/icra2019/out/localizations_cnn.bag"
    bloc_bag = "/home/matous/bag_files/uav_detect/icra2019/out/localizations_both.bag"

    cinfo_msg = load_rosbag_msg(cinfo_bag, "/uav42/rs_d435/color/camera_info", skip_time=0)
    tf_static_msgs = load_rosbag_msgs(tf_bag, "/tf_static", skip_time=0)
    tf_msgs = load_rosbag_msgs(tf_bag, "/tf", skip_time=skip_time, skip_time_end=skip_time_end)
    depth_msgs = load_rosbag_msgs(depth_bag, "/uav42/uav_detection/detections", skip_time=skip_time, skip_time_end=skip_time_end)
    cnn_msgs = load_rosbag_msgs(cnn_bag, "/uav42/cnn_detect/detections", skip_time=skip_time, skip_time_end=skip_time_end)
    dloc_msgs = load_rosbag_msgs(dloc_bag, "/uav42/uav_localization/localized_uav", skip_time=skip_time, skip_time_end=skip_time_end)
    cloc_msgs = load_rosbag_msgs(cloc_bag, "/uav42/uav_localization/localized_uav", skip_time=skip_time, skip_time_end=skip_time_end)
    bloc_msgs = load_rosbag_msgs(bloc_bag, "/uav42/uav_localization/localized_uav", skip_time=skip_time, skip_time_end=skip_time_end)

    tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=10)
    tf_static_pub = rospy.Publisher("/tf_static", TFMessage, queue_size=10)
    for msg in tf_static_msgs:
        tf_static_pub.publish(msg)

    if show_imgs:
        cv2.namedWindow("MAV detection")
        if not gts_loaded:
            cv2.setMouseCallback("MAV detection", mouse_callback)

    cmodel = PinholeCameraModel()
    cmodel.fromCameraInfo(cinfo_msg)

#            left view,          returned to view
    times = [(1567515829.515217, 1567515832.846460),
             (1567515970.712066, 1567515972.843928),
             (1567516025.318885, 9999999999.999999)]
    FP_dist = 100
    N = 6

    depth_it = 0
    cnn_it = 0
    dloc_it = 0
    cloc_it = 0
    bloc_it = 0
    gt_it = 0
    tf_it = 0

    bag = rosbag.Bag(detector_bag)
    skip = rospy.Duration.from_sec(skip_time)
    start_time = rospy.Time.from_sec(bag.get_start_time()) + skip
    skip_end = rospy.Duration.from_sec(skip_time_end)
    end_time = rospy.Time.from_sec(bag.get_end_time()) - skip_end

    rospy.loginfo("Initialization done!")
    #########################################################################################################################################################

    TPs = [0]*N
    FPs = [0]*N
    FNs = [0]*N
    precision = [0]*N
    recall = [0]*N

    for topic, img_msg, cur_stamp in bag.read_messages(topics="/uav42/rs_d435/color/image_rect_color", start_time=start_time, end_time=end_time):
        cur_t = img_msg.header.stamp.to_sec()

        (tf_msg, tf_it) = find_msg_tf(tf_msgs, img_msg.header.stamp, tf_it)
        (depth_msg, depth_it) = find_msg(depth_msgs, img_msg.header.stamp, depth_it)
        (cnn_msg, cnn_it) = find_msg(cnn_msgs, img_msg.header.stamp, cnn_it)
        (dloc_msg, dloc_it) = find_msg(dloc_msgs, img_msg.header.stamp, dloc_it)
        (cloc_msg, cloc_it) = find_msg(cloc_msgs, img_msg.header.stamp, cloc_it)
        (bloc_msg, bloc_it) = find_msg(bloc_msgs, img_msg.header.stamp, bloc_it)
        (gt, gt_it) = find_msg_t(gts, cur_t, gt_it)

        tf_pub.publish(tf_msg)

        should_detect = True
        for ts in times:
            if cur_t > ts[0] and cur_t < ts[1]:
                should_detect = False

        detected = [False]*N

        # #{ evaluate depth
        for det in depth_msg.detections:
            w = det.roi.width
            h = det.roi.height
            pxx = w*det.x
            pxy = h*det.y
            pxr = 500/det.depth
            pxw = pxr
            pxh = pxr
            if show_imgs:
                cv2.rectangle(img, (int(pxx - pxw/2.0), int(pxy - pxh/2.0)), (int(pxx + pxw/2.0), int(pxy + pxh/2.0)), (0, 0, 255), 2)
            ptdet = np.array([pxx, pxy])
            if np.linalg.norm(gt - ptdet) < FP_dist:
                detected[0] = True
            else:
                FPs[0] += 1
        # #} end of evaluate depth
        
        # #{ evaluate cnn
        for det in cnn_msg.detections:
            w = det.roi.width
            h = det.roi.height
            pxx = w*det.x
            pxy = h*det.y
            pxw = w*det.width
            pxh = h*det.height
            if show_imgs:
                cv2.rectangle(img, (int(pxx - pxw/2.0), int(pxy - pxh/2.0)), (int(pxx + pxw/2.0), int(pxy + pxh/2.0)), (0, 0, 255), 2)
            ptdet = np.array([pxx, pxy])
            if np.linalg.norm(gt - ptdet) < FP_dist:
                detected[1] = True
            else:
                FPs[1] += 1
        # #} end of evaluate depth
        
        # #{ evaluate both
        if detected[0] or detected[1]:
            detected[2] = True
        FPs[2] = FPs[0] + FPs[1]
        # #} end of evaluate depth
        
        # #{ evaluate dloc
        dloc = loc_to_pxpt(dloc_msg, tf_buffer, cmodel)
        if show_imgs:
            cv2.circle(img, (int(dloc[0]), int(dloc[1])), 40, (0, 0, 255), 2)
        if np.linalg.norm(gt - dloc) < FP_dist:
            detected[3] = True
        else:
            FPs[3] += 1
        # #} end of evaluate dloc

        # #{ evaluate cloc
        cloc = loc_to_pxpt(cloc_msg, tf_buffer, cmodel)
        if show_imgs:
            cv2.circle(img, (int(cloc[0]), int(cloc[1])), 40, (0, 0, 255), 2)
        if np.linalg.norm(gt - cloc) < FP_dist:
            detected[4] = True
        else:
            FPs[4] += 1
        # #} end of evaluate cloc

        # #{ evaluate bloc
        bloc = loc_to_pxpt(bloc_msg, tf_buffer, cmodel)
        if show_imgs:
            cv2.circle(img, (int(bloc[0]), int(bloc[1])), 40, (0, 0, 255), 2)
        if np.linalg.norm(gt - bloc) < FP_dist:
            detected[5] = True
        else:
            FPs[5] += 1
        # #} end of evaluate bloc

        for it in range(0, N):
            if detected[it]:
                if should_detect:
                    TPs[it] += 1
                else:
                    FPs[it] += 1
            elif should_detect:
                FNs[it] += 1

            precision[it] = calc_precision(TPs[it], FPs[it])
            recall[it] = calc_recall(TPs[it], FNs[it])

        rospy.loginfo("type \tTPs\tFPs\tFNs\tprec\t\trec")
        rospy.loginfo("depth\t{:d}\t{:d}\t{:d}\t{:f}\t{:f}".format(TPs[0], FPs[0], FNs[0], precision[0], recall[0]))
        rospy.loginfo("cnn  \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}".format(TPs[1], FPs[1], FNs[1], precision[1], recall[1]))
        rospy.loginfo("both \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}".format(TPs[2], FPs[2], FNs[2], precision[2], recall[2]))

        if rospy.is_shutdown():
            break
            # rospy.loginfo("loc  \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}\t{:f}".format(TPs[3], FPs[3], FNs[3], precision[3], recall[3], max_range_loc))

    # while not rospy.is_shutdown():
    #     if last_img_ready_cnn and last_img_ready_depth:

    #         pt3 = get_tf()
    #         dist = np.linalg.norm(pt3)


    #         precision = [0, 0, 0, 0]
    #         recall = [0, 0, 0, 0]
    #         precision[0] = calc_precision(TPs[0], FPs[0])
    #         recall[0] = calc_recall(TPs[0], FNs[0])
    #         precision[1] = calc_precision(TPs[1], FPs[1])
    #         recall[1] = calc_recall(TPs[1], FNs[1])
    #         precision[2] = calc_precision(TPs[2], FPs[2])
    #         recall[2] = calc_recall(TPs[2], FNs[2])
    #         precision[3] = calc_precision(TPs[3], FPs[3])
    #         recall[3] = calc_recall(TPs[3], FNs[3])

    #         rospy.loginfo("type \tTPs\tFPs\tFNs\tprec\t\trec\t\trange")
    #         rospy.loginfo("cnn  \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}\t{:f}".format(TPs[0], FPs[0], FNs[0], precision[0], recall[0], max_range_cnn))
    #         rospy.loginfo("depth\t{:d}\t{:d}\t{:d}\t{:f}\t{:f}\t{:f}".format(TPs[1], FPs[1], FNs[1], precision[1], recall[1], max_range_depth))
    #         rospy.loginfo("both \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}\t{:f}".format(TPs[2], FPs[2], FNs[2], precision[2], recall[2], np.max((max_range_cnn, max_range_depth))))
    #         rospy.loginfo("loc  \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}\t{:f}".format(TPs[3], FPs[3], FNs[3], precision[3], recall[3], max_range_loc))

    #         if show_imgs:
    #             cv2.circle(last_img, (int(gt[0]), int(gt[1])), FP_dist, (255, 0, 0))
    #             cv2.imshow("MAV detection", last_img)
    #             cv2.waitKey(1)
    #         last_img = None
    #         last_img_ready_cnn = False
    #         last_img_ready_depth = False
    #         last_img_ready_loc = False
    # if not gts_loaded:
    #     with open("gts.pkl", "wb") as ofile:
    #         pickle.dump(gts, ofile)
    #     rospy.loginfo("Dumped ground-truth positions to gts.pkl")
