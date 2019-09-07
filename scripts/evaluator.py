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

def loc_to_pxpt(loc_msg, tf_buffer, cmodel):
    # pose_lo = geometry_msgs.msg.PointStamped()
    # pose_lo.header = loc_msg.header
    # pose_lo.point = loc_msg.pose.pose.position
    # pose_tfd = tf_buffer.transform(pose_lo, "rs_d435_color_optical_frame", rospy.Duration(0.1))
    pt3 = np.array([loc_msg.point.x, loc_msg.point.y, loc_msg.point.z])
    ptdet = cmodel.project3dToPixel(pt3)
    ptdet = np.array(ptdet)
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
    return (msg, it, it != prev_it)

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

    gts_loaded = False
    gts = None
    if os.path.isfile("gts.pkl"):
        with open("gts.pkl", "rb") as ifile:
            gts = pickle.load(ifile)
            gts_loaded = True
            rospy.loginfo("gts loaded")

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    bridge = CvBridge()

    show_imgs = True

    # skip_time = 70
    # skip_time_end = 90
    # FP_dist = 150
    # type      TPs     FPs     FNs     prec            rec
    # depth     1506    28      959     0.981747        0.610953
    # cnn       1108    438     1357    0.716688        0.449493
    # both      1960    466     505     0.807914        0.795132
    # dloc      1398    3       1067    0.997859        0.567140
    # cloc      1224    16      1241    0.987097        0.496552
    # bloc      1797    12      668     0.993367        0.729006

    skip_time = 70
    skip_time_end = 90
    FP_dist = 100
    # type      TPs     FPs     FNs     prec            re
    # depth     1500    35      965     0.977199        0.60851
    # cnn       1096    450     1369    0.708926        0.44462
    # both      1953    485     512     0.801066        0.79229
    # dloc      1373    28      1092    0.980014        0.55699
    # cloc      1221    19      1244    0.984677        0.49533
    # bloc      1779    30      686     0.983416        0.72170

    detector_bag = "/home/matous/bag_files/uav_detect/icra2019/_2019-09-03-15-03-02_detector.bag"
    cinfo_bag = "/home/matous/bag_files/uav_detect/icra2019/out/camera_info.bag"
    tf_bag = "/home/matous/bag_files/uav_detect/icra2019/out/tf.bag"
    depth_bag = "/home/matous/bag_files/uav_detect/icra2019/out/depth_detections.bag"
    cnn_bag = "/home/matous/bag_files/uav_detect/icra2019/out/cnn_detections.bag"
    dloc_bag = "/home/matous/bag_files/uav_detect/icra2019/out/localizations_depth_tfd.bag"
    cloc_bag = "/home/matous/bag_files/uav_detect/icra2019/out/localizations_cnn_tfd.bag"
    bloc_bag = "/home/matous/bag_files/uav_detect/icra2019/out/localizations_both_tfd.bag"

    cinfo_msg = load_rosbag_msg(cinfo_bag, "/uav42/rs_d435/color/camera_info", skip_time=0)
    # tf_static_msgs = load_rosbag_msgs(tf_bag, "/tf_static", skip_time=0)
    # tf_msgs = load_rosbag_msgs(tf_bag, "/tf", skip_time=skip_time, skip_time_end=skip_time_end)
    depth_msgs = load_rosbag_msgs(depth_bag, "/uav42/uav_detection/detections", skip_time=skip_time, skip_time_end=skip_time_end)
    cnn_msgs = load_rosbag_msgs(cnn_bag, "/uav42/cnn_detect/detections", skip_time=skip_time, skip_time_end=skip_time_end)
    dloc_msgs = load_rosbag_msgs(dloc_bag, "/uav42/uav_localization/localized_uav/tfd", skip_time=skip_time, skip_time_end=skip_time_end)
    cloc_msgs = load_rosbag_msgs(cloc_bag, "/uav42/uav_localization/localized_uav/tfd", skip_time=skip_time, skip_time_end=skip_time_end)
    bloc_msgs = load_rosbag_msgs(bloc_bag, "/uav42/uav_localization/localized_uav/tfd", skip_time=skip_time, skip_time_end=skip_time_end)

    # tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=10)
    # tf_static_pub = rospy.Publisher("/tf_static", TFMessage, queue_size=10)
    # for msg in tf_static_msgs:
        # tf_static_pub.publish(msg)

    if show_imgs:
        cv2.namedWindow("MAV detection", cv2.WINDOW_GUI_EXPANDED)
        if not gts_loaded:
            cv2.setMouseCallback("MAV detection", mouse_callback)

    cmodel = PinholeCameraModel()
    cmodel.fromCameraInfo(cinfo_msg)

#            left view,          returned to view
    times = [(1567515829.515217, 1567515832.846460),
             (1567515970.712066, 1567515972.843928),
             (1567516025.318885, 9999999999.999999)]
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
    t_start = None

    for topic, img_msg, cur_stamp in bag.read_messages(topics="/uav42/rs_d435/color/image_rect_color", start_time=start_time, end_time=end_time):
        cur_t = img_msg.header.stamp.to_sec()
        if t_start is None:
            t_start = cur_t

        # (tf_msg, tf_it) = find_msg_tf(tf_msgs, img_msg.header.stamp, tf_it)
        (depth_msg, depth_it, depth_updated) = find_msg(depth_msgs, img_msg.header.stamp, depth_it)
        (cnn_msg, cnn_it, cnn_updated) = find_msg(cnn_msgs, img_msg.header.stamp, cnn_it)
        (dloc_msg, dloc_it, dloc_updated) = find_msg(dloc_msgs, img_msg.header.stamp, dloc_it)
        (cloc_msg, cloc_it, cloc_updated) = find_msg(cloc_msgs, img_msg.header.stamp, cloc_it)
        (bloc_msg, bloc_it, bloc_updated) = find_msg(bloc_msgs, img_msg.header.stamp, bloc_it)
        (gt, gt_it) = find_msg_t(gts, cur_t, gt_it)
        gt = np.array(gt)

        # tf_pub.publish(tf_msg)

        should_detect = True
        for ts in times:
            if cur_t > ts[0] and cur_t < ts[1]:
                should_detect = False

        detected = [False]*N

        img = None
        if show_imgs:
            try:
              img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            except CvBridgeError as e:
              print(e)

        # #{ evaluate depth
        if depth_updated:
            for det in depth_msg.detections:
                w = det.roi.width
                h = det.roi.height
                pxx = w*det.x
                pxy = h*det.y
                pxr = 500/det.depth
                pxw = pxr
                pxh = pxr
                if show_imgs:
                    cv2.rectangle(img, (int(pxx - pxw/2.0), int(pxy - pxh/2.0)), (int(pxx + pxw/2.0), int(pxy + pxh/2.0)), (0, 255, 0), 2)
                ptdet = np.array([pxx, pxy])
                if np.linalg.norm(gt - ptdet) < FP_dist:
                    detected[0] = True
                else:
                    FPs[0] += 1
        # #} end of evaluate depth
        
        # #{ evaluate cnn
        if cnn_updated:
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
        if depth_updated or cnn_updated:
            if detected[0] or detected[1]:
                detected[2] = True
            FPs[2] = FPs[0] + FPs[1]
        # #} end of evaluate depth
        
        # #{ evaluate dloc
        if dloc_updated:
            dloc = loc_to_pxpt(dloc_msg, tf_buffer, cmodel)
            # if show_imgs:
            #     cv2.circle(img, (int(dloc[0]), int(dloc[1])), 40, (0, 255, 0), 2)
            if np.linalg.norm(gt - dloc) < FP_dist:
                detected[3] = True
            else:
                FPs[3] += 1
        # #} end of evaluate dloc

        # #{ evaluate cloc
        if cloc_updated:
            cloc = loc_to_pxpt(cloc_msg, tf_buffer, cmodel)
            # if show_imgs:
            #     cv2.circle(img, (int(cloc[0]), int(cloc[1])), 40, (0, 0, 255), 2)
            if np.linalg.norm(gt - cloc) < FP_dist:
                detected[4] = True
            else:
                FPs[4] += 1
        # #} end of evaluate cloc

        # #{ evaluate bloc
        if bloc_updated:
            bloc = loc_to_pxpt(bloc_msg, tf_buffer, cmodel)
            # if show_imgs:
            #     cv2.circle(img, (int(bloc[0]), int(bloc[1])), 40, (255, 0, 0), 2)
            if np.linalg.norm(gt - bloc) < FP_dist:
                detected[5] = True
            else:
                FPs[5] += 1
        # #} end of evaluate bloc

        pxx = 30
        # if show_imgs:
        #     cv2.putText(img, "TPs\tFPs\tFNs", (pxx, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
        for it in range(0, N):
            pxx = 30
            pxy = 60 + it*30
            # if show_imgs:
            #     cv2.putText(img, "{:d}\t{:d}\t{:d}".format(TPs[it], FPs[it], FNs[it]), (pxx, pxy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            pxx = 400
            if detected[it]:
                if should_detect:
                    # if show_imgs:
                    #     cv2.putText(img, "TP", (pxx, pxy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                    TPs[it] += 1
                else:
                    # if show_imgs:
                    #     cv2.putText(img, "FP!!", (pxx, pxy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                    FPs[it] += 1
            elif should_detect:
                FNs[it] += 1
                # if show_imgs:
                #     cv2.putText(img, "FN!!!!!", (pxx, pxy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

            precision[it] = calc_precision(TPs[it], FPs[it])
            recall[it] = calc_recall(TPs[it], FNs[it])

        rospy.loginfo("time {:f}".format(cur_t - t_start))
        rospy.loginfo("type \tTPs\tFPs\tFNs\tprec\t\trec")
        rospy.loginfo("depth\t{:d}\t{:d}\t{:d}\t{:f}\t{:f}".format(TPs[0], FPs[0], FNs[0], precision[0], recall[0]))
        rospy.loginfo("cnn  \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}".format(TPs[1], FPs[1], FNs[1], precision[1], recall[1]))
        rospy.loginfo("both \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}".format(TPs[2], FPs[2], FNs[2], precision[2], recall[2]))
        rospy.loginfo("dloc \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}".format(TPs[3], FPs[3], FNs[3], precision[3], recall[3]))
        rospy.loginfo("cloc \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}".format(TPs[4], FPs[4], FNs[4], precision[4], recall[4]))
        rospy.loginfo("bloc \t{:d}\t{:d}\t{:d}\t{:f}\t{:f}".format(TPs[5], FPs[5], FNs[5], precision[5], recall[5]))

        if show_imgs:
            # cv2.circle(img, (int(gt[0]), int(gt[1])), FP_dist, (0, 0, 0))
            cv2.imshow("MAV detection", img)
            key = cv2.waitKey(1)
            if key == ord('p'):
                waiting = True
                while waiting:
                    cv2.imshow("MAV detection", img)
                    key = cv2.waitKey(100)
                    if key == ord('r'):
                        waiting = False

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
