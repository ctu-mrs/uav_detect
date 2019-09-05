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
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point

gt_pos = None
def mouse_callback(event, x, y, flags, param):
    global gt_pos
    gt_pos = np.array((x, y))

if __name__ == '__main__':
    rospy.init_node('cnn_detect_evaluator')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    bridge = CvBridge()

    skip_time = 50
    skip_time_end = 40

    detector_bag = "/home/matous/bag_files/uav_detect/icra2019/_2019-09-03-15-03-02_detector.bag"

    cv2.namedWindow("MAV detection")
    cv2.setMouseCallback("MAV detection", mouse_callback)

#            left view,          returned to view
    times = [(1567515829.515217, 1567515832.846460),
             (1567515970.712066, 1567515972.843928),
             (1567516025.318885, 9999999999.999999)]
    FP_dist = 100

    img_topic = "/uav42/rs_d435/color/image_rect_color"
    bag = rosbag.Bag(detector_bag)
    n_msgs = bag.get_message_count(topic_filters=img_topic)
    skip = rospy.Duration.from_sec(skip_time)
    start_time = rospy.Time.from_sec(bag.get_start_time()) + skip
    skip_end = rospy.Duration.from_sec(skip_time_end)
    end_time = rospy.Time.from_sec(bag.get_end_time()) - skip_end

    rospy.loginfo("Initialization done!")
    #########################################################################################################################################################

    gt_it = 0
    gts = [(None, None, None)]*n_msgs
    ready = False
    for topic, img_msg, cur_stamp in bag.read_messages(topics=img_topic, start_time=start_time, end_time=end_time):

        cur_t = img_msg.header.stamp.to_sec()
        img = None

        try:
          img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
          print(e)

        while gt_pos is None or not ready:
            cv2.imshow("MAV detection", img)
            k = cv2.waitKey(10)
            if k == ord('r'):
                ready = True

        cv2.circle(img, (int(gt_pos[0]), int(gt_pos[1])), FP_dist, (255, 0, 0), 2)
        cv2.imshow("MAV detection", img)
        cv2.waitKey(10)
        gt = (cur_t, gt_pos[0], gt_pos[1])
        print(gt)
        gts[gt_it] = gt
        gt_it += 1

        if rospy.is_shutdown():
            exit(130)

    with open("gts.pkl", "wb") as ofile:
        pickle.dump(gts, ofile)
    rospy.loginfo("Dumped ground-truth positions to gts.pkl")
