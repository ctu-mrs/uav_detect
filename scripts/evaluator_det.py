#!/usr/bin/env python
import rospy
# import tf
# from tf.transformations import quaternion_from_euler
import rosbag
from image_geometry import PinholeCameraModel

import pickle
import numpy as np
from numpy import cos
from numpy import sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv

# #{ 

def load_csv_data(csv_fname):
    rospy.loginfo("Using CSV file {:s}".format(csv_fname))

    n_pos = sum(1 for line in open(csv_fname)) - 1
    positions = np.zeros((n_pos, 3))
    times = np.zeros((n_pos,))
    it = 0
    with open(csv_fname, 'r') as fhandle:
        first_loaded = False
        csvreader = csv.reader(fhandle, delimiter=',')
        for row in csvreader:
            if not first_loaded:
                first_loaded = True
                continue
            positions[it, :] = np.array([float(row[0]), float(row[1]), float(row[2])])
            times[it] = float(row[3])
            it += 1
    return (positions, times)

def msgs_to_pos(msgs):
    positions = list()
    for msg in msgs:
        for pos in msg.positions:
            positions.append(pos)
    return np.matrix(positions)

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

class msg:
    def __init__(self, time, positions):
        self.time = time
        self.positions = positions

def process_msgs(msgs, shift=None):
    ret = len(msgs)*[None]

    offset = None
    if shift is None:
        offset = np.array([0, 0, 0])
        shift = np.array([0, 0, 0])

    for it in range(0, len(msgs)):
        cur_msg = msgs[it]
        out_msg = msg(cur_msg.header.stamp.to_sec(), list())
        if hasattr(cur_msg, "points"):
            for det in cur_msg.points:
                xyz = np.array([det.x, det.y, det.z])
                if offset is None:
                    offset = xyz
                xyz = xyz - offset + shift
                # xyz = np.dot(xyz, R)
                out_msg.positions.append(xyz)
        else:
            x = cur_msg.pose.pose.position.x
            y = cur_msg.pose.pose.position.y
            z = cur_msg.pose.pose.position.z
            xyz = np.array([x, y, z])
            if offset is None:
                offset = xyz
            xyz = xyz - offset + shift
            out_msg.positions.append(xyz)
        ret[it] = out_msg
    return ret


def find_closest(time, msgs):
    closest_it = 0
    closest_diff = float('Inf')
    for it in range(0, len(msgs)):
        cur_time = msgs[it].time
        cur_diff = abs(time - cur_time)
        # print(cur_diff, closest_diff)
        if cur_diff <= closest_diff:
            closest_it = it
            closest_diff = abs(time - cur_time)
        else:
            break
    return (closest_it, closest_diff)


def find_closest_pos(pos, positions):
    closest_pos = None
    closest_diff = float('Inf')
    for cpos in positions:
        cur_diff = np.linalg.norm(cpos - pos)
        # print(cur_diff, closest_diff)
        if cur_diff <= closest_diff:
            closest_pos = cpos
            closest_diff = cur_diff
        else:
            break
    return closest_pos


def calc_statistics(positions1, times1, msgs, FP_error):
    TPs = 0
    TNs = 0
    FPs = 0
    FNs = 0

    max_dt = 0.15
    errors = len(positions1)*[None]
    for it in range(0, len(positions1)):
        time1 = times1[it]
        (closest_it, closest_diff) = find_closest(time1, msgs)
        if closest_diff > max_dt:
            FNs += 1
            continue

        if len(msgs[closest_it].positions) > 0:
            closest_pos = find_closest_pos(positions1[it, :], msgs[closest_it].positions)
            cur_err = np.linalg.norm(positions1[it, :] - closest_pos)
            if closest_pos[1] > 5 or cur_err > FP_error:
                FPs += len(msgs[closest_it].positions)
            else:
                FPs += len(msgs[closest_it].positions) - 1
                TPs += 1
                errors[it] = np.linalg.norm(positions1[it, :] - closest_pos)
        else:
            FNs += 1

    errors = np.array(errors, dtype=float)
    nn_errors = errors[~np.isnan(errors)]
    maxerr = float("NaN")
    meanerr = float("NaN")
    stderr = float("NaN")
    if len(nn_errors) > 0:
        maxerr = np.max(nn_errors)
        meanerr = np.mean(nn_errors)
        stderr = np.std(nn_errors)
    rospy.loginfo("Max. error: {:f}".format(maxerr))
    rospy.loginfo("Mean error: {:f}, std.: {:f}".format(meanerr, stderr))
    return (TPs, TNs, FPs, FNs)

# #} end of 

def main():
    rospy.init_node('localization_evaluator', anonymous=True)
    # out_fname = rospy.get_param('~output_filename')
    # in_fname = rospy.get_param('~input_filename')
    det_bag_fname = rospy.get_param('~detection_bag_name')
    gt_bag_fname = rospy.get_param('~ground_truth_bag_name')
    det_topic_name = rospy.get_param('~detection_topic_name')
    gt_topic_name = rospy.get_param('~ground_truth_topic_name')
    loc_out_fname = rospy.get_param('~localization_out_fname')
    gt_out_fname = rospy.get_param('~ground_truth_out_fname')

    # msgs = load_pickle(in_fname)
    FP_error = 666.0 # meters

    rospy.loginfo("Loading data from rosbags {:s}".format(det_bag_fname))
    skip_time = 35
    skip_time_end = 70
    det_msgs = load_rosbag_msgs(det_bag_fname, det_topic_name, skip_time=skip_time, skip_time_end=skip_time_end)

    rospy.loginfo("Loading data from rosbags {:s}".format(gt_bag_fname))
    gt_msgs = load_rosbag_msgs(gt_bag_fname, gt_topic_name, skip_time=skip_time, skip_time_end=skip_time_end)

    if det_msgs is None or gt_msgs is None:
        exit(1)

    rospy.loginfo("Loaded {:d} localization messages".format(len(det_msgs)))
    msgs = process_msgs(det_msgs, gt_positions[0, :] + np.array([1, -1.1, 0]))
    det_positions = msgs_to_pos(msgs)


    rospy.loginfo("Done loading positions")

    TPs, TNs, FPs, FNs = calc_statistics(gt_positions, gt_times, msgs, FP_error)
    rospy.loginfo("TPs, TNs, FPs, FNs: {:d}, {:d}, {:d}, {:d}".format(TPs, TNs, FPs, FNs))

    precision = TPs/float(TPs + FPs)
    recall = TPs/float(TPs + FNs)
    rospy.loginfo("precision, recall: {:f}, {:f}".format(precision, recall))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    loc_x = np.array(det_positions[:, 0]).flatten()
    loc_y = np.array(det_positions[:, 1]).flatten()
    loc_z = np.array(det_positions[:, 2]).flatten()
    rospy.loginfo(loc_x.shape)
    rospy.loginfo(loc_y.shape)
    rospy.loginfo(loc_z.shape)
    ax.plot(loc_x, loc_y, loc_z, 'g.')
    ax.plot([det_positions[0, 0]], [det_positions[0, 1]], [det_positions[0, 2]], 'gx')
    ax.plot([det_positions[-1, 0]], [det_positions[-1, 1]], [det_positions[-1, 2]], 'go')

    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'b.')
    ax.plot([gt_positions[0, 0]], [gt_positions[0, 1]], [gt_positions[0, 2]], 'rx')
    ax.plot([gt_positions[-1, 0]], [gt_positions[-1, 1]], [gt_positions[-1, 2]], 'ro')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_aspect('equal')
    plt.xlim([-10, 60])
    plt.ylim([-20, 20])
    plt.show()

if __name__ == '__main__':
    main()
