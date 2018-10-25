# DEPRECATED: use branch `depth` for the latest version

--------------------------------------------------------------------------------------------------------------------------------------

# MAV camera detection using tiny-yolo neural network, Darknet, ROS and OpenCL.

This repository contains the different variants of the detection and relative localization system.
The variants are separated in different branches:
 - leader_follower: contains the most up-to-date code for detection and relative localization - this code was used in the experiments
 - master: contains this overview readme file and out of date code, including an attempt at an implementation of the Kalman Filter
 - no_darknet: compilible without the Darknet NN framework, used for evaluation of experimental results, includes evaluation code
 - particle: contains an attempt at an implementation of a particle filter, out of date (code contains even some mistakes)
 - testing: code for testing on a dataset or in a simulation
 - testing_distance: out of date code for testing on a dataset or in a simulation

All the variants are meant to be compiled as a ROS package using the catkin build system.
The tested ROS version is ROS Kinetic.

For details on how to install Darknet with OpenCL on MRS MAV platform, see https://mrs.felk.cvut.cz/gitlab/vrbamato/yolo_ocl_install.

