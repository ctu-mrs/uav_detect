# Markerless MAV detection project

The project contains two variants of MAV detection, which are located in the different git branches:
 - `depth`: This is the depthmap-based MAV detection method, as described in the paper [1].
 - `cnn`: This is the convolutional neural network-based MAV detection method, as described in the paper [2].
 - `mbzirc`: This is the LiDAR-based detection of an MAV with a suspended ball for the MBZIRC2020 competition, as described in the paper [3].
 - `master` (this branch): Contains only this readme and no code.

All the variants are meant to be compiled as a ROS package using the catkin build system (the tested ROS version is ROS Melodic).
Another common requirement is the [`mrs_lib`](https://github.com/ctu-mrs/mrs_lib) code library (contains utility functions eg. for parameter loading from `rosparam` server etc. and is documented in https://ctu-mrs.github.io/mrs_lib/).
This project is part of the [MRS UAV system](https://github.com/ctu-mrs/mrs_uav_system) (which is documented at [ctu-mrs.github.io](https://ctu-mrs.github.io) and in the paper [4]).
For more information about the different methods, checkout the respective git branch.

----
References:
 * [1]: M. Vrba, D. Heřt and M. Saska, "Onboard Marker-Less Detection and Localization of Non-Cooperating Drones for Their Safe Interception by an Autonomous Aerial System," in IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3402-3409, Oct. 2019, doi: 10.1109/LRA.2019.2927130.
 * [2]: M. Vrba and M. Saska, "Marker-Less Micro Aerial Vehicle Detection and Localization Using Convolutional Neural Networks," in IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 2459-2466, April 2020, doi: 10.1109/LRA.2020.2972819.
 * [3]: M. Vrba, Y. Stasinchuk, T. Báča, V. Spurný, M. Petrlík, D. Heřt, D. Žaitlík and M. Saska, "Autonomous Capturing of Agile Flying Objects using MAVs: The *MBZIRC 2020* Challenge", submitted to the IEEE Transactions on Systems, Man and Cybernetics - Systems 2021.
 * [4]: T. Báča, M. Petrlík, M. Vrba, V. Spurný, R. Pěnička, D. Heřt and M. Saska, "The MRS UAV System: Pushing the Frontiers of Reproducible Research, Real-world Deployment, and Education with Autonomous Unmanned Aerial Vehicles", eprint arXiv: 2008.08050, August 2020 (https://arxiv.org/abs/2008.08050).
