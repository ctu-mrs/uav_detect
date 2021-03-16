# Detection and localization of a MAV with a suspended ball for the MBZIRC 2020 competition

Intended to be used with the Ouster OS1-Gen1 64-line LiDAR sensor.
Other related packages:
 * the filtration & estimation algorithm: [`ball_filter`](https://github.com/ctu-mrs/mbzirc2020_ball_filter) [1]
 * the planning algorithm: [`ball_planner`](https://github.com/ctu-mrs/mbzirc2020_ball_planner) [1]
 * the MRS UAV control and estimation pipeline: [`mrs_uav_system`](https://github.com/ctu-mrs/mrs_uav_system) [2]

## The working principle

Principles of the algorithms implemented in this repository are described in [1].

## Description of the provided interface and other info

### The following launchfiles are provided:
 * **detect_pcl.launch**: Starts the detection and localization nodelet.

### The following config files are used by the nodes:
 * **detection_params_pcl.yaml:** Contains parameters for the algorithm, tuned for the MBZIRC 2020 competition at real-world data. Parameters are documented in the file itself.
Most parameters (those which make sense) from the above files are dynamically reconfigurable.

### To launch simulation, detection, localization and visualization:
See the [`ball_planner`](https://github.com/ctu-mrs/mbzirc2020_ball_planner) package to launch the simulation.

----
References:
 * [1]: M. Vrba, Y. Stasinchuk, T. Báča, V. Spurný, M. Petrlík, D. Heřt, D. Žaitlík and M. Saska, "Autonomous Capturing of Agile Flying Objects using MAVs: The *MBZIRC 2020* Challenge", submitted to the IEEE Transactions on Systems, Man and Cybernetics - Systems 2021.
 * [2]: T. Báča, M. Petrlík, M. Vrba, V. Spurný, R. Pěnička, D. Heřt and M. Saska, "The MRS UAV System: Pushing the Frontiers of Reproducible Research, Real-world Deployment, and Education with Autonomous Unmanned Aerial Vehicles", eprint arXiv: 2008.08050, August 2020 (https://arxiv.org/abs/2008.08050).
