# Detection and localization of a MAV with a suspended ball for the MBZIRC 2020 competition

Intended to be used with the Ouster OS1-Gen1 64-line LiDAR sensor.

## The working principle

Principles of the algorithms, implemented in this repository, are described in [1].

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
