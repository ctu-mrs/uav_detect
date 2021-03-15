# MAV detection using a convolutional neural network

Implementation of a drone detection method using a CNN, used in the papers [1] and [2].

*Note:* The code in this branch is old, messy and not well documented.
Since the time it was written, many new and better CNNs for object detection were developed than the deprecated version of tiny-YOLO [3] used in this work.
Consider this repository just an example implementation - for real deployment, you'll be better off using some of the more recent CNN frameworks and architectures, such as TensorFlow and the CenterNet [4].

## Requirements:
 - ROS Melodic
 - [darknet](https://pjreddie.com/darknet/) (the code uses an unofficial OpenCL implementation of darknet, but modification to use the official one should be simple)

## Description of the provided interface and other info

### The following launchfiles are provided:
 * **uav_detect.launch**: Starts the detection of drones from RGB camera image.
 * **display_detections.launch**: Starts visualization of the detections to a ROS topic.
 * **detection_simulation.launch**: Starts the simulation world with some trees and a grass pane. Also starts static transform broadcasters for the Realsense camera coordinate frames! *Note:* Don't use this launchfile to start the simulation manually. Instead, use the prepared tmux scripts in the `uav_localize` package.

### The following config files are used by the nodes:
 * **uav_detect.yaml:** Contains parameters for the drone detection, tuned to detect drones from real-world data using the Realsense D435 sensor.
Most parameters (those which make sense) from the above files are dynamically reconfigurable.

----
References:

 * [1]: M. Vrba, D. Heřt and M. Saska, "Onboard Marker-Less Detection and Localization of Non-Cooperating Drones for Their Safe Interception by an Autonomous Aerial System," in IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3402-3409, Oct. 2019, doi: 10.1109/LRA.2019.2927130.
 * [2]: M. Vrba and M. Saska, "Marker-Less Micro Aerial Vehicle Detection and Localization Using Convolutional Neural Networks," in IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 2459-2466, April 2020, doi: 10.1109/LRA.2020.2972819.
 * [3]: J. Redmon and A. Fahradi, "YOLO9000: Better, Faster, Stronger", eprint arXiv: 1612.08242, December 2016 (https://arxiv.org/abs/1612.08242).
 * [4]: K. Duan, S. Bai, L. Xie, H. Qi, Q. Huang and Q. Tian, "CenterNet: Keypoint Triplets for Object Detection", eprint arXiv: 1904.08189, April 2019 (https://arxiv.org/abs/1904.08189).
