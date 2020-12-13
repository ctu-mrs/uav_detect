# MAV detection and localization from depthmap

Implementation of the marker-less drone detection algorithm, presented in [1].

Rudimentary knowledge of the ROS system is assumed.
Tested with the Intel Realsense D435 depth camera sensor and the Nerian SceneScan Pro + Karmin2 sensor.

## The working principle

### Detection of blobs in a depthmap image
The algorithm is pretty self-explanatory. For each input depthmap image, the following steps are executed:
 1) The depthmap image is preprocessed based on the selected parameters (`dilate_iterations`, `erode_iterations`, `gaussianblur_size` etc.).
 2) The preprocessed image is thresholded by a certain distance to convert it to binary image (ie. all pixels further than x meters are black, rest is white).
 3) Blobs are found in the binarized image using a contour detecton algorithm.
 4) The found blobs are filtered based on the selected parameters (`filter_by_area`, `filter_by_circularity`, `filter_by_orientation` etc.).
 5) Remaining blobs are added to a set of selected blobs.
 6) Steps 2) to 5) are repeated for the selected distance range with the specified distance step (eg. from 1 meter to 13 meters with a step of 0.5 meter).
 7) When the whole distance range has been scanned for blobs, blobs in the resulting set of selected blobs which are closer than `min_dist_between` are grouped together to form a single output blob, which is added to a set of output blobs.
 8) If the number of blobs, which were grouped together to form an output blob, is less than `min_repeatability`, the output blob is kicked out of the set of output blobs.
 9) The set of output blobs is published as the detections.

For a more thorough description and evaluation of the algorithm implemented in this repository, see the paper [1].

## Description of the provided interface and other info

### The following launchfiles are provided:
 * **depth_detect.launch**: Starts the detection of blobs from a depth image.
 * **display_detections.launch**: Starts visualization of the detections to OpenCV windows (raw depthmap, processed depthmap + detected blobs and raw RGB image are displayed).
 * **simulation.launch**: Starts the simulation world with some trees and a grass pane. Also starts static transform broadcasters for the Realsense camera coordinate frames! *Note:* Don't use this launchfile to start the simulation manually. Instead, use the prepared tmux scripts in the `uav_localize` package.

### The following config files are used by the nodes:
 * **detection_params_realsense.yaml:** Contains parameters for the blob detection, tuned to detect drones from real-world data using the Realsense D435 sensor. Parameters are documented in the file itself.
 * **detection_params_nerian.yaml:** Same as above, but for the Nerian SceneScan Pro + Karmin2 sensor.
Most parameters (those which make sense) from the above files are dynamically reconfigurable.

----
References:

 * [1]: M. Vrba, D. Heřt and M. Saska, "Onboard Marker-Less Detection and Localization of Non-Cooperating Drones for Their Safe Interception by an Autonomous Aerial System," in IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3402-3409, Oct. 2019, doi: 10.1109/LRA.2019.2927130.
