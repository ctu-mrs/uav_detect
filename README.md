# MAV detection and localization from depthmap

Intended to be used with Intel Realsense D435 depth camera sensor.

### The following launchfiles are provided:
 * *depth_detect.launch*: Starts the detection of blobs from a depth image.
 * *display_detections.launch*: Starts visualization of the detections to OpenCV windows (raw depthmap, processed depthmap + detected blobs and raw RGB image are displayed).
 * *localize_single.launch*: Starts 3D localization and filtering of a single UAV position from the detected blobs using Kalman Filters.
 * *backproject_location.launch*: Starts visualization of the detected UAV 3D location. The location is backprojected to the RGB image and displayed in an OpenCV window.
 * Variants of the above launchfiles, starting with *sim_* prependix are provided for use with simulation.
 * *simulation.launch*: Starts the simulation world with some trees and a grass pane. Also starts static transform broadcasters for the Realsense camera coordinate frames!

### The following config files are used by the nodes:
 * *detection_params_F550.yaml:* Contains parameters for the blob detection, tuned to detect the F550 from real-world data. Parameters are documented in the file itself.
 * *localization_params_F550.yaml:* Contains parameters for the single UAV localization, tuned to detect the F550 from real-world data. Parameters are documented in the file itself.
 * Variants of the above files with the *sim_* prependix are provided for use with simulation.
Most parameters (those which make sense) from the above files are dynamically reconfigurable.

### To launch simulation, detection, localization and visualization:
 1) In folder *tmux_scripts/simulation* launch the tmuxinator session to start simulation with two drones: `uav1` and `uav2`. `uav1` is the interceptor with the Realsense sensor and `uav2` is an intruder with a random flier controlling it.
 2) In folder *tmux_scripts/detection_and_visualization* launch the tmuxinator session to start detection, localization and visualization nodes. It is recommended to use install Tomáš's layout manager before-hand (just update to latest linux-setup) to enable automatic application of a layout (otherwise the windows will be quite cluttered).
 3) You can adjust the detection and localization parameters according to Rviz or according to the OpenCV visualization using the *rqt_reconfigure* (which is automatically launched in the *detection_and_visualization* session).
