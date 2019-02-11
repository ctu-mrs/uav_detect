#ifndef UAV_DETECT_H
#define UAV_DETECT_H

#include "detector.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cnn_detect/Detection.h>
#include <cnn_detect/Detections.h>

#include <mrs_lib/ParamLoader.h>

#include <iomanip>


#endif // UAV_DETECT_H
