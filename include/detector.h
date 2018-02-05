#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <locale.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "region_layer.h"
#include "option_list.h"
// this actually include OpenCL related function definitions
#include "cuda.h"


struct Detection
{
  float probability;
  box bounding_box;
  int class_ID;
};

class MRS_Detector
{
public:
  MRS_Detector(
          const char *data_fname,
          const char *cfg_fname,
          const char *weights_fname,
          float nms,
          int thresh,
          int n_classes);
  ~MRS_Detector();

  bool initialize();

  std::vector<Detection> detect(const cv::Mat &image,
                                 float thresh,
                                 float hier_thresh);
  private:
    // Configuration variables
    bool _init_OK;  

    std::string _data_fname;
    std::string _cfg_fname;
    std::string _weights_fname;

    float _nms;
    int _thresh;
    int _n_classes;
    char **_class_names;

    // Detection related variables
    box *_boxes;
    float **_probs;
    /* bool bSetup; */
    network _net;
    layer _last_l;
};


#endif // DETECTOR_H
