#include "detector.h"

MRS_Detector::MRS_Detector(
                            const char *data_fname,
                            const char *names_fname,
                            const char *cfg_fname,
                            const char *weights_fname,
                            float nms,
                            int thresh,
                            int n_classes)
{
  _init_OK = false;

  setlocale(LC_NUMERIC,"C");
  _data_fname = std::string(data_fname);
  _names_fname = std::string(names_fname);
  _cfg_fname = std::string(cfg_fname);
  printf("'%s'\n", weights_fname);
  _weights_fname = std::string(weights_fname);
  printf("'%s'\n", _weights_fname.c_str());
  _nms = nms;
  _thresh = thresh;
  _n_classes = n_classes;
}

MRS_Detector::~MRS_Detector()
{
  if(_boxes)
      free(_boxes);
  if(_probs)
      free_ptrs((void **)_probs, _last_l.w*_last_l.h*_last_l.n);
  opencl_deinit();
}

bool MRS_Detector::initialize()
{
  opencl_init(NULL, NULL, NULL);

  //printf("MRS_Detector: Reading data file '%s'\n", _data_fname.c_str());
  //list *options = read_data_cfg((char*)_data_fname.c_str());
  //char *names_fname = option_find_str(options, "names", (char*)_names_fname.c_str());

  /*if(!names_fname)
  {
    fprintf(stderr, "Names file not specified in data file '%s'\n", _data_fname.c_str());
    return false;
  }*/

  char **class_names_tmp = get_labels((char*)_names_fname.c_str());
  if(!class_names_tmp)
  {
    fprintf(stderr, "Invalid names file '%s'\n", _names_fname.c_str());
    return false;
  }
  _class_names.reserve(_n_classes);
  for (int it = 0; it < _n_classes; it++)
  {
    _class_names.push_back(std::string(class_names_tmp[it]));
    free(class_names_tmp[it]);
  }
  free(class_names_tmp);

  printf("MRS_Detector: Creating network from file '%s'\n", _cfg_fname.c_str());
  _net = parse_network_cfg((char*)_cfg_fname.c_str());
  /* DPRINTF("Setup: net.n = %d\n", net.n); */
  /* DPRINTF("net.layers[0].batch = %d\n", net.layers[0].batch); */

  printf("MRS_Detector: Loading weights from file '%s'\n", _weights_fname.c_str());
  load_weights(&_net, (char*)_weights_fname.c_str());
  set_batch_network(&_net, 1);
  _last_l = _net.layers[_net.n-1];
  /* DPRINTF("Setup: layers = %d, %d, %d\n", l.w, l.h, l.n); */

  // Class limiter
  if(_last_l.classes != _n_classes)
  {
    fprintf(stderr, "Last layer number of classes (%d) does not match labels file (%d)\n", _last_l.classes, _n_classes);
    return false;
  }

  /* int expectedHeight = _net.h; */
  /* int expectedWidth = _net.w; */
  /* DPRINTF("Image expected w,h = [%d][%d]!\n", net.w, net.h); */

  _boxes = (box*)calloc(_last_l.w*_last_l.h*_last_l.n, sizeof(box));
  _probs = (float**)calloc(_last_l.w*_last_l.h*_last_l.n, sizeof(float *));

  // initialize probabilities
  for(int it = 0; it < _last_l.w*_last_l.h*_last_l.n; ++it)
  {
    _probs[it] = (float*)calloc(_last_l.classes + 1, sizeof(float));
  }


  printf("Initialization of MRS detector successful\n");
  _init_OK = true;
  return true;
}





std::vector<uav_detect::Detection> MRS_Detector::detect(
            const cv::Mat &image,
            float thresh,
            float hier_thresh)
{
  std::vector<uav_detect::Detection> ret;

  // Input sanity checks
  if(!_init_OK)
  {
    fprintf(stderr, "Initialization of MRS detector is not done!\n");
    return ret;
  }

  if(image.empty())
  {
    fprintf(stderr, "Input image is empty\n");
    return ret;
  }

  /** Convert the input image to proper format for darknet **/
  // Convert the bytes to float
  cv::Mat image_f;
  image.convertTo(image_f, CV_32FC3, 1/255.0);

  if (image_f.rows != _net.h || image_f.cols != _net.w)
  {
    resize(image_f, image_f, cv::Size(_net.w, _net.h));
  }
  // Get the image to suit darknet
  std::vector<cv::Mat> image_channels(3);
  split(image_f, image_channels);
  vconcat(image_channels, image_f);

  /** Detection itself **/
  int n_detections = 0;
  // Predict
  network_predict(_net, (float*)image_f.data);
  get_region_boxes(_last_l, 1, 1, thresh, _probs, _boxes, 0, 0, hier_thresh);

  /* DPRINTF("l.softmax_tree = %p, nms = %f\n", l.softmax_tree, nms); */
  if (_last_l.softmax_tree && _nms)
  {
    do_nms_obj(_boxes, _probs, _last_l.w*_last_l.h*_last_l.n, _last_l.classes, _nms);
  }
  else if (_nms)
  {
    do_nms_sort(_boxes, _probs, _last_l.w*_last_l.h*_last_l.n, _last_l.classes, _nms);
  }

  // Count the number of detections so that enough space can be reserved in the return vector
  for (int it = 0; it < (_last_l.w*_last_l.h*_last_l.n); it++)
  {
    int class_ID = max_index(_probs[it], _last_l.classes);
    float prob = _probs[it][class_ID];
    if (prob > thresh)
    {
      n_detections++;
    }
  }

  /** Get the detections **/
  int count = 0;
  ret.reserve(n_detections);

  /* if(!_boxes || !probs || !outLabels || !outBoxes) */
  /* { */
  /*     EPRINTF("Error NULL boxes/probs, %p, %p !\n", boxes, probs); */
  /*     return false; */
  /* } */
  for(int it = 0; it < (_last_l.w*_last_l.h*_last_l.n); it++)
  {
      int class_ID = max_index(_probs[it], _last_l.classes);
      float prob = _probs[it][class_ID];
      if(prob > _thresh)
      {
        uav_detect::Detection det;
        det.confidence = prob;
        det.class_ID = class_ID;
        det.x = _boxes[it].x;
        det.y = _boxes[it].y;
        det.width = _boxes[it].w;
        det.height = _boxes[it].h;
        ret.push_back(det);
        count++;
      }
  }
  return ret;
}

std::string MRS_Detector::get_class_name(int class_ID)
{
  return _class_names.at(class_ID);
}
