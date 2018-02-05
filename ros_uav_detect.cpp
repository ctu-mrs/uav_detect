#include "ros_uav_detect.h"

using namespace cv;
using namespace std;

int main(int argc, const char **argv)
{
  printf("Creating detector object\n");
  MRS_Detector detector("./mrs/mrs.data", "./mrs/tiny-yolo-mrs.cfg", "./mrs/tiny-yolo-mrs_final.weights", 0.2, 0.1, 1);
  printf("Initializing detector object\n");
  detector.initialize();

  VideoCapture cap(0); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
    return -1;

  for(;;)
  {
    Mat camera_frame;
    cap >> camera_frame; // get a new frame from camera
    auto detections = detector.detect(
            camera_frame,
            0.1,
            0.1);
    for (auto det : detections)
    {
      cout << "Object detected:" << std::endl;
      Point pt1((det.bounding_box.x - det.bounding_box.w/2.0)*camera_frame.cols,
                (det.bounding_box.y - det.bounding_box.h/2.0)*camera_frame.rows);
      Point pt2((det.bounding_box.x + det.bounding_box.w/2.0)*camera_frame.cols,
                (det.bounding_box.y + det.bounding_box.h/2.0)*camera_frame.rows);
      rectangle(camera_frame, pt1, pt2, Scalar(0, 0, 255));
    }
    cout << "End of frame." << std::endl;
    imshow("edges", camera_frame);
    if(waitKey(30) >= 0) break;
  }
}
