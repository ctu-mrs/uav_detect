int32 class_ID                    # class of the detected object (-1 means unused)
sensor_msgs/RegionOfInterest roi  # ROI of the image which was used for this detection
float32 x                         # x coordinate of center of the detection relative to ROI width
float32 y                         # y coordinate of center of the detection relative to ROI height
float32 depth                     # depth of the detected object in meters (-1 means unused)
float32 w                         # width of bounding box of the detection relative to ROI width (-1 means unused)
float32 h                         # height of bounding box of the detection relative to ROI height (-1 means unused)
float32 confidence                # confidence of the detection (-1 means unused)
