all: ros_uav_detect.o mrs_detect.o
	g++ -DGPU -DOPENCV -DOPENCL -Wall -o ros_uav_detect objs/ros_uav_detect.o objs/mrs_detect.o -L../ -I../src -ldarknet-cpp-shared -std=c++11 -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_viz -lopencv_core -lopencv_videoio -lopencv_highgui

ros_uav_detect.o: ros_uav_detect.cpp objs
	g++ -DGPU -DOPENCV -DOPENCL -Wall -L../ -I../src -c -o objs/ros_uav_detect.o ros_uav_detect.cpp -ldarknet-cpp-shared -std=c++11 -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_viz -lopencv_core -lopencv_videoio -lopencv_highgui

mrs_detect.o: mrs_detect.cpp objs
	g++ -DGPU -DOPENCV -DOPENCL -Wall -L../ -I../src -c -o objs/mrs_detect.o mrs_detect.cpp -ldarknet-cpp-shared -std=c++11 -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_viz -lopencv_core -lopencv_videoio -lopencv_highgui

objs:
	mkdir objs
