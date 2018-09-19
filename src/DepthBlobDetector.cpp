#include "DepthBlobDetector.h"

using namespace cv;
using namespace std;
using namespace dbd;

DepthBlobDetector::DepthBlobDetector(const Params& parameters)
{
  params = parameters;
}

/* method void DepthBlobDetector::findBlobs(cv::Mat image, cv::Mat binaryImage, std::vector<Center>& centers) const //{ */
/* inspired by https://github.com/opencv/opencv/blob/3.4/modules/features2d/src/blobdetector.cpp */
void DepthBlobDetector::findBlobs(cv::Mat image, cv::Mat binaryImage, std::vector<Center>& centers) const
{
    centers.clear();

    std::vector < std::vector<Point> > contours;
    Mat tmpBinaryImage = binaryImage.clone();
    findContours(tmpBinaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR
    Mat keypointsImage;
    cvtColor( binaryImage, keypointsImage, CV_GRAY2RGB );
    
    Mat contoursImage;
    cvtColor( binaryImage, contoursImage, CV_GRAY2RGB );
    drawContours( contoursImage, contours, -1, Scalar(0,255,0) );
    imshow("contours", contoursImage );
#endif

    for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
    {
        Center center;
        center.confidence = 1;
        Moments moms = moments(Mat(contours[contourIdx]));
        if (params.filter_by_area)
        {
            double area = moms.m00;
            if (area < params.min_area || area >= params.max_area)
                continue;
        }

        if (params.filter_by_circularity)
        {
            double area = moms.m00;
            double perimeter = arcLength(Mat(contours[contourIdx]), true);
            double ratio = 4 * CV_PI * area / (perimeter * perimeter);
            if (ratio < params.min_circularity || ratio >= params.max_circularity)
                continue;
        }

        if (params.filter_by_inertia)
        {
            double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
            const double eps = 1e-2;
            double ratio;
            if (denominator > eps)
            {
                double cosmin = (moms.mu20 - moms.mu02) / denominator;
                double sinmin = 2 * moms.mu11 / denominator;
                double cosmax = -cosmin;
                double sinmax = -sinmin;

                double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
                double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
                ratio = imin / imax;
            }
            else
            {
                ratio = 1;
            }

            if (ratio < params.min_inertia_ratio || ratio >= params.max_inertia_ratio)
                continue;

            center.confidence = ratio * ratio;
        }

        if (params.filter_by_convexity)
        {
            std::vector < Point > hull;
            convexHull(Mat(contours[contourIdx]), hull);
            double area = contourArea(Mat(contours[contourIdx]));
            double hullArea = contourArea(Mat(hull));
            if (fabs(hullArea) < DBL_EPSILON)
                continue;
            double ratio = area / hullArea;
            if (ratio < params.min_convexity || ratio >= params.max_convexity)
                continue;
        }

        if(moms.m00 == 0.0)
            continue;
        center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

        if (params.filter_by_color)
        {
            if (binaryImage.at<uchar> (cvRound(center.location.y), cvRound(center.location.x)) != params.color)
                continue;
        }

        //compute blob radius
        {
            std::vector<double> dists;
            for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
            {
                Point2d pt = contours[contourIdx][pointIdx];
                dists.push_back(norm(center.location - pt));
            }
            std::sort(dists.begin(), dists.end());
            center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
        }

        centers.push_back(center);


#ifdef DEBUG_BLOB_DETECTOR
        circle( keypointsImage, center.location, 1, Scalar(0,0,255), 1 );
#endif
    }
#ifdef DEBUG_BLOB_DETECTOR
    imshow("bk", keypointsImage );
    waitKey(1);
#endif
}
//}

/* method void DepthBlobDetector::detect(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat mask) //{ */
/* inspired by https://github.com/opencv/opencv/blob/3.4/modules/features2d/src/blobdetector.cpp */
void DepthBlobDetector::detect(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat mask)
{
    keypoints.clear();
    assert(params.min_repeatability != 0);
    Mat grayscaleImage;
    if (image.channels() == 3 || image.channels() == 4)
        cvtColor(image, grayscaleImage, COLOR_BGR2GRAY);
    else
        grayscaleImage = image;

    if (grayscaleImage.type() != CV_8UC1) {
        ROS_ERROR("Blob detector only supports 8-bit images!");
    }

    std::vector < std::vector<Center> > centers;
    for (double thresh = params.min_threshold; thresh < params.max_threshold; thresh += params.threshold_step)
    {
        Mat binarizedImage;
        threshold(grayscaleImage, binarizedImage, thresh, 255, THRESH_BINARY);

        std::vector < Center > curCenters;
        findBlobs(grayscaleImage, binarizedImage, curCenters);
        std::vector < std::vector<Center> > newCenters;
        for (size_t i = 0; i < curCenters.size(); i++)
        {
            bool isNew = true;
            for (size_t j = 0; j < centers.size(); j++)
            {
                double dist = norm(centers[j][ centers[j].size() / 2 ].location - curCenters[i].location);
                isNew = dist >= params.min_dist_between && dist >= centers[j][ centers[j].size() / 2 ].radius && dist >= curCenters[i].radius;
                if (!isNew)
                {
                    centers[j].push_back(curCenters[i]);

                    size_t k = centers[j].size() - 1;
                    while( k > 0 && centers[j][k].radius < centers[j][k-1].radius )
                    {
                        centers[j][k] = centers[j][k-1];
                        k--;
                    }
                    centers[j][k] = curCenters[i];

                    break;
                }
            }
            if (isNew)
                newCenters.push_back(std::vector<Center> (1, curCenters[i]));
        }
        std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));
    }

    for (size_t i = 0; i < centers.size(); i++)
    {
        if (centers[i].size() < params.min_repeatability)
            continue;
        Point2d sumPoint(0, 0);
        double normalizer = 0;
        for (size_t j = 0; j < centers[i].size(); j++)
        {
            sumPoint += centers[i][j].confidence * centers[i][j].location;
            normalizer += centers[i][j].confidence;
        }
        sumPoint *= (1. / normalizer);
        KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius) * 2.0f);
        keypoints.push_back(kpt);
    }

    if (!mask.empty())
    {
        KeyPointsFilter::runByPixelsMask(keypoints, mask);
    }
}
//}
