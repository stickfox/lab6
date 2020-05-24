#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include "../include/ObjectTracking.h"

using namespace cv;
using namespace std;

/* Constructor */
ObjectTracking::ObjectTracking() {
}

std::vector<cv::Point2f> ObjectTracking::getTrackingPoints(cv::Mat frame, cv::Mat previous_frame, std::vector<cv::Point2f> matched_points) {
	std::vector<cv::Point2f> matching_points;
	std::vector<uchar> status;
	std::vector<float> error;
	cv::calcOpticalFlowPyrLK(frame, previous_frame, matched_points, matching_points, status, error);
	return matching_points;
}