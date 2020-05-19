#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "ObjectTracking.h"

using namespace cv;
using namespace std;

/* Constructor */
ObjectTracking::ObjectTracking(cv::String path, double half_FoV) {
	angle = half_FoV;
	files = std::vector<cv::String>();
	cv::glob(path, files);
	if (files.size() == 0) {
		cout << "Error: unable to load files from path " << path << endl;
	}
	matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
	orb = cv::ORB::create(MAX_FEATURES);
}
