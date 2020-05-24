#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>
#include <iostream>


using namespace cv;
using namespace std;

class ObjectTracking
{
public:

	const float RATIO = 5.0f; // RATIO for matches refinement
	const int MAX_FEATURES = 1500; // maximum of features found by ORB
	double angle;  // half FoV of the camera
	cv::Ptr<cv::Feature2D> orb;
	cv::BFMatcher matcher;
	std::vector<cv::String> files;
	std::vector<cv::Mat> images;
	std::vector<cv::Mat> projections;
	std::vector<cv::Mat> panoramics;

	/* Constructor */
	ObjectTracking();

	/* Track the object points throw the frames */
	std::vector<cv::Point2f> getTrackingPoints(cv::Mat frame, cv::Mat previous_frame, std::vector<cv::Point2f> matched_points);

private:

	/* Loads images from a path */
	

};
