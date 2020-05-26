#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class ObjectTracker
{
public:

	const float RATIO = 3.0f; // RATIO for matches refinement
	const int MAX_FEATURES = 5000; // maximum of features found by ORB

	/* Constructor */
	ObjectTracker(cv::VideoCapture vid_cap, cv::Mat obj, cv::Scalar color);

	/* Compute the initial cutting points in two consecutive images */
	std::vector<cv::Point2f> getMatchingPoints(cv::Mat video_image, cv::Mat object_image, int max_features1, int max_features2);

	/* Track the object points throw the frames */
	std::vector<cv::Point2f> getTrackingPoints(cv::Mat frame, cv::Mat previous_frame, std::vector<cv::Point2f> matched_points);

private:

	/* Compute features of an image using ORB */
	void ObjectTracker::computeFeatures(cv::Mat projection, std::vector<cv::KeyPoint>* keypoint, cv::Mat* descriptor, int max_features);

	/* Draw a rectangul on the regnized object */
	void drawRectangle(cv::Mat* image, std::vector<cv::Point2f> points, std::vector<cv::Point2f> object_points, double norm);

	/////////////////////////////////////////////////////
	// Attributes
	/////////////////////////////////////////////////////

	cv::VideoCapture video;
	cv::Mat object;
	cv::BFMatcher matcher;
	std::vector<cv::Point2f> object_inliers;
	std::vector<cv::Point2f> prev_scene_corners;
	cv::Scalar line_color;
};
