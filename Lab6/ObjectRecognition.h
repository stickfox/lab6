#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class ObjectRecognition
{
public:

	const float RATIO = 3.0f; // RATIO for matches refinement
	const int MAX_FEATURES = 500; // maximum of features found by ORB
	cv::VideoCapture video;
	cv::Mat object;
	cv::Ptr<cv::Feature2D> orb;
	cv::BFMatcher matcher;
	std::vector<cv::Mat> projections;
	std::vector<cv::Mat> panoramics;

	/* Constructor */
	ObjectRecognition(cv::VideoCapture vid_cap, cv::Mat obj);

	/* Find final panorama merging together the images */
	void getMatching();

private:

	/* Compute features of an image using ORB */
	void computeFeatures(cv::Mat projection, std::vector<cv::KeyPoint>* keypoint, cv::Mat* descriptor);

	/* Compute the initial cutting points in two consecutive images */
	void getTranslation(cv::Mat video_image, cv::Mat object_image);

};
