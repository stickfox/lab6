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

	const float RATIO = 3.0f;	   // RATIO for matches refinement
	const int MAX_FEATURES = 5000; // maximum of features found by ORB

	/* Constructor */
	ObjectTracker(Mat obj, Scalar color);

	/* Compute the initial cutting points in two consecutive images */
	vector<Point2f> ObjectTracker::getMatchingPoints(Mat object_image, Mat clean_image, Mat* output_image, int use_orb, int max_features1, int max_features2);

	/* Track the object points throw the frames */
	vector<Point2f> ObjectTracker::getTrackingPoints(Mat clean_frame, Mat* output_frame, Mat previous_frame, vector<Point2f> matched_points);

private:

	/* Compute features of an image using ORB */
	void ObjectTracker::computeFeaturesOrb(Mat projection, vector<KeyPoint>* keypoint, Mat* descriptor, int max_features);
	void ObjectTracker::computeFeaturesSift(Mat projection, vector<KeyPoint>* keypoint, Mat* descriptor);

	/* Draw a rectangul on the regnized object */
	void ObjectTracker::computeHomography(Mat* image, vector<Point2f> current_points, double norm, vector<uchar> status);
	void ObjectTracker::drawRectangle(Mat* image);
	void ObjectTracker::drawExtractedFeatures(Mat* image);

	/////////////////////////////////////////////////////
	// Attributes
	/////////////////////////////////////////////////////

	Mat object;
	BFMatcher matcher;
	vector<Point2f> object_inliers;
	vector<Point2f> prev_scene_corners;
	vector<Point2f> prev_scene_features_points;
	Scalar line_color;
};
