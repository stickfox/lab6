#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <iostream>
#include "../include/ObjectTracker.h"

using namespace cv;
using namespace std;

/* Constructor */
ObjectTracker::ObjectTracker(Mat obj, Scalar color) {
	object = obj;

	//matcher = BFMatcher(NORM_HAMMING, true);	
	matcher = BFMatcher(NORM_L1, false);

	prev_scene_corners = vector<Point2f>(4);
	line_color = color;
}


/* Compute features of an image using ORB */
void ObjectTracker::computeFeaturesOrb(Mat projection, vector<KeyPoint>* keypoint, Mat* descriptor, int max_features) {
	Ptr<Feature2D> orb = ORB::create(max_features);
	orb->detectAndCompute(projection, Mat(), *keypoint, *descriptor);
}

void ObjectTracker::computeFeaturesSift(Mat projection, vector<KeyPoint>* keypoint, Mat* descriptor) {
	Ptr<xfeatures2d::SIFT> f2d = xfeatures2d::SIFT::create();
	f2d->detect(projection, *keypoint);
	f2d->compute(projection, *keypoint, *descriptor);
}

/* Compute the initial cutting points in two consecutive images */
vector<Point2f> ObjectTracker::getMatchingPoints(Mat clean_image, Mat* output_image, Mat object_image, int use_orb, int max_features1, int max_features2) {
	vector<KeyPoint> keypoint1, keypoint2;
	Mat descriptor1, descriptor2;
	vector<DMatch> matches;

	// Extract keypoints and their descriptors, then match the keypoints
	if (!use_orb)
	{
		computeFeaturesSift(object_image, &keypoint1, &descriptor1);
		computeFeaturesSift(clean_image, &keypoint2, &descriptor2);
	}
	else
	{
		computeFeaturesOrb(object_image, &keypoint1, &descriptor1, max_features1);
		computeFeaturesOrb(clean_image, &keypoint2, &descriptor2, max_features2);
	}

	matcher.match(descriptor1, descriptor2, matches);

	// Refine the matches using distance
	vector<DMatch> good_matches;
	double min_dist = DBL_MAX;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance < min_dist)
			min_dist = matches[i].distance;
	}
	double dist_thresold = RATIO * min_dist;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance <= max(dist_thresold, 0.02))
			good_matches.push_back(matches[i]);
	}

	// Extract location of good matches
	vector<Point2f> points1, points2;
	for (size_t i = 0; i < good_matches.size(); i++) {
		points1.push_back(keypoint1[good_matches[i].queryIdx].pt);
		points2.push_back(keypoint2[good_matches[i].trainIdx].pt);
	}

	// Find the mask that highlights the inliers
	Mat h;
	Mat inlier_mask;
	float ransacThreshold = 3.0;
	h = findHomography(points1, points2, RANSAC, ransacThreshold, inlier_mask);
	cout << h << endl;

	// Get the corners from the image ( the object to be "detected" )
	vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f(object_image.cols - 1, 1);
	obj_corners[2] = Point2f(object_image.cols - 1, object_image.rows - 1);
	obj_corners[3] = Point2f(1, object_image.rows - 1);

	perspectiveTransform(obj_corners, prev_scene_corners, h);

	drawRectangle(output_image);

	// Retrive only the inliers using the mask
	vector<Point2f> inliers1, inliers2;
	vector<DMatch> inlier_matches;
	for (unsigned i = 0; i < points1.size(); i++) {
		if (inlier_mask.at<uchar>(i)) {
			inliers1.push_back(points1[i]);
			inliers2.push_back(points2[i]);
			inlier_matches.push_back(good_matches[i]);
		}
	}

	object_inliers = inliers1;

	// Draw the found features
	prev_scene_features_points = vector<Point2f>();
	for (int i = 0; i < inliers2.size(); i++)
		prev_scene_features_points.push_back(inliers2[i]);
	drawExtractedFeatures(output_image);

	/* Draw matches */
	//Mat matching;
	//drawMatches(object_image, keypoint1, *output_image, keypoint2, inlier_matches, matching);
	//namedWindow("MATCH");
	//imshow("MATCH", matching);
	//waitKey(0);
	//destroyAllWindows();

	return inliers2;
}

void ObjectTracker::drawExtractedFeatures(Mat* image) {
	// Draw a circle for each found point.
	for (int i = 0; i < prev_scene_features_points.size(); i++)
		circle(*image, prev_scene_features_points[i], 2, line_color, 2);
}

void ObjectTracker::drawRectangle(Mat* image) {
	// Draw lines between the corners (the mapped object in the scene )
	line(*image, prev_scene_corners[0], prev_scene_corners[1], line_color, 2);
	line(*image, prev_scene_corners[1], prev_scene_corners[2], line_color, 2);
	line(*image, prev_scene_corners[2], prev_scene_corners[3], line_color, 2);
	line(*image, prev_scene_corners[3], prev_scene_corners[0], line_color, 2);
}

/* Draw a rectangle on the recognized object */
void ObjectTracker::computeHomography(Mat* image, vector<Point2f> current_points, double norm, vector<uchar> status) {

	// Find the mask that highlights the inliers
	Mat h;
	Mat inlier_mask;
	float ransacThreshold = 3.0;

	// compute the homography only if the points positions significatively change
	if (norm > 3)
	{
		h = findHomography(object_inliers, current_points, RANSAC, ransacThreshold, inlier_mask);
		//cout << h << endl;

		// Get the corners from the image ( the object to be "detected" )
		vector<Point2f> obj_corners(4);
		obj_corners[0] = Point2f(10, 10);
		obj_corners[1] = Point2f(object.cols - 10, 10);
		obj_corners[2] = Point2f(object.cols - 10, object.rows - 10);
		obj_corners[3] = Point2f(10, object.rows - 10);

		perspectiveTransform(obj_corners, prev_scene_corners, h);

		// convert features points
		//perspectiveTransform(current_points, prev_scene_features_points, h);
		prev_scene_features_points = vector<Point2f>();
		for (int i = 0; i < current_points.size(); i++)
		{
			if (status[i])
				prev_scene_features_points.push_back(current_points[i]);
		}

		//prev_scene_features_points = current_points;
	}

	drawRectangle(image);
	drawExtractedFeatures(image);
}

/* Track points on the actual frame starting from the ones on the previous frame */
vector<Point2f> ObjectTracker::getTrackingPoints(Mat clean_frame, Mat* output_frame, Mat previous_frame, vector<Point2f> matched_points) {
	vector<Point2f> matching_points;
	vector<uchar> status;
	vector<float> error;

	Size search_window_size(7, 7);
	int maximal_pyramid_level = 3;
	TermCriteria stop_criteria(TermCriteria::Type::COUNT | TermCriteria::Type::EPS, 10, 1);
	calcOpticalFlowPyrLK(previous_frame, clean_frame, matched_points, matching_points, status, error, search_window_size, maximal_pyramid_level/*, stop_criteria*/);

	vector<Point2f> diff;
	// old points - new points
	absdiff(matched_points, matching_points, diff);
	double norm = cv::norm(diff);

	//cout << "norm = " << norm << endl;
	computeHomography(output_frame, matching_points, norm, status);

	return matching_points;
}