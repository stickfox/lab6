#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "../include/ObjectTracker.h"

using namespace cv;
using namespace std;

/* Constructor */
ObjectTracker::ObjectTracker(cv::VideoCapture vid_cap, cv::Mat obj) {
	video = vid_cap;
	object = obj;
	matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
	orb = cv::ORB::create(MAX_FEATURES);
}


/* Compute features of an image using ORB */
void ObjectTracker::computeFeatures(cv::Mat projection, std::vector<cv::KeyPoint>* keypoint, cv::Mat* descriptor) {
	orb->detectAndCompute(projection, cv::Mat(), *keypoint, *descriptor);
}


/* Compute the initial cutting points in two consecutive images */
std::vector<cv::Point2f> ObjectTracker::getMatchingPoints(cv::Mat video_image, cv::Mat object_image) {
	std::vector<cv::KeyPoint> keypoint1, keypoint2;
	cv::Mat descriptor1, descriptor2;
	std::vector<cv::DMatch> matches;

	/* Extract keypoints and their descriptors, then match the keypoints */
	computeFeatures(object_image, &keypoint1, &descriptor1);
	computeFeatures(video_image, &keypoint2, &descriptor2);
	matcher.match(descriptor1, descriptor2, matches);

	/* Refine the matches using distance */
	std::vector<cv::DMatch> good_matches;
	double min_dist = 500;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance < min_dist)
			min_dist = matches[i].distance;
	}
	double dist_thresold = RATIO * min_dist;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance <= max(dist_thresold, 0.02))
			good_matches.push_back(matches[i]);
	}

	/* Extract location of good matches */
	std::vector<Point2f> points1, points2;
	for (size_t i = 0; i < good_matches.size(); i++) {
		points1.push_back(keypoint1[good_matches[i].queryIdx].pt);
		points2.push_back(keypoint2[good_matches[i].trainIdx].pt);
	}

	/* Find the mask that highlights the inliers */
	cv::Mat h;
	cv::Mat inlier_mask;
	float ransacThreshold = 3.0;
	h = cv::findHomography(points1, points2, RANSAC, ransacThreshold, inlier_mask);
	cout << h << endl;

	// Get the corners from the image ( the object to be "detected" )
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cv::Point2f(0, 0);
	obj_corners[1] = cv::Point2f(object_image.cols - 1, 1);
	obj_corners[2] = cv::Point2f(object_image.cols - 1, object_image.rows - 1);
	obj_corners[3] = cv::Point2f(1, object_image.rows - 1);
	std::vector<cv::Point2f> scene_corners(4);
	scene_corners[0] = cv::Point2f(1, 1);

	perspectiveTransform(obj_corners, scene_corners, h);
	//Draw lines between the corners (the mapped object in the scene )
	cv::line(video_image, scene_corners[0], scene_corners[1], cv::Scalar(0, 0, 255), 2);
	cv::line(video_image, scene_corners[1], scene_corners[2], cv::Scalar(0, 0, 255), 2);
	cv::line(video_image, scene_corners[2], scene_corners[3], cv::Scalar(0, 0, 255), 2);
	cv::line(video_image, scene_corners[3], scene_corners[0], cv::Scalar(0, 0, 255), 2);

	/* Retrive only the inliers using the mask */
	std::vector<cv::Point2f> inliers1, inliers2;
	std::vector<cv::DMatch> inlier_matches;
	for (unsigned i = 0; i < points1.size(); i++) {
		if (inlier_mask.at<uchar>(i)) {
			inliers1.push_back(points1[i]);
			inliers2.push_back(points2[i]);
			inlier_matches.push_back(good_matches[i]);
		}
	}

	/* Draw matches */
	cv::Mat matching;
	cv::drawMatches(object_image, keypoint1, video_image, keypoint2, inlier_matches, matching);
	cv::namedWindow("MATCH");
	cv::imshow("MATCH", matching);
	cv::waitKey(0);
	cv::destroyAllWindows();

	object_inliers = inliers1;

	return inliers2;
}


/* Draw a rectangul on the regnized object */
void ObjectTracker::drawRectangle(cv::Mat* image, std::vector<cv::Point2f> previous_points, std::vector<cv::Point2f> current_points) {

	/* Find the mask that highlights the inliers */
	cv::Mat h;
	cv::Mat inlier_mask;
	float ransacThreshold = 3.0;
	h = cv::findHomography(object_inliers, current_points, RANSAC, ransacThreshold, inlier_mask);
	//cout << h << endl;

	// Get the corners from the image ( the object to be "detected" )
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cv::Point2f(0, 0);
	obj_corners[1] = cv::Point2f(object.cols - 1, 1);
	obj_corners[2] = cv::Point2f(object.cols - 1, object.rows - 1);
	obj_corners[3] = cv::Point2f(1, object.rows - 1);
	std::vector<cv::Point2f> scene_corners(4);
	scene_corners[0] = cv::Point2f(1, 1);

	perspectiveTransform(obj_corners, scene_corners, h);
	//Draw lines between the corners (the mapped object in the scene )
	cv::line(*image, scene_corners[0], scene_corners[1], cv::Scalar(0, 0, 255), 2);
	cv::line(*image, scene_corners[1], scene_corners[2], cv::Scalar(0, 0, 255), 2);
	cv::line(*image, scene_corners[2], scene_corners[3], cv::Scalar(0, 0, 255), 2);
	cv::line(*image, scene_corners[3], scene_corners[0], cv::Scalar(0, 0, 255), 2);
}


std::vector<cv::Point2f> ObjectTracker::getTrackingPoints(cv::Mat frame, cv::Mat previous_frame, std::vector<cv::Point2f> matched_points) {
	std::vector<cv::Point2f> matching_points;
	std::vector<uchar> status;
	std::vector<float> error;

	cv::Size search_window_size(7, 7);
	int maximal_pyramid_level = 3;
	cv::TermCriteria stop_criteria(cv::TermCriteria::Type::COUNT | cv::TermCriteria::Type::EPS, 30, 0.001);
	cv::calcOpticalFlowPyrLK(previous_frame, frame, matched_points, matching_points, status, error, search_window_size, maximal_pyramid_level/*, stop_criteria*/);

	drawRectangle(&frame, matched_points, matching_points);

	return matching_points;
}