#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "ObjectRecognition.h"

using namespace cv;
using namespace std;

/* Constructor */
ObjectRecognition::ObjectRecognition(cv::VideoCapture vid_cap, cv::Mat obj) {
	video = vid_cap;
	object = obj;
	matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
	orb = cv::ORB::create(MAX_FEATURES);
}


/* Compute features of an image using ORB */
void ObjectRecognition::computeFeatures(cv::Mat projection, std::vector<cv::KeyPoint>* keypoint, cv::Mat* descriptor) {
	orb->detectAndCompute(projection, cv::Mat(), *keypoint, *descriptor);
}


/* Compute the initial cutting points in two consecutive images */
void ObjectRecognition::getTranslation(cv::Mat video_image, cv::Mat object_image) {
	std::vector<cv::KeyPoint> keypoint1, keypoint2;
	cv::Mat descriptor1, descriptor2;
	std::vector<cv::DMatch> matches;

	cv::Size size(video_image.cols * 0.5, video_image.rows * 0.5);
	cv::resize(video_image, video_image, size);
	cv::Size size2(object_image.cols * 0.5, object_image.rows * 0.5);
	cv::resize(object_image, object_image, size2);

	/* Extract keypoints and their descriptors, then match the keypoints */
	computeFeatures(video_image, &keypoint1, &descriptor1);
	computeFeatures(object_image, &keypoint2, &descriptor2);
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
	

	// Get the corners from the image ( the object to be "detected" )
	std::vector<cv::Point3f> obj_corners(4);
	obj_corners[0] = cv::Point3f(1, 1, 1);
	obj_corners[1] = cv::Point3f(object_image.cols - 1, 1, 1);
	obj_corners[2] = cv::Point3f(object_image.cols - 1, object_image.rows - 1, 1);
	obj_corners[3] = cv::Point3f(1, object_image.rows - 1, 1);
	std::vector<cv::Point3f> scene_corners(4);
	scene_corners[0] = cv::Point3f(1, 1, 1);
	cv::Mat pointMat1 = (cv::Mat_<float>(3, 1) << obj_corners[0].x, obj_corners[0].y, obj_corners[0].z);
	//cv::Mat cam_to_world(4, 4, CV_32FC1);
	//cv::Mat pointA_cam = cam_to_world * Mat(cv::Vec3f(obj_corners[0].x, obj_corners[0].y, obj_corners[0].z));
	//cv::Point3f point_A_wld(cam_to_world.at<float>(0, 0), cam_to_world.at<float>(1, 0), cam_to_world.at<float>(2, 0));

	cv::Mat pointMatRes1;

	cout << scene_corners[0] << "  " << h << "  " << pointMat1 << endl;

	//pointMatRes1 = h * Mat(cv::Vec3f(obj_corners[0].x, obj_corners[0].y, obj_corners[0].z));

	float r = h.at<double>(1, 0);
	cv::Point3f tmp = cv::Point3f(1, 1, 1);
	scene_corners[0].x = h.at<double>(0, 0) * obj_corners[0].x + h.at<double>(0, 1) * obj_corners[0].y + h.at<double>(0, 2) * obj_corners[0].z;
	scene_corners[0].y = h.at<double>(1, 0) * obj_corners[0].x + h.at<double>(1, 1) * obj_corners[0].y + h.at<double>(1, 2) * obj_corners[0].z;
	scene_corners[0].z = h.at<double>(2, 0) * obj_corners[0].x + h.at<double>(2, 1) * obj_corners[0].y + h.at<double>(2, 2) * obj_corners[0].z;



	cout << "MATRIX " << scene_corners[0] << "MATRIX " << endl;


	cv::Mat ht = Mat(2, 3, CV_64F);
	//cout << mask << endl;
	ht.at<double>(0, 0) = h.at<double>(0, 0); ht.at<double>(0, 1) = h.at<double>(0, 1); ht.at<double>(0, 2) = h.at<double>(0, 2);
	ht.at<double>(1, 0) = h.at<double>(1, 0); ht.at<double>(1, 1) = h.at<double>(1, 1); ht.at<double>(1, 2) = h.at<double>(1, 2);
	
/*
	p_transformed_homogenous = h * p_origin_homogenous =

		(h0, h1, h2)    (x)   (h0 * x + h1 * y + h2)   (tx)
		(h3, h4, h5) * (y) = (h3 * x + h4 * y + h5) = (ty)
		(h6, h7, h8)    (1)   (h6 * x + h7 * y + h8)   (tz)
		Convert p_transformed_homogenous to p_transformed_cartesian :

	(tx, ty, tz) = > (tx / tz, ty / tz)
		Your code translated :

	px = tx / tz;
	py = ty / tz;
	Z = 1 / tz;
	

	perspectiveTransform(obj_corners, scene_corners, h);
	// Draw lines between the corners (the mapped object in the scene )
	line(video_image, scene_corners[0], scene_corners[1], cv::Scalar(255, 0, 0), 4);
	line(video_image, scene_corners[1], scene_corners[2], cv::Scalar(255, 0, 0), 4);
	line(video_image, scene_corners[2], scene_corners[3], cv::Scalar(255, 0, 0), 4);
	line(video_image, scene_corners[3], scene_corners[0], cv::Scalar(255, 0, 0), 4);
	*/
	cout << obj_corners[0] << " -- " << obj_corners[1] << " -- " << obj_corners[2] << " -- " << obj_corners[3] << endl;
	cout << scene_corners[0] << " -- " << scene_corners[1] << " -- " << scene_corners[2] << " -- " << scene_corners[3] << endl;

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

	cv::Mat h1 = cv::findHomography(inliers1, inliers2, RANSAC, ransacThreshold, inlier_mask);
	cout << h1 << endl;

	//cv::line(object_image, point1, point2, Scalar(255, 255, 255), 2, 8);
	//cv::line(video_image, point_res1, point_res2, Scalar(255, 255, 255), 2, 8 );
	cv::namedWindow("LINE");
	cv::imshow("LINE", video_image);
	cv::waitKey(0);
	cv::Mat imReg;
	warpPerspective(video_image, imReg, h, object_image.size());
	cv::imshow("LINE", imReg);
	cv::waitKey(0);

	/* Draw matches */
	cv::Mat matching;
	cv::drawMatches(video_image, keypoint1, object_image, keypoint2, inlier_matches, matching);
	cv::namedWindow("MATCH");
	cv::imshow("MATCH", matching);
	cv::waitKey(0);

}

void ObjectRecognition::getMatching() {
	int k = 0;
	cv::Mat frame;

	if (video.isOpened()) {
		for (;;) {
			video >> frame;
			if (k == 0) {
				getTranslation(frame, object);
			}
			k++;
		}
	}

}