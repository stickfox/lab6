#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>

#include "../include/PanoramicImage.h"

// debug define to print main step images
//#define SHOW_ORIGINAL
//#define SHOW_MODIFIED
//#define SHOW_K_POINTS
//#define SHOW_MATCHES

using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////
PanoramicImage::PanoramicImage(const double verticalFOV, const string path)
	: vFOV(verticalFOV)
{
	// save the images array inside class
	original_set = loadImages(path);

#ifdef SHOW_ORIGINAL
	printSet(original_set, 0);
#endif // SHOW_ORIGINAL
}

///////////////////////////////////////////////////////////
// sortStringVector : Private method to sort images namefiles
///////////////////////////////////////////////////////////
void PanoramicImage::sortStringVector(vector<string>& vec)
{
	vector<string> sorted_imgs;
	string min;
	string temp;
	for (int i = 0; i < vec.size() - 1; i++)
	{
		min = vec[i];
		for (int j = i + 1; j < vec.size(); j++)
		{
			if (vec[j].compare(min) < 0)
			{
				min = vec[j];
				temp = min;
				vec[j] = vec[i];
				vec[i] = temp;
			}
		}
	}
}

///////////////////////////////////////////////////////////
// loadImages : Method to load a set of images
///////////////////////////////////////////////////////////
vector<Mat> PanoramicImage::loadImages(string path)
{
	vector<String> filenames;
	vector<string> stdFileNames;
	string str;
	glob(path, filenames, false);
	for (int i = 0; i < filenames.size(); i++)
	{
		//cout << filenames[i] << endl;
		str = string(filenames[i]); // from cv::String to std::String
		stdFileNames.push_back(str);
	}
	// sort images using names alphabetical order to have them in the right order
	sortStringVector(stdFileNames);
	vector<Mat> total_imgs;
	for (int i = 0; i < stdFileNames.size(); i++)
	{
		total_imgs.push_back(imread(stdFileNames[i]));
	}

	cout << "There are " << total_imgs.size() << " images to merge." << endl;
	return total_imgs;
}

///////////////////////////////////////////////////////////
// computePanoramic : Compute the panoramic and save it inside the class
///////////////////////////////////////////////////////////
void PanoramicImage::computePanoramic(double ratio)
{
#ifdef SHOW_MODIFIED
	printSet(im_set, 0);
#endif // SHOW_MODIFIED

	// Project the images on a cylinder surface
	cylProj();

	// Extract the ORB features from the images
	orbFeaturesExtractor();

	// Compute the match between the different features of each
	// (consecutive) couples of images
	orbMatcher();

	//// Extract the SIFT features from the images
	//siftFeaturesExtractor();
	//siftMatcher();

	// Refine the matches found
	matchesRefiner(ratio);

	// Computation of the set of inliers
	inliersRetriver();

	// Find the pixel distance
	findDistance();

	// Compute the merge between images by using translation
	//PanoramicImage::mergeImg( pano, totalMeanDist, totalInliersGoodMatches );
	mergeImg();
}

void PanoramicImage::print_panoramic_error(string method_name)
{
	cout << "The panoramic image has not been compute yet. Call 'computePanoramic(ratio)' before the method " << method_name << "." << endl;
	//exit(1);
}

///////////////////////////////////////////////////////////
// showPanoramic : Show the computed panoramic
///////////////////////////////////////////////////////////
void PanoramicImage::showPanoramic()
{
	// if the image has not been computed yet, print an error
	if (panoramic.empty())
	{
		print_panoramic_error("showPanoramic");
		return;
	}

	imshow("Panoramic", panoramic);
	waitKey(0);
	destroyAllWindows();
}

///////////////////////////////////////////////////////////
// savePanoramic : Save the panoramic in a png file
///////////////////////////////////////////////////////////
void PanoramicImage::savePanoramic(std::string dst_path)
{
	// if the image has not been computed yet, print an error
	if (panoramic.empty())
	{
		print_panoramic_error("savePanoramic");
		return;
	}

	imwrite(dst_path, panoramic);
	cout << "\n\nPanoramic image saved at " << dst_path << endl;
}

///////////////////////////////////////////////////////////
// cylProj : Method to project images on a cylinder surface
///////////////////////////////////////////////////////////
void PanoramicImage::cylProj()
{
	for (int i = 0; i < original_set.size(); i++)
	{
		im_set.push_back(PanoramicUtils::cylindricalProj(original_set[i], vFOV / 2));
	}
}

///////////////////////////////////////////////////////////
// orbFeaturesExtractor : Extract the ORB features from couples of consecutive images
///////////////////////////////////////////////////////////
void PanoramicImage::orbFeaturesExtractor()
{
	Ptr<Feature2D> orb = ORB::create();
	vector<KeyPoint> OutputKeyPoints;
	all_descriptors.resize(im_set.size());

	for (int i = 0; i < im_set.size(); i++)
	{
		orb->detectAndCompute(im_set[i], Mat(), OutputKeyPoints, all_descriptors[i]);
		all_k_points.push_back(OutputKeyPoints);
	}

#ifdef SHOW_K_POINTS
	printKeySet(0);
#endif // SHOW_K_POINTS
}


///////////////////////////////////////////////////////////
// SiftFeaturesExtractor : Extract the SIFT features from couples of consecutive images
///////////////////////////////////////////////////////////
void PanoramicImage::siftFeaturesExtractor()
{
	vector<KeyPoint> keypoints;
	Mat descriptors;
	// Mat output;

	for (int i = 0; i < im_set.size(); i++)
	{
		Ptr<xfeatures2d::SiftFeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create();
		Ptr<xfeatures2d::SiftDescriptorExtractor> extractor = xfeatures2d::SiftDescriptorExtractor::create();
		detector->detect(im_set[i], keypoints);
		extractor->compute(im_set[i], keypoints, descriptors);
		all_k_points.push_back(keypoints);
		all_descriptors.push_back(descriptors);
		// drawKeypoints( pano.imSet[i], keypoints, output );
		// namedWindow("SIFT result wawa");
		// imshow( "SIFT result wawa", output );
		// waitKey(10);

	}
}

///////////////////////////////////////////////////////////
// matcher :
// Compute the match between the different features of each
// (consecutive) couples of images
///////////////////////////////////////////////////////////
void PanoramicImage::orbMatcher()
{
	cout << "\nFinding matches..." << endl;
	vector<DMatch> matches;
	Ptr<BFMatcher> bf_matcher = BFMatcher::create(NORM_HAMMING, true);
	for (int i = 0; i < all_descriptors.size() - 1; i++)
	{
		bf_matcher->match(all_descriptors[i], all_descriptors[i + 1], matches);
		all_matches.push_back(matches);
	}
	cout << "Matches found!" << endl;
}

void PanoramicImage::siftMatcher()
{
	cout << "\nFinding matches..." << endl;
	vector<DMatch> matches;
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L1, false);
	for (int i = 0; i < all_descriptors.size() - 1; i++)
	{
		matcher->match(all_descriptors[i], all_descriptors[i + 1], matches);
		all_matches.push_back(matches);
	}
	cout << "Matches found!" << endl;
}

///////////////////////////////////////////////////////////
// matchesRefiner :
// Refine the matches found above by selecting the matches 
// with distance less than ratio* min_distance,
// where ratio is a user - defined threshold
// and min_distance is the minimum distance found among the matches.
///////////////////////////////////////////////////////////
void PanoramicImage::matchesRefiner(double ratio)
{
	// Find the minimum distance
	float current_dist;
	float min_dist; // candidate to be the minimum
	vector<DMatch> current_refined_matches;

	vector<float> min_distances; // totalDist[i] contains the minDist between the matches of (i) and (i+1) images.
	for (int i = 0; i < all_matches.size(); i++)
	{
		min_dist = all_matches[i][0].distance;
		for (int j = 0; j < all_matches[i].size(); j++)
		{
			current_dist = all_matches[i][j].distance;
			if (current_dist < min_dist)
			{
				min_dist = current_dist;
			}
		}
		min_distances.push_back(min_dist);
	}

	// Refinement
	if (ratio == 0)
	{
		ratio = 3;
	}
	for (int i = 0; i < all_matches.size(); i++)
	{
		for (int j = 0; j < all_matches[i].size(); j++)
		{
			current_dist = all_matches[i][j].distance;
			/*if (dist <= totalDist[i] * ratio)*/
			if (current_dist <= min_distances[i] * ratio)
			{
				current_refined_matches.push_back(all_matches[i][j]);
			}
		}
		all_refined_matches.push_back(current_refined_matches);
		current_refined_matches.clear(); // clear refinedMatches

	}
	cout << "\nRefining: DONE!" << endl;

#ifdef SHOW_MATCHES
	for (int i = 0; i < all_matches.size(); i++)
	{
		Mat imMatches;
		drawMatches(im_set[i], all_k_points[i], im_set[i + 1], all_k_points[i + 1], all_refined_matches[i], imMatches);
		imshow("Matches", imMatches);
		waitKey(0);
	}
#endif // SHOW_MATCHES

}

///////////////////////////////////////////////////////////
// inliersRetriver : Computation of the set of inliers
///////////////////////////////////////////////////////////
void PanoramicImage::inliersRetriver()
{
	vector<DMatch> good_matches;

	// Extract location of good matches
	vector<Point2f> points1, points2;
	Mat hmask;
	vector<Mat> all_hmask;
	for (int i = 0; i < all_k_points.size() - 1; i++)
	{
		for (int j = 0; j < all_refined_matches[i].size(); j++)
		{
			points1.push_back(all_k_points[i][all_refined_matches[i][j].queryIdx].pt);
			points2.push_back(all_k_points[i + 1][all_refined_matches[i][j].trainIdx].pt);
		}
		findHomography(points1, points2, hmask, RANSAC);
		all_hmask.push_back(hmask);
		points1.clear();
		points2.clear();
	}
	for (int i = 0; i < all_refined_matches.size(); i++)
	{
		for (int j = 0; j < all_refined_matches[i].size(); j++)
		{
			if ((int)all_hmask[i].Mat::at<uchar>(j, 0))
			{
				good_matches.push_back(all_refined_matches[i][j]);
			}
		}
		all_inliers_good_matches.push_back(good_matches);
		good_matches.clear();
	}
	cout << "\nInliers retrieved." << endl;
}


///////////////////////////////////////////////////////////
// findDistance : 
// to compute the width of panoramic image, 
// consider the projected images widths, and the translations along x.
///////////////////////////////////////////////////////////
void PanoramicImage::findDistance()
{
	Point2f point1, point2;
	float dist;
	vector<float> distance;
	for (int i = 0; i < all_k_points.size() - 1; i++)
	{
		for (int j = 0; j < all_inliers_good_matches[i].size(); j++)
		{
			point1 = all_k_points[i][all_inliers_good_matches[i][j].queryIdx].pt;
			point2 = all_k_points[i + 1][all_inliers_good_matches[i][j].trainIdx].pt;
			dist = im_set[i].cols - point1.x + point2.x;
			distance.push_back(dist);
		}
		all_distances.push_back(distance);
		distance.clear();
	}
}


void PanoramicImage::mergeImg()
{
	// Compute the mean distance between couples of images
	float dist;
	float numInliers;
	for (int i = 0; i < all_inliers_good_matches.size(); i++)
	{
		dist = 0;
		numInliers = all_inliers_good_matches[i].size();
		for (int j = 0; j < numInliers; j++)
		{
			dist = dist + all_distances[i][j];
		}
		dist = dist / numInliers;
		mean_dist.push_back(dist);
	}
	cout << "\nMerging images..." << endl;

	// Apply translation of mean distance value
	Mat shiftMat(2, 3, CV_64F, Scalar(0.0));
	shiftMat.Mat::at<double>(0, 0) = 1;
	shiftMat.Mat::at<double>(1, 1) = 1;
	shiftMat.Mat::at<double>(0, 1) = 0;
	shiftMat.Mat::at<double>(1, 0) = 0;
	shiftMat.Mat::at<double>(1, 2) = 0;
	Mat dst;
	panoramic = im_set[0];
	for (int i = 0; i < im_set.size() - 1; i++)
	{
		shiftMat.Mat::at<double>(0, 2) = -mean_dist[i];
		warpAffine(im_set[i + 1], dst, shiftMat, Size(im_set[i + 1].cols - mean_dist[i], im_set[i + 1].rows), INTER_CUBIC, BORDER_CONSTANT, Scalar());
		hconcat(panoramic, dst, panoramic);
	}
	cout << "\nOperation Finished!" << endl;
}


///////////////////////////////////////////////////////////
// Utils
///////////////////////////////////////////////////////////

// printSet : Method to print images that belong to an image set.
void PanoramicImage::printSet(std::vector<cv::Mat> set, const double t)
{
	char PATH_NAME[35];
	for (int i = 0; i < set.size(); i++)
	{
		sprintf(PATH_NAME, "IMG_%d", i + 1);
		namedWindow(PATH_NAME);
		imshow(PATH_NAME, set[i]);
		waitKey(t);
		destroyWindow(PATH_NAME);
	}
}

// printKeySet : Method to print keypoints of a set of images.
void PanoramicImage::printKeySet(const double t)
{
	char PATH_NAME[35];
	Mat output;
	for (int i = 0; i < im_set.size(); i++)
	{
		sprintf(PATH_NAME, "IMG_%d", i + 1);
		namedWindow(PATH_NAME);
		drawKeypoints(im_set[i], all_k_points[i], output);
		imshow(PATH_NAME, output);
		waitKey(t);
		destroyWindow(PATH_NAME);
	}
}