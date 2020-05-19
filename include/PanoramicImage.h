#include <opencv2/core.hpp>
#include <vector>

#include "../src/panoramic_utils.cpp"

class PanoramicImage
{
	///////////////////////////////////////////////////////////
	// Public methods
	///////////////////////////////////////////////////////////
public:

	PanoramicImage(const double verticalFOV, const std::string path);

	void computePanoramic(double ratio);

	void showPanoramic();

	void savePanoramic(std::string dstPathName);

	///////////////////////////////////////////////////////////
	// Private methods
	///////////////////////////////////////////////////////////

private:

	void print_panoramic_error(std::string method_name);

	void sortStringVector(std::vector<std::string>& vec);

	std::vector<cv::Mat> loadImages(std::string path);

	void cylProj();

	void orbFeaturesExtractor();

	void orbMatcher();

	void siftFeaturesExtractor();

	void siftMatcher();

	void matchesRefiner(double ratio);

	void inliersRetriver();

	void findDistance();

	void mergeImg();

	// Utils
	void printSet(std::vector<cv::Mat> set, const double t);

	void printKeySet(const double t);

	///////////////////////////////////////////////////////////
	// Attributes
	///////////////////////////////////////////////////////////

	// Original images set
	std::vector<cv::Mat> original_set;
	// For image cylindrical projection.
	std::vector<cv::Mat> im_set;

	const double vFOV; // vertical field of view [degrees]

	// computed panoramic image
	cv::Mat panoramic;

	// For extraction of ORB/SIFT features.
	std::vector<std::vector<cv::KeyPoint>> all_k_points;
	std::vector<cv::Mat> all_descriptors;

	// For the matcher
	std::vector<std::vector<cv::DMatch>> all_matches;

	// For refinement of the matches
	std::vector<std::vector<cv::DMatch>> all_refined_matches;

	// For inliers retrivement
	std::vector<std::vector<cv::DMatch>> all_inliers_good_matches;

	// For merging
	std::vector<float> mean_dist;

	// For find distance and merge image
	std::vector<std::vector<float>> all_distances;
};
