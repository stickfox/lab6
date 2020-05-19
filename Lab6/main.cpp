#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "ObjectRecognition.h"
#include "ObjectTracking.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	/* Read the video file and object image file */
	cv::String video_path;
	cv::String object_path;
	if (argc != 3) {
		cout << "If you don't set the file paths from the command line, I select video '../data/video.mov' and object image '../data/objects/obj2.png'\n" << endl;
		video_path = "../data/video.mov";
		object_path = "../data/objects/obj2.png";
	}
	else {
		video_path = argv[1];
		object_path = argv[2];
	}
	cv::VideoCapture video(video_path);
	cv::Mat object = cv::imread(object_path);
	if (object.empty()) {
		cout << "Error: unable to read file " << object_path << endl;
	}
	ObjectRecognition recognition = ObjectRecognition(video, object);
	
	
	/* Create PanoramicImage object and compute final panorama image */
	recognition.getMatching();

	/*
	cv::namedWindow("Final Panorama", WINDOW_AUTOSIZE);
	cv::imshow("Final Panorama", result);
	cv::waitKey(0);
	cv::destroyAllWindows();*/

	return 0;
}