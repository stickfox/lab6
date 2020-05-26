#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "ObjectTracker.cpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	cv::Mat loading = cv::imread("../data/loading.png");
	cv::namedWindow("Frame");
	imshow("Frame", loading);
	waitKey(1000);


	/* Read the video file and object image file */
	cv::String video_path;
	cv::String object_path1;
	cv::String object_path2;
	cv::String object_path3;
	cv::String object_path4;
	if (argc != 3) {
		cout << "If you don't set the file paths from the command line, I select video '../data/video.mov' and object image '../data/objects/obj2.png'\n" << endl;
		video_path = "../data/video.mov";
		object_path1 = "../data/objects/obj1.png";
		object_path2 = "../data/objects/obj2.png";
		object_path3 = "../data/objects/obj3.png";
		object_path4 = "../data/objects/obj4.png";
	}
	else {
		video_path = argv[1];
		//object_path = argv[2];
	}
	cv::VideoCapture video_cap(video_path);
	cv::Mat object1 = cv::imread(object_path1);
	cv::Mat object2 = cv::imread(object_path2);
	cv::Mat object3 = cv::imread(object_path3);
	cv::Mat object4 = cv::imread(object_path4);
	//if (object.empty()) {
	//	cout << "Error: unable to read file " << object_path << endl;
	//}

	/* Resize the image to improve visualization */
	cv::Size size2(object1.cols * 0.5, object1.rows * 0.5);
	cv::resize(object1, object1, size2);

	cv::Size size3(object2.cols * 0.5, object2.rows * 0.5);
	cv::resize(object2, object2, size3);

	cv::Size size4(object3.cols * 0.5, object3.rows * 0.5);
	cv::resize(object3, object3, size4);

	cv::Size size5(object4.cols * 0.5, object4.rows * 0.5);
	cv::resize(object4, object4, size5);

	ObjectTracker tracker1 = ObjectTracker(video_cap, object1, cv::Scalar(0, 0, 255));
	ObjectTracker tracker2 = ObjectTracker(video_cap, object2, cv::Scalar(0, 255, 0));
	ObjectTracker tracker3 = ObjectTracker(video_cap, object3, cv::Scalar(255, 0, 0));
	ObjectTracker tracker4 = ObjectTracker(video_cap, object4, cv::Scalar(255, 100, 250));

	char c = 0;
	int frame_width = 0, frame_height = 0;
	cv::Mat frame, previous_frame;
	std::vector<cv::Point2f> matching_points1, matched_points1;
	std::vector<cv::Point2f> matching_points2, matched_points2;
	std::vector<cv::Point2f> matching_points3, matched_points3;
	std::vector<cv::Point2f> matching_points4, matched_points4;
	std::vector<char> error;
	cv::VideoWriter video_writer;
	
	int i = 0;
	if (video_cap.isOpened()) {
		//for (int i = 0;; i++) {
		//	if ((i % 10) != 0)
		//		continue;
		while (video_cap.read(frame))
		{
			video_cap.read(frame);
			video_cap.read(frame);
			video_cap.read(frame);
			video_cap.read(frame);
			video_cap.read(frame);
			video_cap.read(frame);
			video_cap.read(frame);
			video_cap.read(frame);
			video_cap.read(frame);

			//video_cap >> frame;
			if (frame.empty())
				break;

			cv::Size size(frame.cols * 0.5, frame.rows * 0.5);
			cv::resize(frame, frame, size); // resize also the frame
			if (i == 0) {
				frame_width = video_cap.get(cv::CAP_PROP_FRAME_WIDTH);
				frame_height = video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
				video_writer = cv::VideoWriter("outcpp.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height)); //fourcc('P', 'I', 'M', '1')
				matched_points1 = tracker1.getMatchingPoints(frame, object1, 6500, 30000);
				matched_points2 = tracker2.getMatchingPoints(frame, object2, 450, 450);
				matched_points3 = tracker3.getMatchingPoints(frame, object3, 500, 500);
				matched_points4 = tracker4.getMatchingPoints(frame, object4, 6500, 30000);

				frame.copyTo(previous_frame);
				matching_points1 = matched_points1;
				matching_points2 = matched_points2;
				matching_points3 = matched_points3;
				matching_points4 = matched_points4;
			}
			else {
				matching_points1 = tracker1.getTrackingPoints(frame, previous_frame, matching_points1);
				matching_points2 = tracker2.getTrackingPoints(frame, previous_frame, matching_points2);
				matching_points3 = tracker3.getTrackingPoints(frame, previous_frame, matching_points3);
				matching_points4 = tracker4.getTrackingPoints(frame, previous_frame, matching_points4);

				frame.copyTo(previous_frame);
			}

			//video_writer.write(frame);
			imshow("Frame", frame);
			char c = (char)waitKey(1);
			if (c == 27)
				break;

			i++;
		}
	}
	else {
		cout << "Error opening video stream" << endl;
		return -1;
	}

	/* When everything done, release the video capture and write object */
	video_cap.release();
	video_writer.release();

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}