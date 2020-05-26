#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "ObjectTracker.cpp"

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
	cv::VideoCapture video_cap(video_path);
	cv::Mat object = cv::imread(object_path);
	if (object.empty()) {
		cout << "Error: unable to read file " << object_path << endl;
	}

	/* Resize the image to improve visualization */
	cv::Size size2(object.cols * 0.5, object.rows * 0.5);
	cv::resize(object, object, size2);

	ObjectTracker tracker = ObjectTracker(video_cap, object);
	
	char c = 0;
	int frame_width = 0, frame_height = 0;
	cv::Mat frame, previous_frame;
	std::vector<cv::Point2f> matching_points, matched_points;
	std::vector<char> error;
	cv::VideoWriter video_writer;
	cv::namedWindow("Frame");
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

			//video_cap >> frame;
			if (frame.empty())
				break;

			cv::Size size(frame.cols * 0.5, frame.rows * 0.5);
			cv::resize(frame, frame, size); // resize also the frame
			if (i == 0) {
				frame_width = video_cap.get(cv::CAP_PROP_FRAME_WIDTH);
				frame_height = video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
				video_writer = cv::VideoWriter("outcpp.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height)); //fourcc('P', 'I', 'M', '1')
				matched_points = tracker.getMatchingPoints(frame, object);
				previous_frame = frame;
				matching_points = matched_points;
			}
			else {
				matching_points = tracker.getTrackingPoints(frame, previous_frame, matching_points);
				//previous_frame = frame;
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