#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>  // Video write
#include <iostream>
#include "ObjectTracker.cpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	// Loading splash image
	Mat loading = imread("../data/loading.png");
	if (loading.empty())
		loading = imread("../../data/loading.png");
	if (!loading.empty())
	{
		namedWindow("Frame");
		imshow("Frame", loading);
		waitKey(1000);
	}

	// Read the video file and object image files
	String video_path;
	String objects_path;

	if (argc < 4) {
		cout << "If you don't set the file paths from the command line, I select video '../data/video.mov' and object image '../data/objects/obj2.png'\n" << endl;
		video_path = "../data/video.mov";
		objects_path = "../data/objects/obj*.png";
	}
	else {
		video_path = argv[1];
		objects_path = argv[2];
		objects_path.append("*.");
		objects_path.append(argv[3]);
	}

	VideoCapture video_cap(video_path);

	vector<String> filenames;
	glob(objects_path, filenames, false);
	vector<Mat> objects;
	for (int i = 0; i < filenames.size(); i++)
	{
		cout << "Filename " << i << ": " << filenames[i] << endl;
		Mat tmp = imread(filenames[i]);

		if (!tmp.empty())
			objects.push_back(tmp);
	}
	
	if (objects.size() < 1) {
		cout << "Error: unable to read file " << objects_path << endl;
		cout << "**** Parameters are video_path, images folder + images prefix, images extensions." << endl;
		cout << "		executable ../data/video.mov ../data/objects/obj png" << endl;
	}

	cout << "\n\n" << endl;

	 // Resize the image to improve visualization
	for (int i = 0; i < objects.size(); i++)
	{
		Size size2(objects[i].cols * 0.5, objects[i].rows * 0.5);
		resize(objects[i], objects[i], size2);

	}

	// Initialize Trackers
	vector<Scalar> colors{ Scalar(0, 0, 255),
							Scalar(0, 255, 216),
							Scalar(255, 0, 0),
							Scalar(180, 130, 255) };
	vector<ObjectTracker> trackers;
	for (int i = 0; i < objects.size(); i++)
		trackers.push_back(ObjectTracker(objects[i], colors[i % colors.size()]));

	char c = 0;
	int frame_width = 0, frame_height = 0;
	Mat frame, previous_frame;
	vector<vector<Point2f>> matching_points;
	vector<char> error;
	VideoWriter video_writer;
	
	int i = 0;
	Mat clean_copy;
	if (video_cap.isOpened()) {
		while (video_cap.read(frame))
		{
			// Skip some frames to enhance speed
			video_cap.read(frame);
			//video_cap.read(frame);
			//video_cap.read(frame);
			//video_cap.read(frame);
			//video_cap.read(frame);
			//video_cap.read(frame);
			//video_cap.read(frame);
			//video_cap.read(frame);
			//video_cap.read(frame);

			if (frame.empty())
				break;

			// Resize also the current frame
			Size size(frame.cols * 0.5, frame.rows * 0.5);
			resize(frame, frame, size);

			// First iteration, initialize values and find features
			if (i == 0) {
				frame.copyTo(clean_copy);

				frame_width = frame.size().width;
				frame_height = frame.size().height;
				video_writer.open("output.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'), video_cap.get(CAP_PROP_FPS), Size(frame_width, frame_height), true);

				// Use SIFT
				for (int i = 0; i < objects.size(); i++)
				{
					// if 4th param is 0 -> use SIFT, if is 1 -> use ORB
					matching_points.push_back(trackers[i].getMatchingPoints(clean_copy, &frame, objects[i], 0, 0, 0));
				}
				// Use ORB
				//for (int i = 0; i < objects.size(); i++)
				//{
				//	vector<int> max1{ 7500, 500, 500, 6500 };
				//	vector<int> max2{ 40000, 500, 500, 30000 };

				//	matching_points.push_back(trackers[i].getMatchingPoints(clean_copy, &frame, objects[i], 1, max1[i % max1.size()], max2[i % max2.size()]));
				//}

				clean_copy.copyTo(previous_frame);
			}
			else {
				// Other iterations, find flow and draw rectangle and features for each object
				frame.copyTo(clean_copy);

				for (int i = 0; i < objects.size(); i++)
					matching_points[i] = trackers[i].getTrackingPoints(clean_copy, &frame, previous_frame, matching_points[i]);

				clean_copy.copyTo(previous_frame);
			}

			// Export frame and show it
			video_writer.write(frame);
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

	cout << "\nFINISH!\n\n" << endl;

	waitKey(0);
	destroyAllWindows();

	return 0;
}