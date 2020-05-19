#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "PanoramicImage.cpp"


#define FOCAL_LEN 18 //mm
#define vFOV 66 //degrees
#define halfvFOV 33 // degrees
#define RATIO 3 // int non-zero value. If zero, default value = 3

using namespace cv;
using namespace std;
using namespace xfeatures2d;

void printLoader(string folder)
{
	cout << "\n\n *** MERGING images from folder " << folder << " *** " << endl;
}

int main(int argc, char** argv)
{
	std::string folder;
	std::string files_extension;
	std::string images_path;
	
	////////////////////////////////////////////////////////////////////
	// Lab images
	////////////////////////////////////////////////////////////////////
	folder = "lab";
	files_extension = "bmp";
	images_path = "../data/" + folder + "/*." + files_extension;
	
	printLoader(folder);
	PanoramicImage p_image = PanoramicImage(vFOV, images_path);

	// Show and save lab.png
	// If you want to use a different dataset change the following path. Remember
	// to change also FOCAL_LEN and vFOV parameters, and the images extension on the path.
	p_image.computePanoramic(RATIO);
	p_image.showPanoramic();
	p_image.savePanoramic("../data/result/lab.png");

	////////////////////////////////////////////////////////////////////
	// dataset_lab_19_automatic
	////////////////////////////////////////////////////////////////////
	folder = "dataset_lab_19_automatic";
	files_extension = "png";
	images_path = "../data/" + folder + "/*." + files_extension;
	
	printLoader(folder);
	PanoramicImage p_image2 = PanoramicImage(vFOV, images_path);

	p_image2.computePanoramic(RATIO);
	p_image2.showPanoramic();
	p_image2.savePanoramic("../data/result/dataset_lab_19_automatic.png");

	////////////////////////////////////////////////////////////////////
	// dataset_lab_19_manual
	////////////////////////////////////////////////////////////////////
	folder = "dataset_lab_19_manual";
	files_extension = "png";
	images_path = "../data/" + folder + "/*." + files_extension;

	printLoader(folder);
	PanoramicImage p_image3 = PanoramicImage(vFOV, images_path);

	p_image3.computePanoramic(RATIO);
	p_image3.showPanoramic();
	p_image3.savePanoramic("../data/result/dataset_lab_19_manual.png");

	////////////////////////////////////////////////////////////////////
	// kitchen
	////////////////////////////////////////////////////////////////////
	folder = "kitchen";
	files_extension = "bmp";
	images_path = "../data/" + folder + "/*." + files_extension;
	
	printLoader(folder);
	PanoramicImage p_image4 = PanoramicImage(vFOV, images_path);

	p_image4.computePanoramic(RATIO);
	p_image4.showPanoramic();
	p_image4.savePanoramic("../data/result/kitchen.png");

	////////////////////////////////////////////////////////////////////
	// dolomites
	////////////////////////////////////////////////////////////////////
	folder = "dolomites";
	files_extension = "png";
	images_path = "../data/" + folder + "/*." + files_extension;
	
	//int newvFOV = 54;
	//PanoramicImage p_image5 = PanoramicImage(newvFOV, images_path);
	printLoader(folder);
	PanoramicImage p_image5 = PanoramicImage(vFOV, images_path);

	p_image5.computePanoramic(RATIO);
	p_image5.showPanoramic();
	p_image5.savePanoramic("../data/result/dolomites.png");

	return 0;
}
