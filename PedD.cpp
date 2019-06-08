#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\imgproc\imgproc_c.h"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <vector> 
#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

int thresh=100;
int max_thresh = 255;

Mat src,srcgray,res;

CascadeClassifier myfullbodydetector;
CascadeClassifier myCardetector;
HOGDescriptor hog;
Net net;

#define RED Scalar(0,0,255)
#define GREEN Scalar(0,255,0)
#define BLUE Scalar(255,0,0)
#define WHITE Scalar(255,255,255)
#define BLACK Scalar(0,0,0)
struct AssoBodies
{
	vector<int> C1;
	vector<int> C2;
	vector<float> IoU;
};


void ontrackbar(int,void*)
{
	printf("\nthresh is:%d",thresh);
}

Mat cameraMatrix, distCoeffs,temp;

int readcameraparam()
{
    
    const string inputSettingsFile = "out_camera_data.xml";
    
	FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl; 
        return 0;
    }

	fs["Camera_Matrix" ] >> cameraMatrix;
	fs["Distortion_Coefficients" ] >> distCoeffs;
    fs.release();  

}

vector<cv::Rect> CarDetector(Mat img)
{
	vector<cv::Rect> Cars;
	cvtColor(src, srcgray, CV_BGR2GRAY);
	equalizeHist(srcgray, srcgray);
	myCardetector.detectMultiScale(srcgray, Cars, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(3, 3));
	return Cars;
}
vector<cv::Rect> HaarDetector(Mat img)
{
	vector<cv::Rect> Bodies;
	cvtColor(img, srcgray, CV_BGR2GRAY);
	equalizeHist(srcgray, srcgray);
	myfullbodydetector.detectMultiScale(srcgray, Bodies, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE);// Size(5, 5)
	return Bodies;
}
vector<cv::Rect> SVMDetector(Mat img)
{
	vector<Rect> found, found_filtered, Bodies;
	size_t i, j;
	hog.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);

	for (i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		for (j = 0; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	for (i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		Bodies.push_back(r);
	}
	return Bodies;
}
vector<cv::Rect> CNNDetector(Mat img)
{
	vector<cv::Rect> Bodies;
	const size_t inWidth = 640;
	const size_t inHeight = 480;
	const float inScaleFactor = 0.007843f;
	const float meanVal = 127.5;

	cvtColor(img, img, COLOR_BGRA2BGR);

	Mat inputBlob = blobFromImage(img, inScaleFactor,
		Size(inWidth, inHeight),
		Scalar(meanVal, meanVal, meanVal),
		false); //Convert Mat to batch of images
	
	//! [Set input blob]
	net.setInput(inputBlob); //set the network input
	//! [Set input blob]

	//! [Make forward pass]
	Mat detection = net.forward(); //compute output
	//! [Make forward pass]
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	float confidenceThreshold = 0.4;//parser.get<float>("min_confidence");
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
			if (objectClass == 15)
			{
				int left = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
				int top = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
				int right = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
				int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);
				Rect r = Rect(Point(left, top), Point(right, bottom));
				Bodies.push_back(r);
			}
		}
	}

	return Bodies;
}
float IoU_Cal(Rect r1, Rect r2)
{
	float IoU = 0;
	Rect I;
	if (r1.x >= r2.x)
		I.x = r1.x;
	else
		I.x = r2.x;
	
	if (r1.y >= r2.y)
		I.y = r1.y;
	else
		I.y = r2.y;

	int tw1 = r1.x + r1.width;
	int tw2 = r2.x + r2.width;
	int th1 = r1.y + r1.height;
	int th2 = r2.y + r2.height;
	
	if (tw1 <= tw2)
		I.width = tw1 - I.x;
	else
		I.width = tw2 - I.x;

	if (th1 <= th2)
		I.height = th1 - I.y;
	else
		I.height = th2 - I.y;

	if (I.width > 0 & I.height > 0)
	{
		float Intersection = I.width * I.height;
		float Area1 = r1.width * r1.height;
		float Area2 = r2.width * r2.height;
		IoU = Intersection / (Area1 + Area2 - Intersection);
	}
	else
		IoU = 0;


	return IoU;
	
	////Test methode
	//Rect r1 = Rect(0, 0, 100, 100);
	//Rect r2 = Rect(0, 25, 100, 100);
	//float i = IoU_Cal(r1, r2);
	//printf("IoU : %2.2f ", i);
	//getchar();
}
AssoBodies Association(vector<cv::Rect> C1, vector<cv::Rect> C2)
{
	AssoBodies A;
	
	for (size_t i = 0; i < C1.size(); i++)
	{
		float iou = 0;
		size_t j = 0;
		for (j = 0; j < C2.size(); j++)
		{
			float temp = IoU_Cal(C1.at(i), C2.at(j));
			if (temp>iou)
				iou = temp;

		}
		A.C1.push_back(i);
		A.C2.push_back(j);
		A.IoU.push_back(iou);
	}
	return A;
}
vector<float> weights()
{
	int img_num = 10;
	vector<float> weis;
	vector<cv::Rect> HaarBodies;
	vector<cv::Rect> SVMBodies;
	vector<cv::Rect> RCNNBodies;
	float s = 0, c = 0, h = 0;

	for (int i = 0; i < img_num; i++)
	{
		char str[20];
		sprintf_s(str, "vp\\body\\%d.jpg", i + 1);
		src = imread(str, 1);
		if (src.empty())
		{
			printf("Error Openning File...");
			break;
		}

		HaarBodies = HaarDetector(src);
		for (int i = 0; i < HaarBodies.size(); i++)
		{
			rectangle(src, HaarBodies[i], BLUE, 1);
			h++;
		}

		SVMBodies = SVMDetector(src);
		for (int i = 0; i < SVMBodies.size(); i++)
		{
			rectangle(src, SVMBodies[i], GREEN, 1);
			s++;
		}

		RCNNBodies = CNNDetector(src);
		for (int i = 0; i < RCNNBodies.size(); i++)
		{
			rectangle(src, RCNNBodies[i], WHITE, 1);
			c++;
		}

		imshow("Weights", src);
		if (waitKey(500) == 27) break;
	}

	weis.push_back(h / (h + s + c));
	weis.push_back(s / (h + s + c));
	weis.push_back(c / (h + s + c));

	return weis;
}
void main(void)
{
	vector<cv::Rect> HaarBodies;
	vector<cv::Rect> SVMBodies;
	vector<cv::Rect> RCNNBodies;
	vector<cv::Rect> GTBodies;

	//printf("\nInput num 0,1,2,3,4,5,6 to exit 100:");
	//scanf_s("%d", &in);
	char* resWindow = "Output Window";
	char* srcWindow = "Input Window";
	
	VideoWriter vw;
	vw.open("PedDetector.avi", CV_FOURCC('M', 'P', 'E', 'G'), 30, Size(640, 480));
	
	VideoCapture myvideo("vp\\Ped\\set00_V010.avi");
	if (!myvideo.isOpened())
	{
		printf("Error Openning Video File...");
		return;
	}

	//Load Haar training file
	if (myfullbodydetector.load("haarcascade_fullbody.xml") != true)
	{
		printf("Error Openning Haar_fullbody.xml File...");
		return;
	}
	//Load SVM Detector
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	///Load Car detector
	//if (myCardetector.load("cars.xml") != true)
	//{
	//	printf("Error Openning Haar Car detector File...");
	//	return;
	//}
	//Load CNN Detector
	String modelConfiguration = "Models\\MobileNetSSD_deploy.prototxt";
	String modelBinary = "Models\\MobileNetSSD_deploy.caffemodel";
	CV_Assert(!modelConfiguration.empty() && !modelBinary.empty());

	//! [Initialize network]
	net = dnn::readNetFromCaffe(modelConfiguration, modelBinary);
	//! [Initialize network]

	////Calculate Weights
	//vector<float> w = weights();
	//printf("\nHaar weight:%2.2f", w.at(0));
	//printf("\nSVM weight:%2.2f", w.at(1));
	//printf("\nCNN weight:%2.2f", w.at(2));
	//getchar();
	////!Calculate Weights
	
	char str[50];
	string info;
	string line;
	int i = 0;
	while (true)
	{

		//Read From CVC
		sprintf_s(str, "CVC-02\\CVC-02-CG\\data\\color\\frame%04d.png", i);
		//printf("image = %s    \r", str);
		src = imread(str, 1);

		sprintf_s(str, "CVC-02\\CVC-02-CG\\data\\annotations\\frame%04d.txt", i);
		ifstream infile(str);

		while (getline(infile, line))
		{
			std::istringstream iss(line);
			int a, b, c, d;
			if (!(iss >> a >> b >> c >> d)) { break; } // error
			GTBodies.push_back(Rect(a - (c / 2), b - (d / 2), c, d));
		}
		i++;
		if (i == 100)
			break;
		//End Read From CVC

		//myvideo >> src;

		HaarBodies = HaarDetector(src);
		for (int i = 0; i < HaarBodies.size(); i++)
		{
			rectangle(src, HaarBodies[i], BLUE, 1);
		}

		SVMBodies = SVMDetector(src);
		for (int i = 0; i < SVMBodies.size(); i++)
		{
			rectangle(src, SVMBodies[i], GREEN, 1);
		}

		RCNNBodies = CNNDetector(src);
		for (int i = 0; i < RCNNBodies.size(); i++)
		{
			rectangle(src, RCNNBodies[i], WHITE, 1);
		}

		//Calculate IoU
		printf("\n======== GT BB : %d ========", GTBodies.size());
		if (RCNNBodies.size() != 0)
		{
			AssoBodies AA = Association(GTBodies, RCNNBodies);
			printf("\nRCNN and GT : %d\n", RCNNBodies.size());
			for (size_t i = 0; i < AA.IoU.size(); i++)
				printf("%2.2f   ", AA.IoU.at(i));
		}
		if (SVMBodies.size() != 0)
		{
			AssoBodies AA = Association(GTBodies, SVMBodies);
			printf("\nSVM and GT : %d\n", SVMBodies.size());
			for (size_t i = 0; i < AA.IoU.size(); i++)
				printf("%2.2f   ", AA.IoU.at(i));
		}

		if (HaarBodies.size() != 0)
		{
			AssoBodies AA = Association(GTBodies, HaarBodies);
			printf("\nHaar and GT : %d\n", HaarBodies.size());
			for (size_t i = 0; i < AA.IoU.size(); i++)
				printf("%2.2f   ", AA.IoU.at(i));
		}
		//draw GroundTruth
		for (int i = 0; i < GTBodies.size(); i++)
		{
			rectangle(src, GTBodies[i], RED, 1);
		}
		
		vw << src;
		imshow(resWindow, src);
		if (waitKey() == 27) break;
		GTBodies.clear();
	}
	vw.release();
	destroyAllWindows();

}




