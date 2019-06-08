#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\imgproc\imgproc_c.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2\xfeatures2d.hpp"

#include <vector> 

using namespace std;
using namespace cv;

Mat map1x, map1y, map2x, map2y;
Mat Ro, Tr;
void loadCameraparam()
{
	printf("Starting Calibration\n");
	Mat CM1 = Mat(3, 3, CV_64FC1);
	Mat CM2 = Mat(3, 3, CV_64FC1);
	Mat D1, D2;
	Mat R, T, E, F;

	FileStorage fs1("mystereocalib.yml", FileStorage::READ);
	fs1["CM1"] >> CM1;
	fs1["CM2"] >> CM2;
	fs1["D1"] >> D1;
	fs1["D2"] >> D2;
	fs1["R"] >> R;
	fs1["T"] >> T;
	fs1["E"] >> E;
	fs1["F"] >> F;
	Ro = R;
	Tr = T;
	printf("Starting Rectification\n");

	Mat R1, R2, P1, P2, Q;
	//stereoRectify(CM1, D1, CM2, D2, Size(1280, 720), R, T, R1, R2, P1, P2, Q);
	fs1["R1"] >> R1;
	fs1["R2"] >> R2;
	fs1["P1"] >> P1;
	fs1["P2"] >> P2;
	fs1["Q"] >> Q;
	fs1.release();
	printf("Done Rectification\n");

	printf("Applying Undistort\n");

	initUndistortRectifyMap(CM1, D1, R1, P1, Size(1280, 720), CV_32FC1, map2x, map2y);
	initUndistortRectifyMap(CM2, D2, R2, P2, Size(1280, 720), CV_32FC1, map1x, map1y);

	printf("Undistort complete\n");

}
void Sshow(string windowName,Mat M1, Mat M2,int scale=0,bool drawLines = false)
{
	Mat show;
	Mat Ssrc(Size(M1.cols * 2, M1.rows), M1.type(), Scalar::all(0));
	M1.copyTo(Ssrc(Rect(0, 0, M1.cols, M1.rows)));
	M2.copyTo(Ssrc(Rect(M1.cols, 0, M1.cols, M1.rows)));
	if (scale != 0)
		pyrDown(Ssrc, Ssrc, Size(Ssrc.cols / scale, Ssrc.rows / scale));
	
	if (drawLines)
		for (int i = 0; i < 19; i++)
			line(Ssrc, Point(0, i * 20), Point(1280, i * 20), Scalar(0, 0, 255));
	
	imshow(windowName, Ssrc);
}
void AreaOfIntrst(Mat src)
{
	Point P1(0, 0);
	Point P2(1280, 0);
	Point P3(1280, 500);
	Point P4(800, 250);
	Point P5(550, 250);
	Point P6(0, 500);

	Mat black(src.rows, src.cols, src.type(), cv::Scalar::all(0));
	Mat mask(src.rows, src.cols, CV_8UC1, cv::Scalar(0));


	vector< vector<Point> > co_ordinates;
	co_ordinates.push_back(vector<Point>());
	co_ordinates[0].push_back(P1);
	co_ordinates[0].push_back(P2);
	co_ordinates[0].push_back(P3);
	co_ordinates[0].push_back(P4);
	co_ordinates[0].push_back(P5);
	co_ordinates[0].push_back(P6);
	drawContours(mask, co_ordinates, 0, Scalar(255), CV_FILLED, 8);

	black.copyTo(src, mask);
}
Mat filter(Mat src)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat src_gray, grad;

	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);
	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	return grad;
}
Mat lineHough(Mat srcGray)
{
	vector<Vec4f> lines;

	HoughLinesP(srcGray, lines, 1, CV_PI / 180, 10, 20, 0);

	srcGray = Scalar::all(0);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		double x = l[2] - l[0];
		double y = l[3] - l[1];
		double teta = 0;
		if (x != 0)
		{
			teta = atan(y / x) * 180 / CV_PI;
		}
		if (teta > 30 || teta < -30)
			line(srcGray, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 3, CV_AA);
	}
	return srcGray;
}
vector<int> lHistogram(Mat src)
{
	Mat img_column[1280];
	int histSize = 2;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat hist[1280];

	for (int x = 0; x < src.cols; x++)
	{
		img_column[x] = Mat(src.rows, 1, src.type());
		for (int y = 0; y < src.rows; y++)
		{
			img_column[x].at<unsigned char>(y) = src.at<unsigned char>(y, x);
		}
		//img_column[x] = src.col(x);
	}

	Mat histImage = src.clone();
	histImage.convertTo(histImage, CV_8UC3);
	float pf = 720.0;
	int x[2];
	for (int i = 0; i < (src.cols/2); i++)
	{
		calcHist(&img_column[i], 1, 0, Mat(), hist[i], 1, &histSize, &histRange, uniform, accumulate);
		float f = hist[i].at<float>(0);
		if (f < pf)
		{
			pf = f;
			x[0] = i;
		}
			
		line(histImage, Point(i, 720),Point(i,cvRound(f)), Scalar(255, 255, 255));
	}
	for (int i = (src.cols/2) ; i < src.cols; i++)
	{
		calcHist(&img_column[i], 1, 0, Mat(), hist[i], 1, &histSize, &histRange, uniform, accumulate);
		float f = hist[i].at<float>(0);
		if (f < pf)
		{
			pf = f;
			x[1] = i;
		}

		line(histImage, Point(i, 720), Point(i, cvRound(f)), Scalar(255, 255, 255));
	}
	//line(histImage, Point(x, 0), Point(x, 720), Scalar(255, 255, 255),50);
	//cout << x << endl;
	imshow("calcHist Demo", histImage);
	vector<int> res;
	res.push_back(x[0]);
	res.push_back(x[1]);
	return res;
}
vector<int> cHistogram(Mat src)
{
	vector<int> res;
	int x[2];
	Mat histImage = src.clone();
	int* histogram1 = new int[src.cols/2];
	int* histogram2 = new int[src.cols/2];
	int pre = 0;
	for (int i = 0; i < src.cols / 2; i++)
	{
		histogram1[i] = src.rows - countNonZero(src.col(i));
		line(histImage, Point(i, 720), Point(i, histogram1[i]), Scalar(255, 255, 255));

		if ((src.rows - histogram1[i]) > pre)
		{
			pre = src.rows - histogram1[i];
			x[0] = i;
		}
			
	}
	pre = 0;
	for (int i = src.cols/2; i < src.cols; i++)
	{
		histogram2[i- src.cols / 2] = src.rows - countNonZero(src.col(i));
		line(histImage, Point(i, 720), Point(i, histogram2[i- src.cols / 2]), Scalar(255, 255, 255));
		if ((src.rows - histogram2[i - src.cols / 2]) > pre)
		{
			pre = src.rows - histogram2[i - src.cols / 2];
			x[1] = i;
		}
			
	}

	imshow("calcHist Demo", histImage);
	//Mat histogram(Size(src.cols, 1), CV_32S, Scalar(0));
	//for (int j = 0; j<src.rows; j++)
	//	histogram += (src.row(j) == 0);
	//histogram /= 255;

	res.push_back(x[0]);
	res.push_back(x[1]);

	return res;

}
int main()
{
	Mat left, leftU;
	vector<int> x;
	Rect region_of_interest;
	Point2f src[4];
	Point2f dst[4];
	VideoWriter vw;
	vw.open("LaneDeparture.mpeg", CV_FOURCC('M', 'P', 'E', 'G'), 30, Size(640, 480));
	VideoCapture mypic1("vp\\set00_V000.avi");

	//VideoCapture mypic1(2);
	//mypic1.set(CAP_PROP_FRAME_WIDTH, 1280);
	//mypic1.set(CAP_PROP_FRAME_HEIGHT, 720);


	if (!mypic1.isOpened())return 1;

	//loadCameraparam();



	////Image 1280 720
	//src[0] = Point2f(550, 430);
	//src[1] = Point2f(850, 430);
	//src[2] = Point2f(1280,620);
	//src[3] = Point2f(0, 620);

	//dst[0] = Point2f(0, 0);
	//dst[1] = Point2f(1280, 0);
	//dst[2] = Point2f(1280, 720);
	//dst[3] = Point2f(   0, 720);

	//image 640 480
	src[0] = Point2f(280, 200);
	src[1] = Point2f(400, 200);
	src[2] = Point2f(640, 400);
	src[3] = Point2f(0, 400);

	dst[0] = Point2f(0, 0);
	dst[1] = Point2f(640, 0);
	dst[2] = Point2f(640, 480);
	dst[3] = Point2f(0, 480);

	Mat M = getPerspectiveTransform(src, dst);
	Mat Mr = getPerspectiveTransform(dst,src);
	Mat res,warp;
	while (true)
	{
		mypic1 >> leftU;
		//remap(left, leftU, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		
		//AreaOfIntrst(leftU);
		//leftU = filter(leftU);
		//inRange(leftU, 100, 255, leftU);
		//threshold(leftU, leftU, 100, 255, THRESH_BINARY);
		//leftU = lineHough(leftU);

		warpPerspective(leftU, left, M, left.size());
		cvtColor(left, res, CV_BGR2GRAY);
		threshold(res, res, 210, 255, THRESH_BINARY);
		res.convertTo(res, CV_8U);
		x = cHistogram(res);

		//line(left, Point(x[0], 0), Point(x[0], 720), Scalar(0, 0, 255),30);
		//line(left, Point(x[1], 0), Point(x[1], 720), Scalar(0, 0, 255),30);
		rectangle(left, Rect(x[0], 0, x[1]-x[0] , 720), Scalar(0, 255, 0),CV_FILLED);

		warpPerspective(left, warp, Mr, left.size());
		addWeighted(leftU, 0.7, warp, 0.3, 1.0,res);
		imshow("src", left);
		vw.write(res);
		//imshow("warp", leftU);
		imshow("result", res);		

		if(waitKey(30) == 27)
			break;
	}
	vw.release();
	mypic1.release();

	return(0);
}