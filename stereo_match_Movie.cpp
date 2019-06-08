#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <opencv2/video.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat map1x, map1y, map2x, map2y,Qfour;

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

	printf("Starting Rectification\n");

	Mat R1, R2, P1, P2;
	//stereoRectify(CM1, D1, CM2, D2, Size(1280, 720), R, T, R1, R2, P1, P2, Q);
	fs1["R1"] >> R1;
	fs1["R2"] >> R2;
	fs1["P1"] >> P1;
	fs1["P2"] >> P2;
	fs1["Q"] >> Qfour;
	fs1.release();
	printf("Done Rectification\n");

	printf("Applying Undistort\n");

	initUndistortRectifyMap(CM1, D1, R1, P1, Size(1280, 720), CV_32FC1, map2x, map2y);
	initUndistortRectifyMap(CM2, D2, R2, P2, Size(1280, 720), CV_32FC1, map1x, map1y);

	printf("Undistort complete\n");

}
Mat cimage(Mat gray)
{
	Mat color_image = Mat(gray.size(), CV_8UC3);

	for (int i = 0; i < gray.rows; i++)
	{
		for (int j = 0; j < gray.cols; j++)
		{
			uchar H = (240 - gray.at<unsigned char>(i, j))/2;
			uchar S = 255;
			uchar V = 255;

			color_image.at<Vec3b>(i, j)[0] = H;
			color_image.at<Vec3b>(i, j)[1] = S;
			color_image.at<Vec3b>(i, j)[2] = V;
		}
	}
	cvtColor(color_image, color_image, CV_HSV2BGR);
	return color_image;
}
void Sshow(Mat M1, Mat M2, int scale = 0)
{
	//hconcat(M1, M2, Ssrc); //just this Line can do following commands

	Mat Ssrc(Size(M1.cols + M2.cols, M2.rows), M1.type(), Scalar::all(0));
	M1.copyTo(Ssrc(Rect(0, 0, M1.cols, M1.rows)));
	M2.copyTo(Ssrc(Rect(M1.cols, 0, M2.cols, M2.rows)));
	for (int i = 0; i < 38; i++)
	{
		Scalar s = i % 2 == 0 ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
		line(Ssrc, Point(0, i * 20), Point(Ssrc.cols, i * 20), s);
	}
	if (scale != 0)
		pyrDown(Ssrc, Ssrc, Size(Ssrc.cols / scale, Ssrc.rows / scale));

	imshow("Stereo Display", Ssrc);
}
Mat SharpenImage(Mat src)
{
	// Create a kernel that we will use to sharpen our image
	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1);

	Mat imgLaplacian;
	filter2D(src, imgLaplacian, CV_32F, kernel);
	Mat sharp;
	src.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	return imgResult;
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
int main(int argc, char** argv)
{

	VideoWriter vw;
	vw.open("Desparity.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(1280, 720));

	VideoCapture capL("vp\\Stereo1.avi");
	VideoCapture capR("vp\\Stereo2.avi");

	//VideoCapture capL(1);
	//VideoCapture capR(2);
	//capL.set(CAP_PROP_FRAME_WIDTH, 1280);
	//capL.set(CAP_PROP_FRAME_HEIGHT, 720);
	//capR.set(CAP_PROP_FRAME_WIDTH, 1280);
	//capR.set(CAP_PROP_FRAME_HEIGHT, 720);
	Mat img1, img2;

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4 };
    int alg = STEREO_SGBM;
    int SADWindowSize, numberOfDisparities;
    bool no_display;

    Ptr<StereoBM> bm = StereoBM::create(16,9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);

    numberOfDisparities = 96;
    SADWindowSize = 3;
    no_display = false;

    int color_mode = alg == STEREO_BM ? 0 : -1;


    Size img_size = img1.size();

    Rect roi1, roi2;

	loadCameraparam();

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(1);
    bm->setUniquenessRatio(300);
    bm->setSpeckleWindowSize(150);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = img1.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(10);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(40);
    sgbm->setSpeckleWindowSize(80);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    if(alg==STEREO_HH)
        sgbm->setMode(StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)
        sgbm->setMode(StereoSGBM::MODE_SGBM);
    else if(alg==STEREO_3WAY)
        sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

    Mat disp, disp8;
	Mat tmp1, tmp2;
	int64 t;
	Mat video;
	capL >> img1;//sink
	while (true)
	{
		capL >> img1;
		capR >> img2;
		if (img1.empty() | img2.empty())
		{
			printf("ERROR in Images...");
			break;
		}

		remap(img1, img1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		remap(img2, img2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		//img1 = SharpenImage(tmp1);
		//img2 = SharpenImage(tmp2);
		//AreaOfIntrst(img1);
		//AreaOfIntrst(img2);

		t = getTickCount();
		if (alg == STEREO_BM)
		{
			cvtColor(img1, img1, CV_BGR2GRAY);
			cvtColor(img2, img2, CV_BGR2GRAY);
			bm->compute(img1, img2, disp);
		}
		else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY)
			sgbm->compute(img1, img2, disp);
		t = getTickCount() - t;
		//printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());

		//disp = dispp.colRange(numberOfDisparities, img1p.cols);
		if (alg != STEREO_VAR)
			disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
		else
			disp.convertTo(disp8, CV_8U);

		cvtColor(disp8, video, CV_GRAY2BGR);

		Mat Csrc = cimage(disp8);
		if (!no_display)
		{
			Sshow(img1, img2, 2.0);
			//imshow("disparity", disp8);
			imshow("Color", Csrc);
		}
		vw.write(video);
		char c = waitKey(200);
		if (c == 27)
			break;

		switch (c)
		{
		case ' ':
			c = waitKey();
			if (c == 's')
				imwrite("output.jpg", Csrc);
			break;
		}
		
	}
	capL.release();
	capR.release();

	vw.release();
    return 0;
}
