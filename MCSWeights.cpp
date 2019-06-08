#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\imgproc\imgproc_c.h"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <vector> 

using namespace cv;
using namespace std;


int thresh=100;
int max_thresh = 255;
RNG rng(12345);

Mat src,srcgray,res,YCrCb,hsv;

vector<Mat> spl;

#define RED Scalar(0,0,255)
#define GREEN Scalar(0,255,0)


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


void main(void)
{
	int in=0,exit=0;

	while(!exit)
	{
		
		printf("\nInput num 0,1,2,3,4,5,6 to exit 100:");
		scanf_s("%d",&in);
		char* resWindow="Output Window";
		char* srcWindow="Input Window";


		switch(in)
		{
		case 13://HaarCascade Pedestrian Detector
			{
				//VideoCapture mypic(0);
				VideoCapture mypic("vp\\Stereo1.avi");

				bool found;
				//vector<cv::Rect> Faces;
				vector<cv::Rect> Bodies;
				//vector<cv::Rect> LBodies;
				//vector<cv::Rect> UBodies;
				//vector<cv::Rect> Tears;

				//CascadeClassifier myfacedetector;
				//CascadeClassifier myeyedetector;
				//CascadeClassifier mymouthdetector;
				//CascadeClassifier myteardetector;
				
				CascadeClassifier myfullbodydetector;
				//CascadeClassifier myUpperbodydetector;
				//CascadeClassifier myLowerbodydetector;

				//found = myfacedetector.load( "haarcascade_frontalface_alt2.xml" );
				//found = myeyedetector.load("haarcascade_eye.xml");
				//found = mymouthdetector.load("haarcascade_mcs_mouth.xml");
				//found = myteardetector.load("haarcascade_mcs_leftear.xml");
				//
				found = myfullbodydetector.load("haarcascade_fullbody.xml");
				if (found != true)
				{
					printf("Error Openning File...");
					break;
				}

				//found = myUpperbodydetector.load("haarcascade_upperbody.xml");
				//if (found != true)
				//{
				//	printf("Error Openning File...");
				//	break;
				//}

				//found = myLowerbodydetector.load("haarcascade_lowerbody.xml");
				//if (found != true)
				//{
				//	printf("Error Openning File...");
				//	break;
				//}

				for (int i = 0; i < 10; i++)
				{
					char str[20];
					sprintf_s(str,"vp\\body\\%d.jpg",i+1);
					src = imread(str, 1);
					if (src.empty())
					{
						printf("Error Openning File...");
						break;
					}
					cvtColor(src, srcgray, CV_BGR2GRAY);
					equalizeHist(srcgray, srcgray);

					//myfullbodydetector.detectMultiScale(srcgray, Bodies, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
					myfullbodydetector.detectMultiScale(srcgray, Bodies);
					for (int i = 0; i < Bodies.size(); i++)
					{
						rectangle(src, Bodies[i], Scalar(255, 0, 0), 2);
					}

					//myUpperbodydetector.detectMultiScale(srcgray, UBodies);
					//for (int i = 0; i < UBodies.size(); i++)
					//{
					//	rectangle(src, UBodies[i], Scalar(0, 255, 255), 2);
					//}

					//myLowerbodydetector.detectMultiScale(srcgray, LBodies);
					//for (int i = 0; i < LBodies.size(); i++)
					//{
					//	rectangle(src, LBodies[i], Scalar(0, 0, 255), 2);
					//}

					//myfacedetector.detectMultiScale(srcgray, Faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(5, 5));
					//for (int i = 0; i < Faces.size(); i++)
					//{
					//	rectangle(src, Faces[i], Scalar(0, 0, 255), 2);
					//}

					imshow(resWindow, src);
					waitKey();
				}





				while(true)
				{
					mypic >> src;


					cvtColor( src, srcgray, CV_BGR2GRAY );
					equalizeHist( srcgray, srcgray );
					//-- Detect faces
					//myfacedetector.detectMultiScale( srcgray, Faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
					myfullbodydetector.detectMultiScale(srcgray, Bodies, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(5, 5));
					//myteardetector.detectMultiScale( srcgray, Tears, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
					/*
					for( int i = 0; i < Faces.size(); i++ )
					{
						rectangle(src,Faces[i],Scalar( 0, 0, 255 ),2);
						
						Point center( Faces[i].x + Faces[i].width*0.5, Faces[i].y + Faces[i].height*0.5 );
						//ellipse( src, center, Size( Faces[i].width*0.5, Faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
						
						
						//Mat faceROI = srcgray( Faces[i] );
						Mat faceROI=Mat(srcgray,Faces[i]);

						std::vector<Rect> eyes;
						//-- In each face, detect eyes
						myeyedetector.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

						for( int j = 0; j < eyes.size(); j++ )
						{
							Point center( Faces[i].x + eyes[j].x + eyes[j].width*0.5, Faces[i].y + eyes[j].y + eyes[j].height*0.5 );
							int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
							circle( src, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
						}


						std::vector<Rect> mouth;
						mymouthdetector.detectMultiScale( faceROI, mouth, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30) );

						for( int j = 0; j < mouth.size(); j++ )
						{
							Point center( Faces[i].x + mouth[j].x + mouth[j].width*0.5, Faces[i].y + mouth[j].y + mouth[j].height*0.5 );
							int radius = cvRound( (mouth[j].width + mouth[j].height)*0.25 );
							if(center.y > Faces[i].y + Faces[i].height / 2)
								circle( src, center, radius, Scalar( 255, 0, 255 ), 4, 8, 0 );
							//rectangle(src,mouth[j],Scalar( 255, 0, 255 ),2);
						}

					}

					for( int i = 0; i < Tears.size(); i++ )
					{
						rectangle(src,Tears[i],Scalar( 255,255, 255 ),2);
					}
					
					for (int i = 0; i < Faces.size(); i++)
					{
						rectangle(src, Faces[i], Scalar(0, 0, 255), 2);
					}
*/
					for (int i = 0; i < Bodies.size(); i++)
					{
						rectangle(src, Bodies[i], Scalar(0, 255, 0), 2);
					}

					imshow(resWindow,src);
					if(waitKey(30)== 27) break;
				}
				break;
			}
		case 14://HaarCascade Car detector
			{
				VideoCapture myvideo("vp\\Stereo1.avi");

				bool found;
				vector<cv::Rect> Cars;


				CascadeClassifier myCardetector;

				found = myCardetector.load("cars.xml");
				if (found != true)
				{
					printf("Error Openning File...");
					break;
				}

				while (true)
				{
					myvideo >> src;


					cvtColor(src, srcgray, CV_BGR2GRAY);
					equalizeHist(srcgray, srcgray);
					myCardetector.detectMultiScale(srcgray, Cars, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(3, 3));

					for (int i = 0; i < Cars.size(); i++)
					{
						rectangle(src, Cars[i], Scalar(0, 0, 255), 2);
					}

					imshow(resWindow, src);
					if (waitKey(30) == 27) break;
				}
				break;
			}
		case 15://SVM Pedestrian Detector
			{
				VideoWriter vw;
				vw.open("PedDetector.avi", CV_FOURCC('M', 'P', 'E', 'G'), 20, Size(640, 480));
				VideoCapture mypic("vp\\set00_V010.avi");

				HOGDescriptor hog;
				hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

				while(true)
				{
					mypic >> src;


					vector<Rect> found, found_filtered;
					size_t i, j;

					// run the detector with default parameters. to get a higher hit-rate
					// (and more false alarms, respectively), decrease the hitThreshold and
					// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
					hog.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);


					for( i = 0; i < found.size(); i++ )
					{
						Rect r = found[i];
						for( j = 0; j < found.size(); j++ )
							if( j != i && (r & found[j]) == r)
								break;
						if( j == found.size() )
							found_filtered.push_back(r);
					}
					for( i = 0; i < found_filtered.size(); i++ )
					{
						Rect r = found_filtered[i];
						// the HOG detector returns slightly larger rectangles than the real objects.
						// so we slightly shrink the rectangles to get a nicer output.
						r.x += cvRound(r.width*0.1);
						r.width = cvRound(r.width*0.8);
						r.y += cvRound(r.height*0.07);
						r.height = cvRound(r.height*0.8);
						rectangle(src, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
					}

					imshow(resWindow,src);
					vw.write(src);
					if(waitKey(30)== 27) break;
				}
				vw.release();
				break;
			}		

		case 100:
			{
				exit=1;
				break;
			}
		}

		destroyAllWindows();
	}
}




