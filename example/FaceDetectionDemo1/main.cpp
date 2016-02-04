#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
string path="F:/SourceCode/GitWorking/Face/Face/data";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

int main( int argc, char** argv )
{
	Mat frame;

	//-- 1. Load the cascades
	if( !face_cascade.load( path+"/haarcascade_frontalface_alt.xml" ) )
	{ 
		printf("--(!)Error loading\n"); 
		return -1; 
	};
	if( !eyes_cascade.load( path+"/haarcascade_eye_tree_eyeglasses.xml" ) )
	{ 
		printf("--(!)Error loading\n");
		return -1; 
	};

	//-- 2. Read the video stream
 	CvCapture* capture = cvCaptureFromCAM(0);	// ����ͷ��ȡ�ļ�����
	//VideoCapture capture("Sample.avi");

	if( capture/*.isOpened()*//*capture*/ )	// ����ͷ��ȡ�ļ�����
	{
		while( true )
		{
			frame = cvQueryFrame( capture );	// ����ͷ��ȡ�ļ�����
			//capture >> frame;

			//-- 3. Apply the classifier to the frame
			if( !frame.empty() )
			{ 
				detectAndDisplay( frame ); 
				//imshow("1",frame);
			}
			else
			{ 
				printf(" --(!) No captured frame -- Break!"); 
				break; 
			}

			int c = waitKey(10);
			if( (char)c == 'c' ) { break; } 

		}
	}
	return 0;
}

/**
* @function detectAndDisplay
*/
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	for( size_t i = 0; i < faces.size(); i++ )
	{
		Point center( int(faces[i].x + faces[i].width*0.5), int(faces[i].y + faces[i].height*0.5) );
		ellipse( frame, center, Size( int(faces[i].width*0.5), int(faces[i].height*0.5)), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
		rectangle(frame,faces[i],Scalar( 255, 0, 0 ), 3, 8, 0 );
		Mat faceROI = frame_gray( faces[i] );
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

		for( size_t j = 0; j < eyes.size(); j++ )
		{
			Point center( int(faces[i].x + eyes[j].x + eyes[j].width*0.5), int(faces[i].y + eyes[j].y + eyes[j].height*0.5) ); 
			int radius = cvRound( (eyes[j].width + eyes[i].height)*0.25 );
			circle( frame, center, radius, Scalar( 0, 0, 255 ), 3, 8, 0 );
		}
	} 
	//-- Show what you got
	imshow( window_name, frame );
}