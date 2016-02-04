#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
using namespace std;
using namespace cv;
string path="F:/SourceCode/GitWorking/Face/Face/data";

CascadeClassifier face_cc;

int tic = 0;

void detect(Mat img){
    vector<Rect> faces;
    vector<int> rejLevel;
    vector<double> levelW;
    Mat grayimg;
    cvtColor(img, grayimg, CV_RGB2GRAY);
    equalizeHist(grayimg, grayimg);
    int minl = min(img.rows, img.cols);
    face_cc.detectMultiScale(grayimg, faces, rejLevel, levelW, 1.1, 3, 0, Size(), Size(), true);
    //face_cc.detectMultiScale(grayimg, faces, 1.1);
    for ( int i = 0; i < faces.size(); i++ )
    {
		if ( rejLevel[i] < 0.3 )
		{
			continue;
		}
        if ( levelW[i] < 1.9 )
        {
            continue;
        }
        stringstream text1, text2;
        text1 << "rejLevel:" << rejLevel[ i ];
        text2 << "levelW:" << levelW[ i ];
        string ttt = text1.str();
        rectangle(img, faces[ i ], Scalar(255, 255, 0), 2, 8, 0);
        putText(img, ttt, cvPoint(faces[ i ].x, faces[ i ].y - 3), 1, 1, Scalar(0,255,255));
        ttt = text2.str();
        putText(img, ttt, cvPoint(faces[ i ].x, faces[ i ].y + 12), 1, 1, Scalar(255, 0, 255));
    }
	namedWindow("IMG",WINDOW_NORMAL);
    imshow("IMG", img);
    //waitKey(0);
}

int main(){
    if ( !face_cc.load(path+"/lbpcascade_frontalface.xml") )
    {
        cout << "load error!\n";
        return -1;
    }
	Mat frame;
	CvCapture* capture = cvCaptureFromCAM(0);

	if( capture)	
	{
		while( true )
		{
			frame = cvQueryFrame( capture );
			if( !frame.empty() )
			{ 
				detect(frame);
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
