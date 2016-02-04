#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;
string path="F:/SourceCode/GitWorking/Face/Face/data";

void process(CascadeClassifier &cascade,Mat img_color)
{
	Mat img_gray;
	const float scale_factor(1.2f);
	const int min_neighbors(3);

	if (img_color.channels() == 3) {
		cvtColor(img_color, img_gray, CV_BGR2GRAY);
	}
	else if (img_color.channels() == 4) {
		cvtColor(img_color, img_gray, CV_BGRA2GRAY);
	}
	else {
		// 直接使用输入图像，既然它已经是灰度图像
		img_gray = img_color;
	}
	equalizeHist(img_gray, img_gray);
	vector<Rect> objs;
	vector<int> reject_levels;
	vector<double> level_weights;
	cascade.detectMultiScale(img_gray, objs, reject_levels, level_weights, scale_factor, min_neighbors, 0, Size(), Size(), true);
	for (int n = 0; n < objs.size(); n++) {
		rectangle(img_color, objs[n], Scalar(255,0,0), 8);
		putText(img_color, std::to_string((long double)level_weights[n]),
			Point(objs[n].x, objs[n].y), 1, 1, Scalar(0,0,255));
	}
	imshow("VJ Face Detector", img_color);
	waitKey(0);
}

int main(int argc, char **argv) {

	CascadeClassifier cascade;
	if (!cascade.load(path+"/lbpcascade_frontalface.xml")) 
	{
		cerr<<"error:couldn't load face detector (";  
		cerr<<"lbpcascade_frontalface.xml)!"<<endl;  
		exit(1); 
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
				process(cascade,frame);
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