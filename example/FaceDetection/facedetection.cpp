#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include<iostream>
#include<vector>
using namespace std;
using namespace cv;

string path="F:/SourceCode/GitWorking/Face/Face/data";
const double DESIRED_LEFT_EYE_X = 0.16;     // ���ƴ���������Ķ��ٲ����ǿɼ���
const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;         // Ӧ������Ϊ0.5
const double FACE_ELLIPSE_H = 0.80;         //������������ĸ߶�
CascadeClassifier faceDetector;  
//CascadeClassifier  eyeDetector1;
//CascadeClassifier  eyeDetector2;//δ��ʼ������
CascadeClassifier  eyeDetector1;
CascadeClassifier  eyeDetector2;
CascadeClassifier  eyeDetector3;
/*--------------------------------------Ŀ����-------------------------------------*/
void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors);
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth,bool &hasFace);
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth,bool &hasFace);
/*------------------------------------- end------------------------------------------*/

void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2,CascadeClassifier &eyeCascade3, Point &leftEye, Point &rightEye, Rect *searchedLeftEye, Rect *searchedRightEye);
Mat getPreprocessedFace(Mat &srcImg, int desiredFaceWidth, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye, Rect *searchedLeftEye, Rect *searchedRightEye);

void initDetector()
{
	try{  
		//faceDetector.load(path+"/haarcascade_frontalface_alt.xml");
		faceDetector.load(path+"/haarcascade_frontalface_alt_tree.xml");
		//faceDetector.load(path+"/lbpcascade_frontalface.xml");  
		//eyeDetector1.load(path+"/haarcascade_eye.xml");
		//eyeDetector2.load(path+"/haarcascade_eye_tree_eyeglasses.xml");
		eyeDetector1.load(path+"/haarcascade_mcs_lefteye.xml");
		eyeDetector2.load(path+"/haarcascade_mcs_righteye.xml");


	}catch (cv::Exception e){}  
	if(faceDetector.empty())  
	{  
		cerr<<"error:couldn't load face detector (";  
		cerr<<"lbpcascade_frontalface.xml)!"<<endl;  
		exit(1);  
	}  
}

void process(Mat image)
{
	Rect largestObject;
	const int scaledWidth=320;
	bool hasFace=false;
	detectLargestObject(image,faceDetector,largestObject,scaledWidth,hasFace);
	if (hasFace)
	{
		Mat img_rect(image,largestObject);
		Point leftEye,rightEye;
		Rect searchedLeftEye,searchedRightEye;
		detectBothEyes(img_rect,eyeDetector1,eyeDetector2,eyeDetector3,leftEye,rightEye,&searchedLeftEye,&searchedRightEye);
		//����任
		Point2f eyesCenter;
		eyesCenter.x=(leftEye.x+rightEye.x)*0.5f;
		eyesCenter.y=(leftEye.y+rightEye.y)*0.5f;
		cout<<"������������ "<<leftEye.x<<" and "<<leftEye.y<<endl;
		cout<<"������������ "<<rightEye.x<<" and "<<rightEye.y<<endl;
		//��ȡ�������۵ĽǶ�
		double dy=(rightEye.y-leftEye.y);
		double dx=(rightEye.x-leftEye.x);
		double len=sqrt(dx*dx+dy*dy);
		cout<<"dx is "<<dx<<endl;
		cout<<"dy is "<<dy<<endl;
		cout<<"len is "<<len<<endl;
		double angle=atan2(dy,dx)*180.0/CV_PI;
		const double DESIRED_RIGHT_EYE_X=1.0f-0.16;
		//�õ�������Ҫ�ĳ߶Ȼ���С
		const int DESIRED_FACE_WIDTH=100;
		const int DESIRED_FACE_HEIGHT=100;
		double desiredLen=(DESIRED_RIGHT_EYE_X-0.16);
		cout<<"desiredlen is "<<desiredLen<<endl;
		double scale=desiredLen*DESIRED_FACE_WIDTH/len;
		cout<<"the scale is "<<scale<<endl;
		Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
		double ex=DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
		double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y-eyesCenter.y;
		rot_mat.at<double>(0, 2) += ex;
		rot_mat.at<double>(1, 2) += ey;
		Mat warped = Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH,CV_8U, Scalar(128));
		warpAffine(img_rect, warped, rot_mat, warped.size());

		//ֱ��ͼ���⻯,������Ҿ���,ȫ����

		Mat faceImg_temp,faceImg(warped.rows,warped.cols,warped.type());
		warped.copyTo(faceImg_temp);
		cvtColor(faceImg_temp,faceImg,CV_BGR2GRAY);
		imshow("faceImg",faceImg);
		int w=faceImg.cols;
		int h=faceImg.rows;
		Mat wholeFace;
		equalizeHist(faceImg,wholeFace);
		int midX=w/2;
		Mat leftSide=faceImg(Rect(0,0,midX,h));
		Mat rightSide=faceImg(Rect(midX,0,w-midX,h));
		equalizeHist(leftSide,leftSide);
		equalizeHist(rightSide,rightSide);


		for(int y=0;y<h;y++)
		{
			for(int x=0;x<w;x++)
			{
				int v;
				if(x<w/4)
				{
					v=leftSide.at<uchar>(y,x);
				}else if(x<w*2/4)
				{
					int lv=leftSide.at<uchar>(y,x);
					int wv=wholeFace.at<uchar>(y,x);
					float f=(x-w*1/4)/(float)(w/4);
					v=cvRound((1.0f-f)*lv+(f)*wv);
				}else if(x<w*3/4)
				{
					int rv=rightSide.at<uchar>(y,x-midX);
					int wv=wholeFace.at<uchar>(y,x);
					float f=(x-w*2/4)/(float)(w/4);
					v=cvRound((1.0f-f)*wv+(f)*rv);

				}else 
				{
					v=rightSide.at<uchar>(y,x-midX);
				}
				faceImg.at<uchar>(y,x)=v;
			}
		}
		//ƽ��ͼ��

		imshow("��Ϻ� ",faceImg);
		Mat filtered=Mat(warped.size(),CV_8U);
		bilateralFilter(faceImg,filtered,0,20.0,2.0);
		imshow("˫���˲���",filtered);
		//��Բ������
		Mat mask=Mat(warped.size(),CV_8UC1,Scalar(255));
		double dw=DESIRED_FACE_WIDTH;
		double dh=DESIRED_FACE_HEIGHT;
		Point faceCenter=Point(cvRound(dw*0.5),cvRound(dh*0.4));
		Size size=Size(cvRound(dw*0.5),cvRound(dh*0.8));
		ellipse(mask,faceCenter,size,0,0,360,Scalar(0),CV_FILLED);
		filtered.setTo(Scalar(128),mask);

		imshow("filtered",filtered);


		imshow("warped",warped);

		rectangle(image,Point(largestObject.x,largestObject.y),Point(largestObject.x+largestObject.width,largestObject.y+largestObject.height),Scalar(0,0,255),2,8);
		rectangle(img_rect,Point(searchedLeftEye.x,searchedLeftEye.y),Point(searchedLeftEye.x+searchedLeftEye.width,searchedLeftEye.y+searchedLeftEye.height),Scalar(0,255,0),2,8);
		rectangle(img_rect,Point(searchedRightEye.x,searchedRightEye.y),Point(searchedRightEye.x+searchedRightEye.width,searchedRightEye.y+searchedRightEye.height),Scalar(0,255,0),2,8);


		//getPreprocessedFace
		imshow("img_rect",img_rect);
		imwrite("img_rect.jpg",img_rect);
		imshow("img",image);
	}
	else
	{
		imshow("img",image);
	}
}
int main(int argc,char **argv)
{
	initDetector();
	//-- 2. Read the video stream
	Mat frame;
	CvCapture* capture = cvCaptureFromCAM(0);
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
				//detectAndDisplay( frame ); 
				process(frame);
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
}

/*
1�����ø����Ĳ�����ͼ����Ѱ��Ŀ�꣬��������
2������ʹ��Haar����������LBP��������������⣬���������۾������ӣ��������
3��Ϊ��ʹ�����죬����ͼ����ʱ����С��'scaledWidth'����ΪѰ������200�ĳ߶��Ѿ��㹻�ˡ�
*/
void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{

	//��������ͼ���ǻҶ�ͼ��,��ô��BRG����BGRA��ɫͼ��ת��Ϊ�Ҷ�ͼ��
	Mat gray;
	if (img.channels() == 3) {
		cvtColor(img, gray, CV_BGR2GRAY);
	}
	else if (img.channels() == 4) {
		cvtColor(img, gray, CV_BGRA2GRAY);
	}
	else {
		// ֱ��ʹ������ͼ�񣬼�Ȼ���Ѿ��ǻҶ�ͼ��
		gray = img;
	}

	// ���ܵ���Сͼ���Ǽ�������
	Mat inputImg;

	float scale = img.cols / (float)scaledWidth;
	if (img.cols > scaledWidth) {
		// ��Сͼ�񲢱���ͬ���Ŀ�߱�
		int scaledHeight = cvRound(img.rows / scale);
		resize(gray, inputImg, Size(scaledWidth, scaledHeight));
	}
	else {
		// ֱ��ʹ������ͼ�񣬼�Ȼ���Ѿ�С��
		inputImg = gray;
	}

	//��׼�����ȺͶԱȶ������ư���ͼ��
	Mat equalizedImg;
	equalizeHist(inputImg, equalizedImg);

	// ��С�Ļ�ɫͼ���м���Ŀ��
	cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

	// ���ͼ���ڼ��֮ǰ��ʱ�ı���С�ˣ���Ŵ���ͼ��
	if (img.cols > scaledWidth) {
		for (int i = 0; i < (int)objects.size(); i++ ) {
			objects[i].x = cvRound(objects[i].x * scale);
			objects[i].y = cvRound(objects[i].y * scale);
			objects[i].width = cvRound(objects[i].width * scale);
			objects[i].height = cvRound(objects[i].height * scale);
		}
	}

	//ȷ��Ŀ��ȫ����ͼ���ڲ����Է����ڱ߽��� 
	for (int i = 0; i < (int)objects.size(); i++ ) {
		if (objects[i].x < 0)
			objects[i].x = 0;
		if (objects[i].y < 0)
			objects[i].y = 0;
		if (objects[i].x + objects[i].width > img.cols)
			objects[i].x = img.cols - objects[i].width;
		if (objects[i].y + objects[i].height > img.rows)
			objects[i].y = img.rows - objects[i].height;
	}

	// ���ؼ�⵽���������Σ��洢��objects��
}

/*
1����Ѱ��ͼ���еĵ���Ŀ�꣬���������������洢�����largestObject
2������ʹ��Haar����������LBP��������������⣬���������۾������ӣ��������
3��Ϊ��ʹ�����죬����ͼ����ʱ����С��'scaledWidth'����ΪѰ������200�ĳ߶��Ѿ��㹻�ˡ�
4��ע�ͣ�detectLargestObject()Ҫ�� detectManyObjects()�졣
*/
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth,bool &hasFace)
{
	//��Ѱ��һ��Ŀ�� (ͼ��������).
	int flags = CV_HAAR_FIND_BIGGEST_OBJECT;//CV_HAAR_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
	// ��С��Ŀ���С.
	Size minFeatureSize = Size(20, 20);
	// Ѱ��ϸ��,�߶�����,�����1��
	float searchScaleFactor = 1.1f;
	// ���ټ����Ӧ�����˵�������������ļ��ϵͳ�Ƕ໵,���minNeighbors=2 ��������good or bad ����⵽�����
	// minNeighbors=6����ζ��ֻgood�����������һЩ��©�������ɿ��� VS  �����������
	int minNeighbors = 4;

	// ִ��Ŀ�����������⣬��Ѱ��һ��Ŀ�꣨ͼ�������ģ�
	vector<Rect> objects;
	detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
	if (objects.size() > 0) {
		// ���ؽ���⵽��Ŀ��
		largestObject = (Rect)objects.at(0);
		hasFace=true;
	}
	else {
		// ����һ����Ч�ľ���
		largestObject = Rect(-1,-1,-1,-1);
		hasFace=false;
	}
}

void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth,bool &hasFace)
{
	// Ѱ��ͼ���е����Ŀ��
	int flags = CV_HAAR_SCALE_IMAGE;

	// ��С��Ŀ���С.
	Size minFeatureSize = Size(20, 20);
	//  Ѱ��ϸ��,�߶�����,�����1��
	float searchScaleFactor = 1.1f;
	// ���ټ����Ӧ�����˵�������������ļ��ϵͳ�Ƕ໵,���minNeighbors=2 ��������good or bad ����⵽�����
	// minNeighbors=6����ζ��ֻgood�����������һЩ��©�������ɿ��� VS  �����������
	int minNeighbors = 4;

	// ִ��Ŀ�����������⣬Ѱ��ͼ���е����Ŀ��
	detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
	if (objects.size() > 0) 
	{
		hasFace=true;
	}
	else 
	{
		hasFace=false;
	}
}
/*
1���ڸ���������ͼ����Ѱ��˫�ۣ��������ۺ����۵����ģ�������Ҳ�������ʱ,��������ΪPoint(-1,-1)
2��ע�����������������ͬ�ļ�����Ѱ�����ۣ�����Դ��ݵڶ������ۼ���������������ʹ�õ�һ���������ۼ�����ʹ��۾������ۼ����һ���ã��������ۼ���������ۼ����һ���ã�
��������㲻��ڶ��������������һ��δ��ʼ�������������
3�������Ҫ�Ļ���Ҳ���Դ洢��⵽�����ۺ����۵�����
*/
void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, CascadeClassifier &eyeCascade3,Point &leftEye, Point &rightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
	//���������߽磬��Ϊ���Ǿ�����ͷ���Ͷ��䣬�ⲻ�����ǹ��ĵ� 
	/*
	// For "2splits.xml": Finds both eyes in roughly 60% of detected faces, also detects closed eyes.
	const float EYE_SX = 0.12f;
	const float EYE_SY = 0.17f;
	const float EYE_SW = 0.37f;
	const float EYE_SH = 0.36f;
	*/

	// For mcs.xml: Finds both eyes in roughly 80% of detected faces, also detects closed eyes.
	const float EYE_SX = 0.10f;
	const float EYE_SY = 0.19f;
	const float EYE_SW = 0.40f;
	const float EYE_SH = 0.36f;


	// For default eye.xml or eyeglasses.xml: Finds both eyes in roughly 40% of detected faces, but does not detect closed eyes.
	//haarcascade_eye.xml�������������ȷ���������������������š�
	/*
	const float EYE_SX = 0.16f;//x
	const float EYE_SY = 0.26f;//y
	const float EYE_SW = 0.30f;//width
	const float EYE_SH = 0.28f;//height
	*/
	int leftX = cvRound(face.cols * EYE_SX);
	int topY = cvRound(face.rows * EYE_SY);
	int widthX = cvRound(face.cols * EYE_SW);
	int heightY = cvRound(face.rows * EYE_SH);
	int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // ���۵Ŀ�ʼ����

	Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
	Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));
	Rect leftEyeRect, rightEyeRect;

	// �����Ҫ�Ļ���Ȼ���������Ĵ��ڸ�������
	if (searchedLeftEye)
		*searchedLeftEye = Rect(leftX, topY, widthX, heightY);
	if (searchedRightEye)
		*searchedRightEye = Rect(rightX, topY, widthX, heightY);

	// Ѱ��������Ȼ��������ʹ�õ�һ�����ۼ����
	bool isFace;
	detectLargestObject(topLeftOfFace, eyeCascade1, leftEyeRect, topLeftOfFace.cols,isFace);
	detectLargestObject(topRightOfFace, eyeCascade2, rightEyeRect, topRightOfFace.cols,isFace);

	// �������û�м�⵽����������һ����ͬ�ļ��������
	if (leftEyeRect.width <= 0 && !eyeCascade3.empty()) {
		detectLargestObject(topLeftOfFace, eyeCascade3, leftEyeRect, topLeftOfFace.cols,isFace);
		//if (leftEyeRect.width > 0)
		//    cout << "2nd eye detector LEFT SUCCESS" << endl;
		//else
		//    cout << "2nd eye detector LEFT failed" << endl;
	}
	//else
	//    cout << "1st eye detector LEFT SUCCESS" << endl;

	// �������û�м�⵽����������һ����ͬ�ļ��������
	if (rightEyeRect.width <= 0 && !eyeCascade3.empty()) {
		detectLargestObject(topRightOfFace, eyeCascade3, rightEyeRect, topRightOfFace.cols,isFace);
		//if (rightEyeRect.width > 0)
		//    cout << "2nd eye detector RIGHT SUCCESS" << endl;
		//else
		//    cout << "2nd eye detector RIGHT failed" << endl;
	}
	//else
	//    cout << "1st eye detector RIGHT SUCCESS" << endl;

	if (leftEyeRect.width > 0) {   // ������Ƿ񱻼�⵽
		leftEyeRect.x += leftX;    //�������۾��Σ���Ϊ�����߽类ȥ������ 
		leftEyeRect.y += topY;
		leftEye = Point(leftEyeRect.x + leftEyeRect.width/2, leftEyeRect.y + leftEyeRect.height/2);
	}
	else {
		leftEye = Point(-1, -1);    // ����һ����Ч�ĵ�
	}

	if (rightEyeRect.width > 0) { //������Ƿ񱻼�⵽
		rightEyeRect.x += rightX; // �������۾��Σ���Ϊ����ͼ����ұ߽翪ʼ
		rightEyeRect.y += topY;  // �������۾��Σ���Ϊ�����߽类ȥ������
		rightEye = Point(rightEyeRect.x + rightEyeRect.width/2, rightEyeRect.y + rightEyeRect.height/2);
	}
	else {
		rightEye = Point(-1, -1);    // ����һ����Ч�ĵ�
	}
}


