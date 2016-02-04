#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include<iostream>
#include<vector>
using namespace std;
using namespace cv;

string path="F:/SourceCode/GitWorking/Face/Face/data";
const double DESIRED_LEFT_EYE_X = 0.16;     // 控制处理后人脸的多少部分是可见的
const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;         // 应当至少为0.5
const double FACE_ELLIPSE_H = 0.80;         //控制人脸掩码的高度
CascadeClassifier faceDetector;  
//CascadeClassifier  eyeDetector1;
//CascadeClassifier  eyeDetector2;//未初始化不用
CascadeClassifier  eyeDetector1;
CascadeClassifier  eyeDetector2;
CascadeClassifier  eyeDetector3;
/*--------------------------------------目标检测-------------------------------------*/
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
		//仿射变换
		Point2f eyesCenter;
		eyesCenter.x=(leftEye.x+rightEye.x)*0.5f;
		eyesCenter.y=(leftEye.y+rightEye.y)*0.5f;
		cout<<"左眼中心坐标 "<<leftEye.x<<" and "<<leftEye.y<<endl;
		cout<<"右眼中心坐标 "<<rightEye.x<<" and "<<rightEye.y<<endl;
		//获取两个人眼的角度
		double dy=(rightEye.y-leftEye.y);
		double dx=(rightEye.x-leftEye.x);
		double len=sqrt(dx*dx+dy*dy);
		cout<<"dx is "<<dx<<endl;
		cout<<"dy is "<<dy<<endl;
		cout<<"len is "<<len<<endl;
		double angle=atan2(dy,dx)*180.0/CV_PI;
		const double DESIRED_RIGHT_EYE_X=1.0f-0.16;
		//得到我们想要的尺度化大小
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

		//直方图均衡化,左均衡右均衡,全均衡

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
		//平滑图像

		imshow("混合后 ",faceImg);
		Mat filtered=Mat(warped.size(),CV_8U);
		bilateralFilter(faceImg,filtered,0,20.0,2.0);
		imshow("双边滤波后",filtered);
		//椭圆形掩码
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

	if( capture/*.isOpened()*//*capture*/ )	// 摄像头读取文件开关
	{
		while( true )
		{
			frame = cvQueryFrame( capture );	// 摄像头读取文件开关
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
1、采用给出的参数在图像中寻找目标，例如人脸
2、可以使用Haar级联器或者LBP级联器做人脸检测，或者甚至眼睛，鼻子，汽车检测
3、为了使检测更快，输入图像暂时被缩小到'scaledWidth'，因为寻找人脸200的尺度已经足够了。
*/
void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{

	//如果输入的图像不是灰度图像,那么将BRG或者BGRA彩色图像转换为灰度图像
	Mat gray;
	if (img.channels() == 3) {
		cvtColor(img, gray, CV_BGR2GRAY);
	}
	else if (img.channels() == 4) {
		cvtColor(img, gray, CV_BGRA2GRAY);
	}
	else {
		// 直接使用输入图像，既然它已经是灰度图像
		gray = img;
	}

	// 可能的缩小图像，是检索更快
	Mat inputImg;

	float scale = img.cols / (float)scaledWidth;
	if (img.cols > scaledWidth) {
		// 缩小图像并保持同样的宽高比
		int scaledHeight = cvRound(img.rows / scale);
		resize(gray, inputImg, Size(scaledWidth, scaledHeight));
	}
	else {
		// 直接使用输入图像，既然它已经小了
		inputImg = gray;
	}

	//标准化亮度和对比度来改善暗的图像
	Mat equalizedImg;
	equalizeHist(inputImg, equalizedImg);

	// 在小的灰色图像中检索目标
	cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

	// 如果图像在检测之前暂时的被缩小了，则放大结果图像
	if (img.cols > scaledWidth) {
		for (int i = 0; i < (int)objects.size(); i++ ) {
			objects[i].x = cvRound(objects[i].x * scale);
			objects[i].y = cvRound(objects[i].y * scale);
			objects[i].width = cvRound(objects[i].width * scale);
			objects[i].height = cvRound(objects[i].height * scale);
		}
	}

	//确保目标全部在图像内部，以防它在边界上 
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

	// 返回检测到的人脸矩形，存储在objects中
}

/*
1、仅寻找图像中的单个目标，例如最大的人脸，存储结果到largestObject
2、可以使用Haar级联器或者LBP级联器做人脸检测，或者甚至眼睛，鼻子，汽车检测
3、为了使检测更快，输入图像暂时被缩小到'scaledWidth'，因为寻找人脸200的尺度已经足够了。
4、注释：detectLargestObject()要比 detectManyObjects()快。
*/
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth,bool &hasFace)
{
	//仅寻找一个目标 (图像中最大的).
	int flags = CV_HAAR_FIND_BIGGEST_OBJECT;//CV_HAAR_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
	// 最小的目标大小.
	Size minFeatureSize = Size(20, 20);
	// 寻找细节,尺度因子,必须比1大
	float searchScaleFactor = 1.1f;
	// 多少检测结果应当被滤掉，这依赖于你的检测系统是多坏,如果minNeighbors=2 ，大量的good or bad 被检测到。如果
	// minNeighbors=6，意味着只good检测结果，但是一些将漏掉。即可靠性 VS  检测人脸数量
	int minNeighbors = 4;

	// 执行目标或者人脸检测，仅寻找一个目标（图像中最大的）
	vector<Rect> objects;
	detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
	if (objects.size() > 0) {
		// 返回仅检测到的目标
		largestObject = (Rect)objects.at(0);
		hasFace=true;
	}
	else {
		// 返回一个无效的矩阵
		largestObject = Rect(-1,-1,-1,-1);
		hasFace=false;
	}
}

void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth,bool &hasFace)
{
	// 寻找图像中的许多目标
	int flags = CV_HAAR_SCALE_IMAGE;

	// 最小的目标大小.
	Size minFeatureSize = Size(20, 20);
	//  寻找细节,尺度因子,必须比1大
	float searchScaleFactor = 1.1f;
	// 多少检测结果应当被滤掉，这依赖于你的检测系统是多坏,如果minNeighbors=2 ，大量的good or bad 被检测到。如果
	// minNeighbors=6，意味着只good检测结果，但是一些将漏掉。即可靠性 VS  检测人脸数量
	int minNeighbors = 4;

	// 执行目标或者人脸检测，寻找图像中的许多目标
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
1、在给出的人脸图像中寻找双眼，返回左眼和右眼的中心，如果当找不到人眼时,或者设置为Point(-1,-1)
2、注意如果你想用两个不同的级联器寻找人眼，你可以传递第二个人眼检测器，例如如果你使用的一个常规人眼检测器和带眼镜的人眼检测器一样好，或者左眼检测器和右眼检测器一样好，
或者如果你不想第二个检测器，仅传一个未初始化级联检测器。
3、如果需要的话，也可以存储检测到的左眼和右眼的区域
*/
void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, CascadeClassifier &eyeCascade3,Point &leftEye, Point &rightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
	//跳过人脸边界，因为它们经常是头发和耳朵，这不是我们关心的 
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
	//haarcascade_eye.xml检测器在由下面确定的人脸区域内搜索最优。
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
	int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // 右眼的开始区域

	Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
	Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));
	Rect leftEyeRect, rightEyeRect;

	// 如果需要的话，然后搜索到的窗口给调用者
	if (searchedLeftEye)
		*searchedLeftEye = Rect(leftX, topY, widthX, heightY);
	if (searchedRightEye)
		*searchedRightEye = Rect(rightX, topY, widthX, heightY);

	// 寻找左区域，然后右区域使用第一个人眼检测器
	bool isFace;
	detectLargestObject(topLeftOfFace, eyeCascade1, leftEyeRect, topLeftOfFace.cols,isFace);
	detectLargestObject(topRightOfFace, eyeCascade2, rightEyeRect, topRightOfFace.cols,isFace);

	// 如果人眼没有检测到，尝试另外一个不同的级联检测器
	if (leftEyeRect.width <= 0 && !eyeCascade3.empty()) {
		detectLargestObject(topLeftOfFace, eyeCascade3, leftEyeRect, topLeftOfFace.cols,isFace);
		//if (leftEyeRect.width > 0)
		//    cout << "2nd eye detector LEFT SUCCESS" << endl;
		//else
		//    cout << "2nd eye detector LEFT failed" << endl;
	}
	//else
	//    cout << "1st eye detector LEFT SUCCESS" << endl;

	// 如果人眼没有检测到，尝试另外一个不同的级联检测器
	if (rightEyeRect.width <= 0 && !eyeCascade3.empty()) {
		detectLargestObject(topRightOfFace, eyeCascade3, rightEyeRect, topRightOfFace.cols,isFace);
		//if (rightEyeRect.width > 0)
		//    cout << "2nd eye detector RIGHT SUCCESS" << endl;
		//else
		//    cout << "2nd eye detector RIGHT failed" << endl;
	}
	//else
	//    cout << "1st eye detector RIGHT SUCCESS" << endl;

	if (leftEyeRect.width > 0) {   // 检查眼是否被检测到
		leftEyeRect.x += leftX;    //矫正左眼矩形，因为人脸边界被去除掉了 
		leftEyeRect.y += topY;
		leftEye = Point(leftEyeRect.x + leftEyeRect.width/2, leftEyeRect.y + leftEyeRect.height/2);
	}
	else {
		leftEye = Point(-1, -1);    // 返回一个无效的点
	}

	if (rightEyeRect.width > 0) { //检查眼是否被检测到
		rightEyeRect.x += rightX; // 矫正左眼矩形，因为它从图像的右边界开始
		rightEyeRect.y += topY;  // 矫正右眼矩形，因为人脸边界被去除掉了
		rightEye = Point(rightEyeRect.x + rightEyeRect.width/2, rightEyeRect.y + rightEyeRect.height/2);
	}
	else {
		rightEye = Point(-1, -1);    // 返回一个无效的点
	}
}


