#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "MatrixReaderWriter.h"

using namespace cv;
using namespace std;

/*
The project for the third lab visualizes a 3D point cloud. The task is to improve the original projects:



- Add mouse-controlling to the project, similarly to Meshlab. 
The object is rotated if the left mouse button is pressed and the mouse position is changed (dragged). 
Vertical and horizontal movement changes the angles u and v, respectively. (3%)

- Animate the object, independently to the user's control. 
Rotate the objects with a constant rotation w.r.t time, all the three principal axes should be used for the rotation (2%), 
periodically scale the object between scale factors 0.5 and 2.0 (3%)
*/


float v = 1.0;
float u = 0.5;
float rad = 100.0;

int cnt = 0;

MatrixReaderWriter* mrw;
Mat resImg;


void drawPoints(MatrixReaderWriter* mrw, float u, float v, float rad, Mat& resImg) {
	int NUM = mrw->rowNum;



	Mat C(3, 3, CV_32F);
	Mat R(3, 3, CV_32F);
	Mat T(3, 1, CV_32F);

	float tx = cos(u) * sin(v);
	float ty = sin(u) * sin(v);
	float tz = cos(v);



	//Intrincic parameters

	C.at<float>(0, 0) = 3000.0f;
	C.at<float>(0, 1) = 0.0f;
	C.at<float>(0, 2) = 400.0f;

	C.at<float>(1, 0) = 0.0f;
	C.at<float>(1, 1) = 3000.0f;
	C.at<float>(1, 2) = 300.0f;

	C.at<float>(2, 0) = 0.0f;
	C.at<float>(2, 1) = 0.0f;
	C.at<float>(2, 2) = 1.0f;

	T.at<float>(0, 0) = rad * tx;
	T.at<float>(1, 0) = rad * ty;
	T.at<float>(2, 0) = rad * tz;


	//Mirror?
	int HowManyPi = (int)floor(v / 3.1415);



	//Axes:
	Point3f Z(-1.0 * tx, -1.0 * ty, -1.0 * tz);
	Point3f X(sin(u) * sin(v), -cos(u) * sin(v), 0.0f);
	if (HowManyPi % 2)
		X = (1.0 / sqrt(X.x * X.x + X.y * X.y + X.z * X.z)) * X;
	else
		X = (-1.0 / sqrt(X.x * X.x + X.y * X.y + X.z * X.z)) * X;

	Point3f up = X.cross(Z);  //Axis Y

/*
	printf("%f\n",X.x*X.x+X.y*X.y+X.z*X.z);
	printf("%f\n",up.x*up.x+up.y*up.y+up.z*up.z);
	printf("%f\n",Z.x*Z.x+Z.y*Z.y+Z.z*Z.z);

	printf("(%f,%f)\n",u,v);
*/


	R.at<float>(2, 0) = Z.x;
	R.at<float>(2, 1) = Z.y;
	R.at<float>(2, 2) = Z.z;

	R.at<float>(1, 0) = up.x;
	R.at<float>(1, 1) = up.y;
	R.at<float>(1, 2) = up.z;

	R.at<float>(0, 0) = X.x;
	R.at<float>(0, 1) = X.y;
	R.at<float>(0, 2) = X.z;




	for (int i = 0;i < NUM;i++) {
		Mat vec(3, 1, CV_32F);
		vec.at<float>(0, 0) = mrw->data[3 * i];
		vec.at<float>(1, 0) = mrw->data[3 * i + 1];
		vec.at<float>(2, 0) = mrw->data[3 * i + 2];

		int red = 255;
		int green = 255;
		int blue = 255;

		Mat trVec = R * (vec - T);
		trVec = C * trVec;
		trVec = trVec / trVec.at<float>(2, 0);
		//		printf("(%d,%d)",(int)trVec.at<float>(0,0),(int)trVec.at<float>(1,0));

		circle(resImg, Point((int)trVec.at<float>(0, 0), (int)trVec.at<float>(1, 0)), 2.0, Scalar(blue, green, red), 2, 8);
	}
}


bool drag = false;
Point2d mouse;

void MouseCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_MOUSEWHEEL)
	{
		printf("Hello%d\n", cnt++);
		if (getMouseWheelDelta(flags) > 0) {
			rad *= (float)1.1;
		}
		else if (getMouseWheelDelta(flags) < 0) {
			rad /= (float)1.1;
		}

		printf("Hello%d\n", cnt++);

	}
	else if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Drag Start" << endl;
		mouse = Point2d(x, y);
		drag = true;
	}
	else if (event == EVENT_LBUTTONUP)
	{
		cout << "Drag End" << endl;
		drag = false;
	}

	if (drag) {
		cout << "Original X: " << mouse.x << " New X: " << x <<endl;
		cout << "Original Y: " << mouse.y << " New Y: " << y <<endl;
		if (mouse.x < x) {
			u -= 0.1;
		}
		else if(mouse.x > x)
		{
			u += 0.1;
		}

		if (mouse.y < y) {
			v -= 0.1;
		}
		else if (mouse.y > y)
		{
			v += 0.1;
		}
		//cout << "U: " << u << endl;
		//cout << "V: " << u << endl;


		mouse = Point2d(x, y);
	}
	resImg = Mat::zeros(600, 800, CV_8UC3);
	drawPoints(mrw, u, v, rad, resImg);
	imshow("Display window", resImg);
}


float timeX = 0;
int timeConstant = 1;

int main(int argc, const char** argv) {

	

	if (argc != 2) { printf("Usage: FV filename\n");exit(0); }

	mrw = new MatrixReaderWriter(argv[1]);

	printf("%d %d\n", mrw->rowNum, mrw->columnNum);


	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	setMouseCallback("Display window", MouseCallBackFunc, NULL);


	int index = 0;

	int radStart = 1200;
	bool animate = true;

	v = 1.0;
	u = 0.5;
	rad = radStart;

	resImg = Mat::zeros(600, 800, CV_8UC3);
	drawPoints(mrw, u, v, rad, resImg);
	imshow("Display window", resImg);                   // Show our image inside it.


	char key;
	while (true) {
		key = cvWaitKey(50);
		if (key == 27) break;

		if (animate) {
			rad = radStart *(1.25 + 0.75 * sin(timeX * timeConstant));
			u += 0.1;
			v += 0.1;
			//cout << "U: " << u << endl;
		}


		switch (key) {
		case 'q': //Left
			u += 0.1;
			break;
		case 'a'://Right
			u -= 0.1;
			break;
		case 'w'://Up
			v += 0.1;
			break;
		case 's'://Down
			v -= 0.1;
			break;
		case 'e':
			rad *= 1.1;
			break;
		case 'd':
			rad /= 1.1;
			break;
		case 'l':
			animate = !animate;
			break;
		}
		

		if (animate) {
			timeX += 0.1;
		}
		//cout << "Rad: " << rad << endl;
		resImg = Mat::zeros(600, 800, CV_8UC3);
		drawPoints(mrw, u, v, rad, resImg);
		imshow("Display window", resImg);                   // Show our image inside it.
	}

}
