// CV_Practise_RANSAC.cpp : Defines the entry point for the console application.
//


#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;


void GenerateData(vector<Point2d> &points,
	double noise,
	int pointNumber,
	int outlierNumber,
	Size size);

void DrawPoints(vector<Point2d> &points,
	Mat image);

void FitLineRANSAC(const vector<Point2d> * const points,
	vector<int> &inliers,
	Mat &line,
	double threshold,
	int iteration_number,
	Mat image);

void FitLineLSQ(const vector<Point2d> * const points,
	vector<int> &inliers,
	Mat &line);

int main(int argc, _TCHAR* argv[])
{
	vector<Point2d> points;
	Mat image = Mat::zeros(600, 600, CV_8UC3);

	GenerateData(points, 8, 100, 300, Size(image.cols, image.rows));

	DrawPoints(points, image);

	imshow("Image", image);
	waitKey(0);

	vector<int> inliers;
	Mat bestLine;
	FitLineRANSAC(&points, inliers, bestLine, 2.0, 1000, image);

	FitLineLSQ(&points, inliers, bestLine);

	cout << bestLine << endl;
	double a = bestLine.at<double>(0), b = bestLine.at<double>(1), c = bestLine.at<double>(2);

	Point2d pt1(0, -c / b);
	Point2d pt2(image.cols, -(c + image.cols * a) / b);
	cv::line(image, pt1, pt2, Scalar(0, 255, 0), 2);
	
	imshow("Final result", image);
	waitKey(0);

	return 0;
}

// Draw points to the image
void DrawPoints(vector<Point2d> &points,
	Mat image)
{
	for (int i = 0; i < points.size(); ++i)
	{
		circle(image, points[i], 2, Scalar(255, 255, 255));
	}
}

// Generate a synthetic line and sample that. Then add outliers to the data.
void GenerateData(vector<Point2d> &points, 
	double noise,
	int pointNumber,
	int outlierNumber,
	Size size)
{
	// Generate random line by its normal direction and a center
	Point2d center;
	center.x = size.width * (rand() / (double)RAND_MAX);
	center.y = size.height * (rand() / (double)RAND_MAX);

	double a, b, c;
	a = rand() / (double)RAND_MAX;
	b = rand() / (double)RAND_MAX;
	c = -a * center.x - b * center.y;

	// Generate random points on that line
	points.resize(pointNumber + outlierNumber);
	for (int i = 0; i < pointNumber; ++i)
	{
		double x = size.width * (rand() / (double)RAND_MAX);
		double y = -(a * x + c) / b;
		points[i].x = x + (rand() / (double)RAND_MAX) * noise;
		points[i].y = y + (rand() / (double)RAND_MAX) * noise;
	}

	// Add outliers
	for (int i = 0; i < outlierNumber; ++i)
	{
		double x = size.width * (rand() / (double)RAND_MAX);
		double y = size.height * (rand() / (double)RAND_MAX);
		points[pointNumber + i].x = x;
		points[pointNumber + i].y = y;
	}
}

// Apply RANSAC to fit points to a 2D line
void FitLineRANSAC(const vector<Point2d> * const points,
	vector<int> &inliers, 
	Mat &line,
	double threshold,
	int iteration_number,
	Mat image)
{
	int it = 0;
	int bestInlierNumber = 0;
	vector<int> bestInliers;
	Mat bestLine;
	Point2d bestPt1, bestPt2;

	while (it++ < iteration_number)
	{
		Mat img = image.clone();

		// Select MSS
		vector<int> indices(points->size());
		for (int i = 0; i < indices.size(); ++i)
			indices[i] = i;

		vector<int> mss(2);
		for (int i = 0; i < mss.size(); ++i)
		{
			int idx = round((rand() / (double)RAND_MAX) * (indices.size() - 1));
			mss[i] = indices[idx];
			indices.erase(indices.begin() + idx);
		}

		// Calculate line parameters
		double a, b, c;
		Point2d v = points->at(mss[1]) - points->at(mss[0]);
		v = v / norm(v);
		Point2d n(-v.x, v.y);

		a = n.x;
		b = n.y;
		c = -a * points->at(mss[1]).x - b * points->at(mss[1]).y;

		// Get the inliers
		int inlierNumber = 0;
		vector<int> inliers;
		for (int i = 0; i < points->size(); ++i)
		{
			double distance = abs(a * points->at(i).x + b * points->at(i).y + c);
			if (distance < threshold)
			{
				++inlierNumber;
				inliers.push_back(i);

				circle(img, points->at(i), 2, Scalar(0, 0, 255));
			}
		}

		Point pt1(-c / a, 0);
		Point pt2(-(c + b*image.rows) / a, image.rows);

		cv::line(img, pt1, pt2, Scalar(0, 0, 255), 1);
		if (bestInlierNumber > 0)
			cv::line(img, bestPt1, bestPt2, Scalar(0, 255, 0), 1);

		//imshow("Current line", img);
		//waitKey(200);

		// Store the best model
		if (inlierNumber > bestInlierNumber)
		{
			bestInlierNumber = inlierNumber;
			bestInliers = inliers;
			bestLine = (Mat_<double>(3, 1) << a, b, c);

			bestPt1 = pt1;
			bestPt2 = pt2;
		}
	}

	inliers = bestInliers;
	line = bestLine;
}

// Apply Least-Squares line fitting (PCL).
void FitLineLSQ(const vector<Point2d> * const points,
	vector<int> &inliers,
	Mat &line)
{
	vector<Point2d> pts(inliers.size());
	Point2d massPoint;

	for (int i = 0; i < pts.size(); ++i)
		pts[i] = points->at(inliers[i]);

	Mat A(pts.size(), 3, CV_64F);
	for (int i = 0; i < pts.size(); ++i)
	{
		A.at<double>(i, 0) = pts[i].x;
		A.at<double>(i, 1) = pts[i].y;
		A.at<double>(i, 2) = 1;
	}

	Mat AtA = A.t() * A;
	Mat eValues, eVectors;
	eigen(AtA, eValues, eVectors);
	
	line = eVectors.row(2);
}