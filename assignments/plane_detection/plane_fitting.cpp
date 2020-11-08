#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/cvconfig.h>
#include <iostream>
#include <string>
#include "MatrixReaderWriter.h"
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

using namespace cv;
using namespace std;

void FitPlaneLORANSAC(
	const vector<Point3f> &points,
	vector<int> &inliers,
	Mat &line,
	double threshold,
	int iteration_number,
	double confidence_,
	Mat *image = nullptr,
	bool shouldDraw = false);

size_t GetIterationNumber(
	const double &inlierRatio_,
	const double &confidence_,
	const size_t &sampleSize_);

vector<Point3f> MRWTo3DPoints(const MatrixReaderWriter &mrw);
MatrixReaderWriter PointsToMRW(const vector<Point3f> &points, int rowNum, int columnNum, const vector<int> &colorPoints);
void print3dpoints(const vector<Point3f> &points, int num = 10000);

int rowNum;
int columnNum;

int localOptimizationType = 0;

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		std::cout << "Usage: XYZ Filename Iteration number" << endl;
		return -1;
	}
	MatrixReaderWriter mrw(argv[1]);
	rowNum = mrw.rowNum;
	columnNum = mrw.columnNum;
	Mat results = Mat::zeros(rowNum, columnNum, CV_8UC3); // The image where we draw results.

	//show3DPoints(mrw);
	vector<Point3f> points3D = MRWTo3DPoints(mrw);

	//print3dpoints(points3D);

	//cout << points3D[0].x << " BLUB " << points3D[1].x << endl;
	//cout << rowNum*columnNum << " Points " << points3D.size() << endl;

	vector<int> color;
	for (int i = 0; i < 100; i++)
	{
		color.push_back(i);
	}

	//show3DPoints(tmp);

	// The indices of the points of the line
	vector<int> inliers;
	// The parameters of the line
	Mat bestPlane;

	//DrawPoints(points, results);

	FitPlaneLORANSAC(
		points3D,	   // The generated 2D points
		inliers,	   // Output: the indices of the inliers
		bestPlane,	   // Output: the parameters of the found 2D line
		0.1,		   // The inlier-outlier threshold
		atoi(argv[2]), // The number of iterations
		0.99,		   // The confidence required in the results
		&results,	   // Optional: the image where we can draw results
		false);

	waitKey(0);
}

size_t GetIterationNumber(
	const double &inlierRatio_,
	const double &confidence_,
	const size_t &sampleSize_)
{
	double a =
		log(1.0 - confidence_);
	double b =
		log(1.0 - std::pow(inlierRatio_, sampleSize_));

	if (abs(b) < std::numeric_limits<double>::epsilon())
		return std::numeric_limits<size_t>::max();

	return a / b;
}

void print3dpoints(const vector<Point3f> &points, int num)
{
	for (int i = 0; i < points.size() && i < num; i++)
	{
		std::cout << points[i].x << " " << points[i].y << " " << points[i].z << endl;
	}
}

vector<Point3f> MRWTo3DPoints(const MatrixReaderWriter &mrw)
{
	int NUM = mrw.rowNum;
	vector<Point3f> re;
	std::cout << "COLUMN: " << mrw.columnNum << endl;

	for (int i = 0; i < NUM; i++)
	{
		if (mrw.columnNum == 3)
		{
			re.push_back(Point3f(mrw.data[3 * i], mrw.data[3 * i + 1], mrw.data[3 * i + 2]));
		}
		else if (mrw.columnNum == 6)
		{
			re.push_back(Point3f(mrw.data[6 * i], mrw.data[6 * i + 1], mrw.data[6 * i + 2]));
		}
		else if (mrw.columnNum == 4)
		{
			re.push_back(Point3f(mrw.data[4 * i], mrw.data[4 * i + 1], mrw.data[4 * i + 2]));
		}
	}

	return re;
}

MatrixReaderWriter PointsToMRW(const vector<Point3f> &points, int rowNum, int columnNum, const vector<int> &colorPoints)
{

	if (columnNum != 6)
	{
		columnNum = 6;
	}

	MatrixReaderWriter re(rowNum, columnNum);

	for (int i = 0; i < points.size(); i++)
	{
		re.data[i * columnNum] = points[i].x;
		re.data[i * columnNum + 1] = points[i].y;
		re.data[i * columnNum + 2] = points[i].z;
		// if (colorPoints != nullptr && find(colorPoints->begin(), colorPoints->end(), points[i]) != colorPoints->end()){
		// 	re.data[i*columnNum+3] = 255;
		// 	re.data[i*columnNum+4] = 255;
		// 	re.data[i*columnNum+5] = 255;
		// } else {
		// 	re.data[i*columnNum+3] = 0;
		// 	re.data[i*columnNum+4] = 0;
		// 	re.data[i*columnNum+5] = 0;
		// }
		re.data[i * columnNum + 3] = 0;
		re.data[i * columnNum + 4] = 0;
		re.data[i * columnNum + 5] = 0;
	}

	for (int i = 0; i < colorPoints.size(); ++i)
	{
		re.data[colorPoints[i] * columnNum + 3] = 255;
		re.data[colorPoints[i] * columnNum + 4] = 255;
		re.data[colorPoints[i] * columnNum + 5] = 255;
	}

	return re;
}

void LocalOptimization(const vector<int> &inliners, const vector<Point3f> &points, vector<int> &optimizedInliners, Mat &bestPlane, double threshold_);

// Apply RANSAC to fit points to a 2D line
void FitPlaneLORANSAC(
	const vector<Point3f> &points_,
	vector<int> &inliers_,
	Mat &plane_,
	double threshold_,
	int maximum_iteration_number_,
	double confidence_,
	Mat *image_,
	bool shouldDraw)
{
	// The current number of iterations
	int iterationNumber = 0;
	// The number of inliers of the current best model
	int bestInlierNumber = 0;
	// The indices of the inliers of the current best model
	vector<int> bestInliers, inliers;
	bestInliers.reserve(points_.size());
	inliers.reserve(points_.size());
	// The parameters of the best line
	Mat bestPlane(4, 1, CV_64F);
	// Helpers to draw the line if needed
	Point2d bestPt1, bestPt2;
	// The sample size, i.e., 2 for 2D lines
	constexpr int kSampleSize = 3;
	// The current sample
	std::vector<int> sample(kSampleSize);

	cv::Mat tmp_image;
	size_t maximumIterations = maximum_iteration_number_;

	// RANSAC:
	// 1. Select a minimal sample, i.e., in this case, 2 random points.
	// 2. Fit a line to the points.
	// 3. Count the number of inliers, i.e., the points closer than the threshold.
	// 4. Store the inlier number and the line parameters if it is better than the previous best.

	while (iterationNumber++ < maximumIterations)
	{

		// 1. Select a minimal sample, i.e., in this case, 2 random points.
		for (size_t sampleIdx = 0; sampleIdx < kSampleSize; ++sampleIdx)
		{
			do
			{
				// Generate a random index between [0, pointNumber]
				sample[sampleIdx] =
					round((points_.size() - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

				// If the first point is selected we don't have to check if
				// that particular index had already been selected.
				if (sampleIdx == 0)
					break;

				// If the second point is being generated,
				// it should be checked if the index had been selected beforehand.
				if (sampleIdx == 1 &&
					sample[0] != sample[1])
					break;

				if (sampleIdx == 2 &&
					sample[0] != sample[1] && sample[0] != sample[2] && sample[1] != sample[2])
					break;
			} while (true);
		}

		if (shouldDraw)
		{
			vector<Point3f> color;
			color.push_back(points_[sample[0]]); // The thickness of the circle's outline. -1 = filled circle
			color.push_back(points_[sample[1]]); // The thickness of the circle's outline. -1 = filled circle
			color.push_back(points_[sample[2]]);
			//PointsToMRW(points_, rowNum, columnNum, &color).save("Selected_points.xyz");
			//cout << "Selected points" << endl;
			//getchar();  // The thickness of the circle's outline. -1 = filled circle
		}

		// 2. Fit a line to the points.
		const Point3f &p1 = points_[sample[0]]; // First point selected
		const Point3f &p2 = points_[sample[1]]; // Second point select
		const Point3f &p3 = points_[sample[2]]; // Second point select

		Point3f helperP1 = p2 - p1;
		Point3f helperP2 = p3 - p1;
		double a = helperP1.y * helperP2.z - helperP2.y * helperP1.x;
		double b = helperP2.x * helperP1.z - helperP1.x * helperP2.z;
		double c = helperP1.x * helperP2.y - helperP1.y * helperP2.x;
		double d = (-a * p1.x - b * p1.y - c * p1.z);

		// - Distance of a line and a point
		// - Line's implicit equations: a * x + b * y + c = 0
		// - a, b, c - parameters of the line
		// - x, y - coordinates of a point on the line
		// - n = [a, b] - the normal of the line
		// - Distance(line, point) = | a * x + b * y + c | / sqrt(a * a + b * b)
		// - If ||n||_2 = 1 then sqrt(a * a + b * b) = 1 and I don't have do the division.

		// 3. Count the number of inliers, i.e., the points closer than the threshold.
		vector<Point3f> coloredPoints;
		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx)
		{
			const Point3f &point = points_[pointIdx];
			const double distance =
				static_cast<double>(abs(a * point.x + b * point.y + c * point.z + d)) /
				static_cast<double>(sqrt(a * a + b * b + c * c));

			//cout << "Distance: " << distance << endl;

			if (distance < threshold_)
			{
				inliers.emplace_back(pointIdx);

				if (shouldDraw)
				{
					//coloredPoints.push_back(points_[pointIdx]);					 			  // The thickness of the circle's outline. -1 = filled circle
				}
			}
		}

		//std::cout << "Inliner number: " << inliers.size() << endl;
		// 4. Store the inlier number and the line parameters if it is better than the previous best.
		if (inliers.size() > bestInliers.size())
		{
			//PointsToMRW(points_, rowNum, columnNum, inliers).save("Inliners.xyz");
			std::cout << "Bestinliner num: " << inliers.size() << endl;
			LocalOptimization(inliers, points_, bestInliers, bestPlane, threshold_);
			std::cout << "Optimized Inliner num: " << bestInliers.size() << endl;
			//PointsToMRW(points_, rowNum, columnNum, bestInliers).save("OptimizedInliners.xyz");
			//getchar();

			// bestInliers.swap(inliers);
			inliers.clear();
			inliers.resize(0);

			// bestPlane.at<double>(0) = a;
			// bestPlane.at<double>(1) = b;
			// bestPlane.at<double>(2) = c;
			// bestPlane.at<double>(4) = d;

			maximumIterations = GetIterationNumber(
				static_cast<double>(bestInliers.size()) / static_cast<double>(points_.size()),
				confidence_,
				kSampleSize);
			cout << "New maxiteration number: " << maximumIterations << "\n"
				 << endl;
		}

		if (shouldDraw)
		{
			PointsToMRW(points_, rowNum, columnNum, inliers).save("Inliners.xyz");
			std::cout << "Inliners" << endl;
		}
	}

	inliers_ = bestInliers;
	plane_ = bestPlane;

	vector<Point3f> bestColoredPoints;

	std::cout << "Final BestInliner number: " << bestInliers.size() << endl;

	PointsToMRW(points_, rowNum, columnNum, bestInliers).save("BestInliners.xyz");
}

void innerRANSAC(const vector<int> &currentInliners, const vector<Point3f> &points_, 
vector<int> &optimizedInliners, Mat &bestPlane,
				 int maxIterationNumber = 20, double threshold_ = 0.02);

void LocalOptimization(const vector<int> &inliners, const vector<Point3f> &points, vector<int> &optimizedInliners, Mat &bestPlane, double threshold_)
{
	if (localOptimizationType == 0)
	{
		innerRANSAC(inliners, points, optimizedInliners, bestPlane, 20, threshold_);
	}
}

void innerRANSAC(const vector<int> &currentInliners, const vector<Point3f> &points_, vector<int> &optimizedInliners, Mat &bestPlane,
				 int maxIterationNumber, double threshold_)
{

	int iterationNumber = 0;
	int bestInlierNumber = currentInliners.size();
	vector<int> bestInliers, inliers;
	bestInliers.reserve(points_.size());
	inliers.reserve(points_.size());
	constexpr int kSampleSize = 3;
	std::vector<int> sample(kSampleSize);

	bestInliers = currentInliners;

	while (iterationNumber++ < maxIterationNumber)
	{
		// Inner RANSAC
		for (size_t sampleIdx = 0; sampleIdx < kSampleSize; ++sampleIdx)
		{
			do
			{
				sample[sampleIdx] =
					round((currentInliners.size() - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

				if (sampleIdx == 0)
					break;

				if (sampleIdx == 1 &&
					sample[0] != sample[1])
					break;

				if (sampleIdx == 2 &&
					sample[0] != sample[1] && sample[0] != sample[2] && sample[1] != sample[2])
					break;
			} while (true);
		}

		const Point3f &p1 = points_[currentInliners[sample[0]]]; // First point selected
		const Point3f &p2 = points_[currentInliners[sample[1]]]; // Second point select
		const Point3f &p3 = points_[currentInliners[sample[2]]]; // Second point select

		Point3f helperP1 = p2 - p1;
		Point3f helperP2 = p3 - p1;
		double a = helperP1.y * helperP2.z - helperP2.y * helperP1.x;
		double b = helperP2.x * helperP1.z - helperP1.x * helperP2.z;
		double c = helperP1.x * helperP2.y - helperP1.y * helperP2.x;
		double d = (-a * p1.x - b * p1.y - c * p1.z);

		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx)
		{
			const Point3f &point = points_[pointIdx];
			const double distance =
				static_cast<double>(abs(a * point.x + b * point.y + c * point.z + d)) /
				static_cast<double>(sqrt(a * a + b * b + c * c));

			if (distance < threshold_)
			{
				inliers.emplace_back(pointIdx);
			}
		}

		if (inliers.size() > bestInliers.size())
		{
			bestInliers.swap(inliers);
			inliers.clear();
			inliers.resize(0);

			bestPlane.at<double>(0) = a;
			bestPlane.at<double>(1) = b;
			bestPlane.at<double>(2) = c;
			bestPlane.at<double>(4) = d;
		}
	}
	optimizedInliners = bestInliers;
}