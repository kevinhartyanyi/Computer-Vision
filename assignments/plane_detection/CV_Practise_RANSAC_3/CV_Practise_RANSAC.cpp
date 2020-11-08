// CV_Practise_RANSAC.cpp : Defines the entry point for the console application.
//


#include <iostream>
#include <opencv2\cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <time.h>

using namespace cv;
using namespace std;

void GenerateData(vector<Point2d> &points,
	double noise,
	int pointNumber,
	int outlierNumber,
	Size size);

void DrawPoints(vector<Point2d> &points,
	Mat image);

void FitLineRANSAC(
	const vector<Point2d> &points,
	vector<int> &inliers,
	Mat &line,
	double threshold,
	double confidence_,
	int iteration_number,
	Mat *image = nullptr);

void FitLineLSQ(const vector<Point2d> * const points,
	vector<int> &inliers,
	Mat &line);

size_t GetIterationNumber(
	const double &inlierRatio_,
	const double &confidence_,
	const size_t &sampleSize_);

int _tmain(int argc, _TCHAR* argv[])
{
	// Setting the random seed to get random results in each run.
	srand(time(NULL));

	// Task:
	// Implement RANSAC and write a testing environment, where we can test if RANSAC works as expected.
	// 1. First, we should generate synthetic 2D points from a synthetic line and, also, generate outlier.
	// 2. Implement RANSAC and test it on the generated scene. 
	vector<Point2d> points; // The set of 2D points generated randomly
	Mat image = Mat::zeros(600, 600, CV_8UC3); // The image where we draw results. 

	// Generate the 2D synthetic scene
	GenerateData(points, // The output of the function, i.e., a set of data points. 
		10.0, // The noise in pixels
		100, // The number of points generated
		300, // The number of outliers generated
		Size(image.cols, image.rows)); // The image size
	
	// Drawing 2D points to the image
	DrawPoints(points, image);

	// Showing the image
	imshow("Image", image);
	// Waiting for user interaction
	waitKey(0);

	// The indices of the points of the line
	vector<int> inliers; 
	// The parameters of the line
	Mat bestLine;

	// RANSAC to find the line parameters and the inliers
	FitLineRANSAC(
		points, // The generated 2D points
		inliers, // Output: the indices of the inliers
		bestLine, // Output: the parameters of the found 2D line
		10.0, // The inlier-outlier threshold
		0.99, // The confidence required in the results
		1000/*, // The number of iterations
		&image*/); // Optional: the image where we can draw results

	// Drawing the inliers
	for (const auto &inlierIdx : inliers)
	{
		circle(image, // to this image 
			points[inlierIdx], // at this location
			3, // with this radius
			Scalar(0, 255, 0), // and this color
			-1); // The thickness of the circle's outline. -1 = filled circle
	}

	// Draw the line from RANSAC
	const double &a1 = bestLine.at<double>(0);
	const double &b1 = bestLine.at<double>(1);
	const double &c1 = bestLine.at<double>(2);

	// Draw the 2D line
	cv::line(image,
		Point2d(0, -c1 / b1),
		Point2d(image.cols, (-a1 * image.cols - c1) / b1),
		cv::Scalar(0, 255, 0),
		2);

	// Calculate the error or the RANSAC line
	double averageError1 = 0.0;
	for (const auto &inlierIdx : inliers)
	{
		double distance = abs(a1 * points[inlierIdx].x + b1 * points[inlierIdx].y + c1);
		averageError1 += distance;
	}
	averageError1 /= inliers.size();

	FitLineLSQ(&points, 
		inliers, 
		bestLine);
	
	// Draw the line from Least Squares Fitting
	const double &a2 = bestLine.at<double>(0);
	const double &b2 = bestLine.at<double>(1);
	const double &c2 = bestLine.at<double>(2);

	// Draw the 2D line
	cv::line(image,
		Point2d(0, -c2 / b2),
		Point2d(image.cols, (-a2 * image.cols - c2) / b2),
		cv::Scalar(0, 0, 255),
		2);

	// Calculate the error or the Least Squares Fitting line
	double averageError2 = 0.0;
	for (const auto &inlierIdx : inliers)
	{
		double distance = abs(a2 * points[inlierIdx].x + b2 * points[inlierIdx].y + c2);
		averageError2 += distance;
	}
	averageError2 /= inliers.size();

	printf("Avg. RANSAC error = %f px\n", averageError1);
	printf("Avg. LSQ error = %f px\n", averageError2);

	imshow("Final result", image);
	waitKey(0);

	return 0;
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

// Draw points to the image
void DrawPoints(vector<Point2d> &points,
	Mat image)
{
	for (int i = 0; i < points.size(); ++i)
	{
		// Draws a circle
		circle(image, // to this image 
			points[i], // at this location
			3, // with this radius
			Scalar(255, 255, 255), // and this color
			-1); // The thickness of the circle's outline. -1 = filled circle
	}
}

// Generate a synthetic line and sample that. Then add outliers to the data.
void GenerateData(
	vector<Point2d> &points, 
	double noise,
	int inlierNumber,
	int outlierNumber,
	Size size)
{
	// Generate random line by its normal direction and a center
	Point2d center;
	center.x = size.width *
		static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	center.y = size.height *
		static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

	double alpha = 180.0 * 
		static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	Point2d v; // Unit-length direction of the line norm(v) = ||v||_2 = 1
	v.x = cos(alpha);
	v.y = sin(alpha);

	const double diagonal =
		std::sqrt(size.width * size.width + size.height * size.height);
	
	// Generate random points on that line
	points.reserve(inlierNumber + outlierNumber);
	for (int i = 0; i < inlierNumber; ++i)
	{
		// L(t) = center + t * v
		// p_0 is a point in the line
		// v is the (tangent) direction of the line 
		// t is a distance from point p_0 in direction v
		Point2d point;
		do
		{
			double t = diagonal *
				static_cast<double>(rand()) / static_cast<double>(RAND_MAX) -
				diagonal / 2.0;

			point = center + t * v;
		} while (point.x < 0 && point.x > size.width &&
			point.y < 0 && point.y > size.height);

		// Adding noise to the point coordinates
		point.x += noise * static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - noise / 2.0;
		point.y += noise * static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - noise / 2.0;

		// Adding the point to the vector
		points.emplace_back(point);
	}

	// Add outliers
	for (int i = 0; i < outlierNumber; ++i)
	{
		Point2d point;
		point.x += size.width * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		point.y += size.height * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		points.emplace_back(point);
	}
}

// Apply RANSAC to fit points to a 2D line
void FitLineRANSAC(
	const vector<Point2d> &points_,
	vector<int> &inliers_, 
	Mat &line_,
	double threshold_,
	double confidence_,
	int maximum_iteration_number_,
	Mat *image_)
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
	Mat bestLine(3, 1, CV_64F);
	// Helpers to draw the line if needed
	Point2d bestPt1, bestPt2;
	// The sample size, i.e., 2 for 2D lines
	constexpr int kSampleSize = 2;
	// The current sample
	std::vector<int> sample(kSampleSize);

	bool shouldDraw = image_ != nullptr;
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
			} while (true);
		}

		if (shouldDraw)
		{
			tmp_image = image_->clone();

			circle(tmp_image, // to this image 
				points_[sample[0]], // at this location
				5, // with this radius
				Scalar(0, 0, 255), // and this color
				-1); // The thickness of the circle's outline. -1 = filled circle

			circle(tmp_image, // to this image 
				points_[sample[1]], // at this location
				5, // with this radius
				Scalar(0, 0, 255), // and this color
				-1); // The thickness of the circle's outline. -1 = filled circle
		}
			   
		// 2. Fit a line to the points.
		const Point2d &p1 = points_[sample[0]]; // First point selected
		const Point2d &p2 = points_[sample[1]]; // Second point select		
		Point2d v = p2 - p1; // Direction of the line
		// cv::norm(v) = sqrt(v.x * v.x + v.y * v.y)
		v = v / cv::norm(v);
		// Rotate v by 90� to get n.
		Point2d n;
		n.x = -v.y;
		n.y = v.x;
		// To get c use a point from the line.
		double a = n.x;
		double b = n.y;
		double c = -(a * p1.x + b * p1.y);

		// Draw the 2D line
		if (shouldDraw)
		{
			cv::line(tmp_image,
				Point2d(0, -c / b),
				Point2d(tmp_image.cols, (- a * tmp_image.cols - c) / b),
				cv::Scalar(0, 255, 0),
				2);
		}

		// - Distance of a line and a point
		// - Line's implicit equations: a * x + b * y + c = 0
		// - a, b, c - parameters of the line
		// - x, y - coordinates of a point on the line
		// - n = [a, b] - the normal of the line
		// - Distance(line, point) = | a * x + b * y + c | / sqrt(a * a + b * b)
		// - If ||n||_2 = 1 then sqrt(a * a + b * b) = 1 and I don't have do the division.

		// 3. Count the number of inliers, i.e., the points closer than the threshold.
		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx)
		{
			const Point2d &point = points_[pointIdx];
			const double distance =
				abs(a * point.x + b * point.y + c);

			if (distance < threshold_)
			{
				inliers.emplace_back(pointIdx);

				if (shouldDraw)
				{
					circle(tmp_image, // to this image 
						points_[pointIdx], // at this location
						3, // with this radius
						Scalar(0, 255, 0), // and this color
						-1); // The thickness of the circle's outline. -1 = filled circle
				}
			}
		}

		// 4. Store the inlier number and the line parameters if it is better than the previous best. 
		if (inliers.size() > bestInliers.size())
		{
			bestInliers.swap(inliers);
			inliers.clear();
			inliers.resize(0);

			bestLine.at<double>(0) = a;
			bestLine.at<double>(1) = b;
			bestLine.at<double>(2) = c;

			// Update the maximum iteration number
			maximumIterations = GetIterationNumber(
				static_cast<double>(bestInliers.size()) / static_cast<double>(points_.size()),
				confidence_,
				kSampleSize);

			printf("Inlier number = %d\tMax iterations = %d\n", bestInliers.size(), maximumIterations);
		}

		if (shouldDraw)
		{
			cv::line(tmp_image,
				Point2d(0, -bestLine.at<double>(2) / bestLine.at<double>(1)),
				Point2d(tmp_image.cols, (-bestLine.at<double>(0) * tmp_image.cols - bestLine.at<double>(2)) / bestLine.at<double>(1)),
				cv::Scalar(255, 0, 0),
				2);

			cv::imshow("Image", tmp_image);
			cv::waitKey(0);
		}
	}

	inliers_ = bestInliers;
	line_ = bestLine;
}

// Apply Least-Squares line fitting (PCL).
void FitLineLSQ(const vector<Point2d> * const points,
	vector<int> &inliers,
	Mat &line)
{
	vector<Point2d> normalizedPoints;
	normalizedPoints.reserve(inliers.size());

	// Calculating the mass point of the points
	Point2d masspoint(0, 0);

	for (const auto &inlierIdx : inliers)
	{
		masspoint += points->at(inlierIdx);
		normalizedPoints.emplace_back(points->at(inlierIdx));
	}
	masspoint = masspoint * (1.0 / inliers.size());

	// Move the point cloud to have the origin in their mass point
	for (auto &point : normalizedPoints)
		point -= masspoint;

	// Calculating the average distance from the origin
	double averageDistance = 0.0;
	for (auto &point : normalizedPoints)
	{
		averageDistance += cv::norm(point);
		// norm(point) = sqrt(point.x * point.x + point.y * point.y)
	}

	averageDistance /= normalizedPoints.size();
	const double ratio = sqrt(2) / averageDistance;

	// Making the average distance to be sqrt(2)
	for (auto &point : normalizedPoints)
		point *= ratio;

	// Now, we should solve the equation.
	cv::Mat A(normalizedPoints.size(), 2, CV_64F);

	// Building the coefficient matrix
	for (size_t pointIdx = 0; pointIdx < normalizedPoints.size(); ++pointIdx)
	{
		const size_t &rowIdx = pointIdx;

		A.at<double>(rowIdx, 0) = normalizedPoints[pointIdx].x;
		A.at<double>(rowIdx, 1) = normalizedPoints[pointIdx].y;
	}

	cv::Mat evals, evecs;
	cv::eigen(A.t() * A, evals, evecs);

	const cv::Mat &normal = evecs.row(1);
	const double &a = normal.at<double>(0),
		&b = normal.at<double>(1);
	const double c = -(a * masspoint.x + b * masspoint.y);

	line.at<double>(0) = a;
	line.at<double>(1) = b;
	line.at<double>(2) = c;
}