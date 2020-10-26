#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/cvconfig.h>
#include <iostream>
#include <string>


using namespace cv;
using namespace std;

void DrawPoints(vector<Point2d> &points,
	Mat image);

void FitLineRANSAC(
	const vector<Point2d> &points,
	vector<int> &inliers,
	Mat &line,
	double threshold,
	int iteration_number,
	Mat *image = nullptr,
	bool shouldDraw = false);

int main(int argc, char *argv[])
{
	if( argc != 2)
    {
		cout << "Use an iteration number" << endl;
		return -1;
    }

    string img_name = "left.jpg";

    Mat image = imread(img_name);
    Mat contours;

    Canny(image,contours,100,200);

    namedWindow("Image");
    imshow("Image",image);

    namedWindow("Canny");
    imshow("Canny",contours);

    Mat edge_binary = contours / 255;

    // The indices of the points of the line
	vector<int> inliers; 
	// The parameters of the line
	Mat bestLine;

    Mat results = Mat::zeros(edge_binary.rows, edge_binary.cols, CV_8UC3); // The image where we draw results. 
    vector<Point2d> points;

    for (int i=0; i<edge_binary.rows; i++) {
        for (int j=0; j < edge_binary.cols; j++)
        {
            //cout << edge_binary.at<double>(i, j) << endl;
            if (edge_binary.at<bool>(i, j)){
                points.push_back(cv::Point2d(j, i));
            }
        }
    }

    DrawPoints(points, results);

    imshow("Image", results);


    FitLineRANSAC(
		points, // The generated 2D points
		inliers, // Output: the indices of the inliers
		bestLine, // Output: the parameters of the found 2D line
		1, // The inlier-outlier threshold
		atoi(argv[1]), // The number of iterations
		&results,// Optional: the image where we can draw results
		false); 

	imshow("Result", results);

	imwrite("result_" + img_name, results);

    waitKey(0);
} 

void DrawPoints(vector<Point2d> &points,
	Mat image)
{
	for (int i = 0; i < points.size(); ++i)
	{
		// Draws a circle
		circle(image, // to this image 
			points[i], // at this location
			1, // with this radius
			Scalar(255, 255, 255), // and this color
			-1); // The thickness of the circle's outline. -1 = filled circle
	}
}

// Apply RANSAC to fit points to a 2D line
void FitLineRANSAC(
	const vector<Point2d> &points_,
	vector<int> &inliers_, 
	Mat &line_,
	double threshold_,
	int maximum_iteration_number_,
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
	Mat bestLine(3, 1, CV_64F);
	// Helpers to draw the line if needed
	Point2d bestPt1, bestPt2;
	// The sample size, i.e., 2 for 2D lines
	constexpr int kSampleSize = 2;
	// The current sample
	std::vector<int> sample(kSampleSize);

	cv::Mat tmp_image;

	// RANSAC:
	// 1. Select a minimal sample, i.e., in this case, 2 random points.
	// 2. Fit a line to the points.
	// 3. Count the number of inliers, i.e., the points closer than the threshold.
	// 4. Store the inlier number and the line parameters if it is better than the previous best. 

	while (iterationNumber++ < maximum_iteration_number_)
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
				10, // with this radius
				Scalar(255, 0, 0), // and this color
				-1); // The thickness of the circle's outline. -1 = filled circle

			circle(tmp_image, // to this image 
				points_[sample[1]], // at this location
				10, // with this radius
				Scalar(255, 0, 0), // and this color
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
						Scalar(0, 0, 255), // and this color
						-1); // The thickness of the circle's outline. -1 = filled circle
				}
			}
		}

        cout << "Inliner number: " << inliers.size() << endl;
		// 4. Store the inlier number and the line parameters if it is better than the previous best. 
		if (inliers.size() > 200)
		{			

			cv::line(*image_,
				Point2d(0, -c / b),
				Point2d(image_->cols, (- a * image_->cols - c) / b),
				cv::Scalar(0, 0, 255),
				2);
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
}