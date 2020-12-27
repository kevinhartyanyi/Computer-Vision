// CV_Practise_RANSAC.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv\cv.hpp>
#include <opencv\highgui.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <chrono>  
#include <tchar.h>

using namespace std;

// Detecting point correspondences in two images
void readFeatures(
	const string fileName, // The destination image
	std::vector<cv::Point2d> &source_points_, // Points in the source image
	std::vector<cv::Point2d> &destination_points_); // Points in the destination image

// A function estimating the fundamental matrix from point correspondences
// by RANSAC.
void ransacFundamentalMatrix(
	const std::vector<cv::Point2d> &input_source_points_, // Points in the source image
	const std::vector<cv::Point2d> &input_destination_points_, // Points in the destination image
	const std::vector<cv::Point2d> &normalized_input_src_points_, // Normalized points in the source image
	const std::vector<cv::Point2d> &normalized_input_destination_points_, // Normalized points in the destination image
	const cv::Mat &T1_, // Normalizing transformation in the source image
	const cv::Mat &T2_, // Normalizing transformation in the destination image
	cv::Mat &fundamental_matrix_, // The estimated fundamental matrix
	std::vector<size_t> &inliers_, // The inliers of the fundamental matrix
	double confidence_, // The required confidence of RANSAC
	double threshold_); // The inlier-outlier threshold

// A function estimating the fundamental matrix from point correspondences
// by least-squares fitting.
void getFundamentalMatrixLSQ(
	const std::vector<cv::Point2d> &source_points_, // Points in the source image
	const std::vector<cv::Point2d> &destination_points_, // Points in the destination image
	cv::Mat &fundamental_matrix_); // The estimated fundamental matrix

// A function decomposing the essential matrix to the projection matrices
// of the two cameras.
void getProjectionMatrices(
	const cv::Mat &essential_matrix_, // The parameters of the essential matrix
	const cv::Mat &K1_, // The intrinsic camera parameters of the source image
	const cv::Mat &K2_, // The intrinsic camera parameters of the destination image
	const cv::Mat &src_point_, // A point in the source image
	const cv::Mat &dst_point_, // A point in the destination image
	cv::Mat &projection_1_, // The projection matrix of the source image
	cv::Mat &projection_2_); // The projection matrix of the destination image

// A function estimating the 3D point coordinates from a point correspondences
// from the projection matrices of the two observing cameras.
void linearTriangulation(
	const cv::Mat &projection_1_, // The projection matrix of the source image
	const cv::Mat &projection_2_, // The projection matrix of the destination image
	const cv::Mat &src_point_, // A point in the source image
	const cv::Mat &dst_point_, // A point in the destination image
	cv::Mat &point3d_); // The estimated 3D coordinates

// Normalizing the point coordinates for the fundamental matrix estimation
void normalizePoints(
	const std::vector<cv::Point2d> &input_source_points_, // Points in the source image
	const std::vector<cv::Point2d> &input_destination_points_, // Points in the destination image
	std::vector<cv::Point2d> &output_source_points_, // Normalized points in the source image
	std::vector<cv::Point2d> &output_destination_points_, // Normalized points in the destination image
	cv::Mat &T1_, // Normalizing transformation in the source image
	cv::Mat &T2_); // Normalizing transformation in the destination image

// Return the iteration number of RANSAC given the inlier ratio and
// a user-defined confidence value.
int getIterationNumber(int point_number_, // The number of points
	int inlier_number_, // The number of inliers
	int sample_size_, // The sample size
	double confidence_); // The required confidence

// Printing the time to the console
void printTimes(
	const std::chrono::time_point<std::chrono::system_clock> &start, // The starting time
	const std::chrono::time_point<std::chrono::system_clock> &end, // The current time
	const std::string &message); // The message to be written

// Visualize the effect of the point normalization
cv::Mat checkEffectOfNormalization(const std::vector<cv::Point2d> &source_points_,  // Points in the first image 
	const std::vector<cv::Point2d> &destination_points_,   // Points in the second image
	const std::vector<cv::Point2d> &normalized_source_points_,  // Normalized points in the first image 
	const std::vector<cv::Point2d> &normalized_destination_points_, // Normalized points in the second image
	const cv::Mat &T1_, // Normalizing transforcv::Mation in the first image
	const cv::Mat &T2_, // Normalizing transforcv::Mation in the second image
	const std::vector<size_t> &inliers_); // The inliers of the fundamental matrix

int _tmain(int argc, _TCHAR* argv[])
{
	// Using time point and system_clock 
	std::chrono::time_point<std::chrono::system_clock> start, end;

	// Load images
	cv::Mat image1 = cv::imread("assets/2/1.png", cv::IMREAD_ANYCOLOR);
	cv::Mat image2 = cv::imread("assets/2/2.png", cv::IMREAD_ANYCOLOR);

	const std::string fileName1 = "assets/1/W_first_normal_general.mat";
	const std::string fileName2 = "assets/2/W_first_normal_leftright.mat";

	// Read features
	std::vector<cv::Point2d> source_points, destination_points; // Point correspondences
	readFeatures(fileName2, // Second image
		source_points, // Points in the first image 
		destination_points); // Points in the second image 

	// Normalize the coordinates of the point correspondences to achieve numerically stable results
	start = std::chrono::system_clock::now();
	cv::Mat T1, T2; // Normalizing transforcv::Mations
	std::vector<cv::Point2d> normalized_source_points, normalized_destination_points; // Normalized point correspondences
	normalizePoints(source_points, // Points in the first image 
		destination_points,  // Points in the second image
		normalized_source_points,  // Normalized points in the first image
		normalized_destination_points, // Normalized points in the second image
		T1, // Normalizing transforcv::Mation in the first image
		T2); // Normalizing transforcv::Mation in the second image
	end = std::chrono::system_clock::now();
	printTimes(start, end, "feature normalization");

	start = std::chrono::system_clock::now();
	cv::Mat F; // The fundamental matrix
	std::vector<size_t> inliers; // The inliers of the fundamental matrix
	ransacFundamentalMatrix(source_points,  // Points in the first image 
		destination_points,   // Points in the second image
		normalized_source_points,  // Normalized points in the first image 
		normalized_destination_points, // Normalized points in the second image
		T1, // Normalizing transforcv::Mation in the first image
		T2, // Normalizing transforcv::Mation in the second image
		F, // The fundamental matrix
		inliers, // The inliers of the fundamental matrix
		0.99, // The required confidence in the results 
		1.0); // The inlier-outlier threshold
	end = std::chrono::system_clock::now();
	printTimes(start, end, "RANSAC");

	cout << "Correspondence number after filtering = %d" << inliers.size() << endl;

	// Check the effect of the normalization and
	// fit the fundamental matrix to all correspondences 
	F = checkEffectOfNormalization(
		source_points,
		destination_points,
		normalized_source_points,
		normalized_destination_points,
		T1,
		T2,
		inliers);
	
	// Calibration matrix (intrinsic camera parameters)
	start = std::chrono::system_clock::now();
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1262.620252, 0, 934.611657,
											0, 1267.365350, 659.520995,
											0, 0, 1);

	// Essential matrix
	cv::Mat E = K.t() * F * K;
	cv::Mat P1, P2;

	getProjectionMatrices(E,
		K,
		K,
		(cv::Mat)source_points[inliers[0]],
		(cv::Mat)destination_points[inliers[0]],
		P1,
		P2);

	// Draw the points and the corresponding epipolar lines
	constexpr double resize_by = 4.0;
	cv::Mat tmp_image1, tmp_image2;
	resize(image1, tmp_image1, cv::Size(image1.cols / resize_by, image1.rows / resize_by));
	resize(image2, tmp_image2, cv::Size(image2.cols / resize_by, image2.rows / resize_by));

	std::vector<cv::KeyPoint> src_inliers, dst_inliers;
	std::vector<cv::DMatch> inlier_matches;
	src_inliers.reserve(inliers.size());
	dst_inliers.reserve(inliers.size());
	inlier_matches.reserve(inliers.size());
	std::ofstream out_file("triangulated_points.txt");
	start = std::chrono::system_clock::now();
	for (auto inl_idx = 0; inl_idx < inliers.size(); inl_idx += 1)
	{
		const size_t& inlierIdx = inliers[inl_idx];
		const cv::Mat pt1 = static_cast<cv::Mat>(source_points[inlierIdx]);
		const cv::Mat pt2 = static_cast<cv::Mat>(destination_points[inlierIdx]);

		// Estimate the 3D coordinates of the current inlier correspondence
		cv::Mat point3d;
		linearTriangulation(P1,
			P2,
			pt1,
			pt2,
			point3d);

		// Get the color of the point
		const int xi1 = round(source_points[inlierIdx].x);
		const int yi1 = round(source_points[inlierIdx].y);
		const int xi2 = round(destination_points[inlierIdx].x);
		const int yi2 = round(destination_points[inlierIdx].y);

		const cv::Vec3b &color1 = (cv::Vec3b)image1.at<cv::Vec3b>(yi1, xi1);
		const cv::Vec3b &color2 = (cv::Vec3b)image2.at<cv::Vec3b>(yi2, xi2);
		const cv::Vec3b color = 0.5 * (color1 + color2);

		const int blue = color[0];
		const int green = color[1];
		const int red = color[2];

		out_file << point3d.at<double>(0) << " "
			<< point3d.at<double>(1) << " "
			<< point3d.at<double>(2) << " "
			<< blue << " "
			<< green << " "
			<< red << "\n";

		src_inliers.emplace_back(cv::KeyPoint());
		dst_inliers.emplace_back(cv::KeyPoint());
		inlier_matches.emplace_back(cv::DMatch());

		// Construct the cv::Matches std::vector for the drawing
		src_inliers.back().pt = source_points[inliers[inl_idx]] / resize_by;
		dst_inliers.back().pt = destination_points[inliers[inl_idx]] / resize_by;
		inlier_matches.back().queryIdx = src_inliers.size() - 1;
		inlier_matches.back().trainIdx = dst_inliers.size() - 1;
	}
	out_file.close();

	cv::Mat out_image;
	drawMatches(tmp_image1, src_inliers, tmp_image2, dst_inliers, inlier_matches, out_image);

	cv::imshow("Matches", out_image);
	cv::waitKey(0);

	return 0;
}

void printTimes(const std::chrono::time_point<std::chrono::system_clock> &start,
	const std::chrono::time_point<std::chrono::system_clock> &end,
	const std::string &message)
{
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout << "Processing time of the %s was %f secs." << message.c_str() << elapsed_seconds.count() <<endl;
}

void getProjectionMatrices(
	const cv::Mat &essential_matrix_, 
	const cv::Mat &K1_, 
	const cv::Mat &K2_, 
	const cv::Mat &src_point_, 
	const cv::Mat &dst_point_, 
	cv::Mat &projection_1_,
	cv::Mat &projection_2_)
{
	// ****************************************************
	// Calculate the projection matrix of the first camera
	// ****************************************************
	projection_1_ = K1_ * cv::Mat::eye(3, 4, CV_64F);

	// projection_1_.create(3, 4, CV_64F);
	// cv::Mat rotation_1 = cv::Mat::eye(3, 3, CV_64F);
	// cv::Mat translation_1 = cv::Mat::zeros(3, 1, CV_64F);

	// ****************************************************
	// Calculate the projection matrix of the second camera
	// ****************************************************

	// Decompose the essential matrix (extrinsic camera parameters)
	cv::Mat rotation_1, rotation_2, translation;

	cv::SVD svd(essential_matrix_, cv::SVD::FULL_UV);
	// It gives matrices U D Vt

	if (cv::determinant(svd.u) < 0)
		svd.u.col(2) *= -1;
	if (cv::determinant(svd.vt) < 0)
		svd.vt.row(2) *= -1;

	cv::Mat w = (cv::Mat_<double>(3, 3) << 0, -1, 0,
		1, 0, 0,
		0, 0, 1);

	rotation_1 = svd.u * w * svd.vt;
	rotation_2 = svd.u * w.t() * svd.vt;
	translation = svd.u.col(2) / cv::norm(svd.u.col(2));

	// The possible solutions:
	// (rotation_1, translation)
	// (rotation_2, translation)
	// (rotation_1, -translation)
	// (rotation_2, -translation)

	cv::Mat P21 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_1.at<double>(0, 0), rotation_1.at<double>(0, 1), rotation_1.at<double>(0, 2), translation.at<double>(0),
		rotation_1.at<double>(1, 0), rotation_1.at<double>(1, 1), rotation_1.at<double>(1, 2), translation.at<double>(1),
		rotation_1.at<double>(2, 0), rotation_1.at<double>(2, 1), rotation_1.at<double>(2, 2), translation.at<double>(2));
	cv::Mat P22 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_2.at<double>(0, 0), rotation_2.at<double>(0, 1), rotation_2.at<double>(0, 2), translation.at<double>(0),
		rotation_2.at<double>(1, 0), rotation_2.at<double>(1, 1), rotation_2.at<double>(1, 2), translation.at<double>(1),
		rotation_2.at<double>(2, 0), rotation_2.at<double>(2, 1), rotation_2.at<double>(2, 2), translation.at<double>(2));
	cv::Mat P23 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_1.at<double>(0, 0), rotation_1.at<double>(0, 1), rotation_1.at<double>(0, 2), -translation.at<double>(0),
		rotation_1.at<double>(1, 0), rotation_1.at<double>(1, 1), rotation_1.at<double>(1, 2), -translation.at<double>(1),
		rotation_1.at<double>(2, 0), rotation_1.at<double>(2, 1), rotation_1.at<double>(2, 2), -translation.at<double>(2));
	cv::Mat P24 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_2.at<double>(0, 0), rotation_2.at<double>(0, 1), rotation_2.at<double>(0, 2), -translation.at<double>(0),
		rotation_2.at<double>(1, 0), rotation_2.at<double>(1, 1), rotation_2.at<double>(1, 2), -translation.at<double>(1),
		rotation_2.at<double>(2, 0), rotation_2.at<double>(2, 1), rotation_2.at<double>(2, 2), -translation.at<double>(2));

	std::vector< const cv::Mat* > Ps = { &P21, &P22, &P23, &P24 };
	double minDistance = std::numeric_limits<double>::max();

	for (const auto& P2ptr : Ps)
	{
		const cv::Mat& P1 = projection_1_;
		const cv::Mat& P2 = *P2ptr;

		// Estimate the 3D coordinates of a point correspondence
		cv::Mat point3d;
		linearTriangulation(P1,
			P2,
			src_point_,
			dst_point_,
			point3d);
		point3d.push_back(1.0);

		cv::Mat projection1 =
			P1 * point3d;
		cv::Mat projection2 =
			P2 * point3d;

		if (projection1.at<double>(2) < 0 ||
			projection2.at<double>(2) < 0)
			continue;

		projection1 = projection1 / projection1.at<double>(2);
		projection2 = projection2 / projection2.at<double>(2);

		// cv::norm(projection1 - src_point_)
		double dx1 = projection1.at<double>(0) - src_point_.at<double>(0);
		double dy1 = projection1.at<double>(1) - src_point_.at<double>(1);
		double squaredDist1 = dx1 * dx1 + dy1 * dy1;

		// cv::norm(projection2 - dst_point_)
		double dx2 = projection2.at<double>(0) - dst_point_.at<double>(0);
		double dy2 = projection2.at<double>(1) - dst_point_.at<double>(1);
		double squaredDist2 = dx2 * dx2 + dy2 * dy2;

		if (squaredDist1 + squaredDist2 < minDistance)
		{
			minDistance = squaredDist1 + squaredDist2;
			projection_2_ = P2.clone();
		}
	}
}

void linearTriangulation(
	const cv::Mat &projection_1_,
	const cv::Mat &projection_2_,
	const cv::Mat &src_point_,
	const cv::Mat &dst_point_,
	cv::Mat &point3d_)
{
	cv::Mat A(4, 3, CV_64F);
	cv::Mat b(4, 1, CV_64F);

	{
		const double
			& px = src_point_.at<double>(0),
			& py = src_point_.at<double>(1),
			& p1 = projection_1_.at<double>(0, 0),
			& p2 = projection_1_.at<double>(0, 1),
			& p3 = projection_1_.at<double>(0, 2),
			& p4 = projection_1_.at<double>(0, 3),
			& p5 = projection_1_.at<double>(1, 0),
			& p6 = projection_1_.at<double>(1, 1),
			& p7 = projection_1_.at<double>(1, 2),
			& p8 = projection_1_.at<double>(1, 3),
			& p9 = projection_1_.at<double>(2, 0),
			& p10 = projection_1_.at<double>(2, 1),
			& p11 = projection_1_.at<double>(2, 2),
			& p12 = projection_1_.at<double>(2, 3);

		A.at<double>(0, 0) = px * p9 - p1;
		A.at<double>(0, 1) = px * p10 - p2;
		A.at<double>(0, 2) = px * p11 - p3;
		A.at<double>(1, 0) = py * p9 - p5;
		A.at<double>(1, 1) = py * p10 - p6;
		A.at<double>(1, 2) = py * p11 - p7;

		b.at<double>(0) = p4 - px * p12;
		b.at<double>(1) = p8 - py * p12;
	}

	{
		const double
			& px = dst_point_.at<double>(0),
			& py = dst_point_.at<double>(1),
			& p1 = projection_2_.at<double>(0, 0),
			& p2 = projection_2_.at<double>(0, 1),
			& p3 = projection_2_.at<double>(0, 2),
			& p4 = projection_2_.at<double>(0, 3),
			& p5 = projection_2_.at<double>(1, 0),
			& p6 = projection_2_.at<double>(1, 1),
			& p7 = projection_2_.at<double>(1, 2),
			& p8 = projection_2_.at<double>(1, 3),
			& p9 = projection_2_.at<double>(2, 0),
			& p10 = projection_2_.at<double>(2, 1),
			& p11 = projection_2_.at<double>(2, 2),
			& p12 = projection_2_.at<double>(2, 3);

		A.at<double>(2, 0) = px * p9 - p1;
		A.at<double>(2, 1) = px * p10 - p2;
		A.at<double>(2, 2) = px * p11 - p3;
		A.at<double>(3, 0) = py * p9 - p5;
		A.at<double>(3, 1) = py * p10 - p6;
		A.at<double>(3, 2) = py * p11 - p7;

		b.at<double>(2) = p4 - px * p12;
		b.at<double>(3) = p8 - py * p12;
	}

	//cv::Mat x = (A.t() * A).inv() * A.t() * b;
	point3d_ = A.inv(cv::DECOMP_SVD) * b;
}

void normalizePoints(
	const std::vector<cv::Point2d> &input_source_points_,
	const std::vector<cv::Point2d> &input_destination_points_,
	std::vector<cv::Point2d> &output_source_points_,
	std::vector<cv::Point2d> &output_destination_points_,
	cv::Mat &T1_, 
	cv::Mat &T2_)
{
	// The objective: normalize the point set in each image by
	// translating the mass point to the origin and
	// the average distance from the mass point to be sqrt(2).
	T1_ = cv::Mat::eye(3, 3, CV_64F);
	T2_ = cv::Mat::eye(3, 3, CV_64F);

	const size_t pointNumber = input_source_points_.size();
	output_source_points_.resize(pointNumber);
	output_destination_points_.resize(pointNumber);

	// Calculate the mass point
	cv::Point2d mass1(0, 0), mass2(0, 0);

	for (auto i = 0; i < pointNumber; ++i)
	{
		mass1 = mass1 + input_source_points_[i];
		mass2 = mass2 + input_destination_points_[i];
	}

	mass1 = mass1 * (1.0 / pointNumber);
	mass2 = mass2 * (1.0 / pointNumber);

	// Translate the point clouds to the origin
	for (auto i = 0; i < pointNumber; ++i)
	{
		output_source_points_[i] = input_source_points_[i] - mass1;
		output_destination_points_[i] = input_destination_points_[i] - mass2;
	}

	// Calculate the average distances of the points from the origin
	double avgDistance1 = 0.0,
		avgDistance2 = 0.0;

	for (auto i = 0; i < pointNumber; ++i)
	{
		avgDistance1 += cv::norm(output_source_points_[i]);
		avgDistance2 += cv::norm(output_destination_points_[i]);
	}

	avgDistance1 /= pointNumber;
	avgDistance2 /= pointNumber;

	const double multiplier1 =
		sqrt(2) / avgDistance1;
	const double multiplier2 =
		sqrt(2) / avgDistance2;

	for (auto i = 0; i < pointNumber; ++i)
	{
		output_source_points_[i] *= multiplier1;
		output_destination_points_[i] *= multiplier2;
	}

	/*cv::Mat scaling1 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat translation1 = cv::Mat::eye(3, 3, CV_64F);
	scaling1.at<double>(0, 0) = multiplier1;
	scaling1.at<double>(1, 1) = multiplier1;
	translation1.at<double>(0, 2) = -mass1.x;
	translation1.at<double>(0, 1) = -mass1.y;
	T1_ = scaling1 * translation1;

	cv::Mat scaling2 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat translation2 = cv::Mat::eye(3, 3, CV_64F);
	scaling2.at<double>(0, 0) = multiplier2;
	scaling2.at<double>(1, 1) = multiplier2;
	translation2.at<double>(0, 2) = -mass2.x;
	translation2.at<double>(0, 1) = -mass2.y;
	T2_ = scaling2 * translation2;*/

	T1_.at<double>(0, 0) = multiplier1;
	T1_.at<double>(1, 1) = multiplier1;
	T1_.at<double>(0, 2) = -multiplier1 * mass1.x;
	T1_.at<double>(1, 2) = -multiplier1 * mass1.y;

	T2_.at<double>(0, 0) = multiplier2;
	T2_.at<double>(1, 1) = multiplier2;
	T2_.at<double>(0, 2) = -multiplier2 * mass2.x;
	T2_.at<double>(1, 2) = -multiplier2 * mass2.y;

	// Reason: T1_ * point = (Scaling1 * Translation1) * point = Scaling1 * (Translation1 * point)
}


// Visualize the effect of the point normalization
cv::Mat checkEffectOfNormalization(const std::vector<cv::Point2d> &source_points_,  // Points in the first image 
	const std::vector<cv::Point2d> &destination_points_,   // Points in the second image
	const std::vector<cv::Point2d> &normalized_source_points_,  // Normalized points in the first image 
	const std::vector<cv::Point2d> &normalized_destination_points_, // Normalized points in the second image
	const cv::Mat &T1_, // Normalizing transforcv::Mation in the first image
	const cv::Mat &T2_, // Normalizing transforcv::Mation in the second image
	const std::vector<size_t> &inliers_) // The inliers of the fundamental matrix
{
	std::vector<cv::Point2d> source_inliers;
	std::vector<cv::Point2d> destination_inliers;
	std::vector<cv::Point2d> normalized_source_inliers;
	std::vector<cv::Point2d> normalized_destination_inliers;

	for (const auto& inlierIdx : inliers_)
	{
		source_inliers.emplace_back(source_points_[inlierIdx]);
		destination_inliers.emplace_back(destination_points_[inlierIdx]);
		normalized_source_inliers.emplace_back(normalized_source_points_[inlierIdx]);
		normalized_destination_inliers.emplace_back(normalized_destination_points_[inlierIdx]);
	}

	// Estimate the fundamental matrix from the original points
	cv::Mat unnnormalized_fundamental_matrix(3, 3, CV_64F);
	getFundamentalMatrixLSQ(
		source_inliers,
		destination_inliers,
		unnnormalized_fundamental_matrix);
	
	// Estimate the fundamental matrix from the normalized points
	cv::Mat normalized_fundamental_matrix(3, 3, CV_64F);
	getFundamentalMatrixLSQ(
		normalized_source_inliers,
		normalized_destination_inliers,
		normalized_fundamental_matrix);

	normalized_fundamental_matrix = 
		T2_.t() * normalized_fundamental_matrix * T1_; // Denormalize the fundamental matrix

	// Calcualting the error of the normalized and unnormalized fundamental matrices.
	double error1 = 0, error2 = 0;

	for (size_t i = 0; i < inliers_.size(); ++i)
	{
		// Symmetric epipolar distance   
		cv::Mat pt1 = (cv::Mat_<double>(3, 1) << source_inliers[i].x, source_inliers[i].y, 1);
		cv::Mat pt2 = (cv::Mat_<double>(3, 1) << destination_inliers[i].x, destination_inliers[i].y, 1);

		// Calculate the error
		cv::Mat lL = unnnormalized_fundamental_matrix.t() * pt2;
		cv::Mat lR = unnnormalized_fundamental_matrix * pt1;

		// Calculate the distance of point pt1 from lL
		const double
			& aL = lL.at<double>(0),
			& bL = lL.at<double>(1),
			& cL = lL.at<double>(2);

		double tL = abs(aL * source_inliers[i].x + bL * source_inliers[i].y + cL);
		double dL = sqrt(aL * aL + bL * bL);
		double distanceL = tL / dL;

		// Calculate the distance of point pt2 from lR
		const double
			& aR = lR.at<double>(0),
			& bR = lR.at<double>(1),
			& cR = lR.at<double>(2);

		double tR = abs(aR * destination_inliers[i].x + bR * destination_inliers[i].y + cR);
		double dR = sqrt(aR * aR + bR * bR);
		double distanceR = tR / dR;

		double dist = 0.5 * (distanceL + distanceR);
		error1 += dist;
	}

	for (size_t i = 0; i < inliers_.size(); ++i)
	{
		// Symmetric epipolar distance   
		cv::Mat pt1 = (cv::Mat_<double>(3, 1) << source_inliers[i].x, source_inliers[i].y, 1);
		cv::Mat pt2 = (cv::Mat_<double>(3, 1) << destination_inliers[i].x, destination_inliers[i].y, 1);

		// Calculate the error
		cv::Mat lL = normalized_fundamental_matrix.t() * pt2;
		cv::Mat lR = normalized_fundamental_matrix * pt1;

		// Calculate the distance of point pt1 from lL
		const double
			& aL = lL.at<double>(0),
			& bL = lL.at<double>(1),
			& cL = lL.at<double>(2);

		double tL = abs(aL * source_inliers[i].x + bL * source_inliers[i].y + cL);
		double dL = sqrt(aL * aL + bL * bL);
		double distanceL = tL / dL;

		// Calculate the distance of point pt2 from lR
		const double
			& aR = lR.at<double>(0),
			& bR = lR.at<double>(1),
			& cR = lR.at<double>(2);

		double tR = abs(aR * destination_inliers[i].x + bR * destination_inliers[i].y + cR);
		double dR = sqrt(aR * aR + bR * bR);
		double distanceR = tR / dR;

		double dist = 0.5 * (distanceL + distanceR);
		error2 += dist;
	}

	error1 = error1 / inliers_.size();
	error2 = error2 / inliers_.size();

	cout << "Error of the unnormalized fundamental matrix is %f px. " << error1 << endl;
	cout<< "Error of the normalized fundamental matrix is %f px. " << error2 <<endl;

	return normalized_fundamental_matrix;
}

int getIterationNumber(int point_number_,
	int inlier_number_,
	int sample_size_,
	double confidence_)
{
	const double inlier_ratio = 
		static_cast<double>(inlier_number_) / point_number_;

	static const double log1 = log(1.0 - confidence_);
	const double log2 = log(1.0 - pow(inlier_ratio, sample_size_));
	
	const int k = log1 / log2;
	if (k < 0)
		return std::numeric_limits<int>::max();
	return k;
}

void ransacFundamentalMatrix(
	const std::vector<cv::Point2d> &input_src_points_,
	const std::vector<cv::Point2d> &input_destination_points_,
	const std::vector<cv::Point2d> &normalized_input_src_points_,
	const std::vector<cv::Point2d> &normalized_input_destination_points_,
	const cv::Mat &T1_,
	const cv::Mat &T2_,
	cv::Mat &fundamental_matrix_, 
	std::vector<size_t> &inliers_,
	double confidence_, 
	double threshold_)
{
	// The so-far-the-best fundamental matrix
	cv::Mat best_fundamental_matrix;
	// The number of correspondences
	const size_t point_number = input_src_points_.size();
	
	// Initializing the index pool from which the minimal samples are selected
	std::vector<size_t> index_pool(point_number);
	for (size_t i = 0; i < point_number; ++i)
		index_pool[i] = i;

	// The size of a minimal sample
	constexpr size_t sample_size = 8;
	// The minimal sample
	size_t *mss = new size_t[sample_size];

	size_t maximum_iterations = std::numeric_limits<int>::max(), // The maximum number of iterations set adaptively when a new best model is found
		iteration_limit = 5000, // A strict iteration limit which mustn't be exceeded
		iteration = 0; // The current iteration number

	std::vector<cv::Point2d> source_points(sample_size), 
		destination_points(sample_size);

	while (iteration++ < MIN(iteration_limit, maximum_iterations))
	{
		for (auto sample_idx = 0; sample_idx < sample_size; ++sample_idx)
		{
			// Select a random index from the pool
			const size_t idx = round((rand() / (double)RAND_MAX) * (index_pool.size() - 1));
			mss[sample_idx] = index_pool[idx];
			index_pool.erase(index_pool.begin() + idx);

			// Put the selected correspondences into the point containers
			const size_t point_idx = mss[sample_idx];
			source_points[sample_idx] = normalized_input_src_points_[point_idx];
			destination_points[sample_idx] = normalized_input_destination_points_[point_idx];
		}

		// Estimate fundamental matrix
		cv::Mat fundamental_matrix(3, 3, CV_64F);
		getFundamentalMatrixLSQ(source_points, destination_points, fundamental_matrix);
		fundamental_matrix = T2_.t() * fundamental_matrix * T1_; // Denormalize the fundamental matrix

		// Count the inliers
		std::vector<size_t> inliers;
		const double* p = (double *)fundamental_matrix.data;
		for (int i = 0; i < input_src_points_.size(); ++i)
		{
			// Symmetric epipolar distance   
			cv::Mat pt1 = (cv::Mat_<double>(3, 1) << input_src_points_[i].x, input_src_points_[i].y, 1);
			cv::Mat pt2 = (cv::Mat_<double>(3, 1) << input_destination_points_[i].x, input_destination_points_[i].y, 1);

			// Calculate the error
			cv::Mat lL = fundamental_matrix.t() * pt2;
			cv::Mat lR = fundamental_matrix * pt1;

			// Calculate the distance of point pt1 from lL
			const double
				& aL = lL.at<double>(0),
				& bL = lL.at<double>(1),
				& cL = lL.at<double>(2);

			double tL = abs(aL * input_src_points_[i].x + bL * input_src_points_[i].y + cL);
			double dL = sqrt(aL * aL + bL * bL);
			double distanceL = tL / dL;

			// Calculate the distance of point pt2 from lR
			const double
				& aR = lR.at<double>(0),
				& bR = lR.at<double>(1),
				& cR = lR.at<double>(2);

			double tR = abs(aR * input_destination_points_[i].x + bR * input_destination_points_[i].y + cR);
			double dR = sqrt(aR * aR + bR * bR);
			double distanceR = tR / dR;

			double dist = 0.5 * (distanceL + distanceR);

			if (dist < threshold_)
				inliers.push_back(i);
		}

		// Update if the new model is better than the previous so-far-the-best.
		if (inliers_.size() < inliers.size())
		{
			// Update the set of inliers
			inliers_.swap(inliers);
			inliers.clear();
			inliers.resize(0);
			// Update the model parameters
			best_fundamental_matrix = fundamental_matrix;
			// Update the iteration number
			maximum_iterations = getIterationNumber(point_number,
				inliers_.size(),
				sample_size,
				confidence_);
		}

		// Put back the selected points to the pool
		for (size_t i = 0; i < sample_size; ++i)
			index_pool.push_back(mss[i]);
	}

	delete[] mss;

	fundamental_matrix_ = best_fundamental_matrix;
}

void getFundamentalMatrixLSQ(
	const std::vector<cv::Point2d> &source_points_,
	const std::vector<cv::Point2d> &destination_points_,
	cv::Mat &fundamental_matrix_)
{
	const size_t pointNumber = source_points_.size();
	cv::Mat A(pointNumber, 9, CV_64F);

	for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
	{
		const double
			&x1 = source_points_[pointIdx].x,
			&y1 = source_points_[pointIdx].y,
			&x2 = destination_points_[pointIdx].x,
			&y2 = destination_points_[pointIdx].y;

		A.at<double>(pointIdx, 0) = x1 * x2;
		A.at<double>(pointIdx, 1) = x2 * y1;
		A.at<double>(pointIdx, 2) = x2;
		A.at<double>(pointIdx, 3) = y2 * x1;
		A.at<double>(pointIdx, 4) = y2 * y1;
		A.at<double>(pointIdx, 5) = y2;
		A.at<double>(pointIdx, 6) = x1;
		A.at<double>(pointIdx, 7) = y1;
		A.at<double>(pointIdx, 8) = 1;
	}

	cv::Mat evals, evecs;
	cv::Mat AtA = A.t() * A;
	cv::eigen(AtA, evals, evecs);

	cv::Mat x = evecs.row(evecs.rows - 1); // x = [f1 f2 f3 f4 f5 f6 f7 f8 f9]
	fundamental_matrix_.create(3, 3, CV_64F);
	memcpy(fundamental_matrix_.data, x.data, sizeof(double) * 9);
}

void readFeatures(
	const std::string fileName,
	std::vector<cv::Point2d> &source_points_,
	std::vector<cv::Point2d> &destination_points_)
{
	cout << fileName << endl;
	std::ifstream input(fileName);
	std::string line;
	int c = 0;
	std::vector<double> source_U1;
	std::vector<double> source_V1;
	std::vector<double> source_U2;
	std::vector<double> source_V2;
	while (std::getline(input, line))
	{
		//cout << line << endl;
		std::istringstream iss(line);
		std::string item;
		while (std::getline(iss, item, ' ')) {
			if (c == 0)
			{
				source_U1.push_back(std::stod(item));
			}
			else if (c == 1) {
				source_V1.push_back(std::stod(item));

			}
			else if (c == 2) {
				source_U2.push_back(std::stod(item));

			}
			else if (c == 3) {
				source_V2.push_back(std::stod(item));

			}
			else { cout << "ERROR in READING FILE" << endl;};
		}
		
		c++;
	}

	cout << source_U1.size() << endl;
	cout << source_V1.size() << endl;
	cout << source_U2.size() << endl;
	cout << source_V2.size() << endl;

	for (size_t i = 0; i < source_U1.size(); i++)
	{
		source_points_.push_back(cv::Point2d(source_U1[i], source_V1[i]));
		destination_points_.push_back(cv::Point2d(source_U2[i], source_V2[i]));
	}
	//printf("Detected correspondence number: %d\n", source_points_.size());
}