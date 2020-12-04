#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv\cv.hpp>
#include <opencv\highgui.h>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <time.h>

#include "MatrixReaderWriter.h"

using namespace cv;
using namespace std;

int getIterationNumber(int point_number_, // The number of points
    int inlier_number_, // The number of inliers
    int sample_size_, // The sample size
    double confidence_); // The required confidence


void normalizePoints(
    const std::vector<cv::Point2d>& input_source_points_, // Points in the source image
    const std::vector<cv::Point2d>& input_destination_points_, // Points in the destination image
    std::vector<cv::Point2d>& output_source_points_, // Normalized points in the source image
    std::vector<cv::Point2d>& output_destination_points_, // Normalized points in the destination image
    cv::Mat& T1_, // Normalizing transformation in the source image
    cv::Mat& T2_); // Normalizing transformation in the destination image


Mat calcHomography(vector<pair<Point2d, Point2d> > pointPairs) {
    const int ptsNum = pointPairs.size();
    Mat A(2 * ptsNum, 9, CV_64F);
    for (int i = 0; i < ptsNum; i++) {
        double u1 = pointPairs[i].first.x;
        double v1 = pointPairs[i].first.y;

        double u2 = pointPairs[i].second.x;
        double v2 = pointPairs[i].second.y;

        A.at<double>(2 * i, 0) = u1;
        A.at<double>(2 * i, 1) = v1;
        A.at<double>(2 * i, 2) = 1.0;
        A.at<double>(2 * i, 3) = 0.0;
        A.at<double>(2 * i, 4) = 0.0;
        A.at<double>(2 * i, 5) = 0.0;
        A.at<double>(2 * i, 6) = -u2 * u1;
        A.at<double>(2 * i, 7) = -u2 * v1;
        A.at<double>(2 * i, 8) = -u2;

        A.at<double>(2 * i + 1, 0) = 0.0;
        A.at<double>(2 * i + 1, 1) = 0.0;
        A.at<double>(2 * i + 1, 2) = 0.0;
        A.at<double>(2 * i + 1, 3) = u1;
        A.at<double>(2 * i + 1, 4) = v1;
        A.at<double>(2 * i + 1, 5) = 1.0;
        A.at<double>(2 * i + 1, 6) = -v2 * u1;
        A.at<double>(2 * i + 1, 7) = -v2 * v1;
        A.at<double>(2 * i + 1, 8) = -v2;

    }

    Mat eVecs(9, 9, CV_64F), eVals(9, 9, CV_64F);
    //std::cout << A << endl;
    eigen(A.t() * A, eVals, eVecs);

    //std::cout << eVals << endl;
    //std::cout << eVecs << endl;


    Mat H(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) H.at<double>(i / 3, i % 3) = eVecs.at<double>(8, i);

    //std::cout << H << endl;

    //Normalize:
    H = H * (1.0 / H.at<double>(2, 2));
    //std::cout << H << endl;

    return H;
}


Mat ransacHomographyMatrix(
    const std::vector<cv::Point2d>& input_src_points_,
    const std::vector<cv::Point2d>& input_destination_points_,
    const std::vector<cv::Point2d>& normalized_input_src_points_,
    const std::vector<cv::Point2d>& normalized_input_destination_points_,
    const cv::Mat& T1_,
    const cv::Mat& T2_,
    double confidence_,
    double threshold_)
{
    cv::Mat bestH;
    std::vector<size_t> inliers_;
    // The number of correspondences
    const size_t point_number = input_src_points_.size();

    // Initializing the index pool from which the minimal samples are selected
    std::vector<size_t> index_pool(point_number);
    for (size_t i = 0; i < point_number; ++i)
        index_pool[i] = i;

    // The size of a minimal sample
    constexpr size_t sample_size = 4;
    // The minimal sample
    size_t* mss = new size_t[sample_size];

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

        // Estimate homography matrix
        std::vector<pair<Point2d, Point2d>> samplePairs;
        samplePairs.clear();
        samplePairs.resize(0);
        for (int i = 0; i < source_points.size(); ++i) {
            pair<Point2d, Point2d> currPts;
            currPts.first = source_points[i];
            currPts.second = destination_points[i];
            samplePairs.push_back(currPts);
        }
        cv::Mat H(3,3, CV_64F);
        H = calcHomography(samplePairs);        

        H = T2_.inv() * H * T1_;

        // Count the inliers
        std::vector<size_t> inliers;
        for (int i = 0; i < input_src_points_.size(); ++i)
        {  
            cv::Mat pt1 = (cv::Mat_<double>(3, 1) << input_src_points_[i].x, input_src_points_[i].y, 1);
            cv::Mat pt2 = (cv::Mat_<double>(3, 1) << input_destination_points_[i].x, input_destination_points_[i].y, 1);
            
            // Calculate the error
            cv::Mat p2Calc = H * pt1;

            double error = norm(pt2, p2Calc, NORM_L2);
            //cout << "Erro: " << error << endl;

            if (error < threshold_)
                inliers.push_back(i);
        }

        // Update if the new model is better than the previous so-far-the-best.
        std::cout <<"Inliner size:" << inliers.size() << endl;
        if (inliers_.size() < inliers.size())
        {
            // Update the set of inliers
            inliers_.swap(inliers);
            inliers.clear();
            inliers.resize(0);
            // Update the model parameters
            bestH = H;
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

    std::vector<pair<Point2d, Point2d>> bestInliners;
    for (auto ind : inliers_) {
        pair<Point2d, Point2d> currPts;
        currPts.first = normalized_input_src_points_[ind];
        currPts.second = normalized_input_destination_points_[ind];
        bestInliners.push_back(currPts);
    }

    Mat reH = calcHomography(bestInliners);
    std::cout << "Best Inliner number: " << bestInliners.size() << endl;
    return T2_.inv() * reH * T1_;
}

//Tranformation of images
void transformImage(Mat origImg, Mat& newImage, Mat tr, bool isPerspective) {
    Mat invTr = tr.inv();
    const int WIDTH = origImg.cols;
    const int HEIGHT = origImg.rows;

    const int newWIDTH = newImage.cols;
    const int newHEIGHT = newImage.rows;



    for (int x = 0; x < newWIDTH; x++) for (int y = 0; y < newHEIGHT; y++) {
        Mat pt(3, 1, CV_64F);
        pt.at<double>(0, 0) = x;
        pt.at<double>(1, 0) = y;
        pt.at<double>(2, 0) = 1.0;

        Mat ptTransformed = invTr * pt;
        //        std::cout <<pt <<endl;
        //        std::cout <<invTr <<endl;
        //        std::cout <<ptTransformed <<endl;
        if (isPerspective) ptTransformed = (1.0 / ptTransformed.at<double>(2, 0)) * ptTransformed;

        int newX = round(ptTransformed.at<double>(0, 0));
        int newY = round(ptTransformed.at<double>(1, 0));

        if ((newX >= 0) && (newX < WIDTH) && (newY >= 0) && (newY < HEIGHT)) newImage.at<Vec3b>(y, x) = origImg.at<Vec3b>(newY, newX);

        //        printf("x:%d y:%d newX:%d newY:%d\n",x,y,newY,newY);
    }
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cout << " Usage: point_file img1 img2" << endl;
        return -1;
    }

    MatrixReaderWriter mtxrw(argv[1]);

    if ((mtxrw.rowNum != 4) || (mtxrw.columnNum == 0))
    {
        std::cout << "Point file format error" << std::endl;
        return -1;
    }

    srand((unsigned)time(NULL));

    int r = mtxrw.rowNum;
    int c = mtxrw.columnNum;

    ///
    std::vector<cv::Point2d> source_points;
    std::vector<cv::Point2d> destination_points;
    ///
    std::vector<pair<Point2d, Point2d>> pointPairs;


    //Convert the coordinates:
    for (int i = 0; i < mtxrw.columnNum; i++) {
        pair<Point2d, Point2d> currPts;
        currPts.first = Point2d((double)mtxrw.data[i], (double)mtxrw.data[c + i]);
        currPts.second = Point2d((double)mtxrw.data[2 * c + i], (double)mtxrw.data[3 * c + i]);
        ///
        source_points.push_back(currPts.first);
        destination_points.push_back(currPts.second);
        pointPairs.push_back(currPts);
    }

    //std::cout << source_points[0].x << " " << source_points[0].y << " " << destination_points[0].x << " " << destination_points[0].y;
    //return 0;

    std::vector<cv::Point2d> source_points_normalized;
    std::vector<cv::Point2d> destination_points_normalized;
    cv::Mat T1, T2;

    normalizePoints(source_points, // Points in the first image 
        destination_points,  // Points in the second image
        source_points_normalized,  // Normalized points in the first image
        destination_points_normalized, // Normalized points in the second image
        T1, // Normalizing transforcv::Mation in the first image
        T2); // Normalizing transforcv::Mation in the second image

    //std::cout << endl;
    //std::cout << source_points_normalized[0].x << " " << source_points_normalized[0].y << " " << destination_points_normalized[0].x << " " << destination_points_normalized[0].y;
    //return 0;


    //Mat H = RANSAC(pointPairsNormalized, 10, 100);
    Mat H = ransacHomographyMatrix(source_points,  // Points in the first image 
		destination_points,   // Points in the second image
        source_points_normalized,  // Normalized points in the first image 
        destination_points_normalized, // Normalized points in the second image
		T1, // Normalizing transforcv::Mation in the first image
		T2, // Normalizing transforcv::Mation in the second image
		0.99, // The required confidence in the results 
		20.0); // The inlier-outlier threshold; 

    //Mat H = OldcalcHomography(pointPairs);


    Mat image1;
    image1 = imread(argv[2]);   // Read the file

    if (!image1.data)                              // Check for invalid input
    {
        std::cout << "Could not open or find the image: " << argv[2] << std::endl;
        return -1;
    }

    Mat image2;
    image2 = imread(argv[3]);   // Read the file

    if (!image2.data)                              // Check for invalid input
    {
        std::cout << "Could not open or find the image: " << argv[3] << std::endl;
        return -1;
    }


    //    std::cout <<tr.inv() <<endl;

    Mat transformedImage = Mat::zeros(1.5 * image1.size().height, 2.0 * image1.size().width, image1.type());
    transformImage(image2, transformedImage, Mat::eye(3, 3, CV_64F), true);

    //    transformImage(image2,transformedImage,tr2,false);
    transformImage(image1, transformedImage, H, true);

    imwrite("res.png", transformedImage);

    namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display window", transformedImage);                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}

void normalizePoints(
    const std::vector<cv::Point2d>& input_source_points_,
    const std::vector<cv::Point2d>& input_destination_points_,
    std::vector<cv::Point2d>& output_source_points_,
    std::vector<cv::Point2d>& output_destination_points_,
    cv::Mat& T1_,
    cv::Mat& T2_)
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