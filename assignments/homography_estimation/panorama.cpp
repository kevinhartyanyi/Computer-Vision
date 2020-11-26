#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <time.h>

#include "MatrixReaderWriter.h"

using namespace cv;
using namespace std;


Mat calcHomography(vector<pair<Point2f, Point2f> > pointPairs) {
    const int ptsNum = pointPairs.size();
    Mat A(2 * ptsNum, 9, CV_32F);

    //
    float u1Avg = 0;
    float v1Avg = 0;
    float u2Avg = 0;
    float v2Avg = 0;
    vector<float> u1Vec;
    vector<float> v1Vec;
    vector<float> u2Vec;
    vector<float> v2Vec;
    
    for (int i = 0; i < ptsNum; i++) {
        float u1 = pointPairs[i].first.x;
        float v1 = pointPairs[i].first.y;

        float u2 = pointPairs[i].second.x;
        float v2 = pointPairs[i].second.y;

        
        u1Vec.push_back(u1);
        v1Vec.push_back(v1);
        u2Vec.push_back(u2);
        v2Vec.push_back(v2);

        u1Avg += u1;
        v1Avg += v1;
        u2Avg += u2;
        v2Avg += v2;
    }
    //

    /*
    * Data normalization
    * → translation: origo should be at the center of gravity
    * → scale: spread should be set to √2
    */
    const double ptsNumDouble = static_cast<double>(ptsNum);
    u1Avg = u1Avg / ptsNumDouble;
    v1Avg = v1Avg / ptsNumDouble;
    u2Avg = u2Avg / ptsNumDouble;
    v2Avg = v2Avg / ptsNumDouble;

    float dist1 = 0;
    for (int i = 0; i < ptsNum; i++) {
        dist1 += sqrt(pow(u1Vec[i] - u1Avg, 2) + pow(v1Vec[i] - v1Avg, 2));
    }
    float s1 = sqrt(2) * ptsNum / dist1;

    float dist2 = 0;
    for (int i = 0; i < ptsNum; i++) {
        dist2 += sqrt(pow(u2Vec[i] - u2Avg, 2) + pow(v2Vec[i] - v2Avg, 2));
    }
    float s2 = sqrt(2) * ptsNum / dist2;

    Mat T1(3, 3, CV_32F);
    Mat T2(3, 3, CV_32F);

    // T1
    T1.at<float>(0, 0) = 1;
    T1.at<float>(1, 0) = 0;
    T1.at<float>(2, 0) = 0;

    T1.at<float>(0, 1) = 0;
    T1.at<float>(1, 1) = 1;
    T1.at<float>(2, 1) = 0;

    T1.at<float>(0, 2) = -u1Avg;
    T1.at<float>(1, 2) = -v1Avg;
    T1.at<float>(2, 2) = 1 / s1;

    T1 = s1 * T1;

    // T2
    T2.at<float>(0, 0) = 1;
    T2.at<float>(1, 0) = 0;
    T2.at<float>(2, 0) = 0;

    T2.at<float>(0, 1) = 0;
    T2.at<float>(1, 1) = 1;
    T2.at<float>(2, 1) = 0;

    T2.at<float>(0, 2) = -u2Avg;
    T2.at<float>(1, 2) = -v2Avg;
    T2.at<float>(2, 2) = 1 / s2;

    T2 = s2 * T2;
    //

    for (int i = 0; i < ptsNum; i++) {
        float u1 = pointPairs[i].first.x;
        float v1 = pointPairs[i].first.y;

        float u2 = pointPairs[i].second.x;
        float v2 = pointPairs[i].second.y;

        Mat p1(3, 1, CV_32F);
        Mat p2(3, 1, CV_32F);

        p1.at<float>(0, 0) = u1;
        p1.at<float>(1, 0) = v1;
        p1.at<float>(2, 0) = 1;

        p2.at<float>(0, 0) = u2;
        p2.at<float>(1, 0) = v2;
        p2.at<float>(2, 0) = 1;

        //
        Mat p1Hat(3, 1, CV_32F);
        Mat p2Hat(3, 1, CV_32F);

        p1Hat = T1 * p1;
        p2Hat = T2 * p2;

        float u1Hat = p1Hat.at<float>(0, 0);
        float v1Hat = p1Hat.at<float>(1, 0);

        float u2Hat = p2Hat.at<float>(0, 0);
        float v2Hat = p2Hat.at<float>(1, 0);


        A.at<float>(2 * i, 0) = u1Hat;
        A.at<float>(2 * i, 1) = v1Hat;
        A.at<float>(2 * i, 2) = 1.0f;
        A.at<float>(2 * i, 3) = 0.0f;
        A.at<float>(2 * i, 4) = 0.0f;
        A.at<float>(2 * i, 5) = 0.0f;
        A.at<float>(2 * i, 6) = -u2Hat * u1Hat;
        A.at<float>(2 * i, 7) = -u2Hat * v1Hat;
        A.at<float>(2 * i, 8) = -u2Hat;

        A.at<float>(2 * i + 1, 0) = 0.0f;
        A.at<float>(2 * i + 1, 1) = 0.0f;
        A.at<float>(2 * i + 1, 2) = 0.0f;
        A.at<float>(2 * i + 1, 3) = u1Hat;
        A.at<float>(2 * i + 1, 4) = v1Hat;
        A.at<float>(2 * i + 1, 5) = 1.0f;
        A.at<float>(2 * i + 1, 6) = -v2Hat * u1Hat;
        A.at<float>(2 * i + 1, 7) = -v2Hat * v1Hat;
        A.at<float>(2 * i + 1, 8) = -v2Hat;
    }

    //

    Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);
    cout << A << endl;
    eigen(A.t() * A, eVals, eVecs);

    cout << eVals << endl;
    cout << eVecs << endl;


    Mat HHat(3, 3, CV_32F);
    for (int i = 0; i < 9; i++) HHat.at<float>(i / 3, i % 3) = eVecs.at<float>(8, i);

    cout << HHat << endl;

    //Normalize:
    HHat = HHat * (1.0 / HHat.at<float>(2, 2));
    cout << "HHat: " << HHat << endl;


    Mat H(3, 3, CV_32F);
    H = T2.t() * HHat * T1;

    cout << "H: " << H << endl;

    return H;
}


Mat OldcalcHomography(vector<pair<Point2f, Point2f> > pointPairs) {
    const int ptsNum = pointPairs.size();
    Mat A(2 * ptsNum, 9, CV_32F);
    for (int i = 0; i < ptsNum; i++) {
        float u1 = pointPairs[i].first.x;
        float v1 = pointPairs[i].first.y;

        float u2 = pointPairs[i].second.x;
        float v2 = pointPairs[i].second.y;

        A.at<float>(2 * i, 0) = u1;
        A.at<float>(2 * i, 1) = v1;
        A.at<float>(2 * i, 2) = 1.0f;
        A.at<float>(2 * i, 3) = 0.0f;
        A.at<float>(2 * i, 4) = 0.0f;
        A.at<float>(2 * i, 5) = 0.0f;
        A.at<float>(2 * i, 6) = -u2 * u1;
        A.at<float>(2 * i, 7) = -u2 * v1;
        A.at<float>(2 * i, 8) = -u2;

        A.at<float>(2 * i + 1, 0) = 0.0f;
        A.at<float>(2 * i + 1, 1) = 0.0f;
        A.at<float>(2 * i + 1, 2) = 0.0f;
        A.at<float>(2 * i + 1, 3) = u1;
        A.at<float>(2 * i + 1, 4) = v1;
        A.at<float>(2 * i + 1, 5) = 1.0f;
        A.at<float>(2 * i + 1, 6) = -v2 * u1;
        A.at<float>(2 * i + 1, 7) = -v2 * v1;
        A.at<float>(2 * i + 1, 8) = -v2;

    }

    Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);
    cout << A << endl;
    eigen(A.t() * A, eVals, eVecs);

    cout << eVals << endl;
    cout << eVecs << endl;


    Mat H(3, 3, CV_32F);
    for (int i = 0; i < 9; i++) H.at<float>(i / 3, i % 3) = eVecs.at<float>(8, i);

    cout << H << endl;

    //Normalize:
    H = H * (1.0 / H.at<float>(2, 2));
    cout << H << endl;

    return H;
}


Mat RANSAC(const vector<pair<Point2f, Point2f> > pointPairs, double threshold, int iteration_number, bool log = false) {
    Mat bestH;
    std::vector<pair<Point2f, Point2f>> bestInliners;
    int bestInlinerError;
    constexpr int kSampleSize = 4;

    // 1.
    for (size_t i = 0; i < iteration_number; i++)
    {
        // The current sample
        std::vector<int> sample;
        for (size_t sampleIdx = 0; sampleIdx < kSampleSize; ++sampleIdx)
        {
            int r;
            do
            {
                r = round((pointPairs.size() - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
            } while (std::find(sample.begin(), sample.end(), r) != sample.end());
            sample.push_back(r);
        }

        std::vector<pair<Point2f, Point2f>> samplePairs;
        if(log) cout << "Random sample:" << endl;
        for (auto a : sample) {
            if (log) cout << a << endl;
            samplePairs.push_back(pointPairs[a]);
        }

        Mat H = OldcalcHomography(samplePairs);

        // Projection error
        std::vector<pair<Point2f, Point2f>> inliners;
        float errorSum = 0;
        for (int i = 0; i < pointPairs.size(); i++) {
            float u1 = pointPairs[i].first.x;
            float v1 = pointPairs[i].first.y;

            float u2 = pointPairs[i].second.x;
            float v2 = pointPairs[i].second.y;

            Mat p1(3, 1, CV_32F);
            p1.at<float>(0, 0) = u1;
            p1.at<float>(1, 0) = v1;
            p1.at<float>(2, 0) = 1;

            Mat p2(3, 1, CV_32F);
            p2.at<float>(0, 0) = u2;
            p2.at<float>(1, 0) = v2;
            p2.at<float>(2, 0) = 1;

            Mat p2Calc = H * p1;

            if (log) {
                cout << "Point " << i << endl;
                cout << p2 << endl;
                cout << "Point " << i << " Calc" << endl;
                cout << p2Calc << endl;
                cout << endl;
            }

            float error = norm(p2, p2Calc, NORM_L2);
            if (log) {
                cout << "Point " << i << " error: " << error << endl;
                cout << endl;
            }
            errorSum = errorSum + error;

            // Count inliners
            
            if (error < threshold) {
                inliners.push_back(pointPairs[i]);
            }     
        }
        if (log) {
            cout << "Error Sum: " << errorSum << endl;

            cout << "Inliner number: " << inliners.size() << endl;
        }
        // Check if it's better than the best
        if (inliners.size() > bestInliners.size())
        {
            // 2. Choose the H with the largest number of inliers
            bestH = H;
            bestInliners.swap(inliners);  
            bestInlinerError = errorSum;    
        }
    }      
    //cout << "Best Inliner number: " << bestInliners.size() << endl;
    //cout << "Best Inliner projection error: " << bestInlinerError << endl;
    //return bestH;

    // 3. Recompute H with the bestInliners.
    // TODO: Is this correct?
    Mat reH = OldcalcHomography(bestInliners);
    cout << "Best Inliner number: " << bestInliners.size() << endl;
    cout << "Best Inliner projection error: " << bestInlinerError << endl;
    return reH;
}

//Tranformation of images

void transformImage(Mat origImg, Mat& newImage, Mat tr, bool isPerspective) {
    Mat invTr = tr.inv();
    const int WIDTH = origImg.cols;
    const int HEIGHT = origImg.rows;

    const int newWIDTH = newImage.cols;
    const int newHEIGHT = newImage.rows;



    for (int x = 0; x < newWIDTH; x++) for (int y = 0; y < newHEIGHT; y++) {
        Mat pt(3, 1, CV_32F);
        pt.at<float>(0, 0) = x;
        pt.at<float>(1, 0) = y;
        pt.at<float>(2, 0) = 1.0;

        Mat ptTransformed = invTr * pt;
        //        cout <<pt <<endl;
        //        cout <<invTr <<endl;
        //        cout <<ptTransformed <<endl;
        if (isPerspective) ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;

        int newX = round(ptTransformed.at<float>(0, 0));
        int newY = round(ptTransformed.at<float>(1, 0));

        if ((newX >= 0) && (newX < WIDTH) && (newY >= 0) && (newY < HEIGHT)) newImage.at<Vec3b>(y, x) = origImg.at<Vec3b>(newY, newX);

        //        printf("x:%d y:%d newX:%d newY:%d\n",x,y,newY,newY);
    }
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        cout << " Usage: point_file img1 img2" << endl;
        return -1;
    }

    MatrixReaderWriter mtxrw(argv[1]);

    if ((mtxrw.rowNum != 4) || (mtxrw.columnNum == 0))
    {
        cout << "Point file format error" << std::endl;
        return -1;
    }

    srand((unsigned)time(NULL));

    int r = mtxrw.rowNum;
    int c = mtxrw.columnNum;

    //Convert the coordinates:
    vector<pair<Point2f, Point2f> > pointPairs;
    for (int i = 0; i < mtxrw.columnNum; i++) {
        pair<Point2f, Point2f> currPts;
        currPts.first = Point2f((float)mtxrw.data[i], (float)mtxrw.data[c + i]);
        currPts.second = Point2f((float)mtxrw.data[2 * c + i], (float)mtxrw.data[3 * c + i]);
        pointPairs.push_back(currPts);
    }

    Mat H = RANSAC(pointPairs, 10, 100);

    //Mat H = OldcalcHomography(pointPairs);


    Mat image1;
    image1 = imread(argv[2]);   // Read the file

    if (!image1.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image: " << argv[2] << std::endl;
        return -1;
    }

    Mat image2;
    image2 = imread(argv[3]);   // Read the file

    if (!image2.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image: " << argv[3] << std::endl;
        return -1;
    }


    //    cout <<tr.inv() <<endl;

    Mat transformedImage = Mat::zeros(1.5 * image1.size().height, 2.0 * image1.size().width, image1.type());
    transformImage(image2, transformedImage, Mat::eye(3, 3, CV_32F), true);

    //    transformImage(image2,transformedImage,tr2,false);
    transformImage(image1, transformedImage, H, true);

    imwrite("res.png", transformedImage);

    namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display window", transformedImage);                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}