#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include "MatrixReaderWriter.h"

using namespace cv;
using namespace std;


Mat calcHomography(vector<pair<Point2f, Point2f> > pointPairs) {
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

    Mat H = calcHomography(pointPairs);


    Mat image1;
    image1 = imread(argv[2]);   // Read the file

    if (!image1.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << argv[2] << std::endl;
        return -1;
    }

    Mat image2;
    image2 = imread(argv[3]);   // Read the file

    if (!image2.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << argv[3] << std::endl;
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