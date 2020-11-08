#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    double b[5][5] = {
    { 1.96 , -6.49, -0.47, -7.20, -0.65},
    { -6.49,  3.80, -6.39,  1.50, -6.34},
    { -0.47, -6.39,  4.17, -1.51,  2.67},
    { -7.20,  1.50, -1.51,  5.70,  1.80},
    { -0.65, -6.34,  2.67,  1.80, -7.10}
    };

cv::Mat E, V;
cv::Mat M(5,5,CV_64FC1,b);
cv::eigen(M,E,V);

// eigenvalues sorted desc
for(int i=0; i < 5; i++)
        std::cout << E.at<double>(0,i) << " \t";


    return 0;
}