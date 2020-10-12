#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;

int main( )
{
    Mat src1;
    src1 = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
    namedWindow( "Original image", CV_WINDOW_AUTOSIZE );
    imshow( "Original image", src1 );

    Mat gray, edge, draw;
    cvCvtColor(src1, gray, CV_BGR2GRAY);


    cvCanny( gray, edge, 50, 150, 3);

    edge.convertTo(draw, CV_8U);
    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", draw);

    waitKey(0);                                       
    return 0;
} 