#ifndef HANDCAPTUREUTILITIES_H
#define HANDCAPTUREUTILITIES_H

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <math.h>

using namespace cv;
Point midPoint(Point p1, Point p2);
int dist (Point p1, Point p2);
double angleBetween3Points(Point P1, Point P2, Point P3);
Point averageOfPoints(vector<Point> points);
void equalizeRGB(Mat &src);
void intensityMap(const cv::Mat &image, cv::Mat &result);

#endif // HANDCAPTUREUTILITIES_H
