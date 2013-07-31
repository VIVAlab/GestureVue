
#ifndef TUTORIALFUNCTIONS_H
#define TUTORIALFUNCTIONS_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "iostream";


void salt(cv::Mat &image, int n);
void snow(cv::Mat &image, int n);
void colorReduce(cv::Mat &image, int div = 64);
void colorReduce(const cv::Mat &image, cv::Mat &result, int div = 64);
void colorReduce2(cv::Mat &image, int div = 64);
void sharpen(const cv::Mat &image, cv::Mat &result, double centerScale);
void colourSkew(const cv::Mat &image, cv::Mat &result);
void equalize(const cv::Mat &image, cv::Mat &result);
#endif // TUTORIALFUNCTIONS_H
