#ifndef BINARAYFILTER_H
#define BINARAYFILTER_H
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>

using namespace cv;
class binaryFilter{
private:
    Mat memory;
public:
    binaryFilter(int rows, int columns){
        memory = zeroes()
    }

    Mat filterImage(const& input);
};

#endif // BINARAYFILTER_H
