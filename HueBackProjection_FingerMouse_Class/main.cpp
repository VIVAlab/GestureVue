#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <math.h>
#include "iostream"
//#include "tools.h"
//#include "watershedSegmentation.h"
#include "tutorialfunctions.h"
#include "handcaptureutilities.h"

#include <opencv2/photo/photo.hpp>
#include <windows.h>
#include "handTracker.h"








int main(){
    handTracker h1;

    h1.initialize("Blah", false, false);

    h1.process();

    return 1;
}




