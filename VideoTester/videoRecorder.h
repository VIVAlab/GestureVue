#ifndef VIDEORECORDER_H
#define VIDEORECORDER_H

#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "string"

using namespace std;
using namespace cv;

class videoRecorder{

    string filename = "VideoResults\\TestVideoSequences";

    VideoWriter output;

public:
    videoRecorder(int j){
        //do nothing!;
    }

    void startRecording(string vidIn, int width, int height){
        string fileToOpen = filename;
        fileToOpen += vidIn;
        fileToOpen +="_ResultVideo";
        fileToOpen += ".avi";
        output.open( fileToOpen, CV_FOURCC('D','I','V','X'), 30, cv::Size ( width,height), true );



    }

    int addFrame(Mat image){
        output.write ( image );


        return 1;
    }
    void close(){
        output.release();


    }





};

#endif // VIDEORECORDER_H
