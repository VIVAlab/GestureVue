#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "iostream"
#include "fstream"
#include <string>
#include "handTracker.h"


#define video_number 77


//using namespace cv;
//using namespace std;
int main()
{
    handTracker tracker;
    ifstream inputFile("VideoList.txt");
    string inputLine;

    tracker.startExperiment();

    while(inputFile.good()){
        getline (inputFile,inputLine);
        cout << inputLine << endl;
        if(inputLine.length() < 2){
            break;
        }

        string fileName = "Videos/TestVideoSequences";

        stringstream videoStream;
        videoStream << fileName;
        videoStream <<inputLine;
        videoStream<<".avi";

        //VideoCapture cap(videoStream.str());
        cout<<videoStream.str()<<endl;


        //play Video

        Mat image;
        bool playing = true;
        tracker.initialize(inputLine);
        tracker.process(true);
//        while(playing){
//            playing = cap.read(image);
//            if(playing == false){
//                break;
//            }
//            imshow("Current Video", image);
//            waitKey(1);

//        }


    }
    inputFile.close();

    tracker.endExperiment();



//    while(){
//    VideoCapture cap(0);
//    Mat image;


////    while(cap.read(image)){





////            imshow("Processed Video", image);
////            int response = waitKey(1);

////    }
//    }






    return 1;
}
