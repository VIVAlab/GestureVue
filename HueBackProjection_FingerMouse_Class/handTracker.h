#ifndef HANDTRACKER_H
#define HANDTRACKER_H

//Designed By Pavel Popov
//Viva Lab 2013


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


//added for recording purposes
#include "fileEvaluator.h"
#include "videoRecorder.h"







//parameters used for the project
#define big_line_divisor 15
#define min_contour_size 140
#define min_number_of_contour_points 5
#define hbin_number 32
#define sbin_number 32
#define polyAprrox_epsilon 5.0

#define sharpening_parameter 5.0
#define angle_divisor 4.0
#define color_reduce_parameter 2.0

//Hand Scanning Parameters
#define minimumSaturation 65;
int thresholdCode = cv::THRESH_TOZERO;


#define big_line_threshold_scanning 15
#define min_number_of_big_lines 3
#define max_number_of_big_lines 8
#define min_successful_iterations 1

#define angle_divisor_scanning 4.0
#define tip_distance_multiplier 1.0
#define max_finger_factor 4.0
#define innerDist_radius_threshold 1.5
#define min_finger_radius_multiplier 0.8
#define max_finger_radius_multiplier 4.0
#define min_knuckle_radius_multiplier 1.1
#define consecutive_finger_angle_divisor 4.0

//Hand Tracking Parameters
#define big_line_threshold_tracking 10
#define innerDistFactor 5.5
#define palm_filter_angle_divisor 6.0
#define finger_angle_divisor_tracking 3.5
#define tip_width_fingerBase_range 3.5
#define tip_width_palmRadius_range 2.5
#define baseDist_palmRadius_range 3.0
#define fingerLength_palmRadius_divisor 5.0
#define fingerToCenter_minimum_range 1.05
#define proxDist_palmRadius_divisor 4.0
#define referenceRadius_range 2.0
#define min_contour_size_tracking 120

//Finger Mouse Parameters
#define index_click_margin_factor 0.8
#define index_doubleclick_margin_factor 0.65
#define thumb_click_margin_factor 0.15






//max_line_divisor_tracking

#define video_number 66








//enable low pass filter and other options
bool lowPassEnabled = false;
bool colorReduceEnabled = false;
bool equalizeImage = false;
bool useFingerMouse = false;
bool useVideoFeed = false;

bool useColorEqualization = false;

//ashkan test
bool useModifiedThreshold = false;

//Pavel Skin tone
bool useSkinToneThresholds = false;

//Used for intensity map use
bool useIntensityMap = true;

//all these options are bad
bool useIntensityMapScanFiltering = false;
bool useIntensityMapTracking = false;
bool useAddImages = false;

//don't use this option. Ever!!!!!
bool useIntensityAsBackProjection = false;

//Variables for min saturation
bool useMinSat = true;
cv::Mat saturationZone;

//ROI dimensions
int xdim = 30;
int ydim = 30;

using namespace cv;
using namespace std;






//Histogram Variables
// Quantize the hue to 30 levels
// and the saturation to 32 levels
int hbins = hbin_number, sbins = sbin_number;
int histSize[] = {hbins, sbins};
// hue varies from 0 to 179, see cvtColor
float hranges[] = { 0, 180 };
// saturation varies from 0 (black-gray-white) to
// 255 (pure spectrum color)
float sranges[] = { 0, 256 };
const float* ranges[] = { hranges, sranges };

// we compute the histogram from the 0-th and 1-st channels
int channels[] = {0, 1};
//int channels2[] = {0, 1};











//Finger Mouse variables

int noiseIterations = 0;

//tracking variables
//finger mouse variables
bool fingerMouseActive = false;
bool scrollMode = false;
bool LMBpressed = false;
bool LMBhold = false;
bool doubleClicked = false;
bool doubleClickExecuted = false;

bool middlePressed = false;

bool modeSwitchTriggered = false;
bool switchModeClick = false;

bool RMBpressed = false;
bool RMBCLICK = false;
//bool useRightClick = true;

bool movingMouse = true;
bool dragSwitch = false;
bool CLICK = false;
int clickIterations = 0;
int clickHoldIterations = 0;
Mat templateROI;
Mat trackingTemplateIndexFinger;
Point IndexFingerTrackingCenter;
Mat trackingTemplateThumb;
Point ThumbTrackingCenter;
//scroll mode variables
Mat trackingPalmTemplateScrolling;
Point currentPalmTemplatePosition;
Point previousPalmTemplatePosition;
vector<Point> savedPalmPoints;



double distBetweenFingers;
double thumbClickMargin;
double indexClickMargin;
double indexDoubleClickMargin;

Mat trackingTemplatePalm;
Point palmTrackingCenter;

int horizontal = GetSystemMetrics(SM_CXSCREEN);
int vertical = GetSystemMetrics(SM_CYSCREEN);



//Colouration of contours
RNG rng(12345);
bool startInitialization = false;






class handTracker{


    //added for recording data
    //fileEvaluator  dataAcquisition;
    string savedFileName;
    //


    cv::VideoCapture cap;

    //Scanning and Tracking Variables
    cv::Mat image;
    cv::Mat imageCopyForIntensity;
    cv::Mat imageDisplay;
    cv::Mat imageROI;
    cv::Mat src, hsv;


    cv::MatND hist;

    cv::MatND histCopy;

    //hand tracking global variables
    float referenceRadius = 0.0;

    fileEvaluator dataAcquisition;



public:

    void startExperiment(){
        dataAcquisition.startExperiment();
//                for(int i = 0; i<100; i++){
//                    int j = rand();
//                }
    }

    void endExperiment(){
        dataAcquisition.endExperiment();
    }


    void initialize(string filename, bool useVideo = true, bool useMouse = false)
    {
        //activate data acquisition
        savedFileName = filename;
        //dataAcquisition = fileEvaluator(filename);

        useVideoFeed = useVideo;
        useFingerMouse = useMouse;

        //initialization for video file
        if(useVideoFeed){
            stringstream videoStream;
            videoStream << "Videos/TestVideoSequences";
            videoStream <<filename;
            videoStream<<".avi";
            cap = cv::VideoCapture(videoStream.str());
        }else{
            cap = cv::VideoCapture(0);
        }



    }

    int process(bool recordData = false)
    {
        //data acquisition variable
        //if(recordData);
        if(recordData == false){
            savedFileName = "dummyFile";

        }

        dataAcquisition.openFile(savedFileName);

        videoRecorder videoAcquisition(1);







        Mat oldResultImage;

        //Variable for intensityMap function
        //Used to search for hand
        Mat searchRegionMat;


        //Initialization for webcam

        //cv::VideoCapture cap(0);


        cap>>image;
        if(recordData){
        dataAcquisition.processFrame(-1);
        //if(recordData){
        videoAcquisition.startRecording(savedFileName, image.cols, image.rows);


        }

        //equalize the image
        //equalizeRGB(image);

        //image equalization
        if(equalizeImage){
            Mat equalizedImage;
            equalize(image,equalizedImage);
            imshow("Equalized", equalizedImage);
            equalizedImage.copyTo(image);
        }

        //end of equalization


        cv::Mat imageClone;



        int x = image.cols/2 -xdim/2;
        int y = image.rows/2 -ydim/2;

        //cv::namedWindow("ROI");

        imageROI = image(cv::Rect(x, y, xdim, ydim));

        //variables for scan and track
        bool scan = true;
        bool track = true;


        //repeat the method loop
        while(cap.isOpened() && scan && track){



            if(middlePressed){
                middlePressed = false;
                mouse_event(MOUSEEVENTF_MIDDLEUP,0,0,0,0);




            }
            if(LMBpressed){
                LMBpressed = false;
                mouse_event(MOUSEEVENTF_LEFTUP,0,0,0,0);


            }
            if(RMBpressed)RMBpressed = false;

            //capture hand loop

            int successfulIterations = 0;

            bool findHist = true;




            // The hand finding loop
            while(scan){

                int numberOfFingersFoundUponSuccess = -1;


                //variables for persistent ROI scanning
                //Things that look like hands should get scanned more than once
                vector<Point> maybeContour;
                bool maybe = false;
                Point centerMaybe;
                //

                //cap>>image;
                scan = cap.read(image);

                if(scan == false){
                    break;
                }
                //flip horizontally
                cv::flip(image, image, 1);

                image.copyTo(imageClone);
                image.copyTo(imageDisplay);
                image.copyTo(imageCopyForIntensity);



                if(useColorEqualization){
                    uchar R,G,B;
                    uchar *ptr1;
                    ptr1 = image.data;
                    for(int r = 0;r < image.rows;r++)
                    for(int c = 0;c < image.cols;c++)
                    {
                    R = *ptr1;
                    ptr1++;
                    G = *ptr1;
                    ptr1++;
                       B = *ptr1;
                    ptr1--;
                    ptr1--;


                    *ptr1 =(uchar)  ((float)R)/sqrt((float)(R*R + G*G + B*B))*256;
                    //R = *ptr1 ;
                    ptr1++;
                    *ptr1 =(uchar)  ((float)G)/sqrt((float)(R*R + G*G + B*B))*256;
                    //G = *ptr1 ;
                    ptr1++;
                    *ptr1 =(uchar)  ((float)B)/sqrt((float)(R*R + G*G + B*B))*256;
                    //B = *ptr1 ;
                    ptr1++;
                    }

                }

                if(useIntensityMap || useIntensityAsBackProjection){
                    intensityMap(imageCopyForIntensity, searchRegionMat);
                }





                //cascade classifier

                //detectAndDisplay(haarImage);





                //equalize the image
                //equalizeRGB(image);
                //end of equalization

                //colorReduce
                if(colorReduceEnabled){
                    colorReduce2(image,color_reduce_parameter);
                }
                //





                //cv::Mat imageClone;

                //    sharpen(image,imageClone, sharpening_parameter);
                //    imshow("Sharpen", imageClone);
                //    imageClone.copyTo(image);


                // low pass filter code
                if(lowPassEnabled){
                    Mat filtered;
                    Mat kernel;
                    int ind = 0;
                    int  kernel_size = 3 + 2*( ind%5 );
                    kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
                    filter2D(image, image, -1,kernel );
                }

                //imshow("Filtered image", image);

                //end of low pass filter code


                //image equalization
                if(equalizeImage){
                    Mat equalizedImage;
                    equalize(image,equalizedImage);
                    imshow("Equalized", equalizedImage);
                    equalizedImage.copyTo(image);
                }

                //end of equalization









                //Display Captured Image
                cv::rectangle(imageDisplay, cv::Point(x,y), cv::Point(x+xdim, y+ydim), Scalar(0,255,0), 3, 8,0);
                cv::imshow("Capture", imageDisplay);

                if(recordData){
                   videoAcquisition.addFrame(imageDisplay);
                }


                //cv::imshow("ROI", imageROI);

                //This line toggles the ROI saving!

                if(findHist){

                    imageROI.copyTo(src);
                    //cv::imshow("currentROI", src);


                    cvtColor(src, hsv, CV_BGR2HSV);

                    if(useMinSat){
                        vector<cv::Mat> v;
                        cv::split(hsv,v);
                        if(useModifiedThreshold){
                            cv::threshold(v[0],v[0], 10,200,thresholdCode);
                            cv::threshold(v[1],v[1],65,101,thresholdCode);
                        }else if(useSkinToneThresholds){
                            cv::threshold(v[0],v[0], 0,50,thresholdCode);
                            cv::threshold(v[1],v[1],51,173,thresholdCode);
                            cv::threshold(v[2],v[2],89,255,thresholdCode);

                        }else{
                            //Pavel's original code
                        cv::threshold(v[1],v[1],65,255,thresholdCode);
                        }

                        cv::merge(v, hsv);
                    }





                    //calcHist()

                    calcHist( &hsv, 1, channels, cv::Mat(), // do not use mask
                              hist, 2, histSize, ranges,
                              true, // the histogram is uniform
                              false );

                    //            equalizeHist(hist,histCopy);

                    //            histCopy.copyTo(hist);


                    double maxVal=0;
                    //            cv::MatND histClone;
                    //            equalizeHist(hist, histClone);
                    //            histClone.copyTo(hist);
                    minMaxLoc(hist, 0, &maxVal, 0, 0);
                }
                imshow("The Histogram", hist);



                //Colour Reduce the image after a histogram has been taken
                //colorReduce2(image, color_reduce_parameter);

                // imshow("Color Reduced", image);
                //End of colour reduce code




                //Back projecting the histogram
                //hist
                cv::Mat imageCopy;
                image.copyTo(imageCopy);
                imshow("Original Image",image);
                cvtColor(image, image, CV_BGR2HSV);



                //eliminate low saturation
                if(useMinSat){
                    vector<cv::Mat> v;
                    cv::split(image,v);
                    if(useModifiedThreshold){
                        cv::threshold(v[0],v[0], 10,200,thresholdCode);
                        cv::threshold(v[1],v[1],65,101,thresholdCode);
                    }else if(useSkinToneThresholds){
                        cv::threshold(v[0],v[0], 0,50,thresholdCode);
                        cv::threshold(v[1],v[1],51,173,thresholdCode);
                        cv::threshold(v[2],v[2],89,255,thresholdCode);

                    }else{
                        //Pavel's original code
                    cv::threshold(v[1],v[1],65,255,thresholdCode);
                    }
                    //cv::imshow("Saturation",v[1]);

                    v[1].copyTo(saturationZone);


                    merge(v, image);

                }

                if(useSkinToneThresholds){
                Mat dummmyImage;
                cvtColor(image, dummmyImage, CV_HSV2BGR);
                imshow("Skin tone Thresholded HSV image", dummmyImage);
                }

                //


                cv::Mat resultImage;

                cv::calcBackProject(&image,
                                    1,
                                    channels,
                                    hist,
                                    resultImage,
                                    ranges,
                                    255.0);




                //imshow("Back project", resultImage);

                if(useIntensityMapScanFiltering){

                    Mat andedResult;

                    addWeighted(searchRegionMat, 0.5, resultImage, 0.5, 0.0, andedResult);
                    //imshow("Anded Result", andedResult);

                    threshold(andedResult, andedResult, 200, 255, CV_THRESH_BINARY);
                    //imshow("Anded & Thresholded Result", andedResult);

                    //addWeighted(searchRegionMat, 0.5, andedResult, 0.5, 0.0, andedResult);
                    //threshold(andedResult, andedResult, 75, 255, CV_THRESH_BINARY);
                    //imshow("With added elements from search Region", andedResult);

                    andedResult.copyTo(resultImage);


                }else if(useIntensityAsBackProjection){
                    searchRegionMat.copyTo(resultImage);
                }


                //find contours

                std::vector<std::vector<cv::Point> > contours;

                cv::findContours(resultImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);




                //cv::drawContours(resultImage, contours, -1, cv::Scalar(255), 2);

                std::vector<std::vector<cv::Point> > goodContours;

                //cv::imshow("Contours", resultImage);


                for( int i = 0; i < contours.size(); i++ ){
                    if (contours[i].size()>min_contour_size){
                        vector<Point> polyApprox;
                        approxPolyDP(contours[i], polyApprox, polyAprrox_epsilon, true);
                        goodContours.insert(goodContours.begin(), polyApprox);


                    }


                }






                vector<vector<Point> >hull(goodContours.size());
                vector<std::vector<Vec4i> > defect(goodContours.size());

                //vector<std::vector<cv::Vec4i> >defect(goodContours.size());
                vector<std::vector<int> > convexHull_IntIdx(goodContours.size());


                for(int i = 0; i<goodContours.size(); i++){
                    cv::convexHull(goodContours[i],hull[i],false );

                    cv::convexHull(goodContours[i], convexHull_IntIdx[i], false);


                    if(goodContours[i].size() >= min_number_of_contour_points){
                        cv::convexityDefects(goodContours[i],convexHull_IntIdx[i],defect[i]);
                    }



                }







                Mat drawing = Mat::zeros( resultImage.size(), CV_8UC3 );

                Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

                drawContours( drawing, goodContours, -1, color, 2);
                drawContours( drawing, hull, -1, color, 2);


                //cout<<defect[0]<<endl;

                //find hands
                int numberOfBigLines = 0;
                int candidates = 0;

                if(goodContours.size() > 0){
                    for(int j = 0; j < defect.size(); j++){
                        numberOfBigLines = 0;
                        Point prevPoint(-1,-1);
                        int prevDist = 0;
                        bool palmFound = false;
                        vector<Point> innerPoints;
                        vector<Point> startPoints;
                        vector<Point> endPoints;
                        for (int i = 0; i< defect[j].size(); i++){
                            Point startP = goodContours[j][(defect[j][i][0])];
                            Point endP = goodContours[j][(defect[j][i][1])];
                            Point innerP = goodContours[j][(defect[j][i][2])];

                            cv::circle(drawing, startP, 5, Scalar(255,255,255), 3);
                            cv::circle(drawing, endP, 5, Scalar(255,255,255), 3);
                            cv::circle(drawing, innerP, 5, Scalar(255,255,255), 3);
                            Point mid = midPoint(startP,endP);
                            String s1;
                            ostringstream temp;  //temp as in temporary
                            int currentDist = dist(mid, innerP);
                            temp<<currentDist;
                            s1 = temp.str();
                            cv::putText(drawing,  s1, midPoint(mid, innerP), CV_FONT_BLACK, 1, Scalar(255,255,255));
                            cv::circle(drawing, mid, 5, Scalar(255,255,255), 3);

                            //This code makes sure that enough of a certain length of line are found with the
                            //inner points being within a certain distance of each other.
                            //This effectively detects the hand.
                            if(dist(mid, innerP)> big_line_threshold_scanning){
                                if(prevPoint.x < 0){

                                    innerPoints.insert(innerPoints.begin(), innerP);
                                    startPoints.insert(startPoints.begin(), startP);
                                    endPoints.insert(endPoints.begin(), endP);
                                    prevPoint = innerP;
                                    numberOfBigLines++;
                                }else{
                                    //if(dist(prevPoint, innerP) < ((5*currentDist)/2)){

                                    innerPoints.insert(innerPoints.begin(), innerP);
                                    startPoints.insert(startPoints.begin(), startP);
                                    endPoints.insert(endPoints.begin(), endP);

                                    prevPoint = innerP;
                                    numberOfBigLines++;
                                    //}else{
                                    // leave this code out!

                                    //prevDist = 0;
                                    //numberOfBigLines = 0;
                                    //}
                                }

                            }


                            cv::line(drawing, mid, innerP, Scalar(255,255,255), 3 );
                        }

                        //vector<Point>consecutiveFingers;




                        if( numberOfBigLines>=min_number_of_big_lines  && numberOfBigLines <= max_number_of_big_lines && isContourConvex(innerPoints)){

                            float pRadius;
                            Point2f pCenter;
                            minEnclosingCircle(innerPoints, pCenter, pRadius);
                            circle(drawing, pCenter, pRadius, Scalar(0,50,255), 3);
                            int fingers = 0;


                            Point roiCenter(x,y);

                            double distROItoCentre = dist(pCenter, roiCenter);

                            int consecutiveFingers = 0;

                            if(distROItoCentre <= pRadius){




                                for(int i = 0; i< innerPoints.size(); i++){
                                    Point pStart = startPoints[i];
                                    Point pNextEndp = endPoints[((i+1)%endPoints.size())];
                                    Point pInner = innerPoints[i];
                                    circle(drawing, pInner, 6, Scalar(255,0,0), 3);


                                    //next finger tip point

                                    Point pNextInner = innerPoints[((i+1)%innerPoints.size())];
                                    Point currentTipPoint = midPoint(pStart, pNextEndp);
                                    Point fingerBase = midPoint(pInner, pNextInner);
                                    double tipDist = dist(pStart, pNextEndp);
                                    double baseDist = dist(pInner, pNextInner);
                                    double fingerFromCenter = dist(currentTipPoint,pCenter);
                                    double innerD1 = dist(pInner, pCenter);
                                    double innerD2 = dist(pNextInner, pCenter);


                                    //next finger tip point
                                    Point NextTipStart = startPoints[i+1];
                                    Point NextTipNextEndp = endPoints[((i+2)%endPoints.size())];
                                    Point NextTipPoint = midPoint(NextTipStart, NextTipNextEndp);

                                    //where does the line intersect with the circle?
                                    double diffx = fingerBase.x - pCenter.x;
                                    double diffy = fingerBase.y - pCenter.y;
                                    double radiusDiff = sqrt((pow(diffx,2) + pow(diffy, 2)));
                                    double ratioX = diffx/radiusDiff;
                                    double ratioY = diffy/radiusDiff;
                                    double radiusShift = pRadius - radiusDiff;
                                    double shiftX = radiusShift*ratioX;
                                    double shiftY = radiusShift*ratioY;
                                    Point shiftedFingerBase(fingerBase.x + shiftX, fingerBase.y + shiftY);
                                    circle(drawing, shiftedFingerBase, 3, Scalar(200,50,200), 3);

                                    double fingerLength = dist(currentTipPoint, shiftedFingerBase);

                                    //end of code

                                    Point knuckle = midPoint(shiftedFingerBase, currentTipPoint);
                                    double knuckleDist = dist(knuckle, pCenter);




                                    line(drawing, currentTipPoint, pInner, Scalar(255,130,130),3);
                                    line(drawing, currentTipPoint, pNextInner, Scalar(255,130,130),3);

                                    //test the tipe angle and other conditions

                                    if((currentTipPoint.x >= 0 && currentTipPoint.x <= 30) || (currentTipPoint.x >= (imageCopy.cols -30)  && currentTipPoint.x <= imageCopy.cols)  ){

                                    }else{
                                        if((currentTipPoint.y >= 0 && currentTipPoint.y <= 30) || (currentTipPoint.y >= (imageCopy.rows -30)  && currentTipPoint.y <= imageCopy.rows)  ){

                                        }else{

                                            if((angleBetween3Points(currentTipPoint, pInner, pNextInner)<=(3.1415/angle_divisor_scanning))&&((tip_distance_multiplier*tipDist) < baseDist)&&(fingerFromCenter<(max_finger_factor*pRadius))){


                                                // if((innerD1 < 1.5*pRadius) && (innerD2 < 1.5*pRadius)&&(tipDist < (pRadius/2.0))&&(fingerLength > 0.75*pRadius)){

                                                circle(drawing, pInner, 3, Scalar(0,0,255), 3);
                                                circle(drawing, pNextInner, 3, Scalar(0,0,255), 3);

                                                if((innerD1 < innerDist_radius_threshold*pRadius) && (innerD2 < innerDist_radius_threshold*pRadius)&&(fingerLength >= min_finger_radius_multiplier*pRadius)&&(fingerLength <= max_finger_radius_multiplier*pRadius)&&(knuckleDist > min_knuckle_radius_multiplier*pRadius)){


                                                    line(drawing, shiftedFingerBase, currentTipPoint, Scalar(255,34,180),3);



                                                    circle(drawing, currentTipPoint, 3, Scalar(0,0,255), 3);
                                                    line(drawing, pStart, pNextEndp, Scalar(255,34,180),3);
                                                    //line(drawing, pStart, pNextEndp, Scalar(255,34,180),3);
                                                    fingers++;
                                                    //test the consecutive finger angle
                                                    if(angleBetween3Points(pInner, currentTipPoint, NextTipPoint) <= (3.1415/consecutive_finger_angle_divisor)){
                                                        if(consecutiveFingers ==0){
                                                            circle(drawing, currentTipPoint, 8, Scalar(100,50,230), 5);
                                                            circle(drawing, NextTipPoint, 8, Scalar(100,50,230), 5);
                                                            consecutiveFingers =2;
                                                        }else{
                                                            circle(drawing, NextTipPoint, 8, Scalar(100,50,230), 5);
                                                            consecutiveFingers++;
                                                        }
                                                    }

                                                }

                                            }
                                        }
                                    }

                                }
                            }
                            if(fingers == 5){
                                numberOfFingersFoundUponSuccess = fingers;
                                cout<<"hand found!!!!\n";
                                cout<<"Candidate Found!"<<endl;
                                cout<<"Test Message!!\n";
                                candidates++;
                                referenceRadius = pRadius;
                                successfulIterations++; //keep track of iterations
                                cout<<"Search Worked! Fingers:"<<fingers<<" consecutive fingers:"<<consecutiveFingers<<"\n";
                                findHist = false;
                                break;





                            }else if(consecutiveFingers>=3){
                                numberOfFingersFoundUponSuccess = fingers;
                                cout<<"hand found!!!!\n";
                                cout<<"Candidate Found!"<<endl;
                                cout<<"Test Message!!\n";
                                candidates++;
                                referenceRadius = pRadius;
                                successfulIterations++; //keep track of iterations
                                cout<<"Search Worked! Fingers:"<<fingers<<" consecutive fingers:"<<consecutiveFingers<<"\n";
                                findHist = false;
                                break;
                            }else if((fingers >= 3)&&(fingers <= 7)){
                                //Mat copyImage;



                                cout<<"Wrong amount of fingers :"<<fingers<<endl;
                                maybe = true;
                                centerMaybe = pCenter;
                                //imshow("drawing", drawing);
                                cout<<"Worked Maybe! consecutive fingers:"<<consecutiveFingers<<"\n";
                                //waitfor input
                                //waitKey(0);
                            }
                            //else if (fingers>=2){
                            //                        imshow("drawing Bad it", drawing);
                            //                        //waitfor input
                            //                        waitKey(0);
                            //}





                        }else if( j == defect.size() - 1){
                            successfulIterations = 0;

                        }
                    }
                }


                //if no success reassign the ROI
                if(successfulIterations == 0){
                    findHist = true;
                    if(maybe){ //goodContours.size() > 0
                        maybe = false;
                        candidates =0;


                        //reassign ROI to centerPoint of first or biggest contour
                        //find the center of the hand
                        cout<<"Worked Maybe!\n";
                        x = centerMaybe.x;
                        y = centerMaybe.y;
                        //cv::rectangle(imageDisplay, cv::Point(x,y), cv::Point(x+xdim, y+ydim), Scalar(0,0,255), 3, 8,0);
                        //cv::imshow("Capture - NEW ROI", imageDisplay);

                        //waitKey(0);
                        //                int centerX = centerMaybe.x;
                        //                int centerY = centerMaybe.y;
                        ////                cv::Moments m=cv::moments(maybeContour);
                        ////                centerX=m.m10/m.m00;
                        ////                centerY=m.m01/m.m00;
                        //                //cv::circle(drawing, cv::Point2d(centerX, centerY), 5, Scalar(255,255,255), 3);


                        //                if(centerX+(xdim/2)<image.cols && centerY+(ydim/2)<image.rows && centerX>=(xdim/2) && centerY>=(ydim/2)){
                        //                    x = centerX - xdim/2;
                        //                    y = centerY - ydim/2;
                        //                    imageROI = image(cv::Rect(x, y, xdim, ydim));


                        //                }

                    }else{
                        //findHist = true;
                        uchar* satIntensity;
                        //                   uchar intensity = 0;

                        int iterations = 0;
                        //while(intensity == 0){
                        //while(true){
                        x = rand()%(image.cols - xdim);
                        y = rand()%(image.rows - ydim);
                        imageROI = image(cv::Rect(x, y, xdim, ydim));
                        //cout<<"Image Rows:"<<image.rows<<" Image Columns:"<<image.cols<<endl;
                        //cout<<"SatZone Rows:"<<saturationZone.rows<<" saturation Columns:"<<saturationZone.cols<<endl;
                        if(saturationZone.cols > 0){

                            Scalar intensity;
                            if(useIntensityMap){
                                //intensityMap(imageCopyForIntensity, searchRegionMat);
                                intensity = searchRegionMat.at<uchar>(y, x);
                            }else{
                            intensity = saturationZone.at<uchar>(y, x);
                            }

                            while(intensity[0] == 0){
                                x = rand()%(image.cols - xdim);
                                y = rand()%(image.rows - ydim);
                                imageROI = image(cv::Rect(x, y, xdim, ydim));
                                if(useIntensityMap){

                                    intensity = searchRegionMat.at<uchar>(y, x);
                                }else{
                                intensity = saturationZone.at<uchar>(y, x);
                                }
                            }
                            //cout<<"Intensity:"<<intensity[0]<<endl;
                        }
                        //}
                        //cout<<"Intensity:"<<intensity<<endl;
                        //cout<<"Intensity:"<<intensity<<"\n";


                        //iterations++;
                        //}


                        //randomly reassign ROI
                    }



                }

                //no big lines are availables code




                //imshow("Drawing", drawing);

                if(successfulIterations >= min_successful_iterations){
                    cout<<"Palm is Found!"<<endl;
                    if(recordData){
                    dataAcquisition.processFrame(numberOfFingersFoundUponSuccess);
                    }
                    //waitKey(0);
                    break;
                }else{
                    if(recordData){
                    dataAcquisition.processFrame(-1);
                    }

                }



                waitKey(1);
            }









            //    //tracking loop

            //float previousRadius = 0;

            while(track){

                //variables for dataAcquisition

                //imshow("THE PALM USED", src);

                //cap>>image;
                track = cap.read(image);
                if(track == false){
                    break;
                }

                cv::flip(image, image, 1);

                image.copyTo(imageDisplay);

                //back Projection
                cv::Mat imageCopy;
                cv::Mat imageCopy_2;
                cv::Mat imageCopyForIntensityTracking;
                image.copyTo(imageCopy);
                image.copyTo(imageCopyForIntensityTracking);


                if(useColorEqualization){
                    uchar R,G,B;
                    uchar *ptr1;
                    ptr1 = image.data;
                    for(int r = 0;r < image.rows;r++)
                    for(int c = 0;c < image.cols;c++)
                    {
                    R = *ptr1;
                    ptr1++;
                    G = *ptr1;
                    ptr1++;
                       B = *ptr1;
                    ptr1--;
                    ptr1--;


                    *ptr1 =(uchar)  ((float)R)/sqrt((float)(R*R + G*G + B*B))*256;
                    //R = *ptr1 ;
                    ptr1++;
                    *ptr1 =(uchar)  ((float)G)/sqrt((float)(R*R + G*G + B*B))*256;
                    //G = *ptr1 ;
                    ptr1++;
                    *ptr1 =(uchar)  ((float)B)/sqrt((float)(R*R + G*G + B*B))*256;
                    //B = *ptr1 ;
                    ptr1++;
                    }

                }

                if(useIntensityMapTracking || useAddImages || useIntensityAsBackProjection){
                    intensityMap(imageCopyForIntensityTracking, searchRegionMat);
                }

                //imshow("Camera Feed of Tracking Loop", image);












                //sharpen(image,imageClone);
                //imageClone.copyTo(image);





                //equalize the image
                //equalizeRGB(image);
                //end of equalization

                //color Reduce the image!
                //colorReduce2(image,color_reduce_parameter);
                //

                if(colorReduceEnabled){
                    colorReduce2(image, color_reduce_parameter);
                }


                // low pass filter code

                if(lowPassEnabled){
                    Mat filtered;
                    Mat kernel;
                    int ind = 0;
                    int  kernel_size = 3 + 2*( ind%5 );
                    kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
                    filter2D(image, image, -1,kernel );
                    //imshow("Filtered image", image);
                }

                //end of low pass filter code

                //image equalization
                if(equalizeImage){
                    Mat equalizedImage;
                    equalize(image,equalizedImage);
                    //imshow("Equalized", equalizedImage);
                    equalizedImage.copyTo(image);
                }

                //end of equalization



                cvtColor(image, image, CV_BGR2HSV);

                //eliminate low saturation
                if(useMinSat){
                    vector<cv::Mat> v;
                    cv::split(image,v);
                    if(useModifiedThreshold){
                        cv::threshold(v[0],v[0], 10,200,thresholdCode);
                        cv::threshold(v[1],v[1],65,101,thresholdCode);
                    }else if(useSkinToneThresholds){
                        cv::threshold(v[0],v[0], 0,50,thresholdCode);
                        cv::threshold(v[1],v[1],51,173,thresholdCode);
                        cv::threshold(v[2],v[2],89,255,thresholdCode);

                    }else{
                        //Pavel's original code
                    cv::threshold(v[1],v[1],65,255,thresholdCode);
                    }

                    v[1].copyTo(imageCopy_2);

                    merge(v, image);

                }

                //
                //image.copyTo(imageCopy_2);



                cv::Mat resultImage;

                cv::calcBackProject(&image,
                                    1,
                                    channels,
                                    hist,
                                    resultImage,
                                    ranges,
                                    255.0);

                //imshow("Back Projection", resultImage);


                //code for segmentation
                //WatershedSegmenter segmenter;

                //try morpho filtering here



                dilate(resultImage, resultImage, cv::Mat(), cv::Point(-1,-1),2);



                //end of morhpo filters



                //











                Mat resultCopy;
                resultImage.copyTo(resultCopy);



                //imshow("Contours that have been found", resultImage);

                if(useIntensityMapTracking){
                Mat andedResult;

                addWeighted(searchRegionMat, 0.5, resultImage, 0.5, 0.0, andedResult);
                //imshow("Anded Result", andedResult);

                threshold(andedResult, andedResult, 200, 255, CV_THRESH_BINARY);
                //imshow("Anded & Thresholded Result", andedResult);

                //addWeighted(searchRegionMat, 0.5, andedResult, 0.5, 0.0, andedResult);
                //threshold(andedResult, andedResult, 75, 255, CV_THRESH_BINARY);
                //imshow("With added elements from search Region", andedResult);

                andedResult.copyTo(resultImage);



                }else if (useAddImages){
                    //Mat addedResult;
                    add(searchRegionMat, resultImage, resultImage);

                }else if (useIntensityAsBackProjection){
                    searchRegionMat.copyTo(resultImage);
                }







                //find contours

                std::vector<std::vector<cv::Point> > contours;

                cv::findContours(resultImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);






                std::vector<std::vector<cv::Point> > goodContours;





                for( int i = 0; i < contours.size(); i++ ){
                    if (contours[i].size()>min_contour_size_tracking){
                        vector<Point> polyApprox;
                        //approxPolyDP(contours[i], polyApprox, 7.0, true);
                        polyApprox = contours[i];
                        goodContours.insert(goodContours.begin(),polyApprox);

                    }


                }




                //break the loop
                if(goodContours.size() == 0){
                    //code used to test and debug
//                    namedWindow("NO GOOD CONTOURS!!!");
//                    waitKey(0);
                    break;
                }

                Mat contourDrawing = Mat::zeros( resultImage.size(), CV_8UC1 );
                cv::drawContours(contourDrawing , goodContours, -1, cv::Scalar(255), -1);
                //erode(contourDrawing, contourDrawing, Mat(), Point(-1,-1), 3);
                //cv::imshow("Contours with stuff", contourDrawing);
                //contourDrawing.copyTo(imageCopy_2);
                //GO WITH GOOD CONTOURS!!!!!!!!












                Mat drawing = Mat::zeros( resultImage.size(), CV_8UC3 );

                Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

                drawContours( drawing, goodContours, -1, color, 2);
                //drawContours( drawing, hull, -1, color, 2);
                Mat drawing2 = Mat::zeros(resultImage.size(), CV_8UC1);

                drawContours( drawing2, goodContours, -1, Scalar(255), -1);

                //drawContours( drawing2, hull, -1, Scalar(255), 2);

                //imshow("Drawing 2", drawing2);




                std::vector<std::vector<cv::Point> > contoursSecondRound;


                //            cv::findContours(segmentation, contoursSecondRound, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

                //end of segmentation code
                contoursSecondRound = goodContours;

                vector<vector<Point> >hull(contoursSecondRound.size());

                vector<std::vector<Vec4i> > defect(contoursSecondRound.size());
                vector<std::vector<int> > convexHull_IntIdx(contoursSecondRound.size());






                std::vector<std::vector<cv::Point> > approximatedContourShapes(contoursSecondRound.size());






                for(int h = 0; h< contoursSecondRound.size(); h++){
                    //approxPolyDP(contoursSecondRound[h], approximatedContourShapes[h], 1.0, true);
                    approximatedContourShapes[h] = contoursSecondRound[h];
                    //
                    cv::convexHull(approximatedContourShapes[h],hull[h],false );

                    cv::convexHull(approximatedContourShapes[h], convexHull_IntIdx[h], false);
                    if(convexHull_IntIdx[h].size() > 3){
                        cv::convexityDefects(approximatedContourShapes[h],convexHull_IntIdx[h],defect[h]);
                    }



                }


                Mat contoursMat = Mat::zeros( resultImage.size(), CV_8UC3 );
                Mat contoursMat2 = Mat::zeros( resultImage.size(), CV_8UC3 );
                Mat contoursMat3 = Mat::zeros( resultImage.size(), CV_8UC3 );




                cv::drawContours(contoursMat, contoursSecondRound, -1, Scalar(255,134, 175), 3 );
                cv::drawContours(contoursMat2, approximatedContourShapes, -1, Scalar(230,100, 223), 3 );
                cv::drawContours(contoursMat3, approximatedContourShapes, -1, Scalar(230,100, 223), 3 );

                vector<int> distances;
                bool targetFound = false;
                vector<Point> fingersFound;
                vector<double> fingerTipWidth;


                float referenceDist;
                bool referenceDistInit = false;






                for(int index = 0; index<contoursSecondRound.size(); index++){


                    //draw contour rotated rectangle here
                    RotatedRect boundingBox = minAreaRect(contoursSecondRound[index]);

                    //calculate the amount of the box filled
                    double size = contourArea(contoursSecondRound[index], false);
                    double percentFilled = 100.0*(size/boundingBox.size.area());





                    Point2f boundingBoxPnts[4];
                    boundingBox.points(boundingBoxPnts);
                    for( int j = 0; j < 4; j++ )
                        line( contoursMat3, boundingBoxPnts[j], boundingBoxPnts[(j+1)%4], Scalar(255,30,180), 1, 8 );

                    if(percentFilled > 60.0){
                        putText(contoursMat3, "Filled", boundingBoxPnts[0], 0, 2.0, Scalar(255,255,255) );
                    }

                    //draw bounding rectangle here
                    Rect boundRect = boundingRect(contoursSecondRound[index]);
                    rectangle(contoursMat3, boundRect, Scalar(255,255,255), 3);



                    for(int k = 0; k<defect[index].size(); k++){
                        Point startP = approximatedContourShapes[index][(defect[index][k][0])];
                        Point endP = approximatedContourShapes[index][(defect[index][k][1])];
                        Point innerP1 = approximatedContourShapes[index][(defect[index][k][2])];
                        Point mid = midPoint(startP,endP);
                        int distBetween = dist(mid, innerP1);
                        distances.insert(distances.begin(), distBetween);

                        //circle(contoursMat2, innerP1, 3, Scalar(255,255,255),3);


                    }




                    int maxLineVal = 0;

                    if(distances.size() > 0){
                        std::sort(distances.begin(), distances.end());
                        maxLineVal = distances.back();

                        vector<Point> innerPoints;
                        vector<Point> startPoints;
                        vector<Point> endPoints;
                        vector<int> defectIndices;
                        vector<int> innerDistToInnerPoints;
                        vector<double> innerPointAngles;



                        for(int k = 0; k<defect[index].size(); k++){
                            Point startP = approximatedContourShapes[index][(defect[index][k][0])];
                            Point endP = approximatedContourShapes[index][(defect[index][k][1])];
                            Point innerP1 = approximatedContourShapes[index][(defect[index][k][2])];
                            Point mid = midPoint(startP,endP);
                            int distBetween = dist(mid, innerP1);
                            //distances.insert(distance.begin, distBetween);
                            if(innerP1.x > 15 && innerP1.y > 15 && innerP1.x < image.cols - 15 && innerP1.y<image.rows -15 ){
                                if(distBetween > (maxLineVal/big_line_threshold_tracking)){
                                    circle(contoursMat2, innerP1, 3, Scalar(255,255,255),3);
                                    line(contoursMat2, innerP1, mid, Scalar(255,255,255), 2);
                                    double ang = angleBetween3Points(innerP1, startP, endP);

                                    // stringstream angleOut;
                                    //angleOut<<ang;
                                    //string angle = angleOut.str();
                                    //putText(contoursMat2, angle, innerP1, 1, 2.0, Scalar(255,220,144));
                                    innerPoints.insert(innerPoints.begin(), innerP1);
                                    //innerPointAngles are unused
                                    innerPointAngles.insert(innerPointAngles.begin(), ang);
                                    //unused variables above
                                    defectIndices.insert(defectIndices.begin(), k);
                                    innerDistToInnerPoints.insert(innerDistToInnerPoints.begin(), distBetween);
                                    startPoints.insert(startPoints.begin(), startP);
                                    endPoints.insert(endPoints.begin(), endP);

                                }
                            }




                        }


                        Point2f currentCenter;
                        float currentRadius;
                        vector<Point> palmPoints;
                        vector<Point> palmPoints_2;

                        vector<bool> narrowAngle(innerPoints.size());


                        //draw the palm
                        bool palmFound = false;
                        if(innerPoints.size() > 1){
                            //draw all of the inner points so far

                            for(int i = 0; i<innerPoints.size(); i++){
                                circle(contoursMat2, innerPoints[i], 9, Scalar(0,0,255), 3);
                                if(innerPointAngles[i] <= (3.1415/1.7)){
                                    line(contoursMat2, startPoints[i], endPoints[i], Scalar(255,0,0),3);
                                    narrowAngle[i] = true;
                                }else{
                                    line(contoursMat2, startPoints[i], endPoints[i], Scalar(0,255,0),3);
                                    narrowAngle[i] = false;
                                }

                            }

                            //draw all the angle defects
                            //for(int i = 0; i<innerPoints.size(); i++){
                            // if ( innerPointAngles[i] <
                            //}


                            minEnclosingCircle(innerPoints, currentCenter, currentRadius);

                            //compute centroid

                            //contour in question
                            //contoursSecondRound[index]

                            cv::Moments m=cv::moments(contoursSecondRound[index]);
                            double  contourCenterX=m.m10/m.m00;
                            double contourCenterY=m.m01/m.m00;
                            Point contourCenter(contourCenterX, contourCenterY);
                            circle(contoursMat2, contourCenter, 3, Scalar(180,55,200), 3);


                            Point average = averageOfPoints(innerPoints);
                            circle(contoursMat2, average, 3, Scalar(255,30,30), 3);




                            /////// IMPORTANT COOOOODEEEE!!

                            //sift through points
                            //test if the distance to the center
                            for(int i = 0; i<innerPoints.size(); i++){

                                int testDist= dist(currentCenter, innerPoints[i]);
                                int distToAverageCenter = dist(average,innerPoints[i]);

                                //IDEA change this to allow a greater distance away from the calculated average
                                //IDEA IDEA IDEA PALMPOINTS PALMPOINTS PALMPOINTS
                                //                            if(((double)testDist)< innerDistFactor*((double)innerDistToInnerPoints[i])){
                                //                                palmPoints.insert(palmPoints.begin(), innerPoints[i]);

                                //                            }else
                                if(testDist > 0.6*currentRadius){
                                    if(2.5*testDist >distToAverageCenter){
                                        putText(contoursMat2, "Is Palm Point", innerPoints[i], 1, 1.5, Scalar(133,45,201),2);
                                        if(((double)testDist)< innerDistFactor*((double)innerDistToInnerPoints[i])){
                                            double innerX = innerPoints[i].x;
                                            double innerY = innerPoints[i].y;
                                            int margin = 30;
                                            if(innerX > margin && innerX < image.cols - margin && innerY > margin && innerY<image.rows - margin){
                                                palmPoints.insert(palmPoints.begin(), innerPoints[i]);

                                            }
                                        }

                                    }
                                }else{
                                    if(((double)testDist)< innerDistFactor*((double)innerDistToInnerPoints[i])){
                                        putText(contoursMat2, "Is Special\n Palm Point", innerPoints[i], 1, 1.5, Scalar(133,45,201),2);
                                        double innerX = innerPoints[i].x;
                                        double innerY = innerPoints[i].y;
                                        int margin = 25;
                                        if(innerX > margin && innerX < image.cols - margin && innerY > margin && innerY<image.rows - margin){
                                            palmPoints.insert(palmPoints.begin(), innerPoints[i]);

                                        }
                                    }

                                }
                                //                                           else if(innerPointAngles[i] > 2.2){
                                //                                                 palmPoints.insert(palmPoints.begin(), innerPoints[i]);
                                //                                           }
                                palmPoints_2.insert(palmPoints_2.begin(), innerPoints[i]);


                            }

                            //filter out points in palm that are outliers
                            //code is currently NOT working
                            vector<Point> filteredPoints;
                            if(palmPoints.size() >= 4){
                                minEnclosingCircle(palmPoints, currentCenter, currentRadius);
                                for(int i = 0; i< palmPoints.size(); i++){
                                    Point current = palmPoints[i];
                                    Point back = palmPoints[(i-1)%palmPoints.size()];
                                    Point forward = palmPoints[(i+1)%palmPoints.size()];

                                    double angle = 180.0*angleBetween3Points(current, back, forward)/3.1415;
                                    stringstream ang;
                                    ang <<angle;
                                    //putText(contoursMat2, ang.str(), current, 1, 2.0, Scalar(255,255,255));
                                    string s1 = ang.str();


                                    if(angleBetween3Points(current, back, forward) > (3.1415/palm_filter_angle_divisor) || s1[0] =='n'){


                                        filteredPoints.insert(filteredPoints.begin(), current);
                                    }
                                }
                                if(filteredPoints.size()>=5){
                                    palmPoints = filteredPoints;
                                }

                            }



                            //end of filtering







                            // draw the palm
                            if(palmPoints.size() >= 2){
                                for(int i = 0; i< palmPoints.size(); i++){
                                    circle(contoursMat2, palmPoints[i], 6, Scalar(0,255,0),3);
                                }

                                minEnclosingCircle(palmPoints, currentCenter, currentRadius);
                                circle(contoursMat2, currentCenter, currentRadius, Scalar(0,255,0), 3);
                                Point currentPalmOrigin(x,y);
                                int distFromOrigin = dist(currentPalmOrigin, currentCenter);
                                //                            if(previousRadius < 1.0){
                                //                               previousRadius = currentRadius;
                                //                            }

                                //code added to test if point is inside a polygon
                                bool isInside = pointPolygonTest(contoursSecondRound[index], currentPalmOrigin, false);



                                // debugging target lost code

                                circle(contoursMat2, currentPalmOrigin, 3, Scalar(255,230,241), 3);





                                //end of target lost code


                                if((distFromOrigin <= referenceRadius_range*referenceRadius && !isInside)|| (isInside && distFromOrigin < (referenceRadius_range+1)*referenceRadius)){


                                    if((referenceDistInit == false)||(referenceDistInit && distFromOrigin <= referenceDist))
                                    {
                                        if(referenceDistInit == false){
                                            referenceDistInit = true;
                                        }
                                        referenceDist =distFromOrigin;


                                        line(contoursMat2, currentPalmOrigin, currentCenter, Scalar(255,255,255), 3);
                                        //previousRadius = currentRadius;
                                        palmFound = true;
                                        circle(imageCopy, currentCenter, currentRadius, Scalar(0,0,255), 3);
                                        x = currentCenter.x;
                                        y = currentCenter.y;
                                        targetFound = true;
                                        noiseIterations = 0;
                                        savedPalmPoints = palmPoints;

                                        //update radius if 5 fingers are found
                                        if(fingersFound.size() == 5){
                                            if(currentRadius<= 0.5*referenceRadius){
                                                referenceRadius = 0.75*referenceRadius;
                                            }else{
                                                referenceRadius = currentRadius;
                                            }
                                        }

                                    }

                                }
                                //                            else if(palmPoints.size() == 2 && distFromOrigin < 10.0*currentRadius){
                                //                                line(contoursMat2, currentPalmOrigin, currentCenter, Scalar(255,255,255), 3);
                                //                                palmFound = true;
                                //                                circle(imageCopy, currentCenter, currentRadius, Scalar(0,0,255), 3);
                                //                                x = currentCenter.x;
                                //                                y = currentCenter.y;
                                //                                targetFound = true;
                                //                                noiseIterations = 0;
                                //                            }
                                else{
                                    line(contoursMat2, currentPalmOrigin, currentCenter, Scalar(0,0,255), 3);
                                    //cout<<"Bad Iteration";

                                    //waitKey(0);
                                    noiseIterations++;
                                }

                                //putText(contoursMat2, "Palm!", currentCenter, 1,2, Scalar(255,255,255));
                                //putText(imageCopy, "Palm Found", currentCenter, 1,2, Scalar(255,255,255));
                            }else{

                            }

                        }

                        //fingertip detection
                        //this variable is used for detecting the closed hand
                        vector <Point> tipPoints;

                        if(true){ //ignore initiatial condition. It used to be palmFound == true
                            //draw fingers;



                            for(int i = 0; i< defectIndices.size(); i++){
                                int startPindex = (defect[index][(defectIndices[i])][0]);
                                Point startP = approximatedContourShapes[index][startPindex];

                                int innerP1index = (defect[index][(defectIndices[i])][2]);
                                Point innerP1 = approximatedContourShapes[index][innerP1index];


                                int nextEndPIndex = (defect[index][(defectIndices[(i+1)%defectIndices.size()])][1]);
                                Point NEXTendP = approximatedContourShapes[index][nextEndPIndex];

                                int nextInnerP1Index = (defect[index][(defectIndices[(i+1)%defectIndices.size()])][2]);
                                Point NEXTinnerP1 = approximatedContourShapes[index][nextInnerP1Index];

                                //cout<<"nextEndPIndex: "<<nextEndPIndex<<" startPIndex :"<<startPindex<<"\n";


                                // cout<<"StartP index: "<<startPindex<<" InnerP index :"<<innerP1index<<"\n";
                                int differenceSideOne = (innerP1index - startPindex)/2;
                                int differenceSideTwo = (nextEndPIndex- nextInnerP1Index)/2;








                                vector<Point> testContourFinger1;
                                vector<Point> testContourFinger2;
                                bool curvy = true;
                                int badPoints = 0;

                                for (int i = 1; i< differenceSideOne - 1; i++){
                                    Point testPoint = approximatedContourShapes[index][startPindex + i];
                                    Point front = approximatedContourShapes[index][startPindex + i+1];
                                    Point back = approximatedContourShapes[index][startPindex + i-1];
                                    if(angleBetween3Points(testPoint, front, back)< ((10.0/18.0)*(3.1415))){
                                        //circle(contoursMat2, testPoint, 2, Scalar(0,0,255), 4);
                                        //badPoints++;
                                    }
                                    else{

                                        //circle(contoursMat2, testPoint, 2, Scalar(255,255,255), 4);

                                    }
                                    testContourFinger1.insert(testContourFinger1.begin(), testPoint);

                                }

                                for (int i = 1; i< differenceSideTwo - 1; i++){
                                    Point testPoint = approximatedContourShapes[index][nextEndPIndex- i];
                                    Point front = approximatedContourShapes[index][nextEndPIndex - (i+1)];
                                    Point back = approximatedContourShapes[index][nextEndPIndex - (i-1)];

                                    if(angleBetween3Points(testPoint, front, back)< ((10.0/18.0)*(3.1415))){
                                        // circle(contoursMat2, testPoint, 2, Scalar(0,0,255), 4);
                                        //badPoints++;
                                    }
                                    else{

                                        // circle(contoursMat2, testPoint, 2, Scalar(255,255,255), 4);
                                    }

                                    testContourFinger2.insert(testContourFinger2.begin(), testPoint);


                                }











                                Point endP = approximatedContourShapes[index][(defect[index][(defectIndices[i])][1])];




                                int halfWayInd = ((int)(ceil(((double)(defect[index][(defectIndices[i])][2] - defect[index][(defectIndices[i])][0]))/2.0))) + defect[index][(defectIndices[i])][0];
                                Point halfWay = approximatedContourShapes[index][halfWayInd];

                                int quarterWayInd = ((int)(ceil(((double)(halfWayInd - defect[index][(defectIndices[i])][2]))/2.0))) + defect[index][(defectIndices[i])][2];
                                Point quarterWay = approximatedContourShapes[index][quarterWayInd];


                                int NEXThalfWayInd = ((int)(ceil(((double)((defect[index][(defectIndices[(i+1)%defectIndices.size()])][1]) - (defect[index][(defectIndices[(i+1)%defectIndices.size()])][2]) ))/2.0))) + (defect[index][(defectIndices[(i+1)%defectIndices.size()])][2]);
                                Point NEXThalfWay = approximatedContourShapes[index][NEXThalfWayInd];

                                int NEXTquarterWayInd = ((int)(ceil(((double)(NEXThalfWayInd - defect[index][(defectIndices[(i+1)%defectIndices.size()])][2]))/2.0))) + defect[index][(defectIndices[(i+1)%defectIndices.size()])][2];
                                Point NEXTquarterWay = approximatedContourShapes[index][NEXTquarterWayInd];








                                Point tipPoint = midPoint(startP,NEXTendP);
                                tipPoints.insert(tipPoints.begin(), tipPoint);
                                double tipWidth = dist(startP, NEXTendP);


                                //initial tip test
                                Point midLeft = approximatedContourShapes[index][startPindex + differenceSideOne];
                                Point midRight = approximatedContourShapes[index][nextInnerP1Index + differenceSideTwo];
                                if(angleBetween3Points(tipPoint, midLeft, midRight)<(3.1415/4.0)){
                                    circle(contoursMat3, tipPoint, 10, Scalar(255,25,125), 3);
                                }



                                //                                        cv::circle(contoursMat2, tipPoint, 3, Scalar(100,255,100), 3);
                                //                                        cv::line(contoursMat2, tipPoint, innerP1, Scalar(255,255,0), 3);
                                //                                        cv::line(contoursMat2, tipPoint, NEXTinnerP1, Scalar(255,255,0), 3);
                                double line1 = dist(tipPoint, innerP1);
                                double line2 = dist(tipPoint, NEXTinnerP1);
                                double fingerLine = dist(tipPoint, currentCenter);

                                Point fingerBase = midPoint(innerP1, NEXTinnerP1);

                                //code to calculate finger angle relative to palm
                                Point aboveCenter(currentCenter.x, currentCenter.y - currentRadius);





                                //where does the line intersect with the circle?
                                double diffx = fingerBase.x - currentCenter.x;
                                double diffy = fingerBase.y - currentCenter.y;
                                double radiusDiff = sqrt((pow(diffx,2) + pow(diffy, 2)));
                                double ratioX = diffx/radiusDiff;
                                double ratioY = diffy/radiusDiff;
                                double radiusShift = currentRadius - radiusDiff;
                                double shiftX = radiusShift*ratioX;
                                double shiftY = radiusShift*ratioY;
                                Point shiftedFingerBase(fingerBase.x + shiftX, fingerBase.y + shiftY);
                                circle(contoursMat2, shiftedFingerBase, 3, Scalar(200,50,200), 3);
                                line(contoursMat2, tipPoint, shiftedFingerBase, Scalar(244,43,29),3);
                                line(contoursMat2, tipPoint, fingerBase, Scalar(244,190,45),3);

                                double distFromPalm = dist(tipPoint, shiftedFingerBase);

                                //end of code

                                //Move palm points to saved







                                if((angleBetween3Points(tipPoint, innerP1, NEXTinnerP1) < (3.1415/finger_angle_divisor_tracking))){

                                    if(palmFound && palmPoints.size() >= 3){
                                        //code for finger mouse parameter initialization
                                        palmTrackingCenter = currentCenter;
                                        //end of finger mouse code


                                        //cout<<"Palm is present!\n";
                                        //Import if statement has been commented out.
                                        //if((tipWidth < (line1/1.0)) &&(tipWidth < (line2/1.0))){

                                        double dist1 = dist(innerP1, currentCenter);
                                        double dist2 = dist(NEXTinnerP1, currentCenter);
                                        double dist3 = dist(innerP1, NEXTinnerP1);
                                        double baseDist = dist(fingerBase, currentCenter);
                                        double fingerLength = dist(fingerBase, tipPoint);
                                        double fingerToCenter = dist(tipPoint, currentCenter);

                                        double orientationAngle = angleBetween3Points(currentCenter, tipPoint, aboveCenter);


                                        //double angle2 = angleBetween3Points(shiftedFingerBase, currentCenter, tipPoint);
                                        Point bellowShiftedBase(fingerBase.x,fingerBase.y + currentRadius);

                                        bool impossibleFinger = false;

                                        double positionAngle = angleBetween3Points(currentCenter, aboveCenter,fingerBase);
                                        double fingerOrientationAngle = angleBetween3Points(fingerBase, tipPoint, bellowShiftedBase);



                                        if(bellowShiftedBase.y < imageCopy.rows){
                                            //                                        if (angleBetween3Points(fingerBase, tipPoint, bellowShiftedBase) < 3.1415/2.0){
                                            //                                            impossibleFinger = true;
                                            //                                        }else
                                            if (fingerOrientationAngle <= 3.1415/2.0 && positionAngle <= 3.1415/3.0){
                                                //if (positionAngle < (3.1415/3.0)){
                                                impossibleFinger = true;
                                                //}
                                            }
                                        }

                                        //if (angle1 < 3.1415/2.5){
                                        //    if(angle2 < 3.1415/1.9){
                                        //impossibleFinger = true;
                                        //    }
                                        //}

                                        bool clearance = true;
                                        if(fingersFound.size() >= 1){

                                            Point prevTip = fingersFound[0];
                                            double proxDist = dist(tipPoint, prevTip);
                                            if(proxDist < currentRadius/proxDist_palmRadius_divisor){
                                                clearance = false;
                                            }

                                        }



                                        bool wideFinger = false;

                                        if(angleBetween3Points(currentCenter, innerP1, NEXTinnerP1)>((10.0/18.0)*3.1415))wideFinger = true;
                                        int margin = 20;


                                        if((tipPoint.x >= 0 && tipPoint.x <= margin) || (tipPoint.x >= (imageCopy.cols -margin)  && tipPoint.x <= imageCopy.cols)  ){

                                        }else{
                                            if((tipPoint.y >= 0 && tipPoint.y <= margin) || (tipPoint.y >= (imageCopy.rows -margin)  && tipPoint.y <= imageCopy.rows)  ){

                                            }else{
                                                //putText(contoursMat2, "Is Tip", tipPoint, 1, 1.0, Scalar(255,0,255),2);


                                                //if(((dist1 <= 6.0*currentRadius && dist2 <= 1.0*currentRadius)||(dist2 <= 6.0*currentRadius && dist1 <= 1.0*currentRadius))){

                                                if(((dist2 <= 1.0*currentRadius)||(dist1 <= 1.0*currentRadius))){
                                                    putText(contoursMat2, "Is Tip", tipPoint, 1, 1.0, Scalar(255,0,255),2);


                                                    if((tipWidth < dist3*tip_width_fingerBase_range)&&(tipWidth < (tip_width_palmRadius_range*currentRadius))){



                                                        if((baseDist<(baseDist_palmRadius_range*currentRadius))&&(fingerLength>(currentRadius/fingerLength_palmRadius_divisor))){
                                                            //putText(contoursMat2, "Is Tip", tipPoint, 1, 1.0, Scalar(255,0,255),2);

                                                            if(fingerToCenter > (fingerToCenter_minimum_range*currentRadius)){
                                                                //if(distFromPalm >= 0.3*currentRadius && distFromPalm <= 3.0*currentRadius){
                                                                if(orientationAngle < (3.1415/1.7) && impossibleFinger == false && clearance){


                                                                    //Point prevFingerTip = fingersFound[0];

                                                                    cv::circle(contoursMat2, tipPoint, 8, Scalar(255,255,255), 6);
                                                                    cv::circle(imageCopy, tipPoint, 8, Scalar(0,0,255), 6);
                                                                    cv::line(imageCopy, tipPoint, currentCenter, Scalar(0,0,255), 3);
                                                                    fingersFound.insert(fingersFound.begin(), tipPoint);
                                                                    fingerTipWidth.insert(fingerTipWidth.begin(), tipWidth);
                                                                }
                                                                //}
                                                            }
                                                        }
                                                        //                                                        else if((dist1 <0.9*currentRadius)||(dist2 < 0.9*currentRadius)){
                                                        //                                                            cv::circle(imageCopy, tipPoint, 8, Scalar(255,0,0), 6);
                                                        //                                                            cv::line(imageCopy, tipPoint, currentCenter, Scalar(255,0,0), 3);
                                                        //                                                        }
                                                    }

                                                }
                                            }

                                        }
                                        //}
                                    }else{
                                        // cout<<"This section executes!!!!\n";
                                        //cv::circle(contoursMat2, tipPoint, 8, Scalar(255,255,255), 6);
                                        //cv::circle(imageCopy, tipPoint, 8, Scalar(0,0,255), 6);
                                    }





                                }











                                // if((tipDistance < halfWayDist1)&&(tipDistance < halfWayDist2)){


                            }



                        }





                    }


                }


                if(targetFound == false && noiseIterations >=3){
                    //exit the tracking loop
                    //imshow("Target Lost!",contoursMat2);
                    //code to record the data
                    if(recordData){
                    dataAcquisition.processFrame(-1);
                    }
                    //waitKey(0);

                    //reset the mouse buttons
                    if(middlePressed){
                        middlePressed = false;
                        mouse_event(MOUSEEVENTF_MIDDLEUP,0,0,0,0);




                    }
                    if(LMBpressed){
                        LMBpressed = false;
                        mouse_event(MOUSEEVENTF_LEFTUP,0,0,0,0);


                    }
                    if(RMBpressed)RMBpressed = false;

                    break;
                }


                double fieldDimension = imageCopy.cols/20;

                //include code for mouse control via template matching.
                if(targetFound){

                    //code added to record the data
                    if(recordData){
                    dataAcquisition.processFrame(fingersFound.size());
                    }



                    if(fingersFound.size() == 2 && !fingerMouseActive){
                        //Record the template to match
                        cout<<"Finger Mouse Activated!!!\n";
                        fingerMouseActive = true;
                        double dim = fingerTipWidth[0];
                        if(dim == 0)dim++;
                        double dim2 = fingerTipWidth[1];
                        if(dim2 == 0)dim2++;
                        distBetweenFingers = fingersFound[1].y - fingersFound[0].y;
                        indexClickMargin = index_click_margin_factor*distBetweenFingers;
                        indexDoubleClickMargin = index_doubleclick_margin_factor*distBetweenFingers;
                        thumbClickMargin = thumb_click_margin_factor*distBetweenFingers;

                        if(fingersFound[0].x-(dim) > 0 && fingersFound[0].x+(dim) < imageCopy_2.rows && fingersFound[0].y-(dim) > 0 &&  fingersFound[0].y+(dim) < imageCopy_2.cols){
                            trackingTemplateIndexFinger = imageCopy_2(cv::Rect(fingersFound[0].x-(dim), fingersFound[0].y-(dim), 2.0*dim, 2.0*dim));
                            //imshow("Tracking Template Used", trackingTemplate);
                            IndexFingerTrackingCenter = fingersFound[0];



                        }else{
                            fingerMouseActive = false;
                        }

                        if(fingersFound[1].x-(dim2) > 0 && fingersFound[1].x-(dim2) < imageCopy_2.rows && fingersFound[1].y-(dim2) > 0 &&  fingersFound[1].y-(dim2) < imageCopy_2.cols){
                            trackingTemplateThumb = imageCopy_2(cv::Rect(fingersFound[1].x-(dim2), fingersFound[1].y-(dim2), 2.0*dim2, 2.0*dim2));
                            //imshow("Tracking Template Used", trackingTemplate);

                            ThumbTrackingCenter = fingersFound[1];

                        }else{
                            fingerMouseActive = false;
                        }

                        //                        trackingTemplatePalm = imageCopy_2(cv::Rect(palmTrackingCenter.x - (dim), palmTrackingCenter.y - dim, 2.0*dim, 2.0*dim));

                        if(LMBpressed){
                            LMBpressed = false;
                            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                        }

                        if(middlePressed){
                            middlePressed = false;
                            mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0);
                        }
                        if(RMBpressed)RMBpressed = false;

                    }else if(fingersFound.size() == 5 && fingerMouseActive){
                        //deactivate
                        if(LMBpressed){
                            LMBpressed = false;
                            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                        }

                        if(middlePressed){
                            middlePressed = false;
                            mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0);
                        }

                        if(RMBpressed)RMBpressed = false;

                        cout<<"Finger Mouse Deactivated!!!\n";
                        fingerMouseActive = false;


                    }
                    //else if(fingersFound.size() == 5 && scrollMode){
                      //  scrollMode = false;


                    //}

                    //else if(fingersFound.size() == 0 && !scrollMode){

                        //Point templateOrigin = savedPalmPoints[0];


                        //if(templateOrigin.x - (fieldDimension/2) > 0 && templateOrigin.y - (fieldDimension/2) > 0 && templateOrigin.x + (fieldDimension/2) < imageCopy.cols && templateOrigin.y + (fieldDimension/2) < imageCopy.rows){
                        //trackingPalmTemplateScrolling = imageCopy_2(cv::Rect(templateOrigin.x - (fieldDimension/2), templateOrigin.y - fieldDimension/2, fieldDimension, fieldDimension));
                        //imshow("Scroll Template Used", trackingPalmTemplateScrolling);
                        //scrollMode = true;
                        //previousPalmTemplatePosition = templateOrigin;
                        //currentPalmTemplatePosition = templateOrigin;
                        //}
                        //putText(imageCopy, "Scroll Mode", Point(imageCopy.cols/2, imageCopy.rows/2), 1, 2, Scalar(0,30,140),3);
                    //}



                    if(fingerMouseActive && useFingerMouse){
                        Point startRec(imageCopy.cols/6, imageCopy.rows/12);
                        Point endRec(9*imageCopy.cols/10, 4*imageCopy.rows/10);


                        //define region of the image where stuff happens

                        rectangle(imageCopy, startRec, endRec, Scalar(50, 89, 85),3);


                        if(LMBhold == false){



                            if (fingersFound.size()==1 && LMBpressed == false){

                            }else if(fingersFound.size() == 1 && LMBpressed == true && doubleClicked == false){



                            }else if(fingersFound.size() ==2 && LMBpressed == false && doubleClicked == true){


                            }else if(fingersFound.size() ==4 && LMBhold == false && dragSwitch == false){

                                //                            imshow("Contours Mat 3 fingers", contoursMat2);
                                //                            waitKey(0);
//                                waitKey(50);
//                                if(LMBpressed == true){
//                                    LMBpressed = false;
//                                }

//                                dragSwitch = true;



//                                LMBhold = true;

                                //mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);

                            }else if (fingersFound.size() == 2 && dragSwitch == true){
                                dragSwitch = false;
                            }

                        }else{
                            Point imageCenter(imageCopy.cols/2, imageCopy.rows/2);
                            //if(dragSwitch == true){
                            putText(imageCopy, "Drag Activated", imageCenter, 1, 2.0, Scalar(0,100,0), 2);
                            //}

                            if(fingersFound.size() == 2 && dragSwitch == true){
                                Point centerDown(imageCenter.x + 30, imageCenter.y + 30);
                                dragSwitch = false;
                                //putText(imageCopy, "Drag Activated drag Switch is false", centerDown, 1, 2.0, Scalar(0,100,0), 2);
                            }else if(fingersFound.size() == 4 && dragSwitch == false){
                                //putText(imageCopy, "Drag Activated Not!!!!!!", imageCenter, 1, 2.0, Scalar(0,100,0), 2);

                                //                                                    if(LMBpressed){
                                //                                                        LMBpressed = false;
                                //                                                        mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);

                                //                                                    }

                                //dragSwitch = true;
                                //LMBhold = false;
                            }else if (fingersFound.size() == 1 && LMBpressed == false){
                                //                                                    LMBpressed = true;
                                //                                                    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);

                            }else if (fingersFound.size() == 2 && LMBpressed == true){
                                //                                                    LMBpressed = false;
                                //                                                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);

                            }else if (fingersFound.size() == 0){
                                //                            imshow("Contours Mat2 0 fingers", contoursMat2);
                                //                            waitKey(0);
                            }
                        }
                        //IDEA move only when there are 2 fingers present

                        cout<<"Finger Mouse Tracking!!! fingers:"<<fingersFound.size()<<"\n";

                        //index finger
                        Mat result;
                        double minVal; double maxVal; Point minLoc; Point maxLoc;
                        Point minLocPrime;

                        //thumb
                        Mat result2;
                        double minVal2; double maxVal2; Point minLoc2; Point maxLoc2;
                        Point minLocPrime2;





                        //cout<<"Executes this far!!!!\n";

                        if(fingersFound.size() == 2||fingersFound.size() == 3||fingersFound.size() == 1||(fingersFound.size() == 0 && LMBhold)){

                            //Index Finger Variables
                            Point searchingROI_SP = Point(IndexFingerTrackingCenter.x - (1.5*fieldDimension), IndexFingerTrackingCenter.y - (1.5*fieldDimension));
                            Point searchingROI_EP = Point(IndexFingerTrackingCenter.x + 2.5*(fieldDimension), IndexFingerTrackingCenter.y+ 2.5*(fieldDimension));

                            //Thumb Variables
                            Point searchingROI_Thumb_SP = Point(ThumbTrackingCenter.x - (2.5*fieldDimension), ThumbTrackingCenter.y - (2.5*fieldDimension));
                            Point searchingROI_Thumb_EP = Point(ThumbTrackingCenter.x + 3.5*(fieldDimension), ThumbTrackingCenter.y+ 3.5*(fieldDimension));




                            //Bound the rectangle withing the image.

                            // index finger search area bounding
                            if(searchingROI_SP.x < 0)searchingROI_SP.x = 0;
                            if(searchingROI_SP.y < 0)searchingROI_SP.y = 0;


                            if(searchingROI_EP.x >= imageCopy.cols)searchingROI_EP.x = imageCopy.cols-1;
                            if(searchingROI_EP.y >= imageCopy.rows)searchingROI_EP.y = imageCopy.rows-1;



                            //thumb search area bounding
                            if(searchingROI_Thumb_SP.x < 0)searchingROI_Thumb_SP.x = 0;
                            if(searchingROI_Thumb_SP.y < 0)searchingROI_Thumb_SP.y = 0;


                            if(searchingROI_Thumb_EP.x >= imageCopy.cols)searchingROI_Thumb_EP.x = imageCopy.cols-1;
                            if(searchingROI_Thumb_EP.y >= imageCopy.rows)searchingROI_Thumb_EP.y = imageCopy.rows-1;


                            //cout<<"Executes this far tooo!!!!\n";

                            cout<<"Executes this far 2!!!!\n";




                            //end of bounding
                            //Error Checking for index finger
                            bool avoidError = false;
                            if(searchingROI_SP.x == searchingROI_EP.x || searchingROI_SP.y == searchingROI_EP.y ){
                                cout<<"Error incoming!\n";
                                //                            searchingROI_SP.x = 0;
                                //                            searchingROI_SP.y = 0;
                                //                            searchingROI_EP.x = imageCopy.cols-1;
                                //                            searchingROI_EP.y = imageCopy.rows-1;

                                searchingROI_SP.x = imageCopy.cols/2 - 2*fieldDimension;
                                searchingROI_SP.y = imageCopy.rows/2 - 2*fieldDimension;

                                searchingROI_EP.x = imageCopy.cols/2 + 2*fieldDimension;
                                searchingROI_EP.y = imageCopy.rows/2 + 2*fieldDimension;

                                //searchingROI_SP.y = searchingROI_SP.y - 3*trackingTemplateIndexFinger.rows;
                                //searchingROI_SP.x = searchingROI_SP.x - 3*trackingTemplateIndexFinger.cols;

                                //avoidError = true;
                            }

                            //Error checking for thumb
                            bool avoidError2 = false;
                            if(searchingROI_Thumb_SP.x == searchingROI_Thumb_EP.x || searchingROI_Thumb_SP.y == searchingROI_Thumb_EP.y ){
                                cout<<"Error incoming!\n";
                                //                            searchingROI_SP.x = 0;
                                //                            searchingROI_SP.y = 0;
                                //                            searchingROI_EP.x = imageCopy.cols-1;
                                //                            searchingROI_EP.y = imageCopy.rows-1;

                                searchingROI_Thumb_SP.y = searchingROI_Thumb_SP.y - 3*trackingTemplateThumb.rows;
                                searchingROI_Thumb_SP.x = searchingROI_Thumb_SP.x - 3*trackingTemplateThumb.cols;

                                avoidError2 = true;
                            }



                            //searching for the index finger
                            Mat searchingROI = imageCopy_2(Rect(searchingROI_SP, searchingROI_EP));
                            cout<<"Executes this far 3!!!!\n";
                            //                        imshow("Search Area", searchingROI);
                            if(avoidError == false){
                                cout<<"Rows of Searching ROI:"<<searchingROI.rows<<endl;
                                cout<<"Cols of Searching ROI:"<<searchingROI.cols<<endl;
                                cout<<"Rows of Template:"<<trackingTemplateIndexFinger.rows<<endl;
                                cout<<"Cols of Template:"<<trackingTemplateIndexFinger.cols<<endl;

                                //if(searchingROI.rows + trackingTemplateIndexFinger.rows < imageCopy_2.rows && searchingROI.cols + trackingTemplateIndexFinger.cols < imageCopy_2.cols ){

                                result.create(searchingROI.rows, searchingROI.cols, searchingROI.type());
                                matchTemplate(searchingROI, trackingTemplateIndexFinger, result, CV_TM_SQDIFF_NORMED);

                                cout<<"Executes this far 3.5!!!!\n";
                                normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
                                //}

                            }
                            cout<<"Executes this far 4!!!!\n";

                            //Point matchLoc;

                            minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
                            //find the template in the current image;
                            minLocPrime= Point(searchingROI_SP.x+minLoc.x, searchingROI_SP.y +minLoc.y);
                            if(minLocPrime.x>= imageCopy.cols)minLocPrime.x = imageCopy.cols -1;
                            if(minLocPrime.y>= imageCopy.rows)minLocPrime.y = imageCopy.rows -1;


                            cout<<"Executes this far 5!!!!\n";


                            //searching for the thumb
                            Mat searchingROIThumb = imageCopy_2(Rect(searchingROI_Thumb_SP, searchingROI_Thumb_EP));
                            if(avoidError2 == false){
                                result2.create(searchingROIThumb.rows, searchingROIThumb.cols, searchingROIThumb.type());
                                matchTemplate(searchingROIThumb, trackingTemplateThumb, result2, CV_TM_SQDIFF_NORMED);
                                normalize( result2, result2, 0, 1, NORM_MINMAX, -1, Mat() );

                            }

                            minMaxLoc( result2, &minVal2, &maxVal2, &minLoc2, &maxLoc2, Mat() );
                            //find the template in the current image;
                            minLocPrime2= Point(searchingROI_Thumb_SP.x+minLoc2.x, searchingROI_Thumb_SP.y +minLoc2.y);
                            if(minLocPrime2.x>= imageCopy.cols)minLocPrime2.x = imageCopy.cols -1;
                            if(minLocPrime2.y>= imageCopy.rows)minLocPrime2.y = imageCopy.rows -1;

                            cout<<"Executes this far 6!!!!\n";






                            bool shouldMove = false;
                            //if(IndexFingerTrackingCenter.x < minLocPrime.x - 1 || IndexFingerTrackingCenter.x > minLocPrime.x + 1||IndexFingerTrackingCenter.y < minLocPrime.y - 1||IndexFingerTrackingCenter.y > minLocPrime.y + 1){
                            shouldMove = true;
                            IndexFingerTrackingCenter = minLocPrime;
                            //}



                            //stabilize the thumb ROI
                            //imshow("Thumb Before", trackingTemplateThumb);
                            //double dim2 = trackingTemplateThumb.cols;
                            //trackingTemplateThumb = imageCopy_2(cv::Rect(minLocPrime2.x, minLocPrime2.y, dim2, dim2));
                            //imshow("Thumb After", trackingTemplateThumb);
                            //waitKey(0);
                            //imshow("Tracking Template Used", trackingTemplate);

                            ThumbTrackingCenter = minLocPrime2;

                            cout<<"Executes this far 7!!!!\n";






                            //Paint the index finger on the image
                            Point targetRect = Point(minLocPrime.x + (trackingTemplateIndexFinger.cols), minLocPrime.y+(trackingTemplateIndexFinger.rows));
                            if(targetRect.x < imageCopy.cols && targetRect.y < imageCopy.rows){
                                rectangle(imageCopy, Point(minLocPrime.x, minLocPrime.y), targetRect,Scalar(0,0,130),3 );
                            }

                            Point borderStart = Point(minLocPrime.x - 2.5*(fieldDimension), minLocPrime.y - 0.6*(fieldDimension));
                            Point borderEnd = Point(minLocPrime.x + 2.5*(fieldDimension), minLocPrime.y+ indexClickMargin);

                            if(borderStart.x >= 0  && borderStart.y >= 0){
                                if(borderEnd.x < imageCopy.cols && borderEnd.y < imageCopy.rows){
                                    rectangle(imageCopy, borderStart, borderEnd,Scalar(0,0,130),3 );
                                }
                            }


                            Point borderStartDouble = Point(minLocPrime.x - 2.0*(fieldDimension), minLocPrime.y - 0.3*(fieldDimension));
                            Point borderEndDouble = Point(minLocPrime.x + 2.0*(fieldDimension), minLocPrime.y+ indexDoubleClickMargin);



                            if(borderStartDouble.x >= 0  && borderStart.y >= 0){
                                if(borderEndDouble.x < imageCopy.cols && borderEndDouble.y < imageCopy.rows){
                                    rectangle(imageCopy, borderStartDouble, borderEndDouble,Scalar(0,0,130),3 );
                                }
                            }


                            cout<<"Executes this far 8!!!!\n";


                            //Paint the thumb on the image
                            Point targetRect2 = Point(minLocPrime2.x + (trackingTemplateThumb.cols), minLocPrime2.y+(trackingTemplateThumb.rows));
                            if(targetRect2.x < imageCopy.cols && targetRect2.y < imageCopy.rows){
                                rectangle(imageCopy, Point(minLocPrime2.x, minLocPrime2.y), targetRect2,Scalar(0,0,130),3 );
                            }

                            Point borderStart2 = Point(minLocPrime2.x - 2*(fieldDimension), minLocPrime2.y - thumbClickMargin);
                            Point borderEnd2 = Point(minLocPrime2.x + 3*(fieldDimension), minLocPrime2.y+ 2.5*(fieldDimension));

                            if(borderStart2.x >= 0  && borderStart2.y >= 0){
                                if(borderEnd2.x < imageCopy.cols && borderEnd2.y < imageCopy.rows){
                                    rectangle(imageCopy, borderStart2, borderEnd2,Scalar(0,0,130),3 );
                                }
                            }


                            //draw the mode switch line relative to the index finger
                            Point modeSwitchLinePoint1 = Point(minLocPrime2.x - 2*(fieldDimension), minLocPrime.y+ 1.16*distBetweenFingers);
                            Point modeSwitchLinePoint2 = Point(minLocPrime2.x + 3*(fieldDimension), minLocPrime.y+ 1.16*distBetweenFingers);

                            //boundary checking for mode switch line
                            if(modeSwitchLinePoint1.x < 0)modeSwitchLinePoint1.x = 0;
                            if(modeSwitchLinePoint1.x >= imageCopy.cols)modeSwitchLinePoint1.x = imageCopy.cols - 1;
                            if(modeSwitchLinePoint1.y < 0)modeSwitchLinePoint1.y = 0;
                            if(modeSwitchLinePoint1.y >= imageCopy.rows)modeSwitchLinePoint1.y = imageCopy.rows - 1;

                            if(modeSwitchLinePoint2.x < 0)modeSwitchLinePoint2.x = 0;
                            if(modeSwitchLinePoint2.x >= imageCopy.cols)modeSwitchLinePoint2.x = imageCopy.cols - 1;
                            if(modeSwitchLinePoint2.y < 0)modeSwitchLinePoint2.y = 0;
                            if(modeSwitchLinePoint2.y >= imageCopy.rows)modeSwitchLinePoint2.y = imageCopy.rows - 1;




                            //if(useRightClick){
                            line(imageCopy, modeSwitchLinePoint1, modeSwitchLinePoint2, Scalar(244,190,180),2);
                            //}

                           //Draw the right click line relative to the index finger
                            Point rightClickPoint1 = Point(minLocPrime2.x - 2*(fieldDimension), minLocPrime.y+ 1.28*distBetweenFingers);
                            Point rightClickPoint2 = Point(minLocPrime2.x + 3*(fieldDimension), minLocPrime.y+ 1.28*distBetweenFingers);

                            //boundary checking for right click line
                            if(rightClickPoint1.x < 0)rightClickPoint1.x = 0;
                            if(rightClickPoint1.x >= imageCopy.cols)rightClickPoint1.x = imageCopy.cols - 1;
                            if(rightClickPoint1.y < 0)rightClickPoint1.y = 0;
                            if(rightClickPoint1.y >= imageCopy.rows)rightClickPoint1.y = imageCopy.rows - 1;

                            if(rightClickPoint2.x < 0)rightClickPoint2.x = 0;
                            if(rightClickPoint2.x >= imageCopy.cols)rightClickPoint2.x = imageCopy.cols - 1;
                            if(rightClickPoint2.y < 0)rightClickPoint2.y = 0;
                            if(rightClickPoint2.y >= imageCopy.rows)rightClickPoint2.y = imageCopy.rows - 1;

                            line(imageCopy, rightClickPoint1, rightClickPoint2, Scalar(244,190,180),2);




                            //paint the click
                            CLICK = false;

                            if(borderEnd2.x >= borderStart.x && borderEnd.y >= borderStart2.y){
                                Point imageCenter = Point(imageCopy.cols/2, imageCopy.rows/2);
                                if(LMBhold){
                                    putText(imageCopy, "DRAG CLICK!", imageCenter, 1, 2.0, Scalar(0,0,0),2);
                                }else{
                                putText(imageCopy, "CLICK!", imageCenter, 1, 2.0, Scalar(0,0,0),2);
                                }
                                CLICK = true;
                            }

                            //double click code
                            if(borderEnd2.x >= borderStartDouble.x && borderEndDouble.y >= borderStart2.y){
                                Point imageCenter = Point(imageCopy.cols/2, imageCopy.rows/2);
                                if(LMBhold){
                                    putText(imageCopy, "SCROLL!", Point(imageCenter.x - fieldDimension,imageCenter.y - fieldDimension), 1, 2.0, Scalar(0,0,0),2);
                                }else{
                                        putText(imageCopy, "DOUBLE CLICK!", Point(imageCenter.x - fieldDimension,imageCenter.y - fieldDimension), 1, 2.0, Scalar(0,0,0),2);
                                }
                                doubleClicked = true;
                            }



                            switchModeClick = false;
                            //RESET Right Mouse Button Click only when back to natural position;
                            if(borderStart2.y < modeSwitchLinePoint1.y){
                                if(RMBCLICK){
                                    modeSwitchTriggered = false;
                                    RMBCLICK = false;
                                }
                            }

                            //paint the right click
                            if( borderStart2.y >=  rightClickPoint1.y && doubleClicked == false && CLICK == false){
                                 Point imageCenter = Point(imageCopy.cols/2, imageCopy.rows/2);
                                 putText(imageCopy, "Right Click!!", Point(imageCenter.x - fieldDimension,imageCenter.y - fieldDimension), 1, 2.0, Scalar(0,0,0),2);
                                 //switchModeClick = true;
                                 RMBCLICK = true;




                            }
                            //paint the switch click
                            else if(borderStart2.y >=  modeSwitchLinePoint1.y && doubleClicked == false && CLICK == false && RMBCLICK == false){
                                Point imageCenter = Point(imageCopy.cols/2, imageCopy.rows/2);
                                //if(!LMBhold){
                                //if(useRightClick){
                                    putText(imageCopy, "Switch!!", Point(imageCenter.x - fieldDimension,imageCenter.y - fieldDimension), 1, 2.0, Scalar(0,0,0),2);
                                    switchModeClick = true;
                                //}

                                //}
                            }





                            if(LMBhold && doubleClicked && !scrollMode) {
                                scrollMode = true;
                                if(LMBpressed){
                                    LMBpressed = false;
                                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                                    //mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                                    //mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                                }

                                middlePressed = true;
                                mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0);

                            }else if(LMBhold && doubleClicked == false && scrollMode ){
                                scrollMode = false;
                                middlePressed = false;
                                mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0);

                            }

                            else if(CLICK == true && LMBpressed == false){
                                LMBpressed = true;
                                if(LMBhold ==true){
                                    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                                }
                            }else if(CLICK == false && LMBpressed == true){
                                LMBpressed = false;
                                if(LMBhold && doubleClicked == false){
                                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                                }else if (doubleClicked == false){
                                    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);


                                }else if (doubleClicked == true){
                                    doubleClicked = false;
                                    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                                    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);

                                }



                            //code for switching modes from click to drag and vice versa.
                            }else if(modeSwitchTriggered == false && switchModeClick == true && RMBCLICK == false){
                                modeSwitchTriggered = true;



                            }else if (modeSwitchTriggered && switchModeClick == false && RMBCLICK == false){
                                modeSwitchTriggered = false;
                                LMBhold = !LMBhold;


                            //code for right click. Higher priority than switch Mode
                            }else if(RMBpressed== false && RMBCLICK == true){
                                RMBpressed= true;



                            }else if (RMBpressed && RMBCLICK == false){
                                RMBpressed = false;

                                mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);
                                mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0);
                                LMBhold = false;
                            }

                        }











                        //code for movement of cursor
                        if((((fingersFound.size() == 2 || fingersFound.size() == 3|| fingersFound.size() == 1) && CLICK == false) || LMBhold)&& !switchModeClick && !RMBpressed){
                            if(minLocPrime.x >= startRec.x && minLocPrime.y >= startRec.y && minLocPrime.x <= endRec.x && minLocPrime.y <= endRec.y){



                                Point startTarget = Point(minLocPrime.x, minLocPrime.y);
                                Point endTarget = Point(minLocPrime.x + (trackingTemplateIndexFinger.rows), minLocPrime.y+(trackingTemplateIndexFinger.cols));

                                if(endTarget.x >= imageCopy.cols)endTarget.x = imageCopy.cols -1;
                                if(endTarget.y >= imageCopy.rows)endTarget.y = imageCopy.rows -1;

                                rectangle(imageCopy, startTarget, endTarget,Scalar(23,134,130),3 );

                                double xnumerator = minLocPrime.x - startRec.x;
                                double ynumerator = minLocPrime.y - startRec.y;
                                double xdenominator = endRec.x - startRec.x;
                                double ydenominator = endRec.y - startRec.y;



                                //                            int xPosCursor = ((int)((((double)horizontal)/3.0 )* (xnumerator)/(xdenominator) + ((double)horizontal)/3.0));
                                //                            int yPosCursor = ((int)((((double)vertical)/3.0 )* (ynumerator)/(ydenominator) + ((double)vertical)/3.0));
                                int xPosCursor = ((int)((((double)horizontal))* (xnumerator)/(xdenominator)));
                                int yPosCursor = ((int)((((double)vertical) )* (ynumerator)/(ydenominator)));

                                if(yPosCursor > vertical){
                                    yPosCursor = vertical - 1;
                                }
                                //if(yPosCursor > vertical)yPosCursor = vertical;


                                SetCursorPos(xPosCursor, yPosCursor);
                            }
                        }


                        //better performance???
                        //trackingTemplate = imageCopy_2(cv::Rect(minLoc.x, minLoc.y, trackingTemplate.rows, trackingTemplate.cols));






                    }
//                    if(scrollMode && useFingerMouse){
//                        putText(imageCopy, "Scroll Mode", Point(imageCopy.cols/2, imageCopy.rows/2), 1, 2.0, Scalar(0,30,140));


//                        //variables for template searching

//                        //palm
//                        //                    Mat resultPalm;
//                        //                    double minVal3; double maxVal3; Point minLoc3; Point maxLoc3;
//                        //                    Point minLocPrime3;


//                        //                    Point searchingROI_SP = Point(previousPalmTemplatePosition.x - (1.5*fieldDimension), previousPalmTemplatePosition.y - (1.5*fieldDimension));
//                        //                    Point searchingROI_EP = Point(previousPalmTemplatePosition.x + 2.5*(fieldDimension), previousPalmTemplatePosition.y+ 2.5*(fieldDimension));

//                        //                     Mat searchingROI = imageCopy_2(Rect(searchingROI_SP, searchingROI_EP));



//                        //                    //search for template

//                        //                     //if(avoidError2 == false){
//                        //                         resultPalm.create(searchingROI.rows, searchingROI.cols, searchingROI.type());
//                        //                         matchTemplate(searchingROI, trackingPalmTemplateScrolling, resultPalm, CV_TM_SQDIFF_NORMED);
//                        //                         normalize( resultPalm, resultPalm, 0, 1, NORM_MINMAX, -1, Mat() );

//                        //                     //}

//                        //                     minMaxLoc( resultPalm, &minVal3, &maxVal3, &minLoc3, &maxLoc3, Mat() );
//                        //                     //find the template in the current image;
//                        //                     minLocPrime3= Point(searchingROI_SP.x+minLoc3.x, searchingROI_SP.y +minLoc3.y);
//                        //                     if(minLocPrime3.x>= imageCopy.cols)minLocPrime3.x = imageCopy.cols -1;
//                        //                     if(minLocPrime3.y>= imageCopy.rows)minLocPrime3.y = imageCopy.rows -1;

//                        //                     previousPalmTemplatePosition = minLocPrime3;



//                        //                    //draw template location
//                        //                     circle(imageCopy, previousPalmTemplatePosition, 3, Scalar(255,255,255), 3);



//                        //circle(imageCopy, palmTrackingCenter, 3, Scalar(244,180,0),3);
//                    }
                }





                //imshow("The segmentation with contours 1", contoursMat);


                //imshow("The segmentation with contours 2", contoursMat2);
                //imshow("Threshold Map", imageCopy_2);
                imshow("Augmented Image", imageCopy);
                if(recordData){
                    videoAcquisition.addFrame(imageCopy);
                }

                if(track == false){
                    cout<<"Track is False!\n";
                }

                //waitKey(0);





                int resp = waitKey(1);

                if(resp == 112){

                                    imshow("Contours Mat 2", contoursMat2);
                                    cout<<"Saved Palm Points:"<<savedPalmPoints.size()<<endl;
                                    cout<<"Target Found: "<<targetFound<<endl;
                                    cout<<"Noise Iterations: "<<noiseIterations<<endl;

                                    waitKey(0);
                }

                if(resp == 13){
                    std::cout<<"Quit program\n";
                    break;
                    //templateSelected = true;
                    //imageROI.copyTo(templateToMatch);

                };
            }
            //cout<<"Video is over!!!\n";




            int resp = waitKey(1);

            if(resp == 13){
                std::cout<<"Quit program\n";
                break;
                //templateSelected = true;
                //imageROI.copyTo(templateToMatch);

            }
        }

        dataAcquisition.close();
        videoAcquisition.close();
        cout<<"Program is finished\n";

        //cap.release();
        //waitKey(0);





        return 1;
    }



};



#endif // HANDTRACKER_H
