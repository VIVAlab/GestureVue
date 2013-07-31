#ifndef FILEEVALUATOR_H
#define FILEEVALUATOR_H

#include "iostream"
#include "fstream"
#include "math.h"


#include <string>


using namespace std;

string fileNameInput = "VideoProfiles/TestVideoSequences";
string fileNameResult = "VideoResults/TestVideoSequences";


class fileEvaluator
{
  public:
    fileEvaluator();
    void openFile(string inputFile);
    void processFrame(int input);
    void close();
    void startExperiment();

    void endExperiment();


  private:

    bool recordMasterFile = false;

    ifstream inFile;
    ofstream outFile;
    ofstream outMasterFile;
    string header;


    double successFrames;
    double totalFrames;

    //used for tracking if a target has been locked and how many successful frames the algorithm has produced;
    bool isLocked;
    double successFramesWhenLocked;
    double TotalFramesWhenLocked;
    string savedFileName;

    //used for calculating the average
    double sumSuccessRate;
    double sumSuccessRatesWhenLocked;
    double numberOfTrials;
    double numberOfTrialsThatLocked;
};


void fileEvaluator::startExperiment(){
    string masterFile = "VideoResults/ResultsMasterFile.csv";
    outMasterFile.open(masterFile.c_str());
    outMasterFile<<"Video Number, Success Rate, Success Rate With Target Locked,\n";

    recordMasterFile = true;
    sumSuccessRate = 0.0;
    sumSuccessRatesWhenLocked = 0.0;
    numberOfTrials = 0.0;
    numberOfTrialsThatLocked =0.0;
}



fileEvaluator::fileEvaluator(){

}


void fileEvaluator::openFile(string inputFile)    // This is the constructor definition
     // This is called an initialization list
{
    savedFileName = inputFile;
    //initialize
    successFrames = 0.0;
    totalFrames = 0.0;

    isLocked = false;
    successFramesWhenLocked = 0.0;
    TotalFramesWhenLocked = 0.0;





    stringstream videoStream;
    videoStream << fileNameInput;
    videoStream <<inputFile;
    videoStream<<".csv";
    string completeInputFileName = videoStream.str();


    inFile.open(completeInputFileName.c_str());


    stringstream videoStream2;
    videoStream2 << fileNameResult;
    videoStream2 <<inputFile;
    videoStream2 <<"Result.csv";
    string completeInputFileName2 = videoStream2.str();

    outFile.open(completeInputFileName2.c_str());



    getline(inFile, header);
    outFile<<"Frame Number, Number of Ground Truth Fingers, Finger Orientation, Number of Test Fingers, Finger Difference, Match, \n";
    // ...
}

void fileEvaluator::processFrame(int input)
{
    if(inFile.good())
    {

        string FrameNumber;
        string NumberOfFingers;
        string fingerOrientation;


        getline(inFile, FrameNumber, ',');
        cout << "Frame: " << FrameNumber << " " ;

        getline(inFile, NumberOfFingers, ',');
        cout << "Number of Fingers: " << NumberOfFingers << " " ;

        getline(inFile, fingerOrientation, ',');
        cout << "Finger Orientation: " << fingerOrientation << "\n " ;

        getline(inFile, header); //clears new line characters

        //std::string myString = "45";
        int numb;
        istringstream ( NumberOfFingers ) >> numb;;
        bool result = (numb == input);

        if(isLocked == false && input >= 3){
            isLocked = true;
        }else if(isLocked == true && input == -1){
            isLocked = false;
        }



        int diffResult;
        if(numb == -1 && input != -1){
            diffResult = input;
        }else if(numb != -1 && input == -1){
            diffResult = numb;

        }else{
        diffResult = abs(numb - input);
        }

        outFile<<FrameNumber<<", "<<NumberOfFingers<<", "<<fingerOrientation<<", "<< input <<", "<< diffResult <<", "<<result<<",\n";

        totalFrames += 1;

        if(result)successFrames+=1;

        if(isLocked){
            TotalFramesWhenLocked += 1;
            if(result){
                successFramesWhenLocked +=1;
            }
        }

        //if(isLocked && result)successFramesWhenLocked +=1;





    }
}

void fileEvaluator::close(){
    inFile.close();

    double successRate = (successFrames/totalFrames);
    double successRateWhenLocked = (successFramesWhenLocked/TotalFramesWhenLocked);

    outFile<<"Success Rate, "<<successRate<<",\n";
    outFile<<"Success Rate with Target Locked, "<< successRateWhenLocked <<",\n";

    outFile.close();
    if(recordMasterFile){
    outMasterFile<<savedFileName<<","<<successRate<<","<<successRateWhenLocked<<",\n";

    sumSuccessRate  += successRate;
    if(successRateWhenLocked !=successRateWhenLocked){
        //do nothing
    }else{
        sumSuccessRatesWhenLocked += successRateWhenLocked;
        numberOfTrialsThatLocked += 1.0;

    }
    numberOfTrials += 1.0;


    }


}

void fileEvaluator::endExperiment(){
    recordMasterFile = false;
    outMasterFile<<"Average Success Rate: ,"<<(sumSuccessRate/numberOfTrials)<<",\n";
    if(numberOfTrialsThatLocked != 0.0){
    outMasterFile<<"Average Success Rate with Target Locked: ,"<<(sumSuccessRatesWhenLocked/numberOfTrialsThatLocked)<<",\n";
    }

    outMasterFile.close();
}


#endif // FILEEVALUATOR_H
