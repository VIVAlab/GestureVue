#include "tutorialfunctions.h"

void salt(cv::Mat &image, int n)
{
    for(int k=0; k<n; k++){
        int i = rand()%image.cols;
        int j = rand()%image.rows;

        if(image.channels() == 1) {
            image.at<uchar>(j,i) = 255;
        }else if(image.channels() == 3){
            image.at<cv::Vec3b>(j,i)[0] = 255;        image.at<cv::Vec3b>(j,i)[1] = 255; image.at<cv::Vec3b>(j,i)[2] = 255;
        }
    }
}

void snow(cv::Mat &image, int n)
{
    for(int k=0; k<n; k++){
        int i = rand()%image.cols;
        int j = rand()%image.rows;

        if(image.channels() == 1) {
            image.at<uchar>(j,i) = 255;
        }else if(image.channels() == 3){
            image.at<cv::Vec3b>(j,i)[0] = 255; image.at<cv::Vec3b>(abs(j+1),i)[0] = 255; image.at<cv::Vec3b>(abs(j-1),i)[0] = 255; image.at<cv::Vec3b>(j,abs(i+1))[0] = 255; image.at<cv::Vec3b>(j,abs(i-1))[0] = 255;
            image.at<cv::Vec3b>(abs(j+2),i)[0] = 255; image.at<cv::Vec3b>(abs(j-2),i)[0] = 255; image.at<cv::Vec3b>(j,abs(i+2))[0] = 255; image.at<cv::Vec3b>(j,abs(i-2))[0] = 255;
            image.at<cv::Vec3b>(abs(j+1),abs(i+1))[0] = 255; image.at<cv::Vec3b>(abs(j-1),abs(i+1))[0] = 255;
            image.at<cv::Vec3b>(abs(j+1),abs(i-1))[0] = 255; image.at<cv::Vec3b>(abs(j-1),abs(i-1))[0] = 255;

            image.at<cv::Vec3b>(j,i)[1] = 255; image.at<cv::Vec3b>(abs(j+1),i)[1] = 255; image.at<cv::Vec3b>(abs(j-1),i)[1] = 255; image.at<cv::Vec3b>(j,abs(i+1))[1] = 255; image.at<cv::Vec3b>(j,abs(i-1))[1] = 255;
            image.at<cv::Vec3b>(abs(j+2),i)[1] = 255; image.at<cv::Vec3b>(abs(j-2),i)[1] = 255; image.at<cv::Vec3b>(j,abs(i+2))[1] = 255; image.at<cv::Vec3b>(j,abs(i-2))[1] = 255;
            image.at<cv::Vec3b>(abs(j+1),abs(i+1))[1] = 255; image.at<cv::Vec3b>(abs(j-1),abs(i+1))[1] = 255;
            image.at<cv::Vec3b>(abs(j+1),abs(i-1))[1] = 255; image.at<cv::Vec3b>(abs(j-1),abs(i-1))[1] = 255;

            image.at<cv::Vec3b>(j,i)[2] = 255; image.at<cv::Vec3b>(abs(j+1),i)[2] = 255; image.at<cv::Vec3b>(abs(j-1),i)[2] = 255; image.at<cv::Vec3b>(j,abs(i+1))[2] = 255; image.at<cv::Vec3b>(j,abs(i-1))[2] = 255;
            image.at<cv::Vec3b>(abs(j+2),i)[2] = 255; image.at<cv::Vec3b>(abs(j-2),i)[2] = 255; image.at<cv::Vec3b>(j,abs(i+2))[2] = 255; image.at<cv::Vec3b>(j,abs(i-2))[2] = 255;
            image.at<cv::Vec3b>(abs(j+1),abs(i+1))[2] = 255; image.at<cv::Vec3b>(abs(j-1),abs(i+1))[2] = 255;
            image.at<cv::Vec3b>(abs(j+1),abs(i-1))[2] = 255; image.at<cv::Vec3b>(abs(j-1),abs(i-1))[2] = 255;

        }
    }
}

void colorReduce(cv::Mat &image, int div){
    int nl = image.rows;
    int nc = image.cols * image.channels();

    if(image.isContinuous()){
        nc = nc*nl;
        nl = 1; //to make this a 1D Array. More efficient execution.
    }

    for (int j=0; j<nl; j++){


        uchar* data = image.ptr<uchar>(j);
        for(int i=0; i<nc; i++){
            data[i] = data[i]/div*div + div/2;

        }

    }
}

void colorReduce(const cv::Mat &image, cv::Mat &result, int div){
    int nl = image.rows;
    int nc = image.cols * image.channels();
    result.create(image.rows, image.cols, image.type());

    for (int j=0; j<nl; j++){


        const uchar* data_in = image.ptr<uchar>(j);
        uchar* data_out = result.ptr<uchar>(j);
        for(int i=0; i<nc; i++){
            data_out[i] = data_in[i]/div*div + div/2;

        }

    }

}

void colorReduce2(cv::Mat &image, int div){
    //same as colorReduce1 just with iterators
    cv:: Mat_<cv::Vec3b>::iterator it= image.begin<cv::Vec3b>();
    cv:: Mat_<cv::Vec3b>::iterator itend= image.end<cv::Vec3b>();
    for(; it!= itend; ++it){
        //process each pixel
        (*it)[0] = (*it)[0]/div*div + div/2;
        (*it)[1] = (*it)[1]/div*div + div/2;
        (*it)[2] = (*it)[2]/div*div + div/2;

    }

}


void sharpen(const cv::Mat &image, cv::Mat &result, double centerScale){
    result.create(image.rows, image.cols, image.type());

    //result.create(image.size(), image.type());
    //std::cout<<"Number of rows ="<<image.rows;
    //std::cout<<"Number of columns ="<<image.cols;

    int nc = image.cols * image.channels();
    for(int j = 1; j<image.rows-1; j++){

        //for all rows except first and last
        const uchar* previous = image.ptr<const uchar>(j -1); // previous row
        const uchar* current = image.ptr<const uchar>(j); // current row
        const uchar* next = image.ptr<const uchar>(j+1); // next row
        uchar* output = result.ptr<uchar>(j);
        for(int i = 1*image.channels(); i<nc-(1*image.channels()); i++){
            *output++= cv::saturate_cast<uchar>(((int)centerScale*current[i]) - current[i-image.channels()] -current[i+image.channels()] -previous[i] - next[i]);
            //sharpen_pixel = 5*current - left -right -up -down;
        }
    }

    //set the unprocessed pixels to 0;
    result.row(0).setTo(cv::Scalar(0));
    result.row(result.rows - 1).setTo(cv::Scalar(0));
    result.col(0).setTo(cv::Scalar(0));
    result.col(result.cols-1).setTo(cv::Scalar(0));
}



void equalize(const cv::Mat &image, cv::Mat &result){
    result.create(image.rows, image.cols, image.type());

    //result.create(image.size(), image.type());
    //std::cout<<"Number of rows ="<<image.rows;
    //std::cout<<"Number of columns ="<<image.cols;

    int nc = image.cols * image.channels();
    for(int j = 0; j<image.rows; j++){

        //for all rows except first and last
        //const uchar* c = image.ptr<const uchar>(j); // previous row
        const uchar* current = image.ptr<const uchar>(j); // current row
        //const uchar* next = image.ptr<const uchar>(j+1); // next row
        uchar* output = result.ptr<uchar>(j);
        for(int i = 0*image.channels(); i<nc; i=i+image.channels()){

            *output++= cv::saturate_cast<uchar>((int)(2.0*255.0*(((double)(current[i]))/((double)(current[i] + current[i+1] + current[i+2])))));
            *output++= cv::saturate_cast<uchar>((int)(2.0*255.0*(((double)(current[i+1]))/((double)(current[i] + current[i+1] + current[i+2])))));
            *output++= cv::saturate_cast<uchar>((int)(2.0*255.0*(((double)(current[i+2]))/((double)(current[i] + current[i+1] + current[i+2])))));
            //sharpen_pixel = 5*current - left -right -up -down;
        }
    }

    //set the unprocessed pixels to 0;
    //result.row(0).setTo(cv::Scalar(0));
    //result.row(result.rows - 1).setTo(cv::Scalar(0));
    //result.col(0).setTo(cv::Scalar(0));
    //result.col(result.cols-1).setTo(cv::Scalar(0));
}




void colourSkew(const cv::Mat &image, cv::Mat &result){
    result.create(image.rows, image.cols, image.type());

    //result.create(image.size(), image.type());
    //std::cout<<"Number of rows ="<<image.rows;
    //std::cout<<"Number of columns ="<<image.cols;

    int nc = image.cols * image.channels();
    for(int j = 1; j<image.rows-1; j++){

        //for all rows except first and last
        const uchar* previous = image.ptr<const uchar>(j -1); // previous row
        const uchar* current = image.ptr<const uchar>(j); // current row
        const uchar* next = image.ptr<const uchar>(j+1); // next row
        uchar* output = result.ptr<uchar>(j);
        for(int i = 1; i<nc-image.channels(); i++){
            *output++= cv::saturate_cast<uchar>(5*current[i] - current[i-1] -current[i+1] -previous[i] - next[i]);
            //sharpen_pixel = 5*current - left -right -up -down;
        }
    }

    //set the unprocessed pixels to 0;
    result.row(0).setTo(cv::Scalar(0));
    result.row(result.rows - 1).setTo(cv::Scalar(0));
    result.col(0).setTo(cv::Scalar(0));
    result.col(result.cols-1).setTo(cv::Scalar(0));
}
