#include "handcaptureutilities.h"

cv::Point midPoint(cv::Point p1, cv::Point p2){
    Point mid(0,0);
    mid.x= (p1.x - p2.x)/2 + p2.x;
    mid.y= (p1.y - p2.y)/2 + p2.y;
    return mid;
}


int dist (cv::Point p1, cv::Point p2){
    int x =(p1.x - p2.x);
    int y =(p1.y - p2.y);
    int retVal = (x*x) + (y*y);
    return (int)(sqrt((double)retVal));

}


double angleBetween3Points(Point P1, Point P2, Point P3){
    double P13 = dist(P3, P1);
    double P12 = dist(P2, P1);
    double P23 = dist(P2, P3);

    double result = acos((pow(P13, 2) + pow(P12, 2) - pow(P23,2))/(2*P12*P13));
    return result;
}

Point averageOfPoints(vector<Point> points){
    Point retPoint;
    retPoint.x =0;
    retPoint.y = 0;
    for (int i = 0; i<points.size(); i++){
        retPoint.x += points[i].x;
        retPoint.y += points[i].y;

    }
    retPoint.x = (retPoint.x/points.size());
    retPoint.y = (retPoint.y/points.size());

    return retPoint;

}

void equalizeRGB(Mat &src){
    vector<Mat> dst(3);

    vector<Mat> channels(3);
    split(src, channels);


    /// Convert to grayscale
    //cvtColor( src, src, CV_BGR2GRAY );

    /// Apply Histogram Equalization
    equalizeHist( channels[0], dst[0] );
    equalizeHist( channels[1], dst[1] );
    equalizeHist( channels[2], dst[2] );

    /// Display results


    Mat resultImage;
    merge(dst, resultImage);
    resultImage.copyTo(src);

}

void intensityMap(const cv::Mat &image, cv::Mat &result){
    result.create(image.rows, image.cols, CV_8UC1);

    //if(image.type() ==)



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
            double green = ((double)(current[i+1]));
            double blue = ((double)(current[i]));
            double Iprime;
            if(green > blue){
                Iprime = green;
            }else{
                Iprime = blue;
            }


            //*output++= cv::saturate_cast<uchar>((int)(2.0*255.0*(((double)(current[i]))/((double)(current[i] + current[i+1] + current[i+2])))));
            //*output++= cv::saturate_cast<uchar>((int)(2.0*255.0*(((double)(current[i+1]))/((double)(current[i] + current[i+1] + current[i+2])))));
            *output++= cv::saturate_cast<uchar>((int)((((double)(current[i+2]))*0.298936021293775390 + ((double)(current[i+1]))*0.587043074451121360 + ((double)(current[i]))*0.140209042551032500 - Iprime)));
            //sharpen_pixel = 5*current - left -right -up -down;
        }
    }

    threshold(result, result, 4, 255, CV_THRESH_BINARY);

    //set the unprocessed pixels to 0;
    //result.row(0).setTo(cv::Scalar(0));
    //result.row(result.rows - 1).setTo(cv::Scalar(0));
    //result.col(0).setTo(cv::Scalar(0));
    //result.col(result.cols-1).setTo(cv::Scalar(0));
}

