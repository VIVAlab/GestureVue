#if !defined COLHISTOGRAM
#define COLHISTOGRAM

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

class ColourHistogram {

  private:

    int histSize[3];
    float hranges[2];
    const float* ranges[3];
    int channels[3];

  public:

    ColourHistogram() {

        // Prepare arguments for a colour histogram
        histSize[0]= histSize[1]= histSize[2]= 256;
        hranges[0]= 0.0;    // BRG range
        hranges[1]= 255.0;
        ranges[0]= hranges; // all channels have the same range
        ranges[1]= hranges;
        ranges[2]= hranges;
        channels[0]= 0;		// the three channels
        channels[1]= 1;
        channels[2]= 2;
    }

    // Computes the histogram.
    cv::MatND getHistogram(const cv::Mat &image) {

        cv::MatND hist;

        // BGR colour histogram
        hranges[0]= 0.0;    // BRG range
        hranges[1]= 255.0;
        channels[0]= 0;		// the three channels
        channels[1]= 1;
        channels[2]= 2;

        // Compute histogram
        cv::calcHist(&image,
            1,			// histogram of 1 image only
            channels,	// the channel used
            cv::Mat(),	// no mask is used
            hist,		// the resulting histogram
            3,			// it is a 3D histogram
            histSize,	// number of bins
            ranges		// pixel value range
        );

        return hist;
    }

    // Computes the histogram.
    cv::SparseMat getSparseHistogram(const cv::Mat &image) {

        cv::SparseMat hist(3,histSize,CV_32F);

        // BGR colour histogram
        hranges[0]= 0.0;    // BRG range
        hranges[1]= 255.0;
        channels[0]= 0;		// the three channels
        channels[1]= 1;
        channels[2]= 2;

        // Compute histogram
        cv::calcHist(&image,
            1,			// histogram of 1 image only
            channels,	// the channel used
            cv::Mat(),	// no mask is used
            hist,		// the resulting histogram
            3,			// it is a 3D histogram
            histSize,	// number of bins
            ranges		// pixel value range
        );

        return hist;
    }

    // Computes the 1D Hue histogram with a mask.
    // BGR source image is converted to HSV
    // Pixels with low saturation are ignored
    cv::MatND getHueHistogram(const cv::Mat &image,
                             int minSaturation=0) {

        cv::MatND hist;

        // Convert to HSV colour space
        cv::Mat hsv;
        cv::cvtColor(image, hsv, CV_BGR2HSV);

        // Mask to be used (or not)
        cv::Mat mask;

        if (minSaturation>0) {

            // Spliting the 3 channels into 3 images
            std::vector<cv::Mat> v;
            cv::split(hsv,v);

            // Mask out the low saturated pixels
            cv::threshold(v[1],mask,minSaturation,255,
                                 cv::THRESH_BINARY);
        }

        // Prepare arguments for a 1D hue histogram
        hranges[0]= 0.0;
        hranges[1]= 180.0;
        channels[0]= 0; // the hue channel

        // Compute histogram
        cv::calcHist(&hsv,
            1,			// histogram of 1 image only
            channels,	// the channel used
            mask,		// binary mask
            hist,		// the resulting histogram
            1,			// it is a 1D histogram
            histSize,	// number of bins
            ranges		// pixel value range
        );

        return hist;
    }

    // Computes the 1D Hue histogram with a mask.
    // BGR source image is converted to HSV
    cv::MatND getHueHistogram(const cv::Mat &image) {

        cv::MatND hist;

        // Convert to Lab colour space
        cv::Mat hue;
        cv::cvtColor(image, hue, CV_BGR2HSV);

        // Prepare arguments for a 1D hue histogram
        hranges[0]= 0.0;
        hranges[1]= 180.0;
        channels[0]= 0; // the hue channel

        // Compute histogram
        cv::calcHist(&hue,
            1,			// histogram of 1 image only
            channels,	// the channel used
            cv::Mat(),	// no mask is used
            hist,		// the resulting histogram
            1,			// it is a 1D histogram
            histSize,	// number of bins
            ranges		// pixel value range
        );

        return hist;
    }

    cv::Mat colorReduce(const cv::Mat &image, int div=64) {

      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

      cv::Mat_<cv::Vec3b>::const_iterator it= image.begin<cv::Vec3b>();
      cv::Mat_<cv::Vec3b>::const_iterator itend= image.end<cv::Vec3b>();

      // Set output image (always 1-channel)
      cv::Mat result(image.rows,image.cols,image.type());
      cv::Mat_<cv::Vec3b>::iterator itr= result.begin<cv::Vec3b>();

      for ( ; it!= itend; ++it, ++itr) {

        (*itr)[0]= ((*it)[0]&mask) + div/2;
        (*itr)[1]= ((*it)[1]&mask) + div/2;
        (*itr)[2]= ((*it)[2]&mask) + div/2;
      }

      return result;
}

};


#endif
