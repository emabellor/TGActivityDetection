#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

// Standard includes
#include <iostream>
#include <vector>
#include <iomanip>

// Custom includes
#include "FrameInfo.h"

class ImageProcess {
public:
    ImageProcess();

    //Static functions
    static cv::Mat Erode(cv::Mat image, int shape = cv::MORPH_RECT); //Shapes: MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
    static cv::Mat Dilate(cv::Mat image, int shape = cv::MORPH_RECT); //Shapes: MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
    static cv::Mat Threshold(cv::Mat image, double threshValue, int type = cv::THRESH_BINARY); //Types: THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV
    static cv::Mat Grayscale(cv::Mat image);
    static cv::Mat Blur(cv::Mat image);
    static cv::Mat BlurGaussian(cv::Mat image);
    static cv::Mat HistEqualization(cv::Mat image);
    static cv::Mat GetFromCharBuffer(unsigned char* buffer, int len);
    static cv::Mat Resize(cv::Mat image, CvSize size);
    static FrameInfo MatToJPEG(cv::Mat image);
    static void ShowSingleImage(cv::Mat image);
    static void ShowAndWait(cv::Mat mat);
    static double GetMean (cv::Mat mat);
    static double GetAverageColor(cv::Mat mat, cv::Rect region);
    static std::vector<cv::Mat> GetImagesFromVideo(std::string videoPath);
    static std::string Type2Str(int type);
    static cv::Mat LoadFromCSV(std::string csv);
    static std::string GetCSV(cv::Mat matrix);
    static std::string DoubleToStr(double number);

private:
    static double StrToDouble(std::string const str);
    static const int EROSION_SIZE;
    static const int DILATION_SIZE;

};

#endif // IMAGEPROCESS_H
