#include "ImageProcess.h"

using namespace std;

// Variable redefinition
const int ImageProcess::EROSION_SIZE = 1;
const int ImageProcess::DILATION_SIZE = 1;

ImageProcess::ImageProcess()  {
	// Empty constructor
}

cv::Mat ImageProcess::Erode(cv::Mat image, int shape) {
    cv::Mat erosion_dst;
    cv::Mat element = cv::getStructuringElement(shape, cv::Size(2*EROSION_SIZE + 1, 2*EROSION_SIZE+1),
			cv::Point(EROSION_SIZE, EROSION_SIZE));
    erode(image, erosion_dst, element );
    return erosion_dst;
}

cv::Mat ImageProcess::Dilate(cv::Mat image, int shape) {
	cv::Mat dilate_dst;
    cout << "Applying dilating operation" << endl;
    cv::Mat element = cv::getStructuringElement(shape, cv::Size(2*DILATION_SIZE + 1, 2*DILATION_SIZE+1),
    		cv::Point(DILATION_SIZE, DILATION_SIZE));
    cv::dilate(image, dilate_dst, element);
    return dilate_dst;
}

cv::Mat ImageProcess::Threshold(cv::Mat image, double threshValue, int type) {
	cv::Mat result;
	cv::threshold(image, result, threshValue, 255, type);
    return result;
}

cv::Mat ImageProcess::Grayscale(cv::Mat image) {
	cv::Mat result;
	cv::cvtColor(image, result, CV_BGR2GRAY);
    return result;
}


cv::Mat ImageProcess::Blur(cv::Mat image) {
	cv::Mat result;
	cv::blur(image, result, cv::Size(3, 3));
    return result;
}

void ImageProcess::ShowAndWait(cv::Mat image) {
	cv::namedWindow("ImageShow", CV_WINDOW_AUTOSIZE);
	cv::imshow("ImageShow", image);
	cv::waitKey(0);
	cv::destroyWindow("ImageShow");
}

cv::Mat ImageProcess::HistEqualization(cv::Mat image) {
	cv::Mat result;
	cv::equalizeHist(image, result);
    return result;
}

cv::Mat ImageProcess::BlurGaussian(cv::Mat image) {
	cv::Mat result;
	cv::GaussianBlur(image, result, cv::Size(3, 3), 0, 0);
    return result;
}


void ImageProcess::ShowSingleImage(cv::Mat image) {
    cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", image);
    cv::waitKey(0);
    cv::destroyWindow("image");
}

double ImageProcess::GetMean(cv::Mat image) {
    double result;

    if (image.type() != CV_8UC1) {
        cout << "Image is not grayscale" << endl;
        result = -1;
    } else {
        double mean = 0;
        int counter = 0;

        for (int x = 0; x < image.cols; x++) {
            for (int y = 0; y < image.rows; y++) {
            	cv::Scalar intensity = image.at<uchar>(y, x);
                auto value = intensity.val[0];

                mean += value;
                counter++;
            }
        }

        mean = mean / counter;
        result = mean;
    }

    return result;
}

double ImageProcess::GetAverageColor(cv::Mat mat, cv::Rect region) {
	auto imageCrop = mat(region);

	cv::Mat imageHSV;

	cv::cvtColor(imageCrop, imageHSV, CV_BGR2HSV);

	// Hue range: [0, 179]
	double hueCount = 0;
	int pixelCount = 0;
	for (int i = 0; i < imageHSV.rows; i++) {
		for (int j = 0; j < imageHSV.cols; j++) {
			cv::Vec3b pixel = imageHSV.at<cv::Vec3b>(i, j);

			auto hue = pixel.val[0];

			hueCount += (double)hue;
			pixelCount++;
		}
	}

	return hueCount / pixelCount;
}


std::vector<cv::Mat> ImageProcess::GetImagesFromVideo(std::string videoPath) {
	std::vector<cv::Mat> listImages;

	cv::VideoCapture cap(videoPath);
	if (cap.isOpened() == false) {
		std::cerr << "Error opening path " << videoPath;
		exit(1);
	} else {
		for (;;) {
			cv::Mat frame;
			cap >> frame;

			if (frame.empty()) {
				break;
			} else {
				listImages.push_back(frame);
			}
		}
	}

	return listImages;
}

// Taken from: https://stackoverflow.com/questions/14727267/opencv-read-jpeg-image-from-buffer
cv::Mat ImageProcess::GetFromCharBuffer(unsigned char* buffer, int len) {
	cv::Mat rawData(1, len, CV_8UC1, (void*)buffer);
	cv::Mat decodedImage  =  cv::imdecode(rawData, CV_LOAD_IMAGE_COLOR);
	if (decodedImage.data == NULL) {
	    std::cerr << "Error reading raw image data" << std::endl;
	    exit(1);
	}

	return decodedImage;
}

// Resizing
cv::Mat ImageProcess::Resize(cv::Mat image, CvSize size) {
	cv::Mat outputImg;
	cv::resize(image, outputImg, size, 0, 0, CV_INTER_LINEAR);
	return outputImg;
}

// Convert Mat To JPEG buffer
FrameInfo ImageProcess::MatToJPEG(cv::Mat image) {
	FrameInfo response;

	// Buffer from coding
	std::vector<uchar> buff;

	std::vector<int> param(2);
	param[0] = cv::IMWRITE_JPEG_QUALITY;
	param[1] = 80;//default(95) 0-100

	cv::imencode(".jpg", image, buff, param);

	// Mat is vector! - You must be careful in your implementation
	response.LoadFromVector(buff);
	return response;
}

std::string ImageProcess::Type2Str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

		switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}

// Forcing 64F
cv::Mat ImageProcess::LoadFromCSV(string values) {
	int opencv_type = CV_64F;
    cv::Mat m;

    stringstream ss(values);
    string line;
    while (getline(ss, line, ';')) {
        vector<double> dvals;

        stringstream ssline(line);
        string val;
        while (getline(ssline, val, '&')) {
        	double valPrev = StrToDouble(val);
            dvals.push_back(valPrev);
        }

        cv::Mat mline(1, dvals.size(), CV_64F);
        for (uint i = 0; i < dvals.size(); i++) {
        	mline.at<double>(0, i) = dvals.at(i);
        }


        m.push_back(mline);
    }

    int ch = CV_MAT_CN(opencv_type);
    m = m.reshape(ch);
    m.convertTo(m, opencv_type);
    return m;
}

string ImageProcess::GetCSV(cv::Mat matrix) {
	string data = "";
	if (matrix.type() != CV_64F) {
		cerr << "Wrong type - Killing" << endl;
		exit(1);
	} else {
		for (int i = 0; i < matrix.rows; i++) {
			if (i != 0) {
				data += ";";
			}

			for (int j = 0; j < matrix.cols; j++) {
				if (j != 0) {
					data += "&";
				}
				data += DoubleToStr(matrix.at<double>(i, j));
			}
		}
	}

	return data;
}


double ImageProcess::StrToDouble(std::string str) {
	int presicion = 10;
	std::stringstream sstrm;
	sstrm << fixed << setprecision(presicion) << str;

	double d;
	sstrm >> d;

	return d;
}

std::string ImageProcess::DoubleToStr(double number) {
	int presicion = 10;
	std::stringstream ss;
	ss << fixed << setprecision(presicion) << number;
	return ss.str();
}

