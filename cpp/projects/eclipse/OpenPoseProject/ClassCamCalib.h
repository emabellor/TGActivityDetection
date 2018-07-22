/*
 * ClassCamCalib.h
 *
 *  Created on: Apr 9, 2018
 *      Author: mauricio
 */

#ifndef CLASSCAMCALIB_H_
#define CLASSCAMCALIB_H_

// Standard Includes
#include <iostream>
#include <vector>

// OpenCV Includes
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

// Local Includes
#include "ImageProcess.h"

// Custom Includes
#include <nlohmann/json.hpp>

typedef struct _CalibResults {
	cv::Mat cameraMatrix;
	cv::Mat distortion;
	cv::Mat rvec;
	cv::Mat tvec;
} CalibResults;


class ClassCamCalib {
public:
	ClassCamCalib();
	virtual ~ClassCamCalib();

	static cv::Point2f Project3DTo2D(cv::Point3f point, cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec, cv::Mat distorsion);
	static cv::Point3f Project2DTo3D(cv::Point2f point, cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec, cv::Mat distortion);
	static CalibResults CalibrateCamera(std::vector<cv::Point2f> listPoints2D, std::vector<cv::Point3f> listPoints3D);
	static CalibResults CalibrateCamera(std::vector<cv::Point2f> listPoints2D, std::vector<cv::Point3f> listPoints3D, cv::Mat camMatrix, cv::Mat dist);
	static cv::Mat GetHomography(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> objectPoints);
	static cv::Point2f ProjectHomography(cv::Point2f point, cv::Mat homographyMat);
	static CalibResults CalibrateCameraCalib(std::vector<cv::Point2f> listPoints2D, std::vector<cv::Point3f> listPoints3D, cv::Size sizeImage);
	static std::string CalibResultsToString(CalibResults results);
	static CalibResults StringToCalibResults(std::string resultsString);
};

#endif /* CLASSCAMCALIB_H_ */
