/*
 * ClassMain.h
 *
 *  Created on: Feb 4, 2018
 *      Author: mauricio
 */

#ifndef CLASSMAIN_H_
#define CLASSMAIN_H_

// Standard Includes
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>

// JSON includes
#include <nlohmann/json.hpp>

// Boost Includes
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string/replace.hpp>

// OpenCV Includes
#include <opencv2/opencv.hpp>

// Custom Includes
#include "ClassOpenPose.h"
#include "ImageProcess.h"
#include "FileHandler.h"
#include "VideoCapInfo.h"
#include "ClassWrapper.h"
#include "ClassMJPEGReader.h"
#include "AutoResetEvent.h"

typedef struct _StuctDu {
	double arrayX [14];
	double arrayY [14];
	double arrayZ [14];

} StructDu;

typedef struct _DescriptorDu {
	int identifier;
	std::vector<std::string> files;
} DescriptorDu;


class ClassMain {
public:
	// Constructors
	ClassMain();
	virtual ~ClassMain();

	// Functions
	void InitProcess(int argc, char** argv);

	// Static Functions
	static cv::Point2f Project3DTo2D(cv::Point3f point, cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec, cv::Mat distorsion);
	static cv::Point3f Project2DTo3D(cv::Point2f point, cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec, cv::Mat distorsion);


private:
	// Constants
	const std::string defaultImage = "/home/mauricio/Pictures/person.jpg";
	const std::string defaultCompareImage1 = "/home/mauricio/Pictures/test4.png";
	const std::string defaultCompareImage2 = "/home/mauricio/Pictures/test5.png";
	const std::string folderCalibration = "/home/mauricio/Oviedo/CameraCalibration";
	const int rectSize = 14; // Should be multiples of 2
	const double scoreThresh = 0.4;

	// Functions
	void InitLogger();
	void ProcessKeyPoints(int argc, char** argv);
	void ShowKeyPoints(int argc, char** argv);
	void PersonReidentification(int argc, char** argv);
	void ExtractKeyPosesDescriptor(int argc, char** argv);
	void ExtractDescriptorsDu(int argc, char** argv);
	void Clustering(int argc, char** argv);
	void TestDrawPose(int argc, char** argv);
	void TestVideoCapInfo(int argc, char** argv);
	void TestVideoWrapper(int argc, char** argv);
	void TestConvertVideoMJPEG(int argc, char** argv);
	void TaskConvertVideoFolder(int argc, char** argv);
	void SolvePNPTesting(int argc, char** argv);
	void CalibrateCamera(int argc, char** argv);
	void CalibrateCameraCv(int argc, char** argv);
	void CalibrateCameraLoad(int argc, char** argv);
	double GetColorPoses(cv::Mat image, StructPoints pose1, StructPoints pose2);
	std::vector<std::string> SplitStr(std::string str, std::string delimiter);
	void SaveMatToCSV(std::string baseFolder, std::string fileName, cv::Mat matrix);
	void TestJSONSerialization(int argc, char** argv);
	void CheckDoubleToString(int argc, char** argv);
	void TestChessBoardCorners(int argc, char ** argv);
	void CalibrateCameraCustom(int argc, char ** argv);
	void ChessboardCalib(int argc, char** argv);
	void FindHomography(int argc, char** argv);
	std::vector<cv::Point3f> Create3DChessboardCorners(cv::Size boardSize, float squareSize);

	// Static functions
	static void ClickEventHandler(int event, int x, int y, int flags, void* userdata);
	// Properties
	static AutoResetEvent clickEvent;
	static cv::Point2f pointClicked;
	static bool mouseClicked;
};

#endif /* CLASSMAIN_H_ */
