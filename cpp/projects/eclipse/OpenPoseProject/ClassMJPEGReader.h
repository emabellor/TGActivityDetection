/*
 * ClassVideoSaver.h
 *
 *  Created on: Feb 20, 2018
 *      Author: mauricio
 */

#ifndef CLASSMJPEGREADER_H_
#define CLASSMJPEGREADER_H_

// Standard includes
#include <string>
#include <iostream>
#include <fstream>

// OpenCV Includes
#include <opencv2/opencv.hpp>

// Custom includes
#include "ImageProcess.h"
#include "FrameInfo.h"
#include "ClassPoseResults.h"

class ClassMJPEGReader {
public:
	ClassMJPEGReader();
	virtual ~ClassMJPEGReader();

	void OpenFileSave(std::string fileName);
	void WriteFileSave(int data);
	void WriteFileSave(unsigned char* data, int len);
	void WriteFileSave(char* data, int len);
	void WriteFrame(FrameInfo frame, ClassPoseResults results);
	void CloseFileSave();

	static std::vector<FrameInfo> ProcessVideo(std::string fileName);

private:
	std::ofstream* oFile;
	static int ByteArrayToInt(char* byteArray);
	static void IntToBytearray(int number, char* arrayRef);
};

#endif /* CLASSMJPEGREADER_H_ */
