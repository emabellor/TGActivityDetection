/*
 * VideoCapInfo.cpp
 *
 *  Created on: Feb 18, 2018
 *      Author: mauricio
 */

#include "VideoCapInfo.h"

VideoCapInfo::VideoCapInfo(std::string fileName) {
	cap = new cv::VideoCapture(fileName);
	uuid_generate(guid);
}


VideoCapInfo::~VideoCapInfo() {
	if (cap != NULL) {
		delete (cap);
	}
}

std::string VideoCapInfo::GetGUID() {
	char uuidBuffer[37];
	uuid_unparse(guid, uuidBuffer);
	std::string uuidGuid(uuidBuffer);
	return uuidGuid;
}

bool VideoCapInfo::IsOpened() {
	bool isOpened = cap->isOpened();
	std::cout << "Is Opened: " << isOpened << std::endl;
	return isOpened;
}

FrameInfo VideoCapInfo::GetNextImage() {
	cv::Mat frame;
	cap->read(frame);

	if (frame.empty() == true) {
		FrameInfo response;
		response.isEmpty = true;
		return response;
	} else {
		FrameInfo response = ImageProcess::MatToJPEG(frame);
		return response;
	}
}
