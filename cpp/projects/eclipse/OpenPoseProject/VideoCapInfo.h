/*
 * VideoCapInfo.h
 *
 *  Created on: Feb 18, 2018
 *      Author: mauricio
 */

#ifndef VIDEOCAPINFO_H_
#define VIDEOCAPINFO_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <uuid/uuid.h>
#include "ImageProcess.h"


class VideoCapInfo {
public:
	VideoCapInfo(std::string videoPath);
	virtual ~VideoCapInfo();

	std::string GetGUID();
	FrameInfo GetNextImage();
	bool IsOpened();
	std::string GetUUID();

private:
	uuid_t guid;
	cv::VideoCapture* cap;
};

#endif /* VIDEOCAPINFO_H_ */
