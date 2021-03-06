/*
 * FrameInfo.cpp
 *
 *  Created on: Feb 18, 2018
 *      Author: mauricio
 */

#include "FrameInfo.h"


FrameInfo::FrameInfo() {
	isEmpty = false;
	len = 0;
}


void FrameInfo::LoadFromVector(std::vector<uchar> buff) {
	if (buff.size() > MAX_SIZE) {
		std::cerr << "Image size exceeded max size " << MAX_SIZE << ". Aborting" << std::endl;
		exit(1);
	}

	std::copy(buff.begin(), buff.end(), image);
	len = buff.size();
}

void FrameInfo::CopyBuffer(unsigned char* buffer, int len) {
	if (len > MAX_SIZE) {
		std::cerr << "Image size exceeded max size " << MAX_SIZE << ". Aborting" << std::endl;
		exit(1);
	}

	// Copy buffer using memcpy
	memcpy(image, buffer, len);
}

void FrameInfo::CopyBuffer(char* buffer, int len) {
	if (len > MAX_SIZE) {
		std::cerr << "Image size exceeded max size " << MAX_SIZE << ". Aborting" << std::endl;
		exit(1);
	}

	// Copy buffer using memcpy
	memcpy(image, buffer, len);
}

void FrameInfo::CopyBufferTicks(char* bfTicks) {
	// Copy buffer using memcpy
	int len = 8;
	memcpy(ticks, bfTicks, len);
}

