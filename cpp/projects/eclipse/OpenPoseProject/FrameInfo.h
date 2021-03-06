/*
 * FrameInfo.h
 *
 *  Created on: Feb 18, 2018
 *      Author: mauricio
 */

#ifndef FRAMEINFO_H_
#define FRAMEINFO_H_

#include <vector>
#include <opencv2/core/core.hpp> // Types uchar
#include <iostream>

class FrameInfo {
public:
	static const int MAX_SIZE = 100 * 1024; // Fixed Length - Avoid realocating memory continuously

	unsigned char image [MAX_SIZE];
	unsigned char ticks[8] = {0};
	int len;
	bool isEmpty;

	FrameInfo();
	void LoadFromVector(std::vector<uchar> buff);
	void CopyBuffer(unsigned char* buffer, int len);
	void CopyBuffer(char* buffer, int len);
	void CopyBufferTicks(char* bfTicks);
};


#endif /* FRAMEINFO_H_ */
