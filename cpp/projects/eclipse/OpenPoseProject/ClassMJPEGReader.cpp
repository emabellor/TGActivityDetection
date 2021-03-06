/*
 * ClassVideoSaver.cpp
 *
 *  Created on: Feb 20, 2018
 *      Author: mauricio
 */

#include "ClassMJPEGReader.h"

using namespace std;

ClassMJPEGReader::ClassMJPEGReader() {
	oFile = NULL;
}

ClassMJPEGReader::~ClassMJPEGReader() {
	if (oFile != NULL) {
		delete(oFile);
	}
}

void ClassMJPEGReader::OpenFileSave(string fileName) {
	auto substr = fileName.substr(fileName.find_last_of(".") + 1);
	cout << substr << endl;

	if (substr != "mjpegx") {
		cerr << "FileName must have extension mjpegx" << endl;
		exit(1);
	}

	oFile = new ofstream(fileName, ios::binary);
	cout << "File initialized" << endl;
}

void ClassMJPEGReader::CloseFileSave() {
	oFile->close();
}

void ClassMJPEGReader::WriteFileSave(int data) {
	char buffer[4];
	IntToBytearray(data, buffer);
	oFile->write(buffer, 4);
}

void ClassMJPEGReader::WriteFileSave(unsigned char* data, int len) {
	oFile->write(reinterpret_cast<char*>(data), len);
}

void ClassMJPEGReader::WriteFileSave(char* data, int len) {
	oFile->write(data, len);
}

void ClassMJPEGReader::WriteFrame(FrameInfo frame, ClassPoseResults results) {
	//MJPEGX Structure

	auto totalResults = results.GetAllPoints();

	// Write file size
	WriteFileSave(frame.len);

	// Write date time
	WriteFileSave(frame.ticks, 8);

	// Write number of total results
	WriteFileSave(totalResults.size());

	for (uint i = 0; i < totalResults.size(); i++) {
		// Write person
		WriteFileSave(totalResults.at(i).person);

		// Write bodyPart
		WriteFileSave(totalResults.at(i).bodyPart);

		// WriteX
		WriteFileSave(totalResults.at(i).pos.x);

		// WriteY
		WriteFileSave(totalResults.at(i).pos.y);

		// Write score
		WriteFileSave(totalResults.at(i).score * 100);
	}

	// WriteFrame
	WriteFileSave(frame.image, frame.len);

	// Done!
}


// Static Functions
vector<FrameInfo> ClassMJPEGReader::ProcessVideo(string filePath) {
	cout << "Initializing process video " << filePath << endl;

	// Initialize elements
	vector<FrameInfo> listFrames;

	// Loading video
	ifstream file (filePath, ios::in|ios::binary);

	if (file.is_open() == false) {
		cerr << "Error opening file" << endl;
		exit(1);
	} else {
		while(true) {
			// Reading file size
			char fileSizeBin [4];
			file.read(fileSizeBin, 4);
			int bytes = file.gcount();


			if (bytes != 4) {
				if (bytes != 0) {
					cerr << "Error reading file size bytes: " << bytes << endl;
					exit(1);
				} else {
					// End of file - break!
					break;
				}
			}

			int fileSize = ByteArrayToInt(fileSizeBin);

			// Add to vector
			FrameInfo frame;
			frame.isEmpty = false;
			frame.len = fileSize;

			// Reading dateTime
			char dateTimeBin[8];
			file.read(dateTimeBin, 8);
			bytes = file.gcount();

			if (bytes != 8) {
				cerr << "Error reading Date Time bytes: " << bytes << endl;
				exit(1);
			}

			// Copy buffer
			frame.CopyBufferTicks(dateTimeBin);

			// Reading image
			char imageBin[fileSize];
			file.read(imageBin, fileSize);
			bytes = file.gcount();

			if (bytes != fileSize) {
				cerr << "Error reading Image bytes: " << bytes << endl;
				exit(1);
			}

			frame.CopyBuffer(imageBin, fileSize);

			listFrames.push_back(frame);
		}
	}

	return listFrames;
}

int ClassMJPEGReader::ByteArrayToInt(char * b) {
	int i = ((unsigned char)b[3] << 24) | ((unsigned char)b[2] << 16) | ((unsigned char)b[1] << 8) | ((unsigned char)b[0]);
	return i;
}

void ClassMJPEGReader::IntToBytearray(int number, char* arrayRef) {
	// Array reference must be previously initialized
	arrayRef[0] = number & 0xFF;
	arrayRef[1] = (number >> 8) & 0xFF;
	arrayRef[2] = (number >> 16) & 0xFF;
	arrayRef[3] = (number >> 24) & 0xFF;
}

