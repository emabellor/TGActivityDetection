/*
 * ClassWrapper.cpp
 *
 *  Created on: Feb 11, 2018
 *      Author: mauricio
 */

#include "ClassWrapper.h"

// Must have local includes for compatibility
#include "ClassOpenPose.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "ImageProcess.h"
#include "VideoCapInfo.h"
#include <iostream>
#include <mutex>

// Struct definitions
typedef struct _JNI_RESULT {
    jclass cls;
    jmethodID constructorID;
    jfieldID personID;
    jfieldID bodyPartID;
    jfieldID xID;
    jfieldID yID;
    jfieldID scoreID;
} JNI_RESULT;

// Using definitions
using json = nlohmann::json;

// Globals
ClassOpenPose openPose;
std::vector<VideoCapInfo*> listVideos;
std::mutex mutexList;

// Custom Functions
JNI_RESULT LoadJniResultEnv(JNIEnv * env) {
	// Always have to be loaded
	// Class definition changes over time
	JNI_RESULT jniResEnv;

    jniResEnv.cls = env->FindClass("activitybesa/ClassResults");

    if(jniResEnv.cls == NULL){
    	std::cerr << "Can't find class name" << std::endl;
    	exit(1);
    } else {
		jniResEnv.constructorID = env->GetMethodID(jniResEnv.cls, "<init>", "()V");

		if(jniResEnv.constructorID == NULL){
			std::cerr << "Can't find constructor" << std::endl;
			exit(1);
		} else {
			jniResEnv.personID = env->GetFieldID(jniResEnv.cls, "person", "I");
			jniResEnv.bodyPartID = env->GetFieldID(jniResEnv.cls, "bodyPart", "I");
			jniResEnv.xID = env->GetFieldID(jniResEnv.cls, "x", "D");
			jniResEnv.yID = env->GetFieldID(jniResEnv.cls, "y", "D");
			jniResEnv.scoreID = env->GetFieldID(jniResEnv.cls, "score", "D");
		}
    }

    return jniResEnv;
}


// Defined Functions
JNIEXPORT void JNICALL Java_activitybesa_ClassWrapper_InitOpenPose(JNIEnv * env, jobject obj) {
	openPose.InitOpenPose();
	openPose.jniFlag = true;
	std::cout << "OpenPose Initialized and jni set" << std::endl;
}


JNIEXPORT jobject JNICALL Java_activitybesa_ClassWrapper_ProcessImage(JNIEnv * env, jobject obj, jbyteArray imageBin) {
	// Conversion to unsigned char *
    int len = env->GetArrayLength(imageBin);
    unsigned char* buf = new unsigned char[len];
    env->GetByteArrayRegion (imageBin, 0, len, reinterpret_cast<jbyte*>(buf));

    // Image Processing
    cv::Mat image = ImageProcess::GetFromCharBuffer(buf, len);

    // Process
    auto results = openPose.ExtractKeyPoints(image);
    auto resultPeople = results.GetAllPoints();

    // Iterating
    JNI_RESULT jniResEnv = LoadJniResultEnv(env);
    std::cout << "Creating jni object size " <<  resultPeople.size() << std::endl;
    jobjectArray jPosRecArray = env->NewObjectArray(resultPeople.size(), jniResEnv.cls, NULL);
    std::cout << "Object created" << std::endl;

    for (uint i = 0; i < resultPeople.size(); i++) {
    	auto elem = resultPeople.at(i);

    	// Set fields
    	jobject item = env->NewObject(jniResEnv.cls, jniResEnv.constructorID);
    	env->SetIntField(item, jniResEnv.personID, elem.person);
    	env->SetIntField(item, jniResEnv.bodyPartID, elem.bodyPart);
    	env->SetDoubleField(item, jniResEnv.xID, elem.pos.x);
    	env->SetDoubleField(item, jniResEnv.yID, elem.pos.y);
    	env->SetDoubleField(item, jniResEnv.scoreID, elem.score);

    	// Adding to array
    	env->SetObjectArrayElement(jPosRecArray, i, item);
    }

    std::cout << "Array created!" << std::endl;
    return jPosRecArray;
}

std::string ClassWrapper_LoadVideo(std::string path) {
	std::string returnValue;

	VideoCapInfo * capInfo = new VideoCapInfo(path);
	if (capInfo->IsOpened() == false) {
		std::cerr << "Can't open " << path << std::endl;
		delete(capInfo); // ReleaseMemory
		returnValue = "";
	} else {
		// Add cap info to list - Must have exclusion to avoid change of elements
		mutexList.lock();
		listVideos.push_back(capInfo);
		mutexList.unlock();

		returnValue = capInfo->GetGUID();
	}

	return returnValue;
}

JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_LoadVideo (JNIEnv * env, jobject obj, jstring path) {
	// Loading elements
	const char* nativeString = env->GetStringUTFChars(path, 0);
	std::string pathStr(nativeString);
	std::cout << "String: " << pathStr << std::endl;

	// Return value
	std::string returnValue = ClassWrapper_LoadVideo(pathStr);

	// Return JNI call
	return env->NewStringUTF(returnValue.c_str());
}



FrameInfo ClassWrapper_GetNextImage(std::string guidStr) {
	// Check guid in list - Must have exclusion
	FrameInfo frame;
	frame.isEmpty = true;

	// Init mutual exclusion
	mutexList.lock();
	bool found = false;
	uint index = 0;

	for (uint i = 0; i < listVideos.size(); i++) {
		VideoCapInfo* elem = listVideos.at(i);

		if (guidStr.compare(elem->GetGUID()) == 0) {
			index = i;
			found = true;
			break;
		}
	}

	if (found == false) {
		std::cout << "Could not find guid " << guidStr << std::endl;
	} else {
		auto elem = listVideos.at(index);
		frame = elem->GetNextImage();

		if (frame.isEmpty == true) {
			std::cout << "Frame is empty. Removing from list" << std::endl;
			listVideos.erase(listVideos.begin() + index);
			delete(elem);
		}
	}

	// Remove mutual exclusion
	mutexList.unlock();

	// Done!
	return frame;
}

JNIEXPORT jbyteArray JNICALL Java_activitybesa_ClassWrapper_GetNextImage(JNIEnv * env, jobject obj, jstring guid) {
	const char* nativeString = env->GetStringUTFChars(guid, 0);
	std::string guidStr(nativeString);
	FrameInfo frame = ClassWrapper_GetNextImage(guidStr);

	// Return byte array
	if (frame.isEmpty == true) {
	    jbyteArray jarray = env->NewByteArray(0);
	    return jarray;
	} else {
		jbyteArray jarray = env->NewByteArray(frame.len);
	    env->SetByteArrayRegion(jarray, 0, frame.len, reinterpret_cast<signed char*>(frame.image));
	    return jarray;
	}
}

JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_GetHomography(JNIEnv * env, jobject obj, jstring jsonJava) {
	const char* jsonPtr = env->GetStringUTFChars(jsonJava, 0);
	json jsonObj = json::parse(std::string(jsonPtr));

	std::vector<cv::Point2f> imagePoints;
	std::vector<cv::Point2f> objectPoints;

	// Getting points 2d - ForEach
	for (auto& element : jsonObj["imagePoints"]) {
		imagePoints.push_back(cv::Point2f(element["x"], element["y"]));
	}

	// Getting points 3d - ForEach
	for (auto& element : jsonObj["objectPoints"]) {
		objectPoints.push_back(cv::Point2f(element["x"], element["y"]));
	}

	std::cout << "ImagePoints: " << imagePoints << std::endl;
	std::cout << "ObjectPoints: "  << objectPoints << std::endl;

	// Calibrate camera
	auto homographyMat = ClassCamCalib::GetHomography(imagePoints, objectPoints);

	// Convert to string
	auto resultStr = ImageProcess::GetCSV(homographyMat);

	// Print for debugging purposes
	std::cout << "Obtained results: " << resultStr << std::endl;

	// Return JNI call - string
	return env->NewStringUTF(resultStr.c_str());
}

JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_ProjectHomography(JNIEnv * env, jobject obj, jstring jsonJava) {
	const char* jsonPtr = env->GetStringUTFChars(jsonJava, 0);
	json jsonObj = json::parse(std::string(jsonPtr));

	double pointX = jsonObj["point"]["x"];
	double pointY = jsonObj["point"]["y"];
	std::string homographyMatStr = jsonObj["homographyMat"];

	cv::Mat homographyMat = ImageProcess::LoadFromCSV(homographyMatStr);
	auto projectedPoint = ClassCamCalib::ProjectHomography(cv::Point(pointX, pointY), homographyMat);

	json jsonResponse;
	jsonResponse["x"] = projectedPoint.x;
	jsonResponse["y"] = projectedPoint.y;

	auto resultStr = jsonResponse.dump();

	// Return JNI call - string
	return env->NewStringUTF(resultStr.c_str());
}

JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_Convert2DPoint(JNIEnv * env, jobject obj, jstring jsonJava) {
	const char* jsonPtr = env->GetStringUTFChars(jsonJava, 0);
	json jsonObj = json::parse(std::string(jsonPtr));

	int pointX = jsonObj["point"]["x"];
	int pointY = jsonObj["point"]["y"];
	std::string calibStr = jsonObj["calibResuts"];

	cv::Point2f pointCv(pointX, pointY);
	std::cout << "PointCV: " << pointCv << std::endl;

	auto results = ClassCamCalib::StringToCalibResults(calibStr);

	auto result = ClassCamCalib::Project2DTo3D(pointCv, results.cameraMatrix, results.rvec,
			results.tvec, results.distortion);

	// How to convert elements in string
	json jsonResponse;
	jsonResponse["x"] = result.x;
	jsonResponse["y"] = result.y;
	jsonResponse["z"] = result.z;

	auto resultStr = jsonResponse.dump();

	// Return JNI call - string
	return env->NewStringUTF(resultStr.c_str());
}


JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_CalibrateCamera(JNIEnv * env, jobject obj, jstring jsonJava) {
	std::cout << "Initializing calibrate camera" << std::endl;

	const char* jsonPtr = env->GetStringUTFChars(jsonJava, 0);
	json jsonObj = json::parse(std::string(jsonPtr));

	std::vector<cv::Point2f> listPoints2D;
	std::vector<cv::Point3f> listPoints3D;

	// Getting points 2d - ForEach
	for (auto& element : jsonObj["listPoints2D"]) {
		listPoints2D.push_back(cv::Point2d(element["x"], element["y"]));
	}

	// Getting points 3d - ForEach
	for (auto& element : jsonObj["listPoints3D"]) {
		listPoints3D.push_back(cv::Point3d(element["x"], element["y"], element["z"]));
	}

	std::cout << "ImagePoints: " << listPoints2D << std::endl;
	std::cout << "WorldPoints: "  << listPoints3D << std::endl;

	// Calibrate camera
	auto results = ClassCamCalib::CalibrateCamera(listPoints2D, listPoints3D);

	// Convert to string
	auto resultStr = ClassCamCalib::CalibResultsToString(results);

	// Print for debugging purposes
	std::cout << "Obtained results: " << resultStr << std::endl;

	// Return JNI call - string
	return env->NewStringUTF(resultStr.c_str());
}


