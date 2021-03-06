/*
 * ClassWrapper.h
 *
 *  Created on: Feb 11, 2018
 *      Author: mauricio
 */

#ifndef CLASSWRAPPER_H_
#define CLASSWRAPPER_H_

#include <jni.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
// Includes for test functions
#include <iostream>
#include "ImageProcess.h"
#include "ClassMain.h"
#include <nlohmann/json.hpp>
#include "ClassCamCalib.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jobject JNICALL Java_activitybesa_ClassWrapper_ProcessImage(JNIEnv *, jobject, jbyteArray);
JNIEXPORT void JNICALL Java_activitybesa_ClassWrapper_InitOpenPose(JNIEnv *, jobject);
JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_LoadVideo (JNIEnv *, jobject, jstring);
JNIEXPORT jbyteArray JNICALL Java_activitybesa_ClassWrapper_GetNextImage(JNIEnv *, jobject, jstring);
JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_Convert2DPoint(JNIEnv *, jobject, jstring);
JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_CalibrateCamera(JNIEnv *, jobject, jstring);
JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_GetHomography(JNIEnv *, jobject, jstring);
JNIEXPORT jstring JNICALL Java_activitybesa_ClassWrapper_ProjectHomography(JNIEnv *, jobject, jstring);

#ifdef __cplusplus
// Test functions
std::string ClassWrapper_LoadVideo(std::string path);
FrameInfo ClassWrapper_GetNextImage(std::string guid);
FrameInfo ClassWrapper_GetNextImage22(std::string guid);
#endif


#ifdef __cplusplus
}
#endif

#endif /* CLASSWRAPPER_H_ */
