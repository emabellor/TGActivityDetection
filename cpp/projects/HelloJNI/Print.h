/*
 * PrintHello.h
 *
 *  Created on: Feb 9, 2018
 *      Author: mauricio
 */

#ifndef PRINT_H_
#define PRINT_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

// SayHello Method
JNIEXPORT void JNICALL Java_main_HelloJNI_SayHello(JNIEnv *env, jobject obj);

// Sum Method
JNIEXPORT jint JNICALL Java_main_HelloJNI_Sum(JNIEnv * env, jobject obj, jint a, jint b);


#ifdef __cplusplus
}
#endif

#endif /* PRINTHELLO_H_ */
