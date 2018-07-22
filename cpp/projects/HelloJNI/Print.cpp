#include "Print.h"
#include <iostream>

JNIEXPORT void JNICALL Java_main_HelloJNI_SayHello(JNIEnv *env, jobject obj) {
	std::cout << "Hello 22" << std::endl;
}

JNIEXPORT jint JNICALL Java_main_HelloJNI_Sum(JNIEnv * env, jobject obj, jint a, jint b) {
	int result = a + b;
	return result;
}
