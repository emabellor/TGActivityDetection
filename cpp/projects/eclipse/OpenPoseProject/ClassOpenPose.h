/*
 * ClassOpenPose.h
 *
 *  Created on: Feb 3, 2018
 *      Author: mauricio
 */

#ifndef CLASSOPENPOSE_H_
#define CLASSOPENPOSE_H_

#include <string>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <iostream>
#include <vector>
#include "ClassPoseResults.h"

// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

// OpenCV dependencies
#include <opencv2/highgui/highgui.hpp>

// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

// standard dependencies
#include <iostream>


class ClassOpenPose {
public:
	ClassOpenPose();
	virtual ~ClassOpenPose();

	// Variables
	bool jniFlag = false;

	// Functions
	void InitOpenPose();
	void ExtractAndShow(cv::Mat image);
	void DrawPose(std::vector<StructPoints> pose, bool translate = false, double factor = 1);
	ClassPoseResults ExtractKeyPoints(cv::Mat image);


private:
	// The logging level. Integer in the range [0, 255]. 0 will output any log() message, while
	// 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for
	// low priority messages and 4 for important ones.
	const int logging_level = 3;
	// Process the desired image
	const std::string image_path = "/home/mauricio/Programs/openpose/openpose/examples/media/COCO_val2014_000000000192.jpg";
	// Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster),
	// MPI_4_layers (15 keypoints, even faster but less accurate);
	const std::string model_pose = "COCO";
	// Folder path (absolute or relative) where the models (pose, face, ...) are located.")
	const std::string model_folder = "/home/mauricio/Programs/openpose/openpose/models/";
	// "Multiples of 16. If it is increased, the accuracy potentially increases. If it is
    // decreased, the speed increases. For maximum speed-accuracy balance, it should keep the
    // closest aspect ratio possible to the images or videos to be processed. Using `-1` in
    // any of the dimensions, OP will choose the optimal aspect ratio depending on the user's
    // input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,
    // e.g. full HD (1980x1080) and HD (1280x720) resolutions.")
	const std::string net_resolution = "-1x240";
	// The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
	// input image resolution."
	const std::string output_resolution = "-1x-1";
	// GPU device start number."
	const int num_gpu_start = 0;
	// Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
	// If you want to change the initial scale, you actually want to multiply the"
	// net_resolution` by your desired initial scale."
	const double scale_gap = 0.3;
	// "Number of scales to average."
	const int scale_number = 1;
	// If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
	// background, instead of being rendered into the original image. Related: `part_to_show`,"
	// `alpha_pose`, and `alpha_pose`."
	const bool disable_blending = false;
	// Only estimated keypoints whose score confidences are higher than this threshold will be"
	// rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
	// while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
	// more false positives (i.e. wrong detections)."
	const double render_threshold = 0.05;
	// Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	// hide it. Only valid for GPU rendering."
	const double alpha_pose = 0.6;

	// Variables
	op::ScaleAndSizeExtractor* scaleAndSizeExtractor;
	op::PoseExtractorCaffe* poseExtractorCaffe;
	op::FrameDisplayer* frameDisplayer;
	op::CvMatToOpInput* cvMatToOpInput;
	op::CvMatToOpOutput* cvMatToOpOutput;
	op::PoseCpuRenderer* poseRenderer;
	op::OpOutputToCvMat* opOutputToCvMat;
	op::Array<float> outputArray;
	op::Array<float> poseKeypoints;
	op::Array<float> poseScores;
	double scaleInputToOutput;
	bool initialized = false;

	// Functions
	void GetKeyPointsFromImage(cv::Mat inputImage);
};


#endif /* CLASSOPENPOSE_H_ */
