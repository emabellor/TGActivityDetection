/*
 * ClassOpenPose.cpp
 *
 *  Created on: Feb 3, 2018
 *      Author: mauricio
 */

#include "ClassOpenPose.h"


ClassOpenPose::ClassOpenPose() {
	scaleAndSizeExtractor = NULL;
	poseExtractorCaffe = NULL;
	frameDisplayer = NULL;
	cvMatToOpInput = NULL;
	cvMatToOpOutput = NULL;
	poseRenderer = NULL;
	opOutputToCvMat = NULL;
	scaleInputToOutput = 0;
}

ClassOpenPose::~ClassOpenPose() {
	op::log("Freeing resources");

	if (jniFlag == true) {
		op::log("Ignoring by JNI flag!");
	} else {
		op::log("Freeing: scaleAndSizeExtractor");
		if (scaleAndSizeExtractor != NULL) {
			delete (scaleAndSizeExtractor);
		}

		op::log("Freeing: poseExtractorCaffe");
		if (poseExtractorCaffe != NULL) {
			delete(poseExtractorCaffe);
		}

		op::log("Freeing: frameDisplayer");
		if (frameDisplayer != NULL) {
			delete(frameDisplayer);
		}

		op::log("Freeing: cvMatToOpInput");
		if (cvMatToOpInput != NULL) {
			delete (cvMatToOpInput);
		}

		op::log("Freeing: cvMatToOpOutput");
		if (cvMatToOpOutput != NULL) {
			delete (cvMatToOpOutput);
		}

		op::log("Freeing: opOutputToCvMat");
		if (opOutputToCvMat != NULL) {
			delete (opOutputToCvMat);
		}
	}
}


void ClassOpenPose::InitOpenPose() {
	if (initialized == true) {
		std::cerr << "OpenPose already initialized!" << std::endl;
		exit(1);
	} else {
		op::log("Initializing open pose");
		op::log("Waiting for GPU");

		// Step 1 - Set logging level
		// - 0 will output all the logging messages
		// - 255 will output nothing
		op::check(0 <= logging_level && logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
		op::ConfigureLog::setPriorityThreshold((op::Priority)logging_level);
		op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

		// Step 2 - Read Google flags (user defined configuration)
		// outputSize
		const auto outputSize = op::flagsToPoint(output_resolution, "-1x-1");
		// netInputSize
		const auto netInputSize = op::flagsToPoint(net_resolution, "-1x368");
		// poseModel
		const auto poseModel = op::flagsToPoseModel(model_pose);

		// Check no contradictory flags enabled
		if (alpha_pose < 0. || alpha_pose > 1.)
			op::error("Alpha value for blending must be in the range [0,1].",
					  __LINE__, __FUNCTION__, __FILE__);

		if (scale_gap <= 0. && scale_number > 1)
			op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
					  __LINE__, __FUNCTION__, __FILE__);

		// Logging
		op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

		// Step 3 - Initialize all required classes
		scaleAndSizeExtractor = new op::ScaleAndSizeExtractor(netInputSize, outputSize, scale_number, scale_gap);
		cvMatToOpInput = new op::CvMatToOpInput{poseModel};
		cvMatToOpOutput = new op::CvMatToOpOutput{};
		poseExtractorCaffe = new op::PoseExtractorCaffe{poseModel, model_folder, num_gpu_start};

		poseRenderer = 	new op::PoseCpuRenderer{poseModel, (float)render_threshold, !disable_blending, (float)alpha_pose};
		frameDisplayer = new op::FrameDisplayer{"OpenPose Tutorial - Example 1", outputSize};

		// Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
		poseExtractorCaffe->initializationOnThread();
		poseRenderer->initializationOnThread();

		// Step 5 - Set variable and return
		initialized = true;
	}
}


ClassPoseResults ClassOpenPose::ExtractKeyPoints(cv::Mat image) {
	ClassPoseResults poseResults;
	std::cout << "Image key points: " << image.empty() << std::endl;

	if (initialized == false) {
		std::cerr << "ClassOpenPose::InitOpenPose not called" << std::endl;
		exit(1);
	} else {
		GetKeyPointsFromImage(image);

		//Extracting poses
		auto sizeElems = poseKeypoints.getSize();

		if (poseKeypoints.empty() == false) {
			for (auto person = 0; person < poseKeypoints.getSize(0); person++) {

				for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++) {
					StructPoints point;
					point.person = person;
					point.bodyPart = bodyPart;

					std::string valueToPrint;

					for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++) {
						auto value = poseKeypoints[{person, bodyPart, xyscore}];

						switch(xyscore) {
							case 0: {
								point.pos.x = value;
								break;
							}
							case 1: {
								point.pos.y = value;
								break;
							}
							case 2: {
								point.score = value;
								break;
							}
							default: {
								op::log("Index not recognized");
								break;
							}
						}
					}

					poseResults.AddResult(point);
				}
			}
		}
	}

	return poseResults;
}

void ClassOpenPose::GetKeyPointsFromImage(cv::Mat inputImage) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	std::cout << "Init GetKeyPointsFromImage" << std::endl;
	//Step 1 already tested - Image loaded
	const op::Point<int> imageSize{inputImage.cols, inputImage.rows};

	// Step 2 - Get desired scale sizes
	std::vector<double> scaleInputToNetInputs;
	std::vector<op::Point<int>> netInputSizes;
	op::Point<int> outputResolution;

	std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
		= scaleAndSizeExtractor->extract(imageSize);

	// Step 3 - Format input image to OpenPose input and output formats
	const auto netInputArray = cvMatToOpInput->createArray(inputImage, scaleInputToNetInputs, netInputSizes);
	outputArray = cvMatToOpOutput->createArray(inputImage, scaleInputToOutput, outputResolution);

	// Step 4 - Estimate poseKeypoints
	poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);

	poseKeypoints = poseExtractorCaffe->getPoseKeypoints();
	poseScores = poseExtractorCaffe->getPoseScores();
	std::cout << "End GetKeyPointsFromImage" << std::endl;

	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
}


void ClassOpenPose::ExtractAndShow(cv::Mat inputImage) {
	std::cout << "Image key points: " << inputImage.empty() << std::endl;

	op::log("Getting keypoints from image");
	GetKeyPointsFromImage(inputImage);

	// Step 5 - Render poseKey points
	op::log("Rendering pose");
	poseRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput);

	// Step 6 - OpenPose output format to cv::Mat
	op::log("Output to mat");
	auto outputImage = opOutputToCvMat->formatToCvMat(outputArray);

	// ------------------------- SHOWING RESULT AND CLOSING -------------------------
	// Step 1 - Show results
	frameDisplayer->displayFrame(outputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
	// Step 2 - Logging information message
	op::log("Example 1 successfully finished.", op::Priority::High);
}

void ClassOpenPose::DrawPose(std::vector<StructPoints> list, bool translate, double factor) {
	op::log("Drawing pose", op::Priority::High);

	// One pose only
	// person
	// bodypart
	// x y score
	poseKeypoints.reset({1, 18, 3});
	std::cout << "Factor: " << factor << std::endl;

	for (uint i = 0; i < list.size(); i++) {
		auto elem = list.at(i);

		double x, y, score;
		score = elem.score;

		std::cout << "x: " << elem.pos.x << std::endl;
		std::cout << "y: " << elem.pos.y << std::endl;
		std::cout << std::endl;


		if (translate == true) {
			double yOffset = 50;
			x = elem.pos.x * factor + 320;
			y = elem.pos.y * factor + yOffset;
		} else {
			x = elem.pos.x;
			y = elem.pos.y;
		}


		poseKeypoints[{0, (int)i, 0}] = x;
		poseKeypoints[{0, (int)i, 1}] = y;
		poseKeypoints[{0, (int)i, 2}] = score;
	}

	// Create white image: 640 * 480
	cv::Mat image(480, 640, CV_8UC3);
	image.setTo(cv::Scalar(255, 255, 255));

	op::Point<int> outputResolution;
	outputResolution.x = 640;
	outputResolution.y = 480;

	//Create outputArray
	outputArray = cvMatToOpOutput->createArray(image, scaleInputToOutput, outputResolution);

	// Step 5 - Render poseKey points
	poseRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput);

	// Step 6 - OpenPose output format to cv::Mat
	auto outputImage = opOutputToCvMat->formatToCvMat(outputArray);

	// ------------------------- SHOWING RESULT AND CLOSING -------------------------
	// Step 1 - Show results
	frameDisplayer->displayFrame(outputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)

	// Step 2 - Logging information message
	op::log("Image drawed!", op::Priority::High);
}



