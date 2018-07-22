/*
 * ClassCamCalib.cpp
 *
 *  Created on: Apr 9, 2018
 *      Author: mauricio
 */

#include "ClassCamCalib.h"

using json = nlohmann::json;

ClassCamCalib::ClassCamCalib() {
	// TODO Auto-generated constructor stub

}

ClassCamCalib::~ClassCamCalib() {
	// TODO Auto-generated destructor stub
}

cv::Point2f ClassCamCalib::Project3DTo2D(cv::Point3f point, cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec, cv::Mat distortion) {
	std::vector<cv::Point3f> vectorObject;
	vectorObject.push_back(point);

	std::vector<cv::Point2f> listPoints;

	cv::Mat points2d;
	cv::projectPoints(vectorObject, rvec, tvec, cameraMatrix, distortion, listPoints);
	return listPoints[0];
}

cv::Point3f ClassCamCalib::Project2DTo3D(cv::Point2f point, cv::Mat cameraMatrix,
		cv::Mat rvec, cv::Mat tvec, cv::Mat distortion) {
	// Stack overflow references
	// https://stackoverflow.com/questions/48038817/opencv-get-3d-coordinates-from-2d-pixel

	// Como realizar el despeje de las ecuaciones
	// https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point

	// Obtencion de parametros intrinsecos de la camara
	// http://answers.opencv.org/question/17076/conversion-focal-distance-from-mm-to-pixels/

	// Correccion de la distorsion en la imagen
	// https://stackoverflow.com/questions/30881607/camera-calibration-with-opencv-using-the-distortion-and-rotation-translation-ma

	// Formula para la correccion de la distorsion en la imagen
	// https://physics.stackexchange.com/questions/273464/what-is-the-tangential-distortion-of-opencv-actually-tangential-to

	cv::Mat cameraRotation;
	Rodrigues(rvec, cameraRotation);


	// Generating new point
	// Verificar salida undistort
	// https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi
	std::vector<cv::Point2f> input;
	input.push_back(point);
	std::vector<cv::Point2f> output;

	cv::undistortPoints(input, output, cameraMatrix, distortion, cv::noArray(), cameraMatrix);
	std::cout << "output: " << output << std::endl;

	// Remapping distorted points
	cv::Mat pointMat(3, 1, CV_64F);
	pointMat.at<double>(0, 0) = output[0].x;
	pointMat.at<double>(1, 0) = output[0].y;
	pointMat.at<double>(2, 0) = 1;

	// Calculating S
	// View https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point0
	cv::Mat tempMat = cameraRotation.inv() * cameraMatrix.inv() * pointMat;
	cv::Mat tempMat2 = cameraRotation.inv() * tvec;

	double zValue = 0;
	double s = zValue + tempMat2.at<double>(2, 0);
	s = s / tempMat.at<double>(2, 0);

	// Calculating point
	// View https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point0
	cv::Mat point3D = cameraRotation.inv() * (s * cameraMatrix.inv() * pointMat - tvec);
	cv::Point3f point3f(point3D.at<double>(0, 0), point3D.at<double>(1, 0), point3D.at<double>(2, 0));

	return point3f;


	/*
		Old Method
		cv::Mat PMatrix(3, 3, cameraRotation.type()); // T is 3x4
		setIdentity(PMatrix);

		PMatrix(cv::Range(0,3), cv::Range(0, 2)) = cameraRotation(cv::Range(0,3), cv::Range(0,2)) * 1;

		PMatrix.at<double>(0, 2) = tvec.at<double>(0, 0);
		PMatrix.at<double>(1, 2) = tvec.at<double>(1, 0);
		PMatrix.at<double>(2, 2) = tvec.at<double>(2, 0);

		cv::Mat pointResult = PMatrix.inv() * cameraMatrix.inv() * pointMat;

		auto xValue = pointResult.at<double>(0) / pointResult.at<double>(2);
		auto yValue = pointResult.at<double>(1) / pointResult.at<double>(2);
		std::cout << "x" << xValue << std::endl;
		std::cout << "y" << yValue << std::endl;

		return cv::Point3f(xValue, yValue, 0);
	*/
}

CalibResults ClassCamCalib::CalibrateCamera(std::vector<cv::Point2f> listPoints2D, std::vector<cv::Point3f> listPoints3D) {
	CalibResults results;

	// Assuming ideal camera Matrix
	// No need to get intrinsic coefficients
	results.cameraMatrix = cv::Mat(3, 3, cv::DataType<double>::type);
	results.cameraMatrix.at<double>(0,0) = 909;
	results.cameraMatrix.at<double>(0,1) = 0;
	results.cameraMatrix.at<double>(0,2) = 640;

	results.cameraMatrix.at<double>(1,0) = 0;
	results.cameraMatrix.at<double>(1,1) = 909;
	results.cameraMatrix.at<double>(1,2) = 480;

	results.cameraMatrix.at<double>(2,0) = 0;
	results.cameraMatrix.at<double>(2,1) = 0;
	results.cameraMatrix.at<double>(2,2) = 1;

	results.distortion = cv::Mat(4,1,cv::DataType<double>::type);
	results.distortion.at<double>(0) = 0;
	results.distortion.at<double>(1) = 0;
	results.distortion.at<double>(2) = 0;
	results.distortion.at<double>(3) = 0;

	results.rvec = cv::Mat(3,1,cv::DataType<double>::type);
	results.tvec = cv::Mat(3,1,cv::DataType<double>::type);

	cv::solvePnP(listPoints3D, listPoints2D, results.cameraMatrix, results.distortion, results.rvec, results.tvec);
	return results;
}

cv::Mat ClassCamCalib::GetHomography(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> objectPoints) {
	cv::Mat results = cv::findHomography(imagePoints, objectPoints);
	return results;
}

cv::Point2f ClassCamCalib::ProjectHomography(cv::Point2f point, cv::Mat homographyMat) {
	cv::Mat pointMat(3, 1, CV_64F);
	pointMat.at<double>(0, 0) = point.x;
	pointMat.at<double>(1, 0) = point.y;
	pointMat.at<double>(2, 0) = 1;

	cv::Mat projectedPoint = homographyMat * pointMat;

	cv::Point2f result;
	result.x = projectedPoint.at<double>(0, 0) / projectedPoint.at<double>(2, 0);
	result.y = projectedPoint.at<double>(1, 0) / projectedPoint.at<double>(2, 0);

	return result;
}

CalibResults ClassCamCalib::CalibrateCamera(std::vector<cv::Point2f> listPoints2D, std::vector<cv::Point3f> listPoints3D, cv::Mat camMat, cv::Mat dist) {
	CalibResults results;

	// Assuming ideal camera Matrix
	// No need to get intrinsic coefficients
	results.cameraMatrix = camMat;
	results.distortion = dist;

	results.rvec = cv::Mat(3,1,cv::DataType<double>::type);
	results.tvec = cv::Mat(3,1,cv::DataType<double>::type);

	cv::solvePnP(listPoints3D, listPoints2D, results.cameraMatrix, results.distortion, results.rvec, results.tvec);
	return results;
}


CalibResults ClassCamCalib::CalibrateCameraCalib(std::vector<cv::Point2f> listPoints2D, std::vector<cv::Point3f> listPoints3D, cv::Size sizeImage) {
	std::cout << "2D: " << listPoints2D << std::endl;
	std::cout << "3D: " << listPoints3D << std::endl;

	CalibResults results;

	// Assuming ideal camera Matrix
	// No need to get intrinsic coefficients
	results.cameraMatrix = cv::Mat::eye(3, 3, cv::DataType<double>::type);
	results.distortion = cv::Mat::zeros(4,1,cv::DataType<double>::type);
	results.rvec = cv::Mat::zeros(3,1,cv::DataType<double>::type);
	results.tvec = cv::Mat::zeros(3,1,cv::DataType<double>::type);


	std::cout << "Camera matrix: " << results.cameraMatrix << std::endl;
	std::cout << "Dist coeffs: " << results.distortion << std::endl;
	std::cout << "R vector: "  << results.rvec << std::endl;
	std::cout << "T vector: " << results.tvec << std::endl;

	std::vector<std::vector<cv::Point2f>> imagePoints;
	std::vector<std::vector<cv::Point3f>> objectPoints;

	imagePoints.push_back(listPoints2D);
	objectPoints.push_back(listPoints3D);


	cv::calibrateCamera(objectPoints, imagePoints, sizeImage, results.cameraMatrix, results.distortion, results.rvec, results.tvec);


	return results;
}


std::string ClassCamCalib::CalibResultsToString(CalibResults results) {
	json obj;

	obj["cameraMatrix"] = ImageProcess::GetCSV(results.cameraMatrix);
	obj["distortion"] = ImageProcess::GetCSV(results.distortion);
	obj["rvec"] = ImageProcess::GetCSV(results.rvec);
	obj["tvec"] = ImageProcess::GetCSV(results.tvec);

	return obj.dump();
}


CalibResults ClassCamCalib::StringToCalibResults(std::string resultsString) {
	json obj = json::parse(resultsString);
	CalibResults results;

	results.cameraMatrix = ImageProcess::LoadFromCSV(obj["cameraMatrix"]);
	results.distortion = ImageProcess::LoadFromCSV(obj["cameraMatrix"]);
	results.rvec = ImageProcess::LoadFromCSV(obj["rvec"]);
	results.tvec = ImageProcess::LoadFromCSV(obj["tvec"]);

	return results;
}

