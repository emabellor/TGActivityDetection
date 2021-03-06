/*
 * ClassMain.cpp
 *
 *  Created on: Feb 4, 2018
 *      Author: mauricio
 */

#include "ClassMain.h"

using namespace std;
using namespace cv;
using json = nlohmann::json;

// Static variable declaration
AutoResetEvent ClassMain::clickEvent;
cv::Point2f ClassMain::pointClicked;
bool ClassMain::mouseClicked = false;

ClassMain::ClassMain() {
	// TODO Auto-generated constructor stub

}

ClassMain::~ClassMain() {
	// TODO Auto-generated destructor stub
}

void ClassMain::InitLogger() {

}


void ClassMain::InitProcess(int argc, char** argv) {
	cout << "OpenCV Version: " << CV_VERSION << endl;

	cout << "Initializing process" << endl;

	int response = 0;
	while (true) {
		cout << "Select case" << endl;
		cout << "1: ShowKeyPoints" << endl;
		cout << "2: SelectKeyPoints" << endl;
		cout << "3: Person re-identification" << endl;
		cout << "4: Extract key poses descriptors" << endl;
		cout << "5: Clustering pose descriptors" << endl;
		cout << "6: Test Draw Pose" << endl;
		cout << "7: Extract descriptors lu" << endl;
		cout << "8: Test video cap info" << endl;
		cout << "9: Test video wrapper" << endl;
		cout << "10: Convert video MJPEG record" << endl;
		cout << "11: Task convert video Folder" << endl;
		cout << "12: SolvePNP Testing" << endl;
		cout << "13: Calibrate Camera" << endl;
		cout << "14: Calibrate Camera CV" << endl;
		cout << "15: Test JSON serialization" << endl;
		cout << "16: Check DoubleToString" << endl;
		cout << "17: Calibrate CameraLoad" << endl;
		cout << "18: Testing chess board" << endl;
		cout << "19: Test calibration calib" << endl;
		cout << "20: Test chessboard calib" << endl;
		cout << "21: Find Homography" << endl;

		cin >> response;

		if (response < 1 || response > 21) {
			cout << "You must insert a number between 1 and 11" << endl;
		} else {
			break;
		}
	}

	switch (response) {
		case 1: {
			ProcessKeyPoints(argc, argv);
			break;
		}
		case 2: {
			ShowKeyPoints(argc, argv);
			break;
		}
		case 3: {
			PersonReidentification(argc, argv);
			break;
		}
		case 4: {
			ExtractKeyPosesDescriptor(argc, argv);
			break;
		}
		case 5: {
			Clustering(argc, argv);
			break;
		}
		case 6: {
			TestDrawPose(argc, argv);
			break;
		}
		case 7: {
			ExtractDescriptorsDu(argc, argv);
			break;
		}
		case 8: {
			TestVideoCapInfo(argc, argv);
			break;
		}
		case 9: {
			TestVideoWrapper(argc, argv);
			break;
		}
		case 10: {
			TestConvertVideoMJPEG(argc, argv);
			break;
		}
		case 11: {
			TaskConvertVideoFolder(argc, argv);
			break;
		}
		case 12: {
			SolvePNPTesting(argc, argv);
			break;
		}
		case 13: {
			CalibrateCamera(argc, argv);
			break;
		}
		case 14: {
			CalibrateCameraCv(argc, argv);
			break;
		}
		case 15: {
			TestJSONSerialization(argc, argv);
			break;
		}
		case 16: {
			CheckDoubleToString(argc, argv);
			break;
		}
		case 17: {
			CalibrateCameraLoad(argc, argv);
			break;
		}
		case 18: {
			TestChessBoardCorners(argc, argv);
			break;
		}
		case 19: {
			CalibrateCameraCustom(argc, argv);
			break;
		}
		case 20: {
			ChessboardCalib(argc, argv);
			break;
		}
		case 21: {
			FindHomography(argc, argv);
			break;
		}
		default: {
			cout << "Response invalid!" << endl;
		}
	}
}

void ClassMain::ProcessKeyPoints(int argc, char** argv) {
	cout << "ProcessKeyPoints" << endl;

	ClassOpenPose poseDetector;

	cout << "Initializing" << endl;
	poseDetector.InitOpenPose();

	string imagePath = defaultImage;

	if (argc > 2) {
		imagePath = string(argv[1]);
	}

	cout << "ImagePath: " << imagePath << endl;

	Mat image1 = imread(imagePath);

	cout << "Extracting data" << endl;
	poseDetector.ExtractAndShow(image1);

	cout << "ProcessKeyPoints finished" << endl;
}

void ClassMain::ShowKeyPoints(int argc, char** argv) {
	cout << "ShowKeyPoints" << endl;

	ClassOpenPose poseDetector;

	cout << "Initializing" << endl;
	poseDetector.InitOpenPose();

	string imagePath = defaultImage;

	if (argc > 2) {
		imagePath = string(argv[1]);
	}

	cout << "ImagePath: " << imagePath << endl;

	Mat image1 = imread(imagePath);

	cout << "Extracting data" << endl;
	auto results = poseDetector.ExtractKeyPoints(image1);
	auto listElems = results.GetAllPoints();

	cout << "List elements count: " << listElems.size() << endl;
	for (uint i = 0; i < listElems.size(); i++) {
		auto item = listElems[i];
		cout << "Person: " << item.person << " BodyPart: " << item.bodyPart << " x: "  << item.pos.x
				<< " y: " << item.pos.y << " score: " << item.score << endl;

		cv::Point pointInit;
		pointInit.x = item.pos.x - rectSize / 2;
		pointInit.y = item.pos.y - rectSize / 2;

		cv::Point pointEnd;
		pointEnd.x = item.pos.x + rectSize / 2;
		pointEnd.y = item.pos.y + rectSize / 2;

		cv::rectangle(image1, pointInit, pointEnd, cv::Scalar(255, 0, 0));
	}

	cout << "Showing image" << endl;
	cv::namedWindow("winMain", CV_WINDOW_AUTOSIZE);
	cv::imshow("winMain", image1);
	cv::waitKey(0);

	cout << "Process Finished!" << endl;
}

void ClassMain::PersonReidentification(int argc, char** argv) {
	cout << "Person re-identification run" << endl;

	ClassOpenPose poseDetector;
	poseDetector.InitOpenPose();

	cout << "Loading images" << endl;
	cout << "Image1 Path: " << defaultCompareImage1 << endl;
	cout << "Image2 Path: " << defaultCompareImage2 << endl;

	cv::Mat image1 = imread(defaultCompareImage1);
	cv::Mat image2 = imread(defaultCompareImage2);

	cout << "Extract points from image" << endl;

	auto results1 = poseDetector.ExtractKeyPoints(image1);
	auto results2 = poseDetector.ExtractKeyPoints(image2);

	if (results1.GetPeopleAmount() != 1) {
		cout << "People number from image1 is not 1" << endl;
	} else if (results2.GetPeopleAmount() != 1) {
		cout << "People number from image2 is not 1" << endl;
	} else {
		// Right elbow - 3
		// Left elbow - 4
		// Right knee - 9
		// Left knee - 12

		auto pose1RU = results1.GetPose(0, 3);
		auto pose1LU = results1.GetPose(0, 6);

		auto pose2RU = results2.GetPose(0, 3);
		auto pose2LU = results2.GetPose(0, 6);

		double color1U = GetColorPoses(image1, pose1RU, pose1LU);
		double color2U = GetColorPoses(image2, pose2RU, pose2LU);

		cout << endl;
		if (color1U == -1) {
			cout << "Image1 upper ignored" << endl;
		} else if (color2U == -1) {
			cout << "Image2 upper ignored" << endl;
		} else {
			cout << "Color1U " << color1U << endl;
			cout << "Color2U " << color2U << endl;
			cout << endl;
		}

		auto pose1RD = results1.GetPose(0, 9);
		auto pose1LD = results1.GetPose(0, 12);

		auto pose2RD = results2.GetPose(0, 9);
		auto pose2LD = results2.GetPose(0, 12);

		double color1D = GetColorPoses(image1, pose1RD, pose1LD);
		double color2D = GetColorPoses(image2, pose2RD, pose2LD);

		if (color1D == -1) {
			cout << "Image1 lower ignored" << endl;
		} else if (color2D == -1) {
			cout << "Image2 lower ignored" << endl;
		} else {
			cout << "Color1D " << color1D << endl;
			cout << "Color2D " << color2D << endl;
			cout << endl;
		}

		cout << "Program finished successfully" << endl;
	}
}

double ClassMain::GetColorPoses(cv::Mat image, StructPoints pose1, StructPoints pose2) {
	double result = 0;
	int count = 0;

	if (pose1.score >= scoreThresh) {
		cv::Rect rectangle(pose1.pos.x - rectSize / 2, pose1.pos.y - rectSize / 2, rectSize, rectSize);
		auto color = ImageProcess::GetAverageColor(image, rectangle);

		result += color;
		count++;
	}

	if (pose2.score >= 0.5) {
		cv::Rect rectangle(pose2.pos.x - rectSize / 2, pose2.pos.y - rectSize / 2, rectSize, rectSize);
		auto color = ImageProcess::GetAverageColor(image, rectangle);

		result += color;
		count++;
	}

	if (count == 0) {
		result = -1;
	} else {
		result = result / count;
	}

	return result;

}

// Based on Du article
void ClassMain::ExtractDescriptorsDu(int argc, char** argv) {
	cout << "Initializing ExtractDescriptorsDu" << endl;

	vector<DescriptorDu> listDescriptors;
	// Walk
	{
		DescriptorDu desc;
		desc.identifier = 0;
		desc.files = FileHandler::ReadAllFiles("/home/mauricio/Videos/Datasets/Weizzman/walk/");
		listDescriptors.push_back(desc);
	}
	// Run
	{
		DescriptorDu desc;
		desc.identifier = 1;
		desc.files = FileHandler::ReadAllFiles("/home/mauricio/Videos/Datasets/Weizzman/run/");
		listDescriptors.push_back(desc);
	}
	// Jump
	{
		DescriptorDu desc;
		desc.identifier = 2;
		desc.files = FileHandler::ReadAllFiles("/home/mauricio/Videos/Datasets/Weizzman/jump/");
		listDescriptors.push_back(desc);
	}
	// Side
	{
		DescriptorDu desc;
		desc.identifier = 3;
		desc.files = FileHandler::ReadAllFiles("/home/mauricio/Videos/Datasets/Weizzman/side/");
		listDescriptors.push_back(desc);
	}
	// Bend
	{
		DescriptorDu desc;
		desc.identifier = 4;
		desc.files = FileHandler::ReadAllFiles("/home/mauricio/Videos/Datasets/Weizzman/bend/");
		listDescriptors.push_back(desc);
	}


	cout << "Initializing poses" << endl;
	ClassOpenPose poseDetector;
	poseDetector.jniFlag = true;
	poseDetector.InitOpenPose();

	std::string outputFolder = "/home/mauricio/folderdu";


	cout << "Initializing processing" << endl;
	string textToWrite = "";

	for (uint cls = 0; cls < listDescriptors.size(); cls++) {
		auto listVideos = listDescriptors.at(cls).files;
		auto groupId = listDescriptors.at(cls).identifier;

		for (uint index = 0; index < listVideos.size(); index++) {
			auto videoFile = listVideos.at(index);
			cout << "Opening video file" << videoFile << endl;
			auto listImages = ImageProcess::GetImagesFromVideo(videoFile);

			cout << "Total image length: " << listImages.size() << endl;

			// Array struct
			vector<StructDu> listArrays;
			for (uint i = 0; i < listImages.size(); i++) {
				if (i % 10 == 0) {
					cout << "Frame: "  << i << endl;
				}

				auto image = listImages.at(i);
				auto results = poseDetector.ExtractKeyPoints(image);

				cout << "Getting people amount " << results.GetPeopleAmount() << endl;
				if (results.GetPeopleAmount() != 1) {
					cout << "Only one person per image" << endl;
				} else {
					// Step 1 - Same points
					// Must be translated
					int personIndex = 0;
					auto resultPerson = results.GetPointsByPersonTranslate(personIndex);

					// Step 2 - Vector generation: X, Y, Z
					StructDu arrays;

					// Step 3 - Vector serialization
					double min_x = 0;
					double max_x = 0;
					double min_y = 0;
					double max_y = 0;

					for (int i = 0; i < 14; i++) {
						// X
						arrays.arrayX[i] = resultPerson.at(i).pos.x;
						if (resultPerson.at(i).pos.x < min_x) {
							min_x = resultPerson.at(i).pos.x;
						}

						if (resultPerson.at(i).pos.x > max_x) {
							max_x = resultPerson.at(i).pos.x;
						}

						// Y
						arrays.arrayY[i] = resultPerson.at(i).pos.y;
						if (resultPerson.at(i).pos.y < min_y) {
							min_y = resultPerson.at(i).pos.y;
						}

						if (resultPerson.at(i).pos.y > max_y) {
							max_y = resultPerson.at(i).pos.y;
						}

						// Z
						arrays.arrayZ[i] = 0;
					}


					// Step 4 - Normalization
					for (int i = 0; i < 14; i++) {
						arrays.arrayX[i] = 255 * (arrays.arrayX[i] - min_x) / (max_x - min_x);
						arrays.arrayY[i] = 255 * (arrays.arrayY[i] - min_y) / (max_y - min_y);
					}

					listArrays.push_back(arrays);
				}

			}

			// Step 5 - Image creation
			int rows = 14;
			int cols = listArrays.size();

			Mat imageResult(rows, cols, CV_8UC3, Scalar::all(0));
			for (int j = 0; j < cols; j++)  {
				for (int i = 0; i < rows; i++) {
					auto elem = listArrays.at(j);

					Vec3b color;

					color.val[0] = 0; // B
					color.val[1] = (uchar)elem.arrayY[i]; // G
					color.val[2] = (uchar)elem.arrayX[i]; // R

					imageResult.at<Vec3b>(Point(j, i)) = color;
				}
			}

			// Step 6 - Check if output folder exists
			if (FileHandler::DirectoryExists(outputFolder) == false) {
				FileHandler::CreateDirectory(outputFolder);
			}

			// Parsing strategy
			string stem = FileHandler::GetFileNameNoExtension(videoFile);
			string fileName = "video%" + to_string(groupId) + "%" + stem + ".jpg";
			string fullName = FileHandler::JoinPath(outputFolder, fileName);

			cv::Mat imgResize = ImageProcess::Resize(imageResult, Size(64, 64));
			imwrite(fullName, imgResize);
		}
	}

	cout << "Function succeded!" << endl;
}

void ClassMain::ExtractKeyPosesDescriptor(int argc, char** argv) {
	// Based on Weizzman dataset

	vector<string> listVideos;
	listVideos.push_back("/home/mauricio/Videos/Datasets/Weizzman/walk/daria_walk.avi");
	listVideos.push_back("/home/mauricio/Videos/Datasets/Weizzman/jump/daria_jump.avi");
	listVideos.push_back("/home/mauricio/Videos/Datasets/Weizzman/run/daria_run.avi");

	cout << "Extracting poses" << endl;
	ClassOpenPose poseDetector;
	poseDetector.InitOpenPose();

	cout << "Opening file" << endl;
	auto outputFileName = "/home/mauricio/file.csv";
	std::ofstream outFile(outputFileName);

	cout << "Initializing processing" << endl;
	string textToWrite = "";
	for (uint index = 0; index < listVideos.size(); index++) {

		auto videoFile = listVideos.at(index);
		cout << "Opening video file" << videoFile << endl;
		auto listImages = ImageProcess::GetImagesFromVideo(videoFile);
		cout << "Total image length: " << listImages.size() << endl;

		for (uint i = 0; i < listImages.size(); i++) {
			if (i % 10 == 0) {
				cout << "Frame: "  << i << endl;
			}

			auto image = listImages.at(i);
			auto results = poseDetector.ExtractKeyPoints(image);

			cout << "Getting people amount " << results.GetPeopleAmount() << endl;

			for (uint j = 0; j < results.GetPeopleAmount(); j++) {
				string textLocal = "";
				if (i != 0 || j != 0 || index != 0) {
					textLocal += "\n";
				}

				auto partsNorm = results.GetPointsByPersonNorm(j);
				for (uint k = 0; k < partsNorm.size(); k++) {
					if (k != 0) {
						textLocal += ";";
					}

					textLocal += boost::str(boost::format("%.4f") % partsNorm.at(k).pos.x);
					textLocal += ";";
					textLocal += boost::str(boost::format("%.4f") % partsNorm.at(k).pos.y);
				}
				if (textLocal.find("inf") != string::npos) {
					cout << "Discarded by inf" << endl;
				} else {
					textToWrite += textLocal;
				}
			}
		}
	}


	cout << "CSV file: " << textToWrite << endl;
	cout << "Writing to file " << endl;

	outFile.write(textToWrite.c_str(), textToWrite.size());

	cout << "Routine executed" << endl;
}

void ClassMain::Clustering(int argc, char** argv) {
	cout << "Clustering" << endl;

	cout << "Initializing open pose" << endl;
	ClassOpenPose poseDetector;
	poseDetector.InitOpenPose();

	// Extract keypoints dummy image!
	string imagePath = defaultImage;
	cout << "ImagePath: " << imagePath << endl;
	Mat image1 = imread(imagePath);
	auto results = poseDetector.ExtractKeyPoints(image1);

	cout << "Loading file!" << endl;
	auto outputFileName = "/home/mauricio/file.csv";

	std::ifstream file(outputFileName);
	std::stringstream buffer;
	buffer << file.rdbuf();

	auto strFile = buffer.str();
	cout << "StrFile: " << strFile;

	cout << "Parsing File!" << endl;
	auto listElems = SplitStr(strFile, "\n");

	uint totalSamples = listElems.size();
	uint dim = 18 * 2; //2 dimensional points - 18 body parts
	uint newDim = 14 * 2; // 14 bodyParts of interest!

	// fill the data points
	cv::Mat points = cv::Mat::zeros(totalSamples, newDim, CV_32FC1);

	for (uint i = 0; i < listElems.size(); i++) {
		auto listLocal = SplitStr(listElems.at(i), ";");

		if (listLocal.size() != dim) {
			cerr << "Error split string " << listElems.at(i) << " size: " << listLocal.size() << endl;
			exit(1);
		}

		for(uint j = 0; j < newDim; j++) {
			points.at<float>(i, j) = stof(listLocal[j]);
		}
	}

	cout << "Training K-Means" << endl;

	cv::Mat labels; cv::Mat centers;
	uint clusters = 15;
    cv::kmeans(points, clusters, labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), 10, cv::KMEANS_PP_CENTERS, centers);

    for (int j = 0; j < centers.rows; ++j) {
    	// std::cout << centers.row(j) << std::endl;

    	vector<StructPoints> listPoints;
    	for (int k = 0; k < centers.cols; k += 2) {
    		// 14 * 2 = 28
    		StructPoints point;
    		point.person = j;
    		point.bodyPart = k / 2;
    		point.score = 1;
    		point.pos.x = centers.at<float>(j, k);
    		point.pos.y = centers.at<float>(j, k + 1);

    		listPoints.push_back(point);
    	}

    	for (int k = 14; k < 18; k++) {
    		StructPoints point;
    		point.bodyPart = k;
    		point.score = 0;
    		listPoints.push_back(point);
    	}


    	double factor = 40;
    	poseDetector.DrawPose(listPoints, true, factor);
    }



	cout << "Exit!" << endl;
}

vector<string> ClassMain::SplitStr(string str, string delimiter) {
	vector<string> list;

	auto s = str;
	size_t pos = 0;
	std::string token;
	while ((pos = s.find(delimiter)) != std::string::npos) {
	    token = s.substr(0, pos);
	    std::cout << token << std::endl;
	    list.push_back(token);
	    s.erase(0, pos + delimiter.length());
	}

	list.push_back(s);
	return list;
}

void ClassMain::TestDrawPose(int argc, char** argv) {
	cout << "Init Test Draw pose!" << endl;

	cout << "ShowKeyPoints" << endl;

	ClassOpenPose poseDetector;

	cout << "Initializing" << endl;
	poseDetector.InitOpenPose();

	string imagePath = defaultImage;

	if (argc > 2) {
		imagePath = string(argv[1]);
	}

	cout << "ImagePath: " << imagePath << endl;

	Mat image1 = imread(imagePath);

	cout << "Extracting data" << endl;
	auto results = poseDetector.ExtractKeyPoints(image1);
	auto listPoints = results.GetPointsByPerson(0); // First person assuming

	cout << "Showing points!" << endl;
	poseDetector.DrawPose(listPoints);

	listPoints = results.GetPointsByPersonNorm(0);
	auto factor = results.factor;

	cout << "Showing points norm!" << endl;
	poseDetector.DrawPose(listPoints, true, factor);

	cout << "End!" << endl;
}

void ClassMain::TestVideoCapInfo(int argc, char** argv) {
	cout << "Testing video cap info" << endl;
	cout << "Opening file" << endl;

	string fileName = "/home/mauricio/Videos/Datasets/Weizzman/jump/daria_jump.avi";
	auto listFrames = ImageProcess::GetImagesFromVideo(fileName); // Just for testing
	cout << "List frames count: " << listFrames.size() << endl;

	VideoCapInfo info(fileName);
	if (info.IsOpened() == false) {
		cout << "Error loading file" << endl;
	} else {
		cout << "Loading frames" << endl;
		int count = 0;
		while(true) {
			FrameInfo frameInfo = info.GetNextImage();
			if (frameInfo.isEmpty == true) {
				break;
			}

			count++;
			cout << "Counting frame " << count << endl;
			cout << "Len: "  << frameInfo.len << endl;
		}
	}

	cout << "Finished!" << endl;
}

void ClassMain::TestVideoWrapper(int argc, char** argv) {
	cout << "Testing video cap info" << endl;
	cout << "Opening file" << endl;

	string fileName = "/home/mauricio/Videos/Datasets/Weizzman/jump/daria_jump.avi";
	auto listFrames = ImageProcess::GetImagesFromVideo(fileName); // Just for testing
	cout << "List frames count: " << listFrames.size() << endl;



	auto guidStr = ClassWrapper_LoadVideo(fileName);

	if (guidStr.compare("") == 0) {
		cout << "Can't open file" << endl;
	} else {
		cout << "Loading frames" << endl;
		int count = 0;
		while(true) {
			FrameInfo frameInfo = ClassWrapper_GetNextImage("hola");
			if (frameInfo.isEmpty == true) {
				break;
			}

			count++;
			cout << "Counting frame " << count << endl;
			cout << "Len: "  << frameInfo.len << endl;
		}
	}

	cout << "Finished!" << endl;
}

void ClassMain::TestConvertVideoMJPEG(int argc, char** argv) {
	cout << "Testing convert video MJPEG" << endl;

	string fileName = "/home/mauricio/Videos/mjpeg/video.mjpeg";

	cout << "Extract frames" << endl;
	auto list = ClassMJPEGReader::ProcessVideo(fileName);

	cout << "Total videos in list: " << list.size() << endl;

	cout << "Initializing open pose";
	ClassOpenPose openPose;
	openPose.InitOpenPose();

	cout << "Processing frames" << endl;

	ClassMJPEGReader saver;
	saver.OpenFileSave("/home/mauricio/Videos/mjpeg/video.mjpegx");
	for(uint i = 0; i < list.size(); i++) {
		auto item = list.at(i);

		Mat image = ImageProcess::GetFromCharBuffer(item.image, item.len);
		auto results = openPose.ExtractKeyPoints(image);

		// Saving in video
		saver.WriteFrame(item, results);
		cout << "Count: " << i << endl;
	}

	cout << "Closing video file" << endl;
	saver.CloseFileSave();

	cout << "Done!" << endl;
}

void ClassMain::TaskConvertVideoFolder(int argc, char** argv) {
	cout << "Init convert video MJPEG task" << endl;

	cout << "Initializing open pose" << endl;
	ClassOpenPose openPose;
	openPose.InitOpenPose();

	string folderName = "/home/mauricio/Videos/Oviedo/";

	cout << "Reading all files inside folder" << endl;
	auto folderList = FileHandler::ReadAllFilesRecursively(folderName);
	cout << "Total videos in list: " << folderList.size() << endl;

	for(uint i = 0; i < folderList.size(); i++) {
		auto file = folderList.at(i);
		auto fileExtension = FileHandler::GetFileExtension(file);

		if (fileExtension.compare(".mjpegx") == 0) {
			cout << "File extension is .mjpegx - Ignoring";
		} else {
			cout << "File extension is not .mjpegx - Converting";
			auto newFileName = FileHandler::ChangeExtension(file, "mjpegx");

			cout << "Checking if file already exists" << endl;
			if (FileHandler::FileExists(newFileName) == true) {
				cout << "File already exists!" << endl;
			} else {
				cout << "Extract frames" << endl;
				auto list = ClassMJPEGReader::ProcessVideo(file);
				auto countList = list.size();

				ClassMJPEGReader saver;
				saver.OpenFileSave(newFileName);
				for(uint i = 0; i < list.size(); i++) {
					auto item = list.at(i);

					Mat image = ImageProcess::GetFromCharBuffer(item.image, item.len);
					auto results = openPose.ExtractKeyPoints(image);


					// Saving in video
					saver.WriteFrame(item, results);
					cout << "Count: " << i << " count list " << countList << endl;
				}

				cout << "Closing video file" << endl;
				saver.CloseFileSave();
			}
		}
	}

	cout << "Done" << endl;
}

void ClassMain::SolvePNPTesting(int argc, char** argv) {
	cout << "Solving PNP" << endl;

	std::vector<cv::Point2f> imagePoints;
	std::vector<cv::Point3f> objectPoints;

	imagePoints.push_back(cv::Point2f(110.,100.));
	imagePoints.push_back(cv::Point2f(190.,100.));
	imagePoints.push_back(cv::Point2f(100.,200.));
	imagePoints.push_back(cv::Point2f(200.,200.));

	//object points are measured in millimeters because calibration is done in mm also
	objectPoints.push_back(cv::Point3f(500., 500., 0.));
	objectPoints.push_back(cv::Point3f(510.,500.,0.));
	objectPoints.push_back(cv::Point3f(500.,510.,0.));
	objectPoints.push_back(cv::Point3f(510.,510.,0.));

	cv::Mat rvec(1,3,cv::DataType<double>::type);
	cv::Mat tvec(1,3,cv::DataType<double>::type);
	cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);

	// Assuming ideal camera Matrix
	// No need to get intrinsic coefficients
	Mat cameraMatrix(3, 3, DataType<double>::type);
	cameraMatrix.at<double>(0,0) = 295;
	cameraMatrix.at<double>(0,1) = 0;
	cameraMatrix.at<double>(0,2) = 315;

	cameraMatrix.at<double>(1,0) = 0;
	cameraMatrix.at<double>(1,1) = 222;
	cameraMatrix.at<double>(1,2) = 233;

	cameraMatrix.at<double>(2,0) = 0;
	cameraMatrix.at<double>(2,1) = 0;
	cameraMatrix.at<double>(2,2) = 1;

	cout << "Camera Matrix: " << cameraMatrix << endl;
	cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
	distCoeffs.at<double>(0) = 0;
	distCoeffs.at<double>(1) = 0;
	distCoeffs.at<double>(2) = 0;
	distCoeffs.at<double>(3) = 0;

	cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
	cout << "Testing 2D to 3D" << endl;
	for (uint i = 0; i < imagePoints.size(); i++) {
		auto point = ClassCamCalib::Project2DTo3D(imagePoints[i], cameraMatrix, rvec, tvec, distCoeffs);
		cout << "Result: " << point << endl;
	}

	cout << "Testing 3D to 2D" << endl;
	for (uint i = 0; i < imagePoints.size(); i++) {
		auto point = ClassCamCalib::Project3DTo2D(objectPoints[i], cameraMatrix, rvec, tvec, distCoeffs);
		cout << "Result: " << point << endl;
	}


	cout << "Method SolvePNPTesting Done!" << endl;
}

void ClassMain::CalibrateCamera(int argc, char** argv) {
	cout << "Please digit the ID of the camera" << endl;
	int idCam;
	cin >> idCam;

	cout << "The ID of the camera is " << idCam << endl;

	cout << "Please open the sample file" << endl;

	char file[1024];
	FILE *f = popen("zenity --file-selection", "r");
	fgets(file, 1024, f);
	std::string fileNameStr(file);

	// Bug popen - Must replace ocurrences!
	boost::replace_all(fileNameStr, "\r", "");
	boost::replace_all(fileNameStr, "\n", "");

	if (f == NULL) {
		cerr << "FILE not selected -- Finishing...";
	} else {
		cout << "FileName: " << fileNameStr << endl;

		cout << "Loading image" << fileNameStr << endl;
		Mat image = imread(fileNameStr);

		if (image.data == NULL) {
			cerr << "Error loading image " << fileNameStr << " done!" << endl;
		} else {
			cvNamedWindow("showWindow", CV_WINDOW_AUTOSIZE);
			setMouseCallback("showWindow", ClickEventHandler, NULL);
			imshow("showWindow", image);

			vector<Point2f> image_points;
			vector<Point3f> world_points;

			while(image_points.size() < 4) {
				cout << "Click the image to get point " << image_points.size() << endl;

				mouseClicked = false;
				int waitTimeoutMs = 100;

				while (true) {
					cvWaitKey(waitTimeoutMs);

					if (mouseClicked == true) {
						break;
					}
				}

				cout << "Enter points - separated by comma - z must be zero - Only x and y" << endl;
				string input;
				cin >> input;

				vector<string> elems = SplitStr(input, ",");
				if (elems.size() != 2) {
					cerr << "Invalid format! ignoring point" << endl;
				} else {
					image_points.push_back(pointClicked);

					Point3f worldPoint(atoi(elems[0].c_str()), atoi(elems[1].c_str()), 0);
					world_points.push_back(worldPoint);

					cout << "ImagePoint: " << pointClicked << endl;
					cout << "WorldPoint: " << worldPoint << endl;
				}
			}

			cout << "Image Points: " << image_points << endl << endl;
			cout << "World Points: " << world_points << endl << endl;

			// Assuming ideal camera Matrix
			// No need to get intrinsic coefficients
			auto results = ClassCamCalib::CalibrateCamera(image_points, world_points);

			cout << "PNP solved - Click For Testing" << endl;
			int timeoutClickedMs = 100;
			while(true) {
				mouseClicked = false;
				auto keyPressed = cvWaitKey(timeoutClickedMs);

				if (keyPressed == 'q' || keyPressed == 'Q') {
					break;
				} else if (mouseClicked == true){
					mouseClicked = false;
					cout << "Evaluating point" << endl;

					auto pointResult = Project2DTo3D(pointClicked, results.cameraMatrix, results.rvec, results.tvec, results.distortion);
					cout << "Point result: " << pointResult << endl;
				}
			}
		}
	}
}

void ClassMain::ClickEventHandler(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN) {
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		pointClicked = Point2f(x, y);
		mouseClicked = true;
	}
}


cv::Point2f ClassMain::Project3DTo2D(cv::Point3f point, cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec, cv::Mat distorsion) {
	cv::Mat cameraRotation;
	Rodrigues(rvec, cameraRotation);

	cv::Mat pointMat(4, 1, CV_64F);
	pointMat.at<double>(0, 0) = point.x;
	pointMat.at<double>(1, 0) = point.y;
	pointMat.at<double>(2, 0) = point.z;
	pointMat.at<double>(3, 0) = 1;

	cv::Mat PMatrix(3, 4, cameraRotation.type()); // T is 3x4
	PMatrix(cv::Range(0,3), cv::Range(0,3)) = cameraRotation * 1; // copies R into T
	PMatrix(cv::Range(0,3), cv::Range(3,4)) = tvec * 1; // copies tvec into T

	cv::Mat resultPoint = cameraMatrix * PMatrix * pointMat;

	auto xValue = resultPoint.at<double>(0) / resultPoint.at<double>(2);
	auto yValue = resultPoint.at<double>(1) / resultPoint.at<double>(2);
	cout << "x" << xValue << endl;
	cout << "y" << yValue << endl;

	return Point2f(xValue, yValue);
}

cv::Point3f ClassMain::Project2DTo3D(cv::Point2f point, cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec, cv::Mat distorsion) {
	// Stack overflow references
	// https://stackoverflow.com/questions/48038817/opencv-get-3d-coordinates-from-2d-pixel

	// Equation solving
	// https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point

	// Intrinsic parameters retrieving
	// http://answers.opencv.org/question/17076/conversion-focal-distance-from-mm-to-pixels/

	cv::Mat cameraRotation;
	Rodrigues(rvec, cameraRotation);

	cv::Mat pointMat(3, 1, CV_64F);
	pointMat.at<double>(0, 0) = point.x;
	pointMat.at<double>(1, 0) = point.y;
	pointMat.at<double>(2, 0) = 1;

	cv::Mat PMatrix(3, 3, cameraRotation.type()); // T is 3x4
	setIdentity(PMatrix);

	PMatrix(cv::Range(0,3), cv::Range(0, 2)) = cameraRotation(cv::Range(0,3), cv::Range(0,2)) * 1;

	PMatrix.at<double>(0, 2) = tvec.at<double>(0, 0);
	PMatrix.at<double>(1, 2) = tvec.at<double>(1, 0);
	PMatrix.at<double>(2, 2) = tvec.at<double>(2, 0);

	cv::Mat pointResult = PMatrix.inv() * cameraMatrix.inv() * pointMat;

	auto xValue = pointResult.at<double>(0) / pointResult.at<double>(2);
	auto yValue = pointResult.at<double>(1) / pointResult.at<double>(2);
	cout << "x" << xValue << endl;
	cout << "y" << yValue << endl;

	return Point3f(xValue, yValue, 0);
}

void ClassMain::CalibrateCameraCv(int argc, char** argv) {
	cout << "Initialize calibrate camera parameters" << endl;
	cout << "Solving all parameters!" << endl;

	std::vector<std::vector<Point2f>> imagePoints;
	std::vector<std::vector<Point3f>> objectPoints;


	std::vector<Point2f> vectorImage;
	vectorImage.push_back(cv::Point2f(10.,10.));
	vectorImage.push_back(cv::Point2f(10.,20.));
	vectorImage.push_back(cv::Point2f(20.,10.));
	vectorImage.push_back(cv::Point2f(20.,20.));
	imagePoints.push_back(vectorImage);


	//object points are measured in millimeters because calibration is done in mm also
	std::vector<Point3f> vectorObject;
	vectorObject.push_back(cv::Point3f(10,10., 0.));
	vectorObject.push_back(cv::Point3f(10, 20.,0.));
	vectorObject.push_back(cv::Point3f(20.,10.,0.));
	vectorObject.push_back(cv::Point3f(20.,20.,0.));
	objectPoints.push_back(vectorObject);


	vector<Mat> rvec;
	vector<Mat> tvec;
	cv::Mat rotationMatrix;
	cv::Mat cameraMatrix(3,3,CV_32FC1);
	cv::Mat distCoeff;

	setIdentity(cameraMatrix);

	cv::Size imageSize(320, 240);

	// Assuming ideal camera Matrix
	calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeff, rvec, tvec);

	cout << "Camera Matrix" << cameraMatrix << endl;



	cout << "Done!" << endl;
}

void ClassMain::SaveMatToCSV(string baseFolder, string fileName, Mat matrix) {
	if (FileHandler::DirectoryExists(baseFolder) == false) {
		FileHandler::CreateDirectory(baseFolder);
	}

	string fullPath = FileHandler::JoinPath(baseFolder, fileName);
	string matCSV = ImageProcess::GetCSV(matrix);

	std::ofstream out(fullPath);
	out << matCSV;
	out.close();

	// Done
}

void ClassMain::TestJSONSerialization(int argc, char** argv) {
	// Testing JSON serialization

	string jsonString = "{\"argument1\": 1,  \"argument2\": 2}";
	cout << "Performing serialization" << endl;

	auto objJson = json::parse(jsonString);
	cout << objJson.dump() << endl;

	int arg1 = objJson["argument1"];
	int arg2 = objJson["argument2"];


	cout << "Argument 1: " << arg1 << endl;
	cout << "Argument 2: " << arg2 << endl;

	cout << "Done!" << endl;
}

void ClassMain::CheckDoubleToString(int argc, char** argv) {
	cout << "Checking double to string" << endl;

	double number = 0.13567243;
	string numberStr = ImageProcess::DoubleToStr(number);
	cout << "Number String: " << numberStr << endl;
}

void ClassMain::CalibrateCameraLoad(int argc, char** argv) {
	cout << "Calibrating camera load" << endl;

	cout << "Please open the sample file" << endl;

	char file[1024];
	FILE *f = popen("zenity --file-selection", "r");
	fgets(file, 1024, f);
	std::string fileNameStr(file);

	// Bug popen - Must replace ocurrences!
	boost::replace_all(fileNameStr, "\r", "");
	boost::replace_all(fileNameStr, "\n", "");

	if (f == NULL) {
		cerr << "FILE not selected -- Finishing...";
	} else {
		cout << "FileName: " << fileNameStr << endl;

		cout << "Loading image" << fileNameStr << endl;
		Mat image = imread(fileNameStr);

		if (image.data == NULL) {
			cerr << "Error loading image " << fileNameStr << " done!" << endl;
		} else {
			cvNamedWindow("showWindow", CV_WINDOW_AUTOSIZE);
			setMouseCallback("showWindow", ClickEventHandler, NULL);
			imshow("showWindow", image);

			vector<Point2f> image_points;
			vector<Point3f> world_points;

			image_points.push_back(Point2f(148, 227));
			image_points.push_back(Point2f(472, 230));
			image_points.push_back(Point2f(67, 405));
			image_points.push_back(Point2f(562, 409));

			world_points.push_back(Point3f(0, 0, 0));
			world_points.push_back(Point3f(10, 0, 0));
			world_points.push_back(Point3f(0, 10, 0));
			world_points.push_back(Point3f(10, 10, 0));

			cout << "Image Points: " << image_points << endl << endl;
			cout << "World Points: " << world_points << endl << endl;

			// Assuming ideal camera Matrix
			// No need to get intrinsic coefficients
			auto results = ClassCamCalib::CalibrateCamera(image_points, world_points);


			cout << "Camera Matrix: " << results.cameraMatrix << endl;
			cout << "Dist Matrix: " << results.distortion << endl;
			cout << "rvec: " << results.rvec << endl;
			cout << "tvec: " << results.tvec << endl;


			cout << "PNP solved - Click For Testing" << endl;
			int timeoutClickedMs = 100;
			while(true) {
				mouseClicked = false;
				auto keyPressed = cvWaitKey(timeoutClickedMs);

				if (keyPressed == 'q' || keyPressed == 'Q') {
					break;
				} else if (mouseClicked == true){
					mouseClicked = false;
					cout << "Evaluating point" << endl;

					auto pointResult = Project2DTo3D(pointClicked, results.cameraMatrix, results.rvec, results.tvec, results.distortion);
					cout << "Point result: " << pointResult << endl;
				}
			}
		}
	}
}

void ClassMain::TestChessBoardCorners(int argc, char** argv) {
	cout << "Testing chess board corners" << endl;
	cout << "Loading image from file" << endl;

	// Loading image from custom path
	auto image = imread("/home/mauricio/Pictures/solvepnp.jpg", CV_LOAD_IMAGE_COLOR);
	auto imageGray = ImageProcess::Grayscale(image);

	vector<Point2f> imagePoints;
	cv::Size cornersSize(9, 6);			// Must have to count inner corners of the camera

	bool found = findChessboardCorners(image, cornersSize, imagePoints);

	cout << "Result found: " << found << endl;
	cout << "Showing image" << endl;
	cout << "Centers length: " << imagePoints.size() << endl;
	cout << imagePoints << endl;
	cout << "Found: " << found << endl;

	if (found == true) {
		//cornerSubPix(imageGray, imagePoints, cornersSize, Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
		ImageProcess::ShowAndWait(imageGray);
		drawChessboardCorners(imageGray, cornersSize, imagePoints, found);
		ImageProcess::ShowAndWait(imageGray);
	}


	float squareSize = 10.0;
	auto objectPoints = Create3DChessboardCorners(cornersSize, squareSize);

	std::vector<cv::Mat> rotationVectors;
	std::vector<cv::Mat> translationVectors;

	cv::Mat distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
	cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

	int flags = 0;
	cout << "Calibrating camera..." << endl;

	vector<vector<Point2f>> vectorImages;
	vector<vector<Point3f>> vectorObjects;

	vectorImages.push_back(imagePoints);
	vectorObjects.push_back(objectPoints);

	double rms = calibrateCamera(vectorObjects, vectorImages, image.size(), cameraMatrix,
				  distortionCoefficients, rotationVectors, translationVectors, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

	cout << "RMS: " << rms << endl;

	cout << "Camera matrix: " << cameraMatrix << endl;
	cout << "Distortion _coefficients: " << distortionCoefficients << endl;

	cout << "Testing point conversion" << endl;
	cout << "Take first ocurrence" << endl;

	Point2f pointToTest(422.75699, 368.4014);
	auto pointResult = ClassCamCalib::Project2DTo3D(pointToTest, cameraMatrix, rotationVectors[0],
			translationVectors[0], distortionCoefficients);

	cout << "Point projection: " << pointResult << endl;
	cout << "Done!" << endl;
}

std::vector<cv::Point3f> ClassMain::Create3DChessboardCorners(cv::Size boardSize, float squareSize) {
	// This function creates the 3D points of your chessboard in its own coordinate system

	std::vector<cv::Point3f> corners;

	for( int i = 0; i < boardSize.height; i++ ) {
		for( int j = 0; j < boardSize.width; j++ ) {
			corners.push_back(cv::Point3f(float(j*squareSize), float(i*squareSize), 0));
		}
	}

	return corners;
}

void ClassMain::CalibrateCameraCustom(int argc, char ** argv) {
	cout << "Initialize calibrate camera parameters" << endl;
	cout << "Solving all parameters!" << endl;

	std::vector<std::vector<Point2f>> imagePoints;
	std::vector<std::vector<Point3f>> objectPoints;

	std::vector<Point2f> vectorImage;
	vectorImage.push_back(cv::Point2f(100.,100.));
	vectorImage.push_back(cv::Point2f(100.,200.));
	vectorImage.push_back(cv::Point2f(200.,100.));
	vectorImage.push_back(cv::Point2f(200.,200.));
	imagePoints.push_back(vectorImage);

	// Object points are measured in millimeters because calibration is done in mm also
	std::vector<Point3f> vectorObject;
	vectorObject.push_back(cv::Point3f(10,10., 0.));
	vectorObject.push_back(cv::Point3f(10, 20.,0.));
	vectorObject.push_back(cv::Point3f(20.,10.,0.));
	vectorObject.push_back(cv::Point3f(20.,20.,0.));
	objectPoints.push_back(vectorObject);

	cv::Size imageSize(640, 480);
	auto results = ClassCamCalib::CalibrateCameraCalib(vectorImage, vectorObject, imageSize);
	cout << "Camera matrix: " << results.cameraMatrix << endl;
	cout << "Dist coeffs: " << results.distortion << endl;
	cout << "R vector: "  << results.rvec << endl;
	cout << "T vector: " << results.tvec << endl;
}

void ClassMain::ChessboardCalib(int argc, char** argv) {
	double squareSize = 100.0;
	cv::Size imageSize(640, 480);
	cout << "Performing chessboard calibration" << endl;

	auto listFiles = FileHandler::ReadAllFiles("/home/mauricio/Oviedo/Chessboard2");
	bool error = false;

	vector<vector<Point2f>> imagePoints;
	vector<vector<Point3f>> objectPoints;

	for(uint i = 0; i < listFiles.size(); i++) {
		cout << "loading " << listFiles[i] << endl;
		auto image = imread(listFiles[i], CV_LOAD_IMAGE_GRAYSCALE);

		vector<Point2f> points2D;
		cv::Size cornersSize(5, 7);			// Must have to count inner corners of the camera

		bool found = findChessboardCorners(image, cornersSize, points2D);

		if (found == false) {
			cout << "Cant find corners, breaking";
			error = true;
			break;
		} else {
			drawChessboardCorners(image, cornersSize, points2D, found);
			ImageProcess::ShowAndWait(image);
			auto points3D = Create3DChessboardCorners(cornersSize, squareSize);

			imagePoints.push_back(points2D);
			objectPoints.push_back(points3D);
		}
	}

	if (error == false) {
		cout << "Calling camera calib" << endl;

		std::vector<cv::Mat> rotationVectors;
		std::vector<cv::Mat> translationVectors;

		cv::Mat distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
		cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

		int flags = 0;

		double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
					  distortionCoefficients, rotationVectors, translationVectors,flags,
					  cvTermCriteria(CV_TERMCRIT_ITER, 2000, DBL_EPSILON));

		int imIndex = 0;
		int ptIndex = 1;
		cout << "RMS: " << rms << endl;

		cout << "Testing" << endl;
		auto Point3D = ClassCamCalib::Project2DTo3D(imagePoints[imIndex][ptIndex], cameraMatrix, rotationVectors[imIndex],
				translationVectors[imIndex], distortionCoefficients);

		cout << "Point3D: " << Point3D << endl;

		cout << "Testing with solve pnp" << endl;
		auto result = ClassCamCalib::CalibrateCamera(imagePoints[imIndex], objectPoints[imIndex], cameraMatrix, distortionCoefficients);
		Point3D = ClassCamCalib::Project2DTo3D(imagePoints[imIndex][ptIndex], result.cameraMatrix,
				result.rvec, result.tvec, distortionCoefficients);
		cout << "Point3D: " << Point3D << endl;

		cout << "Ground truth Result 3D: " << objectPoints[imIndex][ptIndex] << endl;

		Mat distZeros = Mat::zeros(Size(8, 1), CV_32F);

		cout << "Testing inverse" << endl;
		auto result2D2 = ClassCamCalib::Project3DTo2D(objectPoints[imIndex][ptIndex], cameraMatrix, rotationVectors[imIndex],
					translationVectors[imIndex], distortionCoefficients);
		cout << "Result 2D: " << result2D2 << endl;

		cout << "Ground truth Result 2D: " << imagePoints[imIndex][ptIndex] << endl;
		cout << "Done!" << endl;

	}
}

void ClassMain::FindHomography(int argc, char** argv) {
	// Alternative to use camera calibration
	// Check https://stackoverflow.com/questions/43837455/opencv-for-unity-4-point-calibration-reprojection
	// Check tutorial homography for opencv in python
	// Only need to calculate 4 points to find a homography

	cout << "Init find homography" << endl;

	std::vector<cv::Point2f> imagePoints;
	std::vector<cv::Point2f> objectPoints;

	imagePoints.push_back(cv::Point2f(110.,100.));
	imagePoints.push_back(cv::Point2f(190.,100.));
	imagePoints.push_back(cv::Point2f(100.,200.));
	imagePoints.push_back(cv::Point2f(200.,200.));

	//object points are measured in millimeters because calibration is done in mm also
	objectPoints.push_back(cv::Point2f(500., 500));
	objectPoints.push_back(cv::Point2f(510.,500));
	objectPoints.push_back(cv::Point2f(500.,510));
	objectPoints.push_back(cv::Point2f(510.,510));

	auto matHomography = ClassCamCalib::GetHomography(imagePoints, objectPoints);
	cout << "MatHomography: " << matHomography << endl;

	cout << "Done!" << endl;
	cout << "Calculating projection using matrix" << endl;

	for (uint i = 0; i < imagePoints.size(); i++) {
		cout << "Projecting point" << endl;
		auto projectedPoint = ClassCamCalib::ProjectHomography(imagePoints[i], matHomography);
		cout << "Projected Point: " << projectedPoint << endl;
	}

	cout << "Testing custom value" << endl;
	auto pointCustom = ClassCamCalib::ProjectHomography(Point2f(290, 200), matHomography);
	cout << "Projected Point: " << pointCustom << endl;
}







