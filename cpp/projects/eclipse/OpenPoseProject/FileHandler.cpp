/*
 * FileHandler.cpp
 *
 *  Created on: Feb 17, 2018
 *      Author: mauricio
 */

#include "FileHandler.h"
#include <boost/filesystem.hpp>

FileHandler::FileHandler() {
	// TODO Auto-generated constructor stub

}

FileHandler::~FileHandler() {
	// TODO Auto-generated destructor stub
}

bool FileHandler::DirectoryExists(std::string filePath) {
	// Boost way
	bool result = boost::filesystem::exists(filePath);
	std::cout << "Result: " << result << std::endl;
	return result;
}

void FileHandler::CreateDirectory(std::string filePath) {
	// Boost way
	boost::filesystem::create_directory(filePath);
}

std::string FileHandler::JoinPath(std::string path1, std::string path2) {
	boost::filesystem::path pathBoost1 (path1);
	boost::filesystem::path pathBoost2 (path2);
	boost::filesystem::path full_path = pathBoost1 / pathBoost2;

	return full_path.string();
}

std::vector<std::string> FileHandler::ReadAllFiles(std::string dirPath) {
	std::vector<std::string> response;

	DIR *dir = opendir(dirPath.c_str());
	struct dirent *ent;
	if (dir != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			std::string str(ent->d_name);
			std::string fullPath = JoinPath(dirPath,str);

			struct stat buf;
			stat(fullPath.c_str(), &buf);
			if(S_ISREG(buf.st_mode) == true) {
				// IS FILE!
				response.push_back(fullPath);
			}
		}
		closedir (dir);
	} else {
		// Error directory does not exist, or cant open
	}

	return response;
}

std::vector<std::string> FileHandler::ReadAllFilesRecursively(std::string dirPath) {
	std::vector<std::string> resultList;


	boost::filesystem::path pathBoost(dirPath);
	boost::filesystem::recursive_directory_iterator end_itr;

	for(boost::filesystem::recursive_directory_iterator itr(pathBoost); itr != end_itr; ++itr) {
	    if (boost::filesystem::is_regular_file(itr->path())) {
	        //Do whatever you want
	        resultList.push_back(itr->path().string());
	    }
	}

	return resultList;
}

std::string FileHandler::GetFileNameNoExtension(std::string fileName) {
	boost::filesystem::path p(fileName);
	return p.stem().string();
}

std::string FileHandler::GetFileExtension(std::string fileName) {

	boost::filesystem::path p(fileName);
	std::string extension =  p.extension().string();

	std::cout << "Extension: " << extension << std::endl;

	return extension;
}

std::string FileHandler::ChangeExtension(std::string fileName, std::string extension) {
	std::string result = fileName;
	std::string::size_type i = fileName.rfind('.', fileName.length());

	if (i != std::string::npos) {
		result.replace(i+1, extension.length(), extension);
	}
	return result;
}

bool FileHandler::FileExists(std::string fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}


