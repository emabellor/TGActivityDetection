/*
 * FileHandler.h
 *
 *  Created on: Feb 17, 2018
 *      Author: mauricio
 */

#ifndef FILEHANDLER_H_
#define FILEHANDLER_H_

#include <string>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <dirent.h>
#include <fstream>


class FileHandler {
public:
	FileHandler();
	virtual ~FileHandler();

	static bool DirectoryExists(std::string filePath);
	static void CreateDirectory(std::string filePath);
	static std::string JoinPath(std::string path1, std::string path2);
	static std::vector<std::string> ReadAllFiles(std::string dirPath);
	static std::vector<std::string> ReadAllFilesRecursively(std::string dirPath);
	static std::string GetFileNameNoExtension(std::string fileName);
	static std::string GetFileExtension(std::string fileName);
	static std::string ChangeExtension(std::string fileName, std::string extension);   // Ex: .mjpegx
	static bool FileExists(std::string fileName);
};

#endif /* FILEHANDLER_H_ */
