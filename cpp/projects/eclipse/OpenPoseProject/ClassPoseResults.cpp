/*
 * ClassPoseResults.cpp
 *
 *  Created on: Feb 4, 2018
 *      Author: mauricio
 */

#include "ClassPoseResults.h"

ClassPoseResults::ClassPoseResults() {
	// TODO Auto-generated constructor stub
}

ClassPoseResults::~ClassPoseResults() {
	// TODO Auto-generated destructor stub
}

void ClassPoseResults::AddResult(int person, int bodyPart, int x, int y, double score) {
	StructPoints point;
	point.person = person;
	point.bodyPart = bodyPart;
	point.pos.x = x;
	point.pos.y = y;
	point.score = score;

	results.push_back(point);
}

void ClassPoseResults::AddResult(StructPoints point) {
	AddResult(point.person, point.bodyPart, point.pos.x, point.pos.y, point.score);
}


void ClassPoseResults::ClearResults() {
	results.clear();
}

uint ClassPoseResults::GetPeopleAmount() {
	uint peopleAmount = 0;

	if (results.size() == 0) {
		// Empty
	} else {
		std::vector<int> peopleList;

		for (uint i = 0; i < results.size(); i++) {
			auto item = results[i];

			// Enhance the algorithm hardiness
			if (std::find(peopleList.begin(), peopleList.end(), item.person) == peopleList.end()) { //false
				peopleList.push_back(item.person);
				peopleAmount++;
			}
		}
	}

	return peopleAmount;
}

StructPoints ClassPoseResults::GetPose(int person, int bodyPart) {
	StructPoints point;
	point.person = 0;
	point.bodyPart = 0;

	bool found = false;
	for (uint i = 0; i < results.size(); i++) {
		auto item = results[i];

		if (item.person == person && item.bodyPart == bodyPart) {
			point.pos.x = item.pos.x;
			point.pos.y = item.pos.y;
			point.score = item.score;
			found = true;
			break;
		}
	}

	if (found == false) {
		std::cerr << "Cannot find person " << person << " and bodyPart " << bodyPart << std::endl;
		exit(1);
	}

	return point;
}

std::vector<StructPoints> ClassPoseResults::GetAllPoints() {
	return results;
}

bool ClassPoseResults::SortListCase(StructPoints first, StructPoints second) {
	if (first.person != second.person) {
		return first.person < second.person;
	} else {
		return first.bodyPart < second.bodyPart;
	}
}


std::vector<StructPoints> ClassPoseResults::GetPointsByPerson(int person) {
	std::vector<StructPoints> list;

	for (uint i = 0; i < results.size(); i++) {
		auto item = results.at(i);

		if (item.person == person) {
			list.push_back(item);
		}
	}

	if (list.size() == 0) {
		std::cerr << "Can't find person with index " << person << std::endl;
		exit(1);
	}

	std::sort(list.begin(), list.end(), SortListCase);
	return list;
}


std::vector<StructPoints> ClassPoseResults::GetPointsByPersonNorm(int person) {
	auto list = GetPointsByPerson(person);
	auto newList = std::vector<StructPoints>();

	// Based on the paper
	// An approach to pose-based action recognition
	// Wang et al

	// Euclidean distance - Neck Head
	auto nose = list.at(0);
	auto neck = list.at(1);

	auto distance = GetEuclideanDistance(nose.pos, neck.pos);

	// Normalization
	for (uint i = 0; i < list.size(); i++) {

		auto item = list.at(i);
		StructPoints newItem;
		newItem.bodyPart = item.bodyPart;
		newItem.score = item.score;
		newItem.person = item.person;

		PointF newPoint;
		if (item.pos.x == 0 && item.pos.y == 0) {
			// Default item
			newPoint.x = 0;
			newPoint.y = 0;
		} else {
			// Normalizing
			newPoint.x = (item.pos.x - nose.pos.x) / distance;
			newPoint.y = (item.pos.y - nose.pos.y) / distance;
		}

		newItem.pos = newPoint;

		newList.push_back(newItem);
	}

	factor = distance;
	return newList;
}


std::vector<StructPoints> ClassPoseResults::GetPointsByPersonTranslate(int person) {
	auto list = GetPointsByPerson(person);
	auto newList = std::vector<StructPoints>();

	// Based on the paper
	// An approach to pose-based action recognition
	// Wang et al

	// Euclidean distance - Neck Head
	auto nose = list.at(0);

	// Normalization
	for (uint i = 0; i < list.size(); i++) {

		auto item = list.at(i);
		StructPoints newItem;
		newItem.bodyPart = item.bodyPart;
		newItem.score = item.score;
		newItem.person = item.person;

		PointF newPoint;
		if (item.pos.x == 0 && item.pos.y == 0) {
			// Default item
			newPoint.x = 0;
			newPoint.y = 0;
		} else {
			// Normalizing
			newPoint.x = (item.pos.x - nose.pos.x);
			newPoint.y = (item.pos.y - nose.pos.y);
		}

		newItem.pos = newPoint;

		newList.push_back(newItem);
	}

	return newList;
}

double ClassPoseResults::GetEuclideanDistance(PointF point1, PointF point2) {
	double result = 0;

	result = pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2);
	result = sqrt(result);

	return result;
}


