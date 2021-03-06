/*
 * ClassPoseResults.h
 *
 *  Created on: Feb 4, 2018
 *      Author: mauricio
 */

#ifndef CLASSPOSERESULTS_H_
#define CLASSPOSERESULTS_H_

#include <vector>
#include <sys/types.h>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <math.h>

typedef struct {
	double x;
	double y;
} PointF;

typedef struct _StructPoints{
	int person;
	int bodyPart;
	PointF pos;
	double score;

	_StructPoints() {
		this->person = -1;
		this->bodyPart = -1;
		this->pos = PointF();
		this->score = 0;
	}

	_StructPoints(int person, int bodyPart, PointF pos, double score) {
		this->person = person;
		this->bodyPart = bodyPart;
		this->pos = pos;
		this->score = score;
	}
} StructPoints;


enum bodyParts {
	NOSE = 0,
	NECK = 1,
	R_SHOULDER = 2,
	R_ELBOW = 3,
	R_WRIST = 4,
	L_SHOULDER = 5,
	L_ELBOW = 6,
	L_WRITST = 7,
	R_HIP = 8,
	R_KNEE = 9,
	R_ANKLE = 10,
	L_HIP = 11,
	L_KNEE = 12,
	L_ANKLE = 13,
	R_EYE = 14,
	L_EYE = 15,
	R_EAR = 16,
	L_EAR = 17,
	BACKGROUND = 18
};

class ClassPoseResults {
public:
	ClassPoseResults();
	virtual ~ClassPoseResults();

	// Variables
	double factor = 1;

	// Functions
	void AddResult(int person, int bodyPart, int x, int y, double score);
	void AddResult(StructPoints point);
	void ClearResults();
	uint GetPeopleAmount();
	StructPoints GetPose(int person, int bodyPart);
	std::vector<StructPoints> GetAllPoints();
	std::vector<StructPoints> GetPointsByPerson(int person);
	std::vector<StructPoints> GetPointsByPersonNorm(int person);
	std::vector<StructPoints> GetPointsByPersonTranslate(int person);
	static bool SortListCase(const StructPoints first, const StructPoints second);
	double GetEuclideanDistance(PointF point1, PointF point2);

private:
	std::vector<StructPoints> results;
};

#endif /* CLASSPOSERESULTS_H_ */
