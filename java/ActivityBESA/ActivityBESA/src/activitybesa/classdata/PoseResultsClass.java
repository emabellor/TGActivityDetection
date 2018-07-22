/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.classdata;

import activitybesa.utils.Utils;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;

/**
 *
 * @author mauricio
 */
public class PoseResultsClass {
    List<ResultsClass> listPeopleInfo;

    public PoseResultsClass() {
        listPeopleInfo = new ArrayList<>();
    }
   
    public PoseResultsClass(ResultsClass[] results) {
        listPeopleInfo = Arrays.asList(results);
    }
    
    public void AddPersonInfo(int person, int bodyPart, int x, int y, int score) {
        ResultsClass personInfo = new ResultsClass(person, bodyPart, x, y, score);
        listPeopleInfo.add(personInfo);
    }
    
    public int GetPeopleAmount() {
	int peopleAmount = 0;

	if (listPeopleInfo.isEmpty()) {
            // Empty
	} else {
            List<Integer> peopleList = new ArrayList<>();

            for (int i = 0; i < listPeopleInfo.size(); i++) {
                ResultsClass info = listPeopleInfo.get(i);
                
                if (peopleList.contains(info.person) == false) {
                    peopleList.add(info.person);
                    peopleAmount++;
                }
            }
	}

	return peopleAmount;
    }
    
    private int Compare(ResultsClass lhs, ResultsClass rhs) {
        // -1 - less than, 1 - greater than, 0 - equal, all inversed for descending
        if (lhs.bodyPart > rhs.bodyPart) {
            return 1;
        } else if (lhs.bodyPart > rhs.bodyPart) {
            return -1;
        } else {
            return 0;
        }
    }
    
    public List<ResultsClass> GetPointsByPerson(int person) {
	List<ResultsClass> list = new ArrayList<>();

	for (int i = 0; i < listPeopleInfo.size(); i++) {
            ResultsClass item = listPeopleInfo.get(i);

            if (item.person == person) {
                list.add(item);
            }
	}

	if (list.isEmpty()) {
            System.out.println("Can't find person with index " + person);
            System.exit(1);
	}

        // Sort by bodypart
        Collections.sort(list, (a, b) -> Compare(a, b));
	return list;
    }
}
