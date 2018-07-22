/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa;

import activitybesa.classdata.ResultsClass;
import activitybesa.utils.Point2f;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author mauricio
 */
public class ClassJson {
    public float[][][] vectors;
    public float[][] positions;
    public String[] guids;
    
    public int GetPeopleAmount() {
        return vectors.length;
    }
    
    public List<ResultsClass> GetPointsByPerson(int person) {
        List<ResultsClass> list = new ArrayList<>();
        
        float[][] personVector = vectors[person];
        // Assume 18 positions
        for (int i = 0; i < 18; i++) {
            float[] posVector = personVector[i];
            
            ResultsClass elem = new ResultsClass(person, i, 
                    posVector[0], posVector[1], posVector[2]);
            list.add(elem);
        }
        
        return list;
    }
    
    public float[] GetPositionPerson(int person){
        return positions[person];     
    }
}
