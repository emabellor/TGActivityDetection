/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.reidentification.state;

import BESA.Kernell.Agent.StateBESA;
import activitybesa.classdata.CamRelationClass;
import activitybesa.classdata.PersonInfoClass;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 *
 * @author mauricio
 */
public class ReidentificationState extends StateBESA {
    public List<PersonInfoClass> listPeople;
    public Map<Integer, CamRelationClass> cameraMap;
    
    public ReidentificationState(Map<Integer, CamRelationClass> cameraMap) {
        // Initializing in constructor
        this.listPeople = new ArrayList<>();
        this.cameraMap = cameraMap;
    }
}
