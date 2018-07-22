/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.world.state;

import BESA.Kernell.Agent.StateBESA;
import activitybesa.classdata.CamRelationClass;
import activitybesa.classdata.FrameModelClass;
import activitybesa.classdata.PersonInfoClass;
import activitybesa.model.JFrameHandler;
import java.awt.Image;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 *
 * @author mauricio
 */
public class WorldState extends StateBESA {
    public List<String> listAgentsCams;
    public JFrameHandler map;
    public Date dateGame;
    public Date startDateGame;
    public Date endDateGame;
    public Map<Integer, CamRelationClass> cameraUIMap;
    public int factor;
    public List<FrameModelClass> listFrames;
    
    
    public WorldState(Date startDate, Date endDate, Map<Integer, CamRelationClass> cameraUIMap, int factor) {
        listAgentsCams = new ArrayList<>();
        
        map = new JFrameHandler();
        this.dateGame = startDate;
        this.startDateGame = startDate;
        this.endDateGame = endDate;
        this.cameraUIMap = cameraUIMap;
        
        map.SetDates(startDate, endDate);
        map.SetCameraUIMap(cameraUIMap);
        
        this.factor = factor;
        map.SetFactorLabel(factor);
        
        listFrames = new ArrayList<>();
    }
    
    public void UpdateImage(Image image, int idCam) {
        boolean found = false;
        
        for (int i = 0; i < listFrames.size(); i++) {
            if (idCam == listFrames.get(i).idCam) {
                found = true;
                listFrames.get(i).image = image;
                break;
            }
        }
        
        if (found == false) {
            // List not found - Adding to list
            FrameModelClass newFrame = new FrameModelClass(image, idCam);
            listFrames.add(newFrame);
        }
    }
}