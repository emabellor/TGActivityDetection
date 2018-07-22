/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.world.behavior;

import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.GuardBESA;
import BESA.Kernell.Agent.StateBESA;
import activitybesa.classdata.FrameDescriptorClass;
import activitybesa.classdata.FrameModelClass;
import activitybesa.classdata.PersonInfoClass;
import activitybesa.classdata.ResultsClass;
import activitybesa.utils.LineInfo;
import activitybesa.utils.Utils;
import activitybesa.world.state.WorldState;
import java.awt.Color;
import java.awt.Image;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author mauricio
 * Guard sent from process agent
 */
public class UpdatePeopleGuard extends GuardBESA {
    @Override
    public boolean funcEvalBool(StateBESA objEvalBool) {
        // Draw poses class
        return true;
    }
    
    @Override
    public void funcExecGuard(EventBESA ebesa) {  
        AgentBESA ah = this.getAgent();
        UpdatePeopleData data = (UpdatePeopleData)ebesa.getData();
        WorldState ws = (WorldState)ah.getState();
        
        for (FrameModelClass frame : ws.listFrames) {
            List<PersonInfoClass> listPeople = new ArrayList<>();
            
            for (PersonInfoClass person : data.listPeople) {
                if (person.currentCam == frame.idCam) {
                    listPeople.add(person);
                }
            }
            
            frame.UpdatePersonList(listPeople);
            
            // Draw Image
            Image imgToDraw = frame.GetImage();
            ws.map.SetImage(frame.idCam, imgToDraw); 
        }
        
        // Draw positions in blank rectangle!
        ws.map.SetPositionPoints(data.listPeople);     
    }
}
