/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.reidentification.behavior;

import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.GuardBESA;
import BESA.Kernell.Agent.StateBESA;
import activitybesa.classdata.FrameDescriptorClass;
import activitybesa.classdata.PersonInfoClass;
import activitybesa.reidentification.ReidentificationAgent;
import activitybesa.reidentification.state.ReidentificationState;
import activitybesa.utils.Point2f;
import activitybesa.utils.Utils;
import activitybesa.world.behavior.UpdatePeopleData;
import activitybesa.world.behavior.UpdatePeopleGuard;

/**
 *
 * @author mauricio
 */
public class IdentifyPersonGuard extends GuardBESA {
    @Override
    public boolean funcEvalBool(StateBESA objEvalBool) {
        return true;
    }
    
    @Override
    public void funcExecGuard(EventBESA ebesa) {
        IdentifyPersonData data = (IdentifyPersonData)ebesa.getData();
        ReidentificationAgent ah = (ReidentificationAgent)this.getAgent();
        ReidentificationState rs = (ReidentificationState)ah.getState();

        // Adding frame descriptor class
        FrameDescriptorClass frame = new FrameDescriptorClass(data.results, data.dateImage, data.idCam);
        int idCurrentCam = frame.idCam;  

        Point2f currentPosition = data.position;
             
        boolean personFound = false;
        for (int i = 0; i < rs.listPeople.size(); i++) {
            PersonInfoClass person = rs.listPeople.get(i);
            
            // Try to locate in current list
            if (person.guid.equals(data.guid)) {
                person.AddFrameDescriptor(frame, currentPosition, idCurrentCam);
                personFound = true;
                break;
            }
        }
        
        if (personFound == false) {
            Utils.logger.fine("Person not found - Creating person in list");
            PersonInfoClass newPerson = new PersonInfoClass(data.guid, ah);

            // Color generated!
            newPerson.AddFrameDescriptor(frame, currentPosition, idCurrentCam);
            rs.listPeople.add(newPerson);
        }
       
        // Generate elements to be drawed in BESA
        UpdatePeopleData dataToSend = new UpdatePeopleData(rs.listPeople);         
        Utils.SendEventBesaWorld(ah, UpdatePeopleGuard.class, dataToSend);
    }
}
