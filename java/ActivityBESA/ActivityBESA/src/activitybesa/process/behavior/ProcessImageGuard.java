/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.process.behavior;

import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.GuardBESA;
import BESA.Kernell.Agent.StateBESA;
import activitybesa.ClassJson;
import activitybesa.classdata.ResultsClass;
import activitybesa.process.state.ProcessState;
import activitybesa.reidentification.behavior.IdentifyPersonData;
import activitybesa.reidentification.behavior.IdentifyPersonGuard;
import activitybesa.utils.Point2f;
import activitybesa.utils.Utils;
import java.util.Date;
import java.util.List;
import java.util.logging.Level;

public class ProcessImageGuard extends GuardBESA {
    @Override
    public boolean funcEvalBool(StateBESA objEvalBool) {
        return true;
    }
    
    @Override
    public void funcExecGuard(EventBESA ebesa) {
        AgentBESA ah = this.getAgent();
        ProcessState ps = (ProcessState)ah.getState();
        ProcessImageData data = (ProcessImageData)ebesa.getData();
        Date dateImage = data.image.dateImage;
                
        // Does not require consumer
        // Pose formats are extracted already    
        ClassJson poseResults = data.image.poseResults;
        
        // Must extract pose algorithms
        for (int i = 0; i < poseResults.GetPeopleAmount(); i++) {
            List<ResultsClass> results = poseResults.GetPointsByPerson(i);
            float[] pointsPerson = poseResults.GetPositionPerson(i);
            Point2f pointTransformed = new Point2f(pointsPerson[0], pointsPerson[1]);
            float score = pointsPerson[2];
            
            if (score == 0) {
                // Discard element - Not integrity
            } else {
                // Represents HSV color space
                // Temp information only
                String guid = poseResults.guids[i];
                
                // Transform position to global coordinates
                Utils.logger.log(Level.WARNING, "XValue point {0}", pointTransformed.x);
                Utils.logger.log(Level.WARNING, "YValue point {0}", pointTransformed.y);
                
                // Send data into list
                String agentAlias = "REI";
                IdentifyPersonData dataToSend = new IdentifyPersonData(ps.GetIdCam(), results, dateImage, pointTransformed, guid);
                Utils.SendEventBesa(ah, agentAlias, IdentifyPersonGuard.class, dataToSend);
            }
        }
    }
}
