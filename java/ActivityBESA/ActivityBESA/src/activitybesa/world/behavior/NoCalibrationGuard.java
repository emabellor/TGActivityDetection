/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.world.behavior;

import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.GuardBESA;
import BESA.Kernell.Agent.StateBESA;
import activitybesa.utils.Utils;
import activitybesa.world.state.WorldState;
import java.awt.Image;

/**
 *
 * @author mauricio
 */
public class NoCalibrationGuard extends GuardBESA {
    static final Image NOT_CALIBRATED_IMAGE = Utils.LoadImageFromFile("./resources/notCalibrated.jpg");
    
    @Override
    public boolean funcEvalBool(StateBESA objEvalBool) {
        return true;
    }
    
    @Override
    public void funcExecGuard(EventBESA ebesa) {
        WorldState ws = (WorldState)getAgent().getState();
        NoCalibrationData data = (NoCalibrationData)ebesa.getData();
        ws.map.SetImage(data.idCam, NOT_CALIBRATED_IMAGE);
        
        // Done executing guard!
    }
}
