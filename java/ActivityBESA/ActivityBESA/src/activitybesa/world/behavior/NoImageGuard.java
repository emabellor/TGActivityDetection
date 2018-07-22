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
import activitybesa.utils.Utils;
import activitybesa.world.state.WorldState;
import java.awt.Image;
import java.util.logging.Level;

/**
 *
 * @author mauricio
 */
public class NoImageGuard extends GuardBESA{
    static final Image NOT_FOUND_IMAGE = Utils.LoadImageFromFile("./resources/noimage.jpg");
    
    @Override
    public boolean funcEvalBool(StateBESA objEvalBool) {
        return true;
    }
    
    @Override
    public void funcExecGuard(EventBESA ebesa) {
        WorldState ws = (WorldState)getAgent().getState();
        NoImageData data = (NoImageData)ebesa.getData();
        ws.map.SetImage(data.idCam, NOT_FOUND_IMAGE);
        
        // Done executing guard!
    }
}
