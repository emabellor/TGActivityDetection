/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.camera;

import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.Event.DataBESA;
import BESA.Kernell.Agent.KernellAgentExceptionBESA;
import BESA.Kernell.Agent.StateBESA;
import BESA.Kernell.Agent.StructBESA;
import activitybesa.EnumAgents;
import activitybesa.camera.state.CameraState;
import activitybesa.world.behavior.SubscribeData;
import activitybesa.utils.Utils;
import activitybesa.world.behavior.SubscribeGuard;
import java.util.logging.Level;

/**
 *
 * @author mauricio
 */
public class CameraAgent extends AgentBESA {
    public CameraAgent(String alias, StateBESA state, StructBESA structAgent, double passwd)
            throws KernellAgentExceptionBESA {
        super(alias, state, structAgent, passwd);
    }
    
    @Override
    public void setupAgent() {
        Utils.logger.log(Level.INFO, "SETUP AGENT -> {0}", getAlias());
        CameraState cs = (CameraState)getState();

        // Initializing idCam
        int idCam = Utils.GetIdCamFromAlias(getAlias());
        cs.idCam = idCam;

        // Subscribing
        DataBESA data = new SubscribeData(this.getAlias(), EnumAgents.CAM);
        Utils.SendEventBesaWorld(this, SubscribeGuard.class, data); 
    }
    
    
    @Override 
    public void shutdownAgent() {
        Utils.logger.log(Level.INFO, "SHUTDOWN AGENT -> {0}", getAlias());
    }
}
