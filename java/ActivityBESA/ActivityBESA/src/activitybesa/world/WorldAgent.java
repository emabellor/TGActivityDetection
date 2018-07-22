/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.world;

import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.KernellAgentExceptionBESA;
import BESA.Kernell.Agent.StateBESA;
import BESA.Kernell.Agent.StructBESA;
import activitybesa.classdata.FrameModelClass;
import activitybesa.utils.Utils;
import activitybesa.world.state.WorldState;
import java.util.Date;
import activitybesa.model.IFrameEvents;

/**
 *
 * @author mauricio
 */
public class WorldAgent extends AgentBESA implements IFrameEvents {
    public WorldAgent(String alias, StateBESA state, StructBESA structAgent, double passwd)
            throws KernellAgentExceptionBESA {
        super(alias, state, structAgent, passwd);
    }
    
    @Override
    public void setupAgent() {
        Utils.logger.info("Starting agent world");
        
        Utils.logger.info("Setting UI");
        WorldState ws = (WorldState)getState();
        ws.map.setVisible(true);
        
        // Subscribing for events in map
        ws.map.AddListener(this);
    }
    
    @Override 
    public void shutdownAgent() {
        Utils.logger.info("Shutting down agent world");
    }
    
    @Override
    public void ChangeDate(Date newDate) {
        WorldState ws = (WorldState)getState();
        Utils.logger.info("Changing date for cameras");
        
        ws.dateGame = newDate;
        ws.map.UpdateDate(ws.dateGame);
    }
    
    public void DrawGUI() {
        WorldState ws = (WorldState)getState();
         
        for (int i = 0; i < ws.listFrames.size(); i++) {
            FrameModelClass frame = ws.listFrames.get(i);
            
            
        } 
    }
}