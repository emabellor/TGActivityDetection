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

/**
 *
 * @author mauricio
 */
public class SubscribeGuard extends GuardBESA {
    @Override
    public boolean funcEvalBool(StateBESA objEvalBool) {
        return true;
    }
    
    @Override
    public void funcExecGuard(EventBESA ebesa) {
        SubscribeData data = (SubscribeData)ebesa.getData();
        WorldState ws = (WorldState)this.getAgent().getState();
        
        switch(data.agentType) {
            case CAM: {
                ws.listAgentsCams.add(data.alias);
                break;
            }
            default: {
                Utils.logger.severe("No list for alias");
                break;
            }
        }
        
        // Done
    }
}
