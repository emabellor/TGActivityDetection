/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.world.behavior;

import BESA.ExceptionBESA;
import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.PeriodicGuardBESA;
import activitybesa.camera.behavior.ReadImageData;
import activitybesa.camera.behavior.ReadImageGuard;
import activitybesa.utils.Utils;
import activitybesa.world.state.WorldState;
import java.util.logging.Level;

/**
 *
 * @author mauricio
 */
public class GameGuard extends PeriodicGuardBESA {
    
    // Periodic Execution
    @Override
    public void funcPeriodicExecGuard(EventBESA ebesa) {
        WorldState ws = (WorldState)this.getAgent().getState();
        AgentBESA ag = this.getAgent();
        
        ws.dateGame = Utils.AddDateMs(ws.dateGame, Utils.GAME_PERIODIC_TIME);
 
        
        double diffMs = Utils.GetDifferenceInMs(ws.dateGame, ws.endDateGame);
        
        if (diffMs < 0) {
            Utils.logger.severe("dateGame is greater than endDateGame - Killing Program");
            Utils.KillProgram();
        }
    
        // Continue with the game!
        ws.map.UpdateDate(ws.dateGame);
        

        for (int i = 0; i < ws.listAgentsCams.size(); i++) {
            String alias = ws.listAgentsCams.get(i);
            
            // Send event to read Image
            long ticks = Utils.DateToTicks(ws.dateGame);
            ReadImageData data = new ReadImageData(ticks);
            Utils.SendEventBesa(ag, alias, ReadImageGuard.class, data);
        }
        
        // Done evaluating everything
    }
}
