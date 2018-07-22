/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package World.Behavior;

import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.GuardBESA;
import BESA.Kernell.Agent.PeriodicGuardBESA;
import Data.ActionData;
import World.State.WorldState;

/**
 *
 * @author Andres
 */
public class UpdateGuard extends GuardBESA{

    
    @Override
    public void funcExecGuard(EventBESA ebesa) {
        ActionData data = (ActionData) ebesa.getData();
        WorldState state = (WorldState)this.getAgent().getState();
        switch (data.getAction()) {
            case "clean":
                state.clean(data.getAlias());
                break;
            case "move":
                state.move(data.getAlias(), data.getX(), data.getY());
                break;
        }
    }
    
}
