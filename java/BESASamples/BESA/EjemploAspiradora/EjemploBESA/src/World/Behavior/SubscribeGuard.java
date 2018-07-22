package World.Behavior;

import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.GuardBESA;
import Data.SubscribeData;
import World.State.WorldState;

/**
 *
 * @author Andres
 */
public class SubscribeGuard extends GuardBESA{

    @Override
    public void funcExecGuard(EventBESA ebesa) {
        SubscribeData data = (SubscribeData)ebesa.getData();
        WorldState ws = (WorldState)this.getAgent().getState();
        ws.getBotsAlias().add(data.getAlias());
        ws.getMap().addBot(data.getAlias(), data.getX(), data.getY());
    }
    
}
