/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hellobesa.World.Behavior;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.GuardBESA;
import hellobesa.Data.SubscribeData;
import hellobesa.World.State.WorldState;

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