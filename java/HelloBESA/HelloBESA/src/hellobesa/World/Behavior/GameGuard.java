/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hellobesa.World.Behavior;

import BESA.ExceptionBESA;
import BESA.Kernell.Agent.Event.DataBESA;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.PeriodicGuardBESA;
import BESA.Kernell.System.Directory.AgHandlerBESA;
import BESA.Log.ReportBESA;
import hellobesa.Cleaner.Behavior.SensorGuard;
import hellobesa.Data.SensorData;
import hellobesa.World.State.WorldState;

/**
 *
 * @author mauricio
 */
public class GameGuard extends PeriodicGuardBESA {
    @Override
    public void funcPeriodicExecGuard(EventBESA ebesa) {
        WorldState ws = (WorldState)this.getAgent().getState();
        for (int i = 0; i < ws.getBotsAlias().size(); i++) {
            DataBESA data = new SensorData(ws.getMap().getDust());
            EventBESA event = new EventBESA(SensorGuard.class.getName(), data);
            AgHandlerBESA ah;
            try {
                ah = getAgent().getAdmLocal().getHandlerByAlias(ws.getBotsAlias().get(i));
                ah.sendEvent(event);
            } catch (ExceptionBESA e) {
                ReportBESA.error(e);
            }
        }
    }
    
}
