 package Cleaner;

import BESA.ExceptionBESA;
import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.Event.DataBESA;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.KernellAgentExceptionBESA;
import BESA.Kernell.Agent.StateBESA;
import BESA.Kernell.Agent.StructBESA;
import BESA.Kernell.System.Directory.AgHandlerBESA;
import BESA.Log.ReportBESA;
import Cleaner.State.CleanerState;
import Data.SubscribeData;
import World.Behavior.SubscribeGuard;
import java.util.Random;

/**
 *
 * @author Andres
 */
public class CleanerAgent extends AgentBESA {

    public CleanerAgent(String alias, StateBESA state, StructBESA structAgent, double passwd) throws KernellAgentExceptionBESA {
        super(alias, state, structAgent, passwd);
    }
    
    @Override
    public void setupAgent() {
        ReportBESA.info("SETUP AGENT -> " + getAlias());
        CleanerState cs = (CleanerState)this.getState();
        Random r = new Random();
        int initialx = r.nextInt(cs.getSizeMap());
        int initialy = r.nextInt(cs.getSizeMap());
        cs.setX(initialx);
        cs.setY(initialy);
        DataBESA data = new SubscribeData(this.getAlias(), initialx, initialy);
        EventBESA event = new EventBESA(SubscribeGuard.class.getName(), data);
        AgHandlerBESA ah;
        
        try {
            ah = this.getAdmLocal().getHandlerByAlias("WORLD");
            ah.sendEvent(event);
        } catch (ExceptionBESA e) {
            ReportBESA.error(e);
        }
    }

    @Override
    public void shutdownAgent() {
        ReportBESA.info("SHUTDOWN AGENT -> " + getAlias());
    }
    
}
