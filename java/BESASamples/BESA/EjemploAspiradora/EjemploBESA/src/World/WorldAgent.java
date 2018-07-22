package World;

import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.KernellAgentExceptionBESA;
import BESA.Kernell.Agent.StateBESA;
import BESA.Kernell.Agent.StructBESA;
import BESA.Log.ReportBESA;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Andres
 */
public class WorldAgent extends AgentBESA{

    public WorldAgent(String alias, StateBESA state, StructBESA structAgent, double passwd) throws KernellAgentExceptionBESA {
        super(alias, state, structAgent, passwd);
    }
    
    @Override
    public void setupAgent() {
        ReportBESA.info("SETUP AGENT -> " + getAlias());
    }

    @Override
    public void shutdownAgent() {
        ReportBESA.info("SHUTDOWN AGENT -> " + getAlias());
//        try {
//            killBehaviors(0.91);
//        } catch (KernellAgentExceptionBESA ex) {
//            Logger.getLogger(WorldAgent.class.getName()).log(Level.SEVERE, null, ex);
//        }
    }
    
}
