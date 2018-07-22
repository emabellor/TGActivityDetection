/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hellobesa.Cleaner;

import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.KernellAgentExceptionBESA;
import BESA.Kernell.Agent.StateBESA;
import BESA.Kernell.Agent.StructBESA;
import BESA.Log.ReportBESA;

/**
 *
 * @author mauricio
 */
public class CleanerAgent extends AgentBESA {
    
    public CleanerAgent(String alias, StateBESA state, StructBESA structAgent, double passwd) 
            throws KernellAgentExceptionBESA{
        super(alias, state, structAgent, passwd);
    }
    
    @Override
    public void setupAgent() {
        ReportBESA.info("Setup Agent -> " + getAlias());
        
     
    }
    
    @Override
    public void shutdownAgent() {
        
    }
}
