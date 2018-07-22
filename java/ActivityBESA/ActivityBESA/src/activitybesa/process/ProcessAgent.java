/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.process;

import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.KernellAgentExceptionBESA;
import BESA.Kernell.Agent.StateBESA;
import BESA.Kernell.Agent.StructBESA;
import activitybesa.process.state.ProcessState;
import activitybesa.utils.Utils;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Level;

/**
 *
 * @author mauricio
 */
public class ProcessAgent extends AgentBESA {
    public ProcessAgent(String alias, StateBESA state, StructBESA structAgent, double passwd)
            throws KernellAgentExceptionBESA {
        super(alias, state, structAgent, passwd);
    }
    
    @Override
    public void setupAgent() {
        Utils.logger.log(Level.INFO, "SETUP AGENT -> {0}", getAlias());
        
        ProcessState ps = (ProcessState)getState();
       
        int idCam = Utils.GetIdCamFromAlias(getAlias());
        ps.SetIdCam(idCam);
        
    }
    
    @Override 
    public void shutdownAgent() {
        Utils.logger.log(Level.INFO, "SHUTDOWN AGENT -> {0}", getAlias());
    }
}
