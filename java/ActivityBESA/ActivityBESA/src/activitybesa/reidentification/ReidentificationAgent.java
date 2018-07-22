/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.reidentification;

import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.KernellAgentExceptionBESA;
import BESA.Kernell.Agent.StateBESA;
import BESA.Kernell.Agent.StructBESA;
import BESA.Log.ReportBESA;
import activitybesa.classdata.IElapsedReceiver;
import activitybesa.classdata.PersonInfoClass;
import activitybesa.reidentification.state.ReidentificationState;
import activitybesa.utils.Utils;
import activitybesa.world.behavior.UpdatePeopleData;
import activitybesa.world.behavior.UpdatePeopleGuard;
import java.util.logging.Level;

/**
 *
 * @author mauricio
 */
public class ReidentificationAgent extends AgentBESA implements IElapsedReceiver {
    public ReidentificationAgent(String alias, StateBESA state, StructBESA structAgent, double passwd)
            throws KernellAgentExceptionBESA {
        super(alias, state, structAgent, passwd);
    }
    
    @Override
    public void setupAgent() {
        Utils.logger.log(Level.INFO, "SETUP AGENT -> {0}", getAlias());
    }
    
    @Override 
    public void shutdownAgent() {
        Utils.logger.log(Level.INFO, "SHUTDOWN AGENT -> {0}", getAlias());
    }
    
    @Override
    public void TimerElapsed(PersonInfoClass person) {
        ReidentificationState rs = (ReidentificationState)getState();
        
        Utils.logger.info("Eliminating person from list");
        rs.listPeople.remove(person);
        
        Utils.logger.info("Sending updated list to World");
        
        // Generate elements to be drawed in BESA
        UpdatePeopleData dataToSend = new UpdatePeopleData(rs.listPeople);         
        Utils.SendEventBesaWorld(this, UpdatePeopleGuard.class, dataToSend);
    }
}
