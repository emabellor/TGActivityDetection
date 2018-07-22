/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.world.behavior;

import BESA.Kernell.Agent.Event.DataBESA;
import activitybesa.EnumAgents;

/**
 *
 * @author mauricio
 */
public class SubscribeData extends DataBESA { 
    public String alias;
    public EnumAgents agentType;
    
    public SubscribeData(String alias, EnumAgents agentType) {
        this.alias = alias;
        this.agentType = agentType;
    }
}
