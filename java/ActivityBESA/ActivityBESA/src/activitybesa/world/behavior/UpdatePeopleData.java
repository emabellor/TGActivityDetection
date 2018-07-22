/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.world.behavior;

import BESA.Kernell.Agent.Event.DataBESA;
import activitybesa.classdata.PersonInfoClass;
import java.util.List;

/**
 *
 * @author mauricio
 */
public class UpdatePeopleData extends DataBESA {
    public List<PersonInfoClass> listPeople;
    
    public UpdatePeopleData(List<PersonInfoClass> listPeople) {
        this.listPeople = listPeople;
    }
}
