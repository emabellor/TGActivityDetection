/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hellobesa.Data;

import BESA.Kernell.Agent.Event.DataBESA;
import java.util.List;
import hellobesa.Model.WorldObject;
import java.util.ArrayList;

/**
 *
 * @author Andres
 */
public class SensorData extends DataBESA {
    private List<WorldObject> dust;

    public SensorData() {
        dust = new ArrayList<>();
    }

    public SensorData(List<WorldObject> dust) {
        this.dust = dust;
    }

    public List<WorldObject> getDust() {
        return dust;
    }

    public void setDust(List<WorldObject> dust) {
        this.dust = dust;
    }
    
}