/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.camera.behavior;

import BESA.Kernell.Agent.Event.DataBESA;

/**
 *
 * @author mauricio
 */
public class ReadImageData extends DataBESA {
    public long ticks;
    
    public ReadImageData(long ticks) {
        this.ticks = ticks;
    }
}
