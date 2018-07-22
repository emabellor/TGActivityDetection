/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.world.behavior;

import BESA.Kernell.Agent.Event.DataBESA;

/**
 *
 * @author mauricio
 */
public class UpdateImageData extends DataBESA {
    public String alias;
    public byte[] image;
    
    public UpdateImageData(String alias, byte[] image) {
        this.alias = alias;
        this.image = image;
    }
}
