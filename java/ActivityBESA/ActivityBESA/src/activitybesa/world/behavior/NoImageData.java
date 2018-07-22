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
public class NoImageData extends DataBESA {
    public int idCam;
    
    public NoImageData(int idCam) {
        this.idCam = idCam;
    }
}
