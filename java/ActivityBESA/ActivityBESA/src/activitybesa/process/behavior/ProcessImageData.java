/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.process.behavior;

import activitybesa.classdata.FrameInfoClass;
import BESA.Kernell.Agent.Event.DataBESA;

/**
 *
 * @author mauricio
 */
public class ProcessImageData extends DataBESA {
    public FrameInfoClass image;
    
    public ProcessImageData(FrameInfoClass image) {
        this.image = image;
    }
}
