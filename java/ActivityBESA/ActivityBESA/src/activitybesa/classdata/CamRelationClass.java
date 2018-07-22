/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.classdata;

import java.util.List;

/**
 *
 * @author mauricio
 */
public class CamRelationClass {
    public int idCam;
    public int idCamUI;
    public int[] relationCams;
    
    public CamRelationClass(int idCam, int idCamUI, int[] relationCams)  {
        this.idCam = idCam;
        this.idCamUI = idCamUI;
        this.relationCams = relationCams;
    }
}
