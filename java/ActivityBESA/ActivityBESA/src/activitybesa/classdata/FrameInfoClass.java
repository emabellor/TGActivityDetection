/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.classdata;

import activitybesa.ClassJson;
import java.util.Date;

/**
 *
 * @author mauricio
 */
public class FrameInfoClass {
    public byte[] image;
    public Date dateImage;
    public int idCam;
    public ClassJson poseResults;
    
    public FrameInfoClass(byte[] image, Date dateImage, ClassJson poseResults, int idCam) {
        this.image = image;
        this.dateImage = dateImage;
        this.poseResults = poseResults;
        this.idCam = idCam;
    }
}
