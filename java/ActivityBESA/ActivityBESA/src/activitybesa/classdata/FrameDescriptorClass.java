/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.classdata;

import java.util.Date;
import java.util.List;

/**
 *
 * @author mauricio
 * One frame descriptor per person!
 */
public class FrameDescriptorClass {
    public List<ResultsClass> results;
    public Date dateImage;
    public int idCam;
 
    public FrameDescriptorClass(List<ResultsClass> results, Date dateImage, int idCam) {
        this.results = results;
        this.dateImage = dateImage;
        this.idCam = idCam;
    }
}
