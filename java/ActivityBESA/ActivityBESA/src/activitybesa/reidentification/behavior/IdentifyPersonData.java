/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.reidentification.behavior;

import activitybesa.classdata.ResultsClass;
import BESA.Kernell.Agent.Event.DataBESA;
import activitybesa.utils.Point2f;
import java.util.Date;
import java.util.List;

/**
 *
 * @author mauricio
 */
public class IdentifyPersonData extends DataBESA {
    public int idCam;
    public Date dateImage;
    public List<ResultsClass> results;
    public Point2f position;
    public String guid;
    
    public IdentifyPersonData(int idCam, List<ResultsClass> results, Date dateImage, Point2f position, String guid) {
        this.idCam = idCam;
        this.results = results;
        this.dateImage = dateImage;
        this.position = position;
        this.guid = guid;
    }
}
