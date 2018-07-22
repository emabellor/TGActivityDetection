/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.camera.state;

import activitybesa.classdata.FrameInfoClass;
import BESA.Kernell.Agent.StateBESA;
import activitybesa.ClassWrapper;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author mauricio
 */
public class CameraState extends StateBESA {
    public String videoFolder;
    public List<FrameInfoClass> listFrames;
    public int idCam;
    public String currentVideoPath;
    
    public CameraState(String videoFolder) {
        this.videoFolder = videoFolder;
        listFrames = new ArrayList<>();
        
        idCam = -1; // IdCam is initialized in setup Agent
        currentVideoPath = ""; // It must be initialized when loader runs!
    }
}
