/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.process.state;

import BESA.Kernell.Agent.StateBESA;
import activitybesa.utils.Utils;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.apache.commons.io.FileUtils;

/**
 *
 * @author mauricio
 */
public class ProcessState extends StateBESA {
    private int idCam;      
    
    public ProcessState() {
        idCam = 0;  // IdCam from ProcessState must be initialized in Setup Agent
    }
    
    public int GetIdCam() {
        return idCam;
    }
    
    public void SetIdCam(int idCam) {
        this.idCam = idCam;
    }
}
