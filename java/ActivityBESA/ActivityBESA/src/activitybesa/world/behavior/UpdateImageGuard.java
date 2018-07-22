/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.world.behavior;

import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.GuardBESA;
import BESA.Kernell.Agent.StateBESA;
import activitybesa.classdata.FrameModelClass;
import activitybesa.utils.Utils;
import activitybesa.world.state.WorldState;
import java.awt.Image;
import java.io.IOException;

/**
 *
 * @author mauricio
 */
public class UpdateImageGuard extends GuardBESA {
    @Override
    public boolean funcEvalBool(StateBESA objEvalBool) {
        return true;
    }
    
    @Override
    public void funcExecGuard(EventBESA ebesa) {
        try {     
            // Draw image into model
            UpdateImageData data = (UpdateImageData)ebesa.getData();
            WorldState ws = (WorldState)this.getAgent().getState();

            // Setting image from elements
            // Close
            int idCam = Utils.GetIdCamFromAlias(data.alias);
            Image currentImage = Utils.ByteArrayToImage(data.image);
            boolean found = false;
            for (FrameModelClass model : ws.listFrames){
                if (model.idCam == idCam) {
                    model.UpdateImage(currentImage);

                    // DrawImage
                    Image imgToDraw = model.GetImage();
                    ws.map.SetImage(idCam, imgToDraw);    
                    found = true;

                    break;
                }
            }

            if (found == false) {
                // Adding image to list
                FrameModelClass newFrame = new FrameModelClass(currentImage, idCam);
                ws.listFrames.add(newFrame);

                // Draw Image
                Image imgToDraw = newFrame.GetImage();
                ws.map.SetImage(idCam, imgToDraw);    
            }
        } catch (IOException ex) {
            Utils.logger.severe("IOException thrown. Finishing app");
            Utils.KillProgram();
        }
    }
}
