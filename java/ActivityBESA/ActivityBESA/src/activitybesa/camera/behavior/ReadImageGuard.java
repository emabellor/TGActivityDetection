/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.camera.behavior;

import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.GuardBESA;
import BESA.Kernell.Agent.StateBESA;
import activitybesa.ClassCamCalib;
import activitybesa.ClassJson;
import activitybesa.camera.state.CameraState;
import activitybesa.classdata.FrameInfoClass;
import activitybesa.classdata.PoseResultsClass;
import activitybesa.process.behavior.ProcessImageData;
import activitybesa.world.behavior.UpdateImageData;
import activitybesa.process.behavior.ProcessImageGuard;
import activitybesa.utils.Utils;
import activitybesa.world.behavior.NoCalibrationData;
import activitybesa.world.behavior.NoCalibrationGuard;
import activitybesa.world.behavior.NoImageData;
import activitybesa.world.behavior.NoImageGuard;
import activitybesa.world.behavior.UpdateImageGuard;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.logging.Level;

/**
 *
 * @author mauricio
 */
public class ReadImageGuard extends GuardBESA {
    @Override
    public boolean funcEvalBool(StateBESA objEvalBool) {
        return true;
    }
    
    @Override
    public void funcExecGuard(EventBESA ebesa) { 
        CameraState cs = (CameraState)this.getAgent().getState();
        AgentBESA ag = this.getAgent();
        
        ReadImageData data = (ReadImageData)ebesa.getData();
    
        long ticks = data.ticks;
        String fullFilePath = Utils.GetVideoPath(cs.videoFolder, ticks, cs.idCam);

        if (fullFilePath.compareTo(cs.currentVideoPath) != 0) {
            // Seeek continuity in video
            FrameInfoClass frameToPush = null;
            if (cs.listFrames.isEmpty() == false) {
                frameToPush = cs.listFrames.get(cs.listFrames.size() - 1);
            }

            Utils.logger.warning("Paths are different - Reloading video");
            cs.currentVideoPath = fullFilePath;
            InitializeVideoMjpegx();

            if (frameToPush != null) {
                cs.listFrames.add(0, frameToPush);
            }    
        }

        FrameInfoClass frameToSend = GetFrameByTicks(ticks);

        if (frameToSend == null) {
            // Not frame to send!
            NoImageData dataToSend = new NoImageData(cs.idCam);
            Utils.SendEventBesaWorld(ag, NoImageGuard.class, dataToSend);
        } else {
            // Sending to world   
            UpdateImageData dataWorld = new UpdateImageData(ag.getAlias(), frameToSend.image);
            Utils.SendEventBesaWorld(ag, UpdateImageGuard.class, dataWorld);

            // Sending to proc agent
            int id = Utils.GetIdCamFromAlias(ag.getAlias());
            String aliasAgent = "PROC_" + id;
            ProcessImageData dataProc = new ProcessImageData(frameToSend);
            Utils.SendEventBesa(ag, aliasAgent, ProcessImageGuard.class, dataProc);
        }
         
    }
    

    private void InitializeVideoMjpegx() {
        Utils.logger.info("Loading video mjpegx");
        CameraState cs = (CameraState)getAgent().getState();
        
        // Clearing list
        cs.listFrames.clear();
        
        if (Utils.FileExists(cs.currentVideoPath) == false) {
            Utils.logger.log(Level.WARNING, "Can''t find video {0}. setting zero size list", cs.currentVideoPath);
            // Let list zero size
        } else {
            List<FrameInfoClass> listFrames = ProcessMjpegxVideo(cs.currentVideoPath);
            Utils.logger.log(Level.FINE, "Number of frames: {0}", listFrames.size());

            if (listFrames.isEmpty()) {
                Utils.logger.severe("List of frames equals to zero. Aborting");
                System.exit(1);
            } else {
                // Loading list of frames!
                cs.listFrames = listFrames;
            }
        }
        
        // Done
    }
    
    private List<FrameInfoClass> ProcessMjpegxVideo(String videoPath) {
        // Score has to be multiplied by 100
        
        CameraState cs = (CameraState)getAgent().getState();
        List<FrameInfoClass> results = new ArrayList<>();
         
        try {     
            InputStream stream = new FileInputStream(videoPath);
            int counter = 0;
            while(true) { 
                // Reading file size
                byte[] bufferInt = new byte[4];
                int bytes = stream.read(bufferInt);
                if (bytes != 4) {
                    if (bytes == -1 || bytes == 0) {
                        // End of file - Must break
                        break;
                    } else {
                        throw new IOException("Error reading bytes: " + counter);
                    }
                }
                int fileSize = Utils.BytesToInt(bufferInt);

                // Reading ticks
                byte[] ticksBin = new byte[8];
                bytes = stream.read(ticksBin);
                if (bytes != 8) {
                    throw new IOException("Error reading ticks");
                }

                // Reading image!
                byte[] frameBin = new byte[fileSize];
                bytes = stream.read(frameBin);
                if (bytes != fileSize) {
                    throw new IOException("Error reading image");
                }
           
                ByteBuffer wrapped = ByteBuffer.wrap(frameBin, frameBin.length - 4, 4); // big-endian by default
                wrapped.order(ByteOrder.LITTLE_ENDIAN);
                int sizeDict = wrapped.getInt();
                        
                // Reading total array
                byte[] dictBin = new byte[sizeDict];
                System.arraycopy(frameBin, frameBin.length - 4 - sizeDict, dictBin, 0, sizeDict);
                
                // Converting to Json Object
                String dict = new String(dictBin);
                ClassJson poseResults = Utils.gson.fromJson(dict, ClassJson.class);
                 
                // This is the image
                Date dateImage = Utils.TicksToDate(ticksBin);
                dateImage = Utils.AddDateHours(dateImage, 5); // Bug correction - Colombian UTC -5
                
                results.add(new FrameInfoClass(frameBin, dateImage, poseResults, cs.idCam));
                counter++;
            }
        } catch (FileNotFoundException ex) {
            Utils.logger.log(Level.SEVERE, "FileNotFoundException thrown: {0}", ex.toString());
            results = new ArrayList<>();
        } catch (IOException ex) {
            Utils.logger.log(Level.SEVERE, "IOException thrown: {0}", ex.toString());
            results = new ArrayList<>();
        }
        
        return results;
    }
    
    public FrameInfoClass GetFrameByTicks(long ticks) {
        // Assumes that array of frames is organized
        // Getting element in list
        CameraState cs = (CameraState)getAgent().getState();
        
        Date date = Utils.TicksToDate(ticks);
        
        FrameInfoClass referenceFrame = null;
        
        double diffMinutes = -1;
        for (int i = 0; i < cs.listFrames.size(); i++) {
            FrameInfoClass currentFrame = cs.listFrames.get(i);
            double frameDiffMinutes = Utils.GetDifferenceInMinutes(currentFrame.dateImage, date);
            
            if (frameDiffMinutes > 0) {
                // Only take account positive difference
                if (diffMinutes == -1) {
                    diffMinutes = frameDiffMinutes;
                    referenceFrame = currentFrame;
                } else {
                    if (frameDiffMinutes < diffMinutes) {
                        diffMinutes = frameDiffMinutes;
                        referenceFrame = currentFrame;
                    }
                }
            }
        }
       
        return referenceFrame;    
    }
}
