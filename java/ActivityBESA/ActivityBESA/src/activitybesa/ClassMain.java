package activitybesa;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import BESA.ExceptionBESA;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.PeriodicGuardBESA;
import BESA.Kernell.Agent.StructBESA;
import BESA.Kernell.System.AdmBESA;
import BESA.Kernell.System.Directory.AgHandlerBESA;
import BESA.Util.PeriodicDataBESA;
import activitybesa.classdata.ResultsClass;
import activitybesa.camera.CameraAgent;
import activitybesa.camera.behavior.NewImageGuard;
import activitybesa.camera.behavior.ReadImageGuard;
import activitybesa.camera.state.CameraState;
import activitybesa.classdata.CamRelationClass;
import activitybesa.process.ProcessAgent;
import activitybesa.process.behavior.ProcessImageGuard;
import activitybesa.process.state.ProcessState;
import activitybesa.reidentification.ReidentificationAgent;
import activitybesa.reidentification.behavior.IdentifyPersonGuard;
import activitybesa.reidentification.state.ReidentificationState;
import activitybesa.utils.Utils;
import activitybesa.world.WorldAgent;
import activitybesa.world.behavior.GameGuard;
import activitybesa.world.behavior.NoCalibrationGuard;
import activitybesa.world.behavior.NoImageGuard;
import activitybesa.world.behavior.SubscribeGuard;
import activitybesa.world.behavior.UpdateImageGuard;
import activitybesa.world.behavior.UpdatePeopleGuard;
import activitybesa.world.state.WorldState;
import com.google.gson.Gson;
import java.awt.Point;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.Level;

/**
 *
 * @author mauricio
 */
public class ClassMain {
    /**
     * @param args the command line arguments
     * @throws java.io.IOException
     */
    
    
    public static void main(String[] args) throws IOException, ExceptionBESA {
        Utils.logger.info("Initializing");
        
        // Experiment - Get relative path of application
        File directory = new File("./");
        Utils.logger.info(directory.getAbsolutePath());
      
        Utils.logger.fine("Select option");
        Utils.logger.fine("1: TestJNI");
        Utils.logger.fine("2: Run BESA");
        Utils.logger.fine("3: Test Wrapper");       
        Utils.logger.fine("4: Test JSON reading");
        Utils.logger.fine("5: Calibrate camera");
        Utils.logger.fine("6: Test Calibrate camera");
        Utils.logger.fine("7: Get training images");
        
        int number;
        while (true) {
            Scanner reader = new Scanner(System.in);
            number = reader.nextInt();
            
            if (number < 1 || number > 6)  {
                Utils.logger.warning("Number must be between 1 and 3");
            } else {
                break;
            }
        }
        
        switch (number) {
            case 1: {
                TestJNI();
                break;
            }
            case 2: {
                StartBESA();
                break;
            }
            case 3: {
                TestWrapper();
                break;
            }
            case 4: {
                TestJSONReading();
                break;
            }
            case 5: {
                CalibrateCamera();
                break;
            }
            case 6: {
                TestCalibrateCamera();
                break;
            }
            default: {
                Utils.logger.log(Level.SEVERE, "Invalid number! {0}", number);
                break;
            }
        }
    }
    
    public static void TestJNI() throws IOException {
        Utils.logger.fine("Not Implemented");
    }
    
    public static Map<Integer, CamRelationClass> InitCameraMap() {
        Map<Integer, CamRelationClass> cameraMap = new HashMap<>();
       
        {
            int idCam = 419;
            int idCamUI = 0;
            int[] relationCams = new int[] {428, 420};
            cameraMap.put(idCam, new CamRelationClass(idCam, idCamUI, relationCams));
        }
        {
            int idCam = 420;
            int idCamUI = 1;
            int[] relationCams = new int[] {419, 421, 428, 429, 430};
            cameraMap.put(idCam, new CamRelationClass(idCam, idCamUI, relationCams));
        } 
        {
            int idCam = 421;
            int idCamUI = 2;
            int[] relationCams = new int[] {420, 429, 430};
            cameraMap.put(idCam, new CamRelationClass(idCam, idCamUI, relationCams));
        }
        {
            int idCam = 428;
            int idCamUI = 3;
            int[] relationCams = new int[] {420, 428, 429};
            cameraMap.put(idCam, new CamRelationClass(idCam, idCamUI, relationCams));
        }
        {
            int idCam = 429;
            int idCamUI = 4;
            int[] relationCams = new int[] {419, 421, 428, 420, 430};
            cameraMap.put(idCam, new CamRelationClass(idCam, idCamUI, relationCams));
        }
        {
            int idCam = 430;
            int idCamUI = 5;
            int[] relationCams = new int[] {421, 420, 429};
            cameraMap.put(idCam, new CamRelationClass(idCam, idCamUI, relationCams));
        }
        
        return cameraMap;
    }
    
    public static void StartBESA() throws ExceptionBESA {
        Utils.logger.info("Starting BESA");
     
        // Init game
        Date startDate = Utils.DateFromString("2018-02-24 14:15:00");
        Date endDate = Utils.DateFromString("2018-02-24 15:15:00");
    
        Map<Integer, CamRelationClass> cameraMap = InitCameraMap();
        
        int factor = 1;  // factorX
        InitWorldAgent(startDate, endDate, cameraMap, factor);
        
        InitCameraAgent("CAM_419", "/home/mauricio/Videos/Oviedo");
        InitCameraAgent("CAM_420", "/home/mauricio/Videos/Oviedo");
        InitCameraAgent("CAM_421", "/home/mauricio/Videos/Oviedo");
        InitCameraAgent("CAM_428", "/home/mauricio/Videos/Oviedo");
        InitCameraAgent("CAM_429", "/home/mauricio/Videos/Oviedo");
        InitCameraAgent("CAM_430", "/home/mauricio/Videos/Oviedo");
        
        InitPocessAgent("PROC_419");
        InitPocessAgent("PROC_420");
        InitPocessAgent("PROC_421");
        InitPocessAgent("PROC_428");
        InitPocessAgent("PROC_429");
        InitPocessAgent("PROC_430");
          
        InitReidentificationAgent("REI", cameraMap);
        InitPeriodicGuard(factor);
        
        Utils.logger.info("Environment set!");
    }
    
  
    public static void InitWorldAgent(Date startDate, Date endDate, Map<Integer, CamRelationClass> cameraMap, int factor) throws ExceptionBESA { 
        WorldState wState = new WorldState(startDate, endDate, cameraMap, factor);
        StructBESA wStruct = new StructBESA();
        wStruct.addBehavior("WorldBehavior");
        wStruct.bindGuard("WorldBehavior", GameGuard.class);
        wStruct.bindGuard("WorldBehavior", SubscribeGuard.class);
        wStruct.bindGuard("WorldBehavior", UpdateImageGuard.class);
        wStruct.bindGuard("WorldBehavior", UpdatePeopleGuard.class);
        wStruct.bindGuard("WorldBehavior", NoImageGuard.class);
        wStruct.bindGuard("WorldBehavior", NoCalibrationGuard.class);
        WorldAgent wAgent = new WorldAgent("WORLD", wState, wStruct, 0.91);
        wAgent.start();
    }
    
    public static void InitCameraAgent(String camAlias, String folderPath) throws ExceptionBESA {     
        // Camera alias must be in form of CAM_1 ... CAM_6
        CameraState cState = new CameraState(folderPath);
        StructBESA cStruct = new StructBESA();
        cStruct.addBehavior("CameraBehavior");
        cStruct.bindGuard("CameraBehavior", NewImageGuard.class);
        cStruct.bindGuard("CameraBehavior", ReadImageGuard.class);
        CameraAgent cAgent = new CameraAgent(camAlias, cState, cStruct, 0.91);        
        cAgent.start();
    }
    
    public static void InitPocessAgent(String agentAlias) throws ExceptionBESA {
        ProcessState pState = new ProcessState();
        StructBESA pStruct = new StructBESA();
        pStruct.addBehavior("ProcessBehavior");
        pStruct.bindGuard("ProcessBehavior", ProcessImageGuard.class);
        ProcessAgent pAgent = new ProcessAgent(agentAlias, pState, pStruct, 0.91);
        pAgent.start();
    }
    
    public static void InitReidentificationAgent(String agentAlias, Map<Integer, CamRelationClass> cameraMap) throws ExceptionBESA {
        ReidentificationState rState = new ReidentificationState(cameraMap);
        StructBESA rStruct = new StructBESA();
        rStruct.addBehavior("ReidentificationBehavior");
        rStruct.bindGuard("ReidentificationBehavior", IdentifyPersonGuard.class);
        ReidentificationAgent rAgent = new ReidentificationAgent(agentAlias, rState, rStruct, 0.91);
        rAgent.start();
    }
    
    public static void InitPeriodicGuard(int factor) throws ExceptionBESA{
        AdmBESA admLocal = AdmBESA.getInstance();
        PeriodicDataBESA data  = new PeriodicDataBESA(Utils.GAME_PERIODIC_TIME / factor, Utils.GAME_PERIODIC_DELAY_TIME, 
                PeriodicGuardBESA.START_PERIODIC_CALL);   
        EventBESA startPeriodicEv = new EventBESA(GameGuard.class.getName(), data);
        AgHandlerBESA ah = admLocal.getHandlerByAlias("WORLD");
        ah.sendEvent(startPeriodicEv);
    }
    
    public static void TestWrapper() {
        Utils.logger.info("Not implemented");
    }
    
    public static void TestJSONReading() {
        ClassCamCalib object = new ClassCamCalib();
        
        Point point1 = new Point(0, 0);
        Point point2 = new Point(100, 0);
        
        object.listPointsCalib = new Point[] { point1, point2 };
        Utils.logger.log(Level.INFO, "Total elements: {0}", object.listPointsCalib.length);
        
        Utils.logger.info("Converting to JSON string");
        String result = Utils.gson.toJson(object);
        Utils.logger.log(Level.INFO, "Result: {0}", result);
        
        Utils.logger.info("Converting again to object");
        ClassCamCalib object2 = Utils.gson.fromJson(result, ClassCamCalib.class);
        
        Utils.logger.log(Level.INFO, "Result length: {0}", object2.listPointsCalib.length);
        Utils.logger.info("Termina!");
    }
    
    
    public static void CalibrateCamera() {
        Utils.logger.info("Not implemented");
    }
    
    public static void TestCalibrateCamera() {
        Utils.logger.info("Not implemented");
    }
    
    public static void GetTrainingImages() {
        Utils.logger.info("Getting training images");
        
        
    }
}
