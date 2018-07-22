/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.classdata;

import activitybesa.utils.Point2f;
import java.awt.Color;
import java.awt.Point;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

/**
 *
 * @author mauricio
 */
public class PersonInfoClass {
    public List<FrameDescriptorClass> frames;
            
    // Position is like center!
    public Point2f position;
    
    // Unique identifier setted by the partner
    public String guid;
    
    // Set position of cam
    public int currentCam;
    
    // Set timer for cam
    public Timer timer;

    // Receiver for elapsed event
    public IElapsedReceiver receiver;
    
    public PersonInfoClass(String guid, IElapsedReceiver receiver) {
        // Guid is the only element that must be added in list
        this.guid = guid;
        
        frames = new ArrayList<>();
        
        // Initializing position
        position = new Point2f(0, 0); 
        
        // Initializing current cam
        this.currentCam = 0;
        
        // Setting timer
        timer = null;

        // Save reference for elapsed receiver
        this.receiver = receiver;
    }
    
    public void AddFrameDescriptor(FrameDescriptorClass frame, Point2f position, int currentCam) {
        // Generating position frame!
        this.position = position;
        this.currentCam = currentCam;
        
        frames.add(frame);
        
        if (timer != null) {
            timer.cancel();
        }
        
        timer = new Timer();
        int timeoutMs = 5 * 1000;
       
        PersonInfoClass reference = this;
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                receiver.TimerElapsed(reference);
            }
        }, timeoutMs);
    }
}
