/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.classdata;

import activitybesa.utils.LineInfo;
import activitybesa.utils.Utils;
import activitybesa.world.state.WorldState;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.List;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

/**
 *
 * @author mauricio
 */
public class FrameModelClass {
    public Image image;
    public int idCam;
    public List<PersonInfoClass> listPeople;
    
    public FrameModelClass(Image image, int idCam) {
        this.image = image;
        this.idCam = idCam;
        listPeople = new ArrayList<>();
    }
    
    public void UpdateImage(Image image) {
        this.image = image;
    }
    
    public void UpdatePersonList(List<PersonInfoClass> listPeople) {
        this.listPeople = listPeople;
    }
    
    public Image GetImage() {
        // Create a buffered image with transparency
        BufferedImage bImage = new BufferedImage(image.getWidth(null), image.getHeight(null),
                BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bImage.createGraphics();
        bGr.drawImage(image, 0, 0, null);
        
        for (int i = 0; i < listPeople.size(); i++) {
            PersonInfoClass person = listPeople.get(i);
            
            FrameDescriptorClass lastFrame = person.frames.get(person.frames.size() - 1);
            DrawPointsFromImage(bGr, lastFrame.results);
            DrawRectangleForPerson(bGr, lastFrame.results, person.guid);      
        }
        
        bGr.dispose();
        return bImage;
    }
    
    
    public void DrawPointsFromImage(Graphics2D bGr, List<ResultsClass> poses) {
        
        List<LineInfo> listPoints = new ArrayList<>();
        for (int i = 1; i <= 13; i++) {
            // Draw 13 lines
            // Exceptions
            ResultsClass elem;
            switch (i) {
                case 2:
                case 5:
                case 8:
                case 11: {
                    // Neck
                    elem = poses.get(1);
                    break;
                }
                default: {
                    // Other elements
                    elem = poses.get(i - 1);
                    break;
                }
            }
            
                     
            ResultsClass elemNext = poses.get(i);
            
            if (elemNext.score < Utils.MIN_POSE_SCORE || elem.score < Utils.MIN_POSE_SCORE) {
                // Ignore
            } else {                    
                Point p1 = new Point((int)elem.x, (int)elem.y);
                Point p2 = new Point((int)elemNext.x, (int)elemNext.y);
                Color color = GetColorFromIndex(i);

                listPoints.add(new LineInfo(p1, p2, color));
            }
        }
        
        Utils.DrawLinesGraphics(bGr, listPoints);
    }
    
    private Color GetColorFromIndex(int index) {
        switch(index) {
            case 1: {
                // Neck - Blue
                return new Color(0, 0, 153);
            }
            case 2: {
                // Right showlder - Wine
                return new Color(153, 0, 0);
            }
            case 3: {
                // 
                return new Color(153, 102, 0);
            }
            case 4: {
                // 
                return new Color(153, 153, 0);
            }
            case 5: {
                // 
                return new Color(153, 51, 0);
            }
            case 6: {
                return new Color(102, 153, 0);
            }
            case 7: {
                return new Color(51, 153, 0);
            }
            case 8: {
                return new Color(0, 153, 0);
            }
            case 9: {
                return new Color(0, 153, 51);
            }
            case 10: {
                return new Color(0, 153, 102);
            }
            case 11: {
                return new Color(0, 153, 153);
            }
            case 12: {
                return new Color(0, 102, 153);
            }
            case 13: {
                return new Color(0, 51, 153);
            }
            default: {
                return new Color(255, 255, 255);
            }
        }
    }
    
    private void DrawRectangleForPerson(Graphics2D graphics, List<ResultsClass> poses, String guid) {
        String[] elems = guid.split("-");
        
        Rectangle rect = Utils.CalculateBoundingRect(poses);
        Utils.DrawRectangleGraphics(graphics, rect, elems[0]);
    }
}
