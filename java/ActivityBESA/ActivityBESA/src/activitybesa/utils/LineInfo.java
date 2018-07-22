/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.utils;

import java.awt.Color;
import java.awt.Point;

/**
 *
 * @author mauricio
 */
public class LineInfo {
    public Point pt1;
    public Point pt2;
    public Color color;
    
    public LineInfo(Point pt1, Point pt2, Color color) {
        this.pt1 = pt1;
        this.pt2 = pt2;
        this.color = color;
    }
}
