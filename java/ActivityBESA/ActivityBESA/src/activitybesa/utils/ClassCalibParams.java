/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.utils;

import java.util.List;

/**
 *
 * @author mauricio
 */
class ClassCalibParams {
    public List<Point2f> imagePoints;
    public List<Point2f> objectPoints;
    
    public ClassCalibParams(List<Point2f> imagePoints, List<Point2f> objectPoints) {
        this.imagePoints = imagePoints;
        this.objectPoints = objectPoints;
    }
}
