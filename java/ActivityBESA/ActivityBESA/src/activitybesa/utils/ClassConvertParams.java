/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.utils;

/**
 *
 * @author mauricio
 */
public class ClassConvertParams {
    public Point2f point;
    public String homographyMat;
    
    public ClassConvertParams(Point2f point, String homographyMat) {
        this.point = point;
        this.homographyMat = homographyMat;
    }
}
