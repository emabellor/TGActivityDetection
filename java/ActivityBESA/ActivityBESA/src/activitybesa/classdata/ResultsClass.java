/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.classdata;

/**
 *
 * @author mauricio
 */
public class ResultsClass {
    public int person;
    public int bodyPart;
    public double x;
    public double y;
    public double score; 

    public ResultsClass(int person, int bodyPart, double x, double y, double score) {
        this.person = person;
        this.bodyPart = bodyPart;
        this.x = x;
        this.y = y;
        this.score = score;
    }
}
