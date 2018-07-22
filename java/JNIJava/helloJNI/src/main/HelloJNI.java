/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import java.util.ArrayList;

/**
 *
 * @author mauricio
 */
public class HelloJNI {
    static {
        // Code executes when created class
        System.loadLibrary("HelloJNI");
    } 
    
    public native void SayHello();
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        System.out.println("Inicio de la app");
        HelloJNI instance = new HelloJNI();
        instance.SayHello();
    }
}
