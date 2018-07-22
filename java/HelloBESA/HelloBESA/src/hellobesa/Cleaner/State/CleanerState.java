/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hellobesa.Cleaner.State;

import BESA.Kernell.Agent.StateBESA;

/**
 *
 * @author mauricio
 */
public class CleanerState extends StateBESA  {
    int sizeMap;
    int x;
    int y;
    
    public CleanerState(int sizeMap) {
        this.sizeMap = sizeMap;
    }
    
    public int getSizeMap() {
        return sizeMap;
    }

    public void setSizeMap(int sizeMap) {
        this.sizeMap = sizeMap;
    }

    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }
}
