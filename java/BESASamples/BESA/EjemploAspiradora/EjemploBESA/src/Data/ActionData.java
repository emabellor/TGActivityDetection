package Data;

import BESA.Kernell.Agent.Event.DataBESA;

/**
 *
 * @author Andres
 */
public class ActionData extends DataBESA {
    
    String alias;
    String action;
    int x;
    int y;

    public ActionData(String alias, String action) {
        this.alias = alias;
        this.action = action;
    }

    public ActionData(String alias, String action, int x, int y) {
        this.alias = alias;
        this.action = action;
        this.x = x;
        this.y = y;
    }

    public String getAction() {
        return action;
    }

    public void setAction(String action) {
        this.action = action;
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

    public String getAlias() {
        return alias;
    }

    public void setAlias(String alias) {
        this.alias = alias;
    }
    
    
}
