package Data;

import BESA.Kernell.Agent.Event.DataBESA;

/**
 *
 * @author Andres
 */
public class SubscribeData extends DataBESA{
    
    private String alias;
    private int x;
    private int y;

    public SubscribeData(String alias, int x, int y) {
        this.alias = alias;
        this.x = x;
        this.y = y;
    }

    public String getAlias() {
        return alias;
    }

    public void setAlias(String alias) {
        this.alias = alias;
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
