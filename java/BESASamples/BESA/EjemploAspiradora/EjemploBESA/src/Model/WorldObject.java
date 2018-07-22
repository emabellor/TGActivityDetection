package Model;

/**
 *
 * @author Andres
 */
public class WorldObject {
    private int xpos;
    private int ypos;
    private String alias;

    public WorldObject(int xpos, int ypos, String alias) {
        this.xpos = xpos;
        this.ypos = ypos;
        this.alias = alias;
    }

    public String getAlias() {
        return alias;
    }

    public void setAlias(String alias) {
        this.alias = alias;
    }

    public int getXpos() {
        return xpos;
    }

    public void setXpos(int xpos) {
        this.xpos = xpos;
    }

    public int getYpos() {
        return ypos;
    }

    public void setYpos(int ypos) {
        this.ypos = ypos;
    }
    
    
}
