package Data;

import BESA.Kernell.Agent.Event.DataBESA;
import Model.WorldObject;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Andres
 */
public class SensorData extends DataBESA{
    private List<WorldObject> dust;

    public SensorData() {
        dust = new ArrayList<>();
    }

    public SensorData(List<WorldObject> dust) {
        this.dust = dust;
    }

    public List<WorldObject> getDust() {
        return dust;
    }

    public void setDust(List<WorldObject> dust) {
        this.dust = dust;
    }
    
}
