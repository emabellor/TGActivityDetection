/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hellobesa;

import BESA.ExceptionBESA;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.Agent.PeriodicGuardBESA;
import BESA.Kernell.Agent.StructBESA;
import BESA.Kernell.System.AdmBESA;
import BESA.Kernell.System.Directory.AgHandlerBESA;
import BESA.Util.PeriodicDataBESA;
import hellobesa.Cleaner.Behavior.SensorGuard;
import hellobesa.Cleaner.CleanerAgent;
import hellobesa.Cleaner.State.CleanerState;
import hellobesa.World.Behavior.GameGuard;
import hellobesa.World.Behavior.SubscribeGuard;
import hellobesa.World.Behavior.UpdateGuard;
import hellobesa.World.State.WorldState;
import hellobesa.World.WorldAgent;

/**
 *
 * @author Andres
 */
public class HelloBESA {

    public static int GAME_PERIODIC_TIME = 1000;
    public static int GAME_PERIODIC_DELAY_TIME = 100;
    
    public static void main(String[] args) throws ExceptionBESA {
        
        
        AdmBESA admLocal = AdmBESA.getInstance();
        
        
        WorldState ws = new WorldState(11, 11);
        StructBESA wrlStruct = new StructBESA();
        wrlStruct.addBehavior("WorldBehavior");
        wrlStruct.bindGuard("WorldBehavior", GameGuard.class);
        wrlStruct.addBehavior("ChangeBehavior");
        wrlStruct.bindGuard("ChangeBehavior", SubscribeGuard.class);
        wrlStruct.bindGuard("ChangeBehavior", UpdateGuard.class);
        WorldAgent wa = new WorldAgent("WORLD", ws, wrlStruct, 0.91);
        wa.start();
        
        CleanerState c1State = new CleanerState(11);
        StructBESA c1Struct = new StructBESA();
        c1Struct.addBehavior("playerPerception");
        c1Struct.bindGuard("playerPerception", SensorGuard.class);
        CleanerAgent cleaner1 = new CleanerAgent("C1", c1State, c1Struct, 0.91);
        cleaner1.start();
        
        CleanerState c2State = new CleanerState(11);
        StructBESA c2Struct = new StructBESA();
        c2Struct.addBehavior("playerPerception");
        c2Struct.bindGuard("playerPerception", SensorGuard.class);
        CleanerAgent cleaner2 = new CleanerAgent("C2", c2State, c2Struct, 0.91);
        cleaner2.start();
        

        PeriodicDataBESA data  = new PeriodicDataBESA(GAME_PERIODIC_TIME, GAME_PERIODIC_DELAY_TIME, PeriodicGuardBESA.START_PERIODIC_CALL);
        EventBESA startPeriodicEv = new EventBESA(GameGuard.class.getName(), data);
        AgHandlerBESA ah = admLocal.getHandlerByAlias("WORLD");
        ah.sendEvent(startPeriodicEv);        
    }

}