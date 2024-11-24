package functionality;

import java.io.BufferedReader;
import java.io.FileReader;
import userInterface.ControlFrame;


/**
 *
 * @author mintxo
 */
public class Functionality extends Thread{
    public ControlFrame control;
    public boolean realExec;

    public Functionality(ControlFrame control,boolean realExec) {
        this.control = control;
        this.realExec = realExec;
    }
    
    @Override
    public void run(){
        try (BufferedReader reader = new BufferedReader(new FileReader(control.funFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String info[] = line.split(",");
                if(realExec){
                    for (int i = 1; (i < control.speedMessage.length+1) && (i < info.length); i++) {
                        control.speedMessage[i-1] = Integer.parseInt(info[i]);
                    }
                    control.server.updateSpeed(control.speedMessage);
                    Thread.sleep(Integer.parseInt(info[0]));
                } else{
                    for (int i = 1; (i < control.speedMessagePrev.length+1) && (i < info.length); i++) {
                        control.speedMessagePrev[i-1] = Integer.parseInt(info[i]);
                    }                    
                    Thread.sleep(Integer.parseInt(info[0]));
                }
                control.updateDrawing();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Functionality interrupted");
        }
        
        
        if(!realExec){
            //speedDrawing.dispose();
        }
        
    }
}
