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
                for (int i = 1; (i < control.speedMessage.length+1) && (i < info.length); i++) {
                    control.speedMessage[i-1] = Integer.parseInt(info[i]);
                }
                if(realExec){
                    control.server.updateSpeed(control.speedMessage);
                }
                control.updateDrawing();
                Thread.sleep(Integer.parseInt(info[0]));
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
