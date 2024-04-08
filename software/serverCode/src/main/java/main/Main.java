package main;

import javax.swing.JFrame;
import raspberryserver.InterfaceServer;
import representation.SpeedDrawing;
import userInterface.ConectionFrame;
import userInterface.ControlFrame;

/**
 *
 * @author Mintxoo - mintxosola@gmail.com
 */
public class Main {
    public static void main(String args[]) throws InterruptedException {      
        ConectionFrame conection = new ConectionFrame();
        InterfaceServer server = new InterfaceServer(125,conection); // default port is 125
        server.start();
        
        conection.setTitle("FAN CONECTION");
        conection.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        conection.setResizable(false);
        conection.setLocationRelativeTo(null);
        conection.setVisible(true);
        
        while (!conection.ready){
            Thread.sleep(100); // it is necessary
        }
        server.ready = true;
        while (!server.conected){
            Thread.sleep(100); // it is necessary
        }
        
        conection.dispose(); 
        
        ControlFrame control = new ControlFrame(server);
        SpeedDrawing speedDrawing = new SpeedDrawing(control);
        
        control.setSpeedInstance(speedDrawing);
        control.setTitle("FAN CONTROL");
        control.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        control.setResizable(false);
        control.setLocation(800, 100);
        
        speedDrawing.setVisible(true);
        speedDrawing.setTitle("SPEED SCHEMA");
        speedDrawing.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        speedDrawing.setResizable(false);
        speedDrawing.setLocation(200, 100);
        
        Thread controlThread = new Thread(() -> {
            control.setVisible(true); 
            try { 
                control.run();
            } catch (InterruptedException ex) {
                System.out.println("Error on control");
            }
            control.dispose();  
        });
        
        Thread speedThread = new Thread(() -> {
            speedDrawing.setVisible(true); 
            try {
                speedDrawing.run();
                speedDrawing.dispose();
            } catch (Exception ex) {
                System.out.println("Error on the SpeedDrawing class");
            }
        });
        
        controlThread.start();
        speedThread.start();
        
        controlThread.join();
        speedThread.join();
    }
}
