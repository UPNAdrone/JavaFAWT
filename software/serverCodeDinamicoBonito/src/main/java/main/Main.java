package main;

import javax.swing.JFrame;
import connection.InterfaceServer;
import userInterface.ConectionFrame;
import userInterface.ControlFrame;
import userInterface.PortFrame;

/**
 *
 * @author Mintxoo - mintxosola@gmail.com
 */
public class Main {
    public static void main(String args[]) throws InterruptedException {      
        PortFrame portFrame = new PortFrame();
        portFrame.setTitle("PORT SELECTION");
        portFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        portFrame.setResizable(false);
        portFrame.setLocationRelativeTo(null);
        portFrame.setVisible(true);
        while (!portFrame.ready){
            Thread.sleep(100); // it is necessary
        }
        
        ConectionFrame connection = new ConectionFrame();
        InterfaceServer server = new InterfaceServer(portFrame.port,connection);
        portFrame.dispose(); 
        server.start();
        
        connection.setTitle("FAN CONECTION");
        connection.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        connection.setResizable(false);
        connection.setLocationRelativeTo(null);
        connection.setVisible(true);
        
        while (!connection.ready){
            Thread.sleep(100); // it is necessary
        }
        server.ready = true;
        while (!server.connected){
            Thread.sleep(100); // it is necessary
        }
        
        connection.dispose(); 
        
        ControlFrame control = new ControlFrame(server, connection.rows, connection.cols);
        control.setTitle("FAN CONTROL");
        control.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        control.setResizable(false);
        control.setLocation(0, 0);
        control.setSize(1000, 770);
        
        Thread controlThread = new Thread(() -> {
            control.setVisible(true); 
            try { 
                control.run();
            } catch (InterruptedException ex) {
                System.out.println("Error on control");
            }
            control.dispose();  
        });
        
        controlThread.start();
        controlThread.join();
    }
}
