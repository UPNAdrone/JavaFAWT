package connection;

import java.io.IOException;
import java.net.Socket;

/**
 *
 * @author Mintxoo - mintxosola@gmail.com
 */
public class AcceptClient extends Thread{
    
    String ip;
    int port;

    public AcceptClient(String ip, int port) {
        this.ip = ip;
        this.port = port;
    }

    @Override
    public void run(){
        try {
            new Socket(ip, port);
        } catch (IOException ex) {
            System.out.println("Error with the abstract socket");
        }
    }

}
