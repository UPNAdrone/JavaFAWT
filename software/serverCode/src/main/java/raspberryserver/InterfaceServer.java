package raspberryserver;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import static java.lang.Thread.interrupted;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.LinkedList;
import java.util.List;
import java.util.StringJoiner;
import userInterface.ConectionFrame;

/**
 *
 * @author Mintxoo - mintxosola@gmail.com
 */
public class InterfaceServer extends Thread{
    ConectionFrame conection;
    public boolean ready = false, connected = false;
    public int port;
    private int num = 0;
    
    public List<ClientThread> clients = new LinkedList<>();
    
    public InterfaceServer(int port, ConectionFrame conection) {
        this.port = port;
        this.conection = conection;
    }

    @Override
    public void run(){
        try( ServerSocket serverSocket = new ServerSocket(port); ){
            conection.setIPandPort(InetAddress.getLocalHost().getHostAddress(), port);
            int i = 0;
            
            while(!interrupted() && !ready){
                Socket clientSocket = serverSocket.accept();
                Thread.sleep(100);
                if(!ready)
                {
                    ClientThread clientThread = new ClientThread(clients, clientSocket);
                    clientThread.start();
                    i++;
                    while(clients.size() != i){
                        Thread.sleep(10);
                    }
                    conection.addClient("Client "+i+": "+clientSocket.getInetAddress() +"\n");
                }   
            }
            connected = true;
        }catch(Exception ex){
            System.out.println("El cliente canceló la conexión");
        }
    }
    
    public void updateSpeed(int[] speeds){ 
        for (ClientThread c : clients) {
            try {
                StringJoiner joiner = new StringJoiner(",");
                for (int s : speeds) {
                    joiner.add(String.valueOf(s));
                }
                String resultado = joiner.toString();
                c.out.println(resultado);
            } catch (Exception ex) {
                System.out.println("Problema al escribir en el socket");
            }
        }
    }
    
    public class ClientThread extends Thread{
        public List<ClientThread> clients;
        public Socket socket;
        public String name;
        public PrintWriter out;
        public BufferedReader in;

        public ClientThread(List<ClientThread> clients, Socket socket) {
            this.clients = clients;
            this.socket = socket;
        }

        @Override
        public void run() {
            try {
		num++;
		String name = "Cliente "+num;
		this.setName(name);
		
                System.out.println("Connection from " + 
                        socket.getInetAddress() + ":" + socket.getPort()+" ("+name+")");
                
                BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                out = new PrintWriter(socket.getOutputStream(), true);

                //now that we have managed to stablish proper connection, we add ourselve into the list
                synchronized (clients) { //we must sync because other clients may be iterating over it
                    clients.add(this);
                }
                out.println(num);
                while(in.readLine() != null){
                    System.out.println(in.readLine());
                }
                
            } catch (Exception ex) {
                ex.printStackTrace();
            } finally { //we have finished or failed so let's close the socket and remove ourselves from the list
                try{ socket.close(); } catch(Exception ex){} //this will make sure that the socket closes
                synchronized (clients) {
                    clients.remove(this);
                }
            }
        }        
    } 
}
