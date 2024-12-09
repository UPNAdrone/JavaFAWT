package connection;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import static java.lang.Thread.interrupted;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.StringJoiner;
import java.util.logging.Level;
import java.util.logging.Logger;
import userInterface.ConnectionFrame;

/**
 *
 * @author Mintxoo - mintxosola@gmail.com
 */
public class InterfaceServer extends Thread{
    ConnectionFrame conection;
    public boolean ready = false, connected = false;
    public int port;
    private int num = 0;
    public String ip;
    public List<ClientThread> clients = new LinkedList<>();
    
    public InterfaceServer(int port, ConnectionFrame conection) {
        this.port = port;
        this.conection = conection;
    }

    @Override
    public void run(){
        try {
            Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
            while (interfaces.hasMoreElements()) {
                NetworkInterface networkInterface = interfaces.nextElement();
                if (networkInterface.isLoopback() || !networkInterface.isUp()) {continue;}
                Enumeration<InetAddress> addresses = networkInterface.getInetAddresses();
                while (addresses.hasMoreElements()) {
                    InetAddress inetAddress = addresses.nextElement();
                    if (inetAddress.isSiteLocalAddress()) {
                        if (!inetAddress.getHostAddress().startsWith("127")){
                            ip = inetAddress.getHostAddress();
                        }
                    }
                }
            }
        } catch (SocketException e) {
            e.printStackTrace();
        }
        try( ServerSocket serverSocket = new ServerSocket(port); ){
            System.out.println("El puerto es: "+port);
            conection.setIPandPort(ip, port);
            int i = 0;
            
            while(!interrupted() && !ready){
                Socket clientSocket = serverSocket.accept();
                Thread.sleep(100);
                if(!ready){
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
		String client_name = "Cliente "+num;
		this.setName(client_name);
		
                System.out.println("Connection from " + 
                        socket.getInetAddress() + ":" + socket.getPort()+" ("+client_name+")");
                
                BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                out = new PrintWriter(socket.getOutputStream(), true);

                synchronized (clients) { 
                    clients.add(this);
                }
                out.println(num);
                while(in.readLine() != null){
                    System.out.println(in.readLine());
                }
                
            } catch (IOException ex) {
            } finally { 
                try{ socket.close(); } catch(IOException ex){} 
                synchronized (clients) {
                    clients.remove(this);
                }
            }
        }        
    } 
}
