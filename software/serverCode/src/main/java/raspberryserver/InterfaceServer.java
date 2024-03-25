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
import java.util.Scanner;
import java.util.StringJoiner;
import userInterface.ConectionFrame;

public class InterfaceServer extends Thread{
    ConectionFrame conection;
    public boolean ready = false, conected = false;
    public int port;
    private int num = 0;
    
    public List<ClientThread> clients = new LinkedList<>();
    
    public InterfaceServer(int port, ConectionFrame conection) {
        this.port = port;
        this.conection = conection;
    }

    @Override
    public void run(){
        Scanner scanner = new Scanner(System.in);
        try( ServerSocket serverSocket = new ServerSocket(port); ){
            conection.setIPandPort(InetAddress.getLocalHost().getHostAddress(), port);
            int i = 0;
            
            while(!interrupted() && !ready){
                Socket clientSocket = serverSocket.accept();
                Thread.sleep(100);
                if(!ready) // voy a conectar yo a un cliente para salir de aqui sin excepcion cuando pulse rady
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
            
            /*cliente 1 valores de velocidad pos 0 - pos 5
            cliente 2 valores de pos 6 - pos 11
            ...*/
            conected = true;
            
        }catch(Exception ex){
            System.out.println("El cliente canceló la conexión");
        }
    }
    
    public void updateSpeed(int[] speeds){ 
    // ahora mismo les envio el mensaje con todas las velocidades a todos
    // pero podría enviar a cada cliente el mensaje con sus velocidades
        
        // tengo que obtener los sockets de los clientes a los que quiero enviar el mensaje
        for (ClientThread c : clients) {
            try {
                /*
                // con 7 bits puede representar la velocidad de cada ventilador
                // lo primero es transformar el array de velocidades a una larga cadena de bits
                 // Mensaje en bits que se enviará (como ejemplo, un byte con valor 10101010)
                //byte message = (byte) Integer.parseInt("10101010", 2);
                byte[] bytes;
                byte byteResultante;
                
                for (int i = 0; i < speeds.length; i++) {
                     bytes = ByteBuffer.allocate(4).putInt(speeds[i]).array();
                     byteResultante = bytes[3];
                     System.out.println("i: "+i+" - "+byteResultante);
                }
                // Enviar el mensaje
                //c.out.writeByte(message);
                //c.out.flush();
                //System.out.println("Mensaje en bits enviado correctamente");
                //String s = 
                */
                
                StringJoiner joiner = new StringJoiner(",");
                for (int s : speeds) {
                    joiner.add(String.valueOf(s));
                }

                String resultado = joiner.toString();
                //System.out.println("res: "+resultado);
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

        //only one thread at the time can send messages through the socket
        /*synchronized public void sendMsg(String msg){
            Scanner scanner = new Scanner(System.in);
            try {
                System.out.print("Escribe un mensaje para el cliente: ");
                msg = scanner.nextLine();
                out.println(msg);
            } catch (Exception ex) {
                System.out.println("Problema al escribir en el socket");
            }
        }*/

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
