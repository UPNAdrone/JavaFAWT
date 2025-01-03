package representation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import javax.swing.JButton;
import javax.swing.JLabel;
import userInterface.ControlFrame;

/**
 *
 * @author Mintxoo - mintxosola@gmail.com
 */
public class PressureSensor extends javax.swing.JFrame {
    public ControlFrame control;
    InputStream in;
    String directoryPath = "../../preassureFiles";
    public File preassureFile = null;
    
    public float full_scale_psi = (float) 0.1445092;
    public float full_scale_pa = (float) (full_scale_psi * 6894.76);
            
    public PressureSensor(ControlFrame c) {
        initComponents();
        this.control = c;
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPopupMenu1 = new javax.swing.JPopupMenu();
        jPopupMenu2 = new javax.swing.JPopupMenu();
        jCheckBoxMenuItem1 = new javax.swing.JCheckBoxMenuItem();
        popupMenu1 = new java.awt.PopupMenu();
        jDialog1 = new javax.swing.JDialog();
        popupMenu2 = new java.awt.PopupMenu();
        jLabel1 = new javax.swing.JLabel();

        jCheckBoxMenuItem1.setSelected(true);
        jCheckBoxMenuItem1.setText("jCheckBoxMenuItem1");

        popupMenu1.setLabel("popupMenu1");

        javax.swing.GroupLayout jDialog1Layout = new javax.swing.GroupLayout(jDialog1.getContentPane());
        jDialog1.getContentPane().setLayout(jDialog1Layout);
        jDialog1Layout.setHorizontalGroup(
            jDialog1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 400, Short.MAX_VALUE)
        );
        jDialog1Layout.setVerticalGroup(
            jDialog1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 300, Short.MAX_VALUE)
        );

        popupMenu2.setLabel("popupMenu2");

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jLabel1.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        jLabel1.setText("Pressure Sensor");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(30, 30, 30)
                .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 151, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(55, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 36, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(28, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    /**
     *
     * @throws InterruptedException
     * @throws IOException
     */
    public void run() throws InterruptedException, IOException {
        connectPressureServer();
        byte[] packet = new byte[75]; // Buffer para almacenar el paquete recibido
        
        createIncrementalFile(directoryPath);
        int x = 150,y=30,w=100,h=20;
        JLabel time = new JLabel("Timestap:");
        time.setBounds(20, 55, w, h);
        this.add(time);
        ArrayList<JLabel> sensors = new ArrayList<>();
        int count = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                JLabel l = new JLabel("Sensor "+count+":");
                l.setBounds(20+i*x, 85+j*y, w, h);
                sensors.add(l);
                this.add(l);
                count++;
            }
        }

        while (true) {
            int bytesRead = in.read(packet, 0, packet.length);
            if (bytesRead == -1) { break; }
            try {
                PacketData parsedData = parsePacket(packet);
                String t = parsedData.timestamp.toString();
                time.setText("Timestap: "+t);
                System.out.printf(t+"\n");
                writeLine(t+"\n");
                String s = "";
                for (int i = 0; i < parsedData.pressures.length; i++) {
                    sensors.get(i).setText("Sensor "+count+": "+parsedData.pressures[i]);
                    s = s.concat(", "+parsedData.pressures[i]);
                    System.out.printf("%.2f, ", parsedData.pressures[i]);
                }
                writeLine(s);
                System.out.println("\n\n--------------------\n\n");
                
                Thread.sleep(10);
            } catch (Exception e) {
                System.err.println("Error parsing packet: " + e.getMessage());
            }
        }
    }

    public void connectPressureServer() throws InterruptedException, IOException {
        if (control.pressureSensorIP == null || control.pressureSensorPort == -1 || !control.pressureSensorConected) {
            NewPSensor ps = new NewPSensor(control);
            ps.run();
            ps.dispose();
        }
        try {
            System.out.println("----Buscando Conexión----");
            Socket socket = new Socket(control.pressureSensorIP, control.pressureSensorPort);
            in = socket.getInputStream();
            System.out.println("----Conectado correctamente----");
            System.out.println("Se conecto al socket "+socket);
            System.err.println("CONECTADO al host: "+control.pressureSensorIP+":"+control.pressureSensorPort+"----");
            control.pressureSensorConected = true;
        } catch (IOException e) {
            System.out.println("----No se pudo conectar al host: "+control.pressureSensorIP+":"+control.pressureSensorPort+"----");
            System.err.println("----No se pudo conectar al host: "+control.pressureSensorIP+":"+control.pressureSensorPort+"----");
        }
    }

    private PacketData parsePacket(byte[] packet) throws IOException {
        // Verify header
        ByteBuffer buffer = ByteBuffer.wrap(packet);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        int header = buffer.getShort(0) & 0xFFFF;
        if (header != 0x00FF00) {
            throw new IllegalArgumentException("Invalid packet: incorrect header");
        }

        // Parse timestamp (two 32-bit numbers)
        long seconds = buffer.getInt(3) & 0xFFFFFFFFL;
        long subseconds = buffer.getInt(7) & 0xFFFFFFFFL;
        LocalDateTime timestamp = LocalDateTime.ofEpochSecond(seconds, (int) (subseconds * 1000), ZoneOffset.UTC);

        // Parse 32 channels (16-bit integers)
        double[] pressures = new double[32];
        for (int i = 0; i < 32; i++) {
            int rawValue = buffer.getShort(11 + i * 2) & 0xFFFF;
            pressures[i] = (rawValue - 32768) * (full_scale_pa / 32768);
        }

        return new PacketData(timestamp, pressures);
    }

    // Helper class to store parsed data
    private static class PacketData {
        LocalDateTime timestamp;
        double[] pressures;

        public PacketData(LocalDateTime timestamp, double[] pressures) {
            this.timestamp = timestamp;
            this.pressures = pressures;
        }
    }
    
    public void createIncrementalFile(String directoryPath) throws IOException {
        File dir = new File(directoryPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        File[] txtFiles = dir.listFiles((d, name) -> name.matches("\\d+\\.txt"));
        if (txtFiles != null && txtFiles.length > 0) {
            Arrays.sort(txtFiles, Comparator.comparingInt(f -> Integer.parseInt(f.getName().replace(".txt", ""))));
        }
        int nextFileNumber = (txtFiles != null && txtFiles.length > 0) ?
                Integer.parseInt(txtFiles[txtFiles.length - 1].getName().replace(".txt", "")) + 1 : 1;

        // Crear el nuevo archivo incremental
        String newFileName = nextFileNumber + ".txt";
        preassureFile = new File(directoryPath, newFileName);
        if (preassureFile.createNewFile()) {
            System.out.println("Creado archivo: " + preassureFile.getName());
        } else {
            System.out.println("El archivo ya existe: " + preassureFile.getName());
        }
    }
    
    public void writeLine(String line) throws IOException{
        // Escribir en el archivo
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(preassureFile))) {
            writer.write(line);
            writer.newLine();
        }
        System.out.println("Escritura completada en: " + preassureFile.getName());
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JCheckBoxMenuItem jCheckBoxMenuItem1;
    private javax.swing.JDialog jDialog1;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JPopupMenu jPopupMenu1;
    private javax.swing.JPopupMenu jPopupMenu2;
    private java.awt.PopupMenu popupMenu1;
    private java.awt.PopupMenu popupMenu2;
    // End of variables declaration//GEN-END:variables
}
