package representation;

import java.awt.Color;
import java.awt.Graphics;
import static org.mozilla.javascript.Context.exit;
import userInterface.ControlFrame;

/**
 *
 * @author mintxo
 */
public class SpeedDrawing extends javax.swing.JFrame {
    public ControlFrame control;

    /**
     * Creates new form Speed
     * @param control
     */
    public SpeedDrawing(ControlFrame control) {
        initComponents();
        this.control = control;
    }
    
    public void run() throws InterruptedException{
        
        while(true){
            
            //Thread.sleep(200);
            
        }
    }
    
    public void updateDrawing (){
        repaint();
    }
    
    

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jLabel1.setText(".");

        jLabel2.setText(".");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(100, 100, 100)
                .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 12, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap(200, Short.MAX_VALUE)
                .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 12, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(184, 184, 184))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(100, 100, 100)
                .addComponent(jLabel1)
                .addGap(73, 73, 73)
                .addComponent(jLabel2)
                .addContainerGap(445, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents
    
    int a = 0, b = 75, dim;
    
    @Override
    public void paint(Graphics g) {
        super.paint(g);
        int x1 = 1, x2 = 1, x3 = 1, x4 = 1, x5 = 1, x6 = 1;
        try{
            for (int i = 1; i <= control.speedMessage.length; i++) {
                dim = (int) ((control.speedMessage[i-1])*0.75);
                //System.out.println("fan: "+(i-1)+" "+control.speedMessage[i-1]);
                //System.out.println("fan: "+(i-1)+" "+dim);
                switch (i) {
                    case 1, 2, 7, 8 -> {
                        g.setColor(Color.RED);
                        g.fillOval((a+x1*b)-dim/2, (a+1*b)-dim/2, dim, dim);
                        x1++;
                    }
                    case 3, 4, 9, 10 -> {
                        g.setColor(Color.RED);
                        g.fillOval((a+x2*b)-dim/2, (a+2*b)-dim/2, dim, dim);
                        x2++;
                    }
                    case 5, 6, 11, 12 -> {
                        g.setColor(Color.RED);
                        g.fillOval((a+x3*b)-dim/2, (a+3*b)-dim/2, dim, dim);
                        x3++;
                    }
                    case 13, 14, 19, 20 -> {
                        g.setColor(Color.RED);
                        g.fillOval((a+x4*b)-dim/2, (a+4*b)-dim/2, dim, dim);
                        x4++;
                    }
                    case 15, 16, 21, 22 -> {
                        g.setColor(Color.RED);
                        g.fillOval((a+x5*b)-dim/2, (a+5*b)-dim/2, dim, dim);
                        x5++;
                    }
                    case 17, 18, 23, 24 -> {
                        g.setColor(Color.RED);
                        g.fillOval((a+x6*b)-dim/2, (a+6*b)-dim/2, dim, dim);
                        x6++;
                    }
                    default -> {
                        exit();
                    }
                }
            }

        }catch(Exception e){}        
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    // End of variables declaration//GEN-END:variables
}
