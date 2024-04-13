package functionality;

import java.io.FileInputStream;
import java.io.InputStream;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.ss.usermodel.WorkbookFactory;
import representation.SpeedDrawing;
import userInterface.ControlFrame;

/**
 *
 * @author mintxo
 */
public class Functionality extends Thread{
    public ControlFrame control;
    public SpeedDrawing speedDrawing;
    public boolean realExec;

    public Functionality(ControlFrame control, SpeedDrawing speedDrawing,boolean realExec) {
        this.control = control;
        this.speedDrawing = speedDrawing;
        this.realExec = realExec;
    }
    
    @Override
    public void run(){
        try{                
            InputStream inp = new FileInputStream(control.funFile);
            Workbook wb = WorkbookFactory.create(inp);
            Sheet sheet = wb.getSheetAt(0);
            
            for (Row row : sheet) {
                for (int i = 1; i < control.speedMessage.length+1; i++) {
                    if(row.getCell(i) != null)
                        control.speedMessage[i-1] = (int) row.getCell(i).getNumericCellValue();
                }
                int time = (int) row.getCell(0).getNumericCellValue();
                if(realExec){
                    control.server.updateSpeed(control.speedMessage);
                    speedDrawing.updateDrawing();
                }else{
                    speedDrawing.updateDrawing();
                }
                Thread.sleep(time);
            }
        }catch(Exception e){
            System.out.println("Functionality interrupted");
        }
        if(!realExec){
            speedDrawing.dispose();
        }
    }
}
