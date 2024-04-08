package functionality;

import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.dnd.DnDConstants;
import java.awt.dnd.DropTargetDropEvent;
import java.io.File;
import java.io.IOException;
import java.util.List;
import javax.swing.SwingUtilities;

/**
 *
 * @author Mintxoo - mintxosola@gmail.com
 */
public class Index{

    public void getFunFile(Principal principal) throws InterruptedException{
        SwingUtilities.updateComponentTreeUI(principal);
        principal.setVisible(true);
        principal.setTitle("Drag and Drop file");
        principal.setLocationRelativeTo(null);
    }
    
    
    public static File[] getDropFiles(DropTargetDropEvent d){
        try{
            if (d.getDropAction() == DnDConstants.ACTION_MOVE){
                d.acceptDrop(d.getDropAction());
                final Transferable transferable = d.getTransferable();
                if (transferable.isDataFlavorSupported(DataFlavor.javaFileListFlavor)){
                    List<File> listFiles = (List) transferable.getTransferData(DataFlavor.javaFileListFlavor);
                    d.dropComplete(true);
                    System.out.println(listFiles);
                    return listFiles.toArray(File[]::new);
                }
            }
        }catch(UnsupportedFlavorException | IOException e){
            System.out.println(e);
        }
        return null;
    }
    
    
}
