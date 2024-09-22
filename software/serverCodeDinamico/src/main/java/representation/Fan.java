package representation;

import javax.swing.JButton;

/**
 *
 * @author mintxo
 */
public class Fan {
    public boolean selected = false;
    public int speed = 0;
    public JButton button;

    public Fan(JButton button) {
        this.button = button;
    }
}
