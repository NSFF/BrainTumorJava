import javax.swing.*;
import java.awt.*;
import java.io.File;

public class OpenFileExplorer {

    public static String openFile(){
        // creating a file explorer pop-up window
        FileDialog fd = new FileDialog(new JFrame());
        fd.setVisible(true);
        File[] f = fd.getFiles();
        if(f.length > 0) {
            return fd.getFiles()[0].getAbsolutePath();
        }
        else{
            System.out.println("No File was chosen");
            System.exit(0);
            return "no file selected";
        }
    }
}
