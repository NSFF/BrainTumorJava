import org.nd4j.linalg.api.ndarray.INDArray;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        INDArray testData = null;

        // creating a file explorer pop-up window
        String samplePath = OpenFileExplorer.openFile();
        System.out.println("File opened successfully, changing format...");


        // trying to load the sample
        try {
            testData = CNN.loadDataSample(samplePath);
            System.out.println("File loaded to the right format, predicting...");
        }
        catch(IOException e) {
            e.printStackTrace();
        }

        // predicting the sample to be a tumor or not (0 = no tumor 1 = tumor)
        try {
            CNN.modelPredict(testData);
        }
        catch(IOException e) {
            e.printStackTrace();
        }

        System.exit(0);
    }
}
