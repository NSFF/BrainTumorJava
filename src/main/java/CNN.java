import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class CNN {

    // dataset folder directory with training and test folder
    static final String DATA_DIR_ROOT_FOLDER = System.getProperty("user.dir") + "\\DataSet\\";
    static final int IMG_HEIGHT = 180;
    static final int IMG_WIDTH = 180;
    static final int NUMBER_OF_TRAIN_IMG = 2687;
    static final int NUMBER_OF_TEST_IMG = 313;
    // The number of outcomes that can be classified (in this case YES and NO)
    static final int N_OUTCOMES = 2;

    static int batchSize = 32; // Test batch size

    public static void main(String[] args) throws Exception {
        final Logger log = LoggerFactory.getLogger(CNN.class);

        int nChannels = 1; // Number of input channels
        int nEpochs = 4; // Number of training epochs
        int seed = 753; // randmoizer seed

        log.info("About to load data...");
        DataSetIterator trainData = getDataSetIterator(DATA_DIR_ROOT_FOLDER + "training", NUMBER_OF_TRAIN_IMG);
        log.info("Training set loaded");
        DataSetIterator testData = getDataSetIterator(DATA_DIR_ROOT_FOLDER + "test", NUMBER_OF_TEST_IMG);
        log.info("Test set loaded");

        // Constructing the neural network
        log.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1,1)
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(150).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nOut(N_OUTCOMES)
                        .activation(Activation.SIGMOID)
                        .build())
                .setInputType(InputType.convolutionalFlat(IMG_HEIGHT,IMG_WIDTH,1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model...");
        //Print score every 5 iterations and evaluate on test set every epoch
        model.setListeners(new ScoreIterationListener(5), new EvaluativeListener(testData, 1, InvocationType.EPOCH_END));
        model.fit(trainData, nEpochs);

        String modelPath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),  DATA_DIR_ROOT_FOLDER + "model.zip");

        log.info("Saving model to tmp folder: "+ modelPath);
        model.save(new File(modelPath), true);

        log.info("*************** model finished *******************");
    }
    public static INDArray loadDataSample(String folderPath) throws IOException {
        File imageFile = new File(folderPath);
        //File[] subFolders = folder.listFiles();

        NativeImageLoader nil = new NativeImageLoader(IMG_HEIGHT, IMG_WIDTH);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        INDArray input = Nd4j.create(new int[]{1, IMG_HEIGHT * IMG_WIDTH});

        BufferedImage downSizedImg = resizeImage(ImageIO.read(imageFile), IMG_WIDTH, IMG_HEIGHT);

        INDArray img = nil.asRowVector(downSizedImg);
        scaler.transform(img);
        input.putRow(0,img);

        return input;

    }
    private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {

        File folder = new File(folderPath);
        File[] subFolders = folder.listFiles();

        NativeImageLoader nil = new NativeImageLoader(IMG_HEIGHT, IMG_WIDTH);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        INDArray input = Nd4j.create(new int[]{nSamples, IMG_HEIGHT * IMG_WIDTH});
        INDArray output = Nd4j.create(new int[]{nSamples, N_OUTCOMES});

        int n = 0;
        int labelDigit = 0;
        //scan all yes no  subfolders
        for (File subFolder : subFolders) {
            if (subFolder.getName().equals("no")){
                labelDigit = 0;
            } else{
                labelDigit = 1;
            }


            File[] imageFiles = subFolder.listFiles();
            for (File imageFile : imageFiles) {
                BufferedImage downSizedImg = resizeImage(ImageIO.read(imageFile), IMG_WIDTH, IMG_HEIGHT);

                //File outputfile = new File("D:\\Robin\\School\\Bachelor_Toegepaste_Informatica\\Vakken\\Java_Advanced\\BurstProject\\BrainTumorBurstProject\\DataSet\\image.jpg");
                //ImageIO.write(downSizedImg, "jpg", outputfile);

                //read the image as a one dimensional array of 0..255 values
                INDArray img = nil.asRowVector(downSizedImg);
                //scale the 0..255 integer values into a 0..1 floating range
                scaler.transform(img);
                //copy the img array into the input matrix, in the next row
                input.putRow(n, img);
                output.put(n, labelDigit, 1.0);
                //row counter increment
                n++;
            }
        }

        //Join input and output matrixes into a dataset
        DataSet dataSet = new org.nd4j.linalg.dataset.DataSet(input, output);
        //Convert the dataset into a list
        List<org.nd4j.linalg.dataset.DataSet> listDataSet = dataSet.asList();
        //Shuffle its content randomly
        Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
        //Build and return a dataset iterator that the network can use
        DataSetIterator dataset = new ListDataSetIterator<org.nd4j.linalg.dataset.DataSet>(listDataSet, batchSize);
        return dataset;
    }

     private static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics2D = resizedImage.createGraphics();
        graphics2D.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        graphics2D.dispose();
        return resizedImage;
    }

    public static String modelPredict(INDArray data) throws IOException {
        MultiLayerNetwork model = MultiLayerNetwork.load(new File(DATA_DIR_ROOT_FOLDER + "model.zip"), true);
        int pred = Arrays.stream(model.predict(data)).sum();
        String predString = "";

        if (pred == 0){
            predString = "no";
        }
        else {
            predString = "yes";
        }

        System.out.println("Tumor:" + predString);
        return predString;
        //System.out.println(model.evaluate(data));
    }
}
