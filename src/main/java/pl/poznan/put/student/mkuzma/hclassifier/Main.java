package pl.poznan.put.student.mkuzma.hclassifier;

import org.apache.log4j.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.util.Random;

/**
 * Created by Michał Kuźma on 28.05.2017.
 */
public class Main {
    private static final Logger logger = Logger.getLogger(Main.class);

    public static void main(String[] args) {

        // Check if the dataset filepath was given as the parameter
        if (args.length != 1)
            throw new IllegalArgumentException("Wrong number of arguments given");

        String filePath = args[0];

        File folder = new File("data");
        if (!folder.isDirectory()) {
            logger.error("Cannot read data files.");
            return;
        }
        for (final File fileEntry : folder.listFiles()) {
            testForDataset(fileEntry.getPath());
        }

    }

    private static void testForDataset(String filePath) {
        System.out.println("\nTesting for dataset " + filePath + ":");

        ArffLoader.ArffReader arffReader;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            arffReader = new ArffLoader.ArffReader(reader);
        } catch (FileNotFoundException e) {
            logger.error("Error while reading the dataset file", e);
            return;
        } catch (IOException e) {
            logger.error("Error while creating the arff reader for the dataset file", e);
            return;
        }

        Instances data = arffReader.getData();

        String classAttributeName = data.attribute(data.numAttributes() - 1).name();

        data.setClass(data.attribute(classAttributeName));

        evaluateClassifier(new HClassifier(classAttributeName), data);
        evaluateClassifier(new J48(), data);
    }

    private static void evaluateClassifier(Classifier classifier, Instances data) {
        try {
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier, data, 10, new Random(1));

            System.out.printf("\n\n%s\n----------------------------------------\n\n", classifier.getClass().getSimpleName());

            System.out.println("Confusion matrix:");

            double[][] confusionMatrix = evaluation.confusionMatrix();

            StringBuilder breakline = new StringBuilder();
            for (int i = 0; i < confusionMatrix[0].length; i++)
                breakline.append("--------");
            System.out.println(breakline);

            for (double[] confusionMartixRow : confusionMatrix) {
                StringBuilder row = new StringBuilder();
                row.append("|");
                for (double confusionMatrixCell : confusionMartixRow) {
                    row.append(confusionMatrixCell);
                    row.append("\t|");
                }
                System.out.println(row.toString());
                System.out.println(breakline);
            }

            System.out.printf("Weighted Recall: %f\n", evaluation.weightedRecall());
            System.out.printf("Recall on class #1: %f\n", evaluation.recall(0));
            System.out.printf("Recall on class #2: %f\n", evaluation.recall(1));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
