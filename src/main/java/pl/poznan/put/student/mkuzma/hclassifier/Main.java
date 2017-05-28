package pl.poznan.put.student.mkuzma.hclassifier;

import org.apache.log4j.Logger;
import pl.poznan.put.student.mkuzma.hclassifier.classifiers.TreeClassifier;
import pl.poznan.put.student.mkuzma.hclassifier.type.TypeClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

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

        HClassifier hierarchicalClassifier = new HClassifier();
        try {
            hierarchicalClassifier.buildClassifier(data);
        } catch (Exception e) {
            logger.error("Error while building the classifier", e);
        }

        Instances testingSet = new Instances(data);
        testingSet.deleteAttributeAt(testingSet.attribute("Class").index());

        TypeClassifier typeClassifier = new TypeClassifier(data);
        List<String> predictions = hierarchicalClassifier.classify(testingSet);
        predictions = predictions.stream().map(p -> ("SafeMajority".equals(p) ? typeClassifier.getMajorityClassName() :
                ("SafeMinority".equals(p) ? typeClassifier.getMinorityClassName() : p))).collect(Collectors.toList());

        double precision = 0.0;
        double minorityPrecision = 0.0;
        int minorityCount = 0;
        for (int i = 0; i < testingSet.size(); i++) {
            logger.debug(data.get(i) + "\t\t" + predictions.get(i));
            if (typeClassifier.getMinorityClassName().equals(data.attribute("Class").value((int) data.get(i).value(data.attribute("Class")))))
                minorityCount += 1;
            if (data.attribute("Class").value((int) data.get(i).value(data.attribute("Class"))).equals(predictions.get(i))) {
                if (typeClassifier.getMinorityClassName().equals(predictions.get(i)))
                    minorityPrecision += 1.0;
                precision += 1.0;
            }
        }
        precision /= data.size();
        minorityPrecision /= minorityCount;
        logger.info("HIERARCHICAL CLASSIFIER");
        logger.info("Precision: " + Double.toString(precision));
        logger.info("Minority precision: " + Double.toString(minorityPrecision));

        TreeClassifier treeClassifier = new TreeClassifier();
        try {
            treeClassifier.trainClassifier(data, "Class");
        } catch (Exception e) {
            logger.error("Error while building the classifier", e);
        }

        List<String> treeClassifierPredictions = new ArrayList<>();
        try {
            treeClassifierPredictions = treeClassifier.classify(testingSet);
        } catch (Exception e) {
            logger.error("Error while classifying the test set", e);
        }

        double treeClassifierPrecision = 0.0;
        double treeClassifierMinorityPrecision = 0.0;
        for (int i = 0; i < testingSet.size(); i++) {
            if (typeClassifier.getMinorityClassName().equals(data.attribute("Class").value((int) data.get(i).value(data.attribute("Class")))))
            if (data.attribute("Class").value((int) data.get(i).value(data.attribute("Class"))).equals(treeClassifierPredictions.get(i))) {
                if (typeClassifier.getMinorityClassName().equals(treeClassifierPredictions.get(i)))
                    treeClassifierMinorityPrecision += 1.0;
                treeClassifierPrecision += 1.0;
            }
        }
        treeClassifierPrecision /= data.size();
        treeClassifierMinorityPrecision /= minorityCount;
        logger.info("SIMPLE TREE CLASSIFIER");
        logger.info("Precision: " + Double.toString(treeClassifierPrecision));
        logger.info("Minority precision: " + Double.toString(treeClassifierMinorityPrecision));


    }
}
