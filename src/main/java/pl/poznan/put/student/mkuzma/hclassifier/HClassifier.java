package pl.poznan.put.student.mkuzma.hclassifier;

import org.apache.log4j.Logger;
import pl.poznan.put.student.mkuzma.hclassifier.classifiers.Tier1Classifier;
import pl.poznan.put.student.mkuzma.hclassifier.type.TypeClassifier;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Main class of the hierarchical classifier. It's purpose is to execute consecutive layers in order to find a proper class.
 * Created by Michał Kuźma on 08.05.17.
 */
public class HClassifier {

    private static final Logger logger = Logger.getLogger(HClassifier.class);

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
        Map<Integer, String> results = new HashMap<>();

        Tier1Classifier tier1Classifier = new Tier1Classifier(data);

        List<String> predictions = new ArrayList<>();
        try {
            predictions = tier1Classifier.classify(data);
            for (int i = 0; i < data.size(); i++)
                System.out.println(data.get(i) + "\t\t" + predictions.get(i));
        } catch (Exception e) {
            logger.error("Error while classifing data", e);
        }

        TypeClassifier typeClassifier = new TypeClassifier(data);
        Instances tier2Instances = new Instances(data);
        for (int i = data.size() - 1; i >= 0; i--) {
            if ("SafeMinority".equals(predictions.get(i))) {
                results.put(i, typeClassifier.getMinorityClassName());
                break;
            }
            if ("SafeMajority".equals(predictions.get(i))) {
                results.put(i, typeClassifier.getMajorityClassName());
                break;
            }

            tier2Instances.delete(i);
        }

//        TypeClassifier typeClassifier = new TypeClassifier(data);
//
//        for (Instance instance : data) {
//            System.out.println(typeClassifier.getInstanceType(instance));
//        }

    }
}
