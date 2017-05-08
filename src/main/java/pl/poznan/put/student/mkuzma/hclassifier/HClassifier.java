package pl.poznan.put.student.mkuzma.hclassifier;

import org.apache.log4j.Logger;
import pl.poznan.put.student.mkuzma.hclassifier.type.TypeClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

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

        TypeClassifier typeClassifier = new TypeClassifier(data);

        for (Instance instance : data) {
            System.out.println(typeClassifier.getInstanceType(instance));
        }

    }
}
