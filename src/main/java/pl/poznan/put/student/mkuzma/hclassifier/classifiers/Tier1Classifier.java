package pl.poznan.put.student.mkuzma.hclassifier.classifiers;

import org.apache.log4j.Logger;
import pl.poznan.put.student.mkuzma.hclassifier.type.TypeClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * First tier classifier. It labels the instance as Safe Minority, Safe Majority or Other.
 * Created by Michał Kuźma on 22.05.17.
 */
public class Tier1Classifier {

    private static final Logger logger = Logger.getLogger(Tier1Classifier.class);

    private Instances trainInstances;

    TreeClassifier _classifier;

    public Tier1Classifier(Instances trainInstances) {
        this.trainInstances = trainInstances;

        try {
            preprocessData();
        } catch (Exception e) {
            logger.error("Error while preprocessing data", e);
        }

        try {
            trainClassifier();
        } catch (Exception e) {
            logger.error("Error while training the classifier", e);
        }
    }

    private void preprocessData() throws Exception {
        List<String> instanceTypes = new ArrayList<>();
        instanceTypes.add("SafeMinority");
        instanceTypes.add("SafeMajority");
        instanceTypes.add("Other");
        trainInstances.insertAttributeAt(new Attribute("InstanceType", instanceTypes), trainInstances.numAttributes());

        TypeClassifier typeClassifier = new TypeClassifier(trainInstances);

        for (Instance instance : trainInstances) {
            String instanceType = "Other";
            switch (typeClassifier.getInstanceType(instance)) {
                case SafeMinority:
                    instanceType = "SafeMinority";
                    break;
                case SafeMajority:
                    instanceType = "SafeMajority";
                    break;
                case Borderline:
                case UnsafeMinority:
                    instanceType = "Other";
                    break;
            }
            instance.setValue(trainInstances.numAttributes() - 1, instanceType);
        }

        trainInstances.deleteAttributeAt(trainInstances.attribute("Class").index());

        System.out.println(trainInstances);
    }

    private void trainClassifier() throws Exception {
        this._classifier = new TreeClassifier();
        this._classifier.trainClassifier(trainInstances, "InstanceType");
    }

    public List<String> classify(Instances testingSet) throws Exception {
        return _classifier.classify(testingSet);
    }
}
