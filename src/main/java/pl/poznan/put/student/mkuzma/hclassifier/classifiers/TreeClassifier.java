package pl.poznan.put.student.mkuzma.hclassifier.classifiers;

import org.apache.log4j.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Michał Kuźma on 23.05.17.
 */
public class TreeClassifier {
    private static final Logger logger = Logger.getLogger(TreeClassifier.class);

    private J48 _classifier;

    private Instances trainInstances;
    private String classColumnName;

    public void trainClassifier(Instances instances, String classColumnName) throws Exception {
        this.trainInstances = new Instances(instances);
        this.classColumnName = classColumnName;

        trainInstances.setClass(trainInstances.attribute(classColumnName));

        _classifier = new J48();
        _classifier.buildClassifier(trainInstances);

        logger.info("Training model finished.");
    }

    public List<String> classify(Instances testingSet) throws Exception {
        List<String> result = new ArrayList<>();

        testingSet.setClass(testingSet.attribute(classColumnName));

        Evaluation evaluation = new Evaluation(trainInstances);

        evaluation.evaluateModel(_classifier, testingSet);

        for (int i = 0; i < testingSet.size(); i++) {
            result.add(trainInstances.attribute(classColumnName).value((int) evaluation.predictions().get(i).predicted()));
        }

        return result;
    }
}
