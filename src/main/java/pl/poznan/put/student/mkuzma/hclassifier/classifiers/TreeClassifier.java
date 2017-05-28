package pl.poznan.put.student.mkuzma.hclassifier.classifiers;

import org.apache.log4j.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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

        Instances transformedTestingSet = new Instances(testingSet);
        if (transformedTestingSet.attribute(classColumnName) == null) {
            Attribute trainSetClassAttribute = trainInstances.attribute(classColumnName);

            List<String> values = new ArrayList<>(trainSetClassAttribute.numValues());
            for (int i = 0; i < trainSetClassAttribute.numValues(); i++)
                values.add(trainSetClassAttribute.value(i));
            transformedTestingSet.insertAttributeAt(
                    new Attribute(classColumnName, values),
                    transformedTestingSet.numAttributes());

            Random random = new Random();
            for (Instance instance : transformedTestingSet) {
                instance.setValue(transformedTestingSet.attribute(classColumnName).index(),
                        values.get(random.nextInt(values.size())));
            }
        }

        transformedTestingSet.setClass(transformedTestingSet.attribute(classColumnName));

        Evaluation evaluation = new Evaluation(trainInstances);

        evaluation.evaluateModel(_classifier, transformedTestingSet);

        for (int i = 0; i < transformedTestingSet.size(); i++) {
            result.add(trainInstances.attribute(classColumnName).value((int) evaluation.predictions().get(i).predicted()));
        }

        return result;
    }
}
