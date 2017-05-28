package pl.poznan.put.student.mkuzma.hclassifier.layer;

import org.apache.log4j.Logger;
import pl.poznan.put.student.mkuzma.hclassifier.classifiers.TreeClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Michał Kuźma on 24.05.2017.
 */
public class Layer {
    private static final Logger logger = Logger.getLogger(Layer.class);

    private DatasetTransformer datasetTransformer;
    private final String classColumnName;
    private List<String> classesPassedToNextLayer;
    private Instances trainInstances;

    private List<Layer> nextLayers;

    TreeClassifier _classifier;

    public Layer(DatasetTransformer datasetTransformer, String classColumnName, List<String> classesPassedToNextLayer, Instances trainInstances) {
        this.datasetTransformer = datasetTransformer;
        this.classColumnName = classColumnName;
        this.classesPassedToNextLayer = classesPassedToNextLayer;

        this.trainInstances = this.datasetTransformer.transform(trainInstances);
        trainClassifier();

        this.nextLayers = new ArrayList<>();
    }

    private void trainClassifier() {
        this._classifier = new TreeClassifier();
        try {
            this._classifier.trainClassifier(this.trainInstances, classColumnName);
        } catch (Exception e) {
            logger.error("Error while training classifier", e);
        }
    }

    public List<String> classify(Instances testingSet) {

        List<String> predictions = new ArrayList<>();
        try {
            predictions = _classifier.classify(testingSet);
        } catch (Exception e) {
            logger.error("Error while predicting the testing set", e);
        }

        if (!this.nextLayers.isEmpty()) {
            for (int nextLayerId = 0; nextLayerId < this.nextLayers.size(); nextLayerId++) {
                String classPassedToNextLayer = this.classesPassedToNextLayer.get(nextLayerId);
                Instances testInstancesForNextLayer = new Instances(testingSet);
                for (int i = predictions.size() - 1; i >= 0; i--) {
                    if (!classPassedToNextLayer.equals(predictions.get(i))) {
                        testInstancesForNextLayer.remove(i);
                    }
                }

                List<String> nextLayerPredictions = this.nextLayers.get(nextLayerId).classify(testInstancesForNextLayer);

                int nextLayerPredictionIndex = 0;
                for (int i = 0; i < predictions.size(); i++) {
                    if (classPassedToNextLayer.equals(predictions.get(i))) {
                        predictions.set(i, nextLayerPredictions.get(nextLayerPredictionIndex));
                        nextLayerPredictionIndex += 1;
                    }
                }
            }
        }

        return predictions;
    }

    public Instances getInstancesForNextLayer(Instances instances) {
        Instances result = new Instances(instances);
        result.removeIf(instance -> !classesPassedToNextLayer.equals(getClassValue(instance)));
        return result;
    }

    private String getClassValue(Instance instance) {
        return this.trainInstances.attribute(classColumnName).value((int) instance.value(this.trainInstances.attribute(classColumnName)));
    }

    public List<Layer> getNextLayers() {
        return nextLayers;
    }

    public interface DatasetTransformer {
        Instances transform(Instances data);
    }
}
