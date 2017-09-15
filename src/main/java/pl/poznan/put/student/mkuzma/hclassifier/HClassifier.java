package pl.poznan.put.student.mkuzma.hclassifier;

import org.apache.log4j.Logger;
import pl.poznan.put.student.mkuzma.hclassifier.layer.Layer;
import pl.poznan.put.student.mkuzma.hclassifier.type.InstanceType;
import pl.poznan.put.student.mkuzma.hclassifier.type.TypeClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Main class of the hierarchical classifier. It's purpose is to execute consecutive layers in order to find a proper class.
 * Created by Michał Kuźma on 08.05.17.
 */
public class HClassifier implements Classifier, Serializable {

    private static final Logger logger = Logger.getLogger(HClassifier.class);

    private String minorityClassName;
    private String majorityClassName;
    private String classAttributeName;

    private List<Layer> layers;

    public HClassifier(String classAttributeName) {
        this.classAttributeName = classAttributeName;
        this.layers = new ArrayList<>();
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        layers.clear();

        TypeClassifier typeClassifier = new TypeClassifier(instances, classAttributeName);
        minorityClassName = typeClassifier.getMinorityClassName();
        majorityClassName = typeClassifier.getMajorityClassName();

        Layer safeMajorSafeMinorOtherLayer = new Layer(this::transformSafeMajorSafeMinorOther,
                "InstanceType", Arrays.asList("Other"), instances);
        addLayer(safeMajorSafeMinorOtherLayer);

        Layer borderlineUnsafeMinorLayer = new Layer(this::transformBorderlineUnsafeMinor,
                "InstanceType", Arrays.asList("Borderline", "UnsafeMinority"), instances);
        addLayer(borderlineUnsafeMinorLayer);
        safeMajorSafeMinorOtherLayer.getNextLayers().add(borderlineUnsafeMinorLayer);

        Layer borderlineLayer = new Layer(this::transformBorderline,
                classAttributeName, new ArrayList<>(), instances);
        addLayer(borderlineLayer);
        borderlineUnsafeMinorLayer.getNextLayers().add(borderlineLayer);

        Layer unsafeMinorityLayer = new Layer(this::transformUnsafeMinority,
                classAttributeName, new ArrayList<>(), instances);
        addLayer(unsafeMinorityLayer);
        borderlineUnsafeMinorLayer.getNextLayers().add(unsafeMinorityLayer);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Instances instances = new Instances(instance.dataset(), 0);
        instances.add(instance);
        return instances.attribute(classAttributeName).indexOfValue(classify(instances).get(0));
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        List<Double> distribution = new ArrayList<>(Collections.nCopies(instance.classAttribute().numValues(), 0.0));
        distribution.set((int) classifyInstance(instance), 1.0);
        return distribution.stream().mapToDouble(Double::doubleValue).toArray();
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    private void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    public List<String> classify(Instances testingSet) {
        List<String> result = layers.get(0).classify(testingSet);
        result.replaceAll(s -> {
            if ("SafeMinority".equals(s))
                return minorityClassName;
            if ("SafeMajority".equals(s))
                return majorityClassName;
            return s;
        });
        return result;
    }

    private Instances transformSafeMajorSafeMinorOther(Instances data) {
        Instances newData = new Instances(data);

        List<String> instanceTypes = new ArrayList<>();
        instanceTypes.add("SafeMinority");
        instanceTypes.add("SafeMajority");
        instanceTypes.add("Other");
        newData.insertAttributeAt(new Attribute("InstanceType", instanceTypes), newData.numAttributes());

        TypeClassifier typeClassifier = new TypeClassifier(newData, classAttributeName);

        for (Instance instance : newData) {
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
            instance.setValue(newData.numAttributes() - 1, instanceType);
        }

        newData.setClass(newData.attribute("InstanceType"));
        newData.deleteAttributeAt(newData.attribute(classAttributeName).index());

        return newData;
    }

    private Instances transformBorderlineUnsafeMinor(Instances data) {
        Instances newData = new Instances(data);

        List<String> instanceTypes = new ArrayList<>();
        instanceTypes.add("Borderline");
        instanceTypes.add("UnsafeMinority");
        newData.insertAttributeAt(new Attribute("InstanceType", instanceTypes), newData.numAttributes());

        TypeClassifier typeClassifier = new TypeClassifier(newData, classAttributeName);

        for (int i = newData.size() - 1; i >= 0; i--) {
            Instance instance = newData.instance(i);
            if (InstanceType.UnsafeMinority.equals(typeClassifier.getInstanceType(instance))
                    || InstanceType.Borderline.equals(typeClassifier.getInstanceType(instance)))
                instance.setValue(newData.numAttributes() - 1, typeClassifier.getInstanceType(instance).name());
            else {
                newData.remove(i);
            }
        }

        newData.setClass(newData.attribute("InstanceType"));
        newData.deleteAttributeAt(newData.attribute(classAttributeName).index());

        return newData;
    }

    private Instances transformBorderline(Instances data) {
        Instances newData = new Instances(data);

        TypeClassifier typeClassifier = new TypeClassifier(newData, classAttributeName);
        for (int i = newData.size() - 1; i >= 0; i--) {
            if (!InstanceType.Borderline.equals(typeClassifier.getInstanceType(newData.instance(i))))
                newData.remove(i);
        }

        return newData;
    }

    private Instances transformUnsafeMinority(Instances data) {
        Instances newData = new Instances(data);

        TypeClassifier typeClassifier = new TypeClassifier(newData, classAttributeName);
        for (int i = newData.size() - 1; i >= 0; i--) {
            if (!InstanceType.UnsafeMinority.equals(typeClassifier.getInstanceType(newData.instance(i))))
                newData.remove(i);
        }

        return newData;
    }

}