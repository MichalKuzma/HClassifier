package pl.poznan.put.student.mkuzma.hclassifier;

import org.apache.log4j.Logger;
import pl.poznan.put.student.mkuzma.hclassifier.layer.Layer;
import pl.poznan.put.student.mkuzma.hclassifier.type.InstanceType;
import pl.poznan.put.student.mkuzma.hclassifier.type.TypeClassifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Main class of the hierarchical classifier. It's purpose is to execute consecutive layers in order to find a proper class.
 * Created by Michał Kuźma on 08.05.17.
 */
public class HClassifier extends MultipleClassifiersCombiner {

    private static final Logger logger = Logger.getLogger(HClassifier.class);

    private List<Layer> layers;

    public HClassifier() {
        layers = new ArrayList<>();
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        layers.clear();

        Layer safeMajorSafeMinorOtherLayer = new Layer(HClassifier::transformSafeMajorSafeMinorOther,
                "InstanceType", Arrays.asList("Other"), instances);
        addLayer(safeMajorSafeMinorOtherLayer);

        Layer borderlineUnsafeMinorLayer = new Layer(HClassifier::transformBorderlineUnsafeMinor,
                "InstanceType", Arrays.asList("Borderline", "UnsafeMinority"), instances);
        addLayer(borderlineUnsafeMinorLayer);
        safeMajorSafeMinorOtherLayer.getNextLayers().add(borderlineUnsafeMinorLayer);

        Layer borderlineLayer = new Layer(HClassifier::transformBorderline,
                "Class", new ArrayList<>(), instances);
        addLayer(borderlineLayer);
        borderlineUnsafeMinorLayer.getNextLayers().add(borderlineLayer);

        Layer unsafeMinorityLayer = new Layer(HClassifier::transformUnsafeMinority,
                "Class", new ArrayList<>(), instances);
        addLayer(unsafeMinorityLayer);
        borderlineUnsafeMinorLayer.getNextLayers().add(unsafeMinorityLayer);
    }

    private void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    public List<String> classify(Instances testingSet) {
        return layers.get(0).classify(testingSet);
    }

    private static Instances transformSafeMajorSafeMinorOther(Instances data) {
        Instances newData = new Instances(data);

        List<String> instanceTypes = new ArrayList<>();
        instanceTypes.add("SafeMinority");
        instanceTypes.add("SafeMajority");
        instanceTypes.add("Other");
        newData.insertAttributeAt(new Attribute("InstanceType", instanceTypes), newData.numAttributes());

        TypeClassifier typeClassifier = new TypeClassifier(newData);

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

        newData.deleteAttributeAt(newData.attribute("Class").index());

        return newData;
    }

    private static Instances transformBorderlineUnsafeMinor(Instances data) {
        Instances newData = new Instances(data);

        List<String> instanceTypes = new ArrayList<>();
        instanceTypes.add("Borderline");
        instanceTypes.add("UnsafeMinority");
        newData.insertAttributeAt(new Attribute("InstanceType", instanceTypes), newData.numAttributes());

        TypeClassifier typeClassifier = new TypeClassifier(newData);

        for (int i = newData.size() - 1; i >= 0; i--) {
            Instance instance = newData.instance(i);
            if (InstanceType.UnsafeMinority.equals(typeClassifier.getInstanceType(instance))
                    || InstanceType.Borderline.equals(typeClassifier.getInstanceType(instance)))
                instance.setValue(newData.numAttributes() - 1, typeClassifier.getInstanceType(instance).name());
            else {
                newData.remove(i);
            }
        }

        newData.deleteAttributeAt(newData.attribute("Class").index());

        return newData;
    }

    private static Instances transformBorderline(Instances data) {
        Instances newData = new Instances(data);

        TypeClassifier typeClassifier = new TypeClassifier(newData);
        for (int i = newData.size() - 1; i >= 0; i--) {
            if (!InstanceType.Borderline.equals(typeClassifier.getInstanceType(newData.instance(i))))
                newData.remove(i);
        }

        return newData;
    }

    private static Instances transformUnsafeMinority(Instances data) {
        Instances newData = new Instances(data);

        TypeClassifier typeClassifier = new TypeClassifier(newData);
        for (int i = newData.size() - 1; i >= 0; i--) {
            if (!InstanceType.UnsafeMinority.equals(typeClassifier.getInstanceType(newData.instance(i))))
                newData.remove(i);
        }

        return newData;
    }

}