package pl.poznan.put.student.mkuzma.hclassifier.type;

import org.apache.log4j.Logger;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

import java.util.Comparator;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The classifier used to determine a type of the instance.
 * Created by Michał Kuźma on 08.05.17.
 */
public class TypeClassifier {

    private static final Logger logger = Logger.getLogger(TypeClassifier.class);

    private static final int K_NEIGHBOURS = 5;
    private static final int[] MINORITY_NEIGHBOURS_SAFE = {5, 4};
    private static final int[] MINORITY_NEIGHBOURS_BORDERLINE = {3, 2};

    private final Instances data;
    private LinearNNSearch knn;
    private String classAttributeName;

    private String minorityClassName;
    private String majorityClassName;

    public TypeClassifier(Instances data, String classAttributeName) {
        this.data = data;
        this.knn = new LinearNNSearch(data);
        this.classAttributeName = classAttributeName;

        getClassNames();
    }

    public InstanceType getInstanceType(Instance instance) {

        Instances nearestNeighbours;
        try {
            nearestNeighbours = knn.kNearestNeighbours(instance, K_NEIGHBOURS);
        } catch (Exception e) {
            logger.error("Error while finding the nearest neighbours", e);
            return null;
        }

        Map<String, Long> countMap = nearestNeighbours.stream()
                .map(this::getClassName)
                .collect(Collectors.groupingBy(s -> s, Collectors.counting()));

        long minorityCount = countMap.get(minorityClassName) != null ? countMap.get(minorityClassName) : 0;

        if (IntStream.of(MINORITY_NEIGHBOURS_BORDERLINE).anyMatch(x -> x == minorityCount))
            return InstanceType.Borderline;

        if (minorityClassName.equals(getClassName(instance))) {
            if (IntStream.of(MINORITY_NEIGHBOURS_SAFE).anyMatch(x -> x == minorityCount))
                return InstanceType.SafeMinority;

            else // if (IntStream.of(MINORITY_NEIGHBOURS_UNSAFE).anyMatch(x -> x == minorityCount))
                return InstanceType.UnsafeMinority;
        } else {
            if (nearestNeighbours.stream().anyMatch(i -> minorityClassName.equals(getClassName(i))
                                                    && InstanceType.UnsafeMinority.equals(getInstanceType(i))))
                return InstanceType.UnsafeMinority;

            return InstanceType.SafeMajority;
        }
    }

    private void getClassNames() {
        Map<String, Long> countMap = this.data.stream()
                .map(this::getClassName)
                .collect(Collectors.groupingBy(s -> s, Collectors.counting()));

        Comparator<Map.Entry<String, Long>> countMapComparator =
                (entry1, entry2) -> entry1.getValue() > entry2.getValue() ? 1 : -1;

        Optional<Map.Entry<String, Long>> optionalMinimumEntry = countMap.entrySet().stream().min(countMapComparator);

        if (!optionalMinimumEntry.isPresent()) {
            logger.error("Could not find the minority class");
            return;
        }

        this.minorityClassName = optionalMinimumEntry.get().getKey();

        Optional<Map.Entry<String, Long>> optionalMaximumEntry = countMap.entrySet().stream().max(countMapComparator);

        if (!optionalMaximumEntry.isPresent()) {
            logger.error("Could not find the majority class");
            return;
        }

        this.majorityClassName = optionalMaximumEntry.get().getKey();
    }

    private String getClassName(Instance instance) {
        return this.data.attribute(classAttributeName).value((int) instance.value(this.data.attribute(classAttributeName).index()));
    }

    public String getMinorityClassName() {
        return minorityClassName;
    }

    public String getMajorityClassName() {
        return majorityClassName;
    }
}
