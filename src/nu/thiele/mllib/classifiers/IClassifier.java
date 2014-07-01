package nu.thiele.mllib.classifiers;

import java.util.List;
import java.util.Map;

import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.exceptions.InvalidArgumentException;


public interface IClassifier {
	public Object classify(double[] x);
	public Map<Object,Double> calculateProbabilityForClassifications(double[] x);
	public void loadClassifier();
	public void setTrainingData(List<DataEntry> data) throws InvalidArgumentException;
}
