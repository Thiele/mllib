package nu.thiele.mllib.classifiers;

import java.util.Map;


public interface IClassifier {
	public double classify(double[] x);
	public Map<Double,Double> probability(double[] x);
	public void train(double[][] x, double[] y);
}
