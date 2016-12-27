package nu.thiele.mllib.classifiers;

public interface IMultiClassifier {
	public double[] classifyMultipleOutputs(double[] x);
}
