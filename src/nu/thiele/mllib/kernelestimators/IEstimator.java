package nu.thiele.mllib.kernelestimators;
/**
 * For any kernel, see: http://en.wikipedia.org/wiki/Kernel_(statistics)
 * @author Andreas Thiele
 *
 */
public interface IEstimator {
	public void addValue(double x);
	public double probability(double x);
	
	public enum Estimator{
		COSINE, EPANECHNIKOV, GAUSSIAN, HISTOGRAM, TRIWEIGHT
	}
}
